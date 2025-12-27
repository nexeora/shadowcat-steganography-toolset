#![cfg_attr(not(test), no_std)]
#![cfg_attr(test, feature(core_intrinsics))]
#![cfg_attr(test, allow(internal_features))]
#![feature(likely_unlikely)]

const ANTIPENDING_BITS_NUM : u32 = 8;

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::hint::unreachable_unchecked();
    }
}

macro_rules! assume_assert {
    ($cond:expr) => {
        if !$cond {
            #[cfg(test)]
            {
                println!("assert failed: {}", stringify!($cond));
                core::intrinsics::breakpoint();
            }
            #[cfg(not(test))]
            {
                #[cfg(not(debug_assertions))]   
                {
                    unsafe {core::hint::unreachable_unchecked();}
                }
                #[cfg(debug_assertions)]
                {
                    debug_assert!($cond, "assert failed: {}", stringify!($cond));
                }
            } 

            
        }  
    };
}


trait Buffer<T> : Sized{
    fn is_writeable(&self) -> bool;
    fn write(&mut self, val: T) -> bool;//下一次是否还可以写入
}
trait Stream<T: Copy + Sized> : Sized{
    fn read(&mut self) -> Option<T>;
    fn read_n<const N: usize>(&mut self) -> [Option<T>; N] {
        let mut ret = [None; N];
        for i in 0..N {
            ret[i] = self.read();
        }
        ret
    }
}
struct LinearBuffer<'a, T> {
    base: &'a mut [T],
    pos: usize,
}

impl<'a, T> LinearBuffer<'a, T> {
    fn new(base: &'a mut [T]) -> Self {
        Self {
            base,
            pos: 0,
        }
    }
    unsafe fn from_raw_parts(base: *mut T, len: usize) -> Option<Self> {
        Some(
            Self {
                base: unsafe { to_mut_slice_checked(base, len)? },
                pos: 0,
            }
        )
    }
}

struct ArrayStream<'a, T> {
    base: &'a [T],
    pos: usize,
}

impl<'a, T: Copy> Stream<T> for ArrayStream<'a, T> {
    fn read(&mut self) -> Option<T> {
        if self.pos >= self.base.len() {
            return None;
        }
        let ret = self.base[self.pos];
        self.pos += 1;
        Some(ret)
    }
}

impl<'a, T: Copy> ArrayStream<'a, T> {
    fn new(base: &'a [T]) -> Self {
        Self {
            base,
            pos: 0,
        }
    }
}

impl<'a, T> Buffer<T> for LinearBuffer<'a, T> {
    fn is_writeable(&self) -> bool {
        self.pos < self.base.len()
    }
    fn write(&mut self, val: T) -> bool {
        debug_assert!(self.is_writeable());
        self.base[self.pos] = val;
        self.pos += 1;
        self.is_writeable()
    }
}

struct SymEncoderState {
    inf: u64,
    sup: u64,
}





impl SymEncoderState {
    fn new() -> Self {
        Self {
            inf: 0,
            sup: u64::MAX,
        } // [0, 2^64-1]
    }
    
    fn acc(&mut self, cum: u64, freq: u64) {
        assume_assert!(freq != 0);
        self.inf += cum;
        self.sup = self.inf.wrapping_add(freq).wrapping_sub(1); // [inf, sup]
    }

    fn range(&self) -> u64 {
        (self.sup - self.inf).wrapping_add(1)
    }

    fn is_renormable(&self) -> bool {
        (self.sup - self.inf) < (1u64 << (64 - (8 + ANTIPENDING_BITS_NUM as u64)))
    }

    fn xor_first(&mut self) {
        self.inf ^= 1 << 63;
        self.sup ^= 1 << 63;
    }
    fn renorm(&mut self) -> (u8, u8) {
        //inf: resolved(8 - pending_bit_num bit) | 0 | 1..(pending_bit_num - 1 bit) | 1 | 0..(47 bit)
        //sub: resolved(8 - pending_bit_num bit) | 1 | 0..(pending_bit_num - 1 bit) | 0 | 1 ..(47 bit)
        //range = sub - inf + 1 : 2^48
        //new_inf = (self.inf - inf) * (1^64)/range
        //new_range = self.range * (1^64)/range
        //new_sup = new_inf + new_range - 1
        assume_assert!(self.is_renormable());
        let inf_head = (self.inf >> (64 - 8)) as u8;
        let sup_head = (self.sup >> (64 - 8)) as u8;
        let diff = sup_head ^ inf_head;

        assume_assert!(
               diff == 0b0000_0000 
            || diff == 0b0000_0001 
            || diff == 0b0000_0011 
            || diff == 0b0000_0111 
            || diff == 0b0000_1111 
            || diff == 0b0001_1111
            || diff == 0b0011_1111
            || diff == 0b0111_1111
            || diff == 0b1111_1111
        );
        assume_assert!((diff == 0) || (inf_head.wrapping_add(1) == sup_head));//diff非零时，inf_head + 1 == sup_head
        self.inf <<= 8;
        self.sup <<= 8;
        self.sup += 0b1111_1111;
        /*
        if core::hint::unlikely(diff != 0){
            self.xor_first();
        }
        */
        return (inf_head, diff);
    }

}

struct PendingState {
    count: u32,
    byte_cache: u8,  //resolved(8 - pending_bit_num bit) | 0 | 1..(pending_bit_num - 1 bit)
    mask: u8,        //0(8 - pending_bit_num bit) | 1..(pending_bit_num bit)
}

impl PendingState {
    fn new() -> Self {
        Self {
            count: 0,
            byte_cache: 0,
            mask: 0,
        }
    }
    fn resolve_pending(&self, not_first_resolution_bit: u8) -> u8 {
        debug_assert!(not_first_resolution_bit == 0 || not_first_resolution_bit == 1);
        self.byte_cache ^ (self.mask & (not_first_resolution_bit.wrapping_sub(1)))
    }

}

struct ACEncoder<T: Buffer<u8> + Sized> {
    sym_encoder: SymEncoderState,
    pending_state: PendingState,
    is_resolve_stopped: bool,
    buf: T,
}

impl<T: Buffer<u8> + Sized> ACEncoder<T> {

    fn new(buf: T) -> Self {
        Self {
            sym_encoder: SymEncoderState::new(),
            pending_state: PendingState::new(),
            is_resolve_stopped: false,
            buf,
        }
    }

    fn change_buffer(&mut self, buffer: T) {
        self.buf = buffer;
        if self.is_resolve_stopped {
            if core::hint::unlikely(!self.buf.is_writeable()) {
                return;
            }
            let not_first_resolution_bit = self.pending_state.byte_cache >> 7;
            while self.pending_state.count != 0 {
                self.pending_state.count -= 1;
                if core::hint::unlikely(!self.buf.write((not_first_resolution_bit ^ 1).wrapping_sub(1))) {
                    self.is_resolve_stopped = true;
                    return;
                }
            }
            self._renorm_without_resolving(self.pending_state.byte_cache, self.pending_state.mask);
        }
    }

    fn _renorm_without_resolving(&mut self, inf_head: u8, diff: u8) {
        assume_assert!(
            (self.pending_state.count == 0) 
            || ((diff & (1 << 7)) != 0)
        );
        if core::hint::unlikely(diff != 0) {
            self.sym_encoder.xor_first();
            if self.pending_state.count == 0 {
                self.pending_state.byte_cache = inf_head;
                self.pending_state.mask = diff;
                self.pending_state.count = 1;
            } else if (diff & (1 << 7)) != 0 {
                self.pending_state.count += 1;
            } else {
                
                unsafe {core::hint::unreachable_unchecked();}
            }
        } else {
            self.buf.write(inf_head);
        }
    }

    fn _renorm(&mut self) {
            debug_assert!(self.buf.is_writeable());
            let (inf_head, diff) = self.sym_encoder.renorm();

            if core::hint::unlikely(self.pending_state.count != 0 && (diff & (1 << 7)) == 0) {

                let inf_head = inf_head ^ (1 << 7);
                let not_first_resolution_bit = inf_head >> 7;

                self.buf.write(self.pending_state.resolve_pending(not_first_resolution_bit));
                self.pending_state.count -= 1;
                self.pending_state.byte_cache = inf_head;
                self.pending_state.mask = diff;
                while self.pending_state.count != 0 {
                    self.pending_state.count -= 1;
                    if core::hint::unlikely(!self.buf.write((not_first_resolution_bit ^ 1).wrapping_sub(1))) {
                        self.is_resolve_stopped = true;
                        return;
                    }
                }
                self._renorm_without_resolving(self.pending_state.byte_cache, self.pending_state.mask);
                
            } else {
                self._renorm_without_resolving(inf_head, diff);
            }
            

            
    }

    fn renorm(&mut self) -> Option<u64> {
        while self.sym_encoder.is_renormable() {
            if core::hint::unlikely(!self.buf.is_writeable()) {
                return None;
            }
            self._renorm();
        }
        Some(self.sym_encoder.range())
    }

    fn encode(&mut self, cum: u64, freq: u64) -> Option<u64> {
        self.sym_encoder.acc(cum, freq);
        return self.renorm();
        
}
}



struct SafeCumProbTable<'a>  {
    right:u32,
    probs: &'a [SCatVectorU64_8],
}

impl SafeCumProbTable<'_> {
    fn new(table: &SCatCumProbTable) -> Self {
        Self {
            right: table.right,
            probs: unsafe { core::slice::from_raw_parts(table.probs, ((table.right + 1 + 7) / 8) as usize) },
        }
    }

    fn get_sym(&self, sym: u32) -> (u64, u64) {
        let cum = self[sym as usize];
        let freq = (self[(sym + 1) as usize]).wrapping_sub(cum);
        (cum, freq)
    }
}

impl core::ops::Index<usize> for SafeCumProbTable<'_> {
    type Output = u64;
    fn index(&self, index: usize) -> &Self::Output {
        &(self.probs[index / 8].data[index % 8])
    }
}


struct ACDecoder<T: Stream<u8> + Sized> {
    sym_encoder: SymEncoderState,
    cur_state: u64,
    bytes: T,
}

impl<T: Stream<u8> + Sized> ACDecoder<T> {
    fn new(bytes: T) -> Self {
        let mut cur_state = 0;
        let mut bytes = bytes;
        for _ in 0..8 {
            cur_state <<= 8;
            match bytes.read() {
                Some(byte) => {
                    cur_state |= byte as u64 ;
                },
                None => {}
            }
        }
        Self {
            sym_encoder: SymEncoderState::new(),
            bytes,
            cur_state,
        }
    }
    fn _search(&self, table: &SafeCumProbTable) -> usize {
        let mut left = 0;
        let mut right = table.right;
        assume_assert!(right < ((table.probs.len() as u32) * 8));
        let state = self.cur_state - self.sym_encoder.inf;
        while left != (right - 1) {
            let mid = (left + right) / 2;
            assume_assert!(left <= mid && mid < right);
            if state < table[mid as usize] {
                right = mid;
            } else {
                left = mid;
            }
        }
        left as usize
    }

    fn _decode(&mut self, table: &SafeCumProbTable) -> u32 {
        let sym = self._search(table);
        let cum = table[sym];
        let freq = table[sym + 1].wrapping_sub(cum);
        self.sym_encoder.acc(cum, freq);
        sym as u32
    }


    fn decode(&mut self, table: &SafeCumProbTable) -> (u32, Option<u64>) {
        let sym = self._decode(table);
        while self.sym_encoder.is_renormable() {
            let (head, diff) = self.sym_encoder.renorm();
            let cur_head = (self.cur_state>>56) as u8;
            debug_assert!(cur_head - head <= 1);
            debug_assert!(cur_head >= head);
            match self.bytes.read() {
                Some(byte) => {
                    self.cur_state <<= 8;
                    self.cur_state |= byte as u64;
                },
                None => {
                    return (sym, None);
                },
            }
            if core::hint::unlikely(diff != 0){
                self.sym_encoder.xor_first();
                self.cur_state ^= 1 << 63;
            }
        }
        
        (sym, Some(self.sym_encoder.range()))
    }

    
}

unsafe fn to_slice_checked<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]>{
    if ptr.is_null() {
        return None;
    }
    if len == 0 {
        return None;
    }
    unsafe { Some(core::slice::from_raw_parts(ptr, len)) }

}
unsafe fn to_mut_slice_checked<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]>{
    if ptr.is_null() {
        return None;
    }
    if len == 0 {
        return None;
    }
    unsafe { Some(core::slice::from_raw_parts_mut(ptr, len)) }
}
unsafe fn cast_from_void<'a, T>(ptr: *mut core::ffi::c_void) -> &'a mut T{
    unsafe { &mut *(ptr as *mut T) }
}



type SCatModelCallback = unsafe extern "C" fn(u32,u64) -> SCatCumProbTable;

#[repr(C)]
pub struct SCatBufferChangeResult {
    last_range: u64,
    last_buf_pos: usize,
    
}

#[repr(C)]
pub struct SCatDecodeResult {
    range: u64,
    sym: u32,    
}

#[repr(i32)]
pub enum SCatStatus {
    Ok = 0,
    NoEnoughCache = 1,
    NoCallback = 101,
    InvalidParam = 102,
    InvalidCallbackReturn = 103,
    CallBeforeInit = 104,
}

#[repr(C)]
pub struct SCatVectorU64_8 {
    data: [u64; 8],
}
#[repr(C)]
pub struct SCatCumProbTable {
    right:u32,
    probs: *const SCatVectorU64_8,
}


#[repr(C)]
pub struct SCatStructLayout{
    size:usize,
    align:usize,
}

trait LayoutGetable : Sized{
    fn layout() -> SCatStructLayout{
        SCatStructLayout{
            size: core::mem::size_of::<Self>(),
            align: core::mem::align_of::<Self>(),
        }
    }
}

impl LayoutGetable for ACEncoder<LinearBuffer<'_, u8>> {
    
}
impl LayoutGetable for ACDecoder<ArrayStream<'_, u8>> {
    
}

#[unsafe(no_mangle)]
pub extern "C" fn scat_encoder_layout() -> SCatStructLayout{
    ACEncoder::<LinearBuffer::<u8>>::layout()
}
#[unsafe(no_mangle)]
pub extern "C" fn scat_decoder_layout() -> SCatStructLayout{
    ACDecoder::<ArrayStream::<u8>>::layout()
}


/// 初始化编码器
/// 
/// # Arguments
/// 
/// * `this_` - 编码器指针
/// * `buffer` - 编码器缓冲区指针
/// * `buffer_size` - 编码器缓冲区大小
/// 
/// # Returns
/// 
/// * `Status` - 初始化状态，若为Ok则初始化成功
#[unsafe(no_mangle)]
pub extern "C" fn scat_init_encoder(this_: *mut core::ffi::c_void, buffer: *mut u8, buffer_size: usize) -> SCatStatus {
    if let Some(buf) = unsafe { to_mut_slice_checked(buffer, buffer_size) } {
        let this = unsafe { cast_from_void::<ACEncoder<LinearBuffer<'_, u8>>>(this_) };
        *this = ACEncoder::new(LinearBuffer::new(buf));
        SCatStatus::Ok
    } else {
        SCatStatus::InvalidParam
    }
}


/// 编码一个符号
/// 
/// # Arguments
/// 
/// * `this_` - 编码器指针
/// * `cum` - 符号的累计频率
/// * `freq` - 符号的频率
/// * `range_ptr` - 接收下一个编码范围的指针
/// 
/// # Returns
/// 
/// * `bool` - 缓冲区是否已满，若为false则需要调用change_buffer切换缓冲区，下一个编码范围将延迟到调用change_buffer后给出
#[unsafe(no_mangle)]
pub extern "C" fn scat_encode(this_: *mut core::ffi::c_void, cum: u64, freq: u64, range_ptr: *mut u64) -> bool {
    let this = unsafe { cast_from_void::<ACEncoder<LinearBuffer<'_, u8>>>(this_) };
    match this.encode(cum, freq) {
        Some(range) => {
            unsafe { *range_ptr = range };
            true
        },
        None => {
            false
        },
    }
}


/// 切换编码器缓冲区
/// 
/// # Arguments
/// 
/// * `this_` - 编码器指针
/// * `buffer` - 编码器缓冲区指针
/// * `buffer_size` - 编码器缓冲区大小
/// * `result` - 接收切换结果的指针
/// 
/// # Returns
/// 
/// * `bool` - 缓冲区是否已满，若为true则需要再次调用change_buffer切换缓冲区，下一个编码范围将延迟到下次调用后给出
#[unsafe(no_mangle)]
pub extern "C" fn scat_change_buffer(this_: *mut core::ffi::c_void, buffer: *mut u8, buffer_size: usize, result: *mut SCatBufferChangeResult) -> bool {
    let this = unsafe { cast_from_void::<ACEncoder<LinearBuffer<'_, u8>>>(this_) };
    let last_buf_pos = this.buf.pos;
    this.change_buffer(LinearBuffer::new(unsafe { core::slice::from_raw_parts_mut(buffer, buffer_size) }));
    unsafe { *result = SCatBufferChangeResult {
            last_buf_pos: last_buf_pos,
            last_range: this.renorm().unwrap_or(0),
        };
    }
    this.buf.is_writeable()
}

/// 初始化解码器
/// 
/// # Arguments
/// 
/// * `this_` - 解码器指针
/// * `bytes` - 待解码字节流指针
/// * `bytes_size` - 待解码字节流长度
/// 
/// # Returns
/// 
/// * `Status` - 初始化状态，若为Ok则初始化成功
#[unsafe(no_mangle)]
pub extern "C" fn scat_init_decoder(this_: *mut core::ffi::c_void, bytes: *const u8, bytes_size: usize) -> SCatStatus {
    let decoder = unsafe { cast_from_void::<ACDecoder<ArrayStream<'_, u8>>>(this_) };
    if let Some(bytes_slice) = unsafe { to_slice_checked(bytes, bytes_size) } {
        *decoder = ACDecoder::new(
            ArrayStream::new(bytes_slice), 
        );
        SCatStatus::Ok
    } else {
        SCatStatus::InvalidParam
    }
}
/// 解码一个符号
/// 
/// # Arguments
/// 
/// * `this_` - 解码器指针
/// * `table` - 概率表
/// * `result` - 接收解码结果的指针
/// 
/// # Returns
/// 
/// * `bool` - 是否后续还有需要解码的符号
#[unsafe(no_mangle)]
pub extern "C" fn scat_decode(this_: *mut core::ffi::c_void, table: SCatCumProbTable,result:*mut SCatDecodeResult) -> bool {
    let decoder = unsafe { cast_from_void::<ACDecoder<ArrayStream<'_, u8>>>(this_) };
    let (sym, range) = decoder.decode(&SafeCumProbTable::new(&table));
    unsafe { *result = SCatDecodeResult {
            range: range.unwrap_or(0),
            sym,
        };
    }
    range.is_some()
}

/// 编码所有符号
/// 
/// # Arguments
/// 
/// * `this_` - 编码器指针
/// * `syms` - 待编码符号流指针
/// * `syms_size` - 待编码符号流长度
/// * `callback` - 概率表回调函数指针
/// * `length_ptr` - 接收编码后的字节流长度的指针
/// 
/// # Returns
/// 
/// * `Status` - 编码状态，若为Ok则编码成功, 
/// 注意尚未实现恢复因缓冲区不足而调用encode_all失败的编码器状态，对其使用change_buffer切换缓冲区没有意义
#[unsafe(no_mangle)]
pub extern "C" fn scat_encode_all(this_: *mut core::ffi::c_void, syms: *const u32, syms_size: usize, callback: SCatModelCallback, length_ptr: *mut u32) -> SCatStatus {
    let encoder = unsafe { cast_from_void::<ACEncoder<LinearBuffer<'_, u8>>>(this_) };
    let mut cum_table = unsafe {callback(u32::MAX, 0)};
    let syms = match unsafe { to_slice_checked(syms, syms_size) } {
        Some(syms) => syms,
        None => return SCatStatus::InvalidParam,
    };
    let mut syms = ArrayStream::new(syms);
    let mut sym;
    let mut ret = SCatStatus::NoEnoughCache;
    while let Some(range) = {
        sym = syms.read();
        if let Some(sym) = sym {
            let prob = SafeCumProbTable::new(&cum_table); 
            let (cum, freq) = prob.get_sym(sym); 
            #[cfg(test)]
            {
                //println!("sym: {}, cum: {}, freq: {}", sym, cum, freq);
                if sym == 9 && cum == 987329519148042 && freq == 109703279905338 {
                    //core::intrinsics::breakpoint();
                }
            }
            encoder.encode(cum, freq)
        } else {
            ret = SCatStatus::Ok;
            None
        }
    } {
        cum_table = unsafe {callback(sym.unwrap() as u32, range)};
        
    }
    unsafe { *length_ptr = encoder.buf.pos as u32 };
    ret
}


/// 解码所有符号，解码结果由回调函数自行接收
/// 
/// # Arguments
/// 
/// * `this_` - 解码器指针
/// * `callback` - 概率表回调函数指针
/// 
/// # Returns
/// 
/// * `Status` - 解码状态，若为Ok则解码成功
#[unsafe(no_mangle)]
pub extern "C" fn scat_decode_all(this_: *mut core::ffi::c_void, callback: SCatModelCallback) -> SCatStatus {
    let decoder = unsafe { cast_from_void::<ACDecoder<ArrayStream<'_, u8>>>(this_) };
    let mut cum_table = unsafe {callback(u32::MAX, 0)};
    loop{ 
        match decoder.decode(&SafeCumProbTable::new(&cum_table)){
            (sym,Some(range)) => {
                #[cfg(test)]
                {
                    //println!("sym: {}, range: {}", sym, range);
                    if sym == 11 && range == 480465101258425 {
                        //core::intrinsics::breakpoint();
                    }
                }

                cum_table = unsafe {callback(sym, range)};
            } 
            (sym, None) => {
                unsafe {callback(sym , 1)};
                break;
            }
        }
    }
    SCatStatus::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! read_static {
        ($global:ident) => {
            unsafe{&mut *core::ptr::addr_of_mut!($global)}
        };
    }

    struct RWBuffer<'a, T:Copy + Sized> {
        base: &'a mut [T],
        pos_r: usize,
        pos_w: usize,
    }

    impl<T:Copy + Sized> Buffer<T> for RWBuffer<'_, T> {
        fn is_writeable(&self) -> bool {
            self.pos_w < self.base.len()
        }

        fn write(&mut self, byte: T) -> bool {
            if self.is_writeable() {
                self.base[self.pos_w] = byte;
                self.pos_w += 1;
                return self.is_writeable();
            }
            unreachable!("write to full buffer");
        }
    }

    impl<T:Copy + Sized> Stream<T> for RWBuffer<'_, T> {

        fn read(&mut self) -> Option<T> {
            if self.pos_r < self.pos_w {
                let byte = self.base[self.pos_r];
                self.pos_r += 1;
                Some(byte)
            } else {
                None
            }
        }
    }

    impl<'a, T:Copy + Sized> RWBuffer<'a, T> {
        const fn new(base: &'a mut [T]) -> Self {
            Self{base, pos_r: 0, pos_w: 0}
        }
    }
    const fn zero_vec() -> SCatVectorU64_8 {
        SCatVectorU64_8{data: [0u64; 8]}    
    }
    static mut GLOBAL_BUFFER: RWBuffer<'static, u32> = RWBuffer{
        base: &mut [0u32; 4 * 1024 * 1024],
        pos_r: 0,
        pos_w: 0,
    };

    static mut GLOBAL_TABLE: [SCatVectorU64_8; 4 * 1024] = {
        const INIT: SCatVectorU64_8 = zero_vec();
        [INIT; 4 * 1024]
    };

    fn test_decode(decoder: *mut ACDecoder<ArrayStream<'_, u8>>,encoder: *mut ACEncoder<LinearBuffer<'_, u8>>, bytes: &[u8], callback: SCatModelCallback) {
        scat_init_decoder(decoder as *mut core::ffi::c_void, bytes.as_ptr(), bytes.len());
        let mut table = unsafe {callback(u32::MAX, 0)};
        let mut result: SCatDecodeResult = SCatDecodeResult{sym: 0, range: 0};
        
        while scat_decode(decoder as *mut core::ffi::c_void, table, &mut result) {
            table = unsafe {callback(result.sym, result.range)};
        }
        unsafe {callback(result.sym, result.range)};
        let stream = read_static!(GLOBAL_BUFFER);
        let mut oc = std::boxed::Box::new([0u8; 1024 * 1024]);
        let mut output = RWBuffer::new(&mut *oc);
        scat_init_encoder(encoder as *mut core::ffi::c_void, output.base.as_mut_ptr(), output.base.len());
        let mut length_ptr = 0;
        let s =  {
            scat_encode_all(encoder as *mut core::ffi::c_void, stream.base.as_ptr(), stream.pos_w, callback, &mut length_ptr)
        };
        assert_eq!(s as i32, SCatStatus::Ok as i32);
        output.pos_w = length_ptr as usize;
        //println!("output: {:?}, len: {}", &output.base[0..length_ptr as usize], length_ptr);
        //println!("bytes: {:?}, len: {}", bytes, bytes.len());
        let mut i = 0;
        while let Some(byte) = output.read() {
            assert_eq!(byte, bytes[i], "byte {} not equal, expect {}, but {}", i, bytes[i], byte);
            i += 1;
        }
        assert!(i + unsafe{ encoder.as_mut().unwrap()}.pending_state.count as usize + 7 >= bytes.len(), "output len {} + pending count {} not longer than bytes len {} - 7", i, unsafe{ encoder.as_mut().unwrap()}.pending_state.count, bytes.len());        
    }

    fn test_decode_all(decoder: *mut ACDecoder<ArrayStream<'_, u8>>,encoder: *mut ACEncoder<LinearBuffer<'_, u8>>, bytes: &[u8], callback: SCatModelCallback) {
        scat_init_decoder(decoder as *mut core::ffi::c_void, bytes.as_ptr(), bytes.len());
        let s = scat_decode_all(decoder as *mut core::ffi::c_void, callback);
        assert_eq!(s as i32, SCatStatus::Ok as i32);
        let stream = read_static!(GLOBAL_BUFFER);
        let mut oc = std::boxed::Box::new([0u8; 1024 * 1024]);
        let mut output = RWBuffer::new(&mut *oc);
        scat_init_encoder(encoder as *mut core::ffi::c_void, output.base.as_mut_ptr(), output.base.len());
        let mut length_ptr = 0;
        let s =  {
            scat_encode_all(encoder as *mut core::ffi::c_void, stream.base.as_ptr(), stream.pos_w, callback, &mut length_ptr)
        };
        assert_eq!(s as i32, SCatStatus::Ok as i32);    
        output.pos_w = length_ptr as usize;
        //println!("output: {:?}, len: {}", &output.base[0..length_ptr as usize], length_ptr);
        //println!("bytes: {:?}, len: {}", bytes, bytes.len());
        let mut i = 0;
        while let Some(byte) = output.read() {
            assert_eq!(byte, bytes[i], "byte {} not equal, expect {}, but {}", i, bytes[i], byte);
            i += 1;
        }
        assert!(i + unsafe{ encoder.as_mut().unwrap()}.pending_state.count as usize + 7 >= bytes.len(), "output len {} + pending count {} not longer than bytes len {} - 7", i, unsafe{ encoder.as_mut().unwrap()}.pending_state.count, bytes.len());
        
        
    }

    unsafe extern "C" fn c1<const N: usize>(sym: u32, range: u64) -> SCatCumProbTable {
        if sym != u32::MAX {
            read_static!(GLOBAL_BUFFER).write(sym);
        } else {
            let buf = read_static!(GLOBAL_BUFFER);
            buf.pos_w = 0;
            buf.pos_r = 0;
        }
        let p = if range == 0 {
            1u128<<64
        }else {
            range as u128
        };
        let p = p / N as u128;
        let table = read_static!(GLOBAL_TABLE);
        for i in 0..N {
            table[i/8].data[i%8] = (p*(i as u128)) as u64;
        }
        table[N/8].data[N%8] = range;
        SCatCumProbTable {
            right: N as u32,
            probs: table.as_ptr(),
        }
    }

    unsafe extern "C" fn c2<const FREQ: u8>(sym: u32, range: u64) -> SCatCumProbTable {
        if sym != u32::MAX {
            read_static!(GLOBAL_BUFFER).write(sym);
            assume_assert!(sym < 2);
            let now_prob = u8::MAX - FREQ + 1;
            let next_prob = FREQ;
            let prob = [now_prob, next_prob];
        
            let table = read_static!(GLOBAL_TABLE);
            table[0].data[2] = range;
            table[0].data[0] = 0;
            table[0].data[1] = ((range.wrapping_sub(1)) >> 8) * (prob[sym as usize] as u64);
            SCatCumProbTable {
                right: 2,
                probs: table.as_ptr(),
            }
        } else {
            let buf = read_static!(GLOBAL_BUFFER);
            buf.pos_w = 0;
            buf.pos_r = 0;
            let table = read_static!(GLOBAL_TABLE);
            table[0].data[2] = 0;
            table[0].data[0] = 0;
            table[0].data[1] = ((range.wrapping_sub(1)) >> 8) * (FREQ as u64);
            SCatCumProbTable {
                right: 2,
                probs: table.as_ptr(),
            }
        }
        


    }
    fn bits_polar(bytes: &mut [u8], seed: u32, prob: u32) {
        let a = 1664525u32;
        let c = 1013904223u32;
        let mut x = seed;
        x = a.wrapping_mul(x).wrapping_add(c);
        let mut polar = (x & 1) as u8;
        for i in 0..bytes.len() {
            for _ in 0..8 {
                x = a.wrapping_mul(x).wrapping_add(c);
                if x < prob {
                    polar ^= 1;
                } 
                bytes[i] <<= 1;
                bytes[i] |= polar;
            }
        }
    }

    fn bytes_lcg(bytes: &mut [u8], seed: u32) {
        let a = 1664525u32;
        let c = 1013904223u32;
        let mut x = seed;
        for i in 0..bytes.len() {
            x = a.wrapping_mul(x).wrapping_add(c);
            bytes[i] = x as u8;
        }
    }

    macro_rules! init_all_and_call {
        ($e:ident, $bytes:expr, $callback:ident) => {
            let mut decoder = ACDecoder::<ArrayStream<'_, u8>> {
                sym_encoder: SymEncoderState{
                    inf: 0,
                    sup: 0,
                },
                bytes: ArrayStream{
                    base: &mut [],
                    pos: 0,
                },
                cur_state: 0,
            };
            let mut encoder = ACEncoder::<LinearBuffer<'_, u8>> {
                sym_encoder: SymEncoderState{
                    inf: 0,
                    sup: 0,
                },
                pending_state: PendingState{
                    count: 0,
                    byte_cache: 0,
                    mask: 0,
                },
                buf: LinearBuffer{
                    base: &mut [],
                    pos: 0,
                },
                is_resolve_stopped: false,
            };
            $e(&mut decoder, &mut encoder, $bytes, $callback);
        };
    }

    macro_rules! test_bytes {
        ($bytes:expr) => {
            let callback31 = c1::<31>;
            let c2f40 = c2::<40>;
            //let callback128 = c1::<128>;
            //let callback1024 = c1::<1024>;
            init_all_and_call!(test_decode_all, $bytes, callback31);
            init_all_and_call!(test_decode_all, $bytes, c2f40);
            //init_all_and_call!(test_decode_all, $bytes, callback128);
            //init_all_and_call!(test_decode_all, $bytes, callback1024);
            init_all_and_call!(test_decode, $bytes, callback31);
            init_all_and_call!(test_decode, $bytes, c2f40);
            //init_all_and_call!(test_decode, $bytes, callback128);
            //init_all_and_call!(test_decode, $bytes, callback1024);
        };
    }

    #[test]
    fn it_works() {
        test_bytes!(&[0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8]);

    }

    #[test]
    fn more_bytes() {
        test_bytes!(&[0u8, 1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8, 13u8, 14u8, 15u8]);
        let mut bytes = [0u8; 128];
        for i in 0..128 {
            bytes[i] = ((i as u8).wrapping_add(12)).wrapping_mul(215);
        }
        test_bytes!(&bytes);
    }

    #[test]
    fn test_bits_polar() {
        let mut bytes = [0u8; 128];
        bits_polar(&mut bytes, 0x12345678, u32::MAX/20);
        
        test_bytes!(&bytes);
        bits_polar(&mut bytes, 42, u32::MAX/12);
        test_bytes!(&bytes);
        bits_polar(&mut bytes, 0x1, u32::MAX/2);
        test_bytes!(&bytes);
        bits_polar(&mut bytes, 0x2026, u32::MAX/2);
        test_bytes!(&bytes);
    }

    #[test]
    fn test_long() {
        let mut bytes = std::boxed::Box::new([0u8; 256 * 1024]);
        bits_polar(bytes.as_mut(), 0x1678, u32::MAX/20);
        test_bytes!(&*bytes);
        bits_polar(bytes.as_mut(), 0x1678, u32::MAX/5);
        test_bytes!(&*bytes);
        bytes_lcg(bytes.as_mut(), 0x1678);
        test_bytes!(&*bytes);   
        bytes_lcg(bytes.as_mut(), 0x2026);
        test_bytes!(&*bytes);  
    }
}