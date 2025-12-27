#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 算术编码器头文件
 * 
 * 本头文件定义了这一算术编码器的C语言接口，提供了编码和解码功能的结构体、枚举和函数声明。
 * 算术编码是一种高效的熵编码算法，常用于数据压缩领域，但这个实现主要用于概率整型，即编码数据到确定性的自回归模型描述的符号流概率分布上，故这一实现的常规用法与通常的算术编码相反，（先将二进制数据解码为符号流，再对符号流进行编码恢复原始的二进制数据），下文仍按照常规的转换任意符号流到字节流为编码器，转换字节流到任意符号流为解码器来称呼
 * 下文中有时会不区分频率和其表示的概率，视上下文而定。
 *
 * ##编码范围:
 * 编码器内部表示概率的精度是有限的，且会随着编码的过程动态变化
 * 为了保证编码器在概率分布上采样的精度，量化概率到指定精度需要由概率模型实现，以免二次量化降低精度
 * 编码器返回range字段，表示当前编码范围的宽度，其总满足2^48<=range<=2^64,由于使用无符号64位整形，当编码范围为2^64时range字段的值应该为0(在模2^64的同余意义下等价，或者也可以认为是无符号整型上溢的结果)
 * 若总共有n个符号，我们设对于第i个符号(从0计数)，其频率为f_i, 则其累计频率c_i=sum_{j=0}^{i-1} f_j, 即前i个符号的频率和
 * 特别地，c_0=0, 即第0个符号的累计频率为0
 * 第i个符号的编码时的概率p_i=f_i/range
 * 此时总应该有range = sum(f_j) = c_n, sum(p_i) = 1, 每个符号占据编码范围中[c_i, c_i+p_i)的区间
 * 注意: range始终为2的幂仅当每一次传入的累计频率表的每个元素的频率都是2的倍数时成立，此时算数编码器退化到huffman编码，故绝对不可以在假定range为2的幂的前提下编写回调函数
 * 
 * ## 累计概率表(CumProbTable)
 * 累计概率表是一个长度为n+1的数组，其中第i个元素表示前i个符号的累计频率(从0计数)，第0个元素必然为0，第n+1个元素必然为range, 元素应该始终随下标单调递增
 * 为了计算效率，累计概率表以8个元素为一组进行传递，每个VectorU64_8表示8个符号的累计概率，最后一个VectorU64_8可能不足8个元素，但仍应该填充空余部分
 * 累计概率表的right字段表示符号空间的右边界元素(值为range)的下标(从0计数)，其值也刚好等于符号数量
 * 
 * ## 编码器冲洗
 * 因为算术编码的特殊性，总不可能保证在任意的数据和模型下，编码再解码的结果长度与原数据长度相等
 * 这一实现中，Decoder不会保证输入字节流的最后7个字节包含的信息被编码，其中一部分会留在其状态中，故调用方需要自行在数据末端填充，只有填充随机比特才可以确保尾部概率分布仍保持与模型一致
 * 注意如果待编码的倒数第7个字节所有位与倒数第8个字节的最后一位相同，则可能会在编码的最后保留未决计数（注意不是必然的，取决于模型输出和具体数据），这会导致倒数第8个字节在解码时不被输出，如果末尾连续的1或0的数量更多，不被输出的字节可以任意多
 * 这一实现中，Encoder不能保证在任意模型下，指定长度的数据解码再编码后的长度总相等，特定的概率模型可能会使得Encoder输出最后7字节中的任意个字节，调用方需要自行传递长度字段并截断输出，以确保编码后的长度与输入长度相等
 * (因为这一实现支持流式编码，长度字段可以包含在载荷中)
 *
 * ## 概率分布
 * 当解码器输入的字节流不能与随机比特流区分时，解码器输出同样不能与从概率模型中随机采样的结果区分
 * 解码器输入的统计特性同样会反应在解码结果中，例如如果输入的字节流中0的数量多，解码器输出的在累计概率表中排列靠前的符号数量也会伴随增多
 *
 * ## 模型确定性
 * 如果概率模型不能对相同的输入符号序列总返回相同的累计概率表，则不能保证编解码操作的可逆性
 */


/**
 * @brief 解码结果结构体
 * 
 * 保存解码单个符号的结果信息。
 */
typedef struct {
    uint64_t range;          /**< 解码后的下一个编码范围 */
    uint32_t sym;            /**< 解码出的符号值 */
} SCatDecodeResult;

/**
 * @brief 结构体布局信息结构体
 * 
 * 提供结构体的大小和对齐要求信息，用于动态内存分配。
 */
typedef struct {
    size_t size;             /**< 结构体大小（字节） */
    size_t align;            /**< 结构体对齐要求（字节） */
} SCatStructLayout;

/**
 * @brief 8个64位整数的向量结构体
 * 
 * 用于累计概率表数据，8个为一组便于向量化。
 */
typedef struct {
    uint64_t data[8];        /**< 64位整数数组 */
} SCatVectorU64_8;

/**
 * @brief 累积概率表结构体
 * 
 * 传递符号的累积概率分布，用于编码和解码过程。
 * 最后一个累积概率为符号空间的右边界(应该与传入回调的range相等，不然若实际编码范围落入实际右边界与range之间会编码错误)，用于确定编码范围的上限。
 * 如这一次有4个备选符号，频率比为[2, 1, 2, 3]，上一次的range为16(2 + 1 + 2 + 3 = 8, 16 = 8 * 2)，则累积概率表应为[0, 4(2 * 2 + 0), 6(1 * 2 + 4), 10(2 * 2 + 6), 16(3 * 2 + 10)]，右边界为4(5个元素,最右端下标为4),虽然不足8个累计概率值(5个)，仍然需要传递一整个VectorU64_8。
 * 在wasm32中其刚好为64位，可以直接返回，无需返回指针
 */
typedef struct {
    uint32_t right;          /**< 符号空间的右边界（符号数量） */
    const SCatVectorU64_8* probs;/**< 指向累积概率数组的指针 */
} SCatCumProbTable;

/**
 * @brief 操作状态枚举
 * 
 * 定义各种函数操作可能返回的状态码。
 */
enum SCatStatus {
    STATUS_OK = 0,                  /**< 操作成功 */
    STATUS_NO_ENOUGH_CACHE = 1,     /**< 缓存空间不足 */
    STATUS_NO_CALLBACK = 101,       /**< 回调函数未设置 */
    STATUS_INVALID_PARAM = 102,     /**< 参数无效 */
    STATUS_INVALID_CALLBACK_RETURN = 103, /**< 回调函数返回无效数据 */
    STATUS_CALL_BEFORE_INIT = 104   /**< 初始化前调用 */
};

/**
 * @brief 概率模型回调函数类型定义
 * 
 * 这一回调将被用于获取下一个符号的概率表信息，支持复杂的自回归概率模型
 * 
 * @param sym 上一次概率表中最终采样到的符号的下标，特别地，初始化时因为不存在上一个符号，将传入特殊值UINT32_MAX占位并表示初始化阶段
 * @param range 当前编码范围，回调函数应该将概率量化到这个范围，这是为了避免编码器内部二次进行概率量化，导致不必要地损失精度和性能
 * @return CumProbTable 包含累积概率信息的结构体
 */
typedef SCatCumProbTable (*SCatModelCallback)(uint32_t, uint64_t);

/**
 * @brief 缓冲区切换结果结构体
 * 
 * 保存编码器切换缓冲区时的状态信息，用于后续编码操作的恢复。
 */
typedef struct {
    uint64_t last_range;     /**< 当前的编码范围，由于当缓冲区已满时下一个编码范围无法返回，故需延迟到缓冲区切换后；当上一次缓冲区未满时切换缓冲区这个字段会返回和上次一样的值 */
    size_t last_buf_pos;     /**< 切换前的缓冲区位置 */
} SCatBufferChangeResult;

/**
 * @brief 获取编码器结构体的布局信息
 * 
 * 用于调用方动态分配编码器内存空间。
 * 
 * @return StructLayout 包含编码器结构体大小和对齐要求的信息
 */
SCatStructLayout scat_encoder_layout(void);

/**
 * @brief 获取解码器结构体的布局信息
 * 
 * 用于调用方动态分配解码器内存空间。
 * 
 * @return StructLayout 包含解码器结构体大小和对齐要求的信息
 */
SCatStructLayout scat_decoder_layout(void);

/**
 * @brief 初始化编码器
 * 
 * 配置编码器状态并关联输出缓冲区。
 * 
 * @param this_ 指向为编码器结构体分配的空间的指针
 * @param buffer 输出缓冲区指针
 * @param buffer_size 输出缓冲区大小（字节）
 * @return enum Status 操作状态码
 */
enum SCatStatus scat_init_encoder(void* this_, uint8_t* buffer, size_t buffer_size);

/**
 * @brief 编码单个符号(推荐使用)
 * 
 * 将一个符号按照指定的累积概率和频率进行编码。
 * 
 * @param this_ 编码器结构体指针
 * @param cum 符号的累积概率（相对于总概率空间）
 * @param freq 符号的频率（概率空间大小）
 * @param range_ptr 输出参数，接收更新后的编码范围
 * @return bool 缓冲区是否仍有空间，false表示需要切换缓冲区
 */
bool scat_encode(void* this_, uint64_t cum, uint64_t freq, uint64_t* range_ptr);


/**
 * @brief 编码所有符号
 * 
 * 一次性编码整个符号流，使用回调函数获取动态概率表，目前版本不能保证因编码器缓冲区不足而失败时切换缓冲区可以继续正常编码。
 * 
 * @param this_ 编码器结构体指针
 * @param syms 待编码符号流指针
 * @param syms_size 符号流长度
 * @param callback 概率表回调函数
 * @param length_ptr 输出参数，接收编码后的字节长度
 * @return enum Status 操作状态码
 */
enum SCatStatus scat_encode_all(void* this_, const uint32_t* syms, size_t syms_size, SCatModelCallback callback, uint32_t* length_ptr);

/**
 * @brief 切换编码器缓冲区
 * 
 * 当当前缓冲区空间不足时，切换到新的缓冲区继续编码，在当前缓冲区尚存有空间时也可用。
 * 
 * @param this_ 编码器结构体指针
 * @param buffer 新的输出缓冲区指针
 * @param buffer_size 新的输出缓冲区大小（字节）
 * @param result 输出参数，包括之前的缓冲区使用量（字节）和当前的编码范围（如果切换后缓冲区有足够空间输出编码器中积存的数据）
 * @return bool 新缓冲区是否有足够空间，false表示需要再次切换
 */
bool scat_change_buffer(void* this_, uint8_t* buffer, size_t buffer_size, SCatBufferChangeResult* result);

/**
 * @brief 初始化解码器
 * 
 * 配置解码器状态并关联输入字节流。
 * 
 * @param this_ 指向为解码器结构体分配的空间的指针
 * @param bytes 指向待解码的字节流开头的指针
 * @param bytes_size 待解码字节流的大小（字节）
 * @return enum Status 操作状态码，STATUS_OK(0)表示成功
 */
enum SCatStatus scat_init_decoder(void* this_, const uint8_t* bytes, size_t bytes_size);

/**
 * @brief 解码单个符号
 * 
 * 根据概率表解码出下一个符号。
 * 
 * @param this_ 解码器结构体指针
 * @param table 当前符号集的累积概率表
 * @param result 输出参数，保存解码结果
 * @return bool 是否还有更多符号可解码，false表示输入流结束
 */
bool scat_decode(void* this_, SCatCumProbTable table, SCatDecodeResult* result);

/**
 * @brief 解码所有符号(推荐使用)
 * 
 * 一次性解码整个字节流，解码结果的保存和可能的后续处理由回调函数负责。
 * 
 * @param this_ 解码器结构体指针
 * @param callback 回调函数，用于处理解码结果并返回新的累积概率表
 * @return enum Status 操作状态码
 */
enum SCatStatus scat_decode_all(void* this_, SCatModelCallback callback);

#ifdef __cplusplus
}
#endif