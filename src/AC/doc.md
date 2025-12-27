# 算术编码器（AC）文档

本文档详细介绍了Shadowcat隐写工具集中的算术编码器组件。该算术编码器主要用于概率整型，即将数据编码到确定性的自回归模型描述的符号流概率分布上。

## 1. 概述

算术编码是一种高效的熵编码算法，常用于数据压缩领域。本实现的常规用法与通常的算术编码相反：先将二进制数据解码为符号流，再对符号流进行编码恢复原始的二进制数据。

## 2. 核心概念

### 2.1 编码范围

- 编码器内部表示概率的精度是有限的，且会随着编码的过程动态变化
- 为保证编码器在概率分布上采样的精度，量化概率到指定精度需要由概率模型实现
- 编码器返回的range字段表示当前编码范围的宽度，其总满足2^48<=range<=2^64
- 由于使用无符号64位整形，当编码范围为2^64时range字段的值为0（在模2^64的同余意义下等价）

### 2.2 符号与概率表示

若总共有n个符号，对于第i个符号(从0计数)：
- f_i：表示其频率
- c_i：表示累计频率，c_i=sum_{j=0}^{i-1} f_j
- p_i：表示编码时的概率，p_i=f_i/range
- 符号i占据编码范围中[c_i, c_i+p_i)的区间
- 总应该有range = sum(f_j) = c_n, sum(p_i) = 1

### 2.3 累计概率表(CumProbTable)

- 长度为n+1的数组，其中第i个元素表示前i个符号的累计频率
- 第0个元素必然为0，第n+1个元素必然为range
- 以8个元素为一组进行传递，每个VectorU64_8表示8个符号的累计概率
- right字段表示符号空间的右边界元素(值为range)的下标，其值也刚好等于符号数量

## 3. API 参考

### 3.1 数据结构

#### 3.1.1 SCatDecodeResult
```c
typedef struct {
    uint64_t range;          // 解码后的下一个编码范围
    uint32_t sym;            // 解码出的符号值
} SCatDecodeResult;
```

#### 3.1.2 SCatStructLayout
```c
typedef struct {
    size_t size;             // 结构体大小（字节）
    size_t align;            // 结构体对齐要求（字节）
} SCatStructLayout;
```

#### 3.1.3 SCatVectorU64_8
```c
typedef struct {
    uint64_t data[8];        // 64位整数数组
} SCatVectorU64_8;
```

#### 3.1.4 SCatCumProbTable
```c
typedef struct {
    uint32_t right;          // 符号空间的右边界（符号数量）
    const SCatVectorU64_8* probs; // 指向累积概率数组的指针
} SCatCumProbTable;
```

#### 3.1.5 SCatStatus (枚举)
```c
enum SCatStatus {
    STATUS_OK = 0,                  // 操作成功
    STATUS_NO_ENOUGH_CACHE = 1,     // 缓存空间不足
    STATUS_NO_CALLBACK = 101,       // 回调函数未设置
    STATUS_INVALID_PARAM = 102,     // 参数无效
    STATUS_INVALID_CALLBACK_RETURN = 103, // 回调函数返回无效数据
    STATUS_CALL_BEFORE_INIT = 104   // 初始化前调用
};
```

#### 3.1.6 SCatBufferChangeResult
```c
typedef struct {
    uint64_t last_range;     // 当前的编码范围
    size_t last_buf_pos;     // 切换前的缓冲区位置
} SCatBufferChangeResult;
```

### 3.2 回调函数类型

#### 3.2.1 SCatModelCallback
```c
typedef SCatCumProbTable (*SCatModelCallback)(uint32_t, uint64_t);
```

- 参数1：上一次概率表中最终采样到的符号的下标（初始化时传入UINT32_MAX表示初始化阶段）
- 参数2：当前编码范围
- 返回值：包含累积概率信息的结构体

### 3.3 函数声明

#### 3.3.1 内存布局获取函数

```c
SCatStructLayout scat_encoder_layout(void);
SCatStructLayout scat_decoder_layout(void);
```
- 功能：获取编码器/解码器结构体的布局信息，用于动态分配内存空间
- 返回值：包含结构体大小和对齐要求的信息

#### 3.3.2 初始化函数

```c
enum SCatStatus scat_init_encoder(void* this_, uint8_t* buffer, size_t buffer_size);
enum SCatStatus scat_init_decoder(void* this_, const uint8_t* bytes, size_t bytes_size);
```
- 参数this_：指向为编码器/解码器结构体分配的空间的指针
- 参数buffer/bytes：输出/输入缓冲区指针
- 参数buffer_size/bytes_size：缓冲区大小
- 返回值：操作状态码，STATUS_OK表示成功

#### 3.3.3 编码函数

```c
bool scat_encode(void* this_, uint64_t cum, uint64_t freq, uint64_t* range_ptr);
enum SCatStatus scat_encode_all(void* this_, const uint32_t* syms, size_t syms_size, SCatModelCallback callback, uint32_t* length_ptr);
bool scat_change_buffer(void* this_, uint8_t* buffer, size_t buffer_size, SCatBufferChangeResult* result);
```

**scat_encode**：
- 功能：编码单个符号
- 参数cum：符号的累积概率
- 参数freq：符号的频率
- 参数range_ptr：输出参数，接收更新后的编码范围
- 返回值：缓冲区是否仍有空间，false表示需要切换缓冲区

**scat_encode_all**：
- 功能：一次性编码整个符号流
- 参数syms：待编码符号流指针
- 参数syms_size：符号流长度
- 参数callback：概率表回调函数
- 参数length_ptr：输出参数，接收编码后的字节长度
- 返回值：操作状态码

**scat_change_buffer**：
- 功能：切换编码器缓冲区
- 参数buffer：新的输出缓冲区指针
- 参数buffer_size：新的输出缓冲区大小
- 参数result：输出参数，包括之前的缓冲区使用量和当前的编码范围
- 返回值：新缓冲区是否有足够空间，false表示需要再次切换

#### 3.3.4 解码函数

```c
bool scat_decode(void* this_, SCatCumProbTable table, SCatDecodeResult* result);
enum SCatStatus scat_decode_all(void* this_, SCatModelCallback callback);
```

**scat_decode**：
- 功能：解码单个符号
- 参数table：当前符号集的累积概率表
- 参数result：输出参数，保存解码结果
- 返回值：是否还有更多符号可解码，false表示输入流结束

**scat_decode_all**：
- 功能：一次性解码整个字节流
- 参数callback：回调函数，用于处理解码结果并返回新的累积概率表
- 返回值：操作状态码

## 4. 特殊处理与注意事项

### 4.1 编码器冲洗

- 由于算术编码的特殊性，解码器不会保证输入字节流的最后7个字节包含的信息被编码
- 调用方需要在数据末端填充随机比特，以确保尾部概率分布仍保持与模型一致
- 在特定情况下，末尾连续的1或0可能导致部分字节在解码时不被输出
- 编码器不能保证指定长度的数据解码再编码后的长度总相等，调用方需要自行传递长度字段并截断输出

### 4.2 概率分布

- 当解码器输入的字节流不能与随机比特流区分时，解码器输出同样不能与从概率模型中随机采样的结果区分
- 解码器输入的统计特性会反应在解码结果中

### 4.3 模型确定性

- 如果概率模型不能对相同的输入符号序列总返回相同的累计概率表，则不能保证编解码操作的可逆性

## 5. 使用示例

### 5.1 基本编码流程

```c
// 1. 获取编码器布局并分配内存
SCatStructLayout encoder_layout = scat_encoder_layout();
void* encoder = malloc(encoder_layout.size);

// 2. 初始化编码器
uint8_t output_buffer[1024];
scat_init_encoder(encoder, output_buffer, sizeof(output_buffer));

// 3. 编码符号
uint64_t range;
bool success = scat_encode(encoder, cum_prob, freq, &range);

// 4. 处理缓冲区切换（如果需要）
if (!success) {
    SCatBufferChangeResult result;
    uint8_t new_buffer[1024];
    success = scat_change_buffer(encoder, new_buffer, sizeof(new_buffer), &result);
    // 处理旧缓冲区中的数据 (result.last_buf_pos 字节)
}

// 5. 释放资源
free(encoder);
```

### 5.2 基本解码流程

```c
// 1. 获取解码器布局并分配内存
SCatStructLayout decoder_layout = scat_decoder_layout();
void* decoder = malloc(decoder_layout.size);

// 2. 初始化解码器
const uint8_t* input_data = ...;
size_t input_size = ...;
scat_init_decoder(decoder, input_data, input_size);

// 3. 定义概率模型回调
SCatCumProbTable model_callback(uint32_t last_sym, uint64_t range) {
    // 构建累积概率表
    SCatVectorU64_8 probs;
    // 设置概率值...
    
    SCatCumProbTable table;
    table.right = symbol_count;
    table.probs = &probs;
    return table;
}

// 4. 执行解码
scat_decode_all(decoder, model_callback);

// 5. 释放资源
free(decoder);
```

## 6. 限制与注意事项

1. 编码器和解码器使用动态内存布局，需要通过scat_encoder_layout()和scat_decoder_layout()获取正确的内存大小和对齐要求
2. 概率模型的实现应确保累积概率表的正确性，以保证编码和解码的一致性
3. 由于性能和精度考量，回调函数需要特别注意累积概率的计算
4. 使用时应当特别关注缓冲区管理，及时处理缓冲区切换情况
5. 算术编码器接口不依赖于具体的Rust实现，可以与任何实现了这些接口的后端配合使用