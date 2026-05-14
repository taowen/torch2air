# Packed Weight Tile Layout

问题：压缩权重格式通常有 main payload 和 side data。为了让 AIE external tile kernel
简单稳定，host 应该怎样把它们放进一个 L1 memref？

稳定原则：

- host 负责从模型文件格式解包、重排和补齐。
- tile kernel 只读连续 tile 记录，不解析全局模型格式。
- side data 跟随它服务的 output row，避免第二个 scale buffer 和额外 DMA。

一个可复用 layout 是每个 output row 一个连续记录：

```text
compressed payload: payload_words_per_row
side words:         side_words_per_row
total words:        payload_words_per_row + side_words_per_row
```

验证记录使用 GGUF Q4_K projection。对 `hidden_size=1024`：

```text
blocks_per_row = 4
row_words      = 4 * 36 = 144
scale_words    = 4 * 2  = 8
weight_words   = 152
```

L1 ABI：

```mlir
memref<16x152xi32, 2 : i32>
```

每个 block 的 side words 是 host 从 GGUF Q4_K header 解出的 `d` 和 `dmin` 的
f32 bit pattern：

```text
words[144 + block * 2 + 0] = bitcast_i32(float32(d))
words[144 + block * 2 + 1] = bitcast_i32(float32(dmin))
```

这样 external kernel 只接一个 weight memref，不需要第二个 scale memref，也避免在
AIE tile body 里做 fp16 header conversion。

真实 NPU 结果：

```text
input  embed_tokens -> input_layernorm token 0
weight q_proj.weight rows [0:16)
max_abs 1.1920929e-07
mean_ms 13.684
```

这个 recipe 只证明 layout 和 ABI 正确。当前 dot body 是 correctness-first C++ 写法，
性能不代表正式 projection kernel。
