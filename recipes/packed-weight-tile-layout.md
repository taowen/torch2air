# Packed Weight Tile Layout

问题：压缩权重格式通常有 main payload 和 side data。为了让 AIE external tile kernel 简单
稳定，host 应该怎样把它们放进一个 L1 memref？

## 做法

每个 output row 使用一个连续 record：

```text
compressed payload: payload_words_per_row
side words:         side_words_per_row
total words:        payload_words_per_row + side_words_per_row
```

host 负责从模型文件格式切片、补 side data、必要时做窄类型到 ABI word 的临时 widening。
tile kernel 只消费连续 row records，不解析全局模型文件。

## Q4_K 例子

对 `hidden_size=1024`：

```text
blocks_per_row = 4
payload_words  = 4 * 36 = 144
side_words     = 4 * 2  = 8
weight_words   = 152
```

每个 block 的 side words 是 `d` 和 `dmin` 的 f32 bit pattern：

```text
words[144 + block * 2 + 0] = bitcast_i32(float32(d))
words[144 + block * 2 + 1] = bitcast_i32(float32(dmin))
```

## Q6_K 例子

Q6_K block 是 `105` 个 `uint16` halfword：

```text
ql[128 bytes], qh[64 bytes], scales[16 bytes], d[fp16]
```

当前简单 ABI 可以先把 halfword widen 成 `i32`，再追加每个 block 的 f32 `d`：

```text
blocks_per_row = 4
payload_words  = 4 * 105 = 420
side_words     = 4
weight_words   = 424
```

## 取舍

- 单个 weight memref 比额外 scale buffer 更简单，DMA 更少。
- host 只解 block-level scale side data，不 host-dequantize 完整权重值。
- widening 会增加 L1 footprint；稳定后再单独做 compact i16/byte ABI。
