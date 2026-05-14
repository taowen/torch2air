# Output-Parallel Herd

问题：output feature/channel 维天然独立时，怎样在一个 xclbin 里用多个 AIE columns 并行？

## 做法

每个 column 负责一段 disjoint output slice：

```text
herd sizes:       1 x herd_cols
per-column rows:  output_tile_features
public weight:    all column weight chunks
per-tile weight:  one column weight chunk in L1
public output:    concatenated output slices
per-tile output:  one column output slice in L1
```

每个 column 复用同一个 input L3 buffer，按 `tile_j * output_tile_features` 读取自己的
weight slice，并把结果写回同一个 output L3 buffer 的 disjoint slice。

## 规则

- 每列写回的 output slice 必须不重叠。
- 每列的 L1 weight tile 保持小而固定；不要为了减少 columns 而把单 tile weight 做得过大。
- 共享 input 可以由每列各自 DMA 到 L1，先保证正确和 routing 稳定。
- schedule 并行和 tile-local compute 优化是两层问题；不要用更大的 herd 掩盖 scalar dot
  body。
