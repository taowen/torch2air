# Output-Parallel Herd

问题：output feature/channel 维天然独立时，怎样在一个 xclbin 里用多个 AIE columns
并行？

稳定形态是每个 column 负责一段 disjoint output slice：

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

验证记录使用 GGUF Q4_K projection：

| herd_cols | output rows | max_abs | mean_ms |
| ---: | ---: | ---: | ---: |
| 2 | 32 | `2.3841858e-07` | `13.932` |
| 4 | 64 | `3.5762787e-07` | `13.880` |

结论：

- 这个 schedule 没有 packet routing error。
- 4-column herd 比单 tile 64 rows 更适合当前 L1：每个 tile 仍是 `16x152xi32` weight
  buffer，不触发 `64x152xi32` 的 bank-aware allocation warning。
- 当前总耗时仍由 correctness-first dot body 主导；多列证明了 schedule 可用，但真正提速
  还需要 vectorized tile body。
