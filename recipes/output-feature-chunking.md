# Output Feature Chunking

问题：一个 kernel 暂时只能覆盖部分 output feature/channel 时，怎样先用固定形状 NPU
graph 拼出完整 output tensor？

稳定做法是让一个 xclbin 固定计算一段 output features：

```text
input:        shared input tile or vector
weight_chunk: output_chunk_features x per_feature_words
output_chunk: output_chunk_features
```

host 按 output feature 切完整权重：

```text
rows [0:chunk) -> launch
rows [chunk:2*chunk) -> launch
...
```

每次 launch 的输出写入完整结果数组的对应 slice。这个模式先验证 stage 编排和数值正确性；
正式 runtime 后续应把这些 slice 对应到 shared device BO。

验证记录使用 GGUF Q4_K projection，单列 herd：

| output_tile_rows | launch count | max_abs | mean_ms | 备注 |
| --- | ---: | ---: | ---: | --- |
| 16 | 128 | `2.8610229e-06` | `1748.456` | 无 L1 allocation warning |
| 32 | 64 | `2.8610229e-06` | `1725.858` | `32x152xi32` bank-aware allocation failed |
| 64 | 32 | `2.8610229e-06` | `1696.613` | `64x152xi32` bank-aware allocation failed |

结论：

- 单列 output chunk 正确，但当前 external dot body 太慢。
- 32/64-row tile 虽然能跑通，L1 bank allocation 已经不干净。
- 保守策略是保持较小的 per-tile output chunk；提速优先做 vectorized tile body 和多列
  并行，而不是继续增大单 tile rows。
