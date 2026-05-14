# Output Feature Chunking

问题：一个 fixed-shape NPU kernel 只能覆盖部分 output feature/channel 时，怎样拼出完整
output tensor？

## 做法

让一个 xclbin 固定计算一段 output features：

```text
input:        shared input tile or vector
weight_chunk: output_chunk_features x per_feature_words
output_chunk: output_chunk_features
```

host 按 output feature 切完整权重：

```text
rows [0:chunk)       -> launch
rows [chunk:2*chunk) -> launch
...
```

每次 launch 的输出写入完整结果的对应 slice。生产 runtime 可以把这些 slice 对应到 shared
device BO；验证 runner 可以回读后拼 host array。

## 规则

- chunk size 必须整除 full output feature count。
- public output shape 固定为当前 chunk，不在 AIR 里引入动态 shape。
- 增大单 tile rows 前先检查 L1 footprint 和 bank allocation。
- 如果一个大 tile 触发 L1 warning，优先用多列 herd 或更多 host launches，而不是硬塞进
  单个 tile。
