# Static Token Buckets

问题：NPU graph 需要 fixed shape，但 prefill token 数是动态的。应该怎样把动态 token 长度
落到 AIR/NPU graph？

## 做法

使用固定 token bucket。host 把长序列拆成多个 bucket launch，tail 不足时 pad/mask：

```text
public input:   memref<BUCKETx...>
public weight:  fixed tile weight
public output:  memref<BUCKETx...>
herd:           token lanes x feature lanes
per tile:       small fixed number of tokens
```

## 规则

- bucket size 是 ABI 的一部分，不在 public memref 上表达动态 token 数。
- 长序列由 host 拆成多个完整 bucket launch。
- tail 由 host pad，mask 或 valid length 传给需要 mask 的 stage。
- 单个 tile 内 token loop 要短；长 loop 容易放大 DMA ordering 和 L1 状态问题。
- 更大 bucket 必须用真实 NPU 对拍确认，不能只看编译通过。

## 适用边界

projection 这类逐 token 独立 stage 可以直接拆 bucket。attention/KV 这类依赖历史长度的 stage
还需要显式 `start_position`、`valid_length` 或 mask 策略。
