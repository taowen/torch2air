# Producer Consumer Channel

## 问题

一个 xclbin 内两个 stage 需要传递中间 tile：

```text
input -> producer -> consumer -> output
```

目标是中间结果不出现在 public ABI，也不写回 host-visible BO。

## 做法

producer 和 consumer 放在同一个 `air.segment`，用一条 channel 传 L1 tile：

```text
air.channel @ProducerToConsumer []

producer:
  dma_memcpy_nd input L3 -> L1
  compute
  ChannelPut ProducerToConsumer

consumer:
  ChannelGet ProducerToConsumer
  compute
  dma_memcpy_nd L1 -> output L3
```

## 结论

- `ChannelGet` 可以表达 consumer 等待 producer 的顺序。
- host runtime sequence 只需要 public input fill 和 final output drain。
- tile-to-tile 中间数据不会经过 host。

## 不要

- 不要把 stage 间中间结果先写回 public BO 再读回来。
- 不要在同一个 xclbin 能表达时提前拆成多个 xclbin。

## 检查

lowered IR 应该包含 tile-to-tile `aie.flow`。如果中间结果出现在 public ABI
或 host runtime sequence 里，说明 handoff 仍然经过了 host-visible BO。
