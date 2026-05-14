# Pack Multiple Outputs Into One Channel

## 问题

upstream stage 产生多个强配对小 buffer，例如 `cos` 和 `sin`。如果直接建两条内部 channel：

```text
producer -> Cos channel -> consumer
producer -> Sin channel -> consumer
```

consumer tile 会同时接多个 tile-to-tile stream，再叠加 host input/weight packet，容易触发：

```text
aie.masterset op targets same destination DMA
```

## 做法

把强配对输出 pack 成一个 L1 payload，用一条 channel 传：

```text
trig_l1: memref<2x128xf32, 2>

producer:
  trig_l1[0, :] = cos[:]
  trig_l1[1, :] = sin[:]
  ChannelPut Trig

consumer:
  ChannelGet Trig
  cos[:] = trig_l1[0, :]
  sin[:] = trig_l1[1, :]
  compute(input, weight, cos, sin)
```

## 结论

- 强配对多输出不要靠多条 channel 的隐式顺序。
- pack 后只有一条 tile-to-tile flow，routing 更稳。
- 适合“小 payload、必须成对消费”的 stage handoff。

## 不要

- 不要给 `cos` / `sin` 这类强配对小输出各建一条内部 channel。
- 不要默认用动态 L3 scratch 做 intra-xclbin stage handoff；先验证 lowering 和 runtime
  sequence。

验证状态：`4/16 tokens`、`start_position=0/7` 均在真实 NPU 对拍通过。
