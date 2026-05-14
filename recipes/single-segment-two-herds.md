# Single Segment, Multiple Herds

## 问题

一个 xclbin 内需要放多个互不依赖的 compute island，例如：

```text
input_a -> herd A -> output_a
input_b -> herd B -> output_b
```

## 做法

使用一个 public function、一个 `air.launch`、一个 `air.segment`，segment 内放多个 herd：

```text
air.launch
  air.segment
    air.herd @A
      dma_memcpy_nd input_a L3 -> L1
      compute
      dma_memcpy_nd L1 -> output_a L3
    air.herd @B
      dma_memcpy_nd input_b L3 -> L1
      compute
      dma_memcpy_nd L1 -> output_b L3
```

## 结论

- `air-place-herds` 会把多个 `1x1` herd 放到不同 tile。
- 简单 L3/L1 tile 搬运优先用 `air.dma_memcpy_nd`。
- public ABI 直接保留多个 BO，不需要额外 wrapper。

## 不要

- 不要为了独立 compute island 先拆成多个 xclbin。
- 不要引入 graph/manifest 描述 AIR 已经能表达的拓扑。

验证状态：真实 NPU 通过；观察到两个 herd 分别放到不同 AIE tile。
