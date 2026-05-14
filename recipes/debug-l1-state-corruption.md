# Debug L1 State Corruption

问题：NPU 输出表现为后半 token 变 0、某一行重复前一行，或状态像被旧值污染，怎样系统定位？

先看 L1 address map，而不是先调 tile size。AIE stack、显式 L1 buffer、DMA BD 和 lock
都在同一块 tile memory 里；shape 看起来没超 64 KiB，也可能因为 bank 或地址布局出问题。

检查文件：

```text
<work_dir>/aiecc/input_with_addresses.mlir
<work_dir>/aiecc/*_core_*.ld.script
```

重点看：

- stack 起点和大小；
- `memref<..., 2>` buffer 的地址、bank 和大小；
- bank-aware allocation warning；
- herd 物理行列数是否造成 shim DMA channel 压力；
- producer/consumer 的 channel 顺序是否和 runtime sequence 一致。

稳定经验：

- 8-token 这类固定 bucket 先把物理 token 并行度限制到 4 行，再在 tile 内做短 loop。
- 有顺序约束的多输出优先用 AIR channel 表达，不依赖 loop 里多次 DMA lowering 成期望顺序。
- `air-place-herds` 的 placement 诊断不一定致命；最终以 `air-to-aie`、`aiecc` 和真实 NPU
  对拍为准。
