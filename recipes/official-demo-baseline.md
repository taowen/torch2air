# Official Demo Baseline

问题：自己的 AIR kernel 编译失败、链接失败或 NPU 数值错时，什么时候应该先跑官方 demo？

## 做法

先找同类最小官方 demo，并用更小 shape 跑真实 NPU：

```bash
source scripts/npu-common.sh
run_npu_make baseline "$PWD/path/to/Makefile" run KEY=small VALUE=small
```

官方 demo 跑通，说明硬件、XRT、Peano、`aircc/aiecc`、`pyxrt` 和基本 AIR flow 是通的；
接下来重点查自己的 ABI、cwd、`.o` 放置、tile shape 和数据布局。

官方 demo 跑不通，先不要改算子数学。按顺序查：

```text
command -v aircc
python3 是否来自当前项目
build/air_project/*.o 是否存在
link_with 文件名是否一致
compile-only 是否能生成中间产物
```

## 经验

- RoPE/RMSNorm/attention 这类复杂算子，官方 demo 通常已经有可打印 IR、compile-only、
  run、profile、verify 的入口。
- 官方 RoPE 使用 `private func + link_with + L1 memref + external vector kernel`；这条路
  在真实 NPU 上可行。
- 官方 weighted RMSNorm 使用 Python DSL 的 `vector.transfer_read/write` 和
  `vector.reduction`；纯 Python AIR vector path 也可行。
- 失败定位先分清工具链、链接、DMA、external call、数学实现，不要从最终 output 直接猜。
