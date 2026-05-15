# Multi Launch PDI Packaging

## 问题

一个 public AIR function 内有多个 `air.launch`：

```text
launch 1: source + weight -> normed
launch 2: normed + lut -> output
```

lowering 后可能出现多个 `aie.device` / PDI：

```text
@rms_seg
@rope_seg
@main
```

当前工具链路径下，`aiecc --aie-generate-xclbin` 会为每个 `aie.device` 调一次
`xclbinutil`，但每次都是重新创建/替换同一个 xclbin。最后留下来的通常是最后一个
device，也就是 public wrapper 的 `main.pdi`，compute segment 的 PDI 不在最终 xclbin 里。

真实 NPU 侧表现不是数值误差，而是 segment 没有正确加载：中间 BO 或最终 BO 全 0、只写
前几个 tile/head，或者 runtime 等待 DMA/segment 完成直到 timeout。

## 现象

`aiecc` 日志里能看到多个 AIE_PARTITION 被连续写到同一个 xclbin：

```text
... rope_seg_aie_partition.json -> run.xclbin
... rms_seg_aie_partition.json  -> run.xclbin
... main_aie_partition.json     -> run.xclbin
```

最终 `run.xclbin` 可能只有最后一次写入的 PDI。

如果 `PATH` 里没有 `/opt/xilinx/xrt/bin`，`aiecc` 还可能成功生成 `insts.bin` 和 tmp PDI，
但不生成 xclbin。调用方必须显式检查 xclbin 文件存在。

## 检查

用 XRT 工具 dump 最终 xclbin：

```bash
/opt/xilinx/xrt/bin/xclbinutil \
  --dump-section AIE_PARTITION:JSON:/tmp/aie_partition.json \
  --input run.xclbin

rg -n 'uuid|file_name' /tmp/aie_partition.json
```

如果只看到 `main` 对应的一个 PDI，而 `aiecc/` 目录里还有 `rms_seg.pdi`、
`rope_seg.pdi`，说明 packaging 丢了 compute PDI。

同时检查 lowered runtime sequence：

```text
aie.runtime_sequence @run(...)
  aiex.configure @rms_seg
    aiex.run @rms_seg_sequence(...)
  aiex.configure @rope_seg
    aiex.run @rope_seg_sequence(...)
```

如果 sequence 要 configure segment，但最终 xclbin 没有对应 PDI，host 调用 ABI 看起来仍然
能启动，结果也不会可靠。

## 已验证边界

- `.air.mlir -> .aie.mlir -> .npu.mlir` 能展开出所有 launch，不代表 xclbin 包含所有 PDI。
- `insts.bin` 可以包含 `aiex.configure @segment` / `aiex.run @segment_sequence`，但 xclbin
  里缺 segment PDI 时，运行仍然不会产生正确 output。
- `xclbinutil --add-merge-section AIE_PARTITION:JSON:...` 不支持向已有 AIE_PARTITION 追加
  metadata。
- 手动拼一个包含多个 PDI 的 AIE_PARTITION JSON 可以让 dump 看见多个 UUID，但不等于 runtime
  能按 `aiex.configure @segment` 正确加载这些 segment；实测会 mismatch 或 timeout。
- `aiecc --sequence-name=...` 可以生成指定 sequence 的 `insts.bin`，但这里没有产生可用的
  xclbin 打包结果。
- `aiecc --xclbin-input` 是追加另一个 kernel/PDI 的语义。同名 `MLIR_AIE` kernel 会报
  `Kernel name already exists`；换不同 kernel 名可以生成 xclbin，但仍然不是把
  `rms_seg`、`rope_seg` 合并进同一个 runtime kernel。

## 做法

- multi-launch 产物必须检查最终 xclbin 的 `AIE_PARTITION`，不要只看 aiecc 临时目录。
- 同一个 xclbin 内的生产路径优先做成一个 public function、一个 `air.launch`、一个
  `air.segment`，stage 间用 tile/channel 交接，不用多个 public launch 表达 pipeline。
- 如果中间 BO 必须是 host-visible runtime boundary，就拆成多个 host launch。每个 stage
  单独编译成自己的 xclbin/insts，host 显式先跑 RMS，再跑 RoPE。
- 不把手工 JSON 合并或 `--xclbin-input` 当作当前 multi-launch pipeline 的修复方案。

## 诊断

RoPE 这类 head-major 输出可以用 per-head nonzero 分布判断是不是 segment/DMA 问题。临时
诊断脚本只需要打印每个 head 的非零数、最大误差和前几个值。

用 `timeout` 包住坏 xclbin。timeout 本身也是有效信号：runtime sequence 可能在等待未正确
加载的 segment 或未完成的 DMA。
