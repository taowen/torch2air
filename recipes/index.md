# MLIR-AIR Recipes

这里放短小、自包含的 MLIR-AIR 速查笔记。每篇 recipe 只回答一个可复用问题，不依赖历史
调研目录、脚本或过程记录。

| Recipe | 解决的问题 | 核心结论 |
| --- | --- | --- |
| [uv-air-toolchain.md](uv-air-toolchain.md) | `uv run` 下怎么稳定调用 AIR/AIE/Peano/pyxrt。 | 把 wheel 内真实 binary 放到 `PATH` 前面。 |
| [python-air-dsl-kernel.md](python-air-dsl-kernel.md) | Python kernel 应该怎么写。 | 用 MLIR-AIR Python DSL，不用 Python 拼 MLIR 字符串。 |
| [external-kernel-tile-abi.md](external-kernel-tile-abi.md) | Python AIR 怎么调用 AIE external tile kernel。 | private `func.func` 带 `link_with`，调用 L1 memref。 |
| [packed-weight-tile-layout.md](packed-weight-tile-layout.md) | 压缩权重和 side data 怎么组织成一个 L1 tile。 | 每个 output row 使用连续 packed record。 |
| [vectorized-quantized-dot.md](vectorized-quantized-dot.md) | packed quantized dot body 怎么避免纯标量乘加。 | bit unpack 进 16-lane scratch，dot 用 `aie::vector` 和 `aie::accum`。 |
| [export-aten-linear-kernel.md](export-aten-linear-kernel.md) | `aten.linear.default` 怎么接入自定义 AIR kernel。 | generated export Python 直接映射到 kernel builder。 |
| [output-feature-chunking.md](output-feature-chunking.md) | output 维度太大时怎么固定形状拼完整 tensor。 | 一个 xclbin 计算一段 output features，host 多次 launch。 |
| [output-parallel-herd.md](output-parallel-herd.md) | 怎么沿 output feature 维做多列 AIE 并行。 | 每列负责 disjoint output slice，复制共享 input。 |
| [static-token-buckets.md](static-token-buckets.md) | 动态 token 长度怎么落到固定 NPU graph。 | 用固定 token bucket，长序列由 host 拆 launch。 |
| [single-segment-two-herds.md](single-segment-two-herds.md) | 一个 xclbin 能否包含多个独立 compute herd。 | 用 single launch + one segment + multiple herds。 |
| [producer-consumer-channel.md](producer-consumer-channel.md) | 同一 xclbin 内两个 stage 怎么交接 tile。 | 中间 tile 用 AIR channel，不写 host-visible BO。 |
| [pack-multi-output-channel.md](pack-multi-output-channel.md) | 一个 stage 有多个强配对输出怎么传给下游。 | 先 pack 成一个 L1 payload，再用一条 channel。 |
| [herd-scalar-accumulator.md](herd-scalar-accumulator.md) | herd 内标量累加怎么写。 | 用 L1 scalar memref，不依赖 `scf.for iter_args`。 |
| [debug-air-lowering.md](debug-air-lowering.md) | routing 失败或 NPU 数值错时怎么定位。 | 按 AIR graph、AIE lowering、runtime sequence、L1 地址分层看。 |

维护规则：

- recipe 必须短小、自包含。
- 不引用历史调研脚本、旧 docs 或 `.cache` 路径作为理解前提。
- 只记录稳定结论和可复用边界；历史过程不要写进 recipe。
