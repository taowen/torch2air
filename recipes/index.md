# MLIR-AIR Recipes

这里放短小、自包含的 MLIR-AIR 速查笔记。每篇 recipe 只回答一个可复用问题，不把读者导向
实验脚本才能理解结论。

| Recipe | 解决的问题 | 核心结论 |
| --- | --- | --- |
| [uv-air-toolchain.md](uv-air-toolchain.md) | `uv run` 下怎么稳定调用 AIR/AIE/Peano/pyxrt。 | 把 wheel 内真实 binary 放到 `PATH` 前面。 |
| [single-segment-two-herds.md](single-segment-two-herds.md) | 一个 xclbin 能否包含多个独立 compute herd。 | 用 single launch + one segment + multiple herds。 |
| [producer-consumer-channel.md](producer-consumer-channel.md) | 同一 xclbin 内两个 stage 怎么交接 tile。 | 中间 tile 用 AIR channel，不写 host-visible BO。 |
| [pack-multi-output-channel.md](pack-multi-output-channel.md) | 一个 stage 有多个强配对输出怎么传给下游。 | 先 pack 成一个 L1 payload，再用一条 channel。 |
| [debug-air-lowering.md](debug-air-lowering.md) | routing 失败或 NPU 数值错时怎么定位。 | 按 AIR graph、AIE lowering、runtime sequence、L1 地址分层看。 |
| [attention-head-loop.md](attention-head-loop.md) | 多 head attention 怎么避免 tile 程序膨胀。 | host 常量 offset，tile 内运行时 head loop。 |
| [attention-tile-limits.md](attention-tile-limits.md) | attention tile size 先选多大。 | `KEY_TILE_ROWS=8`，`QUERY_TILE_ROWS<=32`。 |
| [python-air-dsl-kernel.md](python-air-dsl-kernel.md) | Python kernel 应该怎么写。 | 用 MLIR-AIR Python DSL，不要用 Python 拼 MLIR 字符串。 |
| [herd-scalar-accumulator.md](herd-scalar-accumulator.md) | herd 内标量累加怎么写。 | 用 L1 scalar memref，不要依赖 `scf.for iter_args`。 |

维护规则：

- recipe 必须短小、自包含。
- 不引用实验脚本路径作为理解前提。
- 只记录稳定结论；长实验过程留在 spike 文档里。
