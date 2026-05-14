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
| [debug-runtime-dma-await.md](debug-runtime-dma-await.md) | score 对但输出缺贡献时怎么查输入 DMA。 | 先拆 standalone，再看 runtime sequence 是否 await 输入 task。 |
| [debug-l1-state-corruption.md](debug-l1-state-corruption.md) | token 后半段变 0 或重复旧状态时怎么定位。 | 查 AIE address map、stack、bank warning、channel 顺序和 herd 规模。 |
| [attention-head-loop.md](attention-head-loop.md) | 多 head attention 怎么避免 tile 程序膨胀。 | host 常量 offset，tile 内运行时 head loop。 |
| [attention-tile-limits.md](attention-tile-limits.md) | attention tile size 先选多大。 | `KEY_TILE_ROWS=8`，`QUERY_TILE_ROWS<=32`。 |
| [python-air-dsl-kernel.md](python-air-dsl-kernel.md) | Python kernel 应该怎么写。 | 用 MLIR-AIR Python DSL，不要用 Python 拼 MLIR 字符串。 |
| [herd-scalar-accumulator.md](herd-scalar-accumulator.md) | herd 内标量累加怎么写。 | 用 L1 scalar memref，不要依赖 `scf.for iter_args`。 |
| [external-kernel-tile-abi.md](external-kernel-tile-abi.md) | Python AIR 怎么调用 AIE external tile kernel。 | private `func.func` 带 `link_with`，调用 L1 memref，`air-to-std` 后跑 `symbol-dce`。 |
| [packed-weight-tile-layout.md](packed-weight-tile-layout.md) | 压缩权重和 side data 怎么组织成一个 L1 tile。 | host 先重排成 tile kernel 直接消费的连续记录，避免 tile body 解析复杂格式。 |
| [output-feature-chunking.md](output-feature-chunking.md) | output 维度太大时怎么先分段跑通。 | 固定 xclbin 计算一段 output features，host 多次 launch 拼完整输出。 |
| [output-parallel-herd.md](output-parallel-herd.md) | 怎么沿 output feature 维做多列 AIE 并行。 | 每列负责 disjoint output slice，复制共享 input，保持每 tile 的 L1 footprint 小。 |
| [static-token-buckets.md](static-token-buckets.md) | 动态 token 长度怎么落到固定 NPU graph。 | 用固定 token bucket，多余长度由 host 拆 launch，tail 用 pad/mask。 |
| [export-aten-linear-kernel.md](export-aten-linear-kernel.md) | `aten.linear.default` 怎么接入自定义 AIR kernel。 | generated export Python 直接映射到 kernel builder，不包第二层 graph。 |

维护规则：

- recipe 必须短小、自包含。
- 不引用实验脚本路径作为理解前提。
- 只记录稳定结论；长实验过程留在 spike 文档里。
