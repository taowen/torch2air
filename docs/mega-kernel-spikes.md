# Mega-kernel Spikes

本文规划一组递进实验，目标是学会用 MLIR-AIR 的正确方式把更大的计算写进一个
xclbin，而不是继续依赖很多小 xclbin 顺序调用。

当前 `torch2air` 已经能用多个 xclbin 和 shared `pyxrt.bo` 跑通
`quantized_qwen3` 的 16-token full `self_attn`。但这只是验证路径，不是最终性能路径。
当前最明显的问题是：

- `attention_core` 仍然是一个 q head 一个 xclbin。
- `self_attn` 前后 stage 仍然是多个 xclbin。
- runtime 需要按需加载 attention xclbin 来避开 `pyxrt.hw_context` 数量上限。

这个计划的核心不是一次写出完整 fused self-attn，而是用小实验把 AIR 的 launch、segment、
herd、channel、DMA、runtime sequence、placement 和 external kernel 关系逐层搞清楚。

## 原则

- 每个 spike 都必须生成 xclbin，并在真实 NPU 上跑。
- reference 只使用 PyTorch ROCm，不用 NumPy 作为正式对拍依据。
- 先做小的可解释实验，再迁移到 `quantized_qwen3`。
- 每个 spike 都要保存生成的 `.mlir`、`.npu.mlir`、`input_with_addresses.mlir` 和运行结果。
- 不引入 IRON DSL 作为依赖。IRON 只作为算法和 tiling 参考。
- 不增加 graph JSON、manifest 或自定义 runtime wrapper。入口仍然直接使用 `pyxrt` 概念。
- 失败时先定位 AIR/AIE 结构，不直接扩大 tile 或改数值代码。

## 参考材料

本地已经有 IRON：

```text
/var/home/taowen/projects/IRON
```

重点阅读：

- `/var/home/taowen/projects/IRON/aie_kernels/aie2p/mha.cc`
- `/var/home/taowen/projects/IRON/iron/operators/mha/design.py`

IRON 的 `mha.cc` 是 AIE tile 内 microkernel 集合；tiling 和数据流不在这里，而在
`design.py` 的 `ObjectFifo`、`Worker`、`Runtime.fill`、`Runtime.drain` 里。我们当前不直接
使用 IRON DSL，但要学习它的结构：

```text
QK matmul -> partial softmax -> PV matmul/rescale
```

MLIR-AIR 官方示例优先看：

- `mlir-air/programming_examples/multi_segment/multi_segment_channel`
- `mlir-air/programming_examples/matrix_scalar_add/multi_launch_channel`
- `mlir-air/programming_examples/channel_examples/worker_to_worker`
- `mlir-air/programming_examples/channel_examples/dual_herd_packet_switch`
- `mlir-air/programming_examples/flash_attention/dataflow_based`
- `mlir-air/programming_examples/flash_attention/kernel_fusion_based`

## 成功标准

最终成功不是“能生成一个巨大 MLIR 文件”，而是满足下面条件：

```text
一个 xclbin 内完成至少 16 个 q heads 的 attention_core
真实 NPU 运行通过
PyTorch ROCm 对拍通过
没有 host 中转
没有多个 attention hw_context
```

进一步目标是把 `q_norm_rope -> attention_core -> o_proj` 逐步合并进一个 xclbin。完整
`embed -> norm -> q/k/v -> rope -> attention -> o_proj` fusion 不是第一阶段目标。

## Spike 0: 官方机制复现

目标：确认本机工具链下官方 AIR 用法的真实 lowering 形态。

实验内容：

1. 跑官方 `multi_segment_channel`。
2. 跑官方 `multi_launch_channel`。
3. 跑官方 `worker_to_worker`。
4. 跑官方 `flash_attention` 的 dataflow 和 kernel-fusion 示例，至少生成 IR；能跑 NPU 则跑。

必须记录：

- AIR 源里有几个 `air.launch`、`air.segment`、`air.herd`。
- lowering 后 `aie.runtime_sequence` 如何组织 DMA task。
- 多 segment / 多 herd 的 channel 在 `input_with_addresses.mlir` 里怎么变成 locks、BD 和 buffers。
- 哪些 pass 会生成 `dma_free_task`，哪些位置需要 `dma_await_task`。

通过条件：

```text
每个官方例子至少 lower 到 AIE MLIR
至少一个多-herd/channel 例子生成 xclbin 并真实 NPU 运行
把关键 IR 片段记录到 docs/attention.md 或本文件
```

停止条件：

如果官方多-herd/channel 示例在当前工具链不能生成 xclbin，不继续做 Qwen3 mega-kernel，先修
工具链或找到官方推荐的 pass pipeline。

### Spike 0 记录：2026-05-14

本次实验环境：

```text
pyxrt: .venv/lib/python3.12/site-packages/pyxrt.cpython-312-x86_64-linux-gnu.so
device: RyzenAI-npu4
AIR device: npu2_4col
Python: uv 管理的 .venv Python 3.12
```

一个重要工具链细节：

```text
uv run 会把 .venv/bin 放在 PATH 最前面。
.venv/bin/aircc 是已经失效的 Python wrapper，会报：
ImportError: cannot import name 'main' from air.compiler.aircc.main
```

所以跑 MLIR-AIR 官方 `XRTRunner` 示例时，本次使用 `.venv/bin/python` 直接启动，
并显式把下面路径放在 PATH 前面：

```text
.venv/lib/python3.12/site-packages/mlir_air/bin
.venv/lib/python3.12/site-packages/mlir_aie/bin
.venv/lib/python3.12/site-packages/llvm-aie/bin
```

这仍然是本仓库 uv 创建的 Python 3.12 环境，不使用 `torch2vk` 的 `.venv`。

#### worker_to_worker

命令入口：

```text
mlir-air/programming_examples/channel_examples/worker_to_worker/worker_to_worker.py --output-format xclbin
```

产物：

```text
.cache/npu-spikes/mega-spike0-worker-to-worker/build_peano/air.xclbin
.cache/npu-spikes/mega-spike0-worker-to-worker/build_peano/air.insts.bin
.cache/npu-spikes/mega-spike0-worker-to-worker/build_peano/air_project/input_with_addresses.mlir
```

结果：

```text
真实 NPU xclbin 运行 PASS
source AIR: 1 launch, 1 segment, 1 herd
herd size: 2x3
public ABI: input/output 两个 BO
```

关键机制：

- 源 AIR 有 `ChanIn[2,3]`、`WorkerToWorker[2,3]`、`ChanOut[2,3]` 三组 channel。
- launch 侧把 6 个 L3 tile put 到 `ChanIn`，再从 `ChanOut` get 6 个 tile。
- herd 内每个 tile 从 `ChanIn[th,tw]` 读，算完后 put 到下一个 tile 的
  `WorkerToWorker[th_next,tw_next]`，再从自己的 `WorkerToWorker[th,tw]` 读回并输出。
- `input_with_addresses.mlir` 里生成了 tile-to-tile flow，例如：

```text
aie.flow(%tile_1_4, DMA : 1, %tile_0_2, DMA : 1)
aie.flow(%tile_1_2, DMA : 1, %tile_1_3, DMA : 1)
aie.flow(%tile_0_2, DMA : 1, %tile_0_3, DMA : 1)
```

- runtime sequence 对 host input DMA 使用 `dma_free_task`，对 output DMA 使用
  `dma_await_task`。这和我们在 attention 里修过的经验一致：host 只需要等待最终 output
  drain；input fill 不等价于结果完成。

结论：

```text
一个 xclbin 内用 AIR channel 做 worker-to-worker tile 传递是可行的。
这应该作为 Spike 2 的最小正确参考。
```

#### multi_launch_channel

命令入口：

```text
mlir-air/programming_examples/matrix_scalar_add/multi_launch_channel/multi_launch_channel.py
```

官方脚本当前缺少 `import numpy as np`，本次用临时 Python runner 直接调用
`build_module(...)`。

产物：

```text
.cache/npu-spikes/mega-spike0-multi-launch-channel/air.xclbin
.cache/npu-spikes/mega-spike0-multi-launch-channel/air.insts.bin
.cache/npu-spikes/mega-spike0-multi-launch-channel/air_project/input_with_addresses.mlir
```

结果：

```text
xclbin 生成成功
真实 NPU 执行完成
数值失败：511 / 512 elements mismatch，输出基本为 0
source AIR: 4 launch, 4 segment, 4 herd
```

关键机制：

- lowering 后每个 segment 都有自己的 `aie.runtime_sequence @segmentXXXX_sequence`。
- 顶层 `aie.runtime_sequence @copy` 只是顺序执行：

```text
aiex.configure @segment0000 { aiex.run @segment0000_sequence(...) }
aiex.configure @segment0001 { aiex.run @segment0001_sequence(...) }
aiex.configure @segment0100 { aiex.run @segment0100_sequence(...) }
aiex.configure @segment0101 { aiex.run @segment0101_sequence(...) }
```

- 每个 segment sequence 内只有一个 input DMA 和一个 output DMA；input 是
  `dma_free_task`，output 是 `dma_await_task`。

结论：

```text
当前工具链/pyxrt runner 下，多个 air.launch 顺序拼进一个 public function 不是可直接采用的稳定路径。
后续不要先走 multi-launch；优先走 single launch + one segment + 多 herd/channel。
```

这也说明 stage stitching 不能只看 AIR 源里 launch token 串起来，还要看最终
`aie.runtime_sequence @copy` 是否真正表达了期望的执行与等待语义。

#### multi_segment_channel

命令入口：

```text
mlir-air/programming_examples/multi_segment/multi_segment_channel/multi_segment.py --output-format xclbin
```

结果：

```text
未生成 xclbin
air-to-aie 失败：
'air.channel.put' op failed to get S2MM tile for L3 allocation
```

即使显式使用 `target_device=npu2_4col` 和 `use_lock_race_condition_fix=true`，仍在
第二个 L3 input channel 上失败。

结论：

```text
旧的 explicit multi-segment + 多个 L3 channel 写法在当前工具链下不稳。
后续不要把 Qwen3 mega-kernel 设计成多个 sibling segment 直接抢同一组 shim DMA。
```

#### dual_herd_packet_switch

命令入口：

```text
mlir-air/programming_examples/channel_examples/dual_herd_packet_switch/dual_herd_packet_switch.py
```

这个例子更接近当前官方推荐风格：

```text
计算用多个 herd
数据移动先写 air.dma_memcpy_nd
由 air-dma-to-channel 自动转 channel
超过每列 shim DMA 限制时自动升级 packet switching
```

结果：

```text
xclbin 生成成功
官方 xclbin runner 运行失败：ValueError: Too many arguments
原因：XRTBackend xclbin invoker 限制最多 5 个 BO 参数；该例子有 4 input + 2 output
ELF 路径也无法作为替代：当前 pyxrt 没有 xrt.elf API
```

结论：

```text
这个例子的 AIR 写法值得借鉴，但当前 Python runner 不能直接验证 6-BO xclbin。
正式实验要么保持 public ABI <= 5 个 BO，要么使用我们自己的 pyxrt 调用约定明确 group_id。
不能靠猜 group_id；错误组合会让 pyxrt 直接崩溃。
```

#### flash_attention IR

命令入口：

```text
mlir-air/programming_examples/flash_attention/dataflow_based/attn.py -p ...
mlir-air/programming_examples/flash_attention/kernel_fusion_based/attn_npu2.py -p ...
```

产物：

```text
.cache/npu-spikes/mega-spike0-flash-attention-ir/dataflow/source.air.mlir
.cache/npu-spikes/mega-spike0-flash-attention-ir/kernel_fusion/source.air.mlir
```

结构观察：

```text
dataflow_based:
  1 launch, 1 segment, 6 herds, 71 channel ops
  有 L3ToL2、L2ToL1、L1ToL1、cascade channel

kernel_fusion_based:
  1 launch, 1 segment, 1 herd, 110 channel ops
  QKIn/QK2L1/VIn/V2L1 使用 3D channel 和 broadcast_shape
  cascade_gp/cascade_up/cascade_sp 使用 channel_type = "npu_cascade"
```

对我们最重要的启发：

- 官方 flash attention 不是多个 host-visible xclbin stage，而是在一个 launch/segment 内显式组织
  Q/K/V relay、cascade 和 output drain。
- 多 head 维度体现在 channel index 和 segment/herd 维度里，不应该拆成很多 public xclbin。
- `kernel_fusion_based` 的方向更接近我们要的 attention mega-kernel：一个 public ABI，
  内部用 channel 和 external vectorized kernel 组织 QK、softmax state、PV。

#### Spike 0 后的设计收敛

下一步不直接做 full attention。先做两个小实验：

```text
Spike 1:
  single launch + one segment + 两个 independent herd
  public ABI 控制在 <= 5 个 BO
  用 xclbin 真实 NPU 跑通

Spike 2:
  single launch + one segment + producer/consumer channel
  不用 host 中间 BO
  以 worker_to_worker 的 input_with_addresses.mlir 为参考
```

暂时避免：

- 多个 sibling `air.segment` 直接共享 L3 channel。
- 多个 `air.launch` 拼接成一个 public function。
- public ABI 超过当前 `XRTBackend` xclbin invoker 可处理的 5 个 BO。

## Spike 1: 单 xclbin 多 herd，无依赖

目标：掌握一个 public function 里放多个 independent herd 的最小写法。

实验设计：

```text
input_a -> herd A -> output_a
input_b -> herd B -> output_b
```

两个 herd 不互相通信，只证明一个 xclbin 可以包含多个 compute island。

实现要求：

- 用两个很小的 external kernel，例如 `add_one_tile` 和 `mul_two_tile`。
- 一个 `func.func @run_two_independent_herds`。
- 一个或两个 `air.segment` 都可以，但要明确比较两种写法的 lowering 差异。
- Python runner 只加载一个 xclbin。

通过条件：

```text
真实 NPU 输出和 PyTorch ROCm reference 对拍通过
input_with_addresses.mlir 里能看到两个 core/herd 的独立 buffer/lock
```

需要回答的问题：

- 多 herd 放同一个 segment 和多个 segment 的 placement 差异是什么？
- `aiecc --xclbin-instance-name` 对 public function 名字有什么要求？
- 一个 xclbin 里多个 private external function 的 `link_with` 是否稳定？

### Spike 1 记录：2026-05-14

实验脚本：

```text
examples/amd_aie_experiments/mega_spike1_two_herds.py
```

命令：

```text
source scripts/npu-common.sh
uv run --no-sync python examples/amd_aie_experiments/mega_spike1_two_herds.py \
  --tile-size 64 \
  --work-dir .cache/npu-spikes/mega-spike1-two-herds
```

结果：

```text
真实 NPU xclbin 运行 PASS
PyTorch ROCm reference device: AMD Radeon 890M
xclbin: .cache/npu-spikes/mega-spike1-two-herds/air.xclbin
insts: .cache/npu-spikes/mega-spike1-two-herds/air.insts.bin
source AIR: 1 launch, 1 segment, 2 herds, 4 dma_memcpy_nd
public ABI: add_input, mul_input, add_output, mul_output
```

计算：

```text
add_herd: output0 = input0 + 1
mul_herd: output1 = input1 * 2
```

关键 AIR 结构：

```text
air.launch
  air.segment @seg
    air.herd @add_herd
      air.dma_memcpy_nd L3 -> L1
      scalar loop
      air.dma_memcpy_nd L1 -> L3
    air.herd @mul_herd
      air.dma_memcpy_nd L3 -> L1
      scalar loop
      air.dma_memcpy_nd L1 -> L3
```

关键 lowering 结果：

```text
placed.air.mlir:
  add_herd x_loc = 0, y_loc = 2
  mul_herd x_loc = 1, y_loc = 2
  air.wait_all [%add_herd_token, %mul_herd_token] {air.segment_end}

input_with_addresses.mlir:
  aie.flow shim_noc_tile_0_0 -> tile_0_2 -> shim_noc_tile_0_0
  aie.flow shim_noc_tile_1_0 -> tile_1_2 -> shim_noc_tile_1_0
  input DMA: dma_free_task
  output DMA: dma_await_task
```

本次回答的问题：

- 多个 independent herd 放在同一个 segment 内可以稳定 lower 到同一个 xclbin。
- `air-place-herds` 会给两个 1x1 herd 分配不同 tile，而不是重叠到同一个 tile：
  `add_herd` 在 `(0,2)`，`mul_herd` 在 `(1,2)`。
- 用 `air.dma_memcpy_nd` 表达 L3/L1 数据移动更接近官方当前风格；aircc 自动把它转换成
  channel，不需要手写 L3 shim channel。
- public ABI 保持 4 个 BO 后，当前 `XRTBackend` xclbin runner 可以直接跑，不触发
  5 参数上限。

还没回答的问题：

- 多个 private external function 的 `link_with` 是否稳定。Spike 1 为了先验证 AIR/AIE
  结构，故意用了内联 scalar loop，没有 external kernel。

结论：

```text
Spike 1 通过。
后续正式 mega-kernel 应优先采用 single launch + one segment + 多 herd 的结构。
不要优先采用多个 sibling segment 或多个 launch。
```

## Spike 2: 单 xclbin producer -> consumer channel

目标：掌握一个 xclbin 内 stage 之间用 AIR channel 传递 tile。

实验设计：

```text
input -> herd A -> channel -> herd B -> output
```

建议计算：

```text
output = input * 2 + 1
```

实现要求：

- herd A 只写 channel，不写 L3 output。
- herd B 只从 channel 读，再写 L3 output。
- channel depth 先用默认，必要时再显式设置。
- 必须检查 `aie.runtime_sequence` 里 host 侧只负责 input fill 和 output drain。

通过条件：

```text
真实 NPU 对拍通过
没有 host 侧中间 BO
runtime sequence 与 producer/consumer 顺序一致
```

需要回答的问题：

- channel put/get 的 async token 链是否足够表达顺序？
- 哪些 MM2S input DMA task 被 free，哪些需要 await？
- producer/consumer 速度不一致时，channel depth 是否影响死锁或结果？

### Spike 2 记录：2026-05-14

实验脚本：

```text
examples/amd_aie_experiments/mega_spike2_producer_consumer.py
```

命令：

```text
source scripts/npu-common.sh
uv run --no-sync python examples/amd_aie_experiments/mega_spike2_producer_consumer.py \
  --tile-size 64 \
  --work-dir .cache/npu-spikes/mega-spike2-producer-consumer
```

结果：

```text
真实 NPU xclbin 运行 PASS
PyTorch ROCm reference device: AMD Radeon 890M
xclbin: .cache/npu-spikes/mega-spike2-producer-consumer/air.xclbin
insts: .cache/npu-spikes/mega-spike2-producer-consumer/air.insts.bin
source AIR: 1 launch, 1 segment, 2 herds, 2 explicit channel ops, 2 dma_memcpy_nd
public ABI: input, output
```

计算：

```text
producer: mid = input * 2
consumer: output = mid + 1
```

关键 AIR 结构：

```text
air.channel @ProducerToConsumer []

air.launch
  air.segment @seg
    air.herd @producer
      air.dma_memcpy_nd input L3 -> producer L1
      ChannelPut @ProducerToConsumer
    air.herd @consumer
      ChannelGet @ProducerToConsumer
      air.dma_memcpy_nd consumer L1 -> output L3
```

关键 lowering 结果：

```text
placed.air.mlir:
  producer x_loc = 0, y_loc = 2
  consumer x_loc = 1, y_loc = 2
  consumer ChannelGet @ProducerToConsumer 由 channel token 阻塞等待

input_with_addresses.mlir:
  aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
  aie.flow(%tile_0_2, DMA : 0, %tile_1_2, DMA : 0)
  aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
  input DMA: dma_free_task
  output DMA: dma_await_task
```

本次回答的问题：

- channel put/get 能表达 producer/consumer 顺序；consumer herd 可以先启动，但会在
  `ChannelGet @ProducerToConsumer` 上等待 producer 的 put。
- host runtime sequence 只看到 public input/output DMA；producer 到 consumer 的中间 tile 没有
  host BO，也没有 host 读写。
- 对一个 tile、默认 channel depth 的场景，producer/consumer 速度不同不会导致错误或 deadlock。

还没回答的问题：

- 更大 tile、多 token、多 producer 或多 consumer 时 channel depth 是否需要显式调。
- 真实 attention 里 QK/softmax/PV 三阶段是否应该用一条 channel 串行，还是多个 indexed channel。

结论：

```text
Spike 2 通过。
stage stitching 的第一原则应该是：同一个 launch/segment 内用 AIR channel 传 tile，
不要把中间结果写回 host-visible BO。
```

## Spike 3: 单 xclbin 两个现有 Qwen3 stage

目标：把 toy channel 迁移到真实 stage，但先选低风险 stage。

首选组合：

```text
rope_table -> q_norm_rope
```

原因：

- `rope_table` 已经用 channel 修过 cos/sin 输出顺序。
- `q_norm_rope` 是 head-local 的 f32 stage，权重小。
- 不涉及 GGUF 量化权重 tile 的 L1 bank 压力。

实验路径：

1. 先保留 public ABI：`q_proj, q_norm_weight, start_position -> q_norm_rope_output`。
2. 在同一个 xclbin 内生成 cos/sin，并通过 channel 或 L3 scratch 交给 q_norm_rope。
3. 先用 4 tokens，再到 16 tokens。

通过条件：

```text
一个 xclbin 完成 rope_table + q_norm_rope
真实 NPU 对拍 q_norm_rope
不读回 cos/sin 给 host 再写入
```

需要回答的问题：

- 这个组合用 channel 更稳，还是用 xclbin 内 L3 scratch 更稳？
- 多输出 cos/sin 的顺序问题在 fused 版本是否自然消失？
- `q_norm_rope` 的 dma_packet auto-upgrade 在 fused 版本里是否还出现？

### Spike 3 记录：2026-05-14

实验脚本：

```text
examples/amd_aie_experiments/mega_spike3_rope_norm.py
```

最终通过的 AIR 结构：

```text
air.launch
  launch: Start/Weight/Input/Output host channels
  air.segment @seg
    air.herd @rope
      ChannelGet Start
      rope_table_tile(position) -> cos_l1, sin_l1
      pack cos/sin into trig_l1: memref<2x128xf32, 2>
      ChannelPut Trig
    air.herd @norm_rope
      ChannelGet Weight
      ChannelGet Trig
      unpack trig_l1 -> cos_l1, sin_l1
      ChannelGet Input
      rms_norm_rope_tile(input, weight, cos, sin) -> output
      ChannelPut Output
```

命令：

```text
source scripts/npu-common.sh
uv run --no-sync python examples/amd_aie_experiments/mega_spike3_rope_norm.py \
  --sequence-length 4 \
  --head-dim 128 \
  --start-position 0 \
  --work-dir .cache/npu-spikes/mega-spike3-rope-norm

uv run --no-sync python examples/amd_aie_experiments/mega_spike3_rope_norm.py \
  --sequence-length 16 \
  --head-dim 128 \
  --start-position 0 \
  --work-dir .cache/npu-spikes/mega-spike3-rope-norm-16

uv run --no-sync python examples/amd_aie_experiments/mega_spike3_rope_norm.py \
  --sequence-length 16 \
  --head-dim 128 \
  --start-position 7 \
  --work-dir .cache/npu-spikes/mega-spike3-rope-norm-16-pos7
```

结果：

```text
4 tokens, start_position=0: 真实 NPU PASS
16 tokens, start_position=0: 真实 NPU PASS
16 tokens, start_position=7: 真实 NPU PASS
PyTorch ROCm reference device: AMD Radeon 890M
```

关键 lowering 结果：

```text
input_with_addresses.mlir:
  aie.flow(%tile_1_2, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
  aie.flow(%tile_0_2, DMA : 0, %tile_1_2, DMA : 1)
  aie.shim_dma_allocation @air_Output(%shim_noc_tile_1_0, S2MM, 0)
  aie.shim_dma_allocation @air_Start(%shim_noc_tile_0_0, MM2S, 0)
  aie.shim_dma_allocation @air_Weight(%shim_noc_tile_1_0, MM2S, 0)
  aie.shim_dma_allocation @air_Input(%shim_noc_tile_1_0, MM2S, 0)

  rope core link_files = ["rope_table.o"]
  norm_rope core link_files = ["rms_norm_rope.o"]
```

中间失败路径也有价值：

- 直接用两个内部 channel `Cos` / `Sin` 会在 routing 阶段失败：
  `aie.masterset op targets same destination DMA`。原因是 consumer tile 同时接收两个
  tile-to-tile stream，再叠加 host packet input/weight，超过了当前可路由的目的 DMA 形态。
- 改成 L3 scratch 后可以生成 xclbin 并真实 NPU 运行，但数值不对。检查回读 scratch 后看到
  lowering 把动态写入变成了 `cos0, sin0, cos1, sin1, ...` 的实际布局，而不是源 AIR 里写的
  `cos[0..N), sin[0..N)`；`runtime_sequence` 也只给 output 配了第一行任务。这个说明当前写法下
  不应该把 herd 内动态 `dma_memcpy_nd` 到 shared L3 scratch 当作 stage stitching 的默认方案。

本次回答的问题：

- 对 `rope_table -> q_norm_rope` 这种小 tile handoff，单条内部 channel 比 L3 scratch 稳。
- 多输出 cos/sin 的顺序问题不能靠两个独立 channel 赌 lowering 顺序；把它们 pack 到同一个
  L1 tile 后，channel 只传一个对象，顺序自然绑定。
- `q_norm_rope` 的 host 输入仍会被 packet 化，但通过路径里只有一条 tile-to-tile `Trig` flow，
  没有再触发 destination DMA 冲突。

结论：

```text
Spike 3 通过。
正式实现多个 stage 串起来时，优先把同一个 logical handoff 的多个小 buffer 合成一个
tile-local payload，再用一条 AIR channel 传递。
```

## Spike 4: 一个 xclbin 内多个 attention head，串行 head loop

目标：先消灭“一个 head 一个 xclbin”，不追求并行。

当前 attention 模板静态绑定一个 `attention_head_index`。这个 spike 要改成一个 xclbin 内处理
多个 head：

```text
for q_head in 0..q_heads:
  q_col = q_head * 128
  kv_col = (q_head / q_heads_per_kv_head) * 128
  run current q-block / kv-block loop
```

实现要求：

- public ABI 保持 `q, k, v, output` 四个 memref。
- 先做 `q_heads=2, kv_heads=1`。
- 再做 `q_heads=4, kv_heads=2`。
- 最后做 `q_heads=16, kv_heads=8`。
- Python runner 只加载一个 attention xclbin。

已知风险：

- 动态 `q_col` / `kv_col` offset 可能触发 AIR lowering 问题。仓库里已有
  `examples/amd_aie_experiments/air_dynamic_head_offset_repro.mlir`，先用它缩小问题。
- 当前 `_await_attention_input_dma` 是针对 `run_attention_core` 的后处理。合并 head 后仍要
  检查 Q/KV input DMA 是否全部 await。

通过条件：

```text
TOKEN_COUNT=16 Q_HEADS=16 KV_HEADS=8
一个 attention xclbin
真实 NPU attention_core_max_abs <= 当前多 xclbin 量级
没有 CREATE_HWCTX 问题
```

需要回答的问题：

- AIR 是否能稳定 lower 动态 head offset 的 channel put/get？
- head loop 放 launch 侧和 herd 侧是否必须保持完全同构？
- 一个 herd 串行跑 16 heads 的 runtime sequence 会不会过大？

### Spike 4 记录：2026-05-14

实验脚本：

```text
examples/amd_aie_experiments/mega_spike4_attention_multi_head.py
```

关键修正：

- 显式 async AIR 不能直接交给 `compile_runtime`。正确路径是先跑
  `air-place-herds`，再跑 `air-to-aie`，然后把 AIE IR 交给 runtime 编译。
- host 侧保留静态 head 展开来生成常量 DMA offset，避免动态 offset 在 runtime
  lowering 里变成 i64。
- herd 侧不能静态展开 16 个 head。静态展开到 q12/kv6 时 `aiecc` 报 tile
  program memory overflow；改成 `scf.for %head` 运行时循环后，q16/kv8 通过。

真实 NPU 结果：

```text
q_heads=2,  kv_heads=1, sequence_length=16: PASS, mean_ms 18.444
q_heads=4,  kv_heads=2, sequence_length=16: PASS, mean_ms 36.453
q_heads=8,  kv_heads=4, sequence_length=16: PASS, mean_ms 71.726
q_heads=16, kv_heads=8, sequence_length=16: PASS, mean_ms 143.140
q_heads=16, kv_heads=8, sequence_length=32: PASS, mean_ms 552.523
q_heads=16, kv_heads=8, sequence_length=64: PASS, mean_ms 2174.660
```

tile size 结果：

```text
QUERY_TILE_ROWS=8,  sequence_length=16: PASS, mean_ms 142.325
QUERY_TILE_ROWS=16, sequence_length=16: PASS, mean_ms 141.514
QUERY_TILE_ROWS=32, sequence_length=64: PASS, mean_ms 2166.039
QUERY_TILE_ROWS=64, sequence_length=64: FAIL, tile L1 buffer allocation exceeded 64 KiB

KEY_TILE_ROWS=4, sequence_length=16: PASS, mean_ms 145.070
KEY_TILE_ROWS=8, sequence_length=16: PASS, mean_ms 134.269
KEY_TILE_ROWS=16: FAIL, 编译可过但真实 NPU 输出部分保持 0
```

结论：

```text
Spike 4 通过。
正式 attention core 应该用一个 xclbin 处理多个 head；head 数放在 herd 运行时循环里，
不要复制 tile 程序。当前已验证 key tile 上限是 8，query tile 上限受 64 KiB L1
约束，32 行还能通过，64 行会超过本地内存。
```

## Spike 5: 一个 xclbin 内多个 attention head，并行 head groups

目标：在 Spike 4 正确之后，再恢复并行度。

实验设计：

```text
head group 0 -> herd/segment 0
head group 1 -> herd/segment 1
...
```

先从 2 个 head groups 开始，不直接做 16。

实现要求：

- 每个 head group 写 output 的不同列范围。
- K/V 可以广播或重复 fill，先选择更容易正确的方案。
- 每个 group 的 channel 名必须明确隔离，避免 accidental sharing。

通过条件：

```text
q_heads=4 或 q_heads=8 单 xclbin真实 NPU通过
输入 DMA await 不再依赖 fragile 文本匹配，或文本匹配范围明确可检查
```

需要回答的问题：

- 多个 group 共享 K/V channel 可行，还是应该每组独立 channel？
- K/V 重复 DMA 的成本和 channel/routing 稳定性怎么权衡？
- npu2_4col 上可稳定放多少个 attention pipelines？

## Spike 6: IRON-style block attention，单 herd 串行

目标：从 row-wise `4x4` attention 过渡到 block attention，但先不做多 worker pipeline。

IRON 的核心结构是：

```text
QK block matmul
partial_softmax 更新 scale_buffer
PV block matmul
最后 rescale O
```

本 spike 在一个 herd 内串行执行这些步骤：

```text
Q tile -> K tile -> qk_l1
qk_l1 -> p_l1 + scale_l1
p_l1 + V tile -> out_l1
```

建议形状：

```text
B_q=8
B_kv=8
HEAD_DIM=128
dtype=f32 first, bf16 later
```

实现要求：

- 先用 f32 scalar/block kernel 证明 AIR 数据流。
- 再评估是否引入 AIE `aie::vector` / bf16 microkernel。
- scale buffer 显式采用 IRON 的布局：

```text
scale[0:B_q]       = m_old
scale[B_q:2B_q]    = m_new
scale[2B_q:3B_q]   = l
scale[3B_q:4B_q]   = exp(m_old - m_new)
```

通过条件：

```text
TOKEN_COUNT=16, one head, block attention xclbin真实 NPU通过
结果不劣于当前 row-wise attention 阈值
```

需要回答的问题：

- f32 block kernel 的 L1 buffer 是否已经超过 bank 压力？
- bf16 是否能接受来自 q_norm_rope 的 f32 输入，还是需要额外 cast stage？
- `aie::exp2` 路径是否比当前 `fast_exp` 更稳定？

## Spike 7: IRON-style QK / softmax / PV 三 worker pipeline

目标：学习真正的 fused attention pipeline，而不是一个 herd 内串行做完。

实验设计：

```text
worker 0: QK matmul
worker 1: partial softmax
worker 2: PV matmul + rescale
```

stage 之间用 channel/ObjectFIFO 等价结构表达。我们用 AIR channel 写，不引入 IRON DSL。

实现要求：

- 先 one head、`B_q=8`、`B_kv=8`。
- 三个 worker/herd 的 external kernel 分开，便于定位。
- 保留 `scale_buffer` 在 softmax/PV 之间的传递路径。
- 每个 worker 的 L1 buffer 地址和 stack 必须检查。

通过条件：

```text
one head block attention真实 NPU通过
生成的 AIE IR 能清楚看到 QK -> softmax -> PV 的 channel/lock
```

需要回答的问题：

- scale buffer 是 channel 传递，还是每个 q block 固定 L1 buffer 更稳？
- 三 worker pipeline 是否比单 herd 串行更容易触发 channel/deadlock 问题？
- 对 causal mask 和 tail padding，放 softmax worker 是否最合适？

## Spike 8: q_norm_rope -> attention_core 单 xclbin

目标：把 attention 的前置 lightweight stage 融进去，减少 host launch 和 BO 交接。

实验设计：

```text
q_proj output + k_proj output + v_proj output
  -> q_norm_rope / k_norm_rope
  -> attention
  -> output
```

先做 one head 或 two heads，不直接做 full 16 heads。

通过条件：

```text
一个 xclbin 完成 q/k norm+RoPE 和 attention
真实 NPU 对拍 attention_core
host 不读回 q_norm_rope/k_norm_rope
```

需要回答的问题：

- q/k norm+RoPE 输出适合用 channel 直接进 attention，还是写 L3 scratch 再读？
- RoPE cos/sin table 在 xclbin 内生成还是作为 public input？
- 这个融合是否减少 DMA，还是只是把问题搬进更复杂的 runtime sequence？

## Spike 9: attention_core -> o_proj 单 xclbin

目标：验证 attention 输出能否直接接量化 `o_proj`，为 self_attn mega-kernel 做准备。

实验设计：

```text
q/k/v -> attention -> o_proj
```

风险：

- `o_proj` Q4_K weight tile 当前是 `32x304xi32`，已有 L1 bank allocation warning。
- attention output 是 f32，o_proj 读取整行 input；如果直接 channel 化，tile shape 要重新设计。

建议先用 L3 scratch：

```text
attention writes scratch output
o_proj reads scratch output
final output writes public output
```

通过后再考虑 channel 化。

通过条件：

```text
一个 xclbin 完成 attention + o_proj
真实 NPU 对拍 o_proj
没有 host 中间 BO
```

需要回答的问题：

- L3 scratch 在单 xclbin 内是否比 channel 更容易稳定？
- `o_proj` slicing 能否和 attention head layout 对齐？
- 是否需要先降低 Q4_K weight tile 的 L1 压力？

## Spike 10: self_attn mega-kernel

目标：把前面已经通过的组件组合成第一个正式 mega-kernel。

范围：

```text
q_proj output
k_proj output
v_proj output
q_norm_weight
k_norm_weight
o_proj_weight
start_position
  -> q/k norm+RoPE
  -> attention
  -> o_proj
```

暂时不包含：

- embed_tokens
- input_layernorm
- q/k/v projection

原因是 q/k/v projection 的量化权重 tile 和 dispatch slicing 仍有较大 L1/bank 压力，把它们
过早放进 mega-kernel 会让问题难以定位。

通过条件：

```text
TOKEN_COUNT=16
Q_HEADS=16
KV_HEADS=8
一个 self_attn xclbin
真实 NPU 对拍 o_proj
```

## 每个 spike 的记录模板

每完成一个 spike，就在本文件追加结果：

```text
日期:
commit:
命令:
输入形状:
xclbin:
insts:
关键 AIR 结构:
关键 AIE 结构:
PyTorch ROCm 设备:
max_abs:
allclose:
mean_ms:
结论:
下一步:
```

失败也要记录，尤其是：

- AIR verifier error
- `air-place-herds` placement 诊断
- `aiecc` L1/bank warning
- `DRM_IOCTL_AMDXDNA_CREATE_HWCTX`
- NPU hang
- 数值 mismatch 的最大误差和 index

## 当前优先级

立即执行顺序：

1. Spike 0，确认官方多 segment / 多 channel 示例在当前工具链的生成形态。
2. Spike 1，写最小多 herd xclbin。
3. Spike 2，写最小 producer-consumer channel xclbin。
4. Spike 4，先把 2 个 attention heads 合成一个 xclbin。
5. Spike 4 扩到 16 heads。

只有 Spike 4 成功后，才开始做 IRON-style block attention。否则我们会同时面对
“不会写大 xclbin”和“不会写高性能 attention”两个问题，定位成本太高。
