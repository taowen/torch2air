# MLIR-AIR Attention 实现说明

本文记录 MLIR-AIR 的关键机制，以及 `torch2air` 当前如何用这些机制实现
`quantized_qwen3` 的 `attention_core`。这里描述的是正式路径，不包含已经删除的
attention spike。

## MLIR-AIR 的执行模型

MLIR-AIR 把一个算子的设备执行拆成几个层次：

- `memref` 参数表示运行时传入的全局 L3 buffer。这里的 L3 是 stage 的公开 ABI，
  也就是 Python/XRT 侧传给 kernel 的 `pyxrt.bo`。
- `memref<..., 2 : i32>` 表示 AIE tile-local L1 buffer。外部 kernel 只应该操作
  这些 L1 buffer，不直接决定上游大 tensor 如何切块。
- `air.launch` 表示一次设备 launch。它拿到 L3 参数，并安排 launch 级别的数据流。
- `air.segment` 表示 launch 内部的设备区域。
- `air.herd` 表示一组 AIE compute tiles。herd body 里分配 L1 buffer、执行 tile
  计算、通过 channel 和 L3/L2/L1 交换数据。
- `air.channel.put` / `air.channel.get` 是 FIFO 数据流边。AIR lowering 会把它们变成
  具体的 shim、memtile、core DMA 和同步。
- `air.dma_memcpy_nd` 是显式多维 DMA。它适合简单的单输入/单输出 tile 搬运；如果多路
  输入之间需要严格顺序，直接使用 channel 更容易表达。
- `air.launch` 和 `air.herd` 的 region 是隔离的。region 内要用的 index 常量需要在
  region 内重新定义，不能依赖外层 SSA 值自然可见。
- `--air-to-aie stack-size` 会占用 AIE tile L1。stack 太小时，external kernel 的局部
  变量和调用帧可能覆盖相邻 L1 buffer，表现为运行结果被静默污染。

所以 AIR 里真正重要的不是“写一个 C++ kernel”，而是让 MLIR 文件明确表达：

```text
stage ABI -> launch -> tile/dataflow schedule -> L1 buffers -> external tile body
```

当前仓库遵循 MLIR-AIR 官方示例的 external-kernel 风格。官方 `softmax`、
`flash_attention` 和 channel examples 都是这个思路：MLIR/AIR 负责 launch、herd、
channel、L1 buffer 生命周期和搬运；native object 只实现 L1 上的一小段计算。

## torch2air 的使用边界

`torch2air` 不重新包装 `pyxrt` 和 AIR runtime 概念。运行时仍然直接使用：

- `pyxrt.device`
- `pyxrt.xclbin`
- `pyxrt.hw_context`
- `pyxrt.kernel`
- `pyxrt.bo`
- `pyxrt.run`

相邻 operator 之间通过同一个 device buffer 交接。当前 Qwen3 attention pipeline
仍然是多个 xclbin 顺序运行，但中间 hidden、q/k/v、RoPE 后的 q/k 都是 shared
`pyxrt.bo`，不是 host NumPy 中转。

导出侧也保持简单：模型 exporter 直接渲染具体 stage 的 Jinja MLIR 模板。对于
attention，导出物是一个带 tiling/dataflow 的 MLIR 文件：

```text
src/torch2air/export/kernels/templates/attention_core.mlir.j2
  -> src/models/quantized_qwen3/generated/run_attention_core.mlir
  -> air-opt / aiecc
  -> run_attention_core.xclbin + insts.bin
```

`attention_core.cc` 不是 tiling 的来源。它只实现一个 Q row 对一个 K/V tile 的
online-softmax 更新。

## attention_core 的输入输出

当前 attention stage 的公开 ABI 是：

```mlir
func.func @run_attention_core(
    %q: memref<Sx128xf32>,
    %k: memref<Sx128xf32>,
    %v: memref<Sx128xf32>,
    %output: memref<Sx128xf32>)
```

其中：

- `q` 来自 `q_norm_rope` 的输出。
- `k` 来自 `k_norm_rope` 的输出。
- `v` 来自 `v_proj` 的输出。
- `output` 是 attention 输出，后续会接 residual / o_proj 等 stage。

`q/k/v/output` 是公开 runtime ABI。内部可以用更适合 AIR routing 的 channel 组织数据，
但不改变 stage 边界。

## 当前 tile 策略

当前硬件验证过的 attention 参数是：

```text
HEAD_DIM=128
QUERY_TILE_ROWS=4
KEY_TILE_ROWS=4
SEQUENCE_LENGTH=16
```

MLIR 模板声明三个 channel：

```mlir
air.channel @attention_q []
air.channel @attention_kv []
air.channel @attention_output []
```

launch 侧按 Q tile 组织外层循环：

1. 把一个 `4x128` Q tile 放入 `attention_q`。
2. 从第 0 个 KV tile 开始，把每个 `4x128` K tile、`4x128` V tile 依次放入同一个
   `attention_kv` FIFO。
3. 从 `attention_output` 取回这个 Q tile 对应的 `4x128` 输出。

herd 侧对应地执行：

1. 分配 L1 buffer：`q_l1`、`k_l1`、`v_l1`、`row_max_l1`、`row_sum_l1`、`out_l1`。
2. 对每个 Q block，先从 `attention_q` 取 `q_l1`。
3. 对每个 KV block，按顺序从 `attention_kv` 取 `k_l1` 和 `v_l1`。
4. 对 Q tile 里的每一行调用 `attention_core_tile(...)`。
5. 全部 KV block 处理完后，把 `out_l1` 放回 `attention_output`。

K 和 V 使用同一个 FIFO 是有意的。早期尝试把 K/V 放到两个独立 channel 时，AIR placement
可能把它们分配到不同 shim 路径，导致 routing 或运行结果不稳定。单 FIFO 保证 K 后面紧跟
对应的 V，公开 ABI 仍然保持 q/k/v 三个独立参数。

## tile compute body

`src/torch2air/export/kernels/attention_core.cc` 实现的是：

```text
O = softmax(QK^T / sqrt(128)) @ V
```

它每次只处理一个 query row 和一个 4-row K/V tile。为了跨多个 KV tile 流式累计，
L1 中保留三类状态：

- `row_max_l1`: 当前 query row 已见 KV tile 的 running max。
- `row_sum_l1`: online softmax 的 running sum。
- `out_l1`: 未最终归一化前的 value accumulator。

每个 KV tile 的更新过程是：

1. 根据 `q_base + q_row` 和 `kv_base` 应用 causal mask。
2. 计算最多 4 个 score。
3. 用 online softmax 公式合并旧的 `(max, sum, acc)` 和当前 tile 的 score/value。
4. 最后一个 KV tile 结束时，用 `1 / row_sum` 归一化 `out_l1`。

4-token 路径满足 `SEQUENCE_LENGTH == KEY_TILE_ROWS`，因此 `attention_core.cc`
会走更直接的 single-tile 分支。16-token 路径会进入四个 KV tile 的 online softmax
路径，并在真实 NPU 上通过了 full-head `attention` 和 `self_attn` 验证。

真实 Qwen3 q/k 的 score 很容易出现非常尖锐的 softmax。当前 tile body 对
`tile_max - second_max > 20` 的情况直接走 one-hot sharp softmax，避免在 AIE external
kernel 里对实际贡献为 0 的项继续调用 `exp` 近似。非尖锐场景仍走近似 `exp`。

这里没有把完整 `SxS` attention score materialize 到 L3 或 L1。score 只在 tile body
里以标量形式存在，减少了中间 buffer 和 stage 间拷贝。

## 调试经验

这次 full-head `self_attn` 的失败最终收敛到 `attention_core` 的输入 DMA 等待，经验是：

- 先把问题从完整 pipeline 拆成 standalone kernel。`q/k/v -> attention_core` 用同一个
  full-head AIE 产物、同一个真实 NPU 跑，证明 small-score 随机输入能过，大幅值 q/k
  复现失败。这样排除了 shared BO 串接、RoPE、projection slicing 这些方向。
- 不要只看最终 attention output。临时 debug external kernel 把 `score0..score3`、
  `tile_max`、`weight0..weight3` 写回 output buffer，直接确认 AIE 上的 dot product
  本身和 PyTorch ROCm 对得上。
- score 对但 output 错，不一定是 softmax/update path。16-token full-head 场景里，
  失败表现是最后一个 V tile 还没有真正到 L1 就被 external kernel 使用，输出像旧
  accumulator 被缩放后缺了一段 V 贡献。
- `AIRRtToNpuPass.cpp` 对 MM2S input task 默认会 `dma_free_task`，不会等待输入 DMA
  完成。对 attention 这种同一个 K/V FIFO 里反复投递 K、V tile 的场景，正式路径在
  `airrt-to-npu` 后只针对 `run_attention_core` 把 Q/KV 输入 task 改成
  `issue_token=true` + `aiex.dma_await_task`。
- `double` 累加不是实用修复。它能表达更高精度，但在 AIE 上导致 program memory overflow。
  对这个 kernel，减少 tile body 复杂度比盲目提高精度更重要。
- 4-token 场景不需要在线状态。`SEQUENCE_LENGTH == KEY_TILE_ROWS` 时走 single-tile
  attention，既更简单，也避开了 first tile 后立刻从 L1 状态读回的风险。
- sharp softmax 是必要的专门路径。真实模型的 q/k norm+RoPE 后幅值很大，score gap 经常
  远大于 20；这时 PyTorch softmax 实际上已经接近 one-hot，NPU 侧也应该直接表达这个事实。
- 调试产物只留在 `.cache` 或临时工作目录，不进入正式路径。确认结论后，只把简化后的
  `attention_core.cc` 修复提交。

这条定位链比直接改完整 pipeline 更可靠：先证明数据搬运和 score 正确，再缩小到
softmax/update body，最后用真实模型链路回归。

## 系统化调试机制

MLIR-AIR/NPU 问题不要只靠调 tile 参数。现在按四层调试：

第一层是 AIR graph 和 lowering。官方工具里已经有 dependency graph dump 和 pass 级 IR
打印：

```bash
AIR_DUMP_DEP_GRAPHS=1 AIR_VERIFY_EACH=1 \
  UV=uv scripts/run-quantized-qwen3-attention-npu.sh

AIR_PRINT_IR_ROOT=.cache/debug-air/ir AIR_VERIFY_EACH=1 \
  UV=uv scripts/run-quantized-qwen3-attention-npu.sh
```

`AIR_DUMP_DEP_GRAPHS=1` 会生成 `*.dep-graph/combined.dot`、`host.dot` 和 herd/core 子图。
它适合检查 `air.channel.put/get` 是否成对、host runtime sequence 和 herd consumer 顺序
是否一致，以及某个状态 buffer 是否跨 loop 被错误复用。

`AIR_PRINT_IR_ROOT` 使用 MLIR 的 `--mlir-print-ir-after-all` 和
`--mlir-print-ir-tree-dir`，每个 `air-opt` 阶段都会保存 pass 后 IR。它适合定位
“哪一个 pass 把 DMA/channel/loop 变成了意外形态”。

第二层是 AIE lowering 后的地址和锁。`aiecc` 生成的
`<work_dir>/aiecc/input_with_addresses.mlir` 和 `*_core_*.ld.script` 是核心文件：

- `input_with_addresses.mlir` 能看到每个 L1 buffer 的地址、bank、lock、DMA BD 和
  `aie.runtime_sequence`。
- link script 能看到 stack 起点和大小，以及显式 buffer 是否贴得太近。
- 如果输出像“某一段 token/head 变成 0”或“后半段复用旧状态”，先看这里，而不是先改数值
  代码。

第三层是官方 runtime/trace 机制。MLIR-AIR 的 `aircc` / `XRTBackend` 支持
`debug_ir`、`trace_size`、`trace_offset`、`use_lock_race_condition_fix`；trace 通过
`air-to-aie insert-trace-packet-flow=true` 和 `airrt-to-npu trace-size/trace-offset`
写到 output buffer 后缀。它主要看事件和时间线，不直接看数值。对我们的手写 runtime，
trace 还需要把 output BO 预留 trace 后缀后再启用。

第四层是数值状态探针。AIR/AIE 没有方便的 printf；定位 state corruption 最直接的办法是
临时把内部状态写到 debug output：`q_base/kv_base/q_row`、`score0..3`、`tile_max`、
`row_max/row_sum`、`weight0..3`、`out_l1` 的指定几个元素。这个 debug ABI 只进 `.cache`
实验，不进入正式 stage ABI。

还有两个实用开关：

```bash
AIR_USE_LOCK_RACE_CONDITION_FIX=1 ...
AIR_STACK_SIZE=8192 ...
```

`AIR_USE_LOCK_RACE_CONDITION_FIX=1` 会把 `use-lock-race-condition-fix=true` 传给
`air-to-aie`，对应官方 extra dummy DMA BD 的 race workaround。它可以排查 lock/BD race，
但不是数值修复。这次 16-token/full-head/large-score standalone failure 在这些开关下仍然
复现，说明问题不在 stack 或 lock race：

```text
attention_core mismatch: max_abs=0.932426393032074 index=(2,1560)
actual=0.0 expected=0.932426393032074
```

继续插桩后发现 score、softmax 权重和 K tile 都正确，错的是最后一段 V tile 没有在使用前
到达 L1。修复点不是 external kernel 数学，而是 attention 的输入 DMA task 必须显式
`await`。修复后同一个 16-token/full-head standalone 用例通过：

```text
attention_core_max_abs 0.0013238788
allclose True rtol=0.05 atol=0.2
mean_ms 142.982
```

复现这个 standalone 验证用例可以直接用：

```bash
TOKEN_COUNT=16 QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 \
  Q_HEADS=16 KV_HEADS=8 ATTENTION_SCALE=4.0 \
  ATTENTION_RTOL=0.05 ATTENTION_ATOL=0.2 \
  UV=uv scripts/run-quantized-qwen3-attention-npu.sh
```

## 8/16-token 机制实验结论

把 `self_attn` 从 4 tokens 扩到 8/16 tokens 时，主要问题不在 attention 数学本身，而在
AIR lowering 后的硬件资源、数据流顺序和 runtime 资源生命周期。

第一类问题是 AIE stack。默认 `stack-size=1024` 时，8-token `attention_core` 会出现
特定行输出重复或状态污染，例如后半段 query row 读到前面 row 的结果。检查
`aiecc/input_with_addresses.mlir` 和 link script 后可以看到 stack 与 L1 buffer 在同一个
tile memory 里相邻分配。把默认 stack 提到 4096 后，`QUERY_TILE_ROWS=4` 和
`QUERY_TILE_ROWS=1` 的 8-token attention_core 都能稳定对拍通过。

第二类问题是物理 herd 行数。npu2_4col 上把 token 数直接映射成 8 行 herd 会触发 shim
DMA channel 和 placement 压力。现在的做法是把物理 token 并行度限制到 4 行，长一点的
sequence 在每个 tile 内用 `token_i += physical_rows` 的 loop 继续处理。这已经应用到
`input_layernorm`，8-token standalone norm 和完整 pipeline 都能通过。

第三类问题是多输出 DMA 的顺序。`rope_table` 早期在 core loop 里连续写 cos/sin 两个
输出，但 AIR lowering 会把 host 侧 runtime sequence 拆成“先取所有 cos，再取所有 sin”
的形态，和 producer 的 per-token cos/sin 交错顺序不一致。改成显式
`air.channel.put/get` 后，launch 侧和 herd 侧都按 token 顺序传递 cos 再 sin，8-token
RoPE table 在真实 NPU 上通过。这个结论也说明：有顺序约束的多路输入/输出，优先用
AIR channel 明确表达，不依赖循环里的多次 `air.dma_memcpy_nd` 被 lowering 成期望顺序。

第四类问题是 L1 bank 压力。当前 Q/K 的 Q4_K 权重 tile 是 `32x152xi32`，O projection
是 `32x304xi32`，V 的 Q6_K tile 是 `32x424xi32`。这些都超过或接近单个 16KB bank，
`aiecc` 会提示 bank-aware allocation 失败并退到 sequential allocation。现在真实 NPU
能跑通，但这是后续 projection/o_proj 性能和稳定性优化的主要卡点。

第五类问题是 XRT `hw_context` 数量。16-token full self_attn 当前是多个 xclbin 顺序执行：
前置 stage 需要 8 个 context，再一次性加载 16 个 attention head 会在第 17 个 context
触发 `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-22`。验证过逐个加载并释放 attention xclbin
可以连续跑完 16 个 head，所以正式 runner 对 attention 和 o_proj 采用按需加载：保留 shared
`pyxrt.bo`，但 kernel/context 只在对应 stage 执行期间存在。

## 后续定位规则

遇到 attention 或后续 decode/prefill 的 NPU 问题时，按下面顺序定位：

1. 先拆成 standalone kernel。完整 pipeline 只能说明最终错了，不能区分 shared BO 交接、
   AIR channel 顺序、external kernel 数值路径和 L1 资源问题。
2. standalone 也要用真实 NPU 和 PyTorch ROCm reference 对拍。NumPy 只能用于辅助观察，
   不能作为正式 reference。
3. 不只看最终 output。必要时临时把 score、tile max、softmax weight、running sum 或
   running max 写回 output buffer，确认错在搬运、dot product、softmax 还是状态更新。
4. 每次修改 MLIR 模板后检查生成物。重点看 `air.channel.put/get` 的顺序、`air.herd`
   的物理规模、`memref<..., 2>` 的 L1 buffer 形状，以及 lowering 后 runtime sequence
   是否和 producer/consumer 期望一致。
5. 看到结果像“某一行重复前一行”或“后半段 token 被污染”时，优先检查 AIE stack 和 L1
   address map。`aiecc/input_with_addresses.mlir`、link script 和 bank allocation warning
   比最终 Python 异常更有信息量。
6. `air-place-herds` 的 `No valid placement found` 目前可能是非致命诊断；是否真正失败
   以 `air-to-aie`、`aiecc`、真实 NPU 运行和 PyTorch ROCm 对拍为准。
7. 多 xclbin pipeline 如果在 `CREATE_HWCTX` 失败，先数同时存活的 `pyxrt.hw_context`。
   BO 可以跨 kernel 保留，context 不需要为后续 stage 常驻。
8. 不把调试专用 ABI 留进正式路径。调试时可以临时写 debug buffer，但确认结论后，正式
   kernel 仍保持简单的 stage ABI 和 shared `pyxrt.bo` 交接。

## lowering 和运行

standalone attention 的验证脚本是：

```bash
TOKEN_COUNT=16 QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 NPU_ITERATIONS=1 NPU_WARMUP=0 \
  Q_HEADS=16 KV_HEADS=8 ATTENTION_RTOL=0.05 ATTENTION_ATOL=0.2 \
  UV=uv scripts/run-quantized-qwen3-attention-npu.sh
```

脚本会检查生成 MLIR 里存在这些关键结构：

- `air.launch`
- `air.herd`
- `air.channel.put`
- `air.channel.get`
- `scf.for %q_block`
- `scf.for %kv_block`
- `func.call @attention_core_tile`
- `link_with = "attention_core.o"`

然后编译 `attention_core.o`，通过 AIR/AIE 工具链生成 xclbin 和 insts，最后在真实 NPU
上运行，并和 PyTorch ROCm reference 对拍。

完整 Qwen3 self-attn pipeline 的当前验证命令是：

```bash
TOKEN_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 \
  NPU_ITERATIONS=1 NPU_WARMUP=0 \
  UV=uv scripts/run-quantized-qwen3-pipeline-npu.sh self_attn
```

验证链路是：

```text
embed_tokens
  -> input_layernorm
  -> q_proj / k_proj / v_proj
  -> rope_table
  -> q_norm_rope / k_norm_rope
  -> attention_core
  -> o_proj
```

所有 stage 都在真实 NPU 上跑。reference 来自同一边界上的 PyTorch ROCm tensor。
host 读取中间 buffer 只用于最终校验，不参与 operator 之间的交接。

最近一次记录的结果：

```text
standalone attention_core, 16 tokens, 16 q heads / 8 kv heads:
  reference pytorch_rocm AMD Radeon 890M
  attention_core_max_abs 0.0013238788
  allclose True rtol=0.05 atol=0.2
  mean_ms 142.982

full self_attn pipeline, 16 tokens:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope->attention_core->o_proj shared pyxrt BO
  rope_cos_max_abs 2.6116613e-05
  rope_sin_max_abs 3.4570694e-06
  q_norm_rope_max_abs 5.7220459e-06
  k_norm_rope_max_abs 9.1552734e-05
  attention_core_max_abs 0.0022195578
  o_proj_max_abs 0.053668097
  max_abs 0.16560259
  allclose True rtol=0.05 atol=0.2
  mean_ms 19559.311
```

## 为什么现在这样做

这个实现选择的是最朴素、可验证的官方风格：

- stage ABI 直接用 memref 参数，不引入 graph JSON、manifest 或自定义 runtime object。
- tiling 和数据流在 MLIR 模板里表达，不放到 C++ 文件里。
- native object 只做 L1 tile compute body，通过 `link_with` 和 `llvm.emit_c_interface`
  接到 AIR lowering。
- q/k/v 与 output 仍然是独立 runtime buffer，便于和前后 stage 用 shared `pyxrt.bo`
  交接。
- 内部用 AIR FIFO channel 保证 Q tile、K/V tile 和 output tile 的顺序。

这和 `torch2vk` 的思路类似：多个编译产物可以通过 device buffer 交接。区别是这里的
编译产物是 AIR/AIE xclbin，device buffer 是 `pyxrt.bo`。

## 当前限制

- `HEAD_DIM` 固定为 128，匹配 Qwen3 当前 head。
- `KEY_TILE_ROWS` 固定为 4。`attention_core.cc` 现在显式展开了 4 行 K/V，避免 AIE L1
  栈数组带来的不稳定结果。
- `SEQUENCE_LENGTH` 必须能整除 `QUERY_TILE_ROWS` 和 `KEY_TILE_ROWS`。
- 当前完整 full-head `self_attn` 已经验证到 16 tokens。更长 context 需要继续验证
  token loop、channel FIFO 深度和 runtime sequence 的规模。
- 当前默认 `AIR_STACK_SIZE=4096`。它解决了 8-token attention_core 的 L1 状态污染，但
  会减少每个 tile 可用于显式 L1 buffer 的空间。
- 16-head attention 现在依赖按需加载 xclbin 来避开 `hw_context` 数量上限；后续如果做
  stage stitching 或 fusion，需要重新评估 context 生命周期和 BO 分配策略。
- projection 和 o_proj 的量化权重 tile 对 L1 bank 压力较大。现在能跑通，但后续需要把
  Q4_K/Q6_K 的 tile 形状和 ABI 再压低。
- `air-place-herds` 对部分形状会打印 `No valid placement found` 诊断，但当前记录里仍能
  生成 AIE IR、xclbin/insts，并得到通过的硬件结果。
- `q_norm_rope` / `k_norm_rope` 会因为单列输入 channel 压力超过 shim DMA limit 而被
  auto-upgrade 到 `dma_packet`。当前运行正确，但后续多 stage 融合时要继续关注。
- 当前 softmax 使用 tile body 里的近似 `exp`，并对尖锐 score 走 one-hot 分支，所以验证
  阈值按 NPU 数值路径设置为 `rtol=0.05`。

## 后续工作

1. 继续把 16-token 经验推广到 decode/prefill 更长 context，先验证 token loop 和 channel
   顺序，再扩大 heads/tiles。
2. 评估 `QUERY_TILE_ROWS` 和 `KEY_TILE_ROWS` 的组合，先保持单 herd，再考虑多 herd 或
   cascade。
3. 在保持 q/k/v ABI 的前提下，继续减少内部 channel 和 L1 buffer 压力。
4. 优先降低 Q4_K/Q6_K projection 的 L1 weight tile 和 bank 压力，再评估 stitched AIR 或
   fusion 是否值得做。
