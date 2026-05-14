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

当前硬件验证过的参数是：

```text
HEAD_DIM=128
QUERY_TILE_ROWS=4
KEY_TILE_ROWS=4
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

这里没有把完整 `SxS` attention score materialize 到 L3 或 L1。score 只在 tile body
里以标量形式存在，减少了中间 buffer 和 stage 间拷贝。

## lowering 和运行

standalone attention 的验证脚本是：

```bash
TOKEN_COUNT=8 QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 NPU_ITERATIONS=1 NPU_WARMUP=0 \
  ATTENTION_RTOL=0.05 ATTENTION_ATOL=0.05 \
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

完整 Qwen3 attention pipeline 的当前验证命令是：

```bash
TOKEN_IDS=0,1,2,3 QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 NPU_ITERATIONS=1 NPU_WARMUP=0 \
  UV=uv scripts/run-quantized-qwen3-pipeline-npu.sh attention
```

验证链路是：

```text
embed_tokens
  -> input_layernorm
  -> q_proj / k_proj / v_proj
  -> rope_table
  -> q_norm_rope / k_norm_rope
  -> attention_core
```

所有 stage 都在真实 NPU 上跑。reference 来自同一边界上的 PyTorch ROCm tensor。
host 读取中间 buffer 只用于最终校验，不参与 operator 之间的交接。

最近一次记录的结果：

```text
standalone attention_core, 8 tokens:
  reference pytorch_rocm AMD Radeon 890M
  attention_core_max_abs 0.025021695
  allclose True rtol=0.05 atol=0.05
  mean_ms 2.699

full attention pipeline, 4 tokens:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope->attention_core shared pyxrt BO
  attention_core_max_abs 0.065862715
  allclose True rtol=0.05 atol=0.2
  mean_ms 293.233
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
- standalone attention 已经到 8 tokens；完整 pipeline 的 8-token 路径目前卡在
  attention 之前的 `input_layernorm` channel lowering，不是 `attention_core` 本身。
- `air-place-herds` 对部分形状会打印 `No valid placement found` 诊断，但当前记录里仍能
  生成 AIE IR、xclbin/insts，并得到通过的硬件结果。
- 当前 softmax 使用 tile body 里的近似 `exp`，所以验证阈值按 NPU 数值路径设置为
  `rtol=0.05`。

## 后续工作

1. 把 `input_layernorm`、`rope_table`、`q/k_norm_rope` 也改成 stage-local token loop，
   让完整 pipeline 能稳定扩到 8 tokens 以上。
2. 评估 `QUERY_TILE_ROWS` 和 `KEY_TILE_ROWS` 的组合，先保持单 herd，再考虑多 herd 或
   cascade。
3. 在保持 q/k/v ABI 的前提下，继续减少内部 channel 和 L1 buffer 压力。
4. 等多个 stage 的单独 xclbin 路径稳定后，再评估是否需要更紧的 stitched AIR 或 fusion。
