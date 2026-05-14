# Q4_K Linear Spikes

本文只规划 `quantized_qwen3` 里 Q4_K projection 的实验路径。之前尝试把
`aten.linear.default` 展开成纯标量 Python AIR 循环，这条路删掉：它会生成大量标量 IR，
编译和运行都很慢，也没有学习到 MLIR-AIR 推荐的 tile/kernel 分工。

## 目标

第一阶段只做 `self_attn.q_proj` decode 路径：

- 输入来自已经跑通的 `embed_tokens -> input_layernorm`，shape 固定为 `1x1024xf32`。
- 权重使用真实 GGUF 里的 `model.layers.0.self_attn.q_proj.weight`，格式必须是 Q4_K。
- reference 使用同一个 PyTorch ROCm module，不使用 NumPy reference。
- 每个 spike 都要在真实 NPU 上跑出结果；只编译通过不算完成。

成功以后，同一套 Q4_K linear kernel 才扩展到 `k_proj`、`o_proj` 和 prefill chunk。

## 核心假设

Q4_K linear 的正确形态是：

```text
Python AIR kernel:
  表达 stage ABI、launch/segment/herd、L3->L1 DMA、tile 大小和 output chunk

external/vectorized tile kernel:
  只处理 L1 buffer 上的 Q4_K dot body
```

也就是说，tiling 策略属于 Python AIR 生成的 MLIR；Q4_K 解包、scale/min 应用和 dot
accumulation 属于 external tile kernel。不要再把 1024 维 dot product 写成 AIR 标量
load/mul/add 循环。

## 不做什么

- 不做纯标量 AIR dot body。
- 不先做完整 `2048` 行 projection 的单个大 xclbin。
- 不引入新的 graph wrapper 或 manifest。
- 不用动态 memref shape 表达 decode/prefill；NPU graph 使用固定 shape。
- 不把中间结果落回 host 做拼接。必要的 output chunk 可以由 host 多次 launch，但 chunk
  输出仍然是 device BO 语义。

## Spike 1: tile ABI

目的：先证明 Python AIR 调度 external kernel 的 ABI、DMA 和 L1 buffer 生命周期是稳定的。

固定参数：

```text
SEQUENCE_LENGTH = 1
HIDDEN_SIZE = 1024
OUTPUT_TILE_ROWS = 8 或 16
```

AIR 侧只做：

1. 从 L3 复制 `1x1024xf32` hidden 到 L1。
2. 从 L3 复制一个小的 weight tile 到 L1。
3. 调用 external tile function。
4. 把 `1xOUTPUT_TILE_ROWSxf32` 写回 L3。

external kernel 第一版可以输出可预测的非零值，例如每个 row 写 row index，用来排除
`air.dma_memcpy_nd`、`func.call` ABI 和 output DMA 问题。通过以后再换成简单 F32 dot，
最后才接 Q4_K。

完成标准：

- 真实 NPU 输出非零且和 PyTorch ROCm reference 对齐。
- `aiecc` 没有 routing error。
- 记录 external function 的 memref ABI 和 L1 buffer 尺寸。

2026-05-14 结果：

- 代码路径：
  - `src/torch2air/export/q4k_linear_spike1.py`
  - `src/torch2air/export/kernels/q4k_linear_spike1.cc`
  - `src/models/quantized_qwen3/run_q4k_linear_spike1.py`
- NPU: `RyzenAI-npu4`。
- ABI: `hidden memref<1x1024xf32, 2>`、`weight memref<16xi32, 2>`、
  `output memref<1x16xf32, 2>`。
- 真实 NPU 结果：`max_abs=2.3841858e-07`，`rtol=1e-6`，`atol=1e-6`，
  `mean_ms=0.675`。
- lowering 经验：external function 会被 `air-to-aie` 复制进 AIE core；`air-to-std`
  之后必须跑 `symbol-dce`，否则 `aiecc` 会因为同名 private func 重定义失败。
- 稳定经验已沉淀到 `recipes/external-kernel-tile-abi.md`。

## Spike 2: Q4_K tile body

目的：验证一个小 output tile 的真实 Q4_K dot。

输入仍然是 `1x1024`，一个 output tile 包含 8 或 16 行权重。每行权重有 4 个 Q4_K
block。需要比较两种 weight tile ABI：

```text
A. raw Q4_K bytes/words + 单独 scale_l1
B. 每行 144 字节 Q4_K + 追加 d/dmin f32 bit pattern
```

旧实现里 `32x152xi32` 的思路属于 B：`144` 个 Q4_K byte 被视为 `36xi32`，
再追加每行 `4 blocks * 2` 个 f32 side words。它的好处是 external kernel 只拿一个
weight memref；坏处是 L1 压力更高。Spike 需要用真实 NPU 数据决定保留哪种 ABI。

完成标准：

- 使用真实 GGUF `q_proj.weight`。
- 单 tile 输出与 PyTorch ROCm `torch.nn.Linear` 对拍。
- 记录 `max_abs`、`rtol/atol`、tile 行数和 L1 buffer 大小。

2026-05-14 结果：

- 代码路径：
  - `src/torch2air/export/q4k_linear_spike2.py`
  - `src/torch2air/export/kernels/q4k_linear_spike2.cc`
  - `src/models/quantized_qwen3/run_q4k_linear_spike2.py`
- NPU: `RyzenAI-npu4`。
- 输入：真实 `embed_tokens -> input_layernorm` 后的 token 0 hidden。
- 权重：真实 GGUF `model.layers.0.self_attn.q_proj.weight` 的 `[0:16)` 行。
- ABI: `hidden memref<1x1024xf32, 2>`、`weight memref<16x152xi32, 2>`、
  `output memref<1x16xf32, 2>`。
- layout B 通过：每行 `144` 字节 Q4_K 作为 `36xi32`，后面追加
  `4 blocks * 2` 个 f32 bit pattern，所以 `weight_words=152`。
- L1 大小：hidden 4096B、weight 9728B、output 64B，总计约 13.6KiB。
- 真实 NPU 结果：`max_abs=1.1920929e-07`，`rtol=1e-2`，`atol=1e-2`，
  `mean_ms=13.684`。
- 这个 spike 验证的是真实 Q4_K tile layout 和 external-kernel ABI。当前 C++ tile body
  还是 correctness-first 标量写法，吞吐不能作为正式 linear；后续必须继续做 vectorized
  dot body 和 output chunk 实验。
- 稳定经验已沉淀到 `recipes/packed-weight-tile-layout.md`。

## Spike 3: output chunk

目的：把一个 tile 扩展成完整 `q_proj` 的固定 output chunk。

先保持单列 herd，host 按 `output_rows_per_call` 多次 launch：

```text
q_proj[0:chunk]
q_proj[chunk:2*chunk]
...
```

这和 torch2vk 多个 dispatch 拼完整 tensor 的方式一致。这里要学习的是 chunk 大小和
external tile body 的吞吐，而不是急着把所有 output rows 塞进一个 xclbin。

完成标准：

- 完整 `1x2048` q_proj 输出在真实 NPU 上对齐 PyTorch ROCm。
- 记录 `output_rows_per_call = 16/32/64` 的编译结果、运行时间和失败模式。
- 如果多列 herd routing 失败，先保留单列可用版本，并把失败 IR 和 aiecc 日志记下来。

2026-05-14 结果：

- 代码路径：
  - `src/models/quantized_qwen3/run_q4k_linear_spike3.py`
  - 复用 `src/torch2air/export/q4k_linear_spike2.py`
  - 复用 `src/torch2air/export/kernels/q4k_linear_spike2.cc`
- 输入：真实 `embed_tokens -> input_layernorm` 后的 token 0 hidden。
- 权重：完整真实 GGUF `model.layers.0.self_attn.q_proj.weight`，输出 `1x2048`。
- reference：PyTorch ROCm `torch.nn.Linear`，同一个 dequantized Q4_K 权重。
- 单列 host output chunk 结果：

| output_tile_rows | chunk_count | max_abs | mean_ms | 编译/运行状态 |
| --- | ---: | ---: | ---: | --- |
| 16 | 128 | `2.8610229e-06` | `1748.456` | 通过，无 L1 allocation warning |
| 32 | 64 | `2.8610229e-06` | `1725.858` | 通过，有 `32x152xi32` L1 bank allocation warning |
| 64 | 32 | `2.8610229e-06` | `1696.613` | 通过，有 `64x152xi32` L1 bank allocation warning |

- 结论：host chunk 可以工作，和 torch2vk 多次 dispatch 拼完整 tensor 的方式一致。但当前
  external C++ dot body 是 correctness-first 标量写法，减少 launch 数几乎不能改善总时间。
- tile size 不能只按总 L1 字节判断。`32x152xi32` 虽然总量还能放下，但 bank-aware allocation
  已失败；正式路径先保守使用 16-row tile，后续优先做 vectorized tile body 和多列并行。
- 稳定经验已沉淀到 `recipes/output-feature-chunking.md`。

## Spike 4: 多列 herd / packet routing

目的：理解 projection 能不能在一个 launch 里并行多个 output tile。

逐步尝试：

```text
1 column -> 2 columns -> 4 columns
```

每个 column 处理不同 output tile，读取同一个 hidden tile，写不同 output slice。重点观察
AIR lowering 生成的 channel、packet id、lock 和 DMA BD，而不是只调 tile size。

完成标准：

- 至少一个多列版本在真实 NPU 上对拍成功；或者记录官方工具链下可复现的 routing 限制。
- 如果失败，给出继续使用 host chunk 的明确理由。

2026-05-15 结果：

- 代码路径：
  - `src/torch2air/export/q4k_linear_spike4.py`
  - `src/models/quantized_qwen3/run_q4k_linear_spike4.py`
  - 复用 `src/torch2air/export/kernels/q4k_linear_spike2.cc`
- 输入：真实 `embed_tokens -> input_layernorm` 后的 token 0 hidden。
- 权重：真实 GGUF `model.layers.0.self_attn.q_proj.weight`，从 row 0 开始。
- tile 策略：每个 AIE tile 处理 `16` 个 output rows，多个 herd column 处理不同 row slice。
- 真实 NPU 结果：

| herd_cols | output_rows | max_abs | mean_ms | 编译/运行状态 |
| ---: | ---: | ---: | ---: | --- |
| 2 | 32 | `2.3841858e-07` | `13.932` | 通过，无 routing error |
| 4 | 64 | `3.5762787e-07` | `13.880` | 通过，无 routing error |

- 结论：对 Q4_K projection schedule，4-column herd 可以工作。每列独立读取同一个 hidden L3
  和自己的 weight slice，写回不同 output slice，AIR/AIE packet routing 稳定。
- 这比 Spike 3 的 64-row 单 tile 更干净：单 tile 64 rows 会触发 `64x152xi32` L1 bank
  allocation warning；4-column 版本每个 tile 仍然是 `16x152xi32`，没有这个 warning。
- 稳定经验已沉淀到 `recipes/output-parallel-herd.md`。

## Spike 5: prefill chunk

目的：把 decode 的 `S=1` 扩展到固定 prefill chunk。

候选 shape：

```text
SEQUENCE_LENGTH = 8 或 16
HIDDEN_SIZE = 1024
OUTPUT_TILE_ROWS = 8 或 16
```

NPU graph 仍然固定 shape。真实序列长度不足一个 chunk 时由 host pad；attention mask/KV 更新
在后续 attention/KV stage 处理，不让 Q4_K linear 自己发明动态 shape。

完成标准：

- `S=8/16` 的 q_proj 输出和 PyTorch ROCm 对拍。
- 记录 token 维 tile 是否复用 hidden DMA、是否需要把 token loop 放进 external kernel。

2026-05-15 结果：

- 代码路径：
  - `src/torch2air/export/q4k_linear_spike5.py`
  - `src/models/quantized_qwen3/run_q4k_linear_spike5.py`
  - 复用 `src/torch2air/export/kernels/q4k_linear_spike2.cc`
- 输入：真实 `embed_tokens -> input_layernorm` 后的 token chunk hidden。
- 权重：真实 GGUF `model.layers.0.self_attn.q_proj.weight` 的 `[0:16)` 行。
- reference：PyTorch ROCm `torch.nn.Linear`，同一个 dequantized Q4_K 权重。
- 验证过的 token tiling：

| shape | herd | token loop | max_abs | mean_ms | 编译/运行状态 |
| --- | --- | --- | ---: | ---: | --- |
| `S=2, rows=16` | `1x1` | 单 tile 2 token | `1.7881393e-07` | `27.502` | 通过 |
| `S=4, rows=16` | `1x1` | 单 tile 4 token | `5.9604645e-07` | `53.439` | 通过 |
| `S=8, rows=16` | `4x1` | 每 tile 2 token | `5.9604645e-07` | `27.306` | 通过 |
| `S=8, rows=16` token 8-15 | `4x1` | 每 tile 2 token | `1.2516975e-06` | `27.174` | 通过 |

- 失败模式：
  - `S=8, 1x1 herd` 编译通过，但运行时后半 token 出现 0，说明一个 tile 内长 token loop
    对当前 DMA/compute ordering 不可靠。
  - `S=16, 4x1 herd` 编译通过，但第三轮 token 出现 0。
  - `S=16, 4x2 herd` 可以把每个 tile 限制到 2 个 token，但 `air-to-aie` 报
    `failed to map to shim dma channels: out of channels`。
- 结论：当前正确的 prefill bucket 是固定 `S=8`。`S=16` 先由 host 拆成两个 `S=8`
  launch，而不是在一个 xclbin 里硬塞更多 token channel。
- 稳定经验已沉淀到 `recipes/static-token-buckets.md`。

## Spike 6: 接入正式路径

只有前面 spike 跑通以后，才把结果接进正式 exporter：

- `aten.linear.default` 映射到 Q4_K linear Python AIR kernel。
- kernel 代码内聚在 projection 对应模块里，不把 q_proj 的临时字段塞进通用 pipeline 类型。
- `q_proj/k_proj/o_proj` 复用同一个已验证 kernel 配置；不同 projection 只改 tensor 名、
  output rows 和调用顺序。
- reference 继续由导出的 PyTorch module 产生，和 NPU 跑同一个输入。

正式路径的文档再沉淀到 `recipes/`，只保留短小、稳定的经验，例如 external kernel ABI、
Q4_K weight tile layout、多列 routing 限制。

2026-05-15 结果：

- 正式代码路径：
  - `src/torch2air/export/q4k_linear.py`
  - `src/torch2air/export/kernels/q4k_linear.cc`
  - `src/torch2air/runtime/compile.py`
  - `src/models/quantized_qwen3/run_q_proj.py`
- 导出路径：`models.quantized_qwen3.export --stage q_proj --sequence-length 1` 生成的
  Python kernel 直接喂给 `Q4KLinearAirBuilder`，没有再包装一层 graph。
- 运行策略：decode `S=1`，每个 xclbin 计算 `64` 个 output rows；内部是 4-column herd，
  每列 `16` rows。完整 projection 由 host 连续跑 output chunks。
- 真实 NPU 结果：

| stage | weight | output_features | chunks | max_abs | mean_ms |
| --- | --- | ---: | ---: | ---: | ---: |
| q_proj | `model.layers.0.self_attn.q_proj.weight` | 2048 | 32 | `2.8610229e-06` | `435.735` |
| k_proj | `model.layers.0.self_attn.k_proj.weight` | 1024 | 16 | `2.8610229e-06` | `220.341` |

- reference: PyTorch ROCm `torch.nn.Linear`，`rtol=1e-2`，`atol=1e-2`。
- 正式 `compile_runtime` 已改成显式检查 xclbin 和 insts 是否存在，避免把缺产物的编译误判为成功。
- 稳定经验已沉淀到 `recipes/export-aten-linear-kernel.md`。

## 与 KV 动态长度的关系

Q4_K projection 本身不管理 KV cache。后续 decode/prefill 的动态长度策略应该是：

- device BO 为最大上下文或当前 batch bucket 预分配固定 shape。
- NPU xclbin 使用固定 shape，例如 decode `S=1`、prefill `S=16`。
- host 传入 `start_position`、`valid_length` 这类标量或选择固定 bucket。
- tail token 用 padding/mask 处理，而不是在 AIR 里创建动态 memref。

这样 projection kernel 只负责固定 shape tensor 计算，KV 的写入位置和有效长度属于后续
attention/KV stage。
