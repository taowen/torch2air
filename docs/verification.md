# Verification Notes

## AIR Toolchain

The development environment is installed with `uv` and Python 3.12:

```bash
scripts/install-air-tools.sh
```

Current verified wheel versions:

- `mlir-air==0.0.1.2026051305+90dc5e9`
- `mlir-aie==0.0.1.2026050821+b37dc33`
- `llvm-aie==21.0.0.2026051201+3fbeee11`

The `mlir-air/` submodule is pinned to commit
`90dc5e92ad8263bcfb81a6743ac59886508d5551`, matching the installed AIR wheel.

## Spike 1: Direct Tiled Matmul

Run:

```bash
scripts/verify-air-spike1.sh
```

Input:

```text
examples/amd_aie_experiments/air_direct_matmul_tiled.mlir
```

Generated outputs:

```text
examples/amd_aie_experiments/generated/air_direct_matmul_tiled.dma.mlir
examples/amd_aie_experiments/generated/air_direct_matmul_tiled.channel.mlir
examples/amd_aie_experiments/generated/air_direct_matmul_tiled.aie.mlir
```

Verified checks:

- `air-par-to-herd` produces `air.herd`.
- `air-copy-to-dma` converts local/global copies to `air.dma_memcpy_nd`.
- `air-dma-to-channel` produces matching `air.channel.put` and
  `air.channel.get`.
- `air-to-aie` produces `aie.device(npu2_4col)` and `aie.core`.

Notes:

- The current toolchain rejects `device=npu4` with `Invalid aie.device option`.
  The local `RyzenAI-npu4` path is verified through `AIR_DEVICE=npu2_4col`,
  which is also the verification script default. Lowering the upstream
  external-kernel smoke test as `device=npu1` produced all-zero output on this
  machine, so `npu1` is not a valid default here.
- `air-place-herds` needs `row-anchor=2` for this NPU path so generated
  buffers are placed on tiles with local memory.
- The direct fixture intentionally copies A, B, and C into L1, so
  `air-dma-to-channel` warns that three input channels are upgraded to
  `dma_packet`. This is acceptable for the feasibility check and is useful
  evidence for later DMA-pressure experiments.

Conclusion: the central Spike 1 hypothesis is feasible with the current
MLIR-AIR wheel stack. A hand-written schedule with `scf.parallel`,
`memref.subview`, `memref.alloc` in memory space `2`, and `memref.copy` lowers
to herd, DMA, channels, and AIE core IR.

## Spike 2: Memory Space Contract

Run:

```bash
scripts/verify-air-spike2-memory.sh
```

Verified checks:

- `memref<..., 1>` works as the L2/memtile memory-space convention when the
  allocation is outside the inner `air.herd`.
- `memref<..., 2>` works as the L1/core-local memory-space convention inside
  the herd.
- The fixture lowers through DMA, channels, and AIE.
- The real NPU check uses MLIR-AIR's `eltwise_add_with_l2` example and reports
  `PASS!`.

Conclusion: templates should model L2 and L1 as distinct explicit buffers.
Early templates should keep tile shapes static and avoid allocating L2 buffers
inside the herd body.

## Spike 3: DMA Ordering And Channels

Run:

```bash
scripts/verify-air-spike3-dma-order.sh
```

Verified checks:

- The fixture contains separate activation, weight, bias/scale, and output
  copies.
- `air-copy-to-dma` emits at least four distinct `air.dma_memcpy_nd` ops.
- `air-dma-to-channel` emits matching `air.channel.put` and
  `air.channel.get` operations.
- The real NPU check uses MLIR-AIR's `passthrough_channel` example and reports
  `PASS!`.

Conclusion: AIR can preserve an explicit multi-copy tile schedule well enough
for tiled MLIR templates to control DMA order before lowering.

## Spike 4: Double Buffering Skeleton

Run:

```bash
scripts/verify-air-spike4-double-buffer.sh
```

Verified checks:

- The fixture has separate ping/pong activation buffers and ping/pong weight
  buffers.
- The lowered DMA IR contains the staged copies needed for a hand-written
  pipeline.
- The real NPU check uses MLIR-AIR's `herd_dataflow` example with
  `M_SIZE=64`, `N_SIZE=256`, and `AIE_TARGET=aie2p`; it reports `PASS!`.

Conclusion: double buffering should be emitted directly in the schedule rather
than left to a generic optimizer.

## Quantized Qwen3: Generated Embed Tokens

Run:

```bash
scripts/verify-quantized-qwen3-stage.sh embed_tokens
```

Useful scale-up variants:

```bash
AIR_DEVICE=npu2_4col BLOCKS_PER_ROW=1 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh embed_tokens

AIR_DEVICE=npu2_4col BLOCKS_PER_ROW=4 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh embed_tokens

AIR_DEVICE=npu2_4col TOKEN_IDS=0,1 BLOCKS_PER_ROW=4 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh embed_tokens

AIR_DEVICE=npu2_4col BLOCKS_PER_ROW=4 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh input_layernorm

AIR_DEVICE=npu2_4col TOKEN_IDS=0,1 BLOCKS_PER_ROW=4 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh input_layernorm

AIR_DEVICE=npu2_4col TOKEN_IDS=0 OUTPUT_ROWS=64 OUTPUT_TILE_ROWS=16 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh q_proj

AIR_DEVICE=npu2_4col TOKEN_IDS=0,1,2,3,4,5,6,7 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh q_proj

AIR_DEVICE=npu2_4col TOKEN_IDS=0 OUTPUT_ROWS=64 OUTPUT_TILE_ROWS=16 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh k_proj

AIR_DEVICE=npu2_4col TOKEN_IDS=0,1,2,3,4,5,6,7 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh k_proj

AIR_DEVICE=npu2_4col TOKEN_IDS=0 OUTPUT_ROWS=64 OUTPUT_TILE_ROWS=16 NPU_ITERATIONS=1 \
  scripts/run-quantized-qwen3-npu.sh o_proj
```

Inputs:

```text
/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf
tensor: model.embed_tokens.weight
format: GGUF Q4_K
physical shape: [151936, 144] uint32
```

Generated artifacts:

```text
src/models/quantized_qwen3/generated/run_embed_tokens.mlir
src/models/quantized_qwen3/generated/lowered/quantized_qwen3_embed_tokens.dma.mlir
src/models/quantized_qwen3/generated/lowered/quantized_qwen3_embed_tokens.channel.mlir
src/models/quantized_qwen3/generated/lowered/quantized_qwen3_embed_tokens.aie.mlir
.cache/npu-spikes/quantized-qwen3-embed_tokens-*/run_embed_tokens.xclbin
.cache/npu-spikes/quantized-qwen3-embed_tokens-*/run_embed_tokens.insts.bin
```

Verified checks:

- `models.quantized_qwen3.export.export_one(...)` uses `torch.export` and walks
  `program.graph_module.graph.nodes` directly.
- The generated artifact is tiled pre-AIR MLIR with `scf.parallel`,
  `memref.subview`, L1 `memref.alloc`, and `memref.copy`.
- `air-opt` lowers it to `air.herd`, `air.dma_memcpy_nd`, channels, and
  `aie.device(npu2_4col)`.
- Runtime compilation uses `air-opt --air-to-std --airrt-to-npu` followed by
  `aiecc --aie-generate-xclbin --aie-generate-npu-insts`.
- Hardware execution uses MLIR-AIR's `air.backend.xrt.XRTBackend` directly; no
  torch2air pyxrt wrapper is introduced.
- The NPU reads real packed Q4_K rows from GGUF. It decodes per-subblock
  scale/min bytes and q4 nibbles inside the tiled MLIR body.
- The block-level f16 `d` and `dmin` values are temporarily host-decoded into a
  small f32 side input because the current AIE scalar f16 conversion path
  produced incorrect values. Full Q4_K value dequantization is still done by the
  NPU tile.
- `input_layernorm` consumes a hidden-state memref directly and writes the
  normalized hidden-state memref. There is no graph/manifest/`.npy` handoff.
- `pipeline_embed_norm` exports and lowers the two stages independently,
  generates a stitched AIR module for inspection, and runs the stage xclbins
  with a shared `pyxrt.bo` hidden buffer. The hidden state is copied back only
  after both NPU kernels finish so the verifier can check it.
- `embed_tokens_input_layernorm` is a fused L1 handoff spike. It keeps the
  dequantized embedding values in L1 and feeds RMSNorm without writing an
  intermediate global hidden buffer.
- `q_proj` and `k_proj` use official-style AIR external kernel lowering with
  `q4k_linear.o`; `v_proj` uses the same AIR shape with `q6k_linear.o` because
  the real GGUF tensor is Q6_K.
- Attention projection kernels still use the temporary f32 sidecar workaround
  for block-level f16 scales. The Q6_K path also temporarily widens GGUF
  halfwords to i32 words for the external-kernel ABI; quantized values are still
  decoded by the NPU tile body.
- `embed_tokens -> input_layernorm -> q/k/v` runs stage xclbins with shared
  `pyxrt.bo` handoff buffers. It does not copy intermediate hidden states
  through NumPy arrays between operators.
- `rope_table` and `q/k_norm_rope` use official-style AIR external kernels.
  The generated MLIR owns launch, herd placement, L1 buffers, and DMA; the
  external `.cc` files are only tile compute bodies. RoPE theta is compiled
  into `rope_table.o` as `ROPE_INV_FREQ_RATIO`, while the runtime ABI passes
  only the position and output cosine/sine buffers.
- `embed_tokens -> input_layernorm -> q/k/v -> rope_table -> q/k_norm_rope`
  runs seven stage xclbins with shared `pyxrt.bo` handoff buffers. The verifier
  reads intermediate buffers only after the NPU chain finishes, then compares
  against PyTorch ROCm.
- Attention and pipeline verification from the removed `legacy/` tree is not a
  production source. `v_proj`, RoPE, attention core, and full self-attention
  must be re-established through fresh Python AIR experiments before new
  verification records are added here.
- The current `quantized_qwen3` reference path computes expected tensors,
  `allclose`, and max-abs metrics with torch tensors on the ROCm device. NumPy
  is used only for GGUF byte slicing and XRT host buffers.

Latest real NPU results:

```text
2026-05-14 safetensors PyTorch ROCm reference:

embed_tokens, 1 token, 4 Q4_K blocks, hidden_size 1024:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.0054986477
  allclose True rtol=0.01 atol=0.01
  mean_ms 1.435

input_layernorm, 1 token, hidden_size 1024:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 1.9073486e-06
  allclose True rtol=0.0001 atol=1e-05
  mean_ms 1.497

embed_tokens_input_layernorm fused L1 handoff, 1 token, 1 Q4_K block:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.18145949
  allclose True rtol=0.05 atol=0.2
  mean_ms 1.135

pipeline_embed_norm shared BO handoff, 1 token, hidden_size 1024:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  allclose True rtol=0.05 atol=0.2
  mean_ms 2.304

q_proj external Q4_K kernel, 1 token, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.058996558
  allclose True rtol=0.05 atol=0.1
  mean_ms 14.544

q_proj external Q4_K kernel, 2 tokens, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.058996558
  allclose True rtol=0.05 atol=0.1
  mean_ms 15.114

k_proj external Q4_K kernel, 1 token, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.03157258
  allclose True rtol=0.05 atol=0.1
  mean_ms 14.570

k_proj external Q4_K kernel, 2 tokens, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.03342998
  allclose True rtol=0.05 atol=0.1
  mean_ms 15.482

v_proj external Q6_K kernel, 1 token, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.00775823
  allclose True rtol=0.05 atol=0.1
  mean_ms 10.752

v_proj external Q6_K kernel, 2 tokens, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.00775823
  allclose True rtol=0.05 atol=0.1
  mean_ms 11.287

q_proj external Q4_K kernel, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.058996558
  allclose True rtol=0.05 atol=0.1
  mean_ms 27.872

q_proj external Q4_K kernel, 1 token, output_rows 128, output_tile_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.058996558
  allclose True rtol=0.05 atol=0.1
  mean_ms 53.342

q_proj external Q4_K kernel, 2 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.058996558
  allclose True rtol=0.05 atol=0.1
  mean_ms 28.014

k_proj external Q4_K kernel, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.03157258
  allclose True rtol=0.05 atol=0.1
  mean_ms 27.663

k_proj external Q4_K kernel, 2 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.046132445
  allclose True rtol=0.05 atol=0.1
  mean_ms 28.110

v_proj external Q6_K kernel, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.00775823
  allclose True rtol=0.05 atol=0.1
  mean_ms 20.167

v_proj external Q6_K kernel, 2 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  max_abs 0.0078741657
  allclose True rtol=0.05 atol=0.1
  mean_ms 20.814

pipeline_embed_norm_qproj shared BO handoff, 1 token, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  qproj_max_abs 0.081097126
  allclose True rtol=0.05 atol=0.2
  mean_ms 16.258

pipeline_embed_norm_qproj shared BO handoff, 2 tokens, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  qproj_max_abs 0.081097126
  allclose True rtol=0.05 atol=0.2
  mean_ms 17.821

pipeline_embed_norm_qproj shared BO handoff, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  qproj_max_abs 0.081097126
  allclose True rtol=0.05 atol=0.2
  mean_ms 29.340

pipeline_embed_norm_qkv shared BO handoff, 1 token, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.031209335
  allclose True rtol=0.05 atol=0.2
  mean_ms 41.817

pipeline_embed_norm_qkv shared BO handoff, 2 tokens, output_rows 64:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.037101876
  allclose True rtol=0.05 atol=0.2
  mean_ms 43.504

pipeline_embed_norm_qkv shared BO handoff, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.031209335
  allclose True rtol=0.05 atol=0.2
  mean_ms 77.533

pipeline_embed_norm_qkv_rope shared BO handoff, 1 token, output_rows 128, output_tile_rows 32, start_position 1:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.031209335
  rope_cos_max_abs 2.9802322e-07
  rope_sin_max_abs 1.4901161e-07
  q_norm_rope_max_abs 2.8610229e-06
  k_norm_rope_max_abs 1.5258789e-05
  allclose True rtol=0.05 atol=0.2
  mean_ms 81.063

pipeline_embed_norm_qkv_rope_attention shared BO handoff, 1 token, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.031209335
  rope_cos_max_abs 0
  rope_sin_max_abs 0
  q_norm_rope_max_abs 2.8610229e-06
  k_norm_rope_max_abs 1.5258789e-05
  attention_core_max_abs 0
  allclose True rtol=0.05 atol=0.2
  mean_ms 81.621

pipeline_embed_norm_qkv_rope_attention shared BO handoff, 2 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.037101876
  rope_cos_max_abs 2.9802322e-07
  rope_sin_max_abs 1.4901161e-07
  q_norm_rope_max_abs 2.8610229e-06
  k_norm_rope_max_abs 3.0517578e-05
  attention_core_max_abs 0.18556878
  allclose True rtol=0.05 atol=0.2
  mean_ms 152.201

pipeline_embed_norm_qkv_rope_attention shared BO handoff, 4 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0059777498
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.04088977
  rope_cos_max_abs 2.4806999e-05
  rope_sin_max_abs 3.3974648e-06
  q_norm_rope_max_abs 2.8610229e-06
  k_norm_rope_max_abs 3.0517578e-05
  attention_core_max_abs 0.065862715
  allclose True rtol=0.05 atol=0.2
  mean_ms 293.233

standalone tiled attention_core, 8 tokens, head_dim 128, q4/kv4:
  reference pytorch_rocm AMD Radeon 890M
  attention_core_max_abs 0.025021695
  allclose True rtol=0.05 atol=0.05
  mean_ms 2.699

pipeline_embed_norm_qkv shared BO handoff, 2 tokens, output_rows 128, output_tile_rows 32:
  reference safetensors_pytorch_rocm AMD Radeon 890M
  hidden_max_abs 0.0054986477
  max_abs 0.16560259
  q_proj_max_abs 0.081097126
  k_proj_max_abs 0.093719244
  v_proj_max_abs 0.037101876
  allclose True rtol=0.05 atol=0.2
  mean_ms 78.332
```

Notes:

- `air-place-herds` currently prints `No valid placement found` diagnostics for
  the 1x4 and 2x4 variants, but still produces AIE IR, xclbin/insts, and a
  passing hardware result. Track this as a placement diagnostic issue, not as a
  runtime failure.
- Full-head projection verification uses `OUTPUT_ROWS=128` with
  `OUTPUT_TILE_ROWS=32`. The `128/16` shape creates eight projection tiles and
  fails during channel lowering with `failed to map to shim dma channels: out of
  channels`. Q4_K with `OUTPUT_TILE_ROWS=64` reaches hardware, but is slower
  than 32 rows and emits larger L1 bank-allocation warnings.
- Q6_K projection variants currently print AIE bank allocation warnings because
  the spike widens Q6_K halfwords to i32 in L1. `OUTPUT_TILE_ROWS=32` still
  runs and matches the PyTorch ROCm reference, but `OUTPUT_TILE_ROWS=64` fails
  AIE allocation because the widened `memref<64x424xi32>` weight tile is
  108544 bytes before the activation/output buffers. Compact i16 L1 DMA is the
  next cleanup for that kernel.
- Passing `aie.runtime_sequence`/`aiex.dma_configure_task_for` MLIR directly to
  `aiecc` is required for this flow. Pre-lowering to `aiex.npu` instructions
  before `aiecc` produced a hanging run.
- Full hidden `embed_tokens_input_layernorm` fusion is not enabled yet. The
  naive single-core full fusion either overflows program memory when fully
  unrolled, or produces invalid values when the Q4_K block copy/store loop uses
  dynamic block offsets. The working full-hidden path is currently
  `embed_tokens` followed by `input_layernorm`, with a direct hidden memref stage
  boundary.
- The stitched AIR module for `embed_tokens -> input_layernorm` is generated and
  contains two `air.launch` regions with the expected arg-map handoff. The
  stitched AIE lowering now keeps renamed private external declarations and
  reaches AIE IR for the 4-token `embed_tokens -> input_layernorm` debug path.
  Running that exact stitched module through the upstream ELF path is blocked in
  this Python 3.12 environment because `pyxrt` does not expose `elf`/`ext`.
  Running it through `aircc --output-format=xclbin` currently overflows program
  memory for the full Q4_K embedding body. The verified hardware path therefore
  uses two xclbins and one shared `pyxrt.bo` for the intermediate hidden buffer.
- The full four-token attention pipeline now runs end to end on the real NPU.
  The previous shim-DMA channel pressure was removed by keeping the token
  dimension inside stage-local loops for `embed_tokens` and q/k/v projection,
  while the herd maps only block or output-tile columns.
- The full eight-token attention pipeline still fails before attention:
  `input_layernorm` maps token rows directly to an 8-row herd and fails channel
  lowering with `failed to map to shim dma channels: out of channels`. The next
  full-pipeline scaling step is to apply the same stage-local token loop pattern
  to `input_layernorm`, `rope_table`, and q/k norm+RoPE.
- Standalone `attention_core` now reaches 8 tokens with a formal tiled
  online-softmax schedule. It keeps q/k/v as runtime arguments, uses Q and K/V
  AIR FIFO channels internally, and compares against PyTorch ROCm.
- The one-token full attention pipeline remains a real Qwen3 handoff test, but
  it is the causal-attention degenerate case where the output equals the first V
  row. The full two-token run and the standalone 2-token and 4-token
  `attention_core` runs cover non-degenerate causal softmax.

## Spike 6: FlashAttention Schedule Skeleton

Run:

```bash
scripts/verify-air-spike6-flash-attn.sh
```

Verified checks:

- The fixture keeps Q resident in L1 and stages K/V copies before accumulator
  updates.
- The fixture lowers through DMA, channels, and AIE.
- The real NPU check uses MLIR-AIR's upstream
  `flash_attention/kernel_fusion_based` kernel with:

```text
LK=512 LKP=64 LQ=512 LQP=256 DK=64 DV=64 NUM_HEADS=2 NUM_KV_HEADS=2
EXTRA_PY_FLAGS="--output-format xclbin"
```

- It runs on `RyzenAI-npu4` through XRT and reports:

```text
Output 0 correlation: 0.997371 (threshold: 0.99)
PASS!
```

Notes:

- `LQP=64` is not valid for this AIE2p cascade path because the cascade value
  is 16xbf16, or 256 bits, while AIE2p requires 512-bit cascade transfers.
- The current Python 3.12 `pyxrt` binding does not expose `pyxrt.elf`, so the
  verification uses xclbin output instead of the upstream ELF default.

Conclusion: the staged FlashAttention schedule is feasible, but valid tile
sizes must respect AIE2p cascade width.

## NPU Hardware Smoke Test

Device discovered through XRT:

```text
RyzenAI-npu4, architecture aie2p, topology 6x8
```

The installed XRT package provides `pyxrt` for system Python 3.14 only. The
repository environment uses Python 3.12, so `scripts/build-pyxrt.sh` builds
`pyxrt.cpython-312-x86_64-linux-gnu.so` from the XRT source checkout under
`../iree-amd-aie/third_party/XRT` and links it against the installed XRT 2.23
runtime.

Run:

```bash
scripts/build-pyxrt.sh
scripts/run-npu-smoke.sh
```

Verified result:

```text
devices 1
PASS!
```

The smoke test uses MLIR-AIR's upstream i8 matrix multiplication fixture with
`AIE_TARGET=aie2p`, compiles a 1x1 herd kernel, loads it through XRT on the
NPU, and validates the result.
