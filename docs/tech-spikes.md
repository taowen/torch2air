# Technical Spikes

This document records the hardware conclusions that still guide `torch2air`.
Retired learning spikes are summarized here instead of kept as active planning
items.

## Working Model

`torch2air` writes the tile schedule explicitly as MLIR, then uses MLIR-AIR to
lower that schedule to herd, DMA, channel, and AIE forms:

```text
PyTorch export -> direct graph walk -> Jinja tiled MLIR -> air-opt/aiecc -> XRT
```

The generated MLIR owns:

- `air.launch`, `air.segment`, and `air.herd` placement.
- Tile loops with `scf.parallel` / `scf.for`.
- Global slices with `memref.subview` or explicit `air.dma_memcpy_nd`.
- L1 buffers with `memref.alloc(..., 2)`.
- Operator-to-operator ABI as direct memref arguments.

Native `.cc` files are only tile compute bodies. They do not express tiling or
operator scheduling.

## Official External Kernel

Keep `scripts/run-mlir-air-official-external-kernel-spike.sh`. It validates the
smallest upstream external-kernel pattern on the real NPU:

- XRT reports `RyzenAI-npu4`, architecture `aie2p`, topology `6x8`.
- The valid target family on this machine is `npu2_4col`; lowering the official
  fixture as `npu1` returned all-zero output.
- External kernels should attach `link_with` and `llvm.emit_c_interface` to the
  private function declaration:

```mlir
func.func private @tile_kernel(memref<1024xi32, 2>)
    attributes {link_with = "tile_kernel.o", llvm.emit_c_interface}
```

Older herd/core-level `link_with` attachment is deprecated by AIRToAIE.

## Active Hardware Findings

Current verified Qwen3 path:

```bash
TOKEN_IDS=0,1,2,3 OUTPUT_ROWS=128 OUTPUT_TILE_ROWS=32 NPU_ITERATIONS=1 NPU_WARMUP=0 \
  scripts/run-quantized-qwen3-pipeline-npu.sh attention
```

Result:

```text
reference safetensors_pytorch_rocm AMD Radeon 890M
handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope->attention_core shared pyxrt BO
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
```

Key conclusions:

- Full four-token Qwen3 attention is feasible with multiple xclbins and shared
  `pyxrt.bo` handoff buffers.
- Standalone `attention_core` now uses the formal tiled online-softmax path.
  The exported MLIR keeps one `4x128` Q/O tile resident, streams K and V through
  one shared AIR FIFO channel, and preserves the public q/k/v/output ABI:

```bash
TOKEN_COUNT=8 QUERY_TILE_ROWS=4 KEY_TILE_ROWS=4 ATTENTION_RTOL=0.05 ATTENTION_ATOL=0.05 \
  scripts/run-quantized-qwen3-attention-npu.sh
```

```text
attention_core_max_abs 0.025021695
allclose True rtol=0.05 atol=0.05
mean_ms 2.699
```

- The old packed-KV attention experiment was removed after the formal
  `attention_core` path adopted the same tiling strategy. The production path
  keeps q/k/v as separate runtime arguments and uses a shared K/V channel only
  inside the generated AIR.

- Full eight-token Qwen3 attention currently fails before attention in
  `input_layernorm` channel lowering because that stage still maps token rows
  directly to an 8-row herd.
- `OUTPUT_TILE_ROWS=32` is the current practical point for a full 128-wide
  attention head. `OUTPUT_TILE_ROWS=16` creates too many shim channels. Q4_K 64
  rows reaches hardware but is slower. Q6_K 64 rows exceeds tile memory because
  the temporary widened `memref<64x424xi32>` is 108544 bytes.
- The Q6_K path still widens GGUF halfwords to i32 in L1. Compact i16 L1 DMA is
  the next cleanup for that kernel.
- Stitched `embed_tokens -> input_layernorm` AIR can be generated and lowered
  for inspection, but the runnable Python 3.12 path remains multiple xclbins
  with shared `pyxrt.bo` because the current `pyxrt` binding does not expose the
  upstream ELF loader.

## Retired Spikes

The old Spike 1-4 fixtures still document basic AIR behavior and may be useful
for debugging pass regressions:

- `scripts/verify-air-spike1.sh`: direct tiled matmul lowers to herd/DMA/AIE.
- `scripts/verify-air-spike2-memory.sh`: `memref<..., 1>` and `memref<..., 2>`
  cover the practical L2/L1 memory-space convention.
- `scripts/verify-air-spike3-dma-order.sh`: explicit copies lower to ordered
  DMA and channel operations.
- `scripts/verify-air-spike4-double-buffer.sh`: handwritten ping/pong buffers
  survive AIR lowering.

The old Q4_K matvec spike has been removed. It was a useful bridge from toy
matmul to packed GGUF data, but its separate `q4k_matvec.cc` and GGUF manifest
path are now lower value than the real `models.quantized_qwen3` projection
pipeline using `q4k_linear.cc` and `q6k_linear.cc`.

The aggregate `scripts/verify-air-spikes.sh` suite has also been removed. It hid
the distinction between retired learning fixtures and the current Qwen3 hardware
regression commands.

## Next Scaling Work

1. Apply the stage-local token loop pattern to `input_layernorm`, `rope_table`,
   and q/k norm+RoPE so the full pipeline can reach eight tokens.
2. Replace Q6_K i32-widened L1 tiles with compact i16/byte DMA and unpacking in
   the tile body.
3. Split attention into smaller context tiles with explicit synchronization
   before attempting 12+ tokens again.
4. Revisit stitched runtime only after the shared-BO path is stable for longer
   contexts.
