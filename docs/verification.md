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
- `air-to-aie` produces `aie.device(npu1)` and `aie.core`.

Notes:

- The current toolchain rejects `device=npu4` with `Invalid aie.device option`.
  The verification script defaults to `AIR_DEVICE=npu1`; override it with
  `AIR_DEVICE=npu2` or another supported AIR/AIE device name when needed.
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
for `AirVariant` templates to control DMA order.

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

## Spike 5: Real Q4_K Linear Tile

Run:

```bash
scripts/verify-air-spike5-q4k-linear.sh
```

Inputs:

```text
/var/home/taowen/projects/torch2vk/dist/llama_cpp_qwen3/qwen3-0.6b-q4_k_m.gguf
tensor: token_embd.weight
format: GGUF Q4_K
physical shape: [151936, 144] uint32
```

Verified checks:

- The AIR fixture consumes packed Q4_K-shaped `uint32` tiles.
- The host manifest records the exact GGUF tensor metadata.
- The NPU path compiles `examples/amd_aie_experiments/q4k_matvec.cc` with
  Peano and runs `examples/amd_aie_experiments/npu_q4k_matvec.py`.
- The NPU kernel reads real packed Q4_K blocks from GGUF, dequantizes them on
  the AIE tile, and computes a float32 matvec.
- Current default validation uses 64 rows, `k=1024`, and 4 Q4_K blocks per
  row. Payload SHA256:
  `e6d93e48b795cf0ed2844feac19a89ea92708904889c05901dae1da6aec4d56a`.
- The NPU result matches the host Q4_K dequantization reference and reports
  `PASS!`.

Conclusion: packed Q4_K weights do not need to be host-dequantized for this
path. `AirVariant` needs an external-kernel/link field for practical quantized
core kernels.

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

## Full Spike Suite

Run all verified spikes in order:

```bash
scripts/verify-air-spikes.sh
```

Select a subset with:

```bash
SPIKES="5 6" scripts/verify-air-spikes.sh
```

Latest full run on 2026-05-13:

```text
All requested AIR/NPU spikes passed: 1 2 3 4 5 6
```

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
