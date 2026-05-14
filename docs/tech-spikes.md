# Technical Spikes

This document tracks the experiments needed to validate the `torch2air`
direction from [architecture.md](architecture.md):

```text
PyTorch export -> direct graph walk -> Jinja tiled MLIR -> air-opt/aircc -> XRT
```

The central hypothesis is that the tiling strategy is hand-written by
`torch2air` as pre-AIR MLIR, then lowered by MLIR-AIR. AIR is not expected to
auto-discover the Qwen3 schedule from the PyTorch graph. `torch.export`
provides operator and shape facts; the stage template emits MLIR that chooses
tile sizes, memory spaces, copy order, and tile-local work explicitly. The tile
schedule is part of the generated MLIR:

- `scf.parallel` / `scf.for` express the tile loop nest.
- `memref.subview` expresses a global tile slice.
- `memref.alloc(..., 1/2)` expresses L2/L1 local buffers.
- `memref.copy` expresses data movement.
- `air-copy-to-dma` converts copies to `air.dma_memcpy_nd`.
- `air-par-to-herd` maps parallel tiles to `air.herd`.
- `air-dma-to-channel` splits DMA into `air.channel.put/get`.

That is a good fit for Q4_K and FlashAttention-style kernels where tile size,
memory space, DMA order, and double buffering are performance-critical.

Do not use `.cc` kernels to express tiling. A native kernel is acceptable only
as the compute body inside a selected tile/herd, with the generated MLIR still
owning launch/segment/herd placement, DMA, memory spaces, and operator
boundaries. Upstream MLIR-AIR often uses the Python AIR DSL to build that IR;
`torch2air` emits the MLIR text directly so the export artifact is easy to
inspect and keep close to the `torch.export` walk.

## Official External-Kernel Check

On 2026-05-14, `scripts/run-mlir-air-official-external-kernel-spike.sh`
validated the smallest upstream external-kernel pattern on the local real NPU:

- XRT reports `RyzenAI-npu4`, architecture `aie2p`, topology `6x8`.
- MLIR-AIR treats `npu4`/Strix hardware as the `npu2` target family.
- The upstream `test/xrt/05_extern_func` flow passes with
  `air-to-aie="device=npu2_4col row-offset=2 col-offset=0"` and Peano target
  `aie2p-none-unknown-elf`.
- The same fixture returned all zeros when accidentally lowered as `device=npu1`,
  so `npu1` is not a valid default on this machine.
- The current recommended external-kernel attachment is:

```mlir
func.func private @tile_kernel(memref<1024xi32, 2>)
    attributes {link_with = "tile_kernel.o", llvm.emit_c_interface}
```

Attaching `link_with` only to `air.herd` still appears in older upstream tests,
but AIRToAIE now warns that the herd/core attachment is deprecated. The
torch2air convention is therefore: put `link_with` and `llvm.emit_c_interface`
on every external `func.func private` declaration, and keep AIR placement/DMA in
the generated MLIR.

For operator-to-operator experiments, follow the upstream MLIR-AIR llama style:
build each operator as an independent stage first, then stitch or sequence those
stages through AIR/XRT. Full-hidden `embed_tokens -> input_layernorm` now
generates a stitched two-launch AIR artifact, but the runnable Python 3.12 path
uses two stage xclbins with a shared `pyxrt.bo` because the upstream ELF loader
is not exposed by the current `pyxrt` binding and the xclbin multi-launch path
overflows program memory for the current Q4_K embedding body.
The next verified steps, `embed_tokens -> input_layernorm -> q_proj` and
`embed_tokens -> input_layernorm -> q/k/v`, keep the same shared-BO style and
add official AIR external kernels for Q4_K and Q6_K linear tile bodies.
The latest q/k/v verification covers one full Qwen3 attention head
(`OUTPUT_ROWS=128`) on real NPU hardware using 32 output rows per external
kernel tile. The current RoPE step adds `rope_table` plus `q/k_norm_rope` in
the same style: generated MLIR owns launch/herd/DMA/L1 placement, external
objects provide only the tile compute body, and runtime handoff stays in shared
`pyxrt.bo` buffers.

## Baseline IR Shape

The minimum shape to validate is:

```mlir
scf.parallel (%i, %j) = (%c0, %c0) to (%M, %N) step (%tm, %tn) {
  %a_tile = memref.subview %A[%i, 0] [%tm, %K] [1, 1]
      : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
  %b_tile = memref.subview %B[0, %j] [%K, %tn] [1, 1]
      : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
  %c_tile = memref.subview %C[%i, %j] [%tm, %tn] [1, 1]
      : memref<?x?xi32> to memref<?x?xi32, strided<[?, 1], offset: ?>>

  %a_l1 = memref.alloc() : memref<8x16xi32, 2 : i32>
  %b_l1 = memref.alloc() : memref<16x8xi32, 2 : i32>
  %c_l1 = memref.alloc() : memref<8x8xi32, 2 : i32>

  memref.copy %a_tile, %a_l1
      : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<8x16xi32, 2 : i32>
  memref.copy %b_tile, %b_l1
      : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<16x8xi32, 2 : i32>
  memref.copy %c_tile, %c_l1
      : memref<?x?xi32, strided<[?, 1], offset: ?>> to memref<8x8xi32, 2 : i32>

  linalg.matmul
      ins(%a_l1, %b_l1 : memref<8x16xi32, 2 : i32>, memref<16x8xi32, 2 : i32>)
      outs(%c_l1 : memref<8x8xi32, 2 : i32>)

  memref.copy %c_l1, %c_tile
      : memref<8x8xi32, 2 : i32> to memref<?x?xi32, strided<[?, 1], offset: ?>>
}
```

The required pass sequence is:

```bash
air-opt input.mlir \
  --air-par-to-herd \
  --air-copy-to-dma \
  --air-dependency \
  --air-dma-to-channel \
  --air-place-herds \
  --air-to-aie
```

The prior local experiment
`examples/amd_aie_experiments/air_direct_matmul_tiled.mlir` used this pattern
and reached `air.herd`, `air.dma_memcpy_nd`, then `aie.device(npu4)` /
`aie.core`. Since the repository has been reset, the first spike is to restore
that experiment as a permanent fixture.

## Spike 1: Restore Direct Tiled Matmul

Goal: prove that hand-written tile loops and local buffers are enough for
MLIR-AIR to build the spatial hierarchy.

Input:

- `examples/amd_aie_experiments/air_direct_matmul_tiled.mlir`
- Static sizes first: `M=8`, `N=8`, `K=16`
- `scf.parallel` over output tiles
- `memref.subview`, `memref.alloc`, `memref.copy`
- `linalg.matmul` on L1 memrefs

Run:

```bash
air-opt examples/amd_aie_experiments/air_direct_matmul_tiled.mlir \
  --air-par-to-herd \
  --air-copy-to-dma \
  --air-dependency \
  --air-dma-to-channel \
  --air-place-herds \
  --air-to-aie \
  -o examples/amd_aie_experiments/generated/air_direct_matmul_tiled.aie.mlir
```

Success criteria:

- Generated IR contains `air.herd`.
- Generated IR contains `air.dma_memcpy_nd`.
- Generated IR contains `air.channel.put` / `air.channel.get` after channel
  lowering.
- Final IR contains `aie.device(npu2_4col)` on the local `RyzenAI-npu4` machine
  and at least one `aie.core`.
- The file is small enough to be used as a regression test.

Decision after spike:

- If this works, model-stage templates can directly emit tiled MLIR source.
- If this fails, inspect which exact op shape the AIR passes reject and narrow
  the template language to that accepted subset.

## Spike 2: Memory Space Contract

Goal: determine the practical meaning of AIR/AIE memory spaces for generated
templates.

Questions:

- Which memory space value should represent L2?
- Which memory space value should represent L1/core-local memory?
- Does `memref.alloc() : memref<..., 2 : i32>` reliably lower to the desired
  local buffer placement on `npu4`?
- Are dynamic tile sizes accepted, or do templates need static tile memrefs?

Input variants:

- Global -> L1 -> global copies
- Global -> L2 -> L1 -> L2 -> global copies
- Static memrefs only
- One dynamic dimension at a time

Success criteria:

- Document a stable memory-space convention for templates.
- Identify the smallest accepted type syntax for `memref.alloc`.
- Add comments in the example MLIR explaining each memory space.

Decision after spike:

- Encode the memory-space convention in `AirBuffer` / `TileConfig`.
- Keep early Q4_K templates static if dynamic memrefs create unstable lowering.

## Spike 3: DMA Ordering And Channels

Goal: verify that multiple copies inside one tile lower into ordered DMA and
channel operations.

Input:

- One tile with three input copies: activation, weight, optional bias/scale
- One output copy
- Explicit compute op between input and output copies

Run passes:

```text
air-copy-to-dma
air-dependency
air-dma-to-channel
```

Success criteria:

- Input DMAs are visible as separate `air.dma_memcpy_nd` operations before
  channel lowering.
- Channelized IR has matching `air.channel.put/get` pairs.
- The output copy is ordered after compute.
- Dependencies do not serialize independent input copies unless required.

Decision after spike:

- If dependency inference is too conservative, templates should emit more
  explicit ordering structure.
- If dependency inference is too weak, templates should add explicit waits or
  split loop bodies.

## Spike 4: Double Buffering Skeleton

Goal: prove that AIR can express DMA/compute overlap when we write the schedule
explicitly.

Input:

- `scf.for` over K tiles inside one output tile.
- Two activation buffers and two weight buffers.
- Prologue copies tile 0.
- Main loop computes tile `k` while copying tile `k + 1`.
- Epilogue computes the final tile and copies the output back.

Success criteria:

- AIR IR preserves two local buffer sets.
- DMA for the next K tile is not forced after current compute.
- Channel/dependency IR exposes a pipeline structure that can become overlap on
  hardware.

Decision after spike:

- If this is expressible and accepted by `air-to-aie`, Q4_K linear should use
  hand-written double-buffered templates.
- If not, start with single-buffer templates and isolate double buffering as a
  later backend-specific path.

## Spike 5: Q4_K Linear Tile Skeleton

Goal: replace the toy matmul with the first quantized Qwen3-relevant kernel
shape.

Input:

- Activation: f16 tile, for example `[sequence_tile, k_tile]`
- Weight: packed Q4_K_M bytes from GGUF, sliced by output columns and K block
- Output: f16 tile
- Compute body: initially an external placeholder call or `linalg.generic`
  stub, then a real AIE/Peano kernel

Success criteria:

- The AIR template consumes packed bytes, not host-dequantized f16/f32 weights.
- Tiling chooses output-column tiles explicitly.
- The generated IR reaches `aie.device(npu4)`.
- The run log records exact GGUF tensor names and packed formats.

Decision after spike:

- If external core kernels are easiest, stage templates should emit the needed
  the generated MLIR should make that boundary explicit, and the lowering path
  must still start from a tiled `.mlir` artifact.
- If `linalg.generic` is enough for a first implementation, keep the external
  path optional.

## Spike 6: FlashAttention Schedule Skeleton

Goal: verify that AIR templates can express a staged attention schedule where
execution order is the kernel.

Input:

- Q tile resident on local memory.
- Loop over K/V tiles.
- Copy K/V tile, compute scores, update softmax state, accumulate output.
- Use placeholder compute bodies first.

Success criteria:

- The IR can represent the intended stage order without relying on a generic
  optimizer.
- K/V DMA and compute can be separated enough to add double buffering later.
- The generated AIR structure is inspectable and not dominated by compiler
  rewrites that hide the schedule.

Decision after spike:

- If the skeleton lowers cleanly, FlashAttention should be a dedicated model
  stage template, not a generic registry entry.
- If it does not, keep FlashAttention outside the initial Qwen3 milestone.

## Spike 7: Torch2VK-Style Direct Export

Goal: connect the `torch2vk`-style export path to tiled MLIR templates.

Input:

- A small real `nn.Module` exported with `torch.export`.
- A local `export_one(...)` function in the model package.
- Direct iteration over `program.graph_module.graph.nodes`.
- Template variables from node metadata and GGUF weight metadata.

Success criteria:

- Export writes only the stage artifact consumed by the next command.
- No `graph.py`, normalized `graph.json`, registry, variant object, or export
  manifest is introduced.
- The generated tiled MLIR file is accepted by the Spike 1 pass pipeline.
- The generated file contains explicit tile structure: `scf.parallel`,
  `memref.subview`, L1 `memref.alloc`, and `memref.copy`.

Decision after spike:

- If this is straightforward, implement `models/quantized_qwen3/export.py`
  using the same direct `export_one(...)` calling style as `torch2vk`.
- If graph shapes are hard to recover from `torch.export`, add only the minimal
  shape annotation layer needed by template rendering.

## Spike 8: Runtime Boundary And Reference

Goal: prove the generated tiled MLIR artifact can be loaded and compared at the module
boundary.

Input:

- One generated `models.quantized_qwen3` MLIR artifact from Spike 7.
- Real packed GGUF weight buffer.
- Reference output for the same module boundary.

Rules:

- `.npy` is only an optional file container for tool compatibility, not the
  deployment ABI.
- The AIR path must consume packed weights.
- Prefer the generated PyTorch ROCm reference when the module boundary has a
  normal PyTorch equivalent. For `quantized_qwen3`, `export.py` regenerates
  `reference.py` from those exported boundaries and loads the local safetensors
  model. NumPy is not a default model-reference path.

Success criteria:

- A host program loads the AIR/XRT artifact and input/output buffers.
- The reference path produces expected output.
- The comparison runs at the exported module boundary with documented tolerance.
- The script runs on the real NPU, not only through `air-opt`.

Decision after spike:

- If Python AIR runtime bindings are unstable, use a small C++ host runner.
- Keep `iree-run-module` and IREE `vmfb` out of the main path.

Current `quantized_qwen3.embed_tokens` command:

```bash
AIR_DEVICE=npu2_4col TOKEN_IDS=0,1 BLOCKS_PER_ROW=4 \
  scripts/run-quantized-qwen3-npu.sh embed_tokens
```

This command exports from `torch.export`, lowers tiled pre-AIR MLIR through
`air-opt`, compiles the runtime MLIR with `aiecc`, loads `xclbin`/`insts` with
`air.backend.xrt.XRTBackend`, and compares the real NPU output with the
generated safetensors PyTorch ROCm reference.

## Milestone Order

1. Restore and commit the direct tiled matmul example.
2. Freeze the memory-space and pass-pipeline contract.
3. Generate one stage artifact from one real exported PyTorch module.
4. Replace the toy linear with packed Q4_K_M weight handling.
5. Add runtime execution and reference comparison on real NPU.
6. Only then attempt FlashAttention scheduling.

The expected conclusion is not that MLIR-AIR auto-discovers the right schedule.
The useful conclusion is that `torch2air` can write the schedule directly in IR
and use MLIR-AIR to lower that schedule to herd, DMA, channel, and AIE forms.

## Handoff Rule

When adding the next Qwen3 step, first make the stage boundary direct: memref in,
memref out, no graph JSON, manifest, `.npy`, or host repacking between
operators. Then look for adjacent operators that can be fused so the
intermediate stays in L1/L2.

Current status:

- Full `input_layernorm` has been migrated and runs on real NPU for hidden size
  1024.
- A fused `embed_tokens_input_layernorm` L1 handoff runs for one Q4_K block.
- Full hidden fusion is a separate scheduling problem because RMSNorm needs a
  reduction across all Q4_K blocks; the naive single-core full fusion is not yet
  a valid replacement.
