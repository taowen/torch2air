# torch2air Architecture

`torch2air` is planned as a standalone project rooted at this repository. It
uses PyTorch model structure and weight metadata as the frontend, and emits
MLIR-AIR programs where the important tiling and data movement schedule is
explicit in the IR.

The project should not depend on the `iree-amd-aie` repository as its root. A
future `mlir-air/` directory will be a git submodule, and the build/runtime
scripts should use MLIR-AIR tools directly.

## Goals

- Use real PyTorch `nn.Module` objects to discover model structure, names,
  shapes, and call boundaries.
- Use real on-disk quantized weights, especially GGUF Q4_K-style packed
  weights, without host-side dequantization for NPU inputs.
- Generate MLIR-AIR-oriented IR where tile loops, local buffers, and DMA copies
  are written directly.
- Keep kernel-specific performance decisions in `torch2air`, not hidden behind
  a generic graph compiler.
- Make the generated AIR artifacts inspectable and reproducible.

## Non-Goals

- `torch2air` is not an arbitrary PyTorch graph optimizer.
- It should not rely on IREE `vmfb`, IREE HAL dispatch formation, or IREE
  encoding dialect as the core execution path.
- It should not assume PyTorch export can represent quantized packed-weight
  kernels exactly. PyTorch export is used for model structure; backend kernels
  can be custom generated.

## Repository Layout

Planned layout:

```text
torch2air/
  mlir-air/                         # git submodule: Xilinx/mlir-air
  docs/
    architecture.md
  src/torch2air/
    frontend/                       # PyTorch export, module traversal, shape capture
    weights/                        # GGUF and other weight readers
    ir/                             # MLIR text/builders for memref/AIR IR
    kernels/                        # kernel templates and lowering rules
    runtime/                        # XRT/AIR runtime wrappers
  models/
    quantized_qwen3/
      export.py
      kernels/
      generated/
  scripts/
    build-air-tools.sh
    export-quantized-qwen3.sh
    run-quantized-qwen3.sh
```

Only the architecture document exists in the reset repository right now. The
layout above is the intended direction.

## Frontend Model Flow

The PyTorch side should look similar to a normal model export flow:

1. Load or construct a real `nn.Module`.
2. Run `torch.export` or FX tracing with meta tensors for shapes.
3. Preserve module path names, parameter names, and call boundaries.
4. Match known subgraphs or modules to `torch2air` kernel rules.
5. Attach disk weight references such as GGUF tensor names and quantization
   types.

For quantized kernels, the PyTorch graph is not the exact arithmetic source. For
example, a PyTorch `Linear` may conceptually be `x @ W`, but the actual AIR
kernel can consume packed Q4_K bytes and decode on device. The frontend exports
the structural intent; the kernel rule owns the NPU implementation.

## IR Layers

`torch2air` should use four IR layers.

### 1. Model Description

This is a lightweight metadata layer derived from PyTorch:

```text
module path: model.layers.0.self_attn.q_proj
logical op: q_proj
activation shape: tensor<8x1024xf16>
weight source: GGUF tensor model.layers.0.self_attn.q_proj.weight
weight format: Q4_K
output shape: tensor<8x2048xf16>
```

This layer is stable and easy to diff. It should be serializable as JSON or a
small Python object graph.

### 2. Logical Linalg/Marker IR

This optional layer describes the mathematical contract:

```mlir
%out = linalg.generic
    {torch2air.kernel = "q4k_f16_matmul"}
    ins(%activation, %packed_weight : tensor<8x1024xf16>, tensor<2048x1024xi8>)
    outs(%init : tensor<8x2048xf16>) {
  ...
} -> tensor<8x2048xf16>
```

This is useful for provenance, testing, and shape checking. It is not required
to be executable by itself.

### 3. Tiled Memref IR

This is the main source layer for MLIR-AIR. Tiling is explicit in the IR:

```mlir
scf.parallel (%oc) = (%c0) to (%c2048) step (%c8) {
  %w_global = memref.subview %packed_weight[%oc, 0] [8, 1024] [1, 1]
      : memref<2048x1024xi8> to memref<8x1024xi8, strided<[1024, 1], offset: ?>>
  %o_global = memref.subview %output[0, %oc] [8, 8] [1, 1]
      : memref<8x2048xf16> to memref<8x8xf16, strided<[2048, 1], offset: ?>>

  %x_l1 = memref.alloc() : memref<8x1024xf16, 2 : i32>
  %w_l1 = memref.alloc() : memref<8x1024xi8, 2 : i32>
  %o_l1 = memref.alloc() : memref<8x8xf16, 2 : i32>

  memref.copy %activation, %x_l1
      : memref<8x1024xf16> to memref<8x1024xf16, 2 : i32>
  memref.copy %w_global, %w_l1
      : memref<8x1024xi8, strided<[1024, 1], offset: ?>>
        to memref<8x1024xi8, 2 : i32>

  call @q4k_f16_8x1024x8(%x_l1, %w_l1, %o_l1)
      : (memref<8x1024xf16, 2 : i32>,
         memref<8x1024xi8, 2 : i32>,
         memref<8x8xf16, 2 : i32>) -> ()

  memref.copy %o_l1, %o_global
      : memref<8x8xf16, 2 : i32>
        to memref<8x8xf16, strided<[2048, 1], offset: ?>>
}
```

This layer is where `torch2air` directly controls:

- output-column tile size
- activation and weight tile shape
- L1/L2/global memory placement
- copy order
- whether the compute body is a `linalg` op or an external AIE kernel call
- future double buffering and DMA/compute overlap

### 4. AIR/AIE IR

MLIR-AIR tools lower the tiled memref IR:

```text
air-par-to-herd
air-copy-to-dma
air-dependency
air-dma-to-channel
air-place-herds
air-to-aie
air-to-std
airrt-to-npu
```

Expected intermediate artifacts:

- `.tiled.mlir`: hand-scheduled memref/scf IR
- `.air.mlir`: AIR hierarchy and `air.dma_memcpy_nd`
- `.aie.mlir`: `aie.device`, `aie.core`, locks, buffers, DMA BDs
- runtime artifacts produced by `aircc.py`

## Why Tiling Is Written Directly

`air-par-to-herd` maps parallel loops to `air.herd`, but it does not invent an
L1 tiling strategy by itself. To get DMA and local buffers, the input IR must
already contain:

- tile loops
- `memref.subview`
- `memref.alloc` in the desired memory space
- `memref.copy`

Then `air-copy-to-dma` can turn the copies into `air.dma_memcpy_nd`.

MLIR-AIR also has `air-linalg-codegen`, which can generate this style of tiled
IR for some Linalg programs. That pass should be treated as an optional helper,
not as the core design. For Q4_K, FlashAttention, and other kernels where
execution order is the performance contract, `torch2air` should generate the
tiled memref IR explicitly.

## Quantized Qwen3 Direction

The first model target should be one Qwen3 layer, starting with Q projection:

```text
activation:      memref<8x1024xf16>
packed weight:   memref<2048x1024xi8>   # rows contain exact packed GGUF bytes
output:          memref<8x2048xf16>
hardware unit:   q4k_f16_8x1024x8
```

The generated program should tile output columns by 8:

```text
for oc in range(0, 2048, 8):
  DMA activation tile to L1 or reuse resident activation tile
  DMA packed_weight[oc:oc+8, :] to L1
  run q4k_f16_8x1024x8 on AIE core
  DMA output[:, oc:oc+8] back to global
```

The NPU input is packed bytes. The host may package buffers and compare results
against PyTorch ROCm, but it must not dequantize weights for the NPU path.

## Compute Kernel Choices

The compute portion inside each tile can be represented in three ways:

1. `linalg.matmul` or `linalg.generic` on L1 memrefs for simple integer/floating
   kernels.
2. `air-linalg-to-func{link-with=...}` for external AIE core kernels.
3. Lower-level AIE/Peano code for kernels that need exact instruction-level
   behavior.

For Q4_K, the likely route is an external AIE core kernel because packed nibble
decode, scale/min handling, and accumulation order matter.

## Build Strategy

Future bootstrap:

```bash
git submodule add https://github.com/Xilinx/mlir-air.git mlir-air
```

Build scripts should produce:

- `air-opt`
- `air-translate`
- `aircc.py`
- Python `air` package
- required MLIR-AIE/llvm-aie/Peano pieces

The project should expose commands like:

```bash
scripts/export-quantized-qwen3.sh
scripts/compile-air.sh models/quantized_qwen3/generated/q_proj.tiled.mlir
scripts/run-quantized-qwen3.sh
```

## Open Problems

- Decide whether `air-linalg-codegen` is built and used as a helper for simple
  kernels.
- Fix or avoid AIRRt/NPU lowering issues from generated affine forms.
- Define the exact AIR runtime ABI for model execution.
- Decide how to package external AIE core kernels and link them through
  `air-linalg-to-func` or `aircc.py`.
- Add a minimal Q4_K end-to-end path with one tiled Q projection.
