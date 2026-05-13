# torch2air

Export small PyTorch models to MLIR and lower them through the IREE AMD-AIE
AIR pipeline. The generated `*.air.mlir` files play the same role that generated
GLSL shader files play in `torch2vk`: they are target-facing compiler artifacts
checked in next to the exported model.

This is intentionally a narrow first slice. It handles matmul-like PyTorch
graphs and uses IREE's `--iree-amdaie-lower-to-aie-pipeline=air` path to produce
AIR IR and an executable VMFB.

## Install

The plain PyPI `torch-mlir` package is old (`20221213.686`). Use the LLVM
snapshot wheel index when you want the torch-mlir frontend:

```bash
uv venv --python 3.12
uv pip install -e .
uv pip install --pre torch-mlir \
  -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

On the current `iree-amd-aie` checkout, use the parent `.venv` as the shared
Python environment for PyTorch, IREE Python bindings, and torch2air.

Useful environment helpers:

```bash
source scripts/env.sh
scripts/bootstrap-python.sh
scripts/doctor.sh
```

## Export A Demo Model

From the parent `iree-amd-aie` checkout:

```bash
third_party/torch2air/scripts/build-matmul-air.sh
```

This produces:

- `examples/generated/matmul_i32_32.linalg.mlir`
- `examples/generated/matmul_i32_32.air.mlir`
- `examples/generated/matmul_i32_32.vmfb`

Only the MLIR files are intended to be committed. VMFB files are build outputs.
On Strix/NPU4, the AIR artifact is generated with the AIR path, while the
runnable VMFB is built with `objectFifo` by default because the current
AIR-to-AIE placement path fails for this small i32 demo.

To execute the generated VMFB on `xrt-lite`:

```bash
third_party/torch2air/scripts/run-matmul-air.sh
```

The default target is Strix/Ryzen AI NPU4: `TORCH2AIR_TARGET_DEVICE=npu4` with
`XRT_LITE_N_CORE_ROWS=4` and `XRT_LITE_N_CORE_COLS=8`. Override those variables
for Phoenix/NPU1 devices.

To compile and verify the same linalg MLIR through the ROCm backend:

```bash
third_party/torch2air/scripts/run-matmul-rocm.sh
```

## Quantized Qwen3 Kernels

The first model-specific scaffold mirrors `torch2vk`'s model layout under
`src/models/quantized_qwen3`. It starts with one exported Qwen3-0.6B projection
tile:

```bash
third_party/torch2air/scripts/build-quantized-qwen3-air.sh
third_party/torch2air/scripts/run-quantized-qwen3-air.sh
```

The exporter and verifier use the same PyTorch module as the source of truth.
`run-quantized-qwen3-air.sh` is a development smoke test: it uses
`iree-run-module` and temporary `.npy` files only at the command-line tensor I/O
boundary, then compares the IREE output against the PyTorch reference with torch
operations. A deployment path should load the VMFB through the IREE runtime API
and pass application-owned buffers directly, without `iree-run-module` or
`.npy` files.

## CLI

```bash
$PYTHON -m torch2air export-demo --out-dir examples/generated
$PYTHON -m torch2air lower-air examples/generated/matmul_i32_32.linalg.mlir \
  --air-mlir examples/generated/matmul_i32_32.air.mlir \
  --vmfb examples/generated/matmul_i32_32.vmfb
$PYTHON -m torch2air run-demo --out-dir examples/generated
$PYTHON -m torch2air run-demo --out-dir examples/generated \
  --vmfb examples/generated/matmul_i32_32.rocm.vmfb --device hip
```

`--frontend=auto` tries `torch-mlir` first if it is installed. The fallback FX
path keeps this repository usable without a matching torch-mlir wheel.
