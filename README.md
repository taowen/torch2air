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

On the current `iree-amd-aie` checkout you can also reuse the existing
`~/projects/torch2vk/.venv` for PyTorch and call the checked-in scripts directly.

## Export A Demo Model

From the parent `iree-amd-aie` checkout:

```bash
source iree-install-rocm/env.sh
PYTHON=/var/home/taowen/projects/torch2vk/.venv/bin/python \
  third_party/torch2air/scripts/export-matmul-air.sh
```

This produces:

- `examples/generated/matmul_i32_32.linalg.mlir`
- `examples/generated/matmul_i32_32.air.mlir`
- `examples/generated/matmul_i32_32.vmfb`

Only the MLIR files are intended to be committed. VMFB files are build outputs.

To execute the generated VMFB on `xrt-lite`:

```bash
source iree-install-rocm/env.sh
PYTHON=/var/home/taowen/projects/torch2vk/.venv/bin/python \
  third_party/torch2air/scripts/run-matmul-air.sh
```

On the current `iree-amd-aie` checkout this reaches the NPU runtime, but the
AIR/i32 path returns zeros. The script treats that as a verification failure
instead of silently accepting it.

## CLI

```bash
python -m torch2air export-demo --out-dir examples/generated
python -m torch2air lower-air examples/generated/matmul_i32_32.linalg.mlir \
  --air-mlir examples/generated/matmul_i32_32.air.mlir \
  --vmfb examples/generated/matmul_i32_32.vmfb
python -m torch2air run-demo --out-dir examples/generated
```

`--frontend=auto` tries `torch-mlir` first if it is installed. The fallback FX
path keeps this repository usable without a matching torch-mlir wheel.
