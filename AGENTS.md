# Agent Notes

Keep this repository self-contained. Do not commit changes in the parent
`iree-amd-aie` checkout unless the user explicitly asks for that.

## Environment

Use the shared environment script before running manual commands:

```bash
source scripts/env.sh
```

It discovers the parent `iree-amd-aie` checkout, sources
`iree-install/env.sh` when available, picks the parent
`iree-amd-aie/.venv/bin/python`, and adds `src/` to `PYTHONPATH`.

For a local Python environment without RPM packages:

```bash
scripts/bootstrap-python.sh
INSTALL_TORCH_MLIR=1 scripts/bootstrap-python.sh
```

`scripts/bootstrap-python.sh` creates or refreshes the parent
`iree-amd-aie/.venv`; do not create a separate `torch2air/.venv`.

Run a quick environment check with:

```bash
scripts/doctor.sh
```

## Common Commands

Export the demo PyTorch matmul, lower it through the AMD-AIE AIR path, and build
the NPU VMFB:

```bash
scripts/build-matmul-air.sh
```

On Strix/NPU4, `scripts/build-matmul-air.sh` keeps the committed AIR artifact
from `TORCH2AIR_LOWER_TO_AIE_PIPELINE=air`, but builds the runnable VMFB with
`TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE=objectFifo` by default. The AIR-to-AIE
placement path in this checkout currently fails for the small i32 demo on NPU4.

Compatibility entry point with the old name:

```bash
scripts/export-matmul-air.sh
```

Run the generated NPU VMFB on `xrt-lite`:

```bash
scripts/run-matmul-air.sh
ALLOW_MISMATCH=1 scripts/run-matmul-air.sh
TORCH2AIR_TARGET_DEVICE=npu1_4col XRT_LITE_N_CORE_ROWS=4 XRT_LITE_N_CORE_COLS=4 scripts/run-matmul-air.sh
```

The default target is Strix/Ryzen AI NPU4 (`TORCH2AIR_TARGET_DEVICE=npu4`,
`XRT_LITE_N_CORE_ROWS=4`, `XRT_LITE_N_CORE_COLS=8`). Override those variables
when running on Phoenix/NPU1.

Compile and run the same linalg MLIR through ROCm:

```bash
scripts/run-matmul-rocm.sh
ROCM_TARGET=gfx1150 scripts/run-matmul-rocm.sh
```

Build and run the first quantized Qwen3 AIR kernel tile:

```bash
scripts/build-quantized-qwen3-air.sh
scripts/run-quantized-qwen3-air.sh
```

The Qwen3 runner uses PyTorch as the reference implementation. It is a
development smoke test: it loads the VMFB through the IREE Python runtime, runs
it on `xrt-lite`, and compares the output against the PyTorch ROCm reference
with torch operations.

Useful direct CLI commands:

```bash
$PYTHON -m torch2air export-demo --out-dir examples/generated --frontend fx
$PYTHON -m torch2air lower-demo --out-dir examples/generated
$PYTHON -m torch2air run-demo --out-dir examples/generated --device xrt-lite
$PYTHON -m torch2air run-demo --out-dir examples/generated \
  --vmfb examples/generated/matmul_i32_32.rocm.vmfb --device hip
```

Inspect the generated AIR IR:

```bash
rg "air\\.(launch|herd|segment|channel|dma)" examples/generated/matmul_i32_32.air.mlir
```

Generated `*.vmfb` and `*.npy` files are build outputs and should stay
uncommitted. The generated `*.linalg.mlir` and `*.air.mlir` files are source
artifacts for examples and may be committed when intentionally refreshed.
