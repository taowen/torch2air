# torch2air

AIR backend experiments for PyTorch-exported models.

## AIR Tool Environment

Install the MLIR-AIR wheel toolchain with `uv` and Python 3.12:

```bash
scripts/install-air-tools.sh
```

The install script uses the local proxy by default:

```text
http_proxy=http://127.0.0.1:7890
https_proxy=http://127.0.0.1:7890
```

Disable that when needed:

```bash
USE_LOCAL_PROXY=0 scripts/install-air-tools.sh
```

Use the local AIR tools:

```bash
source scripts/air-env.sh
air-opt --version
```

or call the wrapper without sourcing:

```bash
scripts/air-opt.sh --version
```

## Spike Verification

Run all verified AIR/NPU spikes in order:

```bash
scripts/verify-air-spikes.sh
```

Run one spike directly:

```bash
scripts/verify-air-spike1.sh
```

or select a subset:

```bash
SPIKES="5 6" scripts/verify-air-spikes.sh
```

Generated IR is written under:

```text
examples/amd_aie_experiments/generated/
```

## NPU Smoke Test

The system XRT package may only ship `pyxrt` for the system Python. Rebuild the
binding for this repository's Python 3.12 environment:

```bash
scripts/build-pyxrt.sh
```

Then run a minimal AIE2P NPU hardware test:

```bash
scripts/run-npu-smoke.sh
```

Expected output includes:

```text
PASS!
```
