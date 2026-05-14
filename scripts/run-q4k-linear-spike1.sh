#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"

source "$ROOT_DIR/scripts/npu-common.sh"

WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/q4k-linear-spike1}"

check_npu_device
"$UV" run --no-sync python -m models.quantized_qwen3.run_q4k_linear_spike1 \
  --work-dir "$WORK_DIR" \
  "$@"
