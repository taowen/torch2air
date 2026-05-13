#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT_DIR=${OUT_DIR:-"${TORCH2AIR_OUT_DIR}"}
VMFB=${VMFB:-"${OUT_DIR}/matmul_i32_32.vmfb"}
ALLOW_MISMATCH=${ALLOW_MISMATCH:-0}

args=(
  run-demo
  --out-dir "${OUT_DIR}"
  --vmfb "${VMFB}"
  --iree-run-module "${IREE_RUN_MODULE}"
  --device xrt-lite
  --xrt-lite-n-core-rows "${XRT_LITE_N_CORE_ROWS}"
  --xrt-lite-n-core-cols "${XRT_LITE_N_CORE_COLS}"
)

if [[ "${ALLOW_MISMATCH}" == "1" ]]; then
  args+=(--allow-mismatch)
fi

"${PYTHON}" -m torch2air "${args[@]}"
