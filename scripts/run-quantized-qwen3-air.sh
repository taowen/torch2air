#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT_DIR=${OUT_DIR:-"${TORCH2AIR_ROOT}/src/models/quantized_qwen3/generated"}
KERNEL=${KERNEL:-prefill_q_proj_tile_i32}
VMFB=${VMFB:-"${OUT_DIR}/${KERNEL}.vmfb"}

"${PYTHON}" -m models.quantized_qwen3.run \
  --kernel "${KERNEL}" \
  --out-dir "${OUT_DIR}" \
  --vmfb "${VMFB}" \
  --iree-run-module "${IREE_RUN_MODULE}" \
  --device xrt-lite \
  --xrt-lite-n-core-rows "${XRT_LITE_N_CORE_ROWS}" \
  --xrt-lite-n-core-cols "${XRT_LITE_N_CORE_COLS}"
