#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT_DIR=${OUT_DIR:-"${TORCH2AIR_ROOT}/src/models/quantized_qwen3/generated"}
KERNEL=${KERNEL:-all}
BUILD_VMFB=${BUILD_VMFB:-1}

args=(
  -m models.quantized_qwen3.export
  --kernel "${KERNEL}"
  --out-dir "${OUT_DIR}"
  --iree-compile "${IREE_COMPILE}"
  --target-device "${TORCH2AIR_TARGET_DEVICE}"
  --air-tile-pipeline "${TORCH2AIR_TILE_PIPELINE}"
  --air-lower-to-aie-pipeline "${TORCH2AIR_LOWER_TO_AIE_PIPELINE}"
  --vmfb-tile-pipeline "${TORCH2AIR_VMFB_TILE_PIPELINE}"
  --vmfb-lower-to-aie-pipeline "${TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE}"
)

if [[ "${BUILD_VMFB}" == "0" ]]; then
  args+=(--no-vmfb)
fi

"${PYTHON}" "${args[@]}"
