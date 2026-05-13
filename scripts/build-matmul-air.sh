#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT_DIR=${OUT_DIR:-"${TORCH2AIR_OUT_DIR}"}
FRONTEND=${FRONTEND:-"${TORCH2AIR_FRONTEND}"}
BUILD_VMFB=${BUILD_VMFB:-1}

"${PYTHON}" -m torch2air export-demo --out-dir "${OUT_DIR}" --frontend "${FRONTEND}"

lower_args=(
  lower-demo
  --out-dir "${OUT_DIR}"
  --iree-compile "${IREE_COMPILE}"
  --target-device "${TORCH2AIR_TARGET_DEVICE}"
  --tile-pipeline "${TORCH2AIR_TILE_PIPELINE}"
  --lower-to-aie-pipeline "${TORCH2AIR_LOWER_TO_AIE_PIPELINE}"
  --vmfb-tile-pipeline "${TORCH2AIR_VMFB_TILE_PIPELINE}"
  --vmfb-lower-to-aie-pipeline "${TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE}"
)
if [[ "${BUILD_VMFB}" == "0" ]]; then
  lower_args+=(--no-vmfb)
fi

"${PYTHON}" -m torch2air "${lower_args[@]}"
