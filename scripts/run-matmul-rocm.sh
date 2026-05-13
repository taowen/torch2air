#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT_DIR=${OUT_DIR:-"${TORCH2AIR_OUT_DIR}"}
FRONTEND=${FRONTEND:-"${TORCH2AIR_FRONTEND}"}
LINALG_MLIR="${OUT_DIR}/matmul_i32_32.linalg.mlir"
ROCM_VMFB=${ROCM_VMFB:-"${OUT_DIR}/matmul_i32_32.rocm.vmfb"}

if [[ ! -f "${LINALG_MLIR}" ]]; then
  "${PYTHON}" -m torch2air export-demo --out-dir "${OUT_DIR}" --frontend "${FRONTEND}"
fi

"${IREE_COMPILE}" "${LINALG_MLIR}" \
  --iree-hal-target-backends=rocm \
  --iree-rocm-target="${ROCM_TARGET}" \
  -o "${ROCM_VMFB}"

"${PYTHON}" -m torch2air run-demo \
  --out-dir "${OUT_DIR}" \
  --vmfb "${ROCM_VMFB}" \
  --iree-run-module "${IREE_RUN_MODULE}" \
  --device hip
