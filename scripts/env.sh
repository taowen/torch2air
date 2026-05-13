#!/usr/bin/env bash

TORCH2AIR_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export TORCH2AIR_ROOT=${TORCH2AIR_ROOT:-$(cd -- "${TORCH2AIR_SCRIPT_DIR}/.." && pwd)}
export IREE_AMD_AIE_ROOT=${IREE_AMD_AIE_ROOT:-$(cd -- "${TORCH2AIR_ROOT}/../.." && pwd)}
if [[ -z "${XDNA_DRIVER_ROOT:-}" && -d "${IREE_AMD_AIE_ROOT}/../xdna-driver" ]]; then
  export XDNA_DRIVER_ROOT=$(cd -- "${IREE_AMD_AIE_ROOT}/../xdna-driver" && pwd)
fi

if [[ -f "${IREE_AMD_AIE_ROOT}/iree-install-rocm/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${IREE_AMD_AIE_ROOT}/iree-install-rocm/env.sh"
fi

if [[ -n "${XDNA_DRIVER_ROOT:-}" && -f "${XDNA_DRIVER_ROOT}/build-bazzite/env-opt-xrt.sh" ]]; then
  # shellcheck disable=SC1091
  source "${XDNA_DRIVER_ROOT}/build-bazzite/env-opt-xrt.sh" >/dev/null
fi

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${TORCH2AIR_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${TORCH2AIR_ROOT}/.venv/bin/python"
  elif [[ -x "/var/home/taowen/projects/torch2vk/.venv/bin/python" ]]; then
    PYTHON="/var/home/taowen/projects/torch2vk/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

export PYTHON
case ":${PYTHONPATH:-}:" in
  *":${TORCH2AIR_ROOT}/src:"*) ;;
  *) export PYTHONPATH="${TORCH2AIR_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export TORCH2AIR_OUT_DIR=${TORCH2AIR_OUT_DIR:-"${TORCH2AIR_ROOT}/examples/generated"}
export IREE_COMPILE=${IREE_COMPILE:-iree-compile}
export IREE_RUN_MODULE=${IREE_RUN_MODULE:-iree-run-module}
export TORCH2AIR_FRONTEND=${TORCH2AIR_FRONTEND:-fx}
export TORCH2AIR_TARGET_DEVICE=${TORCH2AIR_TARGET_DEVICE:-npu4}
export TORCH2AIR_TILE_PIPELINE=${TORCH2AIR_TILE_PIPELINE:-pack-peel}
export TORCH2AIR_LOWER_TO_AIE_PIPELINE=${TORCH2AIR_LOWER_TO_AIE_PIPELINE:-air}
export TORCH2AIR_VMFB_TILE_PIPELINE=${TORCH2AIR_VMFB_TILE_PIPELINE:-pack-peel}
export TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE=${TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE:-objectFifo}
export ROCM_TARGET=${ROCM_TARGET:-gfx1150}
export XRT_LITE_N_CORE_ROWS=${XRT_LITE_N_CORE_ROWS:-4}
export XRT_LITE_N_CORE_COLS=${XRT_LITE_N_CORE_COLS:-8}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  cat <<EOF
source this script to configure the current shell:
  source ${BASH_SOURCE[0]}

TORCH2AIR_ROOT=${TORCH2AIR_ROOT}
IREE_AMD_AIE_ROOT=${IREE_AMD_AIE_ROOT}
XDNA_DRIVER_ROOT=${XDNA_DRIVER_ROOT:-}
PYTHON=${PYTHON}
IREE_COMPILE=${IREE_COMPILE}
IREE_RUN_MODULE=${IREE_RUN_MODULE}
TORCH2AIR_OUT_DIR=${TORCH2AIR_OUT_DIR}
ROCM_TARGET=${ROCM_TARGET}
TORCH2AIR_TARGET_DEVICE=${TORCH2AIR_TARGET_DEVICE}
TORCH2AIR_TILE_PIPELINE=${TORCH2AIR_TILE_PIPELINE}
TORCH2AIR_LOWER_TO_AIE_PIPELINE=${TORCH2AIR_LOWER_TO_AIE_PIPELINE}
TORCH2AIR_VMFB_TILE_PIPELINE=${TORCH2AIR_VMFB_TILE_PIPELINE}
TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE=${TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE}
XRT_LITE_N_CORE_ROWS=${XRT_LITE_N_CORE_ROWS}
XRT_LITE_N_CORE_COLS=${XRT_LITE_N_CORE_COLS}
EOF
fi
