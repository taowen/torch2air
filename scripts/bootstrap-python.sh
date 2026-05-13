#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

UV=${UV:-uv}
VENV_DIR=${VENV_DIR:-"${TORCH2AIR_ROOT}/.venv"}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TORCH2AIR_EXTRAS=${TORCH2AIR_EXTRAS:-dev}
INSTALL_TORCH_MLIR=${INSTALL_TORCH_MLIR:-0}

"${UV}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
if [[ -n "${TORCH2AIR_EXTRAS}" ]]; then
  "${UV}" pip install --python "${VENV_DIR}/bin/python" -e "${TORCH2AIR_ROOT}[${TORCH2AIR_EXTRAS}]"
else
  "${UV}" pip install --python "${VENV_DIR}/bin/python" -e "${TORCH2AIR_ROOT}"
fi

if [[ "${INSTALL_TORCH_MLIR}" == "1" ]]; then
  "${UV}" pip install --python "${VENV_DIR}/bin/python" --pre torch-mlir \
    -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
fi

cat <<EOF
Python environment is ready:
  export PYTHON=${VENV_DIR}/bin/python
  source ${SCRIPT_DIR}/env.sh
EOF
