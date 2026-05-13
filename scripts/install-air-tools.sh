#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PYTHON_VERSION:=3.12}"
: "${VENV_DIR:=.venv}"
: "${USE_LOCAL_PROXY:=1}"

AIR_WHEELS_URL="${AIR_WHEELS_URL:-https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels}"
AIE_WHEELS_URL="${AIE_WHEELS_URL:-https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-3}"
LLVM_AIE_WHEELS_URL="${LLVM_AIE_WHEELS_URL:-https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly}"

if [[ "$USE_LOCAL_PROXY" == "1" ]]; then
  export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
  export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
  export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
  export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv is required but was not found in PATH." >&2
  exit 1
}

if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PYTHON_VERSION="$("$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "$VENV_PYTHON_VERSION" != "3.12" ]]; then
    echo "Existing $VENV_DIR uses Python $VENV_PYTHON_VERSION; expected 3.12." >&2
    echo "Remove $VENV_DIR or set VENV_DIR to a different path." >&2
    exit 1
  fi
else
  uv venv --python "$PYTHON_VERSION" --allow-existing "$VENV_DIR"
fi

uv pip install --python "$VENV_DIR/bin/python" 'mlir_air[aie]' \
  -f "$AIR_WHEELS_URL" \
  -f "$AIE_WHEELS_URL" \
  -f "$LLVM_AIE_WHEELS_URL"

uv pip show --python "$VENV_DIR/bin/python" mlir-air mlir-aie llvm-aie
