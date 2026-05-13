#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${VENV_DIR:=.venv}"
: "${USE_LOCAL_PROXY:=1}"

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

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python 3.12 "$VENV_DIR"
fi

uv pip install --python "$VENV_DIR/bin/python" -e ".[dev]"
