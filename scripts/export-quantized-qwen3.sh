#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${USE_LOCAL_PROXY:=1}"
: "${UV:=uv}"

if [[ "$USE_LOCAL_PROXY" == "1" ]]; then
  export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
  export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
  export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
  export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
fi

exec "$UV" run --no-sync python -m models.quantized_qwen3.export "$@"
