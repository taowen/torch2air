#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stage verification means hardware execution for quantized_qwen3.
STAGE="${1:-embed_tokens}"
exec "$ROOT_DIR/scripts/run-quantized-qwen3-npu.sh" "$STAGE"
