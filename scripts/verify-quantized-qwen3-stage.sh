#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Stage verification means hardware execution for quantized_qwen3. The run
# scripts also perform export and AIR lowering checks before loading XRT.
STAGE="${1:-embed_tokens}"
if [[ "$STAGE" == "pipeline_embed_norm" ]]; then
  exec "$ROOT_DIR/scripts/run-quantized-qwen3-pipeline-npu.sh"
fi
exec "$ROOT_DIR/scripts/run-quantized-qwen3-npu.sh" "$STAGE"
