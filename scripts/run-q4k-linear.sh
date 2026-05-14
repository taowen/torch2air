#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
STAGE="${STAGE:-q_proj}"
TOKEN_IDS="${TOKEN_IDS:-0}"
EXPORT_DIR="${EXPORT_DIR:-$ROOT_DIR/.cache/q4k-linear-export}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.cache/npu/q4k-linear-$STAGE}"
OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-16}"
FUNCTION_NAME="run_$STAGE"
IFS=',' read -r -a TOKEN_PARTS <<< "$TOKEN_IDS"
TOKEN_COUNT="${#TOKEN_PARTS[@]}"
if [[ "$TOKEN_COUNT" == "8" ]]; then
  OUTPUT_ROWS="${OUTPUT_ROWS:-16}"
else
  OUTPUT_ROWS="${OUTPUT_ROWS:-64}"
fi

source "$ROOT_DIR/scripts/npu-common.sh"

check_npu_device
"$UV" run --no-sync python -m models.quantized_qwen3.export \
  --stage "$STAGE" \
  --sequence-length "$TOKEN_COUNT" \
  --output-dir "$EXPORT_DIR"
"$UV" run --no-sync python -m models.quantized_qwen3.run_linear \
  --stage "$STAGE" \
  --kernel-py "$EXPORT_DIR/$FUNCTION_NAME.py" \
  --function-name "$FUNCTION_NAME" \
  --gguf "$GGUF_PATH" \
  --token-ids "$TOKEN_IDS" \
  --output-rows "$OUTPUT_ROWS" \
  --output-tile-rows "$OUTPUT_TILE_ROWS" \
  --work-dir "$WORK_DIR" \
  "$@"
