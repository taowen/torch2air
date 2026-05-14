#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-0}"
EXPORT_DIR="${EXPORT_DIR:-$ROOT_DIR/.cache/q4k-linear-formal-export}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.cache/npu-spikes/q4k-linear-formal-q-proj}"
OUTPUT_ROWS="${OUTPUT_ROWS:-64}"
OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-16}"

source "$ROOT_DIR/scripts/npu-common.sh"

check_npu_device
"$UV" run --no-sync python -m models.quantized_qwen3.export \
  --stage q_proj \
  --sequence-length 1 \
  --output-dir "$EXPORT_DIR"
"$UV" run --no-sync python -m models.quantized_qwen3.run_q_proj \
  --kernel-py "$EXPORT_DIR/run_q_proj.py" \
  --gguf "$GGUF_PATH" \
  --token-ids "$TOKEN_IDS" \
  --output-rows "$OUTPUT_ROWS" \
  --output-tile-rows "$OUTPUT_TILE_ROWS" \
  --work-dir "$WORK_DIR" \
  "$@"
