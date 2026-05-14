#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-embed_tokens}"
UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-0}"
BLOCKS_PER_ROW="${BLOCKS_PER_ROW:-4}"
RMS_NORM_EPS="${RMS_NORM_EPS:-0.000001}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"

IFS=',' read -r -a TOKEN_PARTS <<< "$TOKEN_IDS"
TOKEN_COUNT="${#TOKEN_PARTS[@]}"
EXPORT_DIR="${EXPORT_DIR:-$ROOT_DIR/.cache/quantized-qwen3-export}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/.cache/npu/quantized-qwen3-$STAGE-${TOKEN_COUNT}tok}"

source "$ROOT_DIR/scripts/npu-common.sh"
check_npu_device

"$UV" run --no-sync python -m models.quantized_qwen3.export \
  --stage "$STAGE" \
  --sequence-length "$TOKEN_COUNT" \
  --output-dir "$EXPORT_DIR"

case "$STAGE" in
  embed_tokens)
    "$UV" run --no-sync python -m models.quantized_qwen3.run_embed_tokens \
      --kernel-py "$EXPORT_DIR/run_embed_tokens.py" \
      --gguf "$GGUF_PATH" \
      --token-ids "$TOKEN_IDS" \
      --blocks-per-row "$BLOCKS_PER_ROW" \
      --work-dir "$WORK_DIR" \
      --warmup "$NPU_WARMUP" \
      --iterations "$NPU_ITERATIONS"
    ;;
  input_layernorm)
    "$UV" run --no-sync python -m models.quantized_qwen3.run_input_layernorm \
      --kernel-py "$EXPORT_DIR/run_input_layernorm.py" \
      --gguf "$GGUF_PATH" \
      --token-ids "$TOKEN_IDS" \
      --blocks-per-row "$BLOCKS_PER_ROW" \
      --rms-norm-eps "$RMS_NORM_EPS" \
      --work-dir "$WORK_DIR" \
      --warmup "$NPU_WARMUP" \
      --iterations "$NPU_ITERATIONS"
    ;;
  q_proj|k_proj|o_proj)
    if [[ "$TOKEN_COUNT" == "8" ]]; then
      OUTPUT_ROWS="${OUTPUT_ROWS:-16}"
    else
      OUTPUT_ROWS="${OUTPUT_ROWS:-64}"
    fi
    OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-16}"
    "$UV" run --no-sync python -m models.quantized_qwen3.run_linear \
      --stage "$STAGE" \
      --kernel-py "$EXPORT_DIR/run_${STAGE}.py" \
      --gguf "$GGUF_PATH" \
      --token-ids "$TOKEN_IDS" \
      --blocks-per-row "$BLOCKS_PER_ROW" \
      --output-rows "$OUTPUT_ROWS" \
      --output-tile-rows "$OUTPUT_TILE_ROWS" \
      --rms-norm-eps "$RMS_NORM_EPS" \
      --work-dir "$WORK_DIR" \
      --warmup "$NPU_WARMUP" \
      --iterations "$NPU_ITERATIONS"
    ;;
  *)
    echo "Unsupported quantized_qwen3 NPU stage: $STAGE" >&2
    exit 2
    ;;
esac
