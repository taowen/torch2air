#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-0}"
BLOCKS_PER_ROW="${BLOCKS_PER_ROW:-4}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/src/models/quantized_qwen3/generated/lowered}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"
RMS_NORM_EPS="${RMS_NORM_EPS:-0.000001}"
STEM="quantized_qwen3_pipeline_embed_norm"
FUNC="run_pipeline_embed_norm"

IFS=',' read -r -a _token_parts <<< "$TOKEN_IDS"
TOKEN_COUNT="${TOKEN_COUNT:-${#_token_parts[@]}}"
if [[ "$BLOCKS_PER_ROW" != "4" ]]; then
  echo "pipeline_embed_norm currently targets full hidden_size=1024, so BLOCKS_PER_ROW must be 4." >&2
  exit 2
fi

AIR_HERD_ROWS="${AIR_HERD_ROWS:-$TOKEN_COUNT}"
AIR_HERD_COLS="${AIR_HERD_COLS:-$BLOCKS_PER_ROW}"
export AIR_HERD_ROWS AIR_HERD_COLS

source "$ROOT_DIR/scripts/npu-common.sh"
source "$ROOT_DIR/scripts/verify-air-common.sh"

EMBED_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_embed_tokens.mlir"
NORM_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_input_layernorm.mlir"
PIPELINE_DMA="$OUT_DIR/$STEM.dma.mlir"
WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/quantized-qwen3-pipeline_embed_norm-${TOKEN_COUNT}tok-${BLOCKS_PER_ROW}block}"

"$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
  --stage pipeline_embed_norm \
  --gguf "$GGUF_PATH" \
  --sequence-length "$TOKEN_COUNT" \
  --blocks-per-row "$BLOCKS_PER_ROW"

check_contains "$EMBED_MLIR" 'scf\.parallel' 'embed_tokens explicit tile loop'
check_contains "$EMBED_MLIR" 'memref\.subview' 'embed_tokens tile subview'
check_contains "$EMBED_MLIR" 'memref\.copy' 'embed_tokens copy before AIR DMA lowering'
check_contains "$NORM_MLIR" 'scf\.parallel' 'input_layernorm explicit tile loop'
check_contains "$NORM_MLIR" 'math\.rsqrt' 'input_layernorm inverse square root'

lower_air_fixture_to_dma "$EMBED_MLIR" "${STEM}_embed_tokens"
EMBED_DMA="$DMA_IR"
lower_air_fixture_to_dma "$NORM_MLIR" "${STEM}_input_layernorm"
NORM_DMA="$DMA_IR"
compile_air_dma_fixture "$EMBED_DMA" "${STEM}_embed_tokens"
EMBED_AIE="$AIE_IR"
compile_air_dma_fixture "$NORM_DMA" "${STEM}_input_layernorm"
NORM_AIE="$AIE_IR"

"$UV" run --no-sync python -m models.quantized_qwen3.stitch_pipeline \
  --embed-dma-mlir "$EMBED_DMA" \
  --norm-dma-mlir "$NORM_DMA" \
  --output "$PIPELINE_DMA" \
  --sequence-length "$TOKEN_COUNT" \
  --blocks-per-row "$BLOCKS_PER_ROW" \
  --function-name "$FUNC"

check_contains "$PIPELINE_DMA" "func.func @$FUNC" 'stitched pipeline function'
check_count_ge "$PIPELINE_DMA" 'air\.launch' 2 'stitched air.launch'
check_contains "$PIPELINE_DMA" '@emb_run_embed_tokens_0' 'renamed embed_tokens segment'
check_contains "$PIPELINE_DMA" '@norm_run_input_layernorm_0' 'renamed input_layernorm segment'

if [[ "${PIPELINE_DEBUG_AIE:-0}" == "1" ]]; then
  compile_air_dma_fixture "$PIPELINE_DMA" "$STEM"
fi

"$ROOT_DIR/.venv/bin/python" -m models.quantized_qwen3.run_pipeline \
  --gguf "$GGUF_PATH" \
  --token-ids "$TOKEN_IDS" \
  --blocks-per-row "$BLOCKS_PER_ROW" \
  --rms-norm-eps "$RMS_NORM_EPS" \
  --embed-aie-mlir "$EMBED_AIE" \
  --norm-aie-mlir "$NORM_AIE" \
  --work-dir "$WORK_DIR" \
  --warmup "$NPU_WARMUP" \
  --iterations "$NPU_ITERATIONS"
