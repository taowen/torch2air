#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
PIPELINE_STAGE="${1:-embed_norm}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-0}"
BLOCKS_PER_ROW="${BLOCKS_PER_ROW:-4}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/src/models/quantized_qwen3/generated/lowered}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"
RMS_NORM_EPS="${RMS_NORM_EPS:-0.000001}"
OUTPUT_ROWS="${OUTPUT_ROWS:-64}"
OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-16}"

case "$PIPELINE_STAGE" in
  embed_norm|pipeline_embed_norm)
    INCLUDE_QPROJ=0
    INCLUDE_QKV=0
    STEM="quantized_qwen3_pipeline_embed_norm"
    FUNC="run_pipeline_embed_norm"
    ;;
  embed_norm_qproj|pipeline_embed_norm_qproj)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=0
    STEM="quantized_qwen3_pipeline_embed_norm_qproj"
    FUNC="run_pipeline_embed_norm_qproj"
    ;;
  embed_norm_qkv|pipeline_embed_norm_qkv)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=1
    STEM="quantized_qwen3_pipeline_embed_norm_qkv"
    FUNC="run_pipeline_embed_norm_qkv"
    ;;
  *)
    echo "Unknown quantized_qwen3 pipeline stage: $PIPELINE_STAGE" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a _token_parts <<< "$TOKEN_IDS"
TOKEN_COUNT="${TOKEN_COUNT:-${#_token_parts[@]}}"
if [[ "$BLOCKS_PER_ROW" != "4" ]]; then
  echo "pipeline_embed_norm currently targets full hidden_size=1024, so BLOCKS_PER_ROW must be 4." >&2
  exit 2
fi
if (( INCLUDE_QPROJ && (OUTPUT_TILE_ROWS <= 0 || OUTPUT_ROWS % OUTPUT_TILE_ROWS != 0) )); then
  echo "OUTPUT_TILE_ROWS must be positive and divide OUTPUT_ROWS." >&2
  exit 2
fi

AIR_HERD_ROWS="${AIR_HERD_ROWS:-$TOKEN_COUNT}"
AIR_HERD_COLS="${AIR_HERD_COLS:-$BLOCKS_PER_ROW}"
export AIR_HERD_ROWS AIR_HERD_COLS

source "$ROOT_DIR/scripts/npu-common.sh"
source "$ROOT_DIR/scripts/verify-air-common.sh"

EMBED_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_embed_tokens.mlir"
NORM_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_input_layernorm.mlir"
QPROJ_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_q_proj.mlir"
KPROJ_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_k_proj.mlir"
VPROJ_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_v_proj.mlir"
PIPELINE_DMA="$OUT_DIR/$STEM.dma.mlir"
WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/quantized-qwen3-${PIPELINE_STAGE}-${TOKEN_COUNT}tok-${BLOCKS_PER_ROW}block}"

"$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
  --stage pipeline_embed_norm \
  --gguf "$GGUF_PATH" \
  --sequence-length "$TOKEN_COUNT" \
  --blocks-per-row "$BLOCKS_PER_ROW"
if (( INCLUDE_QPROJ )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage q_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --output-rows "$OUTPUT_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
fi
if (( INCLUDE_QKV )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage k_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --output-rows "$OUTPUT_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage v_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --output-rows "$OUTPUT_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
fi

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
if (( INCLUDE_QPROJ )); then
  HERD_COLS="$((OUTPUT_ROWS / OUTPUT_TILE_ROWS))"
  PROJ_WEIGHT_WORDS="$((BLOCKS_PER_ROW * 38))"
  check_contains "$QPROJ_MLIR" 'air\.launch' 'q_proj official AIR launch'
  check_contains "$QPROJ_MLIR" 'air\.herd' 'q_proj official AIR herd'
  check_contains "$QPROJ_MLIR" 'air\.dma_memcpy_nd' 'q_proj explicit AIR DMA'
  check_contains "$QPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${PROJ_WEIGHT_WORDS}xi32, 2" 'q_proj packed Q4_K weight L1 tile'
  check_contains "$QPROJ_MLIR" 'func\.call @q4k_linear_tile' 'q_proj external Q4_K tile kernel call'
  check_contains "$QPROJ_MLIR" 'link_with = "q4k_linear\.o"' 'q_proj external Q4_K link object'
  compile_air_dma_fixture "$QPROJ_MLIR" "${STEM}_q_proj"
  QPROJ_AIE="$AIE_IR"
  if (( INCLUDE_QKV )); then
    check_contains "$KPROJ_MLIR" 'air\.launch' 'k_proj official AIR launch'
    check_contains "$KPROJ_MLIR" 'air\.herd' 'k_proj official AIR herd'
    check_contains "$KPROJ_MLIR" 'air\.dma_memcpy_nd' 'k_proj explicit AIR DMA'
    check_contains "$KPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${PROJ_WEIGHT_WORDS}xi32, 2" 'k_proj packed Q4_K weight L1 tile'
    check_contains "$KPROJ_MLIR" 'func\.call @q4k_linear_tile' 'k_proj external Q4_K tile kernel call'
    check_contains "$KPROJ_MLIR" 'link_with = "q4k_linear\.o"' 'k_proj external Q4_K link object'
    compile_air_dma_fixture "$KPROJ_MLIR" "${STEM}_k_proj"
    KPROJ_AIE="$AIE_IR"
    VPROJ_WEIGHT_WORDS="$((BLOCKS_PER_ROW * 106))"
    check_contains "$VPROJ_MLIR" 'air\.launch' 'v_proj official AIR launch'
    check_contains "$VPROJ_MLIR" 'air\.herd' 'v_proj official AIR herd'
    check_contains "$VPROJ_MLIR" 'air\.dma_memcpy_nd' 'v_proj explicit AIR DMA'
    check_contains "$VPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${VPROJ_WEIGHT_WORDS}xi32, 2" 'v_proj packed Q6_K weight L1 tile'
    check_contains "$VPROJ_MLIR" 'func\.call @q6k_linear_tile' 'v_proj external Q6_K tile kernel call'
    check_contains "$VPROJ_MLIR" 'link_with = "q6k_linear\.o"' 'v_proj external Q6_K link object'
    compile_air_dma_fixture "$VPROJ_MLIR" "${STEM}_v_proj"
    VPROJ_AIE="$AIE_IR"
  fi
  HERD_COLS="$BLOCKS_PER_ROW"
fi

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

RUNNER_ARGS=(
  -m models.quantized_qwen3.run_pipeline
  --gguf "$GGUF_PATH"
  --token-ids "$TOKEN_IDS"
  --blocks-per-row "$BLOCKS_PER_ROW"
  --rms-norm-eps "$RMS_NORM_EPS"
  --embed-aie-mlir "$EMBED_AIE"
  --norm-aie-mlir "$NORM_AIE"
  --work-dir "$WORK_DIR"
  --warmup "$NPU_WARMUP"
  --iterations "$NPU_ITERATIONS"
)
if (( INCLUDE_QPROJ )); then
  RUNNER_ARGS+=(
    --qproj-aie-mlir "$QPROJ_AIE"
    --output-rows "$OUTPUT_ROWS"
    --output-tile-rows "$OUTPUT_TILE_ROWS"
  )
fi
if (( INCLUDE_QKV )); then
  RUNNER_ARGS+=(
    --kproj-aie-mlir "$KPROJ_AIE"
    --vproj-aie-mlir "$VPROJ_AIE"
  )
fi

"$UV" run --no-sync python "${RUNNER_ARGS[@]}"
