#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAGE="${1:-embed_tokens}"
UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-0}"
BLOCKS_PER_ROW="${BLOCKS_PER_ROW:-4}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/src/models/quantized_qwen3/generated/lowered}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"
RMS_NORM_EPS="${RMS_NORM_EPS:-0.000001}"
OUTPUT_ROWS="${OUTPUT_ROWS:-64}"
OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-16}"

IFS=',' read -r -a _token_parts <<< "$TOKEN_IDS"
TOKEN_COUNT="${TOKEN_COUNT:-${#_token_parts[@]}}"
case "$STAGE" in
  embed_tokens)
    STEM="quantized_qwen3_embed_tokens"
    FUNC="run_embed_tokens"
    RUNNER_MODULE="models.quantized_qwen3.run_embed_tokens"
    DEFAULT_HERD_COLS="$BLOCKS_PER_ROW"
    ;;
  input_layernorm)
    STEM="quantized_qwen3_input_layernorm"
    FUNC="run_input_layernorm"
    RUNNER_MODULE="models.quantized_qwen3.run_input_layernorm"
    DEFAULT_HERD_COLS="1"
    ;;
  embed_tokens_input_layernorm)
    STEM="quantized_qwen3_embed_tokens_input_layernorm"
    FUNC="run_embed_tokens_input_layernorm"
    RUNNER_MODULE="models.quantized_qwen3.run_embed_tokens_input_layernorm"
    DEFAULT_HERD_COLS="1"
    ;;
  q_proj|k_proj|v_proj)
    if (( OUTPUT_TILE_ROWS <= 0 || OUTPUT_ROWS % OUTPUT_TILE_ROWS != 0 )); then
      echo "OUTPUT_TILE_ROWS must be positive and divide OUTPUT_ROWS." >&2
      exit 2
    fi
    STEM="quantized_qwen3_${STAGE}"
    FUNC="run_${STAGE}"
    RUNNER_MODULE="models.quantized_qwen3.run_q_proj"
    DEFAULT_HERD_COLS="$((OUTPUT_ROWS / OUTPUT_TILE_ROWS))"
    ;;
  *)
    echo "Unknown quantized_qwen3 stage: $STAGE" >&2
    exit 2
    ;;
esac
if [[ "$STAGE" == "embed_tokens_input_layernorm" && "$BLOCKS_PER_ROW" != "1" ]]; then
  echo "embed_tokens_input_layernorm fusion is currently verified only for BLOCKS_PER_ROW=1." >&2
  echo "Use STAGE=input_layernorm with BLOCKS_PER_ROW=4 for the full hidden-size RMSNorm step." >&2
  exit 2
fi

AIR_HERD_ROWS="${AIR_HERD_ROWS:-$TOKEN_COUNT}"
AIR_HERD_COLS="${AIR_HERD_COLS:-$DEFAULT_HERD_COLS}"
export AIR_HERD_ROWS AIR_HERD_COLS

source "$ROOT_DIR/scripts/npu-common.sh"
source "$ROOT_DIR/scripts/verify-air-common.sh"

MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/$FUNC.mlir"
AIE_MLIR="$OUT_DIR/$STEM.aie.mlir"
WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/quantized-qwen3-${STAGE}-${TOKEN_COUNT}tok-${BLOCKS_PER_ROW}block}"

EXPORT_ARGS=(
  --stage "$STAGE" \
  --gguf "$GGUF_PATH" \
  --sequence-length "$TOKEN_COUNT" \
  --blocks-per-row "$BLOCKS_PER_ROW"
)
if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  EXPORT_ARGS+=(--output-rows "$OUTPUT_ROWS" --output-tile-rows "$OUTPUT_TILE_ROWS")
fi
"$ROOT_DIR/scripts/export-quantized-qwen3.sh" "${EXPORT_ARGS[@]}"

if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  check_contains "$MLIR" 'air\.launch' 'official AIR launch in exported MLIR'
  check_contains "$MLIR" 'air\.segment' 'official AIR segment in exported MLIR'
else
  check_contains "$MLIR" 'scf\.parallel' 'explicit tile loop in exported MLIR'
  check_contains "$MLIR" 'memref\.subview' 'explicit tile subview in exported MLIR'
  check_contains "$MLIR" 'memref\.copy' 'explicit copy before AIR DMA lowering'
fi
if [[ "$STAGE" == "embed_tokens" || "$STAGE" == "embed_tokens_input_layernorm" ]]; then
  check_contains "$MLIR" 'memref\.alloc\(\) : memref<1x36xi32, 2>' 'packed Q4_K L1 tile'
  check_contains "$MLIR" 'memref\.alloc\(\) : memref<1x1x2xf32, 2>' 'host-decoded Q4_K scale L1 tile'
fi
if [[ "$STAGE" == "input_layernorm" || "$STAGE" == "embed_tokens_input_layernorm" ]]; then
  check_contains "$MLIR" 'math\.rsqrt' 'RMSNorm inverse square root in fused MLIR'
fi
if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  if [[ "$STAGE" == "v_proj" ]]; then
    PROJ_WEIGHT_WORDS="$((BLOCKS_PER_ROW * 106))"
    PROJ_TILE_FUNC="q6k_linear_tile"
    PROJ_LINK_OBJECT="q6k_linear.o"
  else
    PROJ_WEIGHT_WORDS="$((BLOCKS_PER_ROW * 38))"
    PROJ_TILE_FUNC="q4k_linear_tile"
    PROJ_LINK_OBJECT="q4k_linear.o"
  fi
  check_contains "$MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${PROJ_WEIGHT_WORDS}xi32, 2" 'packed quantized weight L1 tile'
  check_contains "$MLIR" "func\\.call @${PROJ_TILE_FUNC}" 'external quantized tile kernel call'
  check_contains "$MLIR" "link_with = \"${PROJ_LINK_OBJECT}\"" 'external quantized link object'
fi
if [[ "$STAGE" == "embed_tokens_input_layernorm" ]]; then
  check_contains "$MLIR" 'embed_tokens output in L1' 'documented L1 operator handoff'
fi
if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  check_contains "$MLIR" 'air\.herd' 'official AIR herd in exported MLIR'
  check_contains "$MLIR" 'air\.dma_memcpy_nd' 'explicit AIR DMA in exported MLIR'
  compile_air_dma_fixture "$MLIR" "$STEM"
else
  compile_air_fixture "$MLIR" "$STEM"
fi

RUNNER_ARGS=(
  -m "$RUNNER_MODULE"
  --gguf "$GGUF_PATH"
  --token-ids "$TOKEN_IDS"
  --aie-mlir "$AIE_MLIR"
  --work-dir "$WORK_DIR"
  --warmup "$NPU_WARMUP"
  --iterations "$NPU_ITERATIONS"
)
if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  RUNNER_ARGS+=(--proj-name "$STAGE" --output-rows "$OUTPUT_ROWS" --output-tile-rows "$OUTPUT_TILE_ROWS")
else
  RUNNER_ARGS+=(--blocks-per-row "$BLOCKS_PER_ROW")
fi
if [[ "$STAGE" == "input_layernorm" || "$STAGE" == "embed_tokens_input_layernorm" ]]; then
  RUNNER_ARGS+=(--rms-norm-eps "$RMS_NORM_EPS")
fi
if [[ "$STAGE" == "q_proj" || "$STAGE" == "k_proj" || "$STAGE" == "v_proj" ]]; then
  RUNNER_ARGS+=(--rms-norm-eps "$RMS_NORM_EPS")
fi

"$UV" run --no-sync python "${RUNNER_ARGS[@]}"
