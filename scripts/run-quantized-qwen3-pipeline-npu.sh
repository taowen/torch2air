#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
PIPELINE_STAGE="${1:-embed_norm}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_IDS="${TOKEN_IDS:-}"
BLOCKS_PER_ROW="${BLOCKS_PER_ROW:-4}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/src/models/quantized_qwen3/generated/lowered}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"
RMS_NORM_EPS="${RMS_NORM_EPS:-0.000001}"
OUTPUT_ROWS="${OUTPUT_ROWS:-128}"
QPROJ_OUTPUT_ROWS="${QPROJ_OUTPUT_ROWS:-}"
KPROJ_OUTPUT_ROWS="${KPROJ_OUTPUT_ROWS:-}"
VPROJ_OUTPUT_ROWS="${VPROJ_OUTPUT_ROWS:-}"
OPROJ_OUTPUT_ROWS="${OPROJ_OUTPUT_ROWS:-}"
OUTPUT_TILE_ROWS="${OUTPUT_TILE_ROWS:-32}"
PROJECTION_SLICE_ROWS="${PROJECTION_SLICE_ROWS:-$((OUTPUT_TILE_ROWS * 4))}"
START_POSITION="${START_POSITION:-0}"
QUERY_TILE_ROWS="${QUERY_TILE_ROWS:-4}"
KEY_TILE_ROWS="${KEY_TILE_ROWS:-4}"
Q_HEADS="${Q_HEADS:-}"
KV_HEADS="${KV_HEADS:-}"
HEAD_DIM=128

case "$PIPELINE_STAGE" in
  embed_norm|pipeline_embed_norm)
    INCLUDE_QPROJ=0
    INCLUDE_QKV=0
    INCLUDE_ROPE=0
    INCLUDE_ATTENTION=0
    INCLUDE_OPROJ=0
    STEM="quantized_qwen3_pipeline_embed_norm"
    FUNC="run_pipeline_embed_norm"
    ;;
  embed_norm_qproj|pipeline_embed_norm_qproj)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=0
    INCLUDE_ROPE=0
    INCLUDE_ATTENTION=0
    INCLUDE_OPROJ=0
    STEM="quantized_qwen3_pipeline_embed_norm_qproj"
    FUNC="run_pipeline_embed_norm_qproj"
    ;;
  embed_norm_qkv|pipeline_embed_norm_qkv)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=1
    INCLUDE_ROPE=0
    INCLUDE_ATTENTION=0
    INCLUDE_OPROJ=0
    STEM="quantized_qwen3_pipeline_embed_norm_qkv"
    FUNC="run_pipeline_embed_norm_qkv"
    ;;
  embed_norm_qkv_rope|pipeline_embed_norm_qkv_rope)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=1
    INCLUDE_ROPE=1
    INCLUDE_ATTENTION=0
    INCLUDE_OPROJ=0
    STEM="quantized_qwen3_pipeline_embed_norm_qkv_rope"
    FUNC="run_pipeline_embed_norm_qkv_rope"
    ;;
  embed_norm_qkv_rope_attention|pipeline_embed_norm_qkv_rope_attention|attention)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=1
    INCLUDE_ROPE=1
    INCLUDE_ATTENTION=1
    INCLUDE_OPROJ=0
    STEM="quantized_qwen3_pipeline_embed_norm_qkv_rope_attention"
    FUNC="run_pipeline_embed_norm_qkv_rope_attention"
    ;;
  self_attn|pipeline_self_attn)
    INCLUDE_QPROJ=1
    INCLUDE_QKV=1
    INCLUDE_ROPE=1
    INCLUDE_ATTENTION=1
    INCLUDE_OPROJ=1
    STEM="quantized_qwen3_pipeline_self_attn"
    FUNC="run_pipeline_self_attn"
    ;;
  *)
    echo "Unknown quantized_qwen3 pipeline stage: $PIPELINE_STAGE" >&2
    exit 2
    ;;
esac

if [[ -z "$TOKEN_IDS" ]]; then
  if (( INCLUDE_ATTENTION )); then
    TOKEN_IDS="0,1,2,3"
  else
    TOKEN_IDS="0"
  fi
fi

if (( INCLUDE_OPROJ )); then
  Q_HEADS="${Q_HEADS:-16}"
  KV_HEADS="${KV_HEADS:-8}"
  QPROJ_OUTPUT_ROWS="${QPROJ_OUTPUT_ROWS:-$((Q_HEADS * HEAD_DIM))}"
  KPROJ_OUTPUT_ROWS="${KPROJ_OUTPUT_ROWS:-$((KV_HEADS * HEAD_DIM))}"
  VPROJ_OUTPUT_ROWS="${VPROJ_OUTPUT_ROWS:-$((KV_HEADS * HEAD_DIM))}"
  OPROJ_OUTPUT_ROWS="${OPROJ_OUTPUT_ROWS:-1024}"
else
  Q_HEADS="${Q_HEADS:-1}"
  KV_HEADS="${KV_HEADS:-1}"
  QPROJ_OUTPUT_ROWS="${QPROJ_OUTPUT_ROWS:-$OUTPUT_ROWS}"
  KPROJ_OUTPUT_ROWS="${KPROJ_OUTPUT_ROWS:-$OUTPUT_ROWS}"
  VPROJ_OUTPUT_ROWS="${VPROJ_OUTPUT_ROWS:-$OUTPUT_ROWS}"
  OPROJ_OUTPUT_ROWS="${OPROJ_OUTPUT_ROWS:-1024}"
fi

projection_parallel_tiles() {
  local output_rows="$1"
  local output_tiles=$((output_rows / OUTPUT_TILE_ROWS))
  if (( output_tiles < 4 )); then
    echo "$output_tiles"
  else
    echo 4
  fi
}

IFS=',' read -r -a _token_parts <<< "$TOKEN_IDS"
TOKEN_COUNT="${TOKEN_COUNT:-${#_token_parts[@]}}"
if [[ "$BLOCKS_PER_ROW" != "4" ]]; then
  echo "pipeline_embed_norm currently targets full hidden_size=1024, so BLOCKS_PER_ROW must be 4." >&2
  exit 2
fi
if (( INCLUDE_QPROJ && (OUTPUT_TILE_ROWS <= 0 || PROJECTION_SLICE_ROWS <= 0 || PROJECTION_SLICE_ROWS % OUTPUT_TILE_ROWS != 0 || QPROJ_OUTPUT_ROWS % PROJECTION_SLICE_ROWS != 0) )); then
  echo "PROJECTION_SLICE_ROWS must be positive, divisible by OUTPUT_TILE_ROWS, and divide QPROJ_OUTPUT_ROWS." >&2
  exit 2
fi
if (( INCLUDE_QKV && (KPROJ_OUTPUT_ROWS % PROJECTION_SLICE_ROWS != 0 || VPROJ_OUTPUT_ROWS % PROJECTION_SLICE_ROWS != 0) )); then
  echo "PROJECTION_SLICE_ROWS must divide KPROJ_OUTPUT_ROWS and VPROJ_OUTPUT_ROWS." >&2
  exit 2
fi
if (( INCLUDE_OPROJ && OPROJ_OUTPUT_ROWS % PROJECTION_SLICE_ROWS != 0 )); then
  echo "PROJECTION_SLICE_ROWS must divide OPROJ_OUTPUT_ROWS." >&2
  exit 2
fi
if (( INCLUDE_ROPE && (QPROJ_OUTPUT_ROWS != Q_HEADS * HEAD_DIM || KPROJ_OUTPUT_ROWS != KV_HEADS * HEAD_DIM) )); then
  echo "q/k norm+RoPE requires QPROJ_OUTPUT_ROWS=Q_HEADS*128 and KPROJ_OUTPUT_ROWS=KV_HEADS*128." >&2
  exit 2
fi
if (( INCLUDE_ATTENTION && VPROJ_OUTPUT_ROWS != KV_HEADS * HEAD_DIM )); then
  echo "attention_core requires VPROJ_OUTPUT_ROWS=KV_HEADS*128." >&2
  exit 2
fi
if (( INCLUDE_ATTENTION && (TOKEN_COUNT % QUERY_TILE_ROWS != 0 || TOKEN_COUNT % KEY_TILE_ROWS != 0) )); then
  echo "QUERY_TILE_ROWS and KEY_TILE_ROWS must divide TOKEN_COUNT for attention_core." >&2
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
OPROJ_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_o_proj.mlir"
ROPE_TABLE_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_rope_table.mlir"
Q_NORM_ROPE_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_q_norm_rope.mlir"
K_NORM_ROPE_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_k_norm_rope.mlir"
ATTENTION_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_attention_core.mlir"
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
    --sequence-length 1 \
    --output-rows "$PROJECTION_SLICE_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
fi
if (( INCLUDE_QKV )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage k_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length 1 \
    --output-rows "$PROJECTION_SLICE_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage v_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length 1 \
    --output-rows "$PROJECTION_SLICE_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
fi
if (( INCLUDE_ROPE )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage rope_table \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT"
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage q_norm_rope \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --q-heads "$Q_HEADS" \
    --kv-heads "$KV_HEADS"
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage k_norm_rope \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --q-heads "$Q_HEADS" \
    --kv-heads "$KV_HEADS"
fi
if (( INCLUDE_ATTENTION )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage attention_core \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --query-tile-rows "$QUERY_TILE_ROWS" \
    --key-tile-rows "$KEY_TILE_ROWS" \
    --q-heads "$Q_HEADS" \
    --kv-heads "$KV_HEADS"
fi
if (( INCLUDE_OPROJ )); then
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage o_proj \
    --gguf "$GGUF_PATH" \
    --sequence-length 1 \
    --output-rows "$PROJECTION_SLICE_ROWS" \
    --output-tile-rows "$OUTPUT_TILE_ROWS"
fi

check_contains "$EMBED_MLIR" 'scf\.parallel' 'embed_tokens explicit tile loop'
check_contains "$EMBED_MLIR" 'memref\.subview' 'embed_tokens tile subview'
check_contains "$EMBED_MLIR" 'memref\.copy' 'embed_tokens copy before AIR DMA lowering'
check_contains "$NORM_MLIR" 'scf\.parallel' 'input_layernorm explicit tile loop'
check_contains "$NORM_MLIR" 'func\.call @rms_norm_tile' 'input_layernorm external RMSNorm tile call'
check_contains "$NORM_MLIR" 'link_with = "rms_norm\.o"' 'input_layernorm external RMSNorm link object'

lower_air_fixture_to_dma "$EMBED_MLIR" "${STEM}_embed_tokens"
EMBED_DMA="$DMA_IR"
lower_air_fixture_to_dma "$NORM_MLIR" "${STEM}_input_layernorm"
NORM_DMA="$DMA_IR"
HERD_ROWS=1
HERD_COLS="$BLOCKS_PER_ROW"
compile_air_dma_fixture "$EMBED_DMA" "${STEM}_embed_tokens"
EMBED_AIE="$AIE_IR"
HERD_ROWS="$TOKEN_COUNT"
HERD_COLS=1
compile_air_dma_fixture "$NORM_DMA" "${STEM}_input_layernorm"
NORM_AIE="$AIE_IR"
if (( INCLUDE_QPROJ )); then
  HERD_ROWS=1
  HERD_COLS="$(projection_parallel_tiles "$PROJECTION_SLICE_ROWS")"
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
    HERD_COLS="$(projection_parallel_tiles "$PROJECTION_SLICE_ROWS")"
    check_contains "$KPROJ_MLIR" 'air\.launch' 'k_proj official AIR launch'
    check_contains "$KPROJ_MLIR" 'air\.herd' 'k_proj official AIR herd'
    check_contains "$KPROJ_MLIR" 'air\.dma_memcpy_nd' 'k_proj explicit AIR DMA'
    check_contains "$KPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${PROJ_WEIGHT_WORDS}xi32, 2" 'k_proj packed Q4_K weight L1 tile'
    check_contains "$KPROJ_MLIR" 'func\.call @q4k_linear_tile' 'k_proj external Q4_K tile kernel call'
    check_contains "$KPROJ_MLIR" 'link_with = "q4k_linear\.o"' 'k_proj external Q4_K link object'
    compile_air_dma_fixture "$KPROJ_MLIR" "${STEM}_k_proj"
    KPROJ_AIE="$AIE_IR"
    VPROJ_WEIGHT_WORDS="$((BLOCKS_PER_ROW * 106))"
    HERD_COLS="$(projection_parallel_tiles "$PROJECTION_SLICE_ROWS")"
    check_contains "$VPROJ_MLIR" 'air\.launch' 'v_proj official AIR launch'
    check_contains "$VPROJ_MLIR" 'air\.herd' 'v_proj official AIR herd'
    check_contains "$VPROJ_MLIR" 'air\.dma_memcpy_nd' 'v_proj explicit AIR DMA'
    check_contains "$VPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${VPROJ_WEIGHT_WORDS}xi32, 2" 'v_proj packed Q6_K weight L1 tile'
    check_contains "$VPROJ_MLIR" 'func\.call @q6k_linear_tile' 'v_proj external Q6_K tile kernel call'
    check_contains "$VPROJ_MLIR" 'link_with = "q6k_linear\.o"' 'v_proj external Q6_K link object'
    compile_air_dma_fixture "$VPROJ_MLIR" "${STEM}_v_proj"
    VPROJ_AIE="$AIE_IR"
    if (( INCLUDE_ROPE )); then
      HERD_ROWS="$TOKEN_COUNT"
      HERD_COLS=1
      check_contains "$ROPE_TABLE_MLIR" 'air\.launch' 'rope_table official AIR launch'
      check_contains "$ROPE_TABLE_MLIR" 'air\.herd' 'rope_table official AIR herd'
      check_contains "$ROPE_TABLE_MLIR" 'air\.dma_memcpy_nd' 'rope_table explicit AIR DMA'
      check_contains "$ROPE_TABLE_MLIR" 'func\.call @rope_table_tile' 'rope_table external tile kernel call'
      check_contains "$ROPE_TABLE_MLIR" 'link_with = "rope_table\.o"' 'rope_table external link object'
      compile_air_dma_fixture "$ROPE_TABLE_MLIR" "${STEM}_rope_table"
      ROPE_TABLE_AIE="$AIE_IR"

      HERD_ROWS=1
      HERD_COLS=1
      check_contains "$Q_NORM_ROPE_MLIR" 'air\.launch' 'q_norm_rope official AIR launch'
      check_contains "$Q_NORM_ROPE_MLIR" 'air\.herd' 'q_norm_rope official AIR herd'
      check_contains "$Q_NORM_ROPE_MLIR" 'air\.channel\.put' 'q_norm_rope explicit AIR channel put'
      check_contains "$Q_NORM_ROPE_MLIR" 'air\.channel\.get' 'q_norm_rope explicit AIR channel get'
      check_contains "$Q_NORM_ROPE_MLIR" 'func\.call @rms_norm_rope_tile' 'q_norm_rope external tile kernel call'
      check_contains "$Q_NORM_ROPE_MLIR" 'link_with = "rms_norm_rope\.o"' 'q_norm_rope external link object'
      compile_air_dma_fixture "$Q_NORM_ROPE_MLIR" "${STEM}_q_norm_rope"
      Q_NORM_ROPE_AIE="$AIE_IR"

      check_contains "$K_NORM_ROPE_MLIR" 'air\.launch' 'k_norm_rope official AIR launch'
      check_contains "$K_NORM_ROPE_MLIR" 'air\.herd' 'k_norm_rope official AIR herd'
      check_contains "$K_NORM_ROPE_MLIR" 'air\.channel\.put' 'k_norm_rope explicit AIR channel put'
      check_contains "$K_NORM_ROPE_MLIR" 'air\.channel\.get' 'k_norm_rope explicit AIR channel get'
      check_contains "$K_NORM_ROPE_MLIR" 'func\.call @rms_norm_rope_tile' 'k_norm_rope external tile kernel call'
      check_contains "$K_NORM_ROPE_MLIR" 'link_with = "rms_norm_rope\.o"' 'k_norm_rope external link object'
      compile_air_dma_fixture "$K_NORM_ROPE_MLIR" "${STEM}_k_norm_rope"
      K_NORM_ROPE_AIE="$AIE_IR"

      if (( INCLUDE_ATTENTION )); then
        HERD_ROWS=1
        HERD_COLS=1
        check_contains "$ATTENTION_MLIR" 'air\.launch' 'attention_core official AIR launch'
        check_contains "$ATTENTION_MLIR" 'air\.herd' 'attention_core official AIR herd'
        check_contains "$ATTENTION_MLIR" 'air\.channel\.put' 'attention_core explicit AIR channel put'
        check_contains "$ATTENTION_MLIR" 'air\.channel\.get' 'attention_core explicit AIR channel get'
        check_contains "$ATTENTION_MLIR" 'scf\.for %q_block' 'attention_core q-block loop'
        check_contains "$ATTENTION_MLIR" 'scf\.for %kv_block' 'attention_core kv-block loop'
        check_contains "$ATTENTION_MLIR" 'func\.call @attention_core_tile' 'attention_core external tile kernel call'
        check_contains "$ATTENTION_MLIR" 'link_with = "attention_core\.o"' 'attention_core external link object'
        compile_air_dma_fixture "$ATTENTION_MLIR" "${STEM}_attention_core"
        ATTENTION_AIE="$AIE_IR"
      fi
      if (( INCLUDE_OPROJ )); then
        HERD_ROWS=1
        HERD_COLS="$(projection_parallel_tiles "$PROJECTION_SLICE_ROWS")"
        OPROJ_BLOCKS_PER_ROW="$((QPROJ_OUTPUT_ROWS / 256))"
        OPROJ_WEIGHT_WORDS="$((OPROJ_BLOCKS_PER_ROW * 38))"
        check_contains "$OPROJ_MLIR" 'air\.launch' 'o_proj official AIR launch'
        check_contains "$OPROJ_MLIR" 'air\.herd' 'o_proj official AIR herd'
        check_contains "$OPROJ_MLIR" 'air\.dma_memcpy_nd' 'o_proj explicit AIR DMA'
        check_contains "$OPROJ_MLIR" "memref\\.alloc\\(\\) : memref<${OUTPUT_TILE_ROWS}x${OPROJ_WEIGHT_WORDS}xi32, 2" 'o_proj packed Q4_K weight L1 tile'
        check_contains "$OPROJ_MLIR" 'func\.call @q4k_linear_tile' 'o_proj external Q4_K tile kernel call'
        check_contains "$OPROJ_MLIR" 'link_with = "q4k_linear\.o"' 'o_proj external Q4_K link object'
        compile_air_dma_fixture "$OPROJ_MLIR" "${STEM}_o_proj"
        OPROJ_AIE="$AIE_IR"
      fi
    fi
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
    --output-rows "$QPROJ_OUTPUT_ROWS"
    --qproj-output-rows "$QPROJ_OUTPUT_ROWS"
    --output-tile-rows "$OUTPUT_TILE_ROWS"
  )
fi
if (( INCLUDE_QKV )); then
  RUNNER_ARGS+=(
    --kproj-aie-mlir "$KPROJ_AIE"
    --vproj-aie-mlir "$VPROJ_AIE"
    --kproj-output-rows "$KPROJ_OUTPUT_ROWS"
    --vproj-output-rows "$VPROJ_OUTPUT_ROWS"
  )
fi
if (( INCLUDE_ROPE )); then
  RUNNER_ARGS+=(
    --rope-table-aie-mlir "$ROPE_TABLE_AIE"
    --q-norm-rope-aie-mlir "$Q_NORM_ROPE_AIE"
    --k-norm-rope-aie-mlir "$K_NORM_ROPE_AIE"
    --start-position "$START_POSITION"
    --q-heads "$Q_HEADS"
    --kv-heads "$KV_HEADS"
  )
fi
if (( INCLUDE_ATTENTION )); then
  RUNNER_ARGS+=(
    --attention-aie-mlir "$ATTENTION_AIE"
    --query-tile-rows "$QUERY_TILE_ROWS"
    --key-tile-rows "$KEY_TILE_ROWS"
  )
fi
if (( INCLUDE_OPROJ )); then
  RUNNER_ARGS+=(
    --oproj-aie-mlir "$OPROJ_AIE"
    --oproj-output-rows "$OPROJ_OUTPUT_ROWS"
  )
fi

"$UV" run --no-sync python "${RUNNER_ARGS[@]}"
