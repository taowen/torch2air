#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV="${UV:-uv}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf}"
TOKEN_COUNT="${TOKEN_COUNT:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
QUERY_TILE_ROWS="${QUERY_TILE_ROWS:-4}"
KEY_TILE_ROWS="${KEY_TILE_ROWS:-4}"
Q_HEADS="${Q_HEADS:-1}"
KV_HEADS="${KV_HEADS:-1}"
ATTENTION_SEED="${ATTENTION_SEED:-1}"
ATTENTION_SCALE="${ATTENTION_SCALE:-0.25}"
NPU_WARMUP="${NPU_WARMUP:-0}"
NPU_ITERATIONS="${NPU_ITERATIONS:-1}"
ATTENTION_RTOL="${ATTENTION_RTOL:-0.05}"
ATTENTION_ATOL="${ATTENTION_ATOL:-0.05}"

source "$ROOT_DIR/scripts/npu-common.sh"
source "$ROOT_DIR/scripts/verify-air-common.sh"

WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/quantized-qwen3-attention-core-${TOKEN_COUNT}tok-q${Q_HEADS}-kv${KV_HEADS}}"

ATTENTION_AIES=()

for (( head = 0; head < Q_HEADS; head++ )); do
  "$ROOT_DIR/scripts/export-quantized-qwen3.sh" \
    --stage attention_core \
    --gguf "$GGUF_PATH" \
    --sequence-length "$TOKEN_COUNT" \
    --query-tile-rows "$QUERY_TILE_ROWS" \
    --key-tile-rows "$KEY_TILE_ROWS" \
    --q-heads "$Q_HEADS" \
    --kv-heads "$KV_HEADS" \
    --attention-head-index "$head"

  if (( Q_HEADS == 1 )); then
    ATTENTION_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_attention_core.mlir"
    ATTENTION_STEM="quantized_qwen3_attention_core_${TOKEN_COUNT}tok"
  else
    ATTENTION_MLIR="$ROOT_DIR/src/models/quantized_qwen3/generated/run_attention_core_head_${head}.mlir"
    ATTENTION_STEM="quantized_qwen3_attention_core_${TOKEN_COUNT}tok_head_${head}"
  fi

  check_contains "$ATTENTION_MLIR" 'air\.launch' "attention_core head $head official AIR launch"
  check_contains "$ATTENTION_MLIR" 'air\.herd' "attention_core head $head official AIR herd"
  check_contains "$ATTENTION_MLIR" 'air\.channel\.put' "attention_core head $head explicit AIR channel put"
  check_contains "$ATTENTION_MLIR" 'air\.channel\.get' "attention_core head $head explicit AIR channel get"
  check_contains "$ATTENTION_MLIR" 'scf\.for %q_block' "attention_core head $head q-block loop"
  check_contains "$ATTENTION_MLIR" 'scf\.for %kv_block' "attention_core head $head kv-block loop"
  check_contains "$ATTENTION_MLIR" 'func\.call @attention_core_tile' "attention_core head $head external tile kernel call"
  check_contains "$ATTENTION_MLIR" 'link_with = "attention_core\.o"' "attention_core head $head external link object"

  compile_air_dma_fixture "$ATTENTION_MLIR" "$ATTENTION_STEM"
  ATTENTION_AIES+=("$AIE_IR")
done

"$UV" run --no-sync python -m models.quantized_qwen3.run_attention_core \
  --aie-mlir "${ATTENTION_AIES[@]}" \
  --work-dir "$WORK_DIR" \
  --sequence-length "$TOKEN_COUNT" \
  --head-dim "$HEAD_DIM" \
  --query-tile-rows "$QUERY_TILE_ROWS" \
  --key-tile-rows "$KEY_TILE_ROWS" \
  --q-heads "$Q_HEADS" \
  --kv-heads "$KV_HEADS" \
  --seed "$ATTENTION_SEED" \
  --scale "$ATTENTION_SCALE" \
  --warmup "$NPU_WARMUP" \
  --iterations "$NPU_ITERATIONS" \
  --rtol "$ATTENTION_RTOL" \
  --atol "$ATTENTION_ATOL"
