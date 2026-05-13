#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/verify-air-common.sh" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/air-env.sh"

command -v rg >/dev/null 2>&1 || {
  echo "rg is required for verification checks." >&2
  return 1
}

OUT_DIR="${OUT_DIR:-$ROOT_DIR/examples/amd_aie_experiments/generated}"
DEVICE="${AIR_DEVICE:-npu1}"
ROW_OFFSET="${AIR_ROW_OFFSET:-2}"
COL_OFFSET="${AIR_COL_OFFSET:-0}"
HERD_ROWS="${AIR_HERD_ROWS:-1}"
HERD_COLS="${AIR_HERD_COLS:-1}"

AIR_OPT_DIAG_FLAGS=()
if [[ "${AIR_VERBOSE_DIAGNOSTICS:-0}" != "1" ]]; then
  AIR_OPT_DIAG_FLAGS=(--mlir-print-op-on-diagnostic=false --mlir-disable-diagnostic-notes)
fi

mkdir -p "$OUT_DIR"

check_contains() {
  local file="$1"
  local pattern="$2"
  local label="$3"

  if ! rg -q "$pattern" "$file"; then
    echo "FAIL: missing $label in $file" >&2
    exit 1
  fi
  echo "ok: $label"
}

check_count_ge() {
  local file="$1"
  local pattern="$2"
  local expected="$3"
  local label="$4"
  local actual

  actual="$(rg -c "$pattern" "$file" || true)"
  if (( actual < expected )); then
    echo "FAIL: expected at least $expected $label in $file, got $actual" >&2
    exit 1
  fi
  echo "ok: $label count $actual >= $expected"
}

compile_air_fixture() {
  local input="$1"
  local stem="$2"

  DMA_IR="$OUT_DIR/$stem.dma.mlir"
  CHANNEL_IR="$OUT_DIR/$stem.channel.mlir"
  AIE_IR="$OUT_DIR/$stem.aie.mlir"

  air-opt "${AIR_OPT_DIAG_FLAGS[@]}" "$input" \
    --air-par-to-launch='depth=0 has-air-segment=true' \
    --air-par-to-herd='depth=0' \
    --scf-forall-to-for \
    --air-copy-to-dma \
    --canonicalize \
    --cse \
    -o "$DMA_IR"

  check_contains "$DMA_IR" 'air\.herd' 'air.herd after parallel-to-herd'
  check_contains "$DMA_IR" 'air\.dma_memcpy_nd' 'air.dma_memcpy_nd after copy lowering'

  air-opt "${AIR_OPT_DIAG_FLAGS[@]}" "$DMA_IR" \
    --air-dependency \
    --air-dma-to-channel \
    --canonicalize \
    --cse \
    --air-place-herds="num-rows=$HERD_ROWS num-cols=$HERD_COLS row-anchor=$ROW_OFFSET col-anchor=$COL_OFFSET" \
    -o "$CHANNEL_IR"

  check_contains "$CHANNEL_IR" 'air\.channel\.put' 'air.channel.put after channel lowering'
  check_contains "$CHANNEL_IR" 'air\.channel\.get' 'air.channel.get after channel lowering'

  air-opt "${AIR_OPT_DIAG_FLAGS[@]}" "$CHANNEL_IR" \
    --air-to-aie="device=$DEVICE row-offset=$ROW_OFFSET col-offset=$COL_OFFSET" \
    --canonicalize \
    --cse \
    -o "$AIE_IR"

  check_contains "$AIE_IR" "aie\\.device\\($DEVICE\\)" "aie.device($DEVICE)"
  check_contains "$AIE_IR" 'aie\.core' 'aie.core'

  echo "Generated:"
  echo "  $DMA_IR"
  echo "  $CHANNEL_IR"
  echo "  $AIE_IR"
}
