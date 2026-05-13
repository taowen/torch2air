#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_flash_attention_skeleton.mlir}"

compile_air_fixture "$INPUT" air_flash_attention_skeleton
check_contains "$INPUT" 'q_l1' 'Q resident L1 buffer in source'
check_count_ge "$DMA_IR" 'air\.dma_memcpy_nd' 5 'Q/K/V staged DMA ops'
check_count_ge "$INPUT" 'arith\.addf' 2 'staged score/accumulator updates in source'

run_npu_make \
  spike6-flash-attention \
  "$ROOT_DIR/mlir-air/programming_examples/flash_attention/kernel_fusion_based/Makefile" \
  run \
  LK=512 \
  LKP=64 \
  LQ=512 \
  LQP=256 \
  DK=64 \
  DV=64 \
  NUM_HEADS=2 \
  NUM_KV_HEADS=2 \
  EXTRA_PY_FLAGS="--output-format xclbin"
