#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_double_buffer_skeleton.mlir}"

compile_air_fixture "$INPUT" air_double_buffer_skeleton
check_contains "$INPUT" 'a_ping' 'ping activation buffer in source'
check_contains "$INPUT" 'a_pong' 'pong activation buffer in source'
check_count_ge "$DMA_IR" 'air\.dma_memcpy_nd' 5 'double-buffer DMA ops'
check_count_ge "$INPUT" 'linalg\.matmul' 2 'two compute stages in source'

run_npu_make \
  spike4-pipelined-dataflow \
  "$ROOT_DIR/mlir-air/programming_examples/herd_dataflow/Makefile" \
  run \
  AIE_TARGET=aie2p \
  OUTPUT_FORMAT=xclbin \
  M_SIZE=64 \
  N_SIZE=256
