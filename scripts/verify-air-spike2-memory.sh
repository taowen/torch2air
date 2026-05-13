#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_memory_space_contract.mlir}"

compile_air_fixture "$INPUT" air_memory_space_contract
check_contains "$INPUT" 'memref<[^>]+, 1>' 'L2 memory-space memrefs in source'
check_contains "$INPUT" 'memref<[^>]+, 2>' 'L1 memory-space memrefs in source'
check_count_ge "$DMA_IR" 'air\.dma_memcpy_nd' 6 'DMA ops across L3/L2/L1'

run_npu_make \
  spike2-memory-space \
  "$ROOT_DIR/mlir-air/programming_examples/eltwise_add_with_l2/Makefile" \
  run \
  OUTPUT_FORMAT=xclbin
