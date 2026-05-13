#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_dma_ordering_channels.mlir}"

compile_air_fixture "$INPUT" air_dma_ordering_channels
check_count_ge "$DMA_IR" 'air\.dma_memcpy_nd' 4 'input/input/input/output DMA ops'
check_count_ge "$CHANNEL_IR" 'air\.channel\.put' 4 'channel put ops'
check_count_ge "$CHANNEL_IR" 'air\.channel\.get' 4 'channel get ops'

run_npu_make \
  spike3-dma-channels \
  "$ROOT_DIR/mlir-air/programming_examples/passthrough/passthrough_channel/Makefile" \
  run \
  OUTPUT_FORMAT=xclbin
