#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_direct_matmul_tiled.mlir}"

compile_air_fixture "$INPUT" air_direct_matmul_tiled

run_npu_make \
  spike1-direct-matmul \
  "$ROOT_DIR/mlir-air/programming_examples/matrix_multiplication/i8/Makefile" \
  run1x1
