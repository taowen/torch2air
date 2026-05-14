#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${AIR_DEVICE:=npu2_4col}"
: "${AIR_ROW_OFFSET:=2}"
: "${AIR_COL_OFFSET:=0}"
: "${AIR_RUN_TIMEOUT:=90s}"

source "$ROOT_DIR/scripts/npu-common.sh"

WORK_DIR="${WORK_DIR:-$NPU_WORK_ROOT/mlir-air-official-external-kernel}"
MLIR="$ROOT_DIR/examples/amd_aie_experiments/air_official_external_kernel.mlir"
KERNEL_CC="$ROOT_DIR/mlir-air/test/xrt/05_extern_func/chess/beefmaker_kernel.cc"
TARGET_TRIPLE="$AIE_TARGET-none-unknown-elf"
AIEOPT_DIR="$(realpath "$(dirname "$(command -v aie-opt)")/..")"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

check_npu_device

"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target="$TARGET_TRIPLE" \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-unused-command-line-argument -DNDEBUG -I "$AIEOPT_DIR/include" \
  -c "$KERNEL_CC" -o beefmaker_kernel.o

air-opt "$MLIR" \
  -air-dma-to-channel -canonicalize -air-dependency \
  -air-to-aie="device=$AIR_DEVICE row-offset=$AIR_ROW_OFFSET col-offset=$AIR_COL_OFFSET" \
  -air-to-std -symbol-dce -airrt-to-npu -canonicalize -cse \
  -o aie.mlir

aiecc --no-aiesim --no-xchesscc --no-xbridge --peano "$PEANO_INSTALL_DIR" \
  --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
  --xclbin-name=aie.xclbin --npu-insts-name=insts.bin aie.mlir

timeout "$AIR_RUN_TIMEOUT" "$UV" run --no-sync python \
  "$ROOT_DIR/mlir-air/test/xrt/05_extern_func/run.py"
