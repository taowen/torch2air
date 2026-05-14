#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
UV="${UV:-uv}"
source "$ROOT_DIR/scripts/verify-air-common.sh"
source "$ROOT_DIR/scripts/npu-common.sh"

INPUT="${INPUT:-$ROOT_DIR/examples/amd_aie_experiments/air_q4k_linear_skeleton.mlir}"
GGUF_PATH="${GGUF_PATH:-/var/home/taowen/projects/torch2vk/dist/llama_cpp_qwen3/qwen3-0.6b-q4_k_m.gguf}"
MANIFEST="${MANIFEST:-$OUT_DIR/q4k_gguf_manifest.json}"
NPU_MANIFEST="${NPU_MANIFEST:-$OUT_DIR/q4k_matvec_npu_manifest.json}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$UV" run --no-sync python -m torch2air.tools.inspect_gguf \
  --gguf "$GGUF_PATH" \
  --format Q4_K \
  --manifest "$MANIFEST"

compile_air_fixture "$INPUT" air_q4k_linear_skeleton
check_contains "$INPUT" 'memref<8x36xi32' 'Q4_K packed uint32 tile shape'
check_contains "$MANIFEST" '"ggml_type": "Q4_K"' 'Q4_K tensor metadata in manifest'
check_count_ge "$DMA_IR" 'air\.dma_memcpy_nd' 3 'activation/packed-weight/output DMA ops'

source "$ROOT_DIR/scripts/npu-common.sh"
setup_npu_python_shim
check_npu_device
SPIKE5_NPU_DIR="$NPU_WORK_ROOT/spike5-real-q4k-matvec"
rm -rf "$SPIKE5_NPU_DIR"
mkdir -p "$SPIKE5_NPU_DIR/air_project"

AIEOPT_DIR="$(realpath "$(dirname "$(which aie-opt)")/..")"
WARNING_FLAGS=(
  -Wno-parentheses
  -Wno-attributes
  -Wno-macro-redefined
  -Wno-empty-body
  -Wno-unused-command-line-argument
)
"$PEANO_INSTALL_DIR/bin/clang++" \
  -O2 \
  -std=c++20 \
  --target=aie2p-none-unknown-elf \
  "${WARNING_FLAGS[@]}" \
  -DNDEBUG \
  -I "$AIEOPT_DIR/include" \
  -DROWS="${Q4K_ROWS:-64}" \
  -DBLOCKS_PER_ROW="${Q4K_BLOCKS_PER_ROW:-4}" \
  -c "$ROOT_DIR/examples/amd_aie_experiments/q4k_matvec.cc" \
  -o "$SPIKE5_NPU_DIR/air_project/q4k_matvec.o"

(
  cd "$SPIKE5_NPU_DIR"
  "$UV" run --no-sync python "$ROOT_DIR/examples/amd_aie_experiments/npu_q4k_matvec.py" \
    --gguf "$GGUF_PATH" \
    --tensor "${Q4K_TENSOR:-token_embd.weight}" \
    --rows "${Q4K_ROWS:-64}" \
    --manifest "$NPU_MANIFEST" \
    --output-format xclbin
)
check_contains "$NPU_MANIFEST" '"payload_sha256"' 'real GGUF payload checksum in NPU manifest'
check_contains "$NPU_MANIFEST" '"expected_first_4"' 'real Q4_K matvec host reference in NPU manifest'
