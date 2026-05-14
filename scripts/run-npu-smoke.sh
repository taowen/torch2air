#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${XRT_DIR:=/opt/xilinx/xrt}"
: "${AIE_TARGET:=aie2p}"
: "${NPU_SMOKE_DIR:=/tmp/torch2air-npu-smoke}"
: "${UV:=uv}"

source "$XRT_DIR/setup.sh" >/tmp/torch2air-xrt-setup.log
source "$ROOT_DIR/scripts/air-env.sh"

"$UV" run --no-sync python - <<'PY'
import pyxrt
print("pyxrt", pyxrt.__file__)
print("devices", pyxrt.enumerate_devices())
PY

rm -rf "$NPU_SMOKE_DIR"
mkdir -p "$NPU_SMOKE_DIR/bin"
cat > "$NPU_SMOKE_DIR/bin/python3" <<EOF
#!/usr/bin/env bash
cd "$ROOT_DIR"
exec "$UV" run --no-sync python "\$@"
EOF
chmod +x "$NPU_SMOKE_DIR/bin/python3"

export PATH="$NPU_SMOKE_DIR/bin:$PATH"
export XRT_HACK_UNSECURE_LOADING_XCLBIN="${XRT_HACK_UNSECURE_LOADING_XCLBIN:-1}"

cd "$NPU_SMOKE_DIR"
make -f "$ROOT_DIR/mlir-air/programming_examples/matrix_multiplication/i8/Makefile" clean
make -f "$ROOT_DIR/mlir-air/programming_examples/matrix_multiplication/i8/Makefile" \
  run1x1 \
  PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" \
  AIE_TARGET="$AIE_TARGET"
