#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/npu-common.sh" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${XRT_DIR:=/opt/xilinx/xrt}"
: "${AIE_TARGET:=aie2p}"
: "${NPU_WORK_ROOT:=$ROOT_DIR/.cache/npu-spikes}"

source "$XRT_DIR/setup.sh" >/tmp/torch2air-xrt-setup.log
source "$ROOT_DIR/scripts/air-env.sh"

export XRT_HACK_UNSECURE_LOADING_XCLBIN="${XRT_HACK_UNSECURE_LOADING_XCLBIN:-1}"

setup_npu_python_shim() {
  local shim_dir="$NPU_WORK_ROOT/bin"
  mkdir -p "$shim_dir"
  printf '#!/usr/bin/env bash\nexec "%s/.venv/bin/python" "$@"\n' "$ROOT_DIR" > "$shim_dir/python3"
  chmod +x "$shim_dir/python3"
  if [[ ":$PATH:" != *":$shim_dir:"* ]]; then
    export PATH="$shim_dir:$PATH"
  fi
}

check_npu_device() {
  "$ROOT_DIR/.venv/bin/python" - <<'PY'
import pyxrt

devices = pyxrt.enumerate_devices()
print("pyxrt", pyxrt.__file__)
print("devices", devices)
if devices < 1:
    raise SystemExit("No XRT NPU device found")
device = pyxrt.device(0)
print("device", device.get_info(pyxrt.xrt_info_device.name))
PY
}

run_npu_make() {
  local label="$1"
  local makefile="$2"
  local target="$3"
  shift 3

  local work_dir="$NPU_WORK_ROOT/$label"
  mkdir -p "$work_dir"
  setup_npu_python_shim

  echo "NPU run: $label"
  check_npu_device
  make -f "$makefile" -C "$work_dir" clean "$@"
  make -f "$makefile" -C "$work_dir" "$target" \
    PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" \
    AIE_TARGET="$AIE_TARGET" \
    "$@"
}
