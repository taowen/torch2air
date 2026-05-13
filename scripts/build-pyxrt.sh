#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${VENV_DIR:=$ROOT_DIR/.venv}"
: "${XRT_DIR:=${XILINX_XRT:-/opt/xilinx/xrt}}"
: "${IREE_AMD_AIE_DIR:=$(realpath "$ROOT_DIR/../iree-amd-aie" 2>/dev/null || true)}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Python environment not found at $VENV_DIR." >&2
  echo "Run: scripts/install-air-tools.sh" >&2
  exit 1
fi

if [[ ! -d "$XRT_DIR/include" || ! -d "$XRT_DIR/lib" ]]; then
  echo "XRT install not found at $XRT_DIR." >&2
  echo "Set XRT_DIR or source /opt/xilinx/xrt/setup.sh first." >&2
  exit 1
fi

XRT_SOURCE_DIR="${XRT_SOURCE_DIR:-$IREE_AMD_AIE_DIR/third_party/XRT}"
XRT_PYXRT_SRC="$XRT_SOURCE_DIR/src/python/pybind11/src/pyxrt.cpp"
XRT_CORE_INCLUDE="$XRT_SOURCE_DIR/src/runtime_src/core/include"

if [[ ! -f "$XRT_PYXRT_SRC" ]]; then
  echo "XRT pyxrt source not found: $XRT_PYXRT_SRC" >&2
  echo "Set XRT_SOURCE_DIR to a checkout containing src/python/pybind11/src/pyxrt.cpp." >&2
  exit 1
fi

SYSROOT_DIR="${SYSROOT_DIR:-$IREE_AMD_AIE_DIR/.cache/iree-amd-aie/sysroot}"
UUID_INCLUDE_DIR="${UUID_INCLUDE_DIR:-$SYSROOT_DIR/usr/include}"
UUID_LIB="${UUID_LIB:-$SYSROOT_DIR/usr/lib64/libuuid.so}"

if [[ ! -f "$UUID_INCLUDE_DIR/uuid/uuid.h" ]]; then
  echo "uuid/uuid.h not found under $UUID_INCLUDE_DIR." >&2
  echo "Set UUID_INCLUDE_DIR to a directory containing uuid/uuid.h." >&2
  exit 1
fi

if [[ ! -f "$UUID_LIB" ]]; then
  echo "libuuid.so not found at $UUID_LIB." >&2
  echo "Set UUID_LIB to a development libuuid.so path." >&2
  exit 1
fi

BOOST_INCLUDE_DIR="${BOOST_INCLUDE_DIR:-}"
if [[ -z "$BOOST_INCLUDE_DIR" ]] && command -v brew >/dev/null 2>&1; then
  if brew --prefix boost >/dev/null 2>&1; then
    BOOST_INCLUDE_DIR="$(brew --prefix boost)/include"
  fi
fi

if [[ -z "$BOOST_INCLUDE_DIR" || ! -f "$BOOST_INCLUDE_DIR/boost/format.hpp" ]]; then
  echo "Boost headers not found." >&2
  echo "Install with: brew install boost" >&2
  echo "or set BOOST_INCLUDE_DIR to a directory containing boost/format.hpp." >&2
  exit 1
fi

http_proxy="${http_proxy:-http://127.0.0.1:7890}" \
https_proxy="${https_proxy:-http://127.0.0.1:7890}" \
  uv pip install --python "$VENV_DIR/bin/python" 'pybind11>=2.10'

SITE_DIR="$("$VENV_DIR/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
EXT_SUFFIX="$("$VENV_DIR/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
PYBIND_INCLUDES="$("$VENV_DIR/bin/python" -m pybind11 --includes)"
OUT="$SITE_DIR/pyxrt$EXT_SUFFIX"

g++ -O3 -Wall -shared -std=c++17 -fPIC $PYBIND_INCLUDES \
  -I"$XRT_DIR/include" \
  -I"$XRT_CORE_INCLUDE" \
  -I"$UUID_INCLUDE_DIR" \
  -I"$BOOST_INCLUDE_DIR" \
  "$XRT_PYXRT_SRC" \
  -L"$XRT_DIR/lib" -Wl,-rpath,"$XRT_DIR/lib" \
  -lxrt_coreutil "$UUID_LIB" -lpthread \
  -o "$OUT"

if [[ -f "$XRT_DIR/python/pyxrt.pyi" ]]; then
  cp "$XRT_DIR/python/pyxrt.pyi" "$SITE_DIR/pyxrt.pyi"
fi

source "$XRT_DIR/setup.sh" >/tmp/torch2air-xrt-setup.log
"$VENV_DIR/bin/python" - <<'PY'
import pyxrt
print(pyxrt.__file__)
print("devices", pyxrt.enumerate_devices())
PY
