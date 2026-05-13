#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source scripts/air-env.sh" >&2
  exit 2
fi

_torch2air_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VENV_DIR:=$_torch2air_root/.venv}"
_torch2air_python="$VENV_DIR/bin/python"

if [[ ! -x "$_torch2air_python" ]]; then
  echo "AIR Python environment not found at $VENV_DIR." >&2
  echo "Run: scripts/install-air-tools.sh" >&2
  return 1
fi

_torch2air_pyver="$("$_torch2air_python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$_torch2air_pyver" != "3.12" ]]; then
  echo "Expected Python 3.12, got $_torch2air_pyver at $_torch2air_python." >&2
  return 1
fi

_torch2air_site="$("$_torch2air_python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

export MLIR_AIR_INSTALL_DIR="${MLIR_AIR_INSTALL_DIR:-$_torch2air_site/mlir_air}"
export MLIR_AIE_INSTALL_DIR="${MLIR_AIE_INSTALL_DIR:-$_torch2air_site/mlir_aie}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$_torch2air_site/llvm-aie}"

for _torch2air_required_dir in "$MLIR_AIR_INSTALL_DIR" "$MLIR_AIE_INSTALL_DIR" "$PEANO_INSTALL_DIR"; do
  if [[ ! -d "$_torch2air_required_dir" ]]; then
    echo "Missing AIR tool dependency directory: $_torch2air_required_dir" >&2
    echo "Run: scripts/install-air-tools.sh" >&2
    return 1
  fi
done

for _torch2air_bin_dir in "$MLIR_AIR_INSTALL_DIR/bin" "$MLIR_AIE_INSTALL_DIR/bin" "$PEANO_INSTALL_DIR/bin"; do
  if [[ -d "$_torch2air_bin_dir" && ":$PATH:" != *":$_torch2air_bin_dir:"* ]]; then
    export PATH="$_torch2air_bin_dir:$PATH"
  fi
done

for _torch2air_python_dir in "$MLIR_AIR_INSTALL_DIR/python" "$MLIR_AIE_INSTALL_DIR/python"; do
  if [[ -d "$_torch2air_python_dir" && ":${PYTHONPATH:-}:" != *":$_torch2air_python_dir:"* ]]; then
    export PYTHONPATH="$_torch2air_python_dir:${PYTHONPATH:-}"
  fi
done

for _torch2air_lib_dir in "$MLIR_AIR_INSTALL_DIR/lib" "$MLIR_AIE_INSTALL_DIR/lib"; do
  if [[ -d "$_torch2air_lib_dir" && ":${LD_LIBRARY_PATH:-}:" != *":$_torch2air_lib_dir:"* ]]; then
    export LD_LIBRARY_PATH="$_torch2air_lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done

unset _torch2air_root
unset _torch2air_python
unset _torch2air_pyver
unset _torch2air_site
unset _torch2air_required_dir
unset _torch2air_bin_dir
unset _torch2air_python_dir
unset _torch2air_lib_dir
