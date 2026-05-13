#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

check_command() {
  local name=$1
  local value=$2
  if command -v "${value}" >/dev/null 2>&1; then
    printf 'ok   %-22s %s\n' "${name}" "$(command -v "${value}")"
  else
    printf 'miss %-22s %s\n' "${name}" "${value}"
    return 1
  fi
}

status=0

printf 'torch2air root       %s\n' "${TORCH2AIR_ROOT}"
printf 'iree-amd-aie root    %s\n' "${IREE_AMD_AIE_ROOT}"
printf 'output dir           %s\n' "${TORCH2AIR_OUT_DIR}"
printf 'python               %s\n' "${PYTHON}"
printf 'target device        %s\n' "${TORCH2AIR_TARGET_DEVICE}"
printf 'tile pipeline        %s\n' "${TORCH2AIR_TILE_PIPELINE}"
printf 'aie pipeline         %s\n' "${TORCH2AIR_LOWER_TO_AIE_PIPELINE}"
printf 'vmfb tile pipeline   %s\n' "${TORCH2AIR_VMFB_TILE_PIPELINE}"
printf 'vmfb aie pipeline    %s\n' "${TORCH2AIR_VMFB_LOWER_TO_AIE_PIPELINE}"
printf 'xrt-lite topology    %sx%s\n' "${XRT_LITE_N_CORE_ROWS}" "${XRT_LITE_N_CORE_COLS}"

"${PYTHON}" - <<'PY' || status=1
import importlib.util
import sys

print("python executable    " + sys.executable)
for name in ("torch", "numpy", "torch_mlir"):
    spec = importlib.util.find_spec(name)
    print(f"{'ok' if spec else 'miss'}   python package       {name}")
PY

check_command iree-compile "${IREE_COMPILE}" || status=1
check_command iree-run-module "${IREE_RUN_MODULE}" || status=1

if [[ -x "${IREE_AMD_AIE_ROOT}/llvm-aie/bin/opt" ]]; then
  printf 'ok   %-22s %s\n' peano-opt "${IREE_AMD_AIE_ROOT}/llvm-aie/bin/opt"
else
  printf 'miss %-22s %s\n' peano-opt "${IREE_AMD_AIE_ROOT}/llvm-aie/bin/opt"
  status=1
fi

if [[ -f "${IREE_AMD_AIE_ROOT}/iree-install/env.sh" ]]; then
  printf 'ok   %-22s %s\n' iree-env "${IREE_AMD_AIE_ROOT}/iree-install/env.sh"
else
  printf 'miss %-22s %s\n' iree-env "${IREE_AMD_AIE_ROOT}/iree-install/env.sh"
  status=1
fi

exit "${status}"
