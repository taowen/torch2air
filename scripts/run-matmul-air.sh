#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TORCH2AIR_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
IREE_AMD_AIE_ROOT=$(cd -- "${TORCH2AIR_ROOT}/../.." && pwd)

if [[ -f "${IREE_AMD_AIE_ROOT}/iree-install-rocm/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${IREE_AMD_AIE_ROOT}/iree-install-rocm/env.sh"
fi

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/var/home/taowen/projects/torch2vk/.venv/bin/python" ]]; then
    PYTHON="/var/home/taowen/projects/torch2vk/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi

export PYTHONPATH="${TORCH2AIR_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" -m torch2air run-demo --out-dir "${TORCH2AIR_ROOT}/examples/generated"
