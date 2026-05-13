#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SPIKES="${SPIKES:-1 2 3 4 5 6}"

run_spike() {
  local spike="$1"
  local script
  case "$spike" in
    1) script="verify-air-spike1.sh" ;;
    2) script="verify-air-spike2-memory.sh" ;;
    3) script="verify-air-spike3-dma-order.sh" ;;
    4) script="verify-air-spike4-double-buffer.sh" ;;
    5) script="verify-air-spike5-q4k-linear.sh" ;;
    6) script="verify-air-spike6-flash-attn.sh" ;;
    *)
      echo "Unknown spike: $spike" >&2
      echo "Use SPIKES=\"1 2 3 4 5 6\" to select spikes." >&2
      return 2
      ;;
  esac

  echo
  echo "=== Spike $spike: $script ==="
  "$ROOT_DIR/scripts/$script"
}

for spike in $SPIKES; do
  run_spike "$spike"
done

echo
echo "All requested AIR/NPU spikes passed: $SPIKES"
