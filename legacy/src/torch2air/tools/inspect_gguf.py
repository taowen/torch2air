from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch2air.weights.gguf import load_gguf_index, read_tensor_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GGUF packed tensor metadata.")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--format", default="Q4_K")
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    index = load_gguf_index(args.gguf)
    matching = [entry for entry in index.tensors.values() if entry.ggml_type == args.format]
    if not matching:
        raise SystemExit(f"No {args.format} tensors found in {args.gguf}")
    selected = max(matching, key=lambda entry: entry.nbytes)
    prefix = read_tensor_prefix(index.path, selected, size=64)
    manifest = {
        "gguf_path": str(index.path),
        "version": index.version,
        "alignment": index.alignment,
        "tensor_count": len(index.tensors),
        "matching_tensor_count": len(matching),
        "selected_tensor": selected.to_json(),
        "first_64_bytes_hex": prefix.hex(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["selected_tensor"], indent=2, sort_keys=True))
    print(f"manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
