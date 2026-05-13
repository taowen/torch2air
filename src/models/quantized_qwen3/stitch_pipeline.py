from __future__ import annotations

import argparse
from pathlib import Path

from torch2air.export.kernels.stitching import stitch_quantized_qwen3_embed_norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Stitch quantized_qwen3 AIR pipeline stages.")
    parser.add_argument("--embed-dma-mlir", type=Path, required=True)
    parser.add_argument("--norm-dma-mlir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--blocks-per-row", type=int, required=True)
    parser.add_argument("--function-name", default="run_pipeline_embed_norm")
    args = parser.parse_args()

    stitched = stitch_quantized_qwen3_embed_norm(
        embed_dma_mlir=args.embed_dma_mlir.read_text(),
        norm_dma_mlir=args.norm_dma_mlir.read_text(),
        sequence_length=args.sequence_length,
        blocks_per_row=args.blocks_per_row,
        function_name=args.function_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(stitched)
    print(f"stitched {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
