#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from air.ir import *  # noqa: E402,F403
from air.dialects.air import *  # noqa: E402,F403
from air.dialects.func import CallOp, FuncOp  # noqa: E402
from air.dialects.memref import AllocOp, DeallocOp  # noqa: E402
from air.backend.xrt_runner import XRTRunner, type_mapper  # noqa: E402

from torch2air.weights.gguf import (  # noqa: E402
    dequantize_q4_k_blocks,
    load_gguf_index,
    read_tensor_bytes,
)


@module_builder
def build_module(rows: int, blocks_per_row: int):
    row_words = blocks_per_row * 36
    k = blocks_per_row * 256
    weight_words = rows * row_words
    xrt_u32 = type_mapper(np.uint32)
    xrt_f32 = type_mapper(np.float32)

    l3_weight_ty = MemRefType.get([weight_words], xrt_u32)
    l3_x_ty = MemRefType.get([k], xrt_f32)
    l3_out_ty = MemRefType.get([rows], xrt_f32)
    l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_weight_ty = MemRefType.get([weight_words], xrt_u32, memory_space=l1_space)
    l1_x_ty = MemRefType.get([k], xrt_f32, memory_space=l1_space)
    l1_out_ty = MemRefType.get([rows], xrt_f32, memory_space=l1_space)

    q4k_func = FuncOp(
        "q4k_matvec_f32",
        ([l1_weight_ty, l1_x_ty, l1_out_ty], []),
        visibility="private",
    )
    q4k_func.attributes["link_with"] = StringAttr.get("q4k_matvec.o")
    q4k_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(l3_weight_ty, l3_x_ty, l3_out_ty)
    def q4k_matvec(weight, x, output):
        @launch(operands=[weight, x, output])
        def launch_body(weight_l, x_l, output_l):
            @segment(name="seg", operands=[weight_l, x_l, output_l])
            def segment_body(weight_s, x_s, output_s):
                @herd(
                    name="q4k_matvec_herd",
                    sizes=[1, 1],
                    operands=[weight_s, x_s, output_s],
                    link_with="q4k_matvec.o",
                )
                def herd_body(_tx, _ty, _sx, _sy, weight_h, x_h, output_h):
                    weight_l1 = AllocOp(l1_weight_ty, [], [])
                    x_l1 = AllocOp(l1_x_ty, [], [])
                    output_l1 = AllocOp(l1_out_ty, [], [])

                    dma_memcpy_nd(weight_l1, weight_h)
                    dma_memcpy_nd(x_l1, x_h)
                    CallOp(q4k_func, [weight_l1, x_l1, output_l1])
                    dma_memcpy_nd(output_h, output_l1)

                    DeallocOp(weight_l1)
                    DeallocOp(x_l1)
                    DeallocOp(output_l1)


def select_q4k_tensor(gguf_path: Path, tensor_name: str | None):
    index = load_gguf_index(gguf_path)
    if tensor_name is not None:
        selected = index.tensors[tensor_name]
        if selected.ggml_type != "Q4_K":
            raise ValueError(f"{tensor_name} is {selected.ggml_type}, not Q4_K")
    else:
        q4k_tensors = [entry for entry in index.tensors.values() if entry.ggml_type == "Q4_K"]
        if not q4k_tensors:
            raise ValueError(f"No Q4_K tensors found in {gguf_path}")
        selected = max(q4k_tensors, key=lambda entry: entry.nbytes)
    return index, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real GGUF Q4_K matvec on the NPU.")
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("/var/home/taowen/projects/torch2vk/dist/llama_cpp_qwen3/qwen3-0.6b-q4_k_m.gguf"),
    )
    parser.add_argument("--tensor", default="token_embd.weight")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--manifest", type=Path, default=Path("q4k_matvec_manifest.json"))
    parser.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    index, selected = select_q4k_tensor(args.gguf, args.tensor)
    if selected.physical_dtype != "uint32" or len(selected.physical_shape) != 2:
        raise ValueError(f"Expected rank-2 uint32 Q4_K physical tensor, got {selected}")
    row_words = int(selected.physical_shape[1])
    if row_words % 36 != 0:
        raise ValueError(f"Q4_K row word width must be a multiple of 36, got {row_words}")
    blocks_per_row = row_words // 36
    if args.rows <= 0 or args.rows > int(selected.physical_shape[0]):
        raise ValueError(f"Invalid row count {args.rows} for tensor with {selected.physical_shape[0]} rows")

    payload = read_tensor_bytes(args.gguf, selected, offset=0, size=args.rows * row_words * 4)
    weights_words = np.frombuffer(payload, dtype=np.uint32).copy()
    raw_blocks = np.frombuffer(payload, dtype=np.uint8).reshape(args.rows * blocks_per_row, 144)
    k = blocks_per_row * 256
    x = np.linspace(-1.0, 1.0, k, dtype=np.float32)
    dequant = dequantize_q4_k_blocks(raw_blocks).reshape(args.rows, k)
    expected = (dequant.astype(np.float32) * x.reshape(1, k)).sum(axis=1, dtype=np.float32)

    manifest = {
        "gguf_path": str(args.gguf),
        "selected_tensor": selected.to_json(),
        "rows": args.rows,
        "blocks_per_row": blocks_per_row,
        "k": k,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "expected_first_4": [float(v) for v in expected[:4]],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    module = build_module(args.rows, blocks_per_row)
    if args.print_module_only:
        print(module)
        return 0

    print(f"GGUF tensor {selected.name}")
    print(f"rows {args.rows} k {k} blocks_per_row {blocks_per_row}")
    print(f"payload_sha256 {manifest['payload_sha256']}")

    runner = XRTRunner(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="q4k_matvec",
        runtime_loop_tiling_sizes=[4, 4],
    )
    return int(
        runner.run_test(
            module,
            inputs=[weights_words, x],
            expected_outputs=[expected],
            rtol=5e-3,
            atol=5e-2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
