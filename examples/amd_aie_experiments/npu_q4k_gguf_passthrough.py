#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from air.ir import *  # noqa: E402,F403
from air.dialects.air import *  # noqa: E402,F403
from air.dialects.memref import AllocOp, DeallocOp, load, store  # noqa: E402
from air.dialects.func import FuncOp  # noqa: E402
from air.dialects.scf import for_, yield_  # noqa: E402
from air.backend.xrt_runner import XRTRunner, type_mapper  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

from torch2air.weights.gguf import load_gguf_index, read_tensor_prefix  # noqa: E402


range_ = for_


@module_builder
def build_module(word_count: int, tile_words: int):
    assert word_count % tile_words == 0
    xrt_dtype = type_mapper(np.uint32)
    l3_ty = MemRefType.get([word_count], xrt_dtype)
    l1_ty = MemRefType.get(
        [tile_words],
        xrt_dtype,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    @FuncOp.from_py_func(l3_ty, l3_ty)
    def q4k_packed_passthrough(arg0, arg1):
        @launch(operands=[arg0, arg1])
        def launch_body(input_words, output_words):
            @segment(name="seg", operands=[input_words, output_words])
            def segment_body(seg_in, seg_out):
                @herd(name="q4k_weight_herd", sizes=[1, 1], operands=[seg_in, seg_out])
                def herd_body(_tx, _ty, _sx, _sy, herd_in, herd_out):
                    for offset in range_(0, word_count, tile_words):
                        tile_in = AllocOp(l1_ty, [], [])
                        tile_out = AllocOp(l1_ty, [], [])

                        dma_memcpy_nd(
                            tile_in,
                            herd_in,
                            src_offsets=[offset],
                            src_sizes=[tile_words],
                            src_strides=[1],
                        )

                        for j in range_(tile_words):
                            value = load(tile_in, [j])
                            store(value, tile_out, [j])
                            yield_([])

                        dma_memcpy_nd(
                            herd_out,
                            tile_out,
                            dst_offsets=[offset],
                            dst_sizes=[tile_words],
                            dst_strides=[1],
                        )

                        DeallocOp(tile_in)
                        DeallocOp(tile_out)
                        yield_([])


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
    parser = argparse.ArgumentParser(
        description="Run a real GGUF Q4_K packed-weight slice through the NPU."
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("/var/home/taowen/projects/torch2vk/dist/llama_cpp_qwen3/qwen3-0.6b-q4_k_m.gguf"),
    )
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--words", type=int, default=1_048_576)
    parser.add_argument("--tile-words", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--manifest", type=Path, default=Path("q4k_gguf_npu_manifest.json"))
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
    )
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--run-test-only", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.words % args.tile_words != 0:
        parser.error("--words must be divisible by --tile-words")
    index, selected = select_q4k_tensor(args.gguf, args.tensor)
    payload = read_tensor_prefix(index.path, selected, size=args.words * 4)
    if len(payload) != args.words * 4:
        raise ValueError(f"Requested {args.words} words but only read {len(payload) // 4}")
    input_words = np.frombuffer(payload, dtype=np.uint32).copy()
    expected_words = input_words.copy()
    payload_sha256 = hashlib.sha256(payload).hexdigest()

    manifest: dict[str, object] = {
        "gguf_path": str(index.path),
        "selected_tensor": selected.to_json(),
        "word_count": int(args.words),
        "payload_bytes": int(input_words.nbytes),
        "tile_words": int(args.tile_words),
        "payload_sha256": payload_sha256,
        "first_8_words_hex": [f"0x{int(word):08x}" for word in input_words[:8]],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    module = build_module(args.words, args.tile_words)
    if args.print_module_only:
        print(module)
        return 0

    print(f"GGUF tensor {selected.name}")
    print(f"GGUF physical {selected.physical_dtype}{selected.physical_shape}")
    print(f"payload_sha256 {payload_sha256}")

    if args.run_test_only:
        runner = XRTRunner(
            verbose=args.verbose,
            output_format=args.output_format,
            instance_name="q4k_packed_passthrough",
            runtime_loop_tiling_sizes=[4, 4],
        )
        return int(runner.run_test(module, inputs=[input_words], expected_outputs=[expected_words]))

    backend = XRTBackend(
        verbose=args.verbose,
        output_format=args.output_format,
        instance_name="q4k_packed_passthrough",
        runtime_loop_tiling_sizes=[4, 4],
    )
    compiled = backend.compile(module)
    func = backend.load(compiled)
    output = np.zeros_like(expected_words)

    for _ in range(args.warmup):
        actual = func(input_words, output)[1]
        if not np.array_equal(np.reshape(actual, expected_words.shape), expected_words):
            backend.unload()
            raise SystemExit("Warmup output mismatch")

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        output.fill(0)
        start = time.perf_counter()
        actual = func(input_words, output)[1]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        actual = np.reshape(actual, expected_words.shape)
        if not np.array_equal(actual, expected_words):
            backend.unload()
            raise SystemExit("Timed output mismatch")
        latencies_ms.append(elapsed_ms)

    backend.unload()
    sorted_lat = sorted(latencies_ms)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95 = sorted_lat[int(len(sorted_lat) * 0.95) - 1]
    mean = sum(latencies_ms) / len(latencies_ms)
    gib_per_s = (input_words.nbytes * 2) / (mean / 1000.0) / (1024**3)
    manifest["benchmark"] = {
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_ms": mean,
        "p50_ms": p50,
        "p95_ms": p95,
        "effective_bidirectional_gib_s": gib_per_s,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("PASS!")
    print(f"payload_mib {input_words.nbytes / (1024 * 1024):.3f}")
    print(f"mean_ms {mean:.3f}")
    print(f"p50_ms {p50:.3f}")
    print(f"p95_ms {p95:.3f}")
    print(f"effective_bidirectional_gib_s {gib_per_s:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
