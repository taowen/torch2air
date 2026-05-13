from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from torch2air.weights.gguf import dequantize_q4_k_blocks, load_gguf_index, read_tensor_bytes


DEFAULT_GGUF = Path("/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf")
DEFAULT_TENSOR = "model.embed_tokens.weight"


def prepare_inputs(
    *,
    gguf_path: Path,
    tensor_name: str,
    token_ids: list[int],
    blocks_per_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    index = load_gguf_index(gguf_path)
    selected = index.tensors[tensor_name]
    if selected.ggml_type != "Q4_K":
        raise ValueError(f"{tensor_name} is {selected.ggml_type}, not Q4_K")
    if selected.physical_dtype != "uint32" or len(selected.physical_shape) != 2:
        raise ValueError(f"Expected rank-2 uint32 Q4_K tensor, got {selected}")

    vocab_size = int(selected.physical_shape[0])
    model_row_words = int(selected.physical_shape[1])
    if model_row_words % 36 != 0:
        raise ValueError(f"Q4_K row word width must be a multiple of 36, got {model_row_words}")
    model_blocks_per_row = model_row_words // 36
    if blocks_per_row <= 0 or blocks_per_row > model_blocks_per_row:
        raise ValueError(f"blocks_per_row must be in [1, {model_blocks_per_row}], got {blocks_per_row}")
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token id {token_id} is outside [0, {vocab_size})")

    row_words = blocks_per_row * 36
    row_bytes = row_words * 4
    model_row_bytes = model_row_words * 4
    packed_rows = np.empty((len(token_ids), row_words), dtype=np.int32)
    raw_blocks = np.empty((len(token_ids) * blocks_per_row, 144), dtype=np.uint8)

    for row_idx, token_id in enumerate(token_ids):
        payload = read_tensor_bytes(
            index.path,
            selected,
            offset=token_id * model_row_bytes,
            size=row_bytes,
        )
        packed_rows[row_idx, :] = np.frombuffer(payload, dtype=np.int32)
        raw_blocks[row_idx * blocks_per_row : (row_idx + 1) * blocks_per_row, :] = np.frombuffer(
            payload,
            dtype=np.uint8,
        ).reshape(blocks_per_row, 144)

    d = raw_blocks[:, 0:2].view(np.float16).astype(np.float32).reshape(-1)
    dmin = raw_blocks[:, 2:4].view(np.float16).astype(np.float32).reshape(-1)
    block_f16_scales = np.stack([d, dmin], axis=1).reshape(len(token_ids), blocks_per_row, 2)
    expected = dequantize_q4_k_blocks(raw_blocks).reshape(len(token_ids), blocks_per_row * 256)
    info = {
        "tensor": selected.to_json(),
        "token_ids": token_ids,
        "blocks_per_row": blocks_per_row,
        "hidden_size": blocks_per_row * 256,
    }
    return (
        np.ascontiguousarray(packed_rows),
        np.ascontiguousarray(block_f16_scales),
        np.ascontiguousarray(expected),
        info,
    )


def compile_runtime(
    *,
    aie_mlir: Path,
    work_dir: Path,
    instance_name: str,
    peano_install_dir: str,
) -> tuple[Path, Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    aiecc_dir = work_dir / "aiecc"
    shutil.rmtree(aiecc_dir, ignore_errors=True)
    aiecc_dir.mkdir(parents=True, exist_ok=True)

    npu_mlir = work_dir / f"{instance_name}.npu.mlir"
    xclbin = work_dir / f"{instance_name}.xclbin"
    insts = work_dir / f"{instance_name}.insts.bin"

    subprocess.run(
        [
            "air-opt",
            str(aie_mlir),
            "--air-to-std",
            "--airrt-to-npu",
            "--canonicalize",
            "-o",
            str(npu_mlir),
        ],
        check=True,
    )
    subprocess.run(
        [
            "aiecc",
            "--no-aiesim",
            "--no-xchesscc",
            "--no-xbridge",
            "--no-compile-host",
            f"--tmpdir={aiecc_dir}",
            "--aie-generate-xclbin",
            f"--xclbin-name={xclbin}",
            "--aie-generate-npu-insts",
            f"--npu-insts-name={insts}",
            f"--xclbin-instance-name={instance_name}",
            f"--peano={peano_install_dir}",
            "-O",
            "0",
            str(npu_mlir),
        ],
        check=True,
    )
    return npu_mlir, xclbin, insts


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    expected: np.ndarray,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, list[float]]:
    backend = XRTBackend(
        verbose=verbose,
        output_format="xclbin",
        instance_name=instance_name,
    )
    func = backend.load(XRTCompileArtifact(str(xclbin), "MLIR_AIE", str(insts)))
    output = np.zeros_like(expected)
    actual = output
    latencies_ms: list[float] = []
    try:
        for _ in range(warmup):
            output.fill(0)
            actual = np.asarray(func(packed_rows, block_f16_scales, output)[2]).reshape(expected.shape)
            _check_close(actual, expected, rtol=rtol, atol=atol)

        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(packed_rows, block_f16_scales, output)[2]).reshape(expected.shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            _check_close(actual, expected, rtol=rtol, atol=atol)
    finally:
        backend.unload()
    return actual, latencies_ms


def _check_close(actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float) -> None:
    if np.allclose(actual, expected, rtol=rtol, atol=atol):
        return
    diff = np.abs(actual - expected)
    flat_idx = int(np.argmax(diff))
    raise AssertionError(
        "NPU output mismatch: "
        f"max_abs={float(diff.reshape(-1)[flat_idx])} "
        f"index={np.unravel_index(flat_idx, diff.shape)} "
        f"actual={float(actual.reshape(-1)[flat_idx])} "
        f"expected={float(expected.reshape(-1)[flat_idx])}"
    )


def parse_token_ids(value: str) -> list[int]:
    token_ids = [int(part) for part in value.split(",") if part.strip()]
    if not token_ids:
        raise argparse.ArgumentTypeError("expected at least one token id")
    return token_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantized_qwen3 embed_tokens on real NPU.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, required=True)
    parser.add_argument("--aie-mlir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--instance-name", default="run_embed_tokens")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    packed_rows, block_f16_scales, expected, info = prepare_inputs(
        gguf_path=args.gguf,
        tensor_name=args.tensor,
        token_ids=args.token_ids,
        blocks_per_row=args.blocks_per_row,
    )
    print(f"GGUF tensor {info['tensor']['name']} {info['tensor']['ggml_type']}")
    print(f"token_ids {','.join(str(v) for v in args.token_ids)}")
    print(f"blocks_per_row {args.blocks_per_row} hidden_size {info['hidden_size']}")

    npu_mlir, xclbin, insts = compile_runtime(
        aie_mlir=args.aie_mlir,
        work_dir=args.work_dir,
        instance_name=args.instance_name,
        peano_install_dir=peano_install_dir,
    )
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=args.instance_name,
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        expected=expected,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = float(np.max(np.abs(actual - expected)))
    print(f"npu_mlir {npu_mlir}")
    print(f"xclbin {xclbin}")
    print(f"insts {insts}")
    print(f"actual_first8 {actual.reshape(-1)[:8].tolist()}")
    print(f"expected_first8 {expected.reshape(-1)[:8].tolist()}")
    print(f"max_abs {max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
