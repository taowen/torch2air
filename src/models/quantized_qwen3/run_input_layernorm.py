from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from torch2air.weights.gguf import load_gguf_index, read_tensor_bytes

from .run_embed_tokens import DEFAULT_GGUF, _check_close, compile_runtime, parse_token_ids, prepare_inputs
from .run_embed_tokens_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR


def prepare_layernorm_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    _, _, hidden, info = prepare_inputs(
        gguf_path=gguf_path,
        tensor_name="model.embed_tokens.weight",
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
    )
    hidden_size = blocks_per_row * 256
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[rms_weight_tensor]
    if weight_entry.ggml_type != "F32" or weight_entry.physical_dtype != "float32":
        raise ValueError(f"{rms_weight_tensor} must be F32, got {weight_entry}")
    if int(weight_entry.physical_shape[0]) != hidden_size:
        raise ValueError(f"{rms_weight_tensor} shape must match hidden_size={hidden_size}")
    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=hidden_size * 4)
    rms_weight = np.frombuffer(payload, dtype=np.float32).copy()

    variance = np.mean(hidden.astype(np.float32) ** 2, axis=-1, keepdims=True)
    expected = hidden * (1.0 / np.sqrt(variance + eps)).astype(np.float32)
    expected = expected * rms_weight.reshape(1, hidden_size)
    info["rms_weight"] = weight_entry.to_json()
    info["rms_norm_eps"] = eps
    return (
        np.ascontiguousarray(hidden.astype(np.float32, copy=False)),
        np.ascontiguousarray(rms_weight),
        np.ascontiguousarray(expected.astype(np.float32, copy=False)),
        info,
    )


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    hidden: np.ndarray,
    rms_weight: np.ndarray,
    expected: np.ndarray,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, list[float]]:
    backend = XRTBackend(verbose=verbose, output_format="xclbin", instance_name=instance_name)
    func = backend.load(XRTCompileArtifact(str(xclbin), "MLIR_AIE", str(insts)))
    output = np.zeros_like(expected)
    actual = output
    latencies_ms: list[float] = []
    try:
        for _ in range(warmup):
            output.fill(0)
            actual = np.asarray(func(hidden, rms_weight, output)[2]).reshape(expected.shape)
            _check_close(actual, expected, rtol=rtol, atol=atol)
        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(hidden, rms_weight, output)[2]).reshape(expected.shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            _check_close(actual, expected, rtol=rtol, atol=atol)
    finally:
        backend.unload()
    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantized_qwen3 input_layernorm on real NPU.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, required=True)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--aie-mlir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--instance-name", default="run_input_layernorm")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    hidden, rms_weight, expected, info = prepare_layernorm_inputs(
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        blocks_per_row=args.blocks_per_row,
        rms_weight_tensor=args.rms_weight_tensor,
        eps=args.rms_norm_eps,
    )
    print(f"input_source {info['tensor']['name']} dequantized reference buffer")
    print(f"RMS weight {info['rms_weight']['name']} {info['rms_weight']['ggml_type']}")
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
        hidden=hidden,
        rms_weight=rms_weight,
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
