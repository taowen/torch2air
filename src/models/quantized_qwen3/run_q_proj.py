from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from torch2air.weights.gguf import load_gguf_index, read_tensor_bytes

from . import reference
from .reference_runtime import (
    check_close_rocm,
    first_values,
    max_abs_rocm,
    q4k_block_f16_scales_rocm,
)
from .run_embed_tokens import DEFAULT_GGUF, compile_runtime, installed_tool, parse_token_ids, prepare_inputs
from .run_embed_tokens_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR


DEFAULT_Q_PROJ_WEIGHT_TENSOR = "model.layers.0.self_attn.q_proj.weight"
KERNEL_SOURCE = Path(__file__).resolve().parents[2] / "torch2air" / "export" / "kernels" / "q4k_linear.cc"


def prepare_norm_hidden(
    *,
    gguf_path: Path,
    token_ids: list[int],
    rms_weight_tensor: str,
    eps: float,
) -> tuple[np.ndarray, torch.Tensor, dict[str, object]]:
    _, _, embed_expected, info = prepare_inputs(
        gguf_path=gguf_path,
        tensor_name="model.embed_tokens.weight",
        token_ids=token_ids,
        blocks_per_row=4,
    )
    hidden_size = embed_expected.shape[1]
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[rms_weight_tensor]
    if weight_entry.ggml_type != "F32" or weight_entry.physical_dtype != "float32":
        raise ValueError(f"{rms_weight_tensor} must be F32, got {weight_entry}")
    if int(weight_entry.physical_shape[0]) != hidden_size:
        raise ValueError(f"{rms_weight_tensor} shape must match hidden_size={hidden_size}")

    norm_hidden = reference.run_input_layernorm(hidden_states=embed_expected)["mul_1"]
    info["rms_weight"] = weight_entry.to_json()
    info["rms_norm_eps"] = eps
    return np.ascontiguousarray(norm_hidden.detach().cpu().numpy()), norm_hidden, info


def prepare_q_proj_weights(
    *,
    gguf_path: Path,
    tensor_name: str,
    output_rows: int,
    hidden_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    index = load_gguf_index(gguf_path)
    selected = index.tensors[tensor_name]
    if selected.ggml_type != "Q4_K":
        raise ValueError(f"{tensor_name} is {selected.ggml_type}, not Q4_K")
    if selected.physical_dtype != "uint32" or len(selected.physical_shape) != 2:
        raise ValueError(f"Expected rank-2 uint32 Q4_K tensor, got {selected}")
    if int(selected.logical_shape[1]) != hidden_size:
        raise ValueError(f"{tensor_name} input size must be {hidden_size}, got {selected.logical_shape}")
    if output_rows <= 0 or output_rows > int(selected.physical_shape[0]):
        raise ValueError(f"output_rows must be in [1, {selected.physical_shape[0]}], got {output_rows}")

    row_words = int(selected.physical_shape[1])
    if row_words % 36 != 0:
        raise ValueError(f"Q4_K row word width must be a multiple of 36, got {row_words}")
    blocks_per_row = row_words // 36
    if blocks_per_row * 256 != hidden_size:
        raise ValueError(f"{tensor_name} row shape implies hidden_size={blocks_per_row * 256}")

    payload = read_tensor_bytes(
        index.path,
        selected,
        offset=0,
        size=output_rows * row_words * 4,
    )
    packed_weights = np.frombuffer(payload, dtype=np.int32).copy().reshape(output_rows, row_words)
    raw_blocks = np.frombuffer(payload, dtype=np.uint8).reshape(output_rows * blocks_per_row, 144)
    block_scales = q4k_block_f16_scales_rocm(raw_blocks).reshape(output_rows, blocks_per_row, 2)
    scale_bits = (
        block_scales.detach()
        .cpu()
        .numpy()
        .reshape(output_rows, blocks_per_row * 2)
        .view(np.int32)
    )
    packed_with_scales = np.concatenate([packed_weights, scale_bits], axis=1)
    info = {
        "tensor": selected.to_json(),
        "output_rows": output_rows,
        "blocks_per_row": blocks_per_row,
        "hidden_size": hidden_size,
    }
    return (
        np.ascontiguousarray(packed_with_scales),
        info,
    )


def prepare_q_proj_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    tensor_name: str,
    output_rows: int,
    rms_weight_tensor: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, dict[str, object]]:
    norm_hidden, norm_hidden_ref, info = prepare_norm_hidden(
        gguf_path=gguf_path,
        token_ids=token_ids,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    packed_weights, weight_info = prepare_q_proj_weights(
        gguf_path=gguf_path,
        tensor_name=tensor_name,
        output_rows=output_rows,
        hidden_size=norm_hidden.shape[1],
    )
    with torch.no_grad():
        expected = reference.run_q_proj(input=norm_hidden_ref)["linear"].reshape(
            len(token_ids),
            -1,
        )[:, :output_rows]
    output = np.zeros(tuple(expected.shape), dtype=np.float32)
    info["q_proj_weight"] = weight_info["tensor"]
    info["q_proj_output_rows"] = output_rows
    return (
        norm_hidden,
        packed_weights,
        output,
        expected,
        info,
    )


def compile_q4k_linear_object(
    *,
    work_dir: Path,
    peano_install_dir: str,
    output_tile_rows: int,
    blocks_per_row: int,
    hidden_size: int,
) -> Path:
    object_path = work_dir / "q4k_linear.o"
    object_path.parent.mkdir(parents=True, exist_ok=True)
    aie_opt = installed_tool("aie-opt", "MLIR_AIE_INSTALL_DIR")
    include_dir = Path(aie_opt).resolve().parent.parent / "include"
    warning_flags = [
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-unused-command-line-argument",
    ]
    cmd = [
        str(Path(peano_install_dir) / "bin" / "clang++"),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        *warning_flags,
        "-DNDEBUG",
        "-I",
        str(include_dir),
        f"-DOUTPUT_TILE_ROWS={output_tile_rows}",
        f"-DBLOCKS_PER_ROW={blocks_per_row}",
        f"-DHIDDEN_SIZE={hidden_size}",
        "-c",
        str(KERNEL_SOURCE),
        "-o",
        str(object_path),
    ]
    subprocess.run(cmd, check=True)
    return object_path


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    hidden: np.ndarray,
    packed_weights: np.ndarray,
    expected: torch.Tensor,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, list[float]]:
    backend = XRTBackend(verbose=verbose, output_format="xclbin", instance_name=instance_name)
    func = backend.load(XRTCompileArtifact(str(xclbin), "MLIR_AIE", str(insts)))
    expected_shape = tuple(expected.shape)
    output = np.zeros(expected_shape, dtype=np.float32)
    actual = output
    latencies_ms: list[float] = []
    try:
        for _ in range(warmup):
            output.fill(0)
            actual = np.asarray(func(hidden, packed_weights, output)[2]).reshape(expected_shape)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol)
        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(hidden, packed_weights, output)[2]).reshape(expected_shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol)
    finally:
        backend.unload()
    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quantized_qwen3 q_proj on real NPU.")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--tensor", default=DEFAULT_Q_PROJ_WEIGHT_TENSOR)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--output-rows", type=int, default=64)
    parser.add_argument("--output-tile-rows", type=int, default=16)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--aie-mlir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--instance-name", default="run_q_proj")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    hidden, packed_weights, _, expected, info = prepare_q_proj_inputs(
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        tensor_name=args.tensor,
        output_rows=args.output_rows,
        rms_weight_tensor=args.rms_weight_tensor,
        eps=args.rms_norm_eps,
    )
    print(f"input_source {info['tensor']['name']} -> {info['rms_weight']['name']}")
    print(f"Q4_K weight {info['q_proj_weight']['name']} {info['q_proj_weight']['ggml_type']}")
    print(f"token_ids {','.join(str(v) for v in args.token_ids)}")
    print(f"output_rows {args.output_rows} hidden_size {hidden.shape[1]}")
    print(f"reference safetensors_pytorch_rocm {torch.cuda.get_device_name(0)}")

    q4k_object = compile_q4k_linear_object(
        work_dir=args.work_dir,
        peano_install_dir=peano_install_dir,
        output_tile_rows=args.output_tile_rows,
        blocks_per_row=4,
        hidden_size=hidden.shape[1],
    )
    npu_mlir, xclbin, insts = compile_runtime(
        aie_mlir=args.aie_mlir,
        work_dir=args.work_dir,
        instance_name=args.instance_name,
        peano_install_dir=peano_install_dir,
        link_objects=(q4k_object,),
    )
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=args.instance_name,
        hidden=hidden,
        packed_weights=packed_weights,
        expected=expected,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = max_abs_rocm(actual, expected)
    print(f"npu_mlir {npu_mlir}")
    print(f"xclbin {xclbin}")
    print(f"insts {insts}")
    print(f"actual_first8 {actual.reshape(-1)[:8].tolist()}")
    print(f"expected_first8 {first_values(expected)}")
    print(f"max_abs {max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
