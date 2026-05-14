from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact

from torch2air.export.q4k_linear import Q4KLinearAirBuilder
from torch2air.export.q6k_linear import Q6KLinearAirBuilder
from torch2air.runtime.compile import compile_q4k_linear_python_kernel
from torch2air.runtime.compile import compile_q6k_linear_python_kernel
from torch2air.runtime.compile import load_kernel_function
from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index, read_tensor_bytes

from .reference_runtime import (
    check_close_rocm,
    dequantize_q4_k_blocks_rocm,
    dequantize_q6_k_blocks_rocm,
    first_values,
    max_abs_rocm,
    q4k_block_f16_scales_rocm,
    q6k_block_f16_scales_rocm,
    rocm_device,
)
from .run_embed_tokens import DEFAULT_GGUF, parse_token_ids
from .run_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR, prepare_layernorm_inputs

LINEAR_STAGES = ("q_proj", "k_proj", "v_proj", "o_proj")
DEFAULT_STAGE = "q_proj"
DEFAULT_LAYER_INPUT_BLOCKS_PER_ROW = 4


@dataclass(frozen=True, slots=True)
class LinearInputInfo:
    stage: str
    source_name: str
    source_type: str
    rms_weight_name: str
    rms_weight_type: str
    projection_weight: GGUFTensorEntry
    weight_type: str
    token_ids: list[int]
    hidden_size: int
    output_features: int
    output_rows: int
    output_tile_rows: int
    blocks_per_row: int
    weight_words: int


def projection_weight_tensor(stage: str) -> str:
    validate_linear_stage(stage)
    return f"model.layers.0.self_attn.{stage}.weight"


def validate_linear_stage(stage: str) -> None:
    if stage not in LINEAR_STAGES:
        raise ValueError(f"stage must be one of {LINEAR_STAGES}, got {stage!r}")


def prepare_linear_inputs(
    *,
    stage: str,
    gguf_path: Path,
    token_ids: list[int],
    layer_input_blocks_per_row: int,
    rms_weight_tensor: str,
    projection_tensor: str,
    output_rows: int,
    output_tile_rows: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, LinearInputInfo]:
    validate_linear_stage(stage)
    if len(token_ids) not in {1, 8}:
        raise ValueError("formal quantized linear supports decode S=1 or prefill S=8")
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[projection_tensor]
    output_features, row_words = (int(dim) for dim in weight_entry.physical_shape)
    if output_features % output_rows != 0:
        raise ValueError(f"output_features={output_features} must be divisible by {output_rows}")
    if output_rows % output_tile_rows != 0:
        raise ValueError(f"output_rows={output_rows} must be divisible by {output_tile_rows}")
    if output_rows // output_tile_rows > 4:
        raise ValueError("output_rows/output_tile_rows must fit in 4 NPU columns")
    model_blocks_per_row, hidden_size = _linear_weight_shape(
        projection_tensor=projection_tensor,
        weight_entry=weight_entry,
        row_words=row_words,
    )
    hidden, hidden_rocm, source_name, source_type, rms_name, rms_type = _prepare_linear_input(
        stage=stage,
        gguf_path=gguf_path,
        token_ids=token_ids,
        layer_input_blocks_per_row=layer_input_blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
        hidden_size=hidden_size,
    )

    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=weight_entry.nbytes)
    packed_weight, reference = _prepare_linear_weight_and_reference_rocm(
        payload=payload,
        weight_entry=weight_entry,
        hidden_size=hidden_size,
        output_features=output_features,
        row_words=row_words,
        blocks_per_row=model_blocks_per_row,
    )
    expected = reference(hidden_rocm.reshape(1, len(token_ids), hidden_size)).reshape(
        len(token_ids),
        output_features,
    )
    info = LinearInputInfo(
        stage=stage,
        source_name=source_name,
        source_type=source_type,
        rms_weight_name=rms_name,
        rms_weight_type=rms_type,
        projection_weight=weight_entry,
        weight_type=weight_entry.ggml_type,
        token_ids=token_ids,
        hidden_size=hidden_size,
        output_features=output_features,
        output_rows=output_rows,
        output_tile_rows=output_tile_rows,
        blocks_per_row=model_blocks_per_row,
        weight_words=packed_weight.shape[1],
    )
    return (
        hidden,
        np.ascontiguousarray(packed_weight),
        expected,
        info,
    )


def _prepare_linear_input(
    *,
    stage: str,
    gguf_path: Path,
    token_ids: list[int],
    layer_input_blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    hidden_size: int,
) -> tuple[np.ndarray, torch.Tensor, str, str, str, str]:
    if stage in {"q_proj", "k_proj", "v_proj"}:
        _, _, normed_hidden, layernorm_info = prepare_layernorm_inputs(
            gguf_path=gguf_path,
            token_ids=token_ids,
            blocks_per_row=layer_input_blocks_per_row,
            rms_weight_tensor=rms_weight_tensor,
            eps=eps,
        )
        if hidden_size != layernorm_info.hidden_size:
            raise ValueError(
                f"{stage} hidden_size={hidden_size} does not match "
                f"layernorm hidden_size={layernorm_info.hidden_size}"
            )
        return (
            np.ascontiguousarray(normed_hidden.detach().cpu().numpy().astype(np.float32)),
            normed_hidden.to(device=rocm_device(), dtype=torch.float32),
            layernorm_info.source.name,
            layernorm_info.source.ggml_type,
            layernorm_info.rms_weight.name,
            layernorm_info.rms_weight.ggml_type,
        )
    if stage == "o_proj":
        hidden = deterministic_o_proj_input_rocm(token_ids=token_ids, hidden_size=hidden_size)
        return (
            np.ascontiguousarray(hidden.detach().cpu().numpy().astype(np.float32)),
            hidden,
            "deterministic_o_proj_input",
            "F32",
            "none",
            "none",
        )
    raise ValueError(f"unsupported quantized linear stage: {stage}")


def deterministic_o_proj_input_rocm(*, token_ids: list[int], hidden_size: int) -> torch.Tensor:
    device = rocm_device()
    tokens = torch.tensor(token_ids, device=device, dtype=torch.float32).reshape(-1, 1)
    features = torch.arange(hidden_size, device=device, dtype=torch.float32).reshape(1, -1)
    return torch.sin((tokens + 1.0) * 0.17 + (features + 1.0) * 0.013).to(torch.float32)


def _linear_weight_shape(
    *,
    projection_tensor: str,
    weight_entry: GGUFTensorEntry,
    row_words: int,
) -> tuple[int, int]:
    if weight_entry.ggml_type == "Q4_K" and weight_entry.physical_dtype == "uint32":
        if row_words % 36 != 0:
            raise ValueError(f"Q4_K row word width must be a multiple of 36, got {row_words}")
        blocks_per_row = row_words // 36
        return blocks_per_row, blocks_per_row * 256
    if weight_entry.ggml_type == "Q6_K" and weight_entry.physical_dtype == "uint16":
        if row_words % 105 != 0:
            raise ValueError(f"Q6_K row halfword width must be a multiple of 105, got {row_words}")
        blocks_per_row = row_words // 105
        return blocks_per_row, blocks_per_row * 256
    raise ValueError(
        f"{projection_tensor} must be Q4_K uint32 or Q6_K uint16, got {weight_entry}"
    )


def _prepare_linear_weight_and_reference_rocm(
    *,
    payload: bytes,
    weight_entry: GGUFTensorEntry,
    hidden_size: int,
    output_features: int,
    row_words: int,
    blocks_per_row: int,
) -> tuple[np.ndarray, torch.nn.Linear]:
    if weight_entry.ggml_type == "Q4_K":
        raw_blocks = np.frombuffer(payload, dtype=np.uint8).copy().reshape(
            output_features * blocks_per_row,
            144,
        )
        packed_weight = _append_q4k_scale_bits(
            packed_rows=np.frombuffer(payload, dtype=np.int32).copy().reshape(
                output_features,
                row_words,
            ),
            raw_blocks=raw_blocks,
            output_features=output_features,
            blocks_per_row=blocks_per_row,
        )
        reference = _q4k_linear_reference_module_rocm(
            raw_blocks=raw_blocks,
            hidden_size=hidden_size,
            output_features=output_features,
        )
        return np.ascontiguousarray(packed_weight), reference
    if weight_entry.ggml_type == "Q6_K":
        raw_halfwords = np.frombuffer(payload, dtype=np.uint16).copy().reshape(
            output_features,
            row_words,
        )
        raw_blocks = raw_halfwords.reshape(output_features * blocks_per_row, 105)
        packed_weight = _append_q6k_scale_bits(
            packed_rows=raw_halfwords,
            raw_blocks=raw_blocks,
            output_features=output_features,
            blocks_per_row=blocks_per_row,
        )
        reference = _q6k_linear_reference_module_rocm(
            raw_blocks=raw_blocks,
            hidden_size=hidden_size,
            output_features=output_features,
        )
        return np.ascontiguousarray(packed_weight), reference
    raise ValueError(f"unsupported linear weight type: {weight_entry.ggml_type}")


def _append_q4k_scale_bits(
    *,
    packed_rows: np.ndarray,
    raw_blocks: np.ndarray,
    output_features: int,
    blocks_per_row: int,
) -> np.ndarray:
    row_words = blocks_per_row * 36
    scale_values = (
        q4k_block_f16_scales_rocm(raw_blocks)
        .reshape(output_features, blocks_per_row, 2)
        .detach()
        .cpu()
        .numpy()
    )
    scale_bits = np.ascontiguousarray(scale_values.astype(np.float32, copy=False)).view(np.int32)
    packed = np.empty((output_features, row_words + blocks_per_row * 2), dtype=np.int32)
    packed[:, :row_words] = packed_rows
    packed[:, row_words:] = scale_bits.reshape(output_features, blocks_per_row * 2)
    return packed


def _append_q6k_scale_bits(
    *,
    packed_rows: np.ndarray,
    raw_blocks: np.ndarray,
    output_features: int,
    blocks_per_row: int,
) -> np.ndarray:
    row_words = blocks_per_row * 105
    scale_values = (
        q6k_block_f16_scales_rocm(raw_blocks)
        .reshape(output_features, blocks_per_row)
        .detach()
        .cpu()
        .numpy()
    )
    scale_bits = np.ascontiguousarray(scale_values.astype(np.float32, copy=False)).view(np.int32)
    packed = np.empty((output_features, row_words + blocks_per_row), dtype=np.int32)
    packed[:, :row_words] = packed_rows.astype(np.int32, copy=False)
    packed[:, row_words:] = scale_bits.reshape(output_features, blocks_per_row)
    return packed


def _q4k_linear_reference_module_rocm(
    *,
    raw_blocks: np.ndarray,
    hidden_size: int,
    output_features: int,
) -> torch.nn.Linear:
    rows = dequantize_q4_k_blocks_rocm(raw_blocks).reshape(output_features, hidden_size)
    module = torch.nn.Linear(
        hidden_size,
        output_features,
        bias=False,
        device=rocm_device(),
        dtype=torch.float32,
    )
    with torch.no_grad():
        module.weight.copy_(rows)
    module.eval()
    return module


def _q6k_linear_reference_module_rocm(
    *,
    raw_blocks: np.ndarray,
    hidden_size: int,
    output_features: int,
) -> torch.nn.Linear:
    rows = dequantize_q6_k_blocks_rocm(raw_blocks).reshape(output_features, hidden_size)
    module = torch.nn.Linear(
        hidden_size,
        output_features,
        bias=False,
        device=rocm_device(),
        dtype=torch.float32,
    )
    with torch.no_grad():
        module.weight.copy_(rows)
    module.eval()
    return module


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
    expected: torch.Tensor,
    stage: str,
    output_rows: int,
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
    expected_shape = tuple(expected.shape)
    actual = np.zeros(expected_shape, dtype=np.float32)
    latencies_ms: list[float] = []
    try:
        if warmup == 0 and iterations == 0:
            actual = _run_output_chunks(
                func=func,
                hidden=hidden,
                packed_weight=packed_weight,
                output_rows=output_rows,
                expected_shape=expected_shape,
            )
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)

        for _ in range(warmup):
            actual = _run_output_chunks(
                func=func,
                hidden=hidden,
                packed_weight=packed_weight,
                output_rows=output_rows,
                expected_shape=expected_shape,
            )
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)

        for _ in range(iterations):
            start = time.perf_counter()
            actual = _run_output_chunks(
                func=func,
                hidden=hidden,
                packed_weight=packed_weight,
                output_rows=output_rows,
                expected_shape=expected_shape,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)
    finally:
        backend.unload()
    return actual, latencies_ms


def _run_output_chunks(
    *,
    func,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
    output_rows: int,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    sequence_length = expected_shape[0]
    output_features = expected_shape[1]
    actual = np.zeros(expected_shape, dtype=np.float32)
    output_chunk = np.zeros((sequence_length, output_rows), dtype=np.float32)
    for row_offset in range(0, output_features, output_rows):
        output_chunk.fill(0)
        weight_chunk = np.ascontiguousarray(packed_weight[row_offset : row_offset + output_rows])
        result = np.asarray(func(hidden, weight_chunk, output_chunk)[2]).reshape(
            output_chunk.shape
        )
        actual[:, row_offset : row_offset + output_rows] = result
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run formal quantized linear kernels.")
    parser.add_argument("--stage", choices=LINEAR_STAGES, default=DEFAULT_STAGE)
    parser.add_argument("--kernel-py", type=Path, required=True)
    parser.add_argument("--function-name")
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, default=DEFAULT_LAYER_INPUT_BLOCKS_PER_ROW)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--projection-tensor")
    parser.add_argument("--output-rows", type=int, default=64)
    parser.add_argument("--output-tile-rows", type=int, default=16)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    function_name = args.function_name or f"run_{args.stage}"
    projection_tensor = args.projection_tensor or projection_weight_tensor(args.stage)
    hidden, packed_weight, expected, info = prepare_linear_inputs(
        stage=args.stage,
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        layer_input_blocks_per_row=args.blocks_per_row,
        rms_weight_tensor=args.rms_weight_tensor,
        projection_tensor=projection_tensor,
        output_rows=args.output_rows,
        output_tile_rows=args.output_tile_rows,
        eps=args.rms_norm_eps,
    )
    sequence_length, hidden_size, output_features = _kernel_shape(
        args.kernel_py,
        function_name,
        args.output_rows,
        args.output_tile_rows,
        info.weight_type,
    )
    if sequence_length != len(args.token_ids):
        raise SystemExit(f"token count must match exported {args.stage} sequence length")
    if hidden_size != info.hidden_size:
        raise SystemExit(f"input width must match exported {args.stage} hidden size")
    if output_features != args.output_rows:
        raise SystemExit("compiled output rows must match quantized linear output shape")

    print(f"stage {info.stage}")
    print(f"input_source {info.source_name} {info.source_type}")
    print(f"RMS weight {info.rms_weight_name} {info.rms_weight_type}")
    print(f"{info.weight_type} weight {info.projection_weight.name}")
    print(f"token_ids {','.join(str(v) for v in info.token_ids)}")
    print(f"hidden_size {info.hidden_size} output_features {info.output_features}")
    print(f"output_rows {info.output_rows} output_tile_rows {info.output_tile_rows}")
    print(f"blocks_per_row {info.blocks_per_row} weight_words {info.weight_words}")
    print(f"reference pytorch_rocm torch.nn.Linear {torch.cuda.get_device_name(0)}")

    if info.weight_type == "Q4_K":
        source_mlir, aie_mlir, xclbin, insts, object_file = compile_q4k_linear_python_kernel(
            kernel_py=args.kernel_py,
            function_name=function_name,
            work_dir=args.work_dir,
            instance_name=function_name,
            output_features=args.output_rows,
            output_tile_rows=args.output_tile_rows,
        )
    elif info.weight_type == "Q6_K":
        source_mlir, aie_mlir, xclbin, insts, object_file = compile_q6k_linear_python_kernel(
            kernel_py=args.kernel_py,
            function_name=function_name,
            work_dir=args.work_dir,
            instance_name=function_name,
            output_features=args.output_rows,
            output_tile_rows=args.output_tile_rows,
        )
    else:
        raise SystemExit(f"unsupported linear weight type: {info.weight_type}")
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=function_name,
        hidden=hidden,
        packed_weight=packed_weight,
        expected=expected,
        stage=args.stage,
        output_rows=args.output_rows,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = max_abs_rocm(actual, expected)
    print(f"source_mlir_cache {source_mlir}")
    print(f"aie_mlir_cache {aie_mlir}")
    print(f"object_file {object_file}")
    print(f"xclbin {xclbin}")
    print(f"insts {insts}")
    print(f"actual_first8 {actual.reshape(-1)[:8].tolist()}")
    print(f"expected_first8 {first_values(expected)}")
    print(f"max_abs {max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


def _kernel_shape(
    kernel_py: Path,
    function_name: str,
    output_features: int,
    output_tile_rows: int,
    weight_type: str,
) -> tuple[int, int, int]:
    if weight_type == "Q4_K":
        builder = Q4KLinearAirBuilder(
            function_name=function_name,
            output_features=output_features,
            output_tile_rows=output_tile_rows,
        )
    elif weight_type == "Q6_K":
        builder = Q6KLinearAirBuilder(
            function_name=function_name,
            output_features=output_features,
            output_tile_rows=output_tile_rows,
        )
    else:
        raise ValueError(f"unsupported linear weight type: {weight_type}")
    load_kernel_function(kernel_py, function_name)(builder)
    return builder.linear_shape()


if __name__ == "__main__":
    raise SystemExit(main())
