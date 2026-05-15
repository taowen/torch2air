from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact
from ml_dtypes import bfloat16
from transformers import AutoConfig

from torch2air.runtime.compile import CompiledRopeKernels, compile_rope_export_python_kernel
from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index, read_tensor_bytes

from .reference_runtime import (
    check_close_rocm,
    first_values,
    float32_host_array,
    max_abs_rocm,
    rocm_device,
)
from .run_embed_tokens import DEFAULT_GGUF, parse_token_ids
from .run_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR
from .run_linear import prepare_linear_inputs, projection_weight_tensor

ROPE_STAGES = ("q_norm_rope", "k_norm_rope")
DEFAULT_STAGE = "q_norm_rope"
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_LAYER_INPUT_BLOCKS_PER_ROW = 4
NPU_DTYPE = bfloat16


@dataclass(frozen=True, slots=True)
class RopeInputInfo:
    stage: str
    projection_stage: str
    projection_weight: str
    projection_weight_type: str
    norm_weight: GGUFTensorEntry
    token_ids: list[int]
    start_position: int
    sequence_length: int
    head_count: int
    head_dim: int
    hidden_size: int
    rope_theta: float


def projection_stage_for_rope(stage: str) -> str:
    validate_rope_stage(stage)
    if stage == "q_norm_rope":
        return "q_proj"
    return "k_proj"


def norm_weight_tensor(stage: str) -> str:
    validate_rope_stage(stage)
    if stage == "q_norm_rope":
        return "model.layers.0.self_attn.q_norm.weight"
    return "model.layers.0.self_attn.k_norm.weight"


def validate_rope_stage(stage: str) -> None:
    if stage not in ROPE_STAGES:
        raise ValueError(f"stage must be one of {ROPE_STAGES}, got {stage!r}")


def prepare_rope_inputs(
    *,
    stage: str,
    model_id: str,
    gguf_path: Path,
    token_ids: list[int],
    start_position: int,
    layer_input_blocks_per_row: int,
    input_rms_weight_tensor: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, RopeInputInfo]:
    validate_rope_stage(stage)
    projection_stage = projection_stage_for_rope(stage)
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    head_dim = int(config.head_dim)
    rope_theta = float(config.rope_parameters["rope_theta"])
    source_host, source_rocm, linear_info = _prepare_projection_source(
        projection_stage=projection_stage,
        gguf_path=gguf_path,
        token_ids=token_ids,
        layer_input_blocks_per_row=layer_input_blocks_per_row,
        input_rms_weight_tensor=input_rms_weight_tensor,
        eps=eps,
    )
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[norm_weight_tensor(stage)]
    if weight_entry.ggml_type != "F32" or weight_entry.physical_dtype != "float32":
        raise ValueError(f"{weight_entry.name} must be F32, got {weight_entry}")
    if int(weight_entry.physical_shape[0]) != head_dim:
        raise ValueError(f"{weight_entry.name} must have shape [{head_dim}], got {weight_entry}")
    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=head_dim * 4)
    norm_weight_f32 = np.ascontiguousarray(np.frombuffer(payload, dtype=np.float32).copy())
    norm_weight = np.ascontiguousarray(norm_weight_f32.astype(NPU_DTYPE))
    norm_weight_rocm = torch.as_tensor(norm_weight_f32, device=rocm_device(), dtype=torch.bfloat16)
    cos, sin = rope_table_rocm(
        sequence_length=len(token_ids),
        head_dim=head_dim,
        start_position=start_position,
        theta=rope_theta,
    )
    expected = rope_reference_rocm(
        source=source_rocm,
        norm_weight=norm_weight_rocm,
        cos=cos,
        sin=sin,
        eps=eps,
    )
    info = RopeInputInfo(
        stage=stage,
        projection_stage=projection_stage,
        projection_weight=linear_info.projection_weight.name,
        projection_weight_type=linear_info.weight_type,
        norm_weight=weight_entry,
        token_ids=token_ids,
        start_position=start_position,
        sequence_length=len(token_ids),
        head_count=expected.shape[1] // head_dim,
        head_dim=head_dim,
        hidden_size=expected.shape[1],
        rope_theta=rope_theta,
    )
    rope_lut = pack_rope_lut(cos=cos, sin=sin)
    return source_host, norm_weight, rope_lut, expected, info


def _prepare_projection_source(
    *,
    projection_stage: str,
    gguf_path: Path,
    token_ids: list[int],
    layer_input_blocks_per_row: int,
    input_rms_weight_tensor: str,
    eps: float,
):
    _, _, expected, info = prepare_linear_inputs(
        stage=projection_stage,
        gguf_path=gguf_path,
        token_ids=token_ids,
        layer_input_blocks_per_row=layer_input_blocks_per_row,
        rms_weight_tensor=input_rms_weight_tensor,
        projection_tensor=projection_weight_tensor(projection_stage),
        output_rows=64,
        output_tile_rows=16,
        eps=eps,
    )
    source = expected.to(device=rocm_device(), dtype=torch.bfloat16)
    return (
        torch_bfloat16_to_numpy(source),
        source,
        info,
    )


def rope_table_rocm(
    *,
    sequence_length: int,
    head_dim: int,
    start_position: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = rocm_device()
    positions = torch.arange(
        start_position,
        start_position + sequence_length,
        device=device,
        dtype=torch.float32,
    )
    freq_idx = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    inv_freq = torch.pow(
        torch.tensor(theta, device=device, dtype=torch.float32),
        -2.0 * freq_idx / head_dim,
    )
    angles = positions.reshape(sequence_length, 1) * inv_freq.reshape(1, head_dim // 2)
    repeated = torch.cat((angles, angles), dim=1).reshape(1, sequence_length, head_dim)
    return torch.cos(repeated).to(torch.float32), torch.sin(repeated).to(torch.float32)


def rope_reference_rocm(
    *,
    source: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    sequence_length, hidden_size = source.shape
    head_dim = int(norm_weight.shape[0])
    heads = hidden_size // head_dim
    shaped = source.to(torch.bfloat16).reshape(1, sequence_length, heads, head_dim)
    shaped_f32 = shaped.to(torch.float32)
    inv_rms = torch.rsqrt(torch.mean(shaped_f32 * shaped_f32, dim=-1, keepdim=True) + eps)
    normed = (
        shaped_f32
        * inv_rms
        * norm_weight.to(torch.float32).reshape(1, 1, 1, head_dim)
    ).to(torch.bfloat16)
    half_dim = head_dim // 2
    rotated = torch.cat((-normed[..., half_dim:], normed[..., :half_dim]), dim=-1)
    output = (
        normed.to(torch.float32) * cos.to(torch.float32).unsqueeze(-2)
        + rotated.to(torch.float32) * sin.to(torch.float32).unsqueeze(-2)
    ).to(torch.bfloat16)
    return output.reshape(sequence_length, hidden_size)


def torch_bfloat16_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    host = tensor.detach().to(device="cpu", dtype=torch.float32).numpy().astype(NPU_DTYPE)
    return np.ascontiguousarray(host)


def pack_rope_lut(
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> np.ndarray:
    sequence_length = int(cos.shape[1])
    head_dim = int(cos.shape[2])
    lut = np.empty((sequence_length, head_dim * 2), dtype=NPU_DTYPE)
    cos_host = torch_bfloat16_to_numpy(cos.reshape(sequence_length, head_dim).to(torch.bfloat16))
    sin_host = torch_bfloat16_to_numpy(sin.reshape(sequence_length, head_dim).to(torch.bfloat16))
    for token_i in range(sequence_length):
        lut[token_i, :head_dim] = cos_host[token_i]
        lut[token_i, head_dim:] = sin_host[token_i]
    return np.ascontiguousarray(lut)


def run_on_npu(
    *,
    kernels: CompiledRopeKernels,
    source: np.ndarray,
    norm_weight: np.ndarray,
    rope_lut: np.ndarray,
    expected: torch.Tensor,
    stage: str,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, list[float]]:
    expected_shape = tuple(expected.shape)
    normed = np.zeros(expected_shape, dtype=NPU_DTYPE)
    output = np.zeros(expected_shape, dtype=NPU_DTYPE)
    actual = np.zeros(expected_shape, dtype=NPU_DTYPE)
    latencies_ms: list[float] = []

    def run_once() -> np.ndarray:
        normed.fill(0)
        output.fill(0)
        normed_actual = run_rms_norm_kernel(
            kernels=kernels,
            source=source,
            norm_weight=norm_weight,
            normed=normed,
            expected_shape=expected_shape,
            verbose=verbose,
        )
        return run_rope_kernel(
            kernels=kernels,
            normed=normed_actual,
            rope_lut=rope_lut,
            output=output,
            expected_shape=expected_shape,
            verbose=verbose,
        )

    if warmup == 0 and iterations == 0:
        actual = run_once()
        check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)

    for _ in range(warmup):
        actual = run_once()
        check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)

    for _ in range(iterations):
        start = time.perf_counter()
        actual = run_once()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        check_close_rocm(actual, expected, rtol=rtol, atol=atol, label=stage)
    return actual, latencies_ms


def run_rms_norm_kernel(
    *,
    kernels: CompiledRopeKernels,
    source: np.ndarray,
    norm_weight: np.ndarray,
    normed: np.ndarray,
    expected_shape: tuple[int, ...],
    verbose: bool,
) -> np.ndarray:
    backend = XRTBackend(
        verbose=verbose,
        output_format="xclbin",
        instance_name=kernels.rms_instance_name,
    )
    func = backend.load(XRTCompileArtifact(str(kernels.rms_xclbin), "MLIR_AIE", str(kernels.rms_insts)))
    try:
        return np.ascontiguousarray(np.asarray(func(source, norm_weight, normed)[2]).reshape(expected_shape))
    finally:
        backend.unload()


def run_rope_kernel(
    *,
    kernels: CompiledRopeKernels,
    normed: np.ndarray,
    rope_lut: np.ndarray,
    output: np.ndarray,
    expected_shape: tuple[int, ...],
    verbose: bool,
) -> np.ndarray:
    backend = XRTBackend(
        verbose=verbose,
        output_format="xclbin",
        instance_name=kernels.rope_instance_name,
    )
    func = backend.load(XRTCompileArtifact(str(kernels.rope_xclbin), "MLIR_AIE", str(kernels.rope_insts)))
    try:
        return np.ascontiguousarray(np.asarray(func(normed, rope_lut, output)[2]).reshape(expected_shape))
    finally:
        backend.unload()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run exported RoPE aten sequence.")
    parser.add_argument("--stage", choices=ROPE_STAGES, default=DEFAULT_STAGE)
    parser.add_argument("--kernel-py", type=Path, required=True)
    parser.add_argument("--function-name")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--blocks-per-row", type=int, default=DEFAULT_LAYER_INPUT_BLOCKS_PER_ROW)
    parser.add_argument("--input-rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    function_name = args.function_name or f"run_{args.stage}"
    source, norm_weight, rope_lut, expected, info = prepare_rope_inputs(
        stage=args.stage,
        model_id=args.model_id,
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        start_position=args.start_position,
        layer_input_blocks_per_row=args.blocks_per_row,
        input_rms_weight_tensor=args.input_rms_weight_tensor,
        eps=args.rms_norm_eps,
    )

    print(f"stage {info.stage}")
    print(f"projection_stage {info.projection_stage}")
    print(f"{info.projection_weight_type} weight {info.projection_weight}")
    print(f"norm_weight {info.norm_weight.name} {info.norm_weight.ggml_type}")
    print(f"token_ids {','.join(str(v) for v in info.token_ids)}")
    print(f"start_position {info.start_position}")
    print(f"sequence_length {info.sequence_length}")
    print(f"head_count {info.head_count} head_dim {info.head_dim} hidden_size {info.hidden_size}")
    print(f"rope_theta {info.rope_theta:g}")
    print(f"reference pytorch_rocm exported_aten_sequence {torch.cuda.get_device_name(0)}")

    kernels = compile_rope_export_python_kernel(
        kernel_py=args.kernel_py,
        function_name=function_name,
        work_dir=args.work_dir,
        instance_name=function_name,
        eps=args.rms_norm_eps,
    )
    actual, latencies_ms = run_on_npu(
        kernels=kernels,
        source=source,
        norm_weight=norm_weight,
        rope_lut=rope_lut,
        expected=expected,
        stage=args.stage,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )
    max_abs = max_abs_rocm(actual, expected)
    print(f"rms_source_mlir_cache {kernels.rms_source_mlir}")
    print(f"rms_aie_mlir_cache {kernels.rms_aie_mlir}")
    print(f"rms_xclbin {kernels.rms_xclbin}")
    print(f"rms_insts {kernels.rms_insts}")
    print(f"rope_source_mlir_cache {kernels.rope_source_mlir}")
    print(f"rope_aie_mlir_cache {kernels.rope_aie_mlir}")
    print(f"rope_xclbin {kernels.rope_xclbin}")
    print(f"rope_insts {kernels.rope_insts}")
    print(f"actual_first8 {float32_host_array(actual).reshape(-1)[:8].tolist()}")
    print(f"expected_first8 {first_values(expected)}")
    print(f"max_abs {max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
