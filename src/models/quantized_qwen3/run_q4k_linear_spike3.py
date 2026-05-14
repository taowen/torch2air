from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact

from torch2air.export.q4k_linear_spike2 import Q4K_LINEAR_SPIKE2_OUTPUT_TILE_ROWS
from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index, read_tensor_bytes

from .reference_runtime import (
    check_close_rocm,
    dequantize_q4_k_blocks_rocm,
    first_values,
    max_abs_rocm,
    q4k_block_f16_scales_rocm,
    rocm_device,
)
from .run_embed_tokens import DEFAULT_GGUF, parse_token_ids
from .run_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR, prepare_layernorm_inputs
from .run_q4k_linear_spike2 import (
    DEFAULT_Q_PROJ_TENSOR,
    compile_q4k_linear_spike2_kernel,
)

Q4K_LINEAR_SPIKE3_FUNCTION = "q4k_linear_spike3"


@dataclass(frozen=True, slots=True)
class Q4KLinearSpike3InputInfo:
    source: GGUFTensorEntry
    rms_weight: GGUFTensorEntry
    q_proj_weight: GGUFTensorEntry
    token_ids: list[int]
    hidden_size: int
    output_features: int
    output_tile_rows: int
    blocks_per_row: int
    weight_words: int


def prepare_q4k_linear_spike3_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    q_proj_tensor: str,
    output_tile_rows: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, Q4KLinearSpike3InputInfo]:
    if len(token_ids) != 1:
        raise ValueError("Spike 3 is decode-only and expects exactly one token")

    _, _, normed_hidden, layernorm_info = prepare_layernorm_inputs(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[q_proj_tensor]
    if weight_entry.ggml_type != "Q4_K" or weight_entry.physical_dtype != "uint32":
        raise ValueError(f"{q_proj_tensor} must be Q4_K uint32, got {weight_entry}")
    output_features, row_words = (int(dim) for dim in weight_entry.physical_shape)
    if output_features % output_tile_rows != 0:
        raise ValueError(
            f"output_features={output_features} must be divisible by {output_tile_rows}"
        )
    if row_words % 36 != 0:
        raise ValueError(f"Q4_K row word width must be a multiple of 36, got {row_words}")
    model_blocks_per_row = row_words // 36
    hidden_size = model_blocks_per_row * 256
    if hidden_size != layernorm_info.hidden_size:
        raise ValueError(
            f"q_proj hidden_size={hidden_size} does not match "
            f"layernorm hidden_size={layernorm_info.hidden_size}"
        )
    if blocks_per_row != model_blocks_per_row:
        raise ValueError(
            f"Spike 3 expects full rows with {model_blocks_per_row} Q4_K blocks, "
            f"got {blocks_per_row}"
        )

    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=weight_entry.nbytes)
    raw_blocks = np.frombuffer(payload, dtype=np.uint8).copy().reshape(
        output_features * model_blocks_per_row,
        144,
    )
    packed_weight = _append_all_q4k_scale_bits(
        packed_rows=np.frombuffer(payload, dtype=np.int32).copy().reshape(
            output_features,
            row_words,
        ),
        raw_blocks=raw_blocks,
        output_features=output_features,
        blocks_per_row=model_blocks_per_row,
    )
    reference = _q4k_linear_reference_module_rocm(
        raw_blocks=raw_blocks,
        hidden_size=hidden_size,
        output_features=output_features,
    )
    expected = reference(
        normed_hidden.reshape(1, 1, hidden_size).to(device=rocm_device(), dtype=torch.float32)
    ).reshape(1, output_features)
    info = Q4KLinearSpike3InputInfo(
        source=layernorm_info.source,
        rms_weight=layernorm_info.rms_weight,
        q_proj_weight=weight_entry,
        token_ids=token_ids,
        hidden_size=hidden_size,
        output_features=output_features,
        output_tile_rows=output_tile_rows,
        blocks_per_row=model_blocks_per_row,
        weight_words=packed_weight.shape[1],
    )
    return (
        np.ascontiguousarray(normed_hidden.detach().cpu().numpy().astype(np.float32, copy=False)),
        np.ascontiguousarray(packed_weight),
        expected,
        info,
    )


def _append_all_q4k_scale_bits(
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


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
    expected: torch.Tensor,
    output_tile_rows: int,
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
        for _ in range(warmup):
            actual = _run_output_chunks(
                func=func,
                hidden=hidden,
                packed_weight=packed_weight,
                output_tile_rows=output_tile_rows,
                expected_shape=expected_shape,
            )
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike3")

        for _ in range(iterations):
            start = time.perf_counter()
            actual = _run_output_chunks(
                func=func,
                hidden=hidden,
                packed_weight=packed_weight,
                output_tile_rows=output_tile_rows,
                expected_shape=expected_shape,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike3")
    finally:
        backend.unload()
    return actual, latencies_ms


def _run_output_chunks(
    *,
    func,
    hidden: np.ndarray,
    packed_weight: np.ndarray,
    output_tile_rows: int,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    output_features = expected_shape[1]
    actual = np.zeros(expected_shape, dtype=np.float32)
    output_chunk = np.zeros((1, output_tile_rows), dtype=np.float32)
    for row_offset in range(0, output_features, output_tile_rows):
        output_chunk.fill(0)
        weight_chunk = np.ascontiguousarray(
            packed_weight[row_offset : row_offset + output_tile_rows],
        )
        result = np.asarray(func(hidden, weight_chunk, output_chunk)[2]).reshape(
            output_chunk.shape
        )
        actual[:, row_offset : row_offset + output_tile_rows] = result
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full q_proj via Q4_K output chunks.")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/npu-spikes/q4k-linear-spike3"))
    parser.add_argument("--function-name", default=Q4K_LINEAR_SPIKE3_FUNCTION)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, default=4)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--q-proj-tensor", default=DEFAULT_Q_PROJ_TENSOR)
    parser.add_argument("--output-tile-rows", type=int, default=Q4K_LINEAR_SPIKE2_OUTPUT_TILE_ROWS)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    hidden, packed_weight, expected, info = prepare_q4k_linear_spike3_inputs(
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        blocks_per_row=args.blocks_per_row,
        rms_weight_tensor=args.rms_weight_tensor,
        q_proj_tensor=args.q_proj_tensor,
        output_tile_rows=args.output_tile_rows,
        eps=args.rms_norm_eps,
    )
    source_mlir, aie_mlir, xclbin, insts, object_file = compile_q4k_linear_spike2_kernel(
        work_dir=args.work_dir,
        function_name=args.function_name,
        hidden_size=info.hidden_size,
        output_tile_rows=args.output_tile_rows,
    )
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=args.function_name,
        hidden=hidden,
        packed_weight=packed_weight,
        expected=expected,
        output_tile_rows=args.output_tile_rows,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = max_abs_rocm(actual, expected)
    chunk_count = info.output_features // info.output_tile_rows
    print(f"input_source {info.source.name} {info.source.ggml_type}")
    print(f"RMS weight {info.rms_weight.name} {info.rms_weight.ggml_type}")
    print(f"Q4_K weight {info.q_proj_weight.name} {info.q_proj_weight.ggml_type}")
    print(f"token_ids {','.join(str(v) for v in info.token_ids)}")
    print(f"hidden_size {info.hidden_size} output_features {info.output_features}")
    print(f"output_tile_rows {info.output_tile_rows} chunk_count {chunk_count}")
    print(f"blocks_per_row {info.blocks_per_row} weight_words {info.weight_words}")
    print(f"reference pytorch_rocm torch.nn.Linear {torch.cuda.get_device_name(0)}")
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


if __name__ == "__main__":
    raise SystemExit(main())
