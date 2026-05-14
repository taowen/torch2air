from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact

from torch2air.export.q4k_linear_spike5 import (
    Q4K_LINEAR_SPIKE5_FUNCTION,
    Q4K_LINEAR_SPIKE5_HIDDEN_SIZE,
    Q4K_LINEAR_SPIKE5_OUTPUT_TILE_ROWS,
    Q4K_LINEAR_SPIKE5_SEQUENCE_LENGTH,
    build_q4k_linear_spike5_air,
    q4k_linear_spike5_herd_cols,
    q4k_linear_spike5_herd_rows,
)
from torch2air.runtime.compile import compile_runtime, lower_scf_air_to_aie, prepend_air_tool_paths
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
    compile_q4k_linear_spike2_object,
)


@dataclass(frozen=True, slots=True)
class Q4KLinearSpike5InputInfo:
    source: GGUFTensorEntry
    rms_weight: GGUFTensorEntry
    q_proj_weight: GGUFTensorEntry
    token_ids: list[int]
    hidden_size: int
    output_tile_rows: int
    blocks_per_row: int
    weight_words: int


def compile_q4k_linear_spike5_kernel(
    *,
    work_dir: Path,
    function_name: str,
    sequence_length: int,
    hidden_size: int,
    output_tile_rows: int,
) -> tuple[Path, Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    module = build_q4k_linear_spike5_air(
        function_name=function_name,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    source_mlir = work_dir / f"{function_name}.air.mlir"
    source_mlir.write_text(str(module), encoding="utf-8")
    object_file = compile_q4k_linear_spike2_object(
        work_dir=work_dir,
        peano_install_dir=peano,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    aie_mlir = lower_scf_air_to_aie(
        source_mlir=source_mlir,
        work_dir=work_dir,
        stem=function_name,
        herd_rows=q4k_linear_spike5_herd_rows(sequence_length),
        herd_cols=q4k_linear_spike5_herd_cols(sequence_length),
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=function_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts, object_file


def prepare_q4k_linear_spike5_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    q_proj_tensor: str,
    output_tile_rows: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, Q4KLinearSpike5InputInfo]:
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
    if output_tile_rows > output_features:
        raise ValueError(f"output_tile_rows={output_tile_rows} exceeds {output_features}")
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
            f"Spike 5 expects full rows with {model_blocks_per_row} Q4_K blocks, "
            f"got {blocks_per_row}"
        )

    row_bytes = row_words * 4
    payload = read_tensor_bytes(
        index.path,
        weight_entry,
        offset=0,
        size=output_tile_rows * row_bytes,
    )
    raw_blocks = np.frombuffer(payload, dtype=np.uint8).copy().reshape(
        output_tile_rows * model_blocks_per_row,
        144,
    )
    packed_weight = _append_q4k_scale_bits(
        packed_rows=np.frombuffer(payload, dtype=np.int32).copy().reshape(
            output_tile_rows,
            row_words,
        ),
        raw_blocks=raw_blocks,
        output_tile_rows=output_tile_rows,
        blocks_per_row=model_blocks_per_row,
    )
    reference = _q4k_linear_reference_module_rocm(
        raw_blocks=raw_blocks,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    expected = reference(
        normed_hidden.reshape(1, len(token_ids), hidden_size).to(
            device=rocm_device(),
            dtype=torch.float32,
        )
    ).reshape(len(token_ids), output_tile_rows)
    info = Q4KLinearSpike5InputInfo(
        source=layernorm_info.source,
        rms_weight=layernorm_info.rms_weight,
        q_proj_weight=weight_entry,
        token_ids=token_ids,
        hidden_size=hidden_size,
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


def _append_q4k_scale_bits(
    *,
    packed_rows: np.ndarray,
    raw_blocks: np.ndarray,
    output_tile_rows: int,
    blocks_per_row: int,
) -> np.ndarray:
    row_words = blocks_per_row * 36
    scale_values = (
        q4k_block_f16_scales_rocm(raw_blocks)
        .reshape(output_tile_rows, blocks_per_row, 2)
        .detach()
        .cpu()
        .numpy()
    )
    scale_bits = np.ascontiguousarray(scale_values.astype(np.float32, copy=False)).view(np.int32)
    packed = np.empty((output_tile_rows, row_words + blocks_per_row * 2), dtype=np.int32)
    packed[:, :row_words] = packed_rows
    packed[:, row_words:] = scale_bits.reshape(output_tile_rows, blocks_per_row * 2)
    return packed


def _q4k_linear_reference_module_rocm(
    *,
    raw_blocks: np.ndarray,
    hidden_size: int,
    output_tile_rows: int,
) -> torch.nn.Linear:
    rows = dequantize_q4_k_blocks_rocm(raw_blocks).reshape(output_tile_rows, hidden_size)
    module = torch.nn.Linear(
        hidden_size,
        output_tile_rows,
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
    output = np.zeros(expected_shape, dtype=np.float32)
    actual = output
    latencies_ms: list[float] = []
    try:
        for _ in range(warmup):
            output.fill(0)
            actual = np.asarray(func(hidden, packed_weight, output)[2]).reshape(expected_shape)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike5")

        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(hidden, packed_weight, output)[2]).reshape(expected_shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike5")
    finally:
        backend.unload()
    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Q4_K linear Spike 5 prefill chunk test.")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/npu-spikes/q4k-linear-spike5"))
    parser.add_argument("--function-name", default=Q4K_LINEAR_SPIKE5_FUNCTION)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0,1,2,3,4,5,6,7"))
    parser.add_argument("--blocks-per-row", type=int, default=4)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--q-proj-tensor", default=DEFAULT_Q_PROJ_TENSOR)
    parser.add_argument("--sequence-length", type=int, default=Q4K_LINEAR_SPIKE5_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=Q4K_LINEAR_SPIKE5_HIDDEN_SIZE)
    parser.add_argument("--output-tile-rows", type=int, default=Q4K_LINEAR_SPIKE5_OUTPUT_TILE_ROWS)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    if len(args.token_ids) != args.sequence_length:
        raise SystemExit("token count must match sequence-length")
    hidden, packed_weight, expected, info = prepare_q4k_linear_spike5_inputs(
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        blocks_per_row=args.blocks_per_row,
        rms_weight_tensor=args.rms_weight_tensor,
        q_proj_tensor=args.q_proj_tensor,
        output_tile_rows=args.output_tile_rows,
        eps=args.rms_norm_eps,
    )
    if info.hidden_size != args.hidden_size:
        raise SystemExit("hidden-size must match q_proj logical input size")
    source_mlir, aie_mlir, xclbin, insts, object_file = compile_q4k_linear_spike5_kernel(
        work_dir=args.work_dir,
        function_name=args.function_name,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        output_tile_rows=args.output_tile_rows,
    )
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=args.function_name,
        hidden=hidden,
        packed_weight=packed_weight,
        expected=expected,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = max_abs_rocm(actual, expected)
    print(f"input_source {info.source.name} {info.source.ggml_type}")
    print(f"RMS weight {info.rms_weight.name} {info.rms_weight.ggml_type}")
    print(f"Q4_K weight {info.q_proj_weight.name} {info.q_proj_weight.ggml_type}")
    print(f"token_ids {','.join(str(v) for v in info.token_ids)}")
    print(f"sequence_length {len(info.token_ids)} hidden_size {info.hidden_size}")
    print(f"output_tile_rows {info.output_tile_rows} blocks_per_row {info.blocks_per_row}")
    print(f"weight_words {info.weight_words}")
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
