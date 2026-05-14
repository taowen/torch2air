from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact

from torch2air.export.q4k_linear_spike4 import (
    Q4K_LINEAR_SPIKE4_FUNCTION,
    Q4K_LINEAR_SPIKE4_HERD_COLS,
    Q4K_LINEAR_SPIKE4_HIDDEN_SIZE,
    Q4K_LINEAR_SPIKE4_OUTPUT_TILE_ROWS,
    build_q4k_linear_spike4_air,
)
from torch2air.runtime.compile import compile_runtime, lower_scf_air_to_aie, prepend_air_tool_paths

from .reference_runtime import check_close_rocm, first_values, max_abs_rocm
from .run_embed_tokens import DEFAULT_GGUF, parse_token_ids
from .run_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR
from .run_q4k_linear_spike2 import (
    DEFAULT_Q_PROJ_TENSOR,
    compile_q4k_linear_spike2_object,
    prepare_q4k_linear_spike2_inputs,
)


def compile_q4k_linear_spike4_kernel(
    *,
    work_dir: Path,
    function_name: str,
    hidden_size: int,
    output_tile_rows: int,
    herd_cols: int,
) -> tuple[Path, Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    module = build_q4k_linear_spike4_air(
        function_name=function_name,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
        herd_cols=herd_cols,
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
        herd_rows=1,
        herd_cols=herd_cols,
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=function_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts, object_file


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
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike4")

        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(hidden, packed_weight, output)[2]).reshape(expected_shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike4")
    finally:
        backend.unload()
    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Q4_K linear Spike 4 multi-column test.")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/npu-spikes/q4k-linear-spike4"))
    parser.add_argument("--function-name", default=Q4K_LINEAR_SPIKE4_FUNCTION)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, default=4)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--q-proj-tensor", default=DEFAULT_Q_PROJ_TENSOR)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=Q4K_LINEAR_SPIKE4_HIDDEN_SIZE)
    parser.add_argument("--output-tile-rows", type=int, default=Q4K_LINEAR_SPIKE4_OUTPUT_TILE_ROWS)
    parser.add_argument("--herd-cols", type=int, default=Q4K_LINEAR_SPIKE4_HERD_COLS)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    output_rows = args.output_tile_rows * args.herd_cols
    hidden, packed_weight, expected, info = prepare_q4k_linear_spike2_inputs(
        gguf_path=args.gguf,
        token_ids=args.token_ids,
        blocks_per_row=args.blocks_per_row,
        rms_weight_tensor=args.rms_weight_tensor,
        q_proj_tensor=args.q_proj_tensor,
        row_offset=args.row_offset,
        output_tile_rows=output_rows,
        eps=args.rms_norm_eps,
    )
    if info.hidden_size != args.hidden_size:
        raise SystemExit("hidden-size must match q_proj logical input size")
    source_mlir, aie_mlir, xclbin, insts, object_file = compile_q4k_linear_spike4_kernel(
        work_dir=args.work_dir,
        function_name=args.function_name,
        hidden_size=args.hidden_size,
        output_tile_rows=args.output_tile_rows,
        herd_cols=args.herd_cols,
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
    print(f"row_offset {info.row_offset} output_rows {output_rows}")
    print(f"hidden_size {info.hidden_size} blocks_per_row {info.blocks_per_row}")
    print(f"output_tile_rows {args.output_tile_rows} herd_cols {args.herd_cols}")
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
