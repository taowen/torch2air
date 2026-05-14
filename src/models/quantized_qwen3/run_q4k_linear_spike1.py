from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt import XRTBackend, XRTCompileArtifact

from torch2air.export.q4k_linear_spike1 import (
    Q4K_LINEAR_SPIKE1_FUNCTION,
    Q4K_LINEAR_SPIKE1_HIDDEN_SIZE,
    Q4K_LINEAR_SPIKE1_LINK_OBJECT,
    Q4K_LINEAR_SPIKE1_OUTPUT_TILE_ROWS,
    build_q4k_linear_spike1_air,
)
from torch2air.runtime.compile import (
    compile_runtime,
    installed_tool,
    lower_scf_air_to_aie,
    prepend_air_tool_paths,
)

from .reference_runtime import check_close_rocm, first_values, max_abs_rocm

KERNEL_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "torch2air"
    / "export"
    / "kernels"
    / "q4k_linear_spike1.cc"
)


def compile_q4k_linear_spike1_kernel(
    *,
    work_dir: Path,
    function_name: str,
    hidden_size: int,
    output_tile_rows: int,
) -> tuple[Path, Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    module = build_q4k_linear_spike1_air(
        function_name=function_name,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    source_mlir = work_dir / f"{function_name}.air.mlir"
    source_mlir.write_text(str(module), encoding="utf-8")
    object_file = compile_q4k_linear_spike1_object(
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
        herd_cols=1,
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=function_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts, object_file


def compile_q4k_linear_spike1_object(
    *,
    work_dir: Path,
    peano_install_dir: Path,
    hidden_size: int,
    output_tile_rows: int,
) -> Path:
    object_file = work_dir / Q4K_LINEAR_SPIKE1_LINK_OBJECT
    aieopt_dir = Path(installed_tool("aie-opt", "MLIR_AIE_INSTALL_DIR")).resolve().parent.parent
    target_triple = f"{os.environ.get('AIE_TARGET', 'aie2p')}-none-unknown-elf"
    subprocess.run(
        [
            str(peano_install_dir / "bin" / "clang++"),
            "-O2",
            "-std=c++20",
            f"--target={target_triple}",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-Wno-unused-command-line-argument",
            "-DNDEBUG",
            f"-I{aieopt_dir / 'include'}",
            f"-DHIDDEN_SIZE={hidden_size}",
            f"-DOUTPUT_TILE_ROWS={output_tile_rows}",
            "-c",
            str(KERNEL_SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
        cwd=work_dir,
    )
    return object_file


def build_reference(
    *,
    hidden_size: int,
    output_tile_rows: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm reference requires torch.cuda.is_available()")
    if hidden_size % output_tile_rows != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by output_tile_rows={output_tile_rows}"
        )

    device = torch.device("cuda")
    hidden = torch.linspace(-0.5, 0.5, hidden_size, device=device, dtype=torch.float32).reshape(
        1,
        hidden_size,
    )
    weight = torch.arange(output_tile_rows, device=device, dtype=torch.int32) * 3 - 5
    offsets = torch.arange(output_tile_rows, device=device, dtype=torch.long) * (
        hidden_size // output_tile_rows
    )
    expected = hidden[0, offsets] + weight.to(torch.float32)
    return (
        np.ascontiguousarray(hidden.detach().cpu().numpy()),
        np.ascontiguousarray(weight.detach().cpu().numpy()),
        expected.reshape(1, output_tile_rows),
        torch.cuda.get_device_name(device),
    )


def run_on_npu(
    *,
    xclbin: Path,
    insts: Path,
    instance_name: str,
    hidden: np.ndarray,
    weight: np.ndarray,
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
            actual = np.asarray(func(hidden, weight, output)[2]).reshape(expected_shape)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike1")

        for _ in range(iterations):
            output.fill(0)
            start = time.perf_counter()
            actual = np.asarray(func(hidden, weight, output)[2]).reshape(expected_shape)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="q4k_linear_spike1")
    finally:
        backend.unload()
    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Q4_K linear Spike 1 external ABI test.")
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/npu-spikes/q4k-linear-spike1"))
    parser.add_argument("--function-name", default=Q4K_LINEAR_SPIKE1_FUNCTION)
    parser.add_argument("--hidden-size", type=int, default=Q4K_LINEAR_SPIKE1_HIDDEN_SIZE)
    parser.add_argument("--output-tile-rows", type=int, default=Q4K_LINEAR_SPIKE1_OUTPUT_TILE_ROWS)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    hidden, weight, expected, device_name = build_reference(
        hidden_size=args.hidden_size,
        output_tile_rows=args.output_tile_rows,
    )
    source_mlir, aie_mlir, xclbin, insts, object_file = compile_q4k_linear_spike1_kernel(
        work_dir=args.work_dir,
        function_name=args.function_name,
        hidden_size=args.hidden_size,
        output_tile_rows=args.output_tile_rows,
    )
    actual, latencies_ms = run_on_npu(
        xclbin=xclbin,
        insts=insts,
        instance_name=args.function_name,
        hidden=hidden,
        weight=weight,
        expected=expected,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    max_abs = max_abs_rocm(actual, expected)
    print(f"reference_device {device_name}")
    print(f"hidden_shape {hidden.shape} weight_shape {weight.shape} output_shape {actual.shape}")
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
