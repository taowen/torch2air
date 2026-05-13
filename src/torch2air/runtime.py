from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from torch2air.demo_models import MatmulModel, make_matmul_inputs
from torch2air.exporter import Frontend, export_model_to_linalg_mlir, write_exported_mlir
from torch2air.iree_air import IreeAirConfig, lower_linalg_to_air


def export_demo(
    *,
    out_dir: Path,
    frontend: Frontend | str = Frontend.AUTO,
    function_name: str = "forward",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lhs, rhs = make_matmul_inputs()
    exported = export_model_to_linalg_mlir(
        MatmulModel(),
        (lhs, rhs),
        function_name=function_name,
        frontend=frontend,
    )
    linalg_mlir = out_dir / "matmul_i32_32.linalg.mlir"
    write_exported_mlir(exported, linalg_mlir)
    return linalg_mlir, out_dir / "matmul_i32_32.air.mlir"


def lower_demo(
    *,
    out_dir: Path,
    iree_compile: str = "iree-compile",
    build_vmfb: bool = True,
) -> tuple[Path, Path | None]:
    source_mlir = out_dir / "matmul_i32_32.linalg.mlir"
    air_mlir = out_dir / "matmul_i32_32.air.mlir"
    vmfb = out_dir / "matmul_i32_32.vmfb" if build_vmfb else None
    lower_linalg_to_air(
        source_mlir,
        air_mlir,
        vmfb=vmfb,
        config=IreeAirConfig(iree_compile=iree_compile),
    )
    return air_mlir, vmfb


def run_demo(
    *,
    out_dir: Path,
    iree_run_module: str = "iree-run-module",
    device: str = "xrt-lite",
    function_name: str = "forward",
    xrt_lite_n_core_rows: int = 4,
    xrt_lite_n_core_cols: int = 4,
    tolerance: float = 0.0,
) -> float:
    lhs, rhs = make_matmul_inputs()
    expected = MatmulModel()(lhs, rhs).cpu().numpy()

    lhs_path = out_dir / "matmul_i32_32_lhs.npy"
    rhs_path = out_dir / "matmul_i32_32_rhs.npy"
    output_path = out_dir / "matmul_i32_32_output.npy"
    np.save(lhs_path, lhs.cpu().numpy())
    np.save(rhs_path, rhs.cpu().numpy())

    command = [
        iree_run_module,
        f"--device={device}",
        f"--module={out_dir / 'matmul_i32_32.vmfb'}",
        f"--function={function_name}",
        f"--input=@{lhs_path}",
        f"--input=@{rhs_path}",
        f"--output=@{output_path}",
    ]
    if device == "xrt-lite":
        command.extend(
            [
                f"--xrt_lite_n_core_rows={xrt_lite_n_core_rows}",
                f"--xrt_lite_n_core_cols={xrt_lite_n_core_cols}",
            ]
        )
    subprocess.run(command, check=True)
    actual = np.load(output_path)
    max_abs_diff = float(np.max(np.abs(actual - expected)))
    if max_abs_diff > tolerance:
        raise RuntimeError(f"NPU result mismatch: max_abs_diff={max_abs_diff}")
    return max_abs_diff
