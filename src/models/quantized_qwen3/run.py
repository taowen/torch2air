"""Run exported quantized Qwen3 kernel tiles through IREE."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch

from models.quantized_qwen3.kernels import DEFAULT_OUT_DIR, KERNELS, Qwen3Kernel
from models.quantized_qwen3.modules import make_torch_inputs, reference_output


def run_kernel(
    kernel: Qwen3Kernel,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    vmfb: Path | None = None,
    iree_run_module: str = "iree-run-module",
    device: str = "xrt-lite",
    xrt_lite_n_core_rows: int = 4,
    xrt_lite_n_core_cols: int = 8,
    tolerance: float = 0.0,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    lhs, rhs = make_torch_inputs(kernel)
    expected = reference_output(kernel).cpu()

    lhs_path = out_dir / f"{kernel.name}_lhs.npy"
    rhs_path = out_dir / f"{kernel.name}_rhs.npy"
    output_path = out_dir / f"{kernel.name}_output.npy"
    np.save(lhs_path, lhs.cpu().numpy())
    np.save(rhs_path, rhs.cpu().numpy())

    command = [
        iree_run_module,
        f"--device={device}",
        f"--module={vmfb or out_dir / kernel.vmfb_filename}",
        f"--function={kernel.function_name}",
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

    actual = torch.from_numpy(np.load(output_path)).cpu()
    max_abs_diff = torch.max(torch.abs(actual.to(torch.int64) - expected.to(torch.int64))).item()
    if max_abs_diff > tolerance:
        raise RuntimeError(f"{kernel.name} mismatch: max_abs_diff={max_abs_diff}")
    return float(max_abs_diff)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="prefill_q_proj_tile_i32", choices=sorted(KERNELS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vmfb", type=Path)
    parser.add_argument("--iree-run-module", default="iree-run-module")
    parser.add_argument("--device", default="xrt-lite")
    parser.add_argument("--xrt-lite-n-core-rows", type=int, default=4)
    parser.add_argument("--xrt-lite-n-core-cols", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()

    max_abs_diff = run_kernel(
        KERNELS[args.kernel],
        out_dir=args.out_dir,
        vmfb=args.vmfb,
        iree_run_module=args.iree_run_module,
        device=args.device,
        xrt_lite_n_core_rows=args.xrt_lite_n_core_rows,
        xrt_lite_n_core_cols=args.xrt_lite_n_core_cols,
        tolerance=args.tolerance,
    )
    print(f"{args.kernel}: max_abs_diff={max_abs_diff}")


if __name__ == "__main__":
    main()
