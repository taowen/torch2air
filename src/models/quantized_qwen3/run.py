"""Run exported quantized Qwen3 kernel tiles through IREE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.quantized_qwen3.kernels import DEFAULT_OUT_DIR, KERNELS, Qwen3Kernel
from models.quantized_qwen3.modules import Qwen3LinearTile, make_torch_inputs


def _load_iree_module(
    vmfb: Path,
    *,
    device: str,
    xrt_lite_n_core_rows: int,
    xrt_lite_n_core_cols: int,
):
    import iree.runtime as ireert
    from iree.runtime import flags as iree_flags

    if device == "xrt-lite":
        iree_flags.parse_flags(
            f"--xrt_lite_n_core_rows={xrt_lite_n_core_rows}",
            f"--xrt_lite_n_core_cols={xrt_lite_n_core_cols}",
        )
    try:
        return ireert.load_vm_flatbuffer_file(str(vmfb), driver=device)
    except Exception as exc:
        available = ", ".join(ireert.query_available_drivers())
        raise RuntimeError(
            f"failed to load {vmfb} with IREE runtime device {device!r}; "
            f"available drivers: {available}"
        ) from exc


def _reference_device(name: str) -> torch.device:
    if name != "cuda":
        raise ValueError("quantized_qwen3 reference currently requires --reference-device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA PyTorch device is required for reference comparison")
    return torch.device(name)


def _reference_output(
    kernel: Qwen3Kernel,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    model = Qwen3LinearTile().eval().to(device)
    lhs = lhs.to(device=device, non_blocking=True)
    rhs = rhs.to(device=device, non_blocking=True)
    with torch.no_grad():
        return model(lhs, rhs).contiguous()


def run_kernel(
    kernel: Qwen3Kernel,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    vmfb: Path | None = None,
    device: str = "xrt-lite",
    reference_device: str = "cuda",
    xrt_lite_n_core_rows: int = 4,
    xrt_lite_n_core_cols: int = 8,
    tolerance: float = 0.0,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    lhs, rhs = make_torch_inputs(kernel)
    ref_device = _reference_device(reference_device)
    expected = _reference_output(kernel, lhs, rhs, device=ref_device)

    module_path = vmfb or out_dir / kernel.vmfb_filename
    if not module_path.exists():
        raise FileNotFoundError(f"VMFB not found: {module_path}")
    module = _load_iree_module(
        module_path,
        device=device,
        xrt_lite_n_core_rows=xrt_lite_n_core_rows,
        xrt_lite_n_core_cols=xrt_lite_n_core_cols,
    )
    entry = getattr(module, kernel.function_name)
    result = entry(lhs.cpu().numpy(), rhs.cpu().numpy())
    actual = torch.from_numpy(result.to_host()).to(device=ref_device, non_blocking=True)
    diff = torch.abs(actual - expected)
    max_abs_diff = float(torch.max(diff).item())
    if max_abs_diff > tolerance:
        raise RuntimeError(f"{kernel.name} mismatch: max_abs_diff={max_abs_diff}")
    return max_abs_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="prefill_q_proj_tile_f32", choices=sorted(KERNELS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vmfb", type=Path)
    parser.add_argument("--device", default="xrt-lite")
    parser.add_argument("--reference-device", default="cuda")
    parser.add_argument("--xrt-lite-n-core-rows", type=int, default=4)
    parser.add_argument("--xrt-lite-n-core-cols", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    max_abs_diff = run_kernel(
        KERNELS[args.kernel],
        out_dir=args.out_dir,
        vmfb=args.vmfb,
        device=args.device,
        reference_device=args.reference_device,
        xrt_lite_n_core_rows=args.xrt_lite_n_core_rows,
        xrt_lite_n_core_cols=args.xrt_lite_n_core_cols,
        tolerance=args.tolerance,
    )
    print(f"{args.kernel}: max_abs_diff={max_abs_diff}")


if __name__ == "__main__":
    main()
