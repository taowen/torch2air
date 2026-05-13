"""Export quantized Qwen3 kernel tiles to linalg MLIR, AIR MLIR, and VMFB."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.quantized_qwen3.kernels import DEFAULT_OUT_DIR, KERNELS, Qwen3Kernel
from models.quantized_qwen3.modules import Qwen3LinearTile, make_torch_inputs
from torch2air.exporter import Frontend, export_model_to_linalg_mlir, write_exported_mlir
from torch2air.iree_air import IreeAirConfig, lower_linalg_to_air


def export_kernel(
    kernel: Qwen3Kernel,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    iree_compile: str = "iree-compile",
    build_vmfb: bool = True,
    target_device: str | None = None,
    air_tile_pipeline: str | None = None,
    air_lower_to_aie_pipeline: str | None = None,
    vmfb_tile_pipeline: str | None = None,
    vmfb_lower_to_aie_pipeline: str | None = None,
) -> tuple[Path, Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    linalg_mlir = out_dir / kernel.linalg_filename
    air_mlir = out_dir / kernel.air_filename
    vmfb = out_dir / kernel.vmfb_filename if build_vmfb else None

    exported = export_model_to_linalg_mlir(
        Qwen3LinearTile(),
        make_torch_inputs(kernel),
        function_name=kernel.function_name,
        frontend=Frontend.FX,
    )
    write_exported_mlir(exported, linalg_mlir)
    lower_linalg_to_air(
        linalg_mlir,
        air_mlir,
        vmfb=vmfb,
        config=IreeAirConfig(
            iree_compile=iree_compile,
            **_config_kwargs(
                target_device=target_device,
                tile_pipeline=air_tile_pipeline,
                lower_to_aie_pipeline=air_lower_to_aie_pipeline,
            ),
        ),
        vmfb_config=IreeAirConfig(
            iree_compile=iree_compile,
            **_config_kwargs(
                target_device=target_device,
                tile_pipeline=vmfb_tile_pipeline or air_tile_pipeline,
                lower_to_aie_pipeline=vmfb_lower_to_aie_pipeline,
            ),
        )
        if vmfb_lower_to_aie_pipeline is not None or vmfb_tile_pipeline is not None
        else None,
    )
    return linalg_mlir, air_mlir, vmfb


def export_kernels(
    kernel_names: list[str],
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    iree_compile: str = "iree-compile",
    build_vmfb: bool = True,
    target_device: str | None = None,
    air_tile_pipeline: str | None = None,
    air_lower_to_aie_pipeline: str | None = None,
    vmfb_tile_pipeline: str | None = None,
    vmfb_lower_to_aie_pipeline: str | None = None,
) -> list[tuple[Path, Path, Path | None]]:
    return [
        export_kernel(
            KERNELS[name],
            out_dir=out_dir,
            iree_compile=iree_compile,
            build_vmfb=build_vmfb,
            target_device=target_device,
            air_tile_pipeline=air_tile_pipeline,
            air_lower_to_aie_pipeline=air_lower_to_aie_pipeline,
            vmfb_tile_pipeline=vmfb_tile_pipeline,
            vmfb_lower_to_aie_pipeline=vmfb_lower_to_aie_pipeline,
        )
        for name in kernel_names
    ]


def _config_kwargs(
    *,
    target_device: str | None,
    tile_pipeline: str | None,
    lower_to_aie_pipeline: str | None,
) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "target_device": target_device,
            "tile_pipeline": tile_pipeline,
            "lower_to_aie_pipeline": lower_to_aie_pipeline,
        }.items()
        if value is not None
    }


def _selected_kernel_names(name: str) -> list[str]:
    if name == "all":
        return sorted(KERNELS)
    if name not in KERNELS:
        raise KeyError(f"unknown kernel {name!r}; available: {', '.join(sorted(KERNELS))}")
    return [name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--iree-compile", default="iree-compile")
    parser.add_argument("--target-device")
    parser.add_argument("--air-tile-pipeline")
    parser.add_argument("--air-lower-to-aie-pipeline")
    parser.add_argument("--vmfb-tile-pipeline")
    parser.add_argument("--vmfb-lower-to-aie-pipeline")
    parser.add_argument("--no-vmfb", action="store_true")
    args = parser.parse_args()

    for linalg_mlir, air_mlir, vmfb in export_kernels(
        _selected_kernel_names(args.kernel),
        out_dir=args.out_dir,
        iree_compile=args.iree_compile,
        build_vmfb=not args.no_vmfb,
        target_device=args.target_device,
        air_tile_pipeline=args.air_tile_pipeline,
        air_lower_to_aie_pipeline=args.air_lower_to_aie_pipeline,
        vmfb_tile_pipeline=args.vmfb_tile_pipeline,
        vmfb_lower_to_aie_pipeline=args.vmfb_lower_to_aie_pipeline,
    ):
        print(f"wrote {linalg_mlir}")
        print(f"wrote {air_mlir}")
        if vmfb is not None:
            print(f"wrote {vmfb}")


if __name__ == "__main__":
    main()
