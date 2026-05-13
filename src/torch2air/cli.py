from __future__ import annotations

import argparse
from pathlib import Path

from torch2air.exporter import Frontend
from torch2air.iree_air import IreeAirConfig, lower_linalg_to_air
from torch2air.runtime import export_demo, lower_demo, run_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="torch2air")
    subcommands = parser.add_subparsers(dest="command", required=True)

    export_parser = subcommands.add_parser("export-demo")
    export_parser.add_argument("--out-dir", type=Path, default=Path("examples/generated"))
    export_parser.add_argument("--frontend", choices=[item.value for item in Frontend], default="auto")

    lower_parser = subcommands.add_parser("lower-air")
    lower_parser.add_argument("source_mlir", type=Path)
    lower_parser.add_argument("--air-mlir", type=Path, required=True)
    lower_parser.add_argument("--vmfb", type=Path)
    lower_parser.add_argument("--iree-compile", default="iree-compile")

    lower_demo_parser = subcommands.add_parser("lower-demo")
    lower_demo_parser.add_argument("--out-dir", type=Path, default=Path("examples/generated"))
    lower_demo_parser.add_argument("--iree-compile", default="iree-compile")
    lower_demo_parser.add_argument("--no-vmfb", action="store_true")

    run_parser = subcommands.add_parser("run-demo")
    run_parser.add_argument("--out-dir", type=Path, default=Path("examples/generated"))
    run_parser.add_argument("--iree-run-module", default="iree-run-module")
    run_parser.add_argument("--device", default="xrt-lite")
    run_parser.add_argument("--xrt-lite-n-core-rows", type=int, default=4)
    run_parser.add_argument("--xrt-lite-n-core-cols", type=int, default=4)
    run_parser.add_argument("--tolerance", type=float, default=0.0)
    run_parser.add_argument("--allow-mismatch", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "export-demo":
        linalg_mlir, air_mlir = export_demo(out_dir=args.out_dir, frontend=args.frontend)
        print(f"wrote {linalg_mlir}")
        print(f"next AIR path: {air_mlir}")
        return 0

    if args.command == "lower-air":
        lower_linalg_to_air(
            args.source_mlir,
            args.air_mlir,
            vmfb=args.vmfb,
            config=IreeAirConfig(iree_compile=args.iree_compile),
        )
        print(f"wrote {args.air_mlir}")
        if args.vmfb is not None:
            print(f"wrote {args.vmfb}")
        return 0

    if args.command == "lower-demo":
        air_mlir, vmfb = lower_demo(
            out_dir=args.out_dir,
            iree_compile=args.iree_compile,
            build_vmfb=not args.no_vmfb,
        )
        print(f"wrote {air_mlir}")
        if vmfb is not None:
            print(f"wrote {vmfb}")
        return 0

    if args.command == "run-demo":
        max_abs_diff = run_demo(
            out_dir=args.out_dir,
            iree_run_module=args.iree_run_module,
            device=args.device,
            xrt_lite_n_core_rows=args.xrt_lite_n_core_rows,
            xrt_lite_n_core_cols=args.xrt_lite_n_core_cols,
            tolerance=float("inf") if args.allow_mismatch else args.tolerance,
        )
        print(f"max_abs_diff={max_abs_diff}")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
