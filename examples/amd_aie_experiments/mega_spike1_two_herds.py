#!/usr/bin/env python3
import argparse
import os
import sysconfig
from pathlib import Path

import numpy as np
import torch
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.dialects import arith
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air.ir import IntegerAttr, MemRefType

range_ = for_


def prepend_air_tool_paths() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    paths = [
        Path(os.environ.get("MLIR_AIR_INSTALL_DIR", site_packages / "mlir_air")) / "bin",
        Path(os.environ.get("MLIR_AIE_INSTALL_DIR", site_packages / "mlir_aie")) / "bin",
        Path(os.environ.get("PEANO_INSTALL_DIR", site_packages / "llvm-aie")) / "bin",
    ]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join(str(path) for path in paths) + ":" + existing_path


def build_module(tile_size: int):
    @module_builder
    def build():
        xrt_dtype = type_mapper(np.int32)
        l3_type = MemRefType.get([tile_size], xrt_dtype)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
        l1_type = MemRefType.get([tile_size], xrt_dtype, memory_space=l1_space)

        @FuncOp.from_py_func(l3_type, l3_type, l3_type, l3_type)
        def two_independent_herds(add_input, mul_input, add_output, mul_output):
            @launch(operands=[add_input, mul_input, add_output, mul_output])
            def launch_body(add_l3, mul_l3, add_out_l3, mul_out_l3):
                @segment(name="seg", operands=[add_l3, mul_l3, add_out_l3, mul_out_l3])
                def segment_body(add_seg, mul_seg, add_out_seg, mul_out_seg):
                    @herd(name="add_herd", sizes=[1, 1], operands=[add_seg, add_out_seg])
                    def add_herd_body(_tx, _ty, _sx, _sy, add_arg, add_out_arg):
                        tile_in = AllocOp(l1_type, [], [])
                        tile_out = AllocOp(l1_type, [], [])
                        dma_memcpy_nd(
                            tile_in,
                            add_arg,
                            src_offsets=[arith.ConstantOp(T.index(), 0)],
                            src_sizes=[tile_size],
                            src_strides=[1],
                        )
                        for i in range_(tile_size):
                            value = load(tile_in, [i])
                            result = arith.addi(value, arith.constant(xrt_dtype, 1))
                            store(result, tile_out, [i])
                            yield_([])
                        dma_memcpy_nd(
                            add_out_arg,
                            tile_out,
                            dst_offsets=[arith.ConstantOp(T.index(), 0)],
                            dst_sizes=[tile_size],
                            dst_strides=[1],
                        )
                        DeallocOp(tile_in)
                        DeallocOp(tile_out)

                    @herd(name="mul_herd", sizes=[1, 1], operands=[mul_seg, mul_out_seg])
                    def mul_herd_body(_tx, _ty, _sx, _sy, mul_arg, mul_out_arg):
                        tile_in = AllocOp(l1_type, [], [])
                        tile_out = AllocOp(l1_type, [], [])
                        dma_memcpy_nd(
                            tile_in,
                            mul_arg,
                            src_offsets=[arith.ConstantOp(T.index(), 0)],
                            src_sizes=[tile_size],
                            src_strides=[1],
                        )
                        for i in range_(tile_size):
                            value = load(tile_in, [i])
                            result = arith.muli(value, arith.constant(xrt_dtype, 2))
                            store(result, tile_out, [i])
                            yield_([])
                        dma_memcpy_nd(
                            mul_out_arg,
                            tile_out,
                            dst_offsets=[arith.ConstantOp(T.index(), 0)],
                            dst_sizes=[tile_size],
                            dst_strides=[1],
                        )
                        DeallocOp(tile_in)
                        DeallocOp(tile_out)

    return build()


def build_reference(tile_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is required for the Spike 1 reference.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    add_input = torch.arange(tile_size, dtype=torch.int32, device=device)
    mul_input = torch.arange(tile_size, dtype=torch.int32, device=device) - 7
    add_output = add_input + 1
    mul_output = mul_input * 2
    return (
        add_input.cpu().numpy(),
        mul_input.cpu().numpy(),
        add_output.cpu().numpy(),
        mul_output.cpu().numpy(),
        device_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike 1: one xclbin with two independent herds.")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--work-dir", type=Path, default=Path(".cache/npu-spikes/mega-spike1-two-herds"))
    parser.add_argument("--output-format", choices=["xclbin"], default="xclbin")
    args = parser.parse_args()

    prepend_air_tool_paths()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)

    mlir_module = build_module(args.tile_size)
    Path("source.air.mlir").write_text(str(mlir_module))
    add_input, mul_input, add_expected, mul_expected, device_name = build_reference(args.tile_size)

    runner = XRTRunner(
        output_format=args.output_format,
        instance_name="two_independent_herds",
        target_device="npu2_4col",
    )
    result = runner.run_test(
        mlir_module,
        inputs=[add_input, mul_input],
        expected_outputs=[add_expected, mul_expected],
    )
    print(f"reference_device {device_name}")
    print(f"work_dir {Path.cwd()}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
