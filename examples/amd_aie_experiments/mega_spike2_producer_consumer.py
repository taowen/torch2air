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
    Channel,
    ChannelGet,
    ChannelPut,
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

        Channel("ProducerToConsumer")

        @FuncOp.from_py_func(l3_type, l3_type)
        def producer_consumer(input_l3, output_l3):
            @launch(operands=[input_l3, output_l3])
            def launch_body(input_arg, output_arg):
                @segment(name="seg", operands=[input_arg, output_arg])
                def segment_body(input_seg, output_seg):
                    @herd(name="producer", sizes=[1, 1], operands=[input_seg])
                    def producer_body(_tx, _ty, _sx, _sy, producer_input):
                        tile_in = AllocOp(l1_type, [], [])
                        tile_mid = AllocOp(l1_type, [], [])
                        dma_memcpy_nd(
                            tile_in,
                            producer_input,
                            src_offsets=[arith.ConstantOp(T.index(), 0)],
                            src_sizes=[tile_size],
                            src_strides=[1],
                        )
                        for i in range_(tile_size):
                            value = load(tile_in, [i])
                            result = arith.muli(value, arith.constant(xrt_dtype, 2))
                            store(result, tile_mid, [i])
                            yield_([])
                        ChannelPut("ProducerToConsumer", tile_mid)
                        DeallocOp(tile_in)
                        DeallocOp(tile_mid)

                    @herd(name="consumer", sizes=[1, 1], operands=[output_seg])
                    def consumer_body(_tx, _ty, _sx, _sy, consumer_output):
                        tile_mid = AllocOp(l1_type, [], [])
                        tile_out = AllocOp(l1_type, [], [])
                        ChannelGet("ProducerToConsumer", tile_mid)
                        for i in range_(tile_size):
                            value = load(tile_mid, [i])
                            result = arith.addi(value, arith.constant(xrt_dtype, 1))
                            store(result, tile_out, [i])
                            yield_([])
                        dma_memcpy_nd(
                            consumer_output,
                            tile_out,
                            dst_offsets=[arith.ConstantOp(T.index(), 0)],
                            dst_sizes=[tile_size],
                            dst_strides=[1],
                        )
                        DeallocOp(tile_mid)
                        DeallocOp(tile_out)

    return build()


def build_reference(tile_size: int) -> tuple[np.ndarray, np.ndarray, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is required for the Spike 2 reference.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    input_tensor = torch.arange(tile_size, dtype=torch.int32, device=device) - 11
    output_tensor = input_tensor * 2 + 1
    return input_tensor.cpu().numpy(), output_tensor.cpu().numpy(), device_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike 2: producer-consumer AIR channel.")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".cache/npu-spikes/mega-spike2-producer-consumer"),
    )
    parser.add_argument("--output-format", choices=["xclbin"], default="xclbin")
    args = parser.parse_args()

    prepend_air_tool_paths()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)

    mlir_module = build_module(args.tile_size)
    Path("source.air.mlir").write_text(str(mlir_module))
    input_data, expected_output, device_name = build_reference(args.tile_size)

    runner = XRTRunner(
        output_format=args.output_format,
        instance_name="producer_consumer",
        target_device="npu2_4col",
    )
    result = runner.run_test(
        mlir_module,
        inputs=[input_data],
        expected_outputs=[expected_output],
    )
    print(f"reference_device {device_name}")
    print(f"work_dir {Path.cwd()}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
