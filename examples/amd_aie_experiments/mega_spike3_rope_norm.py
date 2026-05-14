#!/usr/bin/env python3
import argparse
import os
import shutil
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
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air.ir import FunctionType, IntegerAttr, MemRefType, StringAttr, UnitAttr

from models.quantized_qwen3.stages.rope import (
    compile_rms_norm_rope_object,
    compile_rope_table_object,
)

range_ = for_


def prepend_air_tool_paths() -> tuple[Path, Path, Path]:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    mlir_air = Path(os.environ.get("MLIR_AIR_INSTALL_DIR", site_packages / "mlir_air"))
    mlir_aie = Path(os.environ.get("MLIR_AIE_INSTALL_DIR", site_packages / "mlir_aie"))
    peano = Path(os.environ.get("PEANO_INSTALL_DIR", site_packages / "llvm-aie"))
    paths = [mlir_air / "bin", mlir_aie / "bin", peano / "bin"]
    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ":".join(str(path) for path in paths) + ":" + existing_path
    os.environ["MLIR_AIR_INSTALL_DIR"] = str(mlir_air)
    os.environ["MLIR_AIE_INSTALL_DIR"] = str(mlir_aie)
    os.environ["PEANO_INSTALL_DIR"] = str(peano)
    return mlir_air, mlir_aie, peano


def external_func(name: str, inputs: list[MemRefType], link_with: str) -> FuncOp:
    func = FuncOp(name=name, type=FunctionType.get(inputs, []), visibility="private")
    func.attributes["llvm.emit_c_interface"] = UnitAttr.get()
    func.attributes["link_with"] = StringAttr.get(link_with)
    return func


def build_module(sequence_length: int, head_dim: int):
    @module_builder
    def build():
        f32 = type_mapper(np.float32)
        i32 = type_mapper(np.int32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        input_l3_t = MemRefType.get([sequence_length, head_dim], f32)
        weight_l3_t = MemRefType.get([head_dim], f32)
        start_l3_t = MemRefType.get([1], i32)
        output_l3_t = MemRefType.get([sequence_length, head_dim], f32)

        row_l1_t = MemRefType.get([1, head_dim], f32, memory_space=l1_space)
        trig_l1_t = MemRefType.get([2, head_dim], f32, memory_space=l1_space)
        weight_l1_t = MemRefType.get([head_dim], f32, memory_space=l1_space)
        start_l1_t = MemRefType.get([1], i32, memory_space=l1_space)

        external_func("rope_table_tile", [start_l1_t, row_l1_t, row_l1_t], "rope_table.o")
        external_func(
            "rms_norm_rope_tile",
            [row_l1_t, weight_l1_t, row_l1_t, row_l1_t, row_l1_t],
            "rms_norm_rope.o",
        )

        Channel("Start")
        Channel("Weight")
        Channel("Input")
        Channel("Trig")
        Channel("Output")

        @FuncOp.from_py_func(input_l3_t, weight_l3_t, start_l3_t, output_l3_t)
        def rope_norm(input_l3, weight_l3, start_position_l3, output_l3):
            @launch(operands=[input_l3, weight_l3, start_position_l3, output_l3])
            def launch_body(input_arg, weight_arg, start_arg, output_arg):
                ChannelPut("Start", start_arg)
                ChannelPut("Weight", weight_arg)

                for token_i in range_(sequence_length):
                    ChannelPut(
                        "Input",
                        input_arg,
                        offsets=[token_i, arith.ConstantOp(T.index(), 0)],
                        sizes=[1, head_dim],
                        strides=[head_dim, 1],
                    )
                    ChannelGet(
                        "Output",
                        output_arg,
                        offsets=[token_i, arith.ConstantOp(T.index(), 0)],
                        sizes=[1, head_dim],
                        strides=[head_dim, 1],
                    )
                    yield_([])

                @segment(name="seg")
                def segment_body():
                    @herd(name="rope", sizes=[1, 1])
                    def rope_body(_tx, _ty, _sx, _sy):
                        start_l1 = AllocOp(start_l1_t, [], [])
                        position_l1 = AllocOp(start_l1_t, [], [])
                        cos_l1 = AllocOp(row_l1_t, [], [])
                        sin_l1 = AllocOp(row_l1_t, [], [])
                        trig_l1 = AllocOp(trig_l1_t, [], [])

                        ChannelGet("Start", start_l1)
                        base_position = load(start_l1, [arith.ConstantOp(T.index(), 0)])
                        for token_i in range_(sequence_length):
                            token_i32 = arith.index_cast(i32, token_i)
                            position = arith.addi(base_position, token_i32)
                            store(position, position_l1, [arith.ConstantOp(T.index(), 0)])
                            CallOp(
                                [],
                                "rope_table_tile",
                                [position_l1, cos_l1, sin_l1],
                            )
                            for dim_i in range_(head_dim):
                                cos_value = load(cos_l1, [arith.ConstantOp(T.index(), 0), dim_i])
                                sin_value = load(sin_l1, [arith.ConstantOp(T.index(), 0), dim_i])
                                store(
                                    cos_value,
                                    trig_l1,
                                    [arith.ConstantOp(T.index(), 0), dim_i],
                                )
                                store(
                                    sin_value,
                                    trig_l1,
                                    [arith.ConstantOp(T.index(), 1), dim_i],
                                )
                                yield_([])
                            ChannelPut("Trig", trig_l1)
                            yield_([])

                        DeallocOp(start_l1)
                        DeallocOp(position_l1)
                        DeallocOp(cos_l1)
                        DeallocOp(sin_l1)
                        DeallocOp(trig_l1)

                    @herd(name="norm_rope", sizes=[1, 1])
                    def norm_rope_body(_tx, _ty, _sx, _sy):
                        input_l1 = AllocOp(row_l1_t, [], [])
                        weight_l1 = AllocOp(weight_l1_t, [], [])
                        trig_l1 = AllocOp(trig_l1_t, [], [])
                        cos_l1 = AllocOp(row_l1_t, [], [])
                        sin_l1 = AllocOp(row_l1_t, [], [])
                        output_l1 = AllocOp(row_l1_t, [], [])

                        ChannelGet("Weight", weight_l1)
                        for _token_i in range_(sequence_length):
                            ChannelGet("Trig", trig_l1)
                            ChannelGet("Input", input_l1)
                            for dim_i in range_(head_dim):
                                cos_value = load(
                                    trig_l1,
                                    [arith.ConstantOp(T.index(), 0), dim_i],
                                )
                                sin_value = load(
                                    trig_l1,
                                    [arith.ConstantOp(T.index(), 1), dim_i],
                                )
                                store(
                                    cos_value,
                                    cos_l1,
                                    [arith.ConstantOp(T.index(), 0), dim_i],
                                )
                                store(
                                    sin_value,
                                    sin_l1,
                                    [arith.ConstantOp(T.index(), 0), dim_i],
                                )
                                yield_([])
                            CallOp(
                                [],
                                "rms_norm_rope_tile",
                                [input_l1, weight_l1, cos_l1, sin_l1, output_l1],
                            )
                            ChannelPut("Output", output_l1)
                            yield_([])

                        DeallocOp(input_l1)
                        DeallocOp(weight_l1)
                        DeallocOp(trig_l1)
                        DeallocOp(cos_l1)
                        DeallocOp(sin_l1)
                        DeallocOp(output_l1)

    return build()


def build_reference(
    sequence_length: int,
    head_dim: int,
    start_position: int,
    rope_theta: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm device is required for the Spike 3 reference.")
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)

    values = torch.arange(sequence_length * head_dim, dtype=torch.float32, device=device)
    input_tensor = (values.reshape(sequence_length, head_dim) - 37.0) / 97.0
    weight = 1.0 + torch.arange(head_dim, dtype=torch.float32, device=device) / 512.0
    position = torch.arange(
        start_position,
        start_position + sequence_length,
        dtype=torch.float32,
        device=device,
    )
    freq_index = torch.arange(head_dim // 2, dtype=torch.float32, device=device)
    theta = torch.tensor(rope_theta, dtype=torch.float32, device=device)
    inv_freq = torch.pow(theta, -2.0 * freq_index / head_dim)
    angles = position[:, None] * inv_freq[None, :]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=1)
    sin = torch.cat([sin_half, sin_half], dim=1)

    variance = torch.mean(input_tensor * input_tensor, dim=1, keepdim=True)
    normed = input_tensor * torch.rsqrt(variance + eps) * weight.reshape(1, head_dim)
    half_dim = head_dim // 2
    rotated = torch.cat([-normed[:, half_dim:], normed[:, :half_dim]], dim=1)
    output = normed * cos + rotated * sin

    return (
        input_tensor.cpu().numpy(),
        weight.cpu().numpy(),
        np.array([start_position], dtype=np.int32),
        output.cpu().numpy(),
        device_name,
    )


def prepare_objects(
    work_dir: Path,
    peano: Path,
    head_dim: int,
    rope_theta: float,
    eps: float,
) -> None:
    object_work_dir = work_dir / "objects"
    rope_object = compile_rope_table_object(
        work_dir=object_work_dir,
        peano_install_dir=str(peano),
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    norm_object = compile_rms_norm_rope_object(
        work_dir=object_work_dir,
        peano_install_dir=str(peano),
        head_dim=head_dim,
        eps=eps,
    )
    for object_dir in (work_dir, work_dir / "air_project"):
        object_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rope_object, object_dir / "rope_table.o")
        shutil.copy2(norm_object, object_dir / "rms_norm_rope.o")


def main() -> int:
    parser = argparse.ArgumentParser(description="Spike 3: fused rope table + RMSNorm/RoPE.")
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--rope-theta", type=float, default=500000.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".cache/npu-spikes/mega-spike3-rope-norm"),
    )
    args = parser.parse_args()

    _, _, peano = prepend_air_tool_paths()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.work_dir)
    prepare_objects(Path.cwd(), peano, args.head_dim, args.rope_theta, args.eps)

    mlir_module = build_module(args.sequence_length, args.head_dim)
    Path("source.air.mlir").write_text(str(mlir_module))
    input_data, weight, start_position, expected_output, device_name = build_reference(
        args.sequence_length,
        args.head_dim,
        args.start_position,
        args.rope_theta,
        args.eps,
    )
    runner = XRTRunner(
        output_format="xclbin",
        instance_name="rope_norm",
        target_device="npu2_4col",
    )
    result = runner.run_test(
        mlir_module,
        inputs=[input_data, weight, start_position],
        expected_outputs=[expected_output],
        rtol=5e-3,
        atol=5e-3,
    )
    print(f"reference_device {device_name}")
    print(f"work_dir {Path.cwd()}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
