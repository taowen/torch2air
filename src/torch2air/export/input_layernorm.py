from __future__ import annotations

import numpy as np
from air.backend.xrt_runner import type_mapper
from air.dialects import arith
from air.dialects.air import MemorySpace, T, dma_memcpy_nd, module_builder
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air._mlir_libs._mlir.ir import IntegerAttr, MemRefType, Module, Type, Value

from torch2air.export.air_dsl import air_herd, air_launch, air_segment, idx
from torch2air.export.builder import KernelAttr, TensorInfo

range_ = for_

ALIAS_TARGETS = frozenset(
    {
        "aten._assert_tensor_metadata.default",
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.detach.default",
        "aten.to.dtype",
        "aten.to.dtype_layout",
    }
)

RMS_NORM_TARGETS = frozenset(
    {
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
    }
)


class InputLayerNormAirBuilder:
    def __init__(self, *, function_name: str, eps: float) -> None:
        self.function_name = function_name
        self.eps = eps
        self.tensors: dict[str, TensorInfo] = {}
        self.outputs: list[str] = []
        self.targets: list[str] = []

    def define_tensor(self, name: str, *, shape: tuple[int, ...], dtype: str) -> None:
        self.tensors[name] = TensorInfo(name=name, shape=shape, dtype=dtype)

    def mark_output(self, name: str) -> None:
        self.outputs.append(name)

    def emit_kernel(
        self,
        target: str,
        *,
        output: str,
        inputs: tuple[str, ...],
        attrs: tuple[KernelAttr, ...] = (),
    ) -> None:
        if target not in ALIAS_TARGETS and target not in RMS_NORM_TARGETS:
            raise ValueError(f"input_layernorm builder does not support {target}")
        self.targets.append(target)

    def build_air_module(self) -> Module:
        sequence_length, hidden_size = self._output_shape()
        return build_input_layernorm_air(
            function_name=self.function_name,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            eps=self.eps,
        )

    def render_air(self) -> str:
        return str(self.build_air_module())

    def _output_shape(self) -> tuple[int, int]:
        if not self.outputs:
            raise ValueError("kernel did not mark an output")
        output = self.tensors[self.outputs[-1]]
        if len(output.shape) != 3 or output.shape[0] != 1:
            raise ValueError(f"expected input_layernorm output shape [1, S, H], got {output.shape}")
        if "aten.rsqrt.default" not in self.targets:
            raise ValueError("input_layernorm graph did not emit aten.rsqrt.default")
        return output.shape[1], output.shape[2]


def build_input_layernorm_air(
    *,
    function_name: str,
    sequence_length: int,
    hidden_size: int,
    eps: float,
) -> Module:
    physical_rows = min(sequence_length, 4)

    @module_builder
    def build() -> None:
        i32: Type = type_mapper(np.int32)
        f32: Type = type_mapper(np.float32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        hidden_l3_type = MemRefType.get([sequence_length, hidden_size], f32)
        weight_l3_type = MemRefType.get([hidden_size], f32)
        output_l3_type = MemRefType.get([sequence_length, hidden_size], f32)

        row_l1_type = MemRefType.get([1, hidden_size], f32, memory_space=l1_space)
        weight_l1_type = MemRefType.get([hidden_size], f32, memory_space=l1_space)
        scalar_l1_type = MemRefType.get([1], f32, memory_space=l1_space)

        @FuncOp.from_py_func(
            hidden_l3_type,
            weight_l3_type,
            output_l3_type,
            name=function_name,
        )
        def input_layernorm(hidden_l3: Value, weight_l3: Value, output_l3: Value) -> None:
            @air_launch(operands=(hidden_l3, weight_l3, output_l3))
            def launch_body(hidden_arg: Value, weight_arg: Value, output_arg: Value) -> None:
                @air_segment(name="seg", operands=(hidden_arg, weight_arg, output_arg))
                def segment_body(
                    hidden_seg: Value,
                    weight_seg: Value,
                    output_seg: Value,
                ) -> None:
                    @air_herd(
                        name="input_layernorm",
                        sizes=[physical_rows, 1],
                        operands=(hidden_seg, weight_seg, output_seg),
                    )
                    def herd_body(
                        tile_i: Value,
                        _tile_j: Value,
                        _size_i: Value,
                        _size_j: Value,
                        hidden: Value,
                        weight: Value,
                        output: Value,
                    ) -> None:
                        hidden_l1 = AllocOp(row_l1_type, [], [])
                        weight_l1 = AllocOp(weight_l1_type, [], [])
                        output_l1 = AllocOp(row_l1_type, [], [])
                        sum_l1 = AllocOp(scalar_l1_type, [], [])

                        dma_memcpy_nd(
                            weight_l1,
                            weight,
                            src_offsets=[idx(0)],
                            src_sizes=[hidden_size],
                            src_strides=[1],
                        )
                        for token_i in range_(tile_i, sequence_length, physical_rows):
                            dma_memcpy_nd(
                                hidden_l1,
                                hidden,
                                src_offsets=[token_i, idx(0)],
                                src_sizes=[1, hidden_size],
                                src_strides=[hidden_size, 1],
                            )
                            _emit_rms_norm_tile(
                                i32=i32,
                                f32=f32,
                                hidden_size=hidden_size,
                                eps=eps,
                                hidden_l1=hidden_l1,
                                weight_l1=weight_l1,
                                output_l1=output_l1,
                                sum_l1=sum_l1,
                            )
                            dma_memcpy_nd(
                                output,
                                output_l1,
                                dst_offsets=[token_i, idx(0)],
                                dst_sizes=[1, hidden_size],
                                dst_strides=[hidden_size, 1],
                            )
                            yield_([])

                        DeallocOp(hidden_l1)
                        DeallocOp(weight_l1)
                        DeallocOp(output_l1)
                        DeallocOp(sum_l1)

    return build()


def _emit_rms_norm_tile(
    *,
    i32: Type,
    f32: Type,
    hidden_size: int,
    eps: float,
    hidden_l1: Value,
    weight_l1: Value,
    output_l1: Value,
    sum_l1: Value,
) -> None:
    sum_squares = _emit_sum_squares(
        f32=f32,
        hidden_size=hidden_size,
        hidden_l1=hidden_l1,
        sum_l1=sum_l1,
    )
    variance = arith.mulf(sum_squares, arith.constant(f32, 1.0 / hidden_size))
    variance_eps = arith.addf(variance, arith.constant(f32, eps))
    inv_rms = _emit_fast_rsqrt(i32=i32, f32=f32, value=variance_eps)

    for dim_i in range_(hidden_size):
        hidden_value = load(hidden_l1, [idx(0), dim_i])
        weight_value = load(weight_l1, [dim_i])
        normed = arith.mulf(hidden_value, inv_rms)
        output_value = arith.mulf(normed, weight_value)
        store(output_value, output_l1, [idx(0), dim_i])
        yield_([])


def _emit_sum_squares(
    *,
    f32: Type,
    hidden_size: int,
    hidden_l1: Value,
    sum_l1: Value,
) -> Value:
    store(arith.constant(f32, 0.0), sum_l1, [idx(0)])
    for dim_i in range_(hidden_size):
        acc = load(sum_l1, [idx(0)])
        hidden_value = load(hidden_l1, [idx(0), dim_i])
        squared = arith.mulf(hidden_value, hidden_value)
        next_acc = arith.addf(acc, squared)
        store(next_acc, sum_l1, [idx(0)])
        yield_([])
    return load(sum_l1, [idx(0)])


def _emit_fast_rsqrt(*, i32: Type, f32: Type, value: Value) -> Value:
    half_value = arith.mulf(value, arith.constant(f32, 0.5))
    value_bits = arith.bitcast(i32, value)
    shifted_bits = arith.shrui(value_bits, arith.constant(i32, 1))
    y_bits = arith.subi(arith.constant(i32, 0x5F3759DF), shifted_bits)
    y = arith.bitcast(f32, y_bits)
    for _ in range(3):
        y_squared = arith.mulf(y, y)
        correction = arith.subf(
            arith.constant(f32, 1.5),
            arith.mulf(half_value, y_squared),
        )
        y = arith.mulf(y, correction)
    return y
