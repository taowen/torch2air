from __future__ import annotations

import numpy as np
from air._mlir_libs._mlir.ir import (
    FunctionType,
    IntegerAttr,
    MemRefType,
    Module,
    StringAttr,
    Type,
    UnitAttr,
    Value,
)
from air.backend.xrt_runner import type_mapper
from air.dialects import arith
from air.dialects.air import MemorySpace, T, dma_memcpy_nd, module_builder
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_

from torch2air.export.air_dsl import air_herd, air_launch, air_segment, idx
from torch2air.export.builder import KernelAttr, TensorInfo

Q6K_LINEAR_TILE_FUNCTION = "q6k_linear_tile"
Q6K_LINEAR_LINK_OBJECT = "q6k_linear.o"
Q6K_LINEAR_PREFILL_BUCKET = 8

range_ = for_


class Q6KLinearAirBuilder:
    def __init__(
        self,
        *,
        function_name: str,
        output_features: int,
        output_tile_rows: int = 16,
    ) -> None:
        self.function_name = function_name
        self.output_features = output_features
        self.output_tile_rows = output_tile_rows
        self.tensors: dict[str, TensorInfo] = {}
        self.outputs: list[str] = []
        self.linear_input: str | None = None
        self.linear_weight: str | None = None
        self.linear_output: str | None = None

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
        if target != "aten.linear.default":
            raise ValueError(f"Q6_K linear builder only supports aten.linear.default, got {target}")
        if len(inputs) != 2:
            raise ValueError("Q6_K linear builder only supports bias-free aten.linear.default")
        if attrs:
            raise ValueError(f"aten.linear.default does not use attrs, got {attrs}")
        self.linear_input = inputs[0]
        self.linear_weight = inputs[1]
        self.linear_output = output

    def build_air_module(self) -> Module:
        sequence_length, hidden_size, output_features = self.linear_shape()
        return build_q6k_linear_air(
            function_name=self.function_name,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            output_features=output_features,
            output_tile_rows=self.output_tile_rows,
        )

    def render_air(self) -> str:
        return str(self.build_air_module())

    def herd_rows(self) -> int:
        sequence_length, _, _ = self.linear_shape()
        if sequence_length == 1:
            return 1
        if sequence_length == Q6K_LINEAR_PREFILL_BUCKET:
            return 4
        raise ValueError(
            f"formal Q6_K linear supports S=1 or S={Q6K_LINEAR_PREFILL_BUCKET}, "
            f"got S={sequence_length}"
        )

    def herd_cols(self) -> int:
        sequence_length, _, output_features = self.linear_shape()
        if sequence_length == 1:
            return output_features // self.output_tile_rows
        return 1

    def linear_shape(self) -> tuple[int, int, int]:
        if self.linear_input is None or self.linear_weight is None or self.linear_output is None:
            raise ValueError("kernel did not emit aten.linear.default")
        input_info = self.tensors[self.linear_input]
        weight_info = self.tensors[self.linear_weight]
        output_info = self.tensors[self.linear_output]
        if len(input_info.shape) != 3 or input_info.shape[0] != 1:
            raise ValueError(f"expected linear input shape [1, S, K], got {input_info.shape}")
        if len(weight_info.shape) != 2:
            raise ValueError(f"expected linear weight shape [N, K], got {weight_info.shape}")
        if len(output_info.shape) != 3 or output_info.shape[0] != 1:
            raise ValueError(f"expected linear output shape [1, S, N], got {output_info.shape}")
        sequence_length = input_info.shape[1]
        hidden_size = input_info.shape[2]
        full_output_features = weight_info.shape[0]
        if sequence_length not in {1, Q6K_LINEAR_PREFILL_BUCKET}:
            raise ValueError(
                f"formal Q6_K linear supports sequence_length=1 or "
                f"{Q6K_LINEAR_PREFILL_BUCKET}, got {sequence_length}"
            )
        if weight_info.shape[1] != hidden_size:
            raise ValueError(
                f"linear input/weight K mismatch: {input_info.shape}, {weight_info.shape}"
            )
        if output_info.shape[1] != sequence_length or output_info.shape[2] != full_output_features:
            raise ValueError(f"linear output shape mismatch: {output_info.shape}")
        if hidden_size % 256 != 0:
            raise ValueError(f"Q6_K input feature count must be divisible by 256, got {hidden_size}")
        if self.output_features <= 0:
            raise ValueError("output_features must be positive")
        if self.output_features > full_output_features:
            raise ValueError(
                f"compiled output_features={self.output_features} exceeds "
                f"linear output size {full_output_features}"
            )
        if self.output_features % self.output_tile_rows != 0:
            raise ValueError(
                f"output_features={self.output_features} must be divisible by "
                f"output_tile_rows={self.output_tile_rows}"
            )
        herd_cols = self.output_features // self.output_tile_rows
        if sequence_length == 1 and herd_cols > 4:
            raise ValueError(f"Q6_K linear herd columns must fit in 4 NPU columns, got {herd_cols}")
        if (
            sequence_length == Q6K_LINEAR_PREFILL_BUCKET
            and self.output_features != self.output_tile_rows
        ):
            raise ValueError(
                f"prefill Q6_K linear uses one output tile per xclbin; "
                f"output_features={self.output_features} must equal "
                f"output_tile_rows={self.output_tile_rows}"
            )
        return sequence_length, hidden_size, self.output_features


def build_q6k_linear_air(
    *,
    function_name: str,
    sequence_length: int,
    hidden_size: int,
    output_features: int,
    output_tile_rows: int,
) -> Module:
    if sequence_length not in {1, Q6K_LINEAR_PREFILL_BUCKET}:
        raise ValueError(
            f"formal Q6_K linear supports sequence_length=1 or "
            f"{Q6K_LINEAR_PREFILL_BUCKET}, got {sequence_length}"
        )
    if hidden_size % 256 != 0:
        raise ValueError(f"hidden_size must be divisible by 256, got {hidden_size}")
    blocks_per_row = hidden_size // 256
    weight_words = blocks_per_row * 106
    decode_herd_cols = output_features // output_tile_rows
    prefill_herd_rows = 4

    @module_builder
    def build() -> None:
        i32: Type = type_mapper(np.int32)
        f32: Type = type_mapper(np.float32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        hidden_l3_type = MemRefType.get([sequence_length, hidden_size], f32)
        weight_l3_type = MemRefType.get([output_features, weight_words], i32)
        output_l3_type = MemRefType.get([sequence_length, output_features], f32)

        hidden_l1_type = MemRefType.get([1, hidden_size], f32, memory_space=l1_space)
        weight_l1_type = MemRefType.get(
            [output_tile_rows, weight_words],
            i32,
            memory_space=l1_space,
        )
        output_l1_type = MemRefType.get([1, output_tile_rows], f32, memory_space=l1_space)

        tile_func = FuncOp(
            Q6K_LINEAR_TILE_FUNCTION,
            FunctionType.get([hidden_l1_type, weight_l1_type, output_l1_type], []),
            visibility="private",
        )
        tile_func.attributes["link_with"] = StringAttr.get(Q6K_LINEAR_LINK_OBJECT)
        tile_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            hidden_l3_type,
            weight_l3_type,
            output_l3_type,
            name=function_name,
        )
        def q6k_linear(hidden_l3: Value, weight_l3: Value, output_l3: Value) -> None:
            @air_launch(operands=(hidden_l3, weight_l3, output_l3))
            def launch_body(hidden_arg: Value, weight_arg: Value, output_arg: Value) -> None:
                @air_segment(name="seg", operands=(hidden_arg, weight_arg, output_arg))
                def segment_body(
                    hidden_seg: Value,
                    weight_seg: Value,
                    output_seg: Value,
                ) -> None:
                    if sequence_length == 1:
                        @air_herd(
                            name="q6k_linear",
                            sizes=[1, decode_herd_cols],
                            operands=(hidden_seg, weight_seg, output_seg),
                        )
                        def herd_body(
                            _tile_i: Value,
                            tile_j: Value,
                            _size_i: Value,
                            _size_j: Value,
                            hidden: Value,
                            weight: Value,
                            output: Value,
                        ) -> None:
                            hidden_l1 = AllocOp(hidden_l1_type, [], [])
                            weight_l1 = AllocOp(weight_l1_type, [], [])
                            output_l1 = AllocOp(output_l1_type, [], [])
                            row_base = arith.muli(tile_j, idx(output_tile_rows))

                            dma_memcpy_nd(
                                hidden_l1,
                                hidden,
                                src_offsets=[idx(0), idx(0)],
                                src_sizes=[1, hidden_size],
                                src_strides=[hidden_size, 1],
                            )
                            dma_memcpy_nd(
                                weight_l1,
                                weight,
                                src_offsets=[row_base, idx(0)],
                                src_sizes=[output_tile_rows, weight_words],
                                src_strides=[weight_words, 1],
                            )
                            CallOp(tile_func, [hidden_l1, weight_l1, output_l1])
                            dma_memcpy_nd(
                                output,
                                output_l1,
                                dst_offsets=[idx(0), row_base],
                                dst_sizes=[1, output_tile_rows],
                                dst_strides=[output_features, 1],
                            )

                            DeallocOp(hidden_l1)
                            DeallocOp(weight_l1)
                            DeallocOp(output_l1)
                    else:
                        @air_herd(
                            name="q6k_linear",
                            sizes=[prefill_herd_rows, 1],
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
                            hidden_l1 = AllocOp(hidden_l1_type, [], [])
                            weight_l1 = AllocOp(weight_l1_type, [], [])
                            output_l1 = AllocOp(output_l1_type, [], [])

                            dma_memcpy_nd(
                                weight_l1,
                                weight,
                                src_offsets=[idx(0), idx(0)],
                                src_sizes=[output_tile_rows, weight_words],
                                src_strides=[weight_words, 1],
                            )
                            for token_i in range_(tile_i, sequence_length, prefill_herd_rows):
                                dma_memcpy_nd(
                                    hidden_l1,
                                    hidden,
                                    src_offsets=[token_i, idx(0)],
                                    src_sizes=[1, hidden_size],
                                    src_strides=[hidden_size, 1],
                                )
                                CallOp(tile_func, [hidden_l1, weight_l1, output_l1])
                                dma_memcpy_nd(
                                    output,
                                    output_l1,
                                    dst_offsets=[token_i, idx(0)],
                                    dst_sizes=[1, output_tile_rows],
                                    dst_strides=[output_tile_rows, 1],
                                )
                                yield_([])

                            DeallocOp(hidden_l1)
                            DeallocOp(weight_l1)
                            DeallocOp(output_l1)

    return build()
