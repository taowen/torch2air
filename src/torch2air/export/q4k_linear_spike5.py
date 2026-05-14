from __future__ import annotations

import numpy as np
from air.backend.xrt_runner import type_mapper
from air.dialects import arith
from air.dialects.air import MemorySpace, T, dma_memcpy_nd, module_builder
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
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

from torch2air.export.air_dsl import air_herd, air_launch, air_segment, idx
from torch2air.export.q4k_linear_spike2 import (
    Q4K_LINEAR_SPIKE2_LINK_OBJECT,
    Q4K_LINEAR_SPIKE2_TILE_FUNCTION,
)

Q4K_LINEAR_SPIKE5_FUNCTION = "q4k_linear_spike5"
Q4K_LINEAR_SPIKE5_SEQUENCE_LENGTH = 8
Q4K_LINEAR_SPIKE5_HIDDEN_SIZE = 1024
Q4K_LINEAR_SPIKE5_OUTPUT_TILE_ROWS = 16

range_ = for_


def q4k_linear_spike5_herd_rows(sequence_length: int) -> int:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    return min(sequence_length, 4)


def q4k_linear_spike5_herd_cols(sequence_length: int) -> int:
    herd_rows = q4k_linear_spike5_herd_rows(sequence_length)
    herd_cols = max(1, (sequence_length + herd_rows * 2 - 1) // (herd_rows * 2))
    if herd_cols > 4:
        raise ValueError(f"Spike 5 only supports up to 4 token columns, got {herd_cols}")
    return herd_cols


def build_q4k_linear_spike5_air(
    *,
    function_name: str = Q4K_LINEAR_SPIKE5_FUNCTION,
    sequence_length: int = Q4K_LINEAR_SPIKE5_SEQUENCE_LENGTH,
    hidden_size: int = Q4K_LINEAR_SPIKE5_HIDDEN_SIZE,
    output_tile_rows: int = Q4K_LINEAR_SPIKE5_OUTPUT_TILE_ROWS,
) -> Module:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if output_tile_rows <= 0:
        raise ValueError(f"output_tile_rows must be positive, got {output_tile_rows}")
    if hidden_size % 256 != 0:
        raise ValueError(f"hidden_size must be divisible by 256, got {hidden_size}")
    physical_rows = q4k_linear_spike5_herd_rows(sequence_length)
    physical_cols = q4k_linear_spike5_herd_cols(sequence_length)
    token_step = physical_rows * physical_cols
    blocks_per_row = hidden_size // 256
    weight_words = blocks_per_row * 38

    @module_builder
    def build() -> None:
        i32: Type = type_mapper(np.int32)
        f32: Type = type_mapper(np.float32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        hidden_l3_type = MemRefType.get([sequence_length, hidden_size], f32)
        weight_l3_type = MemRefType.get([output_tile_rows, weight_words], i32)
        output_l3_type = MemRefType.get([sequence_length, output_tile_rows], f32)

        hidden_l1_type = MemRefType.get([1, hidden_size], f32, memory_space=l1_space)
        weight_l1_type = MemRefType.get(
            [output_tile_rows, weight_words],
            i32,
            memory_space=l1_space,
        )
        output_l1_type = MemRefType.get([1, output_tile_rows], f32, memory_space=l1_space)

        tile_func = FuncOp(
            Q4K_LINEAR_SPIKE2_TILE_FUNCTION,
            FunctionType.get([hidden_l1_type, weight_l1_type, output_l1_type], []),
            visibility="private",
        )
        tile_func.attributes["link_with"] = StringAttr.get(Q4K_LINEAR_SPIKE2_LINK_OBJECT)
        tile_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            hidden_l3_type,
            weight_l3_type,
            output_l3_type,
            name=function_name,
        )
        def q4k_linear_spike5(
            hidden_l3: Value,
            weight_l3: Value,
            output_l3: Value,
        ) -> None:
            @air_launch(operands=(hidden_l3, weight_l3, output_l3))
            def launch_body(
                hidden_arg: Value,
                weight_arg: Value,
                output_arg: Value,
            ) -> None:
                @air_segment(name="seg", operands=(hidden_arg, weight_arg, output_arg))
                def segment_body(
                    hidden_seg: Value,
                    weight_seg: Value,
                    output_seg: Value,
                ) -> None:
                    @air_herd(
                        name="q4k_linear_spike5",
                        sizes=[physical_rows, physical_cols],
                        operands=(hidden_seg, weight_seg, output_seg),
                    )
                    def herd_body(
                        tile_i: Value,
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

                        dma_memcpy_nd(
                            weight_l1,
                            weight,
                            src_offsets=[idx(0), idx(0)],
                            src_sizes=[output_tile_rows, weight_words],
                            src_strides=[weight_words, 1],
                        )
                        token_col_base = arith.muli(tile_j, idx(physical_rows))
                        token_start = arith.addi(token_col_base, tile_i)
                        for token_i in range_(token_start, sequence_length, token_step):
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
