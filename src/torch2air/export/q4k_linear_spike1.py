from __future__ import annotations

import numpy as np
from air.backend.xrt_runner import type_mapper
from air.dialects.air import MemorySpace, T, dma_memcpy_nd, module_builder
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
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

Q4K_LINEAR_SPIKE1_FUNCTION = "q4k_linear_spike1"
Q4K_LINEAR_SPIKE1_TILE_FUNCTION = "q4k_linear_spike1_tile"
Q4K_LINEAR_SPIKE1_LINK_OBJECT = "q4k_linear_spike1.o"
Q4K_LINEAR_SPIKE1_HIDDEN_SIZE = 1024
Q4K_LINEAR_SPIKE1_OUTPUT_TILE_ROWS = 16


def build_q4k_linear_spike1_air(
    *,
    function_name: str = Q4K_LINEAR_SPIKE1_FUNCTION,
    hidden_size: int = Q4K_LINEAR_SPIKE1_HIDDEN_SIZE,
    output_tile_rows: int = Q4K_LINEAR_SPIKE1_OUTPUT_TILE_ROWS,
) -> Module:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if output_tile_rows <= 0:
        raise ValueError(f"output_tile_rows must be positive, got {output_tile_rows}")
    if hidden_size % output_tile_rows != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by output_tile_rows={output_tile_rows}"
        )

    @module_builder
    def build() -> None:
        i32: Type = type_mapper(np.int32)
        f32: Type = type_mapper(np.float32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        hidden_l3_type = MemRefType.get([1, hidden_size], f32)
        weight_l3_type = MemRefType.get([output_tile_rows], i32)
        output_l3_type = MemRefType.get([1, output_tile_rows], f32)

        hidden_l1_type = MemRefType.get([1, hidden_size], f32, memory_space=l1_space)
        weight_l1_type = MemRefType.get([output_tile_rows], i32, memory_space=l1_space)
        output_l1_type = MemRefType.get([1, output_tile_rows], f32, memory_space=l1_space)

        tile_func = FuncOp(
            Q4K_LINEAR_SPIKE1_TILE_FUNCTION,
            FunctionType.get([hidden_l1_type, weight_l1_type, output_l1_type], []),
            visibility="private",
        )
        tile_func.attributes["link_with"] = StringAttr.get(Q4K_LINEAR_SPIKE1_LINK_OBJECT)
        tile_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            hidden_l3_type,
            weight_l3_type,
            output_l3_type,
            name=function_name,
        )
        def q4k_linear_spike1(
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
                        name="q4k_linear_spike1",
                        sizes=[1, 1],
                        operands=(hidden_seg, weight_seg, output_seg),
                    )
                    def herd_body(
                        _tile_i: Value,
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
                            hidden_l1,
                            hidden,
                            src_offsets=[idx(0), idx(0)],
                            src_sizes=[1, hidden_size],
                            src_strides=[hidden_size, 1],
                        )
                        dma_memcpy_nd(
                            weight_l1,
                            weight,
                            src_offsets=[idx(0)],
                            src_sizes=[output_tile_rows],
                            src_strides=[1],
                        )
                        CallOp(tile_func, [hidden_l1, weight_l1, output_l1])
                        dma_memcpy_nd(
                            output,
                            output_l1,
                            dst_offsets=[idx(0), idx(0)],
                            dst_sizes=[1, output_tile_rows],
                            dst_strides=[output_tile_rows, 1],
                        )

                        DeallocOp(hidden_l1)
                        DeallocOp(weight_l1)
                        DeallocOp(output_l1)

    return build()
