from __future__ import annotations

import numpy as np
from air.backend.xrt_runner import type_mapper
from air.dialects import arith
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    module_builder,
)
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.scf import for_, yield_
from air._mlir_libs._mlir.ir import IntegerAttr, MemRefType, Module, Type, Value

from torch2air.export.air_dsl import air_herd, air_launch, air_segment, idx
from torch2air.export.builder import KernelAttr, TensorInfo

range_ = for_


class Q4KEmbeddingAirBuilder:
    def __init__(self, *, function_name: str) -> None:
        self.function_name = function_name
        self.tensors: dict[str, TensorInfo] = {}
        self.outputs: list[str] = []
        self.embedding_output: str | None = None

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
        if target != "aten.embedding.default":
            raise ValueError(
                f"Q4K embedding builder only supports aten.embedding.default, got {target}"
            )
        if len(inputs) != 2:
            raise ValueError(f"aten.embedding.default expects weight and indices, got {inputs}")
        if attrs:
            raise ValueError(f"aten.embedding.default does not use attrs, got {attrs}")
        self.embedding_output = output

    def build_air_module(self) -> Module:
        sequence_length, blocks_per_row, row_words, hidden_size = self._embedding_shape()
        return build_q4k_embedding_air(
            function_name=self.function_name,
            sequence_length=sequence_length,
            blocks_per_row=blocks_per_row,
            row_words=row_words,
            hidden_size=hidden_size,
        )

    def render_air(self) -> str:
        return str(self.build_air_module())

    def _embedding_shape(self) -> tuple[int, int, int, int]:
        if self.embedding_output is None:
            raise ValueError("kernel did not emit aten.embedding.default")
        output = self.tensors[self.embedding_output]
        if len(output.shape) != 3 or output.shape[0] != 1:
            raise ValueError(f"expected embedding output shape [1, S, H], got {output.shape}")
        sequence_length = output.shape[1]
        hidden_size = output.shape[2]
        if hidden_size % 256 != 0:
            raise ValueError(f"Q4_K hidden size must be divisible by 256, got {hidden_size}")
        blocks_per_row = hidden_size // 256
        row_words = blocks_per_row * 36
        return sequence_length, blocks_per_row, row_words, hidden_size


def build_q4k_embedding_air(
    *,
    function_name: str,
    sequence_length: int,
    blocks_per_row: int,
    row_words: int,
    hidden_size: int,
) -> Module:
    @module_builder
    def build() -> None:
        i32: Type = type_mapper(np.int32)
        f32: Type = type_mapper(np.float32)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        packed_l3_type = MemRefType.get([sequence_length, row_words], i32)
        scale_l3_type = MemRefType.get([sequence_length, blocks_per_row, 2], f32)
        output_l3_type = MemRefType.get([sequence_length, hidden_size], f32)

        packed_l1_type = MemRefType.get([1, 36], i32, memory_space=l1_space)
        scale_l1_type = MemRefType.get([1, 1, 2], f32, memory_space=l1_space)
        output_l1_type = MemRefType.get([1, 256], f32, memory_space=l1_space)

        @FuncOp.from_py_func(
            packed_l3_type,
            scale_l3_type,
            output_l3_type,
            name=function_name,
        )
        def q4k_embedding(
            packed_rows: Value,
            block_f16_scales: Value,
            output: Value,
        ) -> None:
            @air_launch(operands=(packed_rows, block_f16_scales, output))
            def launch_body(packed_arg: Value, scales_arg: Value, output_arg: Value) -> None:
                @air_segment(name="seg", operands=(packed_arg, scales_arg, output_arg))
                def segment_body(packed_seg: Value, scales_seg: Value, output_seg: Value) -> None:
                    @air_herd(
                        name="q4k_embedding",
                        sizes=[1, blocks_per_row],
                        operands=(packed_seg, scales_seg, output_seg),
                    )
                    def herd_body(
                        _tile_i: Value,
                        tile_j: Value,
                        _size_i: Value,
                        _size_j: Value,
                        packed: Value,
                        scales: Value,
                        out: Value,
                    ) -> None:
                        packed_l1 = AllocOp(packed_l1_type, [], [])
                        scale_l1 = AllocOp(scale_l1_type, [], [])
                        out_l1 = AllocOp(output_l1_type, [], [])

                        packed_col = arith.muli(tile_j, idx(36))
                        out_col = arith.muli(tile_j, idx(256))
                        for token_i in range_(sequence_length):
                            dma_memcpy_nd(
                                packed_l1,
                                packed,
                                src_offsets=[token_i, packed_col],
                                src_sizes=[1, 36],
                                src_strides=[row_words, 1],
                            )
                            dma_memcpy_nd(
                                scale_l1,
                                scales,
                                src_offsets=[token_i, tile_j, idx(0)],
                                src_sizes=[1, 1, 2],
                                src_strides=[blocks_per_row * 2, 2, 1],
                            )
                            _emit_q4k_dequant_tile(
                                i32=i32,
                                f32=f32,
                                packed_l1=packed_l1,
                                scale_l1=scale_l1,
                                out_l1=out_l1,
                            )
                            dma_memcpy_nd(
                                out,
                                out_l1,
                                dst_offsets=[token_i, out_col],
                                dst_sizes=[1, 256],
                                dst_strides=[hidden_size, 1],
                            )
                            yield_([])

                        DeallocOp(packed_l1)
                        DeallocOp(scale_l1)
                        DeallocOp(out_l1)

    return build()


def _emit_q4k_dequant_tile(*, i32, f32, packed_l1, scale_l1, out_l1) -> None:
    d = load(scale_l1, [idx(0), idx(0), idx(0)])
    dmin = load(scale_l1, [idx(0), idx(0), idx(1)])
    scales: list[tuple[Value, Value]] = []
    for subblock in range(8):
        scale, minimum = _emit_scale_min(i32=i32, packed_l1=packed_l1, subblock=subblock)
        scale_f32 = arith.uitofp(f32, scale)
        min_f32 = arith.uitofp(f32, minimum)
        scales.append((arith.mulf(d, scale_f32), arith.mulf(dmin, min_f32)))

    for pair in range(4):
        for q_word_i in range_(8):
            word_index = arith.addi(q_word_i, idx(4 + pair * 8))
            q_word = load(packed_l1, [idx(0), word_index])
            word_base = arith.muli(q_word_i, idx(4))
            for byte in range(4):
                _emit_quant_byte(
                    i32=i32,
                    f32=f32,
                    out_l1=out_l1,
                    scales=scales,
                    q_word=q_word,
                    word_base=word_base,
                    pair=pair,
                    byte=byte,
                )
            yield_([])


def _emit_scale_min(*, i32, packed_l1, subblock: int):
    if subblock < 4:
        scale_byte = _emit_byte_at(i32=i32, packed_l1=packed_l1, byte_offset=4 + subblock)
        min_byte = _emit_byte_at(i32=i32, packed_l1=packed_l1, byte_offset=8 + subblock)
        return arith.andi(scale_byte, _i32(i32, 63)), arith.andi(min_byte, _i32(i32, 63))

    d_byte = _emit_byte_at(i32=i32, packed_l1=packed_l1, byte_offset=subblock)
    m_byte = _emit_byte_at(i32=i32, packed_l1=packed_l1, byte_offset=4 + subblock)
    md_byte = _emit_byte_at(i32=i32, packed_l1=packed_l1, byte_offset=8 + subblock)

    scale_lo = arith.andi(md_byte, _i32(i32, 15))
    scale_hi_0 = arith.shrui(d_byte, _i32(i32, 2))
    scale_hi = arith.andi(scale_hi_0, _i32(i32, 48))
    scale = arith.ori(scale_lo, scale_hi)

    min_lo = arith.shrui(md_byte, _i32(i32, 4))
    min_hi_0 = arith.shrui(m_byte, _i32(i32, 2))
    min_hi = arith.andi(min_hi_0, _i32(i32, 48))
    minimum = arith.ori(min_lo, min_hi)
    return scale, minimum


def _emit_quant_byte(
    *,
    i32,
    f32,
    out_l1,
    scales: list[tuple[Value, Value]],
    q_word,
    word_base,
    pair: int,
    byte: int,
) -> None:
    even_subblock = pair * 2
    odd_subblock = pair * 2 + 1
    byte_value = _mask_i32(i32, arith.shrui(q_word, _i32(i32, byte * 8)), 255)
    lo = arith.andi(byte_value, _i32(i32, 15))
    hi = arith.shrui(byte_value, _i32(i32, 4))
    lo_f32 = arith.uitofp(f32, lo)
    hi_f32 = arith.uitofp(f32, hi)
    even_d, even_min = scales[even_subblock]
    odd_d, odd_min = scales[odd_subblock]
    even_value = arith.subf(arith.mulf(even_d, lo_f32), even_min)
    odd_value = arith.subf(arith.mulf(odd_d, hi_f32), odd_min)

    out_word_byte = arith.addi(word_base, idx(byte))
    out_even = arith.addi(out_word_byte, idx(pair * 64))
    out_odd = arith.addi(out_word_byte, idx(pair * 64 + 32))
    store(even_value, out_l1, [idx(0), out_even])
    store(odd_value, out_l1, [idx(0), out_odd])


def _emit_byte_at(*, i32, packed_l1, byte_offset: int):
    word_offset = byte_offset // 4
    bit_offset = (byte_offset % 4) * 8
    word = load(packed_l1, [idx(0), idx(word_offset)])
    return _mask_i32(i32, arith.shrui(word, _i32(i32, bit_offset)), 255)


def _mask_i32(i32, value, mask: int):
    return arith.andi(value, _i32(i32, mask))


def _i32(i32, value: int):
    return arith.constant(i32, value)
