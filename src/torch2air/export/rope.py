from __future__ import annotations

from dataclasses import dataclass
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
from air.dialects.memref import collapse_shape as memref_collapse_shape
from air.dialects.scf import for_, yield_
from ml_dtypes import bfloat16

from torch2air.export.air_dsl import air_herd, air_launch, air_segment, idx
from torch2air.export.builder import KernelAttr, TensorInfo

range_ = for_

BF16_RMS_NORM_128_TILE_FUNCTION = "bf16_rms_norm_128_tile"
BF16_ROPE_128_TILE_FUNCTION = "bf16_rope_128_tile"
BF16_RMS_NORM_HEADS_TILE_FUNCTION = "bf16_rms_norm_heads_tile"
BF16_ROPE_HEADS_TILE_FUNCTION = "bf16_rope_heads_tile"
BF16_ELEMENTWISE_LINK_OBJECT = "bf16_elementwise.o"
VECTOR_SIZE = 16

ALIAS_TARGETS = frozenset(
    {
        "aten._assert_tensor_metadata.default",
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.detach.default",
        "aten.reshape.default",
        "aten.to.dtype",
        "aten.to.dtype_layout",
        "aten.unsqueeze.default",
    }
)

ROPE_TARGETS = frozenset(
    {
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
        "aten.slice.Tensor",
        "aten.neg.default",
        "aten.cat.default",
    }
)


@dataclass(frozen=True, slots=True)
class ExportKernelOp:
    target: str
    output: str
    inputs: tuple[str, ...]
    attrs: tuple[KernelAttr, ...]


class RopeExportAirBuilder:
    def __init__(self, *, function_name: str) -> None:
        self.function_name = function_name
        self.tensors: dict[str, TensorInfo] = {}
        self.outputs: list[str] = []
        self.ops: list[ExportKernelOp] = []

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
        if target not in ALIAS_TARGETS and target not in ROPE_TARGETS:
            raise ValueError(f"RoPE export builder does not support {target}")
        self.ops.append(ExportKernelOp(target=target, output=output, inputs=inputs, attrs=attrs))

    def build_rms_norm_air_module(self, *, function_name: str) -> Module:
        sequence_length, head_count, head_dim = self.rope_shape()
        return build_rms_norm_export_air(
            function_name=function_name,
            sequence_length=sequence_length,
            head_count=head_count,
            head_dim=head_dim,
        )

    def build_rope_air_module(self, *, function_name: str) -> Module:
        sequence_length, head_count, head_dim = self.rope_shape()
        return build_rope_apply_export_air(
            function_name=function_name,
            sequence_length=sequence_length,
            head_count=head_count,
            head_dim=head_dim,
        )

    def build_air_module(self) -> Module:
        return self.build_rope_air_module(function_name=self.function_name)

    def render_rms_norm_air(self, *, function_name: str) -> str:
        return str(self.build_rms_norm_air_module(function_name=function_name))

    def render_rope_air(self, *, function_name: str) -> str:
        return str(self.build_rope_air_module(function_name=function_name))

    def render_air(self) -> str:
        return str(self.build_air_module())

    def herd_rows(self) -> int:
        sequence_length, head_count, _ = self.rope_shape()
        return rope_herd_shape(sequence_length=sequence_length, head_count=head_count)[0]

    def herd_cols(self) -> int:
        sequence_length, head_count, _ = self.rope_shape()
        return rope_herd_shape(sequence_length=sequence_length, head_count=head_count)[1]

    def rope_shape(self) -> tuple[int, int, int]:
        self._validate_op_sequence()
        source = self.tensors.get("source")
        weight = self.tensors.get("p_norm_weight")
        cos = self.tensors.get("cos")
        sin = self.tensors.get("sin")
        if source is None or weight is None or cos is None or sin is None:
            raise ValueError("RoPE export must define source, p_norm_weight, cos, and sin tensors")
        for tensor in (source, weight, cos, sin):
            if tensor.dtype != "bfloat16":
                raise ValueError(f"{tensor.name} must use bfloat16 for NPU RoPE, got {tensor.dtype}")
        if len(source.shape) != 3 or source.shape[0] != 1:
            raise ValueError(f"expected source shape [1, S, H], got {source.shape}")
        if len(weight.shape) != 1:
            raise ValueError(f"expected norm weight shape [D], got {weight.shape}")
        if len(cos.shape) != 3 or cos.shape[0] != 1:
            raise ValueError(f"expected cos shape [1, S, D], got {cos.shape}")
        if sin.shape != cos.shape:
            raise ValueError(f"sin shape {sin.shape} does not match cos shape {cos.shape}")
        if not self.outputs:
            raise ValueError("RoPE export did not mark an output")
        output = self.tensors[self.outputs[-1]]
        if output.dtype != "bfloat16":
            raise ValueError(f"RoPE output must use bfloat16 for NPU RoPE, got {output.dtype}")
        if output.shape != source.shape:
            raise ValueError(f"RoPE output shape {output.shape} does not match {source.shape}")
        sequence_length = source.shape[1]
        head_dim = weight.shape[0]
        width = source.shape[2]
        if cos.shape[1] != sequence_length or cos.shape[2] != head_dim:
            raise ValueError(
                f"cos shape {cos.shape} must match sequence_length={sequence_length} "
                f"and head_dim={head_dim}"
            )
        if head_dim <= 0 or width % head_dim != 0:
            raise ValueError(f"source width {width} must be a multiple of head_dim {head_dim}")
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        return sequence_length, width // head_dim, head_dim

    def _validate_op_sequence(self) -> None:
        targets = [op.target for op in self.ops if op.target not in ALIAS_TARGETS]
        expected = [
            "aten.pow.Tensor_Scalar",
            "aten.mean.dim",
            "aten.add.Tensor",
            "aten.rsqrt.default",
            "aten.mul.Tensor",
            "aten.mul.Tensor",
            "aten.slice.Tensor",
            "aten.neg.default",
            "aten.slice.Tensor",
            "aten.cat.default",
            "aten.mul.Tensor",
            "aten.mul.Tensor",
            "aten.add.Tensor",
        ]
        if targets != expected:
            raise ValueError(f"unexpected RoPE aten sequence: {targets}")


def rope_herd_shape(*, sequence_length: int, head_count: int) -> tuple[int, int]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if head_count <= 0:
        raise ValueError("head_count must be positive")
    if sequence_length == 1:
        return 1, min(head_count, 4)
    herd_rows = min(sequence_length, 4)
    if sequence_length % herd_rows != 0:
        raise ValueError(
            f"prefill sequence_length={sequence_length} must be divisible by herd_rows={herd_rows}"
        )
    return herd_rows, 1


def build_rms_norm_export_air(
    *,
    function_name: str,
    sequence_length: int,
    head_count: int,
    head_dim: int,
) -> Module:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if head_count <= 0:
        raise ValueError("head_count must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    if head_dim % VECTOR_SIZE != 0:
        raise ValueError(f"head_dim must be divisible by {VECTOR_SIZE}, got {head_dim}")
    hidden_size = head_count * head_dim
    herd_rows, herd_cols = rope_herd_shape(
        sequence_length=sequence_length,
        head_count=head_count,
    )
    tokens_per_tile = sequence_length // herd_rows

    @module_builder
    def build() -> None:
        bf16: Type = type_mapper(bfloat16)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        source_l3_type = MemRefType.get([sequence_length, hidden_size], bf16)
        source_flat_l3_type = MemRefType.get([sequence_length * hidden_size], bf16)
        weight_l3_type = MemRefType.get([head_dim], bf16)
        norm_l3_type = MemRefType.get([sequence_length, hidden_size], bf16)
        norm_flat_l3_type = MemRefType.get([sequence_length * hidden_size], bf16)

        feature_tile_size = head_dim if sequence_length == 1 else hidden_size
        source_l1_type = MemRefType.get([feature_tile_size], bf16, memory_space=l1_space)
        weight_l1_type = MemRefType.get([head_dim], bf16, memory_space=l1_space)
        norm_l1_type = MemRefType.get([feature_tile_size], bf16, memory_space=l1_space)
        rms_norm_function = (
            BF16_RMS_NORM_128_TILE_FUNCTION
            if sequence_length == 1
            else BF16_RMS_NORM_HEADS_TILE_FUNCTION
        )

        rms_norm_tile_func = FuncOp(
            rms_norm_function,
            FunctionType.get([source_l1_type, weight_l1_type, norm_l1_type], []),
            visibility="private",
        )
        rms_norm_tile_func.attributes["link_with"] = StringAttr.get(BF16_ELEMENTWISE_LINK_OBJECT)
        rms_norm_tile_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            source_l3_type,
            weight_l3_type,
            norm_l3_type,
            name=function_name,
        )
        def rms_norm_export(
            source_l3: Value,
            weight_l3: Value,
            norm_l3: Value,
        ) -> None:
            @air_launch(operands=(source_l3, weight_l3, norm_l3))
            def rms_launch_body(
                source_arg: Value,
                weight_arg: Value,
                norm_arg: Value,
            ) -> None:
                source_flat = memref_collapse_shape(source_flat_l3_type, source_arg, [[0, 1]])
                norm_flat = memref_collapse_shape(norm_flat_l3_type, norm_arg, [[0, 1]])

                @air_segment(
                    name="rms_seg",
                    operands=(source_flat, weight_arg, norm_flat),
                )
                def rms_segment_body(
                    source_seg: Value,
                    weight_seg: Value,
                    norm_seg: Value,
                ) -> None:
                    @air_herd(
                        name="rms_export",
                        sizes=[herd_rows, herd_cols],
                        operands=(source_seg, weight_seg, norm_seg),
                    )
                    def rms_herd_body(
                        tile_i: Value,
                        tile_j: Value,
                        _size_i: Value,
                        _size_j: Value,
                        source: Value,
                        weight: Value,
                        norm: Value,
                    ) -> None:
                        source_l1 = AllocOp(source_l1_type, [], [])
                        weight_l1 = AllocOp(weight_l1_type, [], [])
                        norm_l1 = AllocOp(norm_l1_type, [], [])

                        dma_memcpy_nd(
                            weight_l1,
                            weight,
                            src_offsets=[idx(0)],
                            src_sizes=[head_dim],
                            src_strides=[1],
                        )
                        token_block_base = arith.muli(tile_i, idx(tokens_per_tile))
                        for local_token_i in range_(tokens_per_tile):
                            token_i = arith.addi(token_block_base, local_token_i)
                            if sequence_length == 1:
                                for head_i in range_(tile_j, head_count, herd_cols):
                                    row_base = arith.muli(token_i, idx(hidden_size))
                                    head_base = arith.muli(head_i, idx(head_dim))
                                    flat_base = arith.addi(row_base, head_base)
                                    dma_memcpy_nd(
                                        source_l1,
                                        source,
                                        src_offsets=[flat_base],
                                        src_sizes=[head_dim],
                                        src_strides=[1],
                                    )
                                    CallOp(rms_norm_tile_func, [source_l1, weight_l1, norm_l1])
                                    dma_memcpy_nd(
                                        norm,
                                        norm_l1,
                                        dst_offsets=[flat_base],
                                        dst_sizes=[head_dim],
                                        dst_strides=[1],
                                    )
                                    yield_([])
                            else:
                                row_base = arith.muli(token_i, idx(hidden_size))
                                dma_memcpy_nd(
                                    source_l1,
                                    source,
                                    src_offsets=[row_base],
                                    src_sizes=[hidden_size],
                                    src_strides=[1],
                                )
                                CallOp(rms_norm_tile_func, [source_l1, weight_l1, norm_l1])
                                dma_memcpy_nd(
                                    norm,
                                    norm_l1,
                                    dst_offsets=[row_base],
                                    dst_sizes=[hidden_size],
                                    dst_strides=[1],
                                )
                            yield_([])

                        DeallocOp(source_l1)
                        DeallocOp(weight_l1)
                        DeallocOp(norm_l1)

    return build()


def build_rope_apply_export_air(
    *,
    function_name: str,
    sequence_length: int,
    head_count: int,
    head_dim: int,
) -> Module:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if head_count <= 0:
        raise ValueError("head_count must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    if head_dim % VECTOR_SIZE != 0:
        raise ValueError(f"head_dim must be divisible by {VECTOR_SIZE}, got {head_dim}")
    hidden_size = head_count * head_dim
    herd_rows, herd_cols = rope_herd_shape(
        sequence_length=sequence_length,
        head_count=head_count,
    )
    tokens_per_tile = sequence_length // herd_rows

    @module_builder
    def build() -> None:
        bf16: Type = type_mapper(bfloat16)
        l1_space = IntegerAttr.get(T.i32(), MemorySpace.L1)

        norm_l3_type = MemRefType.get([sequence_length, hidden_size], bf16)
        norm_flat_l3_type = MemRefType.get([sequence_length * hidden_size], bf16)
        rope_lut_l3_type = MemRefType.get([sequence_length, head_dim * 2], bf16)
        rope_lut_flat_l3_type = MemRefType.get([sequence_length * head_dim * 2], bf16)
        output_l3_type = MemRefType.get([sequence_length, hidden_size], bf16)
        output_flat_l3_type = MemRefType.get([sequence_length * hidden_size], bf16)

        feature_tile_size = head_dim if sequence_length == 1 else hidden_size
        row_l1_type = MemRefType.get([feature_tile_size], bf16, memory_space=l1_space)
        rope_lut_l1_type = MemRefType.get([head_dim * 2], bf16, memory_space=l1_space)
        rope_function = (
            BF16_ROPE_128_TILE_FUNCTION
            if sequence_length == 1
            else BF16_ROPE_HEADS_TILE_FUNCTION
        )

        rope_tile_func = FuncOp(
            rope_function,
            FunctionType.get([row_l1_type, rope_lut_l1_type, row_l1_type], []),
            visibility="private",
        )
        rope_tile_func.attributes["link_with"] = StringAttr.get(BF16_ELEMENTWISE_LINK_OBJECT)
        rope_tile_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

        @FuncOp.from_py_func(
            norm_l3_type,
            rope_lut_l3_type,
            output_l3_type,
            name=function_name,
        )
        def rope_apply_export(
            norm_l3: Value,
            rope_lut_l3: Value,
            output_l3: Value,
        ) -> None:
            @air_launch(operands=(norm_l3, rope_lut_l3, output_l3))
            def rope_launch_body(
                norm_arg: Value,
                rope_lut_arg: Value,
                output_arg: Value,
            ) -> None:
                norm_flat = memref_collapse_shape(norm_flat_l3_type, norm_arg, [[0, 1]])
                rope_lut_flat = memref_collapse_shape(
                    rope_lut_flat_l3_type,
                    rope_lut_arg,
                    [[0, 1]],
                )
                output_flat = memref_collapse_shape(output_flat_l3_type, output_arg, [[0, 1]])

                @air_segment(
                    name="rope_seg",
                    operands=(norm_flat, rope_lut_flat, output_flat),
                )
                def rope_segment_body(
                    norm_seg: Value,
                    rope_lut_seg: Value,
                    output_seg: Value,
                ) -> None:
                    @air_herd(
                        name="rope_export",
                        sizes=[herd_rows, herd_cols],
                        operands=(norm_seg, rope_lut_seg, output_seg),
                    )
                    def rope_herd_body(
                        tile_i: Value,
                        tile_j: Value,
                        _size_i: Value,
                        _size_j: Value,
                        norm: Value,
                        rope_lut: Value,
                        output: Value,
                    ) -> None:
                        norm_l1 = AllocOp(row_l1_type, [], [])
                        rope_lut_l1 = AllocOp(rope_lut_l1_type, [], [])
                        output_l1 = AllocOp(row_l1_type, [], [])

                        token_block_base = arith.muli(tile_i, idx(tokens_per_tile))
                        for local_token_i in range_(tokens_per_tile):
                            token_i = arith.addi(token_block_base, local_token_i)
                            lut_base = arith.muli(token_i, idx(head_dim * 2))
                            dma_memcpy_nd(
                                rope_lut_l1,
                                rope_lut,
                                src_offsets=[lut_base],
                                src_sizes=[head_dim * 2],
                                src_strides=[1],
                            )
                            if sequence_length == 1:
                                for head_i in range_(tile_j, head_count, herd_cols):
                                    row_base = arith.muli(token_i, idx(hidden_size))
                                    head_base = arith.muli(head_i, idx(head_dim))
                                    flat_base = arith.addi(row_base, head_base)
                                    dma_memcpy_nd(
                                        norm_l1,
                                        norm,
                                        src_offsets=[flat_base],
                                        src_sizes=[head_dim],
                                        src_strides=[1],
                                    )
                                    CallOp(rope_tile_func, [norm_l1, rope_lut_l1, output_l1])
                                    dma_memcpy_nd(
                                        output,
                                        output_l1,
                                        dst_offsets=[flat_base],
                                        dst_sizes=[head_dim],
                                        dst_strides=[1],
                                    )
                                    yield_([])
                            else:
                                row_base = arith.muli(token_i, idx(hidden_size))
                                dma_memcpy_nd(
                                    norm_l1,
                                    norm,
                                    src_offsets=[row_base],
                                    src_sizes=[hidden_size],
                                    src_strides=[1],
                                )
                                CallOp(rope_tile_func, [norm_l1, rope_lut_l1, output_l1])
                                dma_memcpy_nd(
                                    output,
                                    output_l1,
                                    dst_offsets=[row_base],
                                    dst_sizes=[hidden_size],
                                    dst_strides=[1],
                                )
                            yield_([])

                        DeallocOp(norm_l1)
                        DeallocOp(rope_lut_l1)
                        DeallocOp(output_l1)

    return build()
