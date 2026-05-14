from __future__ import annotations

from torch2air.export.builder import KernelAttr, TensorInfo


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

    def render_air(self) -> str:
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
        return render_q4k_embedding_air(
            function_name=self.function_name,
            sequence_length=sequence_length,
            blocks_per_row=blocks_per_row,
            row_words=row_words,
            hidden_size=hidden_size,
        )


def render_q4k_embedding_air(
    *,
    function_name: str,
    sequence_length: int,
    blocks_per_row: int,
    row_words: int,
    hidden_size: int,
) -> str:
    lines: list[str] = [
        "// Generated from Python kernel by torch2air.export.",
        "module {",
        f"  func.func @{function_name}(",
        f"      %packed_rows: memref<{sequence_length}x{row_words}xi32>,",
        "      %block_f16_scales: "
        f"memref<{sequence_length}x{blocks_per_row}x2xf32>,",
        f"      %output: memref<{sequence_length}x{hidden_size}xf32>) {{",
    ]
    _emit_constants(
        lines,
        sequence_length=sequence_length,
        blocks_per_row=blocks_per_row,
        row_words=row_words,
        hidden_size=hidden_size,
    )
    lines.extend(
        [
            "",
            "    scf.parallel (%launch_i, %launch_j) = (%idx0, %idx0) "
            "to (%idx1, %idx1) step (%idx1, %idx1) {",
            "      scf.parallel (%tile_i, %tile_j) = (%idx0, %idx0) "
            f"to (%idx1, %idx{blocks_per_row}) step (%idx1, %idx1) {{",
            f"        scf.for %token_i = %idx0 to %idx{sequence_length} step %idx1 {{",
        ]
    )
    _emit_q4k_tile(
        lines,
        prefix="tile",
        sequence_length=sequence_length,
        row_words=row_words,
        blocks_per_row=blocks_per_row,
        hidden_size=hidden_size,
    )
    lines.extend(
        [
            "        }",
            "        scf.reduce",
            "      }",
            "      scf.reduce",
            "    }",
            "",
            "    return",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _emit_constants(
    lines: list[str],
    *,
    sequence_length: int,
    blocks_per_row: int,
    row_words: int,
    hidden_size: int,
) -> None:
    index_constants = set(range(0, 257))
    index_constants.update(range(0, row_words + 1, 36))
    index_constants.update(range(0, hidden_size + 1, 256))
    index_constants.update({sequence_length, blocks_per_row, row_words, hidden_size})
    for value in sorted(index_constants):
        lines.append(f"    %idx{value} = arith.constant {value} : index")
    int_constants = set(range(0, 25))
    int_constants.update({48, 63, 255})
    for value in sorted(int_constants):
        lines.append(f"    %i32_{value} = arith.constant {value} : i32")


def _emit_byte_at(lines: list[str], prefix: str, name: str, byte_offset: int) -> None:
    word_offset = byte_offset // 4
    bit_offset = (byte_offset % 4) * 8
    lines.extend(
        [
            f"      %{prefix}_{name}_word = memref.load %{prefix}_packed_l1"
            f"[%idx0, %idx{word_offset}] : memref<1x36xi32, 2>",
            f"      %{prefix}_{name}_shifted = arith.shrui %{prefix}_{name}_word, "
            f"%i32_{bit_offset} : i32",
            f"      %{prefix}_{name} = arith.andi %{prefix}_{name}_shifted, %i32_255 : i32",
        ]
    )


def _emit_q4k_tile(
    lines: list[str],
    *,
    prefix: str,
    sequence_length: int,
    row_words: int,
    blocks_per_row: int,
    hidden_size: int,
) -> None:
    lines.extend(
        [
            f"      %{prefix}_packed_col = arith.muli %tile_j, %idx36 : index",
            f"      %{prefix}_out_col = arith.muli %tile_j, %idx256 : index",
            f"      %{prefix}_packed_tile = memref.subview %packed_rows"
            f"[%token_i, %{prefix}_packed_col] [1, 36] [1, 1]",
            f"          : memref<{sequence_length}x{row_words}xi32> to "
            f"memref<1x36xi32, strided<[{row_words}, 1], offset: ?>>",
            f"      %{prefix}_scale_tile = memref.subview %block_f16_scales"
            f"[%token_i, %tile_j, %idx0] [1, 1, 2] [1, 1, 1]",
            f"          : memref<{sequence_length}x{blocks_per_row}x2xf32> to "
            f"memref<1x1x2xf32, strided<[{blocks_per_row * 2}, 2, 1], offset: ?>>",
            f"      %{prefix}_out_tile = memref.subview %output"
            f"[%token_i, %{prefix}_out_col] [1, 256] [1, 1]",
            f"          : memref<{sequence_length}x{hidden_size}xf32> to memref<1x256xf32, "
            f"strided<[{hidden_size}, 1], offset: ?>>",
            "",
            f"      %{prefix}_packed_l1 = memref.alloc() : memref<1x36xi32, 2>",
            f"      %{prefix}_scale_l1 = memref.alloc() : memref<1x1x2xf32, 2>",
            f"      %{prefix}_out_l1 = memref.alloc() : memref<1x256xf32, 2>",
            "",
            f"      memref.copy %{prefix}_packed_tile, %{prefix}_packed_l1",
            f"          : memref<1x36xi32, strided<[{row_words}, 1], offset: ?>> "
            "to memref<1x36xi32, 2>",
            f"      memref.copy %{prefix}_scale_tile, %{prefix}_scale_l1",
            f"          : memref<1x1x2xf32, strided<[{blocks_per_row * 2}, 2, 1], "
            "offset: ?>> to memref<1x1x2xf32, 2>",
            "",
            f"      %{prefix}_d = memref.load %{prefix}_scale_l1[%idx0, %idx0, %idx0] "
            ": memref<1x1x2xf32, 2>",
            f"      %{prefix}_dmin = memref.load %{prefix}_scale_l1[%idx0, %idx0, %idx1] "
            ": memref<1x1x2xf32, 2>",
            "",
        ]
    )
    _emit_scale_min(lines, prefix)
    _emit_quant_values(lines, prefix)
    lines.extend(
        [
            f"      memref.copy %{prefix}_out_l1, %{prefix}_out_tile",
            "          : memref<1x256xf32, 2> to "
            f"memref<1x256xf32, strided<[{hidden_size}, 1], offset: ?>>",
            "",
            f"      memref.dealloc %{prefix}_packed_l1 : memref<1x36xi32, 2>",
            f"      memref.dealloc %{prefix}_scale_l1 : memref<1x1x2xf32, 2>",
            f"      memref.dealloc %{prefix}_out_l1 : memref<1x256xf32, 2>",
        ]
    )


def _emit_scale_min(lines: list[str], prefix: str) -> None:
    for subblock in range(8):
        if subblock < 4:
            _emit_byte_at(lines, prefix, f"s{subblock}_scale_byte", 4 + subblock)
            _emit_byte_at(lines, prefix, f"s{subblock}_min_byte", 8 + subblock)
            lines.extend(
                [
                    f"      %{prefix}_s{subblock}_scale = arith.andi "
                    f"%{prefix}_s{subblock}_scale_byte, %i32_63 : i32",
                    f"      %{prefix}_s{subblock}_min = arith.andi "
                    f"%{prefix}_s{subblock}_min_byte, %i32_63 : i32",
                ]
            )
        else:
            _emit_byte_at(lines, prefix, f"s{subblock}_d_byte", subblock)
            _emit_byte_at(lines, prefix, f"s{subblock}_m_byte", 4 + subblock)
            _emit_byte_at(lines, prefix, f"s{subblock}_md_byte", 8 + subblock)
            lines.extend(
                [
                    f"      %{prefix}_s{subblock}_scale_lo = arith.andi "
                    f"%{prefix}_s{subblock}_md_byte, %i32_15 : i32",
                    f"      %{prefix}_s{subblock}_scale_hi_0 = arith.shrui "
                    f"%{prefix}_s{subblock}_d_byte, %i32_2 : i32",
                    f"      %{prefix}_s{subblock}_scale_hi = arith.andi "
                    f"%{prefix}_s{subblock}_scale_hi_0, %i32_48 : i32",
                    f"      %{prefix}_s{subblock}_scale = arith.ori "
                    f"%{prefix}_s{subblock}_scale_lo, %{prefix}_s{subblock}_scale_hi : i32",
                    f"      %{prefix}_s{subblock}_min_lo = arith.shrui "
                    f"%{prefix}_s{subblock}_md_byte, %i32_4 : i32",
                    f"      %{prefix}_s{subblock}_min_hi_0 = arith.shrui "
                    f"%{prefix}_s{subblock}_m_byte, %i32_2 : i32",
                    f"      %{prefix}_s{subblock}_min_hi = arith.andi "
                    f"%{prefix}_s{subblock}_min_hi_0, %i32_48 : i32",
                    f"      %{prefix}_s{subblock}_min = arith.ori "
                    f"%{prefix}_s{subblock}_min_lo, %{prefix}_s{subblock}_min_hi : i32",
                ]
            )
        lines.extend(
            [
                f"      %{prefix}_s{subblock}_scale_f32 = arith.uitofp "
                f"%{prefix}_s{subblock}_scale : i32 to f32",
                f"      %{prefix}_s{subblock}_min_f32 = arith.uitofp "
                f"%{prefix}_s{subblock}_min : i32 to f32",
                f"      %{prefix}_s{subblock}_d_scale = arith.mulf %{prefix}_d, "
                f"%{prefix}_s{subblock}_scale_f32 : f32",
                f"      %{prefix}_s{subblock}_min_scale = arith.mulf %{prefix}_dmin, "
                f"%{prefix}_s{subblock}_min_f32 : f32",
            ]
        )


def _emit_quant_values(lines: list[str], prefix: str) -> None:
    for pair in range(4):
        lines.extend(
            [
                f"      scf.for %{prefix}_q_word_i{pair} = %idx0 to %idx8 step %idx1 {{",
                f"        %{prefix}_q_word_index{pair} = arith.addi "
                f"%{prefix}_q_word_i{pair}, %idx{4 + pair * 8} : index",
                f"        %{prefix}_q_word{pair} = memref.load %{prefix}_packed_l1"
                f"[%idx0, %{prefix}_q_word_index{pair}] : memref<1x36xi32, 2>",
                f"        %{prefix}_q_out_word_base{pair} = arith.muli "
                f"%{prefix}_q_word_i{pair}, %idx4 : index",
            ]
        )
        for byte in range(4):
            _emit_quant_byte(lines, prefix, pair, byte)
        lines.append("      }")


def _emit_quant_byte(lines: list[str], prefix: str, pair: int, byte: int) -> None:
    even_subblock = pair * 2
    odd_subblock = pair * 2 + 1
    lines.extend(
        [
            f"        %{prefix}_q{pair}_b{byte}_shifted = arith.shrui "
            f"%{prefix}_q_word{pair}, %i32_{byte * 8} : i32",
            f"        %{prefix}_q{pair}_b{byte}_byte = arith.andi "
            f"%{prefix}_q{pair}_b{byte}_shifted, %i32_255 : i32",
            f"        %{prefix}_q{pair}_b{byte}_lo = arith.andi "
            f"%{prefix}_q{pair}_b{byte}_byte, %i32_15 : i32",
            f"        %{prefix}_q{pair}_b{byte}_hi = arith.shrui "
            f"%{prefix}_q{pair}_b{byte}_byte, %i32_4 : i32",
            f"        %{prefix}_q{pair}_b{byte}_lo_f32 = arith.uitofp "
            f"%{prefix}_q{pair}_b{byte}_lo : i32 to f32",
            f"        %{prefix}_q{pair}_b{byte}_hi_f32 = arith.uitofp "
            f"%{prefix}_q{pair}_b{byte}_hi : i32 to f32",
            f"        %{prefix}_even{pair}_b{byte}_scaled = arith.mulf "
            f"%{prefix}_s{even_subblock}_d_scale, %{prefix}_q{pair}_b{byte}_lo_f32 : f32",
            f"        %{prefix}_even{pair}_b{byte}_value = arith.subf "
            f"%{prefix}_even{pair}_b{byte}_scaled, %{prefix}_s{even_subblock}_min_scale : f32",
            f"        %{prefix}_odd{pair}_b{byte}_scaled = arith.mulf "
            f"%{prefix}_s{odd_subblock}_d_scale, %{prefix}_q{pair}_b{byte}_hi_f32 : f32",
            f"        %{prefix}_odd{pair}_b{byte}_value = arith.subf "
            f"%{prefix}_odd{pair}_b{byte}_scaled, %{prefix}_s{odd_subblock}_min_scale : f32",
            f"        %{prefix}_out_word_byte{pair}_{byte} = arith.addi "
            f"%{prefix}_q_out_word_base{pair}, %idx{byte} : index",
            f"        %{prefix}_out_even{pair}_{byte} = arith.addi "
            f"%{prefix}_out_word_byte{pair}_{byte}, %idx{pair * 64} : index",
            f"        %{prefix}_out_odd{pair}_{byte} = arith.addi "
            f"%{prefix}_out_word_byte{pair}_{byte}, %idx{pair * 64 + 32} : index",
            f"        memref.store %{prefix}_even{pair}_b{byte}_value, %{prefix}_out_l1"
            f"[%idx0, %{prefix}_out_even{pair}_{byte}] : memref<1x256xf32, 2>",
            f"        memref.store %{prefix}_odd{pair}_b{byte}_value, %{prefix}_out_l1"
            f"[%idx0, %{prefix}_out_odd{pair}_{byte}] : memref<1x256xf32, 2>",
        ]
    )
