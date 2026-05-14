from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyxrt as xrt
import torch

from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index

from .. import reference
from ..run_q_proj import (
    ATTENTION_PROJ_NAMES,
    prepare_q_proj_weights,
    prepare_projection_weights,
    reference_projection,
)
from .embed_norm import EmbedNormPrepared, prepare_embed_norm


@dataclass(frozen=True, slots=True)
class QProjPrepared:
    base: EmbedNormPrepared
    qproj_weights: np.ndarray
    qproj_output: np.ndarray
    qproj_expected: torch.Tensor
    qproj_weight_tensor: GGUFTensorEntry
    qproj_output_rows: int


@dataclass(frozen=True, slots=True)
class QKVPrepared:
    base: EmbedNormPrepared
    projection_weights: dict[str, np.ndarray]
    projection_outputs: dict[str, np.ndarray]
    projection_expected: dict[str, torch.Tensor]
    projection_weight_tensors: dict[str, GGUFTensorEntry]
    projection_output_rows: dict[str, int]


def prepare_qproj(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    qproj_tensor: str,
    output_rows: int,
) -> QProjPrepared:
    base = prepare_embed_norm(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    packed_qproj, _ = prepare_q_proj_weights(
        gguf_path=gguf_path,
        tensor_name=qproj_tensor,
        output_rows=output_rows,
        hidden_size=base.norm_expected.shape[1],
    )
    with torch.no_grad():
        qproj_expected = reference.run_q_proj(input=base.norm_expected)["linear"].reshape(
            len(token_ids),
            -1,
        )[:, :output_rows]
    qproj_output = np.zeros(tuple(qproj_expected.shape), dtype=np.float32)
    qproj_entry = load_gguf_index(gguf_path).tensors[qproj_tensor]
    return QProjPrepared(
        base=base,
        qproj_weights=packed_qproj,
        qproj_output=qproj_output,
        qproj_expected=qproj_expected,
        qproj_weight_tensor=qproj_entry,
        qproj_output_rows=output_rows,
    )


def prepare_qkv(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    projection_tensors: dict[str, str],
    projection_output_rows: dict[str, int],
) -> QKVPrepared:
    base = prepare_embed_norm(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    projection_weights: dict[str, np.ndarray] = {}
    projection_outputs: dict[str, np.ndarray] = {}
    projection_expected: dict[str, torch.Tensor] = {}
    index = load_gguf_index(gguf_path)
    projection_weight_tensors: dict[str, GGUFTensorEntry] = {}
    for proj_name, tensor_name in projection_tensors.items():
        if proj_name not in ATTENTION_PROJ_NAMES:
            raise ValueError(f"Unsupported projection {proj_name!r}")
        output_rows = projection_output_rows[proj_name]
        packed_projection, _ = prepare_projection_weights(
            gguf_path=gguf_path,
            tensor_name=tensor_name,
            output_rows=output_rows,
            hidden_size=base.norm_expected.shape[1],
        )
        with torch.no_grad():
            expected = reference_projection(proj_name, base.norm_expected).reshape(
                len(token_ids),
                -1,
            )[:, :output_rows]
        projection_weights[proj_name] = packed_projection
        projection_outputs[proj_name] = np.zeros(tuple(expected.shape), dtype=np.float32)
        projection_expected[proj_name] = expected
        projection_weight_tensors[proj_name] = index.tensors[tensor_name]

    return QKVPrepared(
        base=base,
        projection_weights=projection_weights,
        projection_outputs=projection_outputs,
        projection_expected=projection_expected,
        projection_weight_tensors=projection_weight_tensors,
        projection_output_rows=projection_output_rows,
    )


def projection_blocks_per_row(entry: GGUFTensorEntry) -> int:
    if entry.ggml_type == "Q4_K":
        return int(entry.physical_shape[1]) // 36
    if entry.ggml_type == "Q6_K":
        return int(entry.physical_shape[1]) // 105
    raise ValueError(f"Unsupported projection kernel type {entry.ggml_type}")


def projection_dispatch_rows(output_rows: int, output_tile_rows: int) -> int:
    if output_rows % output_tile_rows != 0:
        raise ValueError(
            f"output_rows={output_rows} must be divisible by output_tile_rows={output_tile_rows}"
        )
    parallel_tiles = min(4, output_rows // output_tile_rows)
    return parallel_tiles * output_tile_rows


def projection_row_offsets(output_rows: int, output_tile_rows: int) -> range:
    dispatch_rows = projection_dispatch_rows(output_rows, output_tile_rows)
    if output_rows % dispatch_rows != 0:
        raise ValueError(
            f"output_rows={output_rows} must be divisible by dispatch_rows={dispatch_rows}"
        )
    return range(0, output_rows, dispatch_rows)


def run_projection_slices(
    *,
    kernel: xrt.kernel,
    instr_v: np.ndarray,
    bo_instr: xrt.bo,
    bo_input: xrt.bo,
    bo_weights: xrt.bo,
    bo_output: xrt.bo,
    input_array: np.ndarray,
    weights_array: np.ndarray,
    output_array: np.ndarray,
    output_tile_rows: int,
) -> None:
    sequence_length, hidden_size = input_array.shape
    output_sequence_length, output_rows = output_array.shape
    if output_sequence_length != sequence_length:
        raise ValueError("projection input/output sequence lengths must match")

    slice_rows = projection_dispatch_rows(output_rows, output_tile_rows)
    input_row_bytes = hidden_size * input_array.dtype.itemsize
    weight_row_bytes = weights_array.shape[1] * weights_array.dtype.itemsize
    weight_slice_bytes = slice_rows * weight_row_bytes
    output_scalar_bytes = output_array.dtype.itemsize
    output_slice_bytes = slice_rows * output_scalar_bytes

    for token_i in range(sequence_length):
        input_slice = xrt.bo(bo_input, input_row_bytes, token_i * input_row_bytes)
        for row_offset in projection_row_offsets(output_rows, output_tile_rows):
            weight_slice = xrt.bo(bo_weights, weight_slice_bytes, row_offset * weight_row_bytes)
            output_offset = (token_i * output_rows + row_offset) * output_scalar_bytes
            output_slice = xrt.bo(bo_output, output_slice_bytes, output_offset)
            projection_run = kernel(
                3, bo_instr, len(instr_v), input_slice, weight_slice, output_slice
            )
            projection_run.wait()
