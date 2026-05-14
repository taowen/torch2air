from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index, read_tensor_bytes

from .. import reference
from ..air_runtime import compile_external_kernel_object
from ..run_embed_tokens import prepare_inputs
from ..run_q_proj import KERNEL_DIR

RMS_NORM_KERNEL_SOURCE = KERNEL_DIR / "rms_norm.cc"


@dataclass(frozen=True, slots=True)
class EmbedNormPrepared:
    packed_rows: np.ndarray
    block_f16_scales: np.ndarray
    rms_weight: np.ndarray
    hidden: np.ndarray
    norm_output: np.ndarray
    embed_expected: torch.Tensor
    norm_expected: torch.Tensor
    embed_tensor: GGUFTensorEntry
    rms_weight_tensor: GGUFTensorEntry
    token_ids: list[int]
    blocks_per_row: int
    model_blocks_per_row: int
    hidden_size: int
    rms_norm_eps: float


def prepare_embed_norm(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
) -> EmbedNormPrepared:
    packed_rows, block_f16_scales, embed_expected, _ = prepare_inputs(
        gguf_path=gguf_path,
        tensor_name="model.embed_tokens.weight",
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
    )
    hidden_size = blocks_per_row * 256
    index = load_gguf_index(gguf_path)
    embed_entry = index.tensors["model.embed_tokens.weight"]
    weight_entry = index.tensors[rms_weight_tensor]
    if weight_entry.ggml_type != "F32" or weight_entry.physical_dtype != "float32":
        raise ValueError(f"{rms_weight_tensor} must be F32, got {weight_entry}")
    if int(weight_entry.physical_shape[0]) < hidden_size:
        raise ValueError(f"{rms_weight_tensor} is too small for hidden_size={hidden_size}")

    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=hidden_size * 4)
    rms_weight = np.frombuffer(payload, dtype=np.float32).copy()

    expected = reference.run_input_layernorm(hidden_states=embed_expected)["mul_1"].reshape(
        len(token_ids),
        hidden_size,
    )
    hidden = np.zeros(tuple(embed_expected.shape), dtype=np.float32)
    output = np.zeros(tuple(expected.shape), dtype=np.float32)

    return EmbedNormPrepared(
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        rms_weight=np.ascontiguousarray(rms_weight),
        hidden=np.ascontiguousarray(hidden),
        norm_output=np.ascontiguousarray(output),
        embed_expected=embed_expected,
        norm_expected=expected,
        embed_tensor=embed_entry,
        rms_weight_tensor=weight_entry,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        model_blocks_per_row=int(embed_entry.physical_shape[1]) // 36,
        hidden_size=hidden_size,
        rms_norm_eps=eps,
    )


def compile_rms_norm_object(
    *,
    work_dir: Path,
    peano_install_dir: str,
    hidden_size: int,
    eps: float,
) -> Path:
    eps_bits = int(np.array([eps], dtype=np.float32).view(np.uint32)[0])
    return compile_external_kernel_object(
        source=RMS_NORM_KERNEL_SOURCE,
        object_name="rms_norm.o",
        work_dir=work_dir,
        peano_install_dir=peano_install_dir,
        defines={
            "HIDDEN_SIZE": hidden_size,
            "RMS_NORM_EPS_BITS": f"0x{eps_bits:08x}",
        },
    )
