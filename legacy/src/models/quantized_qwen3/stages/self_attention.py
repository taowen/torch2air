from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index

from .. import reference
from ..run_q_proj import prepare_projection_weights
from .attention import AttentionPrepared, prepare_attention

DEFAULT_O_PROJ_WEIGHT_TENSOR = "model.layers.0.self_attn.o_proj.weight"


@dataclass(frozen=True, slots=True)
class SelfAttentionPrepared:
    attention: AttentionPrepared
    oproj_weights: np.ndarray
    oproj_output: np.ndarray
    oproj_expected: torch.Tensor
    oproj_weight_tensor: GGUFTensorEntry


def prepare_self_attn(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    projection_tensors: dict[str, str],
    projection_output_rows: dict[str, int],
    q_norm_weight_tensor: str,
    k_norm_weight_tensor: str,
    start_position: int,
    q_heads: int,
    kv_heads: int,
    oproj_tensor: str,
    oproj_output_rows: int,
) -> SelfAttentionPrepared:
    attention = prepare_attention(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
        projection_tensors=projection_tensors,
        projection_output_rows=projection_output_rows,
        q_norm_weight_tensor=q_norm_weight_tensor,
        k_norm_weight_tensor=k_norm_weight_tensor,
        start_position=start_position,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    oproj_weights, _ = prepare_projection_weights(
        gguf_path=gguf_path,
        tensor_name=oproj_tensor,
        output_rows=oproj_output_rows,
        hidden_size=attention.attention_expected.shape[1],
    )
    oproj_expected = reference_oproj(attention.attention_expected)[:, :oproj_output_rows]
    oproj_output = np.zeros(tuple(oproj_expected.shape), dtype=np.float32)
    oproj_entry = load_gguf_index(gguf_path).tensors[oproj_tensor]
    return SelfAttentionPrepared(
        attention=attention,
        oproj_weights=oproj_weights,
        oproj_output=oproj_output,
        oproj_expected=oproj_expected,
        oproj_weight_tensor=oproj_entry,
    )


def reference_oproj(input: torch.Tensor) -> torch.Tensor:
    root = cast(Qwen3ForCausalLM, reference.get_model())
    inner_model = cast(torch.nn.Module, getattr(root, "model"))
    layers = cast(torch.nn.ModuleList, getattr(inner_model, "layers"))
    layer0 = cast(torch.nn.Module, layers[0])
    attn = cast(torch.nn.Module, getattr(layer0, "self_attn"))
    oproj = cast(torch.nn.Module, getattr(attn, "o_proj"))
    with torch.no_grad():
        output = oproj(input.reshape(1, input.shape[0], input.shape[1]))
    return output.reshape(input.shape[0], -1).to(torch.float32)
