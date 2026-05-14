from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..air_runtime import compile_external_kernel_object
from ..run_q_proj import KERNEL_DIR
from .rope import HEAD_DIM, QKVRopePrepared, prepare_qkv_rope

ATTENTION_CORE_KERNEL_SOURCE = KERNEL_DIR / "attention_core.cc"


@dataclass(frozen=True, slots=True)
class AttentionPrepared:
    rope: QKVRopePrepared
    attention_output: np.ndarray
    attention_expected: torch.Tensor


def prepare_attention(
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
) -> AttentionPrepared:
    rope = prepare_qkv_rope(
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
    attention_expected = reference_attention_core(
        q=rope.norm_rope_expected["q_norm_rope"],
        k=rope.norm_rope_expected["k_norm_rope"],
        v=rope.projection.projection_expected["v_proj"],
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    attention_output = np.zeros(tuple(attention_expected.shape), dtype=np.float32)
    return AttentionPrepared(
        rope=rope,
        attention_output=attention_output,
        attention_expected=attention_expected,
    )


def reference_attention_core(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int = 1,
    kv_heads: int = 1,
) -> torch.Tensor:
    q_t = q.to(torch.float32)
    k_t = k.to(device=q_t.device, dtype=torch.float32)
    v_t = v.to(device=q_t.device, dtype=torch.float32)
    sequence_length, q_total_dim = q_t.shape
    if q_total_dim != q_heads * HEAD_DIM:
        raise ValueError(f"q has {q_total_dim} columns, expected {q_heads * HEAD_DIM}")
    if k_t.shape != (sequence_length, kv_heads * HEAD_DIM):
        raise ValueError(
            f"k has shape {tuple(k_t.shape)}, expected {(sequence_length, kv_heads * HEAD_DIM)}"
        )
    if v_t.shape != (sequence_length, kv_heads * HEAD_DIM):
        raise ValueError(
            f"v has shape {tuple(v_t.shape)}, expected {(sequence_length, kv_heads * HEAD_DIM)}"
        )
    if q_heads % kv_heads != 0:
        raise ValueError(f"q_heads={q_heads} must be a multiple of kv_heads={kv_heads}")

    q_by_head = q_t.reshape(sequence_length, q_heads, HEAD_DIM)
    k_by_head = k_t.reshape(sequence_length, kv_heads, HEAD_DIM)
    v_by_head = v_t.reshape(sequence_length, kv_heads, HEAD_DIM)
    mask = torch.triu(
        torch.ones((sequence_length, sequence_length), device=q_t.device, dtype=torch.bool),
        diagonal=1,
    )
    outputs: list[torch.Tensor] = []
    q_heads_per_kv_head = q_heads // kv_heads
    for q_head in range(q_heads):
        kv_head = q_head // q_heads_per_kv_head
        q_head_t = q_by_head[:, q_head, :]
        k_head_t = k_by_head[:, kv_head, :]
        v_head_t = v_by_head[:, kv_head, :]
        scores = torch.matmul(q_head_t, k_head_t.transpose(0, 1)) * (
            1.0 / math.sqrt(float(HEAD_DIM))
        )
        probs = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        outputs.append(torch.matmul(probs, v_head_t).to(torch.float32))
    return torch.stack(outputs, dim=1).reshape(sequence_length, q_total_dim).to(torch.float32)


def compile_attention_core_object(
    *,
    work_dir: Path,
    peano_install_dir: str,
    head_dim: int,
    sequence_length: int,
    query_tile_rows: int,
    key_tile_rows: int,
) -> Path:
    return compile_external_kernel_object(
        source=ATTENTION_CORE_KERNEL_SOURCE,
        object_name="attention_core.o",
        work_dir=work_dir,
        peano_install_dir=peano_install_dir,
        defines={
            "HEAD_DIM": head_dim,
            "SEQUENCE_LENGTH": sequence_length,
            "QUERY_TILE_ROWS": query_tile_rows,
            "KEY_TILE_ROWS": key_tile_rows,
        },
    )
