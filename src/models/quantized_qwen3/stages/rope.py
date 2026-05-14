from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch2air.weights.gguf import GGUFTensorEntry, load_gguf_index, read_tensor_bytes

from .. import reference
from ..air_runtime import compile_external_kernel_object
from ..run_q_proj import KERNEL_DIR
from .projection import QKVPrepared, prepare_qkv

DEFAULT_Q_NORM_WEIGHT_TENSOR = "model.layers.0.self_attn.q_norm.weight"
DEFAULT_K_NORM_WEIGHT_TENSOR = "model.layers.0.self_attn.k_norm.weight"
HEAD_DIM = 128
ROPE_TABLE_KERNEL_SOURCE = KERNEL_DIR / "rope_table.cc"
RMS_NORM_ROPE_KERNEL_SOURCE = KERNEL_DIR / "rms_norm_rope.cc"


@dataclass(frozen=True, slots=True)
class QKVRopePrepared:
    projection: QKVPrepared
    q_norm_weight: np.ndarray
    k_norm_weight: np.ndarray
    start_position: np.ndarray
    norm_rope_outputs: dict[str, np.ndarray]
    cos_expected: torch.Tensor
    sin_expected: torch.Tensor
    norm_rope_expected: dict[str, torch.Tensor]
    q_norm_weight_tensor: GGUFTensorEntry
    k_norm_weight_tensor: GGUFTensorEntry
    rope_theta: float
    rope_start_position: int
    q_heads: int
    kv_heads: int


def prepare_qkv_rope(
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
) -> QKVRopePrepared:
    expected_rows = {
        "q_proj": q_heads * HEAD_DIM,
        "k_proj": kv_heads * HEAD_DIM,
        "v_proj": kv_heads * HEAD_DIM,
    }
    for proj_name, expected_rows_value in expected_rows.items():
        actual_rows = projection_output_rows[proj_name]
        if actual_rows != expected_rows_value:
            raise ValueError(f"{proj_name} rows must be {expected_rows_value}, got {actual_rows}")
    base = prepare_qkv(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
        projection_tensors=projection_tensors,
        projection_output_rows=projection_output_rows,
    )
    missing = [name for name in ("q_proj", "k_proj") if name not in base.projection_expected]
    if missing:
        raise ValueError(f"q/k norm+RoPE requires q_proj and k_proj outputs; missing {missing}")

    q_norm_weight, q_norm_entry = read_f32_vector(
        gguf_path=gguf_path,
        tensor_name=q_norm_weight_tensor,
        length=HEAD_DIM,
    )
    k_norm_weight, k_norm_entry = read_f32_vector(
        gguf_path=gguf_path,
        tensor_name=k_norm_weight_tensor,
        length=HEAD_DIM,
    )

    theta = reference_rope_theta()
    start_position_array = np.array([start_position], dtype=np.int32)
    cos_expected, sin_expected = reference_rope_table(
        sequence_length=len(token_ids),
        start_position=start_position,
        head_dim=HEAD_DIM,
    )
    norm_rope_expected = {
        "q_norm_rope": reference_norm_rope(
            norm_name="q_norm_rope",
            projection=base.projection_expected["q_proj"],
            cos=cos_expected,
            sin=sin_expected,
            head_count=q_heads,
        ),
        "k_norm_rope": reference_norm_rope(
            norm_name="k_norm_rope",
            projection=base.projection_expected["k_proj"],
            cos=cos_expected,
            sin=sin_expected,
            head_count=kv_heads,
        ),
    }
    norm_rope_outputs = {
        name: np.zeros(tuple(expected.shape), dtype=np.float32)
        for name, expected in norm_rope_expected.items()
    }

    return QKVRopePrepared(
        projection=base,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        start_position=start_position_array,
        norm_rope_outputs=norm_rope_outputs,
        cos_expected=cos_expected,
        sin_expected=sin_expected,
        norm_rope_expected=norm_rope_expected,
        q_norm_weight_tensor=q_norm_entry,
        k_norm_weight_tensor=k_norm_entry,
        rope_theta=theta,
        rope_start_position=start_position,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )


def read_f32_vector(
    *, gguf_path: Path, tensor_name: str, length: int
) -> tuple[np.ndarray, GGUFTensorEntry]:
    index = load_gguf_index(gguf_path)
    selected = index.tensors[tensor_name]
    if selected.ggml_type != "F32" or selected.physical_dtype != "float32":
        raise ValueError(f"{tensor_name} must be F32, got {selected}")
    if int(selected.physical_shape[0]) < length:
        raise ValueError(f"{tensor_name} is too small for length={length}")
    payload = read_tensor_bytes(index.path, selected, offset=0, size=length * 4)
    return np.ascontiguousarray(np.frombuffer(payload, dtype=np.float32).copy()), selected


def reference_rope_theta() -> float:
    config = cast(Qwen3ForCausalLM, reference.get_model()).config
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and "rope_theta" in rope_scaling:
        return float(rope_scaling["rope_theta"])
    configured = getattr(config, "rope_theta", None)
    if configured is not None:
        return float(configured)
    return 10000.0


def reference_rope_table(
    *,
    sequence_length: int,
    start_position: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    root = cast(Qwen3ForCausalLM, reference.get_model())
    device = next(root.parameters()).device
    position_ids = torch.arange(
        start_position,
        start_position + sequence_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    dummy = torch.empty((1, sequence_length, head_dim), dtype=torch.float32, device=device)
    with torch.no_grad():
        cos, sin = root.model.rotary_emb(dummy, position_ids)
    return (
        cos.reshape(sequence_length, head_dim).to(torch.float32),
        sin.reshape(sequence_length, head_dim).to(torch.float32),
    )


def reference_norm_rope(
    *,
    norm_name: str,
    projection: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_count: int,
) -> torch.Tensor:
    root = cast(Qwen3ForCausalLM, reference.get_model())
    attn = root.model.layers[0].self_attn
    if norm_name == "q_norm_rope":
        norm = cast(torch.nn.Module, getattr(attn, "q_norm"))
    elif norm_name == "k_norm_rope":
        norm = cast(torch.nn.Module, getattr(attn, "k_norm"))
    else:
        raise ValueError(f"Unsupported norm+RoPE stage {norm_name!r}")

    sequence_length, total_dim = projection.shape
    if total_dim != head_count * HEAD_DIM:
        raise ValueError(f"{norm_name} expected {head_count * HEAD_DIM} columns, got {total_dim}")
    with torch.no_grad():
        projected = projection.reshape(1, sequence_length, head_count, HEAD_DIM).to(
            device=cos.device, dtype=torch.float32
        )
        normed = norm(projected)
        half_dim = HEAD_DIM // 2
        rotated = torch.cat((-normed[..., half_dim:], normed[..., :half_dim]), dim=-1)
        output = normed * cos.reshape(1, sequence_length, 1, HEAD_DIM) + rotated * sin.reshape(
            1,
            sequence_length,
            1,
            HEAD_DIM,
        )
    return output.reshape(sequence_length, total_dim).to(torch.float32)


def compile_rope_table_object(
    *,
    work_dir: Path,
    peano_install_dir: str,
    head_dim: int,
    rope_theta: float,
) -> Path:
    inv_freq_ratio = math.pow(rope_theta, -2.0 / float(head_dim))
    return compile_external_kernel_object(
        source=ROPE_TABLE_KERNEL_SOURCE,
        object_name="rope_table.o",
        work_dir=work_dir,
        peano_install_dir=peano_install_dir,
        defines={
            "HEAD_DIM": head_dim,
            "ROPE_INV_FREQ_RATIO": f"{inv_freq_ratio:.9g}f",
        },
    )


def compile_rms_norm_rope_object(
    *,
    work_dir: Path,
    peano_install_dir: str,
    head_dim: int,
    eps: float,
) -> Path:
    eps_bits = int(np.array([eps], dtype=np.float32).view(np.uint32)[0])
    return compile_external_kernel_object(
        source=RMS_NORM_ROPE_KERNEL_SOURCE,
        object_name="rms_norm_rope.o",
        work_dir=work_dir,
        peano_install_dir=peano_install_dir,
        defines={
            "HEAD_DIM": head_dim,
            "RMS_NORM_EPS_BITS": f"0x{eps_bits:08x}",
        },
    )
