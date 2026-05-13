"""Kernel manifest for the first quantized Qwen3 AIR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PACKAGE_ROOT / "generated"


@dataclass(frozen=True, slots=True)
class Qwen3ShapeConfig:
    vocab_size: int = 151_936
    hidden_size: int = 1024
    q_projection_size: int = 2048
    kv_projection_size: int = 1024
    intermediate_size: int = 3072
    head_dim: int = 128
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8


QWEN3_0_6B_SHAPES = Qwen3ShapeConfig()


@dataclass(frozen=True, slots=True)
class Qwen3Kernel:
    name: str
    function_name: str
    lhs_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    result_shape: tuple[int, int]
    dtype: torch.dtype
    description: str

    @property
    def linalg_filename(self) -> str:
        return f"{self.name}.linalg.mlir"

    @property
    def air_filename(self) -> str:
        return f"{self.name}.air.mlir"

    @property
    def vmfb_filename(self) -> str:
        return f"{self.name}.vmfb"


KERNELS: dict[str, Qwen3Kernel] = {
    "prefill_q_proj_tile_f32": Qwen3Kernel(
        name="prefill_q_proj_tile_f32",
        function_name="qwen3_prefill_q_proj_tile",
        lhs_shape=(32, QWEN3_0_6B_SHAPES.hidden_size),
        rhs_shape=(QWEN3_0_6B_SHAPES.hidden_size, QWEN3_0_6B_SHAPES.head_dim),
        result_shape=(32, QWEN3_0_6B_SHAPES.head_dim),
        dtype=torch.float32,
        description=(
            "Single-head tile of Qwen3 self_attn.q_proj using dequantized f32 weights; "
            "the quantized Q4_K weight loader is a later step."
        ),
    )
}
