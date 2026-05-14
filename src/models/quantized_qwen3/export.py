from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch2air.export import export_one


DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("generated")
STAGES = (
    "embed_tokens",
    "input_layernorm",
    "embed_tokens_input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_norm_rope",
    "k_norm_rope",
    "attention_core",
)


class EmbedTokensInputLayerNorm(torch.nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.embed_tokens = cast(torch.nn.Module, model.model.embed_tokens)
        self.input_layernorm = cast(torch.nn.Module, model.model.layers[0].input_layernorm)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(self.embed_tokens(input_ids))


class RMSNormRope(torch.nn.Module):
    def __init__(self, norm: torch.nn.Module, head_count: int, head_dim: int) -> None:
        super().__init__()
        self.norm = norm
        self.head_count = head_count
        self.head_dim = head_dim

    def forward(self, source: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        leading_shape = source.shape[:-1]
        heads = source.reshape(*leading_shape, self.head_count, self.head_dim)
        normed = self.norm(heads)
        half_dim = self.head_dim // 2
        rotated = torch.cat((-normed[..., half_dim:], normed[..., :half_dim]), dim=-1)
        output = normed * cos.unsqueeze(-2) + rotated * sin.unsqueeze(-2)
        return output.reshape(*leading_shape, self.head_count * self.head_dim)


class AttentionCore(torch.nn.Module):
    def __init__(self, q_heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, _ = q.shape
        query = q.reshape(batch, sequence_length, self.q_heads, self.head_dim).transpose(1, 2)
        key = k.reshape(batch, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = v.reshape(batch, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        repeat = self.q_heads // self.kv_heads
        if repeat != 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return output.transpose(1, 2).reshape(batch, sequence_length, self.q_heads * self.head_dim)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Qwen3 stages to Python AIR kernels.")
    parser.add_argument("--stage", choices=STAGES, default="embed_tokens")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")

    # Kept for script compatibility while the new exporter is rebuilt.
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--blocks-per-row", type=int)
    parser.add_argument("--output-rows", type=int)
    parser.add_argument("--output-tile-rows", type=int)
    parser.add_argument("--query-tile-rows", type=int)
    parser.add_argument("--key-tile-rows", type=int)
    parser.add_argument("--attention-head-index", type=int, default=0)
    args = parser.parse_args()

    if args.dry_run:
        print("stages:")
        for stage in STAGES:
            print(f"  {stage}")
        print(f"output_dir: {args.output_dir}")
        return 0

    output = export_stage(
        stage=args.stage,
        model_id=args.model_id,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        head_dim=args.head_dim,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    print(f"generated {output}")
    return 0


def export_stage(
    *,
    stage: str,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
) -> Path:
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    hidden_size = int(config.hidden_size)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
        module, args = _stage_module_and_args(
            stage=stage,
            model=model,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            head_dim=head_dim,
            q_heads=q_heads,
            kv_heads=kv_heads,
        )
    return export_one(f"run_{stage}", module, args, output_dir=output_dir)


def _stage_module_and_args(
    *,
    stage: str,
    model: Qwen3ForCausalLM,
    sequence_length: int,
    hidden_size: int,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    layer = model.model.layers[0]
    attn = cast(torch.nn.Module, layer.self_attn)
    if stage == "embed_tokens":
        return cast(torch.nn.Module, model.model.embed_tokens), (
            torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),
        )
    if stage == "input_layernorm":
        return cast(torch.nn.Module, layer.input_layernorm), (
            torch.zeros((1, sequence_length, hidden_size), dtype=torch.float32, device="meta"),
        )
    if stage == "embed_tokens_input_layernorm":
        return EmbedTokensInputLayerNorm(model), (
            torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),
        )
    if stage in {"q_proj", "k_proj", "v_proj"}:
        return cast(torch.nn.Module, getattr(attn, stage)), (
            torch.zeros((1, sequence_length, hidden_size), dtype=torch.float32, device="meta"),
        )
    if stage == "o_proj":
        return cast(torch.nn.Module, getattr(attn, stage)), (
            torch.zeros(
                (1, sequence_length, q_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
        )
    if stage == "q_norm_rope":
        return RMSNormRope(cast(torch.nn.Module, getattr(attn, "q_norm")), q_heads, head_dim), (
            torch.zeros(
                (1, sequence_length, q_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
            torch.zeros((1, sequence_length, head_dim), dtype=torch.float32, device="meta"),
            torch.zeros((1, sequence_length, head_dim), dtype=torch.float32, device="meta"),
        )
    if stage == "k_norm_rope":
        return RMSNormRope(cast(torch.nn.Module, getattr(attn, "k_norm")), kv_heads, head_dim), (
            torch.zeros(
                (1, sequence_length, kv_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
            torch.zeros((1, sequence_length, head_dim), dtype=torch.float32, device="meta"),
            torch.zeros((1, sequence_length, head_dim), dtype=torch.float32, device="meta"),
        )
    if stage == "attention_core":
        return AttentionCore(q_heads, kv_heads, head_dim), (
            torch.zeros(
                (1, sequence_length, q_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
            torch.zeros(
                (1, sequence_length, kv_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
            torch.zeros(
                (1, sequence_length, kv_heads * head_dim),
                dtype=torch.float32,
                device="meta",
            ),
        )
    raise ValueError(f"unsupported stage: {stage}")


if __name__ == "__main__":
    raise SystemExit(main())
