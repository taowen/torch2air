from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from torch2air.export import (
    render_exported_reference_function,
    render_reference_module,
    render_to_file,
)
from torch2air.export.kernels import TEMPLATE_DIR as KERNEL_TEMPLATE_DIR
from torch2air.weights.gguf import load_gguf_index


DEFAULT_GGUF = Path("/var/home/taowen/projects/torch2vk/dist/quantized_qwen3/model.gguf")
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
ATTENTION_PROJ_NAMES = ("q_proj", "k_proj", "v_proj")
TEMPLATE_DIR = Path(__file__).with_name("templates")


class EmbedTokensInputLayerNorm(torch.nn.Module):
    def __init__(self, model: Qwen3ForCausalLM) -> None:
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.input_layernorm = model.model.layers[0].input_layernorm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.input_layernorm(self.embed_tokens(input_ids))


def export_one(
    name: str,
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, object] | None = None,
    *,
    output_dir: Path,
    template_dir: Path | list[Path] = TEMPLATE_DIR,
    weight_prefix: str = "",
    shape_exprs: dict[int, str] | None = None,
    template_name: str,
    context: dict[str, object],
) -> None:
    program = torch.export.export(module, args, kwargs=kwargs, strict=False)
    nodes = [
        {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "shape": _shape_of(node),
            "dtype": _dtype_of(node),
        }
        for node in program.graph_module.graph.nodes
    ]
    aten_targets = [node["target"] for node in nodes if node["op"] == "call_function"]
    render_to_file(
        template_dir,
        template_name,
        output_dir / f"{name}.mlir",
        stage_name=name,
        weight_prefix=weight_prefix,
        shape_exprs=shape_exprs or {},
        nodes=nodes,
        aten_targets=aten_targets,
        **context,
    )
    print(f"  {name}: {sum(1 for node in nodes if node['op'] == 'call_function')} aten ops")


def export_embed_tokens(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    blocks_per_row_override: int | None = None,
) -> None:
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model_hidden_size = int(config.hidden_size)
    if model_hidden_size % 256 != 0:
        raise ValueError(f"Q4_K hidden_size must be divisible by 256, got {model_hidden_size}")
    model_blocks_per_row = model_hidden_size // 256
    blocks_per_row = blocks_per_row_override or model_blocks_per_row
    if blocks_per_row <= 0 or blocks_per_row > model_blocks_per_row:
        raise ValueError(f"blocks_per_row must be in [1, {model_blocks_per_row}], got {blocks_per_row}")
    hidden_size = blocks_per_row * 256
    row_words = blocks_per_row * 36
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    export_one(
        "run_embed_tokens",
        model.model.embed_tokens,
        args=(torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),),
        output_dir=output_dir,
        weight_prefix="model.embed_tokens.",
        shape_exprs={sequence_length: "sequence_length"},
        template_name="embed_tokens.mlir.j2",
        context={
            "blocks_per_row": blocks_per_row,
            "hidden_size": hidden_size,
            "model_hidden_size": model_hidden_size,
            "model_blocks_per_row": model_blocks_per_row,
            "row_words": row_words,
            "sequence_length": sequence_length,
            "vocab_size": int(config.vocab_size),
        },
    )


def export_embed_tokens_input_layernorm(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    blocks_per_row_override: int | None = None,
) -> None:
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model_hidden_size = int(config.hidden_size)
    if model_hidden_size % 256 != 0:
        raise ValueError(f"Q4_K hidden_size must be divisible by 256, got {model_hidden_size}")
    model_blocks_per_row = model_hidden_size // 256
    blocks_per_row = blocks_per_row_override or model_blocks_per_row
    if blocks_per_row <= 0 or blocks_per_row > model_blocks_per_row:
        raise ValueError(f"blocks_per_row must be in [1, {model_blocks_per_row}], got {blocks_per_row}")
    hidden_size = blocks_per_row * 256
    row_words = blocks_per_row * 36
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
        module = EmbedTokensInputLayerNorm(model)
    export_one(
        "run_embed_tokens_input_layernorm",
        module,
        args=(torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),),
        output_dir=output_dir,
        template_dir=KERNEL_TEMPLATE_DIR,
        weight_prefix="model.",
        shape_exprs={sequence_length: "sequence_length"},
        template_name="embed_tokens_input_layernorm.mlir.j2",
        context={
            "blocks_per_row": blocks_per_row,
            "hidden_size": hidden_size,
            "model_hidden_size": model_hidden_size,
            "model_blocks_per_row": model_blocks_per_row,
            "rms_norm_eps": float(config.rms_norm_eps),
            "row_words": row_words,
            "sequence_length": sequence_length,
            "vocab_size": int(config.vocab_size),
        },
    )


def export_input_layernorm(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    blocks_per_row_override: int | None = None,
) -> None:
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model_hidden_size = int(config.hidden_size)
    if model_hidden_size % 256 != 0:
        raise ValueError(f"Q4_K hidden_size must be divisible by 256, got {model_hidden_size}")
    model_blocks_per_row = model_hidden_size // 256
    blocks_per_row = blocks_per_row_override or model_blocks_per_row
    if blocks_per_row != model_blocks_per_row:
        raise ValueError(
            f"input_layernorm currently requires the full hidden size: "
            f"blocks_per_row={model_blocks_per_row}, got {blocks_per_row}"
        )
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    export_one(
        "run_input_layernorm",
        model.model.layers[0].input_layernorm,
        args=(torch.zeros((1, sequence_length, model_hidden_size), dtype=torch.float32, device="meta"),),
        output_dir=output_dir,
        template_dir=KERNEL_TEMPLATE_DIR,
        weight_prefix="model.layers.0.input_layernorm.",
        shape_exprs={sequence_length: "sequence_length"},
        template_name="input_layernorm.mlir.j2",
        context={
            "hidden_size": model_hidden_size,
            "model_hidden_size": model_hidden_size,
            "model_blocks_per_row": model_blocks_per_row,
            "rms_norm_eps": float(config.rms_norm_eps),
            "sequence_length": sequence_length,
        },
    )


def export_attention_proj(
    *,
    proj_name: str,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    gguf_path: Path = DEFAULT_GGUF,
    output_rows_override: int | None = None,
    output_tile_rows: int = 32,
) -> None:
    if proj_name not in ATTENTION_PROJ_NAMES:
        raise ValueError(f"proj_name must be one of {ATTENTION_PROJ_NAMES}, got {proj_name!r}")
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    model_hidden_size = int(config.hidden_size)
    if model_hidden_size % 256 != 0:
        raise ValueError(f"Q4_K hidden_size must be divisible by 256, got {model_hidden_size}")
    blocks_per_row = model_hidden_size // 256
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    module = getattr(model.model.layers[0].self_attn, proj_name)
    proj_output_size = int(module.out_features)
    output_rows = output_rows_override or 128
    if output_rows <= 0 or output_rows > proj_output_size:
        raise ValueError(f"output_rows must be in [1, {proj_output_size}], got {output_rows}")
    if output_tile_rows <= 0 or output_rows % output_tile_rows != 0:
        raise ValueError(
            f"output_tile_rows must be positive and divide output_rows={output_rows}, got {output_tile_rows}"
        )
    tensor_name = f"model.layers.0.self_attn.{proj_name}.weight"
    tensor_entry = load_gguf_index(gguf_path).tensors[tensor_name]
    if tensor_entry.ggml_type == "Q4_K":
        template_name = "q4k_linear.mlir.j2"
        weight_words = blocks_per_row * 38
    elif tensor_entry.ggml_type == "Q6_K":
        template_name = "q6k_linear.mlir.j2"
        weight_words = blocks_per_row * 106
    else:
        raise ValueError(f"{tensor_name} is {tensor_entry.ggml_type}, not Q4_K or Q6_K")
    export_one(
        f"run_{proj_name}",
        module,
        args=(torch.zeros((1, sequence_length, model_hidden_size), dtype=torch.float32, device="meta"),),
        output_dir=output_dir,
        template_dir=KERNEL_TEMPLATE_DIR,
        weight_prefix=f"model.layers.0.self_attn.{proj_name}.",
        shape_exprs={sequence_length: "sequence_length"},
        template_name=template_name,
        context={
            "proj_name": proj_name,
            "ggml_type": tensor_entry.ggml_type,
            "blocks_per_row": blocks_per_row,
            "hidden_size": model_hidden_size,
            "model_hidden_size": model_hidden_size,
            "output_rows": output_rows,
            "output_tile_rows": output_tile_rows,
            "output_tiles": output_rows // output_tile_rows,
            "proj_output_size": proj_output_size,
            "row_words": blocks_per_row * 36,
            "weight_words": weight_words,
            "sequence_length": sequence_length,
        },
    )


def export_q_proj(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    gguf_path: Path = DEFAULT_GGUF,
    output_rows_override: int | None = None,
    output_tile_rows: int = 32,
) -> None:
    export_attention_proj(
        proj_name="q_proj",
        model_id=model_id,
        gguf_path=gguf_path,
        output_dir=output_dir,
        sequence_length=sequence_length,
        output_rows_override=output_rows_override,
        output_tile_rows=output_tile_rows,
    )


def export_pipeline_embed_norm(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
    blocks_per_row_override: int | None = None,
) -> None:
    export_embed_tokens(
        model_id=model_id,
        output_dir=output_dir,
        sequence_length=sequence_length,
        blocks_per_row_override=blocks_per_row_override,
    )
    export_input_layernorm(
        model_id=model_id,
        output_dir=output_dir,
        sequence_length=sequence_length,
        blocks_per_row_override=blocks_per_row_override,
    )


def export_reference_module(
    *,
    model_id: str,
    output_dir: Path,
    sequence_length: int,
) -> None:
    config = AutoConfig.from_pretrained(model_id, local_files_only=True)
    hidden_size = int(config.hidden_size)
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
        fused = EmbedTokensInputLayerNorm(model)

    exports = [
        (
            "embed_tokens",
            model.model.embed_tokens,
            (torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),),
            None,
            "root.model.embed_tokens",
        ),
        (
            "input_layernorm",
            model.model.layers[0].input_layernorm,
            (torch.zeros((1, sequence_length, hidden_size), dtype=torch.float32, device="meta"),),
            None,
            "root.model.layers[0].input_layernorm",
        ),
        (
            "embed_tokens_input_layernorm",
            fused,
            (torch.zeros((1, sequence_length), dtype=torch.long, device="meta"),),
            None,
            "_EmbedTokensInputLayerNorm(root)",
        ),
    ]
    for proj_name in ATTENTION_PROJ_NAMES:
        exports.append(
            (
                proj_name,
                getattr(model.model.layers[0].self_attn, proj_name),
                (torch.zeros((1, sequence_length, hidden_size), dtype=torch.float32, device="meta"),),
                None,
                f"root.model.layers[0].self_attn.{proj_name}",
            )
        )
    reference_functions: list[str] = []
    for name, module, args, kwargs, module_expr in exports:
        program = torch.export.export(module, args, kwargs=kwargs, strict=False)
        reference_functions.append(
            render_exported_reference_function(
                program,
                name=name,
                module_expr=module_expr,
            )
        )
    (output_dir / "reference.py").write_text(
        render_reference_module(
            model_id=model_id,
            reference_functions=reference_functions,
        )
    )


def _shape_of(node: object) -> tuple[int | str, ...] | None:
    tensor_meta = getattr(node, "meta", {}).get("tensor_meta")
    if tensor_meta is None:
        return None
    return tuple(dim if isinstance(dim, int) else str(dim) for dim in tensor_meta.shape)


def _dtype_of(node: object) -> str | None:
    tensor_meta = getattr(node, "meta", {}).get("tensor_meta")
    if tensor_meta is None:
        return None
    return str(tensor_meta.dtype).removeprefix("torch.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export quantized Qwen3 AIR stages.")
    parser.add_argument(
        "--stage",
        choices=[
            "embed_tokens",
            "input_layernorm",
            "embed_tokens_input_layernorm",
            "pipeline_embed_norm",
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        default="embed_tokens",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("generated"),
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--blocks-per-row",
        type=int,
        default=None,
        help="Q4_K blocks per token row to export; default exports the model hidden size.",
    )
    parser.add_argument(
        "--output-rows",
        type=int,
        default=None,
        help="attention projection output rows to export; defaults to 128.",
    )
    parser.add_argument(
        "--output-tile-rows",
        type=int,
        default=32,
        help="Q4_K linear output rows computed per AIE tile.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"stage: {args.stage}")
    print(f"model_id: {args.model_id}")
    print(f"gguf: {args.gguf}")
    print(f"output_dir: {args.output_dir}")
    print(f"sequence_length: {args.sequence_length}")
    if args.output_rows is not None:
        print(f"output_rows: {args.output_rows}")
    if args.dry_run:
        print("mode: dry-run")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "embed_tokens":
        export_embed_tokens(
            model_id=args.model_id,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            blocks_per_row_override=args.blocks_per_row,
        )
    elif args.stage == "embed_tokens_input_layernorm":
        export_embed_tokens_input_layernorm(
            model_id=args.model_id,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            blocks_per_row_override=args.blocks_per_row,
        )
    elif args.stage == "input_layernorm":
        export_input_layernorm(
            model_id=args.model_id,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            blocks_per_row_override=args.blocks_per_row,
        )
    elif args.stage == "pipeline_embed_norm":
        export_pipeline_embed_norm(
            model_id=args.model_id,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            blocks_per_row_override=args.blocks_per_row,
        )
    elif args.stage in ATTENTION_PROJ_NAMES:
        export_attention_proj(
            proj_name=args.stage,
            model_id=args.model_id,
            gguf_path=args.gguf,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            output_rows_override=args.output_rows,
            output_tile_rows=args.output_tile_rows,
        )
    export_reference_module(
        model_id=args.model_id,
        output_dir=Path(__file__).parent,
        sequence_length=args.sequence_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
