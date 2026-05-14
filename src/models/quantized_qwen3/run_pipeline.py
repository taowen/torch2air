from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .air_runtime import compile_runtime
from .pipeline_runner import (
    run_on_npu,
    run_on_npu_projections,
    run_on_npu_projections_rope,
    run_on_npu_qproj,
)
from .reference_runtime import first_values, max_abs_rocm
from .run_embed_tokens import DEFAULT_GGUF, parse_token_ids
from .run_embed_tokens_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR
from .run_q_proj import (
    DEFAULT_Q_PROJ_WEIGHT_TENSOR,
    compile_projection_object,
    compile_q4k_linear_object,
    projection_weight_tensor,
)
from .stages.attention import AttentionPrepared, compile_attention_core_object, prepare_attention
from .stages.embed_norm import compile_rms_norm_object, prepare_embed_norm
from .stages.projection import (
    QKVPrepared,
    QProjPrepared,
    prepare_qkv,
    prepare_qproj,
    projection_blocks_per_row,
)
from .stages.rope import (
    DEFAULT_K_NORM_WEIGHT_TENSOR,
    DEFAULT_Q_NORM_WEIGHT_TENSOR,
    HEAD_DIM,
    QKVRopePrepared,
    compile_rms_norm_rope_object,
    compile_rope_table_object,
    prepare_qkv_rope,
)
from .stages.self_attention import (
    DEFAULT_O_PROJ_WEIGHT_TENSOR,
    SelfAttentionPrepared,
    prepare_self_attn,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official-style quantized_qwen3 embed_tokens -> input_layernorm pipeline."
    )
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, required=True)
    parser.add_argument("--embed-chunk-rows", type=int, default=None)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--embed-aie-mlir", type=Path, required=True)
    parser.add_argument("--norm-aie-mlir", type=Path, required=True)
    parser.add_argument("--qproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--qproj-tensor", default=DEFAULT_Q_PROJ_WEIGHT_TENSOR)
    parser.add_argument("--kproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--kproj-tensor", default=projection_weight_tensor("k_proj"))
    parser.add_argument("--vproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--vproj-tensor", default=projection_weight_tensor("v_proj"))
    parser.add_argument("--oproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--oproj-tensor", default=DEFAULT_O_PROJ_WEIGHT_TENSOR)
    parser.add_argument("--rope-table-aie-mlir", type=Path, default=None)
    parser.add_argument("--q-norm-rope-aie-mlir", type=Path, default=None)
    parser.add_argument("--k-norm-rope-aie-mlir", type=Path, default=None)
    parser.add_argument("--attention-aie-mlir", type=Path, nargs="+", default=None)
    parser.add_argument("--q-norm-weight-tensor", default=DEFAULT_Q_NORM_WEIGHT_TENSOR)
    parser.add_argument("--k-norm-weight-tensor", default=DEFAULT_K_NORM_WEIGHT_TENSOR)
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--output-rows", type=int, default=128)
    parser.add_argument("--qproj-output-rows", type=int, default=None)
    parser.add_argument("--kproj-output-rows", type=int, default=None)
    parser.add_argument("--vproj-output-rows", type=int, default=None)
    parser.add_argument("--oproj-output-rows", type=int, default=1024)
    parser.add_argument("--output-tile-rows", type=int, default=32)
    parser.add_argument("--query-tile-rows", type=int, default=4)
    parser.add_argument("--key-tile-rows", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    qproj_aie_mlir: Path | None = args.qproj_aie_mlir
    kproj_aie_mlir: Path | None = args.kproj_aie_mlir
    vproj_aie_mlir: Path | None = args.vproj_aie_mlir
    oproj_aie_mlir: Path | None = args.oproj_aie_mlir
    rope_table_aie_mlir: Path | None = args.rope_table_aie_mlir
    q_norm_rope_aie_mlir: Path | None = args.q_norm_rope_aie_mlir
    k_norm_rope_aie_mlir: Path | None = args.k_norm_rope_aie_mlir
    attention_aie_mlirs: list[Path] | None = args.attention_aie_mlir
    projection_output_rows = {
        "q_proj": args.qproj_output_rows or args.output_rows,
        "k_proj": args.kproj_output_rows or args.output_rows,
        "v_proj": args.vproj_output_rows or args.output_rows,
    }
    embed_chunk_rows = args.embed_chunk_rows or len(args.token_ids)
    if embed_chunk_rows <= 0 or len(args.token_ids) % embed_chunk_rows != 0:
        raise SystemExit("embed_chunk_rows must be positive and divide token count")

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    run_qkv = kproj_aie_mlir is not None or vproj_aie_mlir is not None
    if run_qkv and (qproj_aie_mlir is None or kproj_aie_mlir is None or vproj_aie_mlir is None):
        raise SystemExit(
            "q/k/v pipeline requires --qproj-aie-mlir, --kproj-aie-mlir, and --vproj-aie-mlir"
        )
    run_rope = (
        rope_table_aie_mlir is not None
        or q_norm_rope_aie_mlir is not None
        or k_norm_rope_aie_mlir is not None
    )
    if run_rope and (
        not run_qkv
        or rope_table_aie_mlir is None
        or q_norm_rope_aie_mlir is None
        or k_norm_rope_aie_mlir is None
    ):
        raise SystemExit(
            "q/k norm+RoPE pipeline requires q/k/v projections plus "
            "--rope-table-aie-mlir, --q-norm-rope-aie-mlir, and --k-norm-rope-aie-mlir"
        )
    run_attention = attention_aie_mlirs is not None
    if run_attention and not run_rope:
        raise SystemExit("attention pipeline requires q/k/v projections plus q/k norm+RoPE")
    run_self_attn = oproj_aie_mlir is not None
    if run_self_attn and not run_attention:
        raise SystemExit("self_attn pipeline requires attention_core plus --oproj-aie-mlir")
    if run_attention and (
        args.q_heads <= 0 or args.kv_heads <= 0 or args.q_heads % args.kv_heads != 0
    ):
        raise SystemExit("q_heads must be a positive multiple of kv_heads")
    if (
        run_attention
        and attention_aie_mlirs is not None
        and len(attention_aie_mlirs) != args.q_heads
    ):
        raise SystemExit("--attention-aie-mlir must pass one AIE MLIR per q head")
    if run_attention and len(args.token_ids) % args.query_tile_rows != 0:
        raise SystemExit("attention query tile rows must divide token count")
    if run_attention and len(args.token_ids) % args.key_tile_rows != 0:
        raise SystemExit("attention key tile rows must divide token count")
    if run_attention and args.key_tile_rows > 8:
        raise SystemExit("attention_core currently validates key_tile_rows up to 8")
    qproj_prepared: QProjPrepared | None = None
    projection_prepared: QKVPrepared | None = None
    rope_prepared: QKVRopePrepared | None = None
    attention_prepared: AttentionPrepared | None = None
    self_attn_prepared: SelfAttentionPrepared | None = None
    projection_weights: dict[str, np.ndarray]
    projection_outputs: dict[str, np.ndarray]
    projection_expected: dict[str, torch.Tensor]
    qproj_expected: torch.Tensor | None

    if qproj_aie_mlir is None:
        prepared_base = prepare_embed_norm(
            gguf_path=args.gguf,
            token_ids=args.token_ids,
            blocks_per_row=args.blocks_per_row,
            rms_weight_tensor=args.rms_weight_tensor,
            eps=args.rms_norm_eps,
        )
        projection_weights = {}
        projection_outputs = {}
        projection_expected = {}
        qproj_expected = None
    elif run_qkv:
        assert qproj_aie_mlir is not None
        assert kproj_aie_mlir is not None
        assert vproj_aie_mlir is not None
        if run_attention:
            if run_self_attn:
                self_attn_prepared = prepare_self_attn(
                    gguf_path=args.gguf,
                    token_ids=args.token_ids,
                    blocks_per_row=args.blocks_per_row,
                    rms_weight_tensor=args.rms_weight_tensor,
                    eps=args.rms_norm_eps,
                    projection_tensors={
                        "q_proj": args.qproj_tensor,
                        "k_proj": args.kproj_tensor,
                        "v_proj": args.vproj_tensor,
                    },
                    projection_output_rows=projection_output_rows,
                    q_norm_weight_tensor=args.q_norm_weight_tensor,
                    k_norm_weight_tensor=args.k_norm_weight_tensor,
                    start_position=args.start_position,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                    oproj_tensor=args.oproj_tensor,
                    oproj_output_rows=args.oproj_output_rows,
                )
                attention_prepared = self_attn_prepared.attention
            else:
                attention_prepared = prepare_attention(
                    gguf_path=args.gguf,
                    token_ids=args.token_ids,
                    blocks_per_row=args.blocks_per_row,
                    rms_weight_tensor=args.rms_weight_tensor,
                    eps=args.rms_norm_eps,
                    projection_tensors={
                        "q_proj": args.qproj_tensor,
                        "k_proj": args.kproj_tensor,
                        "v_proj": args.vproj_tensor,
                    },
                    projection_output_rows=projection_output_rows,
                    q_norm_weight_tensor=args.q_norm_weight_tensor,
                    k_norm_weight_tensor=args.k_norm_weight_tensor,
                    start_position=args.start_position,
                    q_heads=args.q_heads,
                    kv_heads=args.kv_heads,
                )
            rope_prepared = attention_prepared.rope
            projection_prepared = rope_prepared.projection
        elif run_rope:
            rope_prepared = prepare_qkv_rope(
                gguf_path=args.gguf,
                token_ids=args.token_ids,
                blocks_per_row=args.blocks_per_row,
                rms_weight_tensor=args.rms_weight_tensor,
                eps=args.rms_norm_eps,
                projection_tensors={
                    "q_proj": args.qproj_tensor,
                    "k_proj": args.kproj_tensor,
                    "v_proj": args.vproj_tensor,
                },
                projection_output_rows=projection_output_rows,
                q_norm_weight_tensor=args.q_norm_weight_tensor,
                k_norm_weight_tensor=args.k_norm_weight_tensor,
                start_position=args.start_position,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
            )
            projection_prepared = rope_prepared.projection
        else:
            projection_prepared = prepare_qkv(
                gguf_path=args.gguf,
                token_ids=args.token_ids,
                blocks_per_row=args.blocks_per_row,
                rms_weight_tensor=args.rms_weight_tensor,
                eps=args.rms_norm_eps,
                projection_tensors={
                    "q_proj": args.qproj_tensor,
                    "k_proj": args.kproj_tensor,
                    "v_proj": args.vproj_tensor,
                },
                projection_output_rows=projection_output_rows,
            )
        prepared_base = projection_prepared.base
        projection_weights = projection_prepared.projection_weights
        projection_outputs = projection_prepared.projection_outputs
        projection_expected = projection_prepared.projection_expected
        qproj_expected = projection_expected["q_proj"]
    else:
        qproj_prepared = prepare_qproj(
            gguf_path=args.gguf,
            token_ids=args.token_ids,
            blocks_per_row=args.blocks_per_row,
            rms_weight_tensor=args.rms_weight_tensor,
            eps=args.rms_norm_eps,
            qproj_tensor=args.qproj_tensor,
            output_rows=args.output_rows,
        )
        prepared_base = qproj_prepared.base
        qproj_expected = qproj_prepared.qproj_expected
        projection_weights = {}
        projection_outputs = {}
        projection_expected = {}

    packed_rows = prepared_base.packed_rows
    block_f16_scales = prepared_base.block_f16_scales
    rms_weight = prepared_base.rms_weight
    hidden = prepared_base.hidden
    output = prepared_base.norm_output
    embed_expected = prepared_base.embed_expected
    expected = prepared_base.norm_expected

    print(f"GGUF tensor {prepared_base.embed_tensor.name} {prepared_base.embed_tensor.ggml_type}")
    print(
        f"RMS weight {prepared_base.rms_weight_tensor.name} {prepared_base.rms_weight_tensor.ggml_type}"
    )
    print(f"token_ids {','.join(str(value) for value in args.token_ids)}")
    print(f"blocks_per_row {args.blocks_per_row} hidden_size {prepared_base.hidden_size}")
    print(f"reference safetensors_pytorch_rocm {torch.cuda.get_device_name(0)}")
    if qproj_aie_mlir is None:
        print("handoff embed_tokens->input_layernorm shared pyxrt BO")
    elif run_qkv:
        assert projection_prepared is not None
        for proj_name, weight_entry in projection_prepared.projection_weight_tensors.items():
            print(f"quantized weight {proj_name} {weight_entry.name} {weight_entry.ggml_type}")
        print(
            "projection_output_rows "
            f"q={projection_output_rows['q_proj']} "
            f"k={projection_output_rows['k_proj']} "
            f"v={projection_output_rows['v_proj']}"
        )
        if self_attn_prepared is not None:
            print(
                f"quantized weight o_proj {self_attn_prepared.oproj_weight_tensor.name} "
                f"{self_attn_prepared.oproj_weight_tensor.ggml_type}"
            )
            print(f"oproj_output_rows {args.oproj_output_rows}")
        if run_rope:
            assert rope_prepared is not None
            print(
                f"q_norm weight {rope_prepared.q_norm_weight_tensor.name} {rope_prepared.q_norm_weight_tensor.ggml_type}"
            )
            print(
                f"k_norm weight {rope_prepared.k_norm_weight_tensor.name} {rope_prepared.k_norm_weight_tensor.ggml_type}"
            )
            print(
                f"attention_heads q={rope_prepared.q_heads} kv={rope_prepared.kv_heads} head_dim {HEAD_DIM}"
            )
            print(
                f"rope_start_position {rope_prepared.rope_start_position} rope_theta {rope_prepared.rope_theta:g}"
            )
            if run_attention:
                if run_self_attn:
                    print(
                        "handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope"
                        "->attention_core->o_proj shared pyxrt BO"
                    )
                else:
                    print(
                        "handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope"
                        "->attention_core shared pyxrt BO"
                    )
            else:
                print(
                    "handoff embed_tokens->input_layernorm->q/k/v->rope_table->q/k_norm_rope shared pyxrt BO"
                )
        else:
            print("handoff embed_tokens->input_layernorm->q/k/v shared pyxrt BO")
    else:
        assert qproj_prepared is not None
        print(
            f"Q4_K weight {qproj_prepared.qproj_weight_tensor.name} {qproj_prepared.qproj_weight_tensor.ggml_type}"
        )
        print(f"output_rows {args.output_rows}")
        print("handoff embed_tokens->input_layernorm->q_proj shared pyxrt BO")

    _, embed_xclbin, embed_insts = compile_runtime(
        aie_mlir=args.embed_aie_mlir,
        work_dir=args.work_dir / "embed_tokens",
        instance_name="run_embed_tokens",
        peano_install_dir=peano_install_dir,
    )
    rms_norm_object = compile_rms_norm_object(
        work_dir=args.work_dir / "input_layernorm",
        peano_install_dir=peano_install_dir,
        hidden_size=hidden.shape[1],
        eps=args.rms_norm_eps,
    )
    _, norm_xclbin, norm_insts = compile_runtime(
        aie_mlir=args.norm_aie_mlir,
        work_dir=args.work_dir / "input_layernorm",
        instance_name="run_input_layernorm",
        peano_install_dir=peano_install_dir,
        link_objects=(rms_norm_object,),
    )
    if qproj_aie_mlir is None:
        actual_hidden, actual_output, latencies_ms = run_on_npu(
            embed_xclbin=embed_xclbin,
            embed_insts=embed_insts,
            norm_xclbin=norm_xclbin,
            norm_insts=norm_insts,
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            rms_weight=rms_weight,
            hidden=hidden,
            output=output,
            embed_expected=embed_expected,
            expected=expected,
            embed_chunk_rows=embed_chunk_rows,
            warmup=args.warmup,
            iterations=args.iterations,
            rtol=args.rtol,
            atol=args.atol,
            verbose=args.verbose,
        )
        actual_qproj = None
        qproj_xclbin = None
        qproj_insts = None
        actual_projections = {}
        projection_xclbins = {}
        projection_insts = {}
        rope_table_xclbin = None
        rope_table_insts = None
        norm_rope_xclbins = {}
        norm_rope_insts = {}
        actual_cos = None
        actual_sin = None
        actual_norm_rope = {}
        attention_xclbins = []
        attention_insts_paths = []
        actual_attention = None
        actual_attention_expected = None
        oproj_xclbin = None
        oproj_insts = None
        actual_oproj = None
        actual_oproj_expected = None
    elif run_qkv:
        assert qproj_aie_mlir is not None
        assert kproj_aie_mlir is not None
        assert vproj_aie_mlir is not None
        assert projection_prepared is not None
        projection_aie_mlirs = {
            "q_proj": qproj_aie_mlir,
            "k_proj": kproj_aie_mlir,
            "v_proj": vproj_aie_mlir,
        }
        projection_xclbins = {}
        projection_insts = {}
        for proj_name, aie_mlir in projection_aie_mlirs.items():
            weight_entry = projection_prepared.projection_weight_tensors[proj_name]
            projection_object = compile_projection_object(
                ggml_type=weight_entry.ggml_type,
                work_dir=args.work_dir / proj_name,
                peano_install_dir=peano_install_dir,
                output_tile_rows=args.output_tile_rows,
                blocks_per_row=projection_blocks_per_row(weight_entry),
                hidden_size=int(weight_entry.logical_shape[1]),
            )
            _, projection_xclbins[proj_name], projection_insts[proj_name] = compile_runtime(
                aie_mlir=aie_mlir,
                work_dir=args.work_dir / proj_name,
                instance_name=f"run_{proj_name}",
                peano_install_dir=peano_install_dir,
                link_objects=(projection_object,),
            )
        if run_rope:
            assert rope_table_aie_mlir is not None
            assert q_norm_rope_aie_mlir is not None
            assert k_norm_rope_aie_mlir is not None
            assert rope_prepared is not None
            rope_table_object = compile_rope_table_object(
                work_dir=args.work_dir / "rope_table",
                peano_install_dir=peano_install_dir,
                head_dim=HEAD_DIM,
                rope_theta=rope_prepared.rope_theta,
            )
            _, rope_table_xclbin, rope_table_insts = compile_runtime(
                aie_mlir=rope_table_aie_mlir,
                work_dir=args.work_dir / "rope_table",
                instance_name="run_rope_table",
                peano_install_dir=peano_install_dir,
                link_objects=(rope_table_object,),
            )
            rms_norm_rope_object = compile_rms_norm_rope_object(
                work_dir=args.work_dir / "rms_norm_rope",
                peano_install_dir=peano_install_dir,
                head_dim=HEAD_DIM,
                eps=args.rms_norm_eps,
            )
            norm_rope_aie_mlirs = {
                "q_norm_rope": q_norm_rope_aie_mlir,
                "k_norm_rope": k_norm_rope_aie_mlir,
            }
            norm_rope_xclbins = {}
            norm_rope_insts = {}
            for stage_name, aie_mlir in norm_rope_aie_mlirs.items():
                _, norm_rope_xclbins[stage_name], norm_rope_insts[stage_name] = compile_runtime(
                    aie_mlir=aie_mlir,
                    work_dir=args.work_dir / stage_name,
                    instance_name=f"run_{stage_name}",
                    peano_install_dir=peano_install_dir,
                    link_objects=(rms_norm_rope_object,),
                )
            if run_attention:
                assert attention_aie_mlirs is not None
                attention_core_object = compile_attention_core_object(
                    work_dir=args.work_dir / "attention_core",
                    peano_install_dir=peano_install_dir,
                    head_dim=HEAD_DIM,
                    sequence_length=len(args.token_ids),
                    query_tile_rows=args.query_tile_rows,
                    key_tile_rows=args.key_tile_rows,
                )
                attention_xclbins = []
                attention_insts_paths = []
                for index, aie_mlir in enumerate(attention_aie_mlirs):
                    runtime_work_dir = args.work_dir / "attention_core" / f"head_{index}"
                    _, attention_xclbin, attention_insts = compile_runtime(
                        aie_mlir=aie_mlir,
                        work_dir=runtime_work_dir,
                        instance_name="run_attention_core",
                        peano_install_dir=peano_install_dir,
                        link_objects=(attention_core_object,),
                    )
                    attention_xclbins.append(attention_xclbin)
                    attention_insts_paths.append(attention_insts)
            else:
                attention_xclbins = []
                attention_insts_paths = []
            if run_self_attn:
                assert oproj_aie_mlir is not None
                assert self_attn_prepared is not None
                oproj_object = compile_projection_object(
                    ggml_type=self_attn_prepared.oproj_weight_tensor.ggml_type,
                    work_dir=args.work_dir / "o_proj",
                    peano_install_dir=peano_install_dir,
                    output_tile_rows=args.output_tile_rows,
                    blocks_per_row=projection_blocks_per_row(
                        self_attn_prepared.oproj_weight_tensor
                    ),
                    hidden_size=int(self_attn_prepared.oproj_weight_tensor.logical_shape[1]),
                )
                _, oproj_xclbin, oproj_insts = compile_runtime(
                    aie_mlir=oproj_aie_mlir,
                    work_dir=args.work_dir / "o_proj",
                    instance_name="run_o_proj",
                    peano_install_dir=peano_install_dir,
                    link_objects=(oproj_object,),
                )
            else:
                oproj_xclbin = None
                oproj_insts = None
            (
                actual_hidden,
                actual_output,
                actual_projections,
                actual_cos,
                actual_sin,
                actual_norm_rope,
                actual_attention,
                actual_attention_expected,
                actual_oproj,
                actual_oproj_expected,
                latencies_ms,
            ) = run_on_npu_projections_rope(
                embed_xclbin=embed_xclbin,
                embed_insts=embed_insts,
                norm_xclbin=norm_xclbin,
                norm_insts=norm_insts,
                projection_xclbins=projection_xclbins,
                projection_insts=projection_insts,
                rope_table_xclbin=rope_table_xclbin,
                rope_table_insts=rope_table_insts,
                norm_rope_xclbins=norm_rope_xclbins,
                norm_rope_insts=norm_rope_insts,
                attention_xclbins=attention_xclbins,
                attention_insts_paths=attention_insts_paths,
                oproj_xclbin=oproj_xclbin,
                oproj_insts=oproj_insts,
                packed_rows=packed_rows,
                block_f16_scales=block_f16_scales,
                rms_weight=rms_weight,
                hidden=hidden,
                norm_output=output,
                projection_weights=projection_weights,
                projection_outputs=projection_outputs,
                output_tile_rows=args.output_tile_rows,
                q_norm_weight=rope_prepared.q_norm_weight,
                k_norm_weight=rope_prepared.k_norm_weight,
                start_position=rope_prepared.start_position,
                cos_output=np.zeros(tuple(rope_prepared.cos_expected.shape), dtype=np.float32),
                sin_output=np.zeros(tuple(rope_prepared.sin_expected.shape), dtype=np.float32),
                norm_rope_outputs=rope_prepared.norm_rope_outputs,
                attention_output=(
                    attention_prepared.attention_output if attention_prepared is not None else None
                ),
                oproj_weights=(
                    self_attn_prepared.oproj_weights if self_attn_prepared is not None else None
                ),
                oproj_output=(
                    self_attn_prepared.oproj_output if self_attn_prepared is not None else None
                ),
                embed_expected=embed_expected,
                norm_expected=expected,
                projection_expected=projection_expected,
                cos_expected=rope_prepared.cos_expected,
                sin_expected=rope_prepared.sin_expected,
                norm_rope_expected=rope_prepared.norm_rope_expected,
                attention_expected=(
                    attention_prepared.attention_expected
                    if attention_prepared is not None
                    else None
                ),
                oproj_expected=(
                    self_attn_prepared.oproj_expected if self_attn_prepared is not None else None
                ),
                embed_chunk_rows=embed_chunk_rows,
                warmup=args.warmup,
                iterations=args.iterations,
                rtol=args.rtol,
                atol=args.atol,
                verbose=args.verbose,
            )
        else:
            actual_hidden, actual_output, actual_projections, latencies_ms = run_on_npu_projections(
                embed_xclbin=embed_xclbin,
                embed_insts=embed_insts,
                norm_xclbin=norm_xclbin,
                norm_insts=norm_insts,
                projection_xclbins=projection_xclbins,
                projection_insts=projection_insts,
                packed_rows=packed_rows,
                block_f16_scales=block_f16_scales,
                rms_weight=rms_weight,
                hidden=hidden,
                norm_output=output,
                projection_weights=projection_weights,
                projection_outputs=projection_outputs,
                output_tile_rows=args.output_tile_rows,
                embed_expected=embed_expected,
                norm_expected=expected,
                projection_expected=projection_expected,
                embed_chunk_rows=embed_chunk_rows,
                warmup=args.warmup,
                iterations=args.iterations,
                rtol=args.rtol,
                atol=args.atol,
                verbose=args.verbose,
            )
            rope_table_xclbin = None
            rope_table_insts = None
            norm_rope_xclbins = {}
            norm_rope_insts = {}
            actual_cos = None
            actual_sin = None
            actual_norm_rope = {}
            attention_xclbins = []
            attention_insts_paths = []
            actual_attention = None
            actual_attention_expected = None
            oproj_xclbin = None
            oproj_insts = None
            actual_oproj = None
            actual_oproj_expected = None
        actual_qproj = actual_projections["q_proj"]
        qproj_xclbin = projection_xclbins["q_proj"]
        qproj_insts = projection_insts["q_proj"]
    else:
        assert qproj_aie_mlir is not None
        assert qproj_prepared is not None
        q4k_object = compile_q4k_linear_object(
            work_dir=args.work_dir / "q_proj",
            peano_install_dir=peano_install_dir,
            output_tile_rows=args.output_tile_rows,
            blocks_per_row=args.blocks_per_row,
            hidden_size=hidden.shape[1],
        )
        _, qproj_xclbin, qproj_insts = compile_runtime(
            aie_mlir=qproj_aie_mlir,
            work_dir=args.work_dir / "q_proj",
            instance_name="run_q_proj",
            peano_install_dir=peano_install_dir,
            link_objects=(q4k_object,),
        )
        assert qproj_expected is not None
        actual_hidden, actual_output, actual_qproj, latencies_ms = run_on_npu_qproj(
            embed_xclbin=embed_xclbin,
            embed_insts=embed_insts,
            norm_xclbin=norm_xclbin,
            norm_insts=norm_insts,
            qproj_xclbin=qproj_xclbin,
            qproj_insts=qproj_insts,
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            rms_weight=rms_weight,
            hidden=hidden,
            norm_output=output,
            qproj_weights=qproj_prepared.qproj_weights,
            qproj_output=qproj_prepared.qproj_output,
            output_tile_rows=args.output_tile_rows,
            embed_expected=embed_expected,
            norm_expected=expected,
            qproj_expected=qproj_expected,
            embed_chunk_rows=embed_chunk_rows,
            warmup=args.warmup,
            iterations=args.iterations,
            rtol=args.rtol,
            atol=args.atol,
            verbose=args.verbose,
        )
        actual_projections = {}
        projection_xclbins = {}
        projection_insts = {}
        rope_table_xclbin = None
        rope_table_insts = None
        norm_rope_xclbins = {}
        norm_rope_insts = {}
        actual_cos = None
        actual_sin = None
        actual_norm_rope = {}
        attention_xclbins = []
        attention_insts_paths = []
        actual_attention = None
        actual_attention_expected = None
        oproj_xclbin = None
        oproj_insts = None
        actual_oproj = None
        actual_oproj_expected = None

    hidden_max_abs = max_abs_rocm(actual_hidden, embed_expected)
    output_max_abs = max_abs_rocm(actual_output, expected)
    print(f"embed_xclbin {embed_xclbin}")
    print(f"embed_insts {embed_insts}")
    print(f"norm_xclbin {norm_xclbin}")
    print(f"norm_insts {norm_insts}")
    if projection_xclbins:
        for proj_name in projection_xclbins:
            print(f"{proj_name}_xclbin {projection_xclbins[proj_name]}")
            print(f"{proj_name}_insts {projection_insts[proj_name]}")
    elif qproj_xclbin is not None and qproj_insts is not None:
        print(f"qproj_xclbin {qproj_xclbin}")
        print(f"qproj_insts {qproj_insts}")
    if rope_table_xclbin is not None and rope_table_insts is not None:
        print(f"rope_table_xclbin {rope_table_xclbin}")
        print(f"rope_table_insts {rope_table_insts}")
        for stage_name in norm_rope_xclbins:
            print(f"{stage_name}_xclbin {norm_rope_xclbins[stage_name]}")
            print(f"{stage_name}_insts {norm_rope_insts[stage_name]}")
    for index, (attention_xclbin, attention_insts) in enumerate(
        zip(attention_xclbins, attention_insts_paths, strict=True)
    ):
        label = "attention_core" if len(attention_xclbins) == 1 else f"attention_core_head{index}"
        print(f"{label}_xclbin {attention_xclbin}")
        print(f"{label}_insts {attention_insts}")
    if oproj_xclbin is not None and oproj_insts is not None:
        print(f"o_proj_xclbin {oproj_xclbin}")
        print(f"o_proj_insts {oproj_insts}")
    print(f"hidden_first8 {actual_hidden.reshape(-1)[:8].tolist()}")
    print(f"output_first8 {actual_output.reshape(-1)[:8].tolist()}")
    if actual_projections:
        for proj_name, actual in actual_projections.items():
            print(f"{proj_name}_first8 {actual.reshape(-1)[:8].tolist()}")
            print(f"{proj_name}_expected_first8 {first_values(projection_expected[proj_name])}")
    elif actual_qproj is not None:
        assert qproj_expected is not None
        print(f"qproj_first8 {actual_qproj.reshape(-1)[:8].tolist()}")
        print(f"qproj_expected_first8 {first_values(qproj_expected)}")
    if actual_cos is not None and actual_sin is not None:
        assert rope_prepared is not None
        print(f"rope_cos_first8 {actual_cos.reshape(-1)[:8].tolist()}")
        print(f"rope_cos_expected_first8 {first_values(rope_prepared.cos_expected)}")
        print(f"rope_sin_first8 {actual_sin.reshape(-1)[:8].tolist()}")
        print(f"rope_sin_expected_first8 {first_values(rope_prepared.sin_expected)}")
        for stage_name, actual in actual_norm_rope.items():
            print(f"{stage_name}_first8 {actual.reshape(-1)[:8].tolist()}")
            print(
                f"{stage_name}_expected_first8 {first_values(rope_prepared.norm_rope_expected[stage_name])}"
            )
    if actual_attention is not None:
        assert actual_attention_expected is not None
        print(f"attention_core_first8 {actual_attention.reshape(-1)[:8].tolist()}")
        print(f"attention_core_expected_first8 {first_values(actual_attention_expected)}")
    if actual_oproj is not None:
        assert actual_oproj_expected is not None
        print(f"o_proj_first8 {actual_oproj.reshape(-1)[:8].tolist()}")
        print(f"o_proj_expected_first8 {first_values(actual_oproj_expected)}")
    print(f"expected_first8 {first_values(expected)}")
    print(f"hidden_max_abs {hidden_max_abs:.8g}")
    print(f"max_abs {output_max_abs:.8g}")
    if actual_projections:
        for proj_name, actual in actual_projections.items():
            projection_max_abs = max_abs_rocm(actual, projection_expected[proj_name])
            print(f"{proj_name}_max_abs {projection_max_abs:.8g}")
    elif actual_qproj is not None:
        assert qproj_expected is not None
        qproj_max_abs = max_abs_rocm(actual_qproj, qproj_expected)
        print(f"qproj_max_abs {qproj_max_abs:.8g}")
    if actual_cos is not None and actual_sin is not None:
        assert rope_prepared is not None
        print(f"rope_cos_max_abs {max_abs_rocm(actual_cos, rope_prepared.cos_expected):.8g}")
        print(f"rope_sin_max_abs {max_abs_rocm(actual_sin, rope_prepared.sin_expected):.8g}")
        for stage_name, actual in actual_norm_rope.items():
            norm_rope_max_abs = max_abs_rocm(actual, rope_prepared.norm_rope_expected[stage_name])
            print(f"{stage_name}_max_abs {norm_rope_max_abs:.8g}")
    if actual_attention is not None:
        assert actual_attention_expected is not None
        attention_max_abs = max_abs_rocm(actual_attention, actual_attention_expected)
        print(f"attention_core_max_abs {attention_max_abs:.8g}")
    if actual_oproj is not None:
        assert actual_oproj_expected is not None
        oproj_max_abs = max_abs_rocm(actual_oproj, actual_oproj_expected)
        print(f"o_proj_max_abs {oproj_max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
