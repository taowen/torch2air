from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyxrt as xrt
import torch

from .air_runtime import compile_runtime, load_xrt_kernel
from .reference_runtime import check_close_rocm, first_values, max_abs_rocm
from .stages.attention import compile_attention_core_object, reference_attention_core
from .stages.rope import HEAD_DIM


@dataclass(frozen=True, slots=True)
class AttentionKernel:
    context: xrt.hw_context
    kernel: xrt.kernel
    instr_v: np.ndarray
    bo_instr: xrt.bo


def deterministic_tensor(*, rows: int, cols: int, seed: int, scale: float) -> torch.Tensor:
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return (
        torch.randn((rows, cols), device=device, generator=generator, dtype=torch.float32) * scale
    )


def run_attention_core_on_npu(
    *,
    xclbins: list[Path],
    insts_paths: list[Path],
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    output: np.ndarray,
    expected: torch.Tensor,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, list[float]]:
    if not xclbins or len(xclbins) != len(insts_paths):
        raise ValueError("xclbins and insts_paths must be non-empty lists with the same length")
    device = xrt.device(0)
    attention_kernels: list[AttentionKernel] = []
    for xclbin, insts in zip(xclbins, insts_paths, strict=True):
        context, kernel, instr_v, bo_instr = load_xrt_kernel(
            device,
            xclbin=xclbin,
            insts=insts,
        )
        attention_kernels.append(
            AttentionKernel(
                context=context,
                kernel=kernel,
                instr_v=instr_v,
                bo_instr=bo_instr,
            )
        )

    first_kernel = attention_kernels[0].kernel
    bo_q = xrt.bo(device, q.nbytes, xrt.bo.host_only, first_kernel.group_id(3))
    bo_k = xrt.bo(device, k.nbytes, xrt.bo.host_only, first_kernel.group_id(4))
    bo_v = xrt.bo(device, v.nbytes, xrt.bo.host_only, first_kernel.group_id(5))
    bo_output = xrt.bo(device, output.nbytes, xrt.bo.host_only, first_kernel.group_id(6))
    for bo, array in (
        (bo_q, q),
        (bo_k, k),
        (bo_v, v),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    actual = output
    latencies_ms: list[float] = []
    for iteration in range(warmup + iterations):
        output.fill(0)
        bo_output.write(output, 0)
        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        start = time.perf_counter()
        for attention_kernel in attention_kernels:
            run = attention_kernel.kernel(
                3,
                attention_kernel.bo_instr,
                len(attention_kernel.instr_v),
                bo_q,
                bo_k,
                bo_v,
                bo_output,
            )
            run.wait()
        if iteration >= warmup:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        actual = bo_output.read(output.nbytes, 0).view(output.dtype).reshape(tuple(output.shape))
        check_close_rocm(actual, expected, rtol=rtol, atol=atol, label="attention_core")

    return actual, latencies_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run standalone quantized_qwen3 attention core on NPU."
    )
    parser.add_argument("--aie-mlir", type=Path, nargs="+", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM)
    parser.add_argument("--query-tile-rows", type=int, default=4)
    parser.add_argument("--key-tile-rows", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if args.head_dim != HEAD_DIM:
        raise SystemExit(f"attention_core currently targets head_dim={HEAD_DIM}")
    if args.sequence_length % args.query_tile_rows != 0:
        raise SystemExit("attention query tile rows must divide sequence length")
    if args.sequence_length % args.key_tile_rows != 0:
        raise SystemExit("attention key tile rows must divide sequence length")
    if args.key_tile_rows > 8:
        raise SystemExit("attention_core currently validates key_tile_rows up to 8")
    if args.q_heads <= 0 or args.kv_heads <= 0 or args.q_heads % args.kv_heads != 0:
        raise SystemExit("q_heads must be a positive multiple of kv_heads")
    if len(args.aie_mlir) != args.q_heads:
        raise SystemExit("--aie-mlir must pass one AIE MLIR per q head")
    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    q_ref = deterministic_tensor(
        rows=args.sequence_length,
        cols=args.q_heads * args.head_dim,
        seed=args.seed,
        scale=args.scale,
    )
    k_ref = deterministic_tensor(
        rows=args.sequence_length,
        cols=args.kv_heads * args.head_dim,
        seed=args.seed + 1,
        scale=args.scale,
    )
    v_ref = deterministic_tensor(
        rows=args.sequence_length,
        cols=args.kv_heads * args.head_dim,
        seed=args.seed + 2,
        scale=0.25,
    )
    expected = reference_attention_core(
        q=q_ref, k=k_ref, v=v_ref, q_heads=args.q_heads, kv_heads=args.kv_heads
    )
    q = np.ascontiguousarray(q_ref.detach().cpu().numpy())
    k = np.ascontiguousarray(k_ref.detach().cpu().numpy())
    v = np.ascontiguousarray(v_ref.detach().cpu().numpy())
    output = np.zeros(tuple(expected.shape), dtype=np.float32)

    object_path = compile_attention_core_object(
        work_dir=args.work_dir,
        peano_install_dir=peano_install_dir,
        head_dim=args.head_dim,
        sequence_length=args.sequence_length,
        query_tile_rows=args.query_tile_rows,
        key_tile_rows=args.key_tile_rows,
    )
    xclbins: list[Path] = []
    insts_paths: list[Path] = []
    for index, aie_mlir in enumerate(args.aie_mlir):
        runtime_work_dir = (
            args.work_dir if len(args.aie_mlir) == 1 else args.work_dir / f"head_{index}"
        )
        _, xclbin, insts = compile_runtime(
            aie_mlir=aie_mlir,
            work_dir=runtime_work_dir,
            instance_name="run_attention_core",
            peano_install_dir=peano_install_dir,
            link_objects=(object_path,),
        )
        xclbins.append(xclbin)
        insts_paths.append(insts)
    actual, latencies_ms = run_attention_core_on_npu(
        xclbins=xclbins,
        insts_paths=insts_paths,
        q=q,
        k=k,
        v=v,
        output=output,
        expected=expected,
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
    )

    print(f"reference pytorch_rocm {torch.cuda.get_device_name(0)}")
    print(
        f"sequence_length {args.sequence_length} head_dim {args.head_dim} "
        f"q_heads {args.q_heads} kv_heads {args.kv_heads} "
        f"query_tile_rows {args.query_tile_rows} key_tile_rows {args.key_tile_rows}"
    )
    for index, (xclbin, insts) in enumerate(zip(xclbins, insts_paths, strict=True)):
        label = "attention_core" if len(xclbins) == 1 else f"attention_core_head{index}"
        print(f"{label}_xclbin {xclbin}")
        print(f"{label}_insts {insts}")
    print(f"attention_core_first8 {actual.reshape(-1)[:8].tolist()}")
    print(f"attention_core_expected_first8 {first_values(expected)}")
    print(f"attention_core_max_abs {max_abs_rocm(actual, expected):.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
