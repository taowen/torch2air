from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyxrt as xrt

from torch2air.weights.gguf import load_gguf_index, read_tensor_bytes

from .run_embed_tokens import (
    DEFAULT_GGUF,
    _check_close,
    compile_runtime,
    parse_token_ids,
    prepare_inputs,
)
from .run_embed_tokens_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR


def prepare_pipeline_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    packed_rows, block_f16_scales, embed_expected, info = prepare_inputs(
        gguf_path=gguf_path,
        tensor_name="model.embed_tokens.weight",
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
    )
    hidden_size = blocks_per_row * 256
    index = load_gguf_index(gguf_path)
    weight_entry = index.tensors[rms_weight_tensor]
    if weight_entry.ggml_type != "F32" or weight_entry.physical_dtype != "float32":
        raise ValueError(f"{rms_weight_tensor} must be F32, got {weight_entry}")
    if int(weight_entry.physical_shape[0]) < hidden_size:
        raise ValueError(f"{rms_weight_tensor} is too small for hidden_size={hidden_size}")

    payload = read_tensor_bytes(index.path, weight_entry, offset=0, size=hidden_size * 4)
    rms_weight = np.frombuffer(payload, dtype=np.float32).copy()

    variance = np.mean(embed_expected.astype(np.float32) ** 2, axis=-1, keepdims=True)
    expected = embed_expected * (1.0 / np.sqrt(variance + eps)).astype(np.float32)
    expected = expected * rms_weight.reshape(1, hidden_size)
    hidden = np.zeros_like(embed_expected, dtype=np.float32)
    output = np.zeros_like(expected, dtype=np.float32)

    info["rms_weight"] = weight_entry.to_json()
    info["rms_norm_eps"] = eps
    return (
        packed_rows,
        block_f16_scales,
        np.ascontiguousarray(rms_weight),
        np.ascontiguousarray(hidden),
        np.ascontiguousarray(output),
        np.ascontiguousarray(embed_expected.astype(np.float32, copy=False)),
        np.ascontiguousarray(expected.astype(np.float32, copy=False)),
        info,
    )


def load_xrt_kernel(device, *, xclbin: Path, insts: Path):
    loaded_xclbin = xrt.xclbin(str(xclbin))
    device.register_xclbin(loaded_xclbin)
    context = xrt.hw_context(device, loaded_xclbin.get_uuid())
    kernel_name = [
        kernel.get_name()
        for kernel in loaded_xclbin.get_kernels()
        if "MLIR_AIE" in kernel.get_name()
    ][0]
    kernel = xrt.kernel(context, kernel_name)
    instr_v = np.frombuffer(insts.read_bytes(), dtype=np.uint32)
    bo_instr = xrt.bo(
        device,
        len(instr_v) * 4,
        xrt.bo.cacheable,
        kernel.group_id(1),
    )
    bo_instr.write(instr_v, 0)
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return context, kernel, instr_v, bo_instr


def run_on_npu(
    *,
    embed_xclbin: Path,
    embed_insts: Path,
    norm_xclbin: Path,
    norm_insts: Path,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    rms_weight: np.ndarray,
    hidden: np.ndarray,
    output: np.ndarray,
    embed_expected: np.ndarray,
    expected: np.ndarray,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if verbose:
        print("pyxrt", xrt.__file__)
    device = xrt.device(0)
    embed_context, embed_kernel, embed_instr_v, embed_bo_instr = load_xrt_kernel(
        device,
        xclbin=embed_xclbin,
        insts=embed_insts,
    )
    norm_context, norm_kernel, norm_instr_v, norm_bo_instr = load_xrt_kernel(
        device,
        xclbin=norm_xclbin,
        insts=norm_insts,
    )

    bo_packed = xrt.bo(
        device,
        packed_rows.nbytes,
        xrt.bo.host_only,
        embed_kernel.group_id(3),
    )
    bo_scales = xrt.bo(
        device,
        block_f16_scales.nbytes,
        xrt.bo.host_only,
        embed_kernel.group_id(4),
    )
    bo_hidden = xrt.bo(
        device,
        hidden.nbytes,
        xrt.bo.host_only,
        embed_kernel.group_id(5),
    )
    bo_weight = xrt.bo(
        device,
        rms_weight.nbytes,
        xrt.bo.host_only,
        norm_kernel.group_id(4),
    )
    bo_output = xrt.bo(
        device,
        output.nbytes,
        xrt.bo.host_only,
        norm_kernel.group_id(5),
    )

    bo_packed.write(packed_rows, 0)
    bo_packed.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_scales.write(block_f16_scales, 0)
    bo_scales.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_weight.write(rms_weight, 0)
    bo_weight.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    actual_hidden = hidden
    actual_output = output
    latencies_ms: list[float] = []
    for _ in range(warmup):
        hidden.fill(0)
        output.fill(0)
        actual_hidden, actual_output = run_shared_bo_once(
            embed_kernel=embed_kernel,
            embed_instr_v=embed_instr_v,
            embed_bo_instr=embed_bo_instr,
            norm_kernel=norm_kernel,
            norm_instr_v=norm_instr_v,
            norm_bo_instr=norm_bo_instr,
            bo_packed=bo_packed,
            bo_scales=bo_scales,
            bo_hidden=bo_hidden,
            bo_weight=bo_weight,
            bo_output=bo_output,
            hidden=hidden,
            output=output,
            embed_expected=embed_expected,
            expected=expected,
        )
        _check_close(actual_hidden, embed_expected, rtol=1e-5, atol=1e-6)
        _check_close(actual_output, expected, rtol=rtol, atol=atol)

    for _ in range(iterations):
        hidden.fill(0)
        output.fill(0)
        start = time.perf_counter()
        actual_hidden, actual_output = run_shared_bo_once(
            embed_kernel=embed_kernel,
            embed_instr_v=embed_instr_v,
            embed_bo_instr=embed_bo_instr,
            norm_kernel=norm_kernel,
            norm_instr_v=norm_instr_v,
            norm_bo_instr=norm_bo_instr,
            bo_packed=bo_packed,
            bo_scales=bo_scales,
            bo_hidden=bo_hidden,
            bo_weight=bo_weight,
            bo_output=bo_output,
            hidden=hidden,
            output=output,
            embed_expected=embed_expected,
            expected=expected,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        _check_close(actual_hidden, embed_expected, rtol=1e-5, atol=1e-6)
        _check_close(actual_output, expected, rtol=rtol, atol=atol)

    _ = embed_context
    _ = norm_context
    return actual_hidden, actual_output, latencies_ms


def run_shared_bo_once(
    *,
    embed_kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr,
    norm_kernel,
    norm_instr_v: np.ndarray,
    norm_bo_instr,
    bo_packed,
    bo_scales,
    bo_hidden,
    bo_weight,
    bo_output,
    hidden: np.ndarray,
    output: np.ndarray,
    embed_expected: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bo_hidden.write(hidden, 0)
    bo_hidden.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_output.write(output, 0)
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    embed_run = embed_kernel(3, embed_bo_instr, len(embed_instr_v), bo_packed, bo_scales, bo_hidden)
    embed_run.wait()
    norm_run = norm_kernel(3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_output)
    norm_run.wait()

    bo_hidden.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(embed_expected.shape)
    actual_output = bo_output.read(output.nbytes, 0).view(output.dtype).reshape(expected.shape)
    return actual_hidden, actual_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official-style quantized_qwen3 embed_tokens -> input_layernorm pipeline."
    )
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--token-ids", type=parse_token_ids, default=parse_token_ids("0"))
    parser.add_argument("--blocks-per-row", type=int, required=True)
    parser.add_argument("--rms-weight-tensor", default=DEFAULT_RMS_WEIGHT_TENSOR)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--embed-aie-mlir", type=Path, required=True)
    parser.add_argument("--norm-aie-mlir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    packed_rows, block_f16_scales, rms_weight, hidden, output, embed_expected, expected, info = (
        prepare_pipeline_inputs(
            gguf_path=args.gguf,
            token_ids=args.token_ids,
            blocks_per_row=args.blocks_per_row,
            rms_weight_tensor=args.rms_weight_tensor,
            eps=args.rms_norm_eps,
        )
    )

    print(f"GGUF tensor {info['tensor']['name']} {info['tensor']['ggml_type']}")
    print(f"RMS weight {info['rms_weight']['name']} {info['rms_weight']['ggml_type']}")
    print(f"token_ids {','.join(str(value) for value in args.token_ids)}")
    print(f"blocks_per_row {args.blocks_per_row} hidden_size {info['hidden_size']}")
    print("handoff embed_tokens->input_layernorm shared pyxrt BO")

    _, embed_xclbin, embed_insts = compile_runtime(
        aie_mlir=args.embed_aie_mlir,
        work_dir=args.work_dir / "embed_tokens",
        instance_name="run_embed_tokens",
        peano_install_dir=peano_install_dir,
    )
    _, norm_xclbin, norm_insts = compile_runtime(
        aie_mlir=args.norm_aie_mlir,
        work_dir=args.work_dir / "input_layernorm",
        instance_name="run_input_layernorm",
        peano_install_dir=peano_install_dir,
    )
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
        warmup=args.warmup,
        iterations=args.iterations,
        rtol=args.rtol,
        atol=args.atol,
        verbose=args.verbose,
    )

    hidden_max_abs = float(np.max(np.abs(actual_hidden - embed_expected)))
    output_max_abs = float(np.max(np.abs(actual_output - expected)))
    print(f"embed_xclbin {embed_xclbin}")
    print(f"embed_insts {embed_insts}")
    print(f"norm_xclbin {norm_xclbin}")
    print(f"norm_insts {norm_insts}")
    print(f"hidden_first8 {actual_hidden.reshape(-1)[:8].tolist()}")
    print(f"output_first8 {actual_output.reshape(-1)[:8].tolist()}")
    print(f"expected_first8 {expected.reshape(-1)[:8].tolist()}")
    print(f"hidden_max_abs {hidden_max_abs:.8g}")
    print(f"max_abs {output_max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
