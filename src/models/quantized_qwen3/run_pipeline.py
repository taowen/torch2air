from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyxrt as xrt
import torch

from torch2air.weights.gguf import load_gguf_index, read_tensor_bytes

from . import reference
from .reference_runtime import (
    check_close_rocm,
    first_values,
    max_abs_rocm,
)
from .run_embed_tokens import (
    DEFAULT_GGUF,
    compile_runtime,
    parse_token_ids,
    prepare_inputs,
)
from .run_embed_tokens_input_layernorm import DEFAULT_RMS_WEIGHT_TENSOR
from .run_q_proj import (
    ATTENTION_PROJ_NAMES,
    DEFAULT_Q_PROJ_WEIGHT_TENSOR,
    compile_q4k_linear_object,
    compile_projection_object,
    prepare_q_proj_weights,
    prepare_projection_weights,
    projection_weight_tensor,
    reference_projection,
)


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
    torch.Tensor,
    torch.Tensor,
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

    expected = reference.run_input_layernorm(hidden_states=embed_expected)["mul_1"].reshape(
        len(token_ids),
        hidden_size,
    )
    hidden = np.zeros(tuple(embed_expected.shape), dtype=np.float32)
    output = np.zeros(tuple(expected.shape), dtype=np.float32)

    info["rms_weight"] = weight_entry.to_json()
    info["rms_norm_eps"] = eps
    return (
        packed_rows,
        block_f16_scales,
        np.ascontiguousarray(rms_weight),
        np.ascontiguousarray(hidden),
        np.ascontiguousarray(output),
        embed_expected,
        expected,
        info,
    )


def prepare_pipeline_qproj_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    qproj_tensor: str,
    output_rows: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
]:
    (
        packed_rows,
        block_f16_scales,
        rms_weight,
        hidden,
        norm_output,
        embed_expected,
        norm_expected,
        info,
    ) = prepare_pipeline_inputs(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    packed_qproj, qproj_info = prepare_q_proj_weights(
        gguf_path=gguf_path,
        tensor_name=qproj_tensor,
        output_rows=output_rows,
        hidden_size=norm_expected.shape[1],
    )
    with torch.no_grad():
        qproj_expected = reference.run_q_proj(input=norm_expected)["linear"].reshape(
            len(token_ids),
            -1,
        )[:, :output_rows]
    qproj_output = np.zeros(tuple(qproj_expected.shape), dtype=np.float32)
    info["q_proj_weight"] = qproj_info["tensor"]
    info["q_proj_output_rows"] = output_rows
    return (
        packed_rows,
        block_f16_scales,
        rms_weight,
        hidden,
        norm_output,
        packed_qproj,
        qproj_output,
        embed_expected,
        norm_expected,
        qproj_expected,
        info,
    )


def prepare_pipeline_projection_inputs(
    *,
    gguf_path: Path,
    token_ids: list[int],
    blocks_per_row: int,
    rms_weight_tensor: str,
    eps: float,
    projection_tensors: dict[str, str],
    output_rows: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    (
        packed_rows,
        block_f16_scales,
        rms_weight,
        hidden,
        norm_output,
        embed_expected,
        norm_expected,
        info,
    ) = prepare_pipeline_inputs(
        gguf_path=gguf_path,
        token_ids=token_ids,
        blocks_per_row=blocks_per_row,
        rms_weight_tensor=rms_weight_tensor,
        eps=eps,
    )
    projection_weights: dict[str, np.ndarray] = {}
    projection_outputs: dict[str, np.ndarray] = {}
    projection_expected: dict[str, torch.Tensor] = {}
    projection_weight_info: dict[str, object] = {}
    for proj_name, tensor_name in projection_tensors.items():
        if proj_name not in ATTENTION_PROJ_NAMES:
            raise ValueError(f"Unsupported projection {proj_name!r}")
        packed_projection, projection_info = prepare_projection_weights(
            gguf_path=gguf_path,
            tensor_name=tensor_name,
            output_rows=output_rows,
            hidden_size=norm_expected.shape[1],
        )
        with torch.no_grad():
            expected = reference_projection(proj_name, norm_expected).reshape(
                len(token_ids),
                -1,
            )[:, :output_rows]
        projection_weights[proj_name] = packed_projection
        projection_outputs[proj_name] = np.zeros(tuple(expected.shape), dtype=np.float32)
        projection_expected[proj_name] = expected
        projection_weight_info[proj_name] = projection_info["tensor"]

    info["projection_weights"] = projection_weight_info
    info["projection_output_rows"] = output_rows
    return (
        packed_rows,
        block_f16_scales,
        rms_weight,
        hidden,
        norm_output,
        projection_weights,
        projection_outputs,
        embed_expected,
        norm_expected,
        projection_expected,
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
    embed_expected: torch.Tensor,
    expected: torch.Tensor,
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
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_output, expected, rtol=rtol, atol=atol)

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
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_output, expected, rtol=rtol, atol=atol)

    _ = embed_context
    _ = norm_context
    return actual_hidden, actual_output, latencies_ms


def run_on_npu_qproj(
    *,
    embed_xclbin: Path,
    embed_insts: Path,
    norm_xclbin: Path,
    norm_insts: Path,
    qproj_xclbin: Path,
    qproj_insts: Path,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    rms_weight: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    qproj_weights: np.ndarray,
    qproj_output: np.ndarray,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    qproj_expected: torch.Tensor,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    if verbose:
        print("pyxrt", xrt.__file__)
    device = xrt.device(0)
    _, embed_kernel, embed_instr_v, embed_bo_instr = load_xrt_kernel(
        device,
        xclbin=embed_xclbin,
        insts=embed_insts,
    )
    _, norm_kernel, norm_instr_v, norm_bo_instr = load_xrt_kernel(
        device,
        xclbin=norm_xclbin,
        insts=norm_insts,
    )
    _, qproj_kernel, qproj_instr_v, qproj_bo_instr = load_xrt_kernel(
        device,
        xclbin=qproj_xclbin,
        insts=qproj_insts,
    )

    bo_packed = xrt.bo(device, packed_rows.nbytes, xrt.bo.host_only, embed_kernel.group_id(3))
    bo_scales = xrt.bo(
        device,
        block_f16_scales.nbytes,
        xrt.bo.host_only,
        embed_kernel.group_id(4),
    )
    bo_hidden = xrt.bo(device, hidden.nbytes, xrt.bo.host_only, embed_kernel.group_id(5))
    bo_weight = xrt.bo(device, rms_weight.nbytes, xrt.bo.host_only, norm_kernel.group_id(4))
    bo_norm_output = xrt.bo(
        device,
        norm_output.nbytes,
        xrt.bo.host_only,
        norm_kernel.group_id(5),
    )
    bo_qproj_weights = xrt.bo(
        device,
        qproj_weights.nbytes,
        xrt.bo.host_only,
        qproj_kernel.group_id(4),
    )
    bo_qproj_output = xrt.bo(
        device,
        qproj_output.nbytes,
        xrt.bo.host_only,
        qproj_kernel.group_id(5),
    )

    for bo, array in (
        (bo_packed, packed_rows),
        (bo_scales, block_f16_scales),
        (bo_weight, rms_weight),
        (bo_qproj_weights, qproj_weights),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    actual_hidden = hidden
    actual_norm = norm_output
    actual_qproj = qproj_output
    latencies_ms: list[float] = []
    for iteration in range(warmup + iterations):
        hidden.fill(0)
        norm_output.fill(0)
        qproj_output.fill(0)
        start = time.perf_counter()
        actual_hidden, actual_norm, actual_qproj = run_shared_bo_qproj_once(
            embed_kernel=embed_kernel,
            embed_instr_v=embed_instr_v,
            embed_bo_instr=embed_bo_instr,
            norm_kernel=norm_kernel,
            norm_instr_v=norm_instr_v,
            norm_bo_instr=norm_bo_instr,
            qproj_kernel=qproj_kernel,
            qproj_instr_v=qproj_instr_v,
            qproj_bo_instr=qproj_bo_instr,
            bo_packed=bo_packed,
            bo_scales=bo_scales,
            bo_hidden=bo_hidden,
            bo_weight=bo_weight,
            bo_norm_output=bo_norm_output,
            bo_qproj_weights=bo_qproj_weights,
            bo_qproj_output=bo_qproj_output,
            hidden=hidden,
            norm_output=norm_output,
            qproj_output=qproj_output,
            embed_expected=embed_expected,
            norm_expected=norm_expected,
            qproj_expected=qproj_expected,
        )
        if iteration >= warmup:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_norm, norm_expected, rtol=rtol, atol=atol)
        check_close_rocm(actual_qproj, qproj_expected, rtol=rtol, atol=atol)

    return actual_hidden, actual_norm, actual_qproj, latencies_ms


def run_shared_bo_qproj_once(
    *,
    embed_kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr,
    norm_kernel,
    norm_instr_v: np.ndarray,
    norm_bo_instr,
    qproj_kernel,
    qproj_instr_v: np.ndarray,
    qproj_bo_instr,
    bo_packed,
    bo_scales,
    bo_hidden,
    bo_weight,
    bo_norm_output,
    bo_qproj_weights,
    bo_qproj_output,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    qproj_output: np.ndarray,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    qproj_expected: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for bo, array in (
        (bo_hidden, hidden),
        (bo_norm_output, norm_output),
        (bo_qproj_output, qproj_output),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    embed_run = embed_kernel(3, embed_bo_instr, len(embed_instr_v), bo_packed, bo_scales, bo_hidden)
    embed_run.wait()
    norm_run = norm_kernel(3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_norm_output)
    norm_run.wait()
    qproj_run = qproj_kernel(
        3,
        qproj_bo_instr,
        len(qproj_instr_v),
        bo_norm_output,
        bo_qproj_weights,
        bo_qproj_output,
    )
    qproj_run.wait()

    for bo in (bo_hidden, bo_norm_output, bo_qproj_output):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    actual_norm = (
        bo_norm_output.read(norm_output.nbytes, 0)
        .view(norm_output.dtype)
        .reshape(tuple(norm_expected.shape))
    )
    actual_qproj = (
        bo_qproj_output.read(qproj_output.nbytes, 0)
        .view(qproj_output.dtype)
        .reshape(tuple(qproj_expected.shape))
    )
    return actual_hidden, actual_norm, actual_qproj


def run_on_npu_projections(
    *,
    embed_xclbin: Path,
    embed_insts: Path,
    norm_xclbin: Path,
    norm_insts: Path,
    projection_xclbins: dict[str, Path],
    projection_insts: dict[str, Path],
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    rms_weight: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    projection_weights: dict[str, np.ndarray],
    projection_outputs: dict[str, np.ndarray],
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    projection_expected: dict[str, torch.Tensor],
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[float]]:
    if verbose:
        print("pyxrt", xrt.__file__)
    device = xrt.device(0)
    _, embed_kernel, embed_instr_v, embed_bo_instr = load_xrt_kernel(
        device,
        xclbin=embed_xclbin,
        insts=embed_insts,
    )
    _, norm_kernel, norm_instr_v, norm_bo_instr = load_xrt_kernel(
        device,
        xclbin=norm_xclbin,
        insts=norm_insts,
    )
    projection_kernels = {}
    for proj_name, xclbin in projection_xclbins.items():
        _, kernel, instr_v, bo_instr = load_xrt_kernel(
            device,
            xclbin=xclbin,
            insts=projection_insts[proj_name],
        )
        projection_kernels[proj_name] = (kernel, instr_v, bo_instr)

    bo_packed = xrt.bo(device, packed_rows.nbytes, xrt.bo.host_only, embed_kernel.group_id(3))
    bo_scales = xrt.bo(
        device,
        block_f16_scales.nbytes,
        xrt.bo.host_only,
        embed_kernel.group_id(4),
    )
    bo_hidden = xrt.bo(device, hidden.nbytes, xrt.bo.host_only, embed_kernel.group_id(5))
    bo_weight = xrt.bo(device, rms_weight.nbytes, xrt.bo.host_only, norm_kernel.group_id(4))
    bo_norm_output = xrt.bo(
        device,
        norm_output.nbytes,
        xrt.bo.host_only,
        norm_kernel.group_id(5),
    )
    bo_projection_weights = {}
    bo_projection_outputs = {}
    for proj_name, (kernel, _, _) in projection_kernels.items():
        weights = projection_weights[proj_name]
        output = projection_outputs[proj_name]
        bo_projection_weights[proj_name] = xrt.bo(
            device,
            weights.nbytes,
            xrt.bo.host_only,
            kernel.group_id(4),
        )
        bo_projection_outputs[proj_name] = xrt.bo(
            device,
            output.nbytes,
            xrt.bo.host_only,
            kernel.group_id(5),
        )

    for bo, array in (
        (bo_packed, packed_rows),
        (bo_scales, block_f16_scales),
        (bo_weight, rms_weight),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for proj_name, bo in bo_projection_weights.items():
        bo.write(projection_weights[proj_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    actual_hidden = hidden
    actual_norm = norm_output
    actual_projections = projection_outputs
    latencies_ms: list[float] = []
    for iteration in range(warmup + iterations):
        hidden.fill(0)
        norm_output.fill(0)
        for output in projection_outputs.values():
            output.fill(0)
        start = time.perf_counter()
        actual_hidden, actual_norm, actual_projections = run_shared_bo_projections_once(
            embed_kernel=embed_kernel,
            embed_instr_v=embed_instr_v,
            embed_bo_instr=embed_bo_instr,
            norm_kernel=norm_kernel,
            norm_instr_v=norm_instr_v,
            norm_bo_instr=norm_bo_instr,
            projection_kernels=projection_kernels,
            bo_packed=bo_packed,
            bo_scales=bo_scales,
            bo_hidden=bo_hidden,
            bo_weight=bo_weight,
            bo_norm_output=bo_norm_output,
            bo_projection_weights=bo_projection_weights,
            bo_projection_outputs=bo_projection_outputs,
            hidden=hidden,
            norm_output=norm_output,
            projection_outputs=projection_outputs,
            embed_expected=embed_expected,
            norm_expected=norm_expected,
            projection_expected=projection_expected,
        )
        if iteration >= warmup:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_norm, norm_expected, rtol=rtol, atol=atol)
        for proj_name, actual in actual_projections.items():
            check_close_rocm(actual, projection_expected[proj_name], rtol=rtol, atol=atol, label=proj_name)

    return actual_hidden, actual_norm, actual_projections, latencies_ms


def run_shared_bo_projections_once(
    *,
    embed_kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr,
    norm_kernel,
    norm_instr_v: np.ndarray,
    norm_bo_instr,
    projection_kernels: dict[str, tuple[object, np.ndarray, object]],
    bo_packed,
    bo_scales,
    bo_hidden,
    bo_weight,
    bo_norm_output,
    bo_projection_weights: dict[str, object],
    bo_projection_outputs: dict[str, object],
    hidden: np.ndarray,
    norm_output: np.ndarray,
    projection_outputs: dict[str, np.ndarray],
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    projection_expected: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    for bo, array in (
        (bo_hidden, hidden),
        (bo_norm_output, norm_output),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for proj_name, bo in bo_projection_outputs.items():
        bo.write(projection_outputs[proj_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    embed_run = embed_kernel(3, embed_bo_instr, len(embed_instr_v), bo_packed, bo_scales, bo_hidden)
    embed_run.wait()
    norm_run = norm_kernel(3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_norm_output)
    norm_run.wait()
    for proj_name, (kernel, instr_v, bo_instr) in projection_kernels.items():
        projection_run = kernel(
            3,
            bo_instr,
            len(instr_v),
            bo_norm_output,
            bo_projection_weights[proj_name],
            bo_projection_outputs[proj_name],
        )
        projection_run.wait()

    for bo in (bo_hidden, bo_norm_output):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    for bo in bo_projection_outputs.values():
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    actual_norm = (
        bo_norm_output.read(norm_output.nbytes, 0)
        .view(norm_output.dtype)
        .reshape(tuple(norm_expected.shape))
    )
    actual_projections = {}
    for proj_name, bo in bo_projection_outputs.items():
        output = projection_outputs[proj_name]
        actual_projections[proj_name] = (
            bo.read(output.nbytes, 0)
            .view(output.dtype)
            .reshape(tuple(projection_expected[proj_name].shape))
        )
    return actual_hidden, actual_norm, actual_projections


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
    embed_expected: torch.Tensor,
    expected: torch.Tensor,
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
    actual_hidden = bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    actual_output = bo_output.read(output.nbytes, 0).view(output.dtype).reshape(tuple(expected.shape))
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
    parser.add_argument("--qproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--qproj-tensor", default=DEFAULT_Q_PROJ_WEIGHT_TENSOR)
    parser.add_argument("--kproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--kproj-tensor", default=projection_weight_tensor("k_proj"))
    parser.add_argument("--vproj-aie-mlir", type=Path, default=None)
    parser.add_argument("--vproj-tensor", default=projection_weight_tensor("v_proj"))
    parser.add_argument("--output-rows", type=int, default=128)
    parser.add_argument("--output-tile-rows", type=int, default=32)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    peano_install_dir = os.environ.get("PEANO_INSTALL_DIR")
    if not peano_install_dir:
        raise SystemExit("PEANO_INSTALL_DIR is not set; source scripts/npu-common.sh first")
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")

    run_qkv = args.kproj_aie_mlir is not None or args.vproj_aie_mlir is not None
    if run_qkv and (args.qproj_aie_mlir is None or args.kproj_aie_mlir is None or args.vproj_aie_mlir is None):
        raise SystemExit("q/k/v pipeline requires --qproj-aie-mlir, --kproj-aie-mlir, and --vproj-aie-mlir")

    if args.qproj_aie_mlir is None:
        packed_rows, block_f16_scales, rms_weight, hidden, output, embed_expected, expected, info = (
            prepare_pipeline_inputs(
                gguf_path=args.gguf,
                token_ids=args.token_ids,
                blocks_per_row=args.blocks_per_row,
                rms_weight_tensor=args.rms_weight_tensor,
                eps=args.rms_norm_eps,
            )
        )
        projection_weights = {}
        projection_outputs = {}
        projection_expected = {}
    elif run_qkv:
        (
            packed_rows,
            block_f16_scales,
            rms_weight,
            hidden,
            output,
            projection_weights,
            projection_outputs,
            embed_expected,
            expected,
            projection_expected,
            info,
        ) = prepare_pipeline_projection_inputs(
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
            output_rows=args.output_rows,
        )
        qproj_expected = projection_expected["q_proj"]
    else:
        (
            packed_rows,
            block_f16_scales,
            rms_weight,
            hidden,
            output,
            qproj_weights,
            qproj_output,
            embed_expected,
            expected,
            qproj_expected,
            info,
        ) = prepare_pipeline_qproj_inputs(
            gguf_path=args.gguf,
            token_ids=args.token_ids,
            blocks_per_row=args.blocks_per_row,
            rms_weight_tensor=args.rms_weight_tensor,
            eps=args.rms_norm_eps,
            qproj_tensor=args.qproj_tensor,
            output_rows=args.output_rows,
        )
        projection_weights = {}
        projection_outputs = {}
        projection_expected = {}

    print(f"GGUF tensor {info['tensor']['name']} {info['tensor']['ggml_type']}")
    print(f"RMS weight {info['rms_weight']['name']} {info['rms_weight']['ggml_type']}")
    print(f"token_ids {','.join(str(value) for value in args.token_ids)}")
    print(f"blocks_per_row {args.blocks_per_row} hidden_size {info['hidden_size']}")
    print(f"reference safetensors_pytorch_rocm {torch.cuda.get_device_name(0)}")
    if args.qproj_aie_mlir is None:
        print("handoff embed_tokens->input_layernorm shared pyxrt BO")
    elif run_qkv:
        for proj_name, weight_info in info["projection_weights"].items():
            print(f"quantized weight {proj_name} {weight_info['name']} {weight_info['ggml_type']}")
        print(f"output_rows {args.output_rows}")
        print("handoff embed_tokens->input_layernorm->q/k/v shared pyxrt BO")
    else:
        print(f"Q4_K weight {info['q_proj_weight']['name']} {info['q_proj_weight']['ggml_type']}")
        print(f"output_rows {args.output_rows}")
        print("handoff embed_tokens->input_layernorm->q_proj shared pyxrt BO")

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
    if args.qproj_aie_mlir is None:
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
        actual_qproj = None
        qproj_xclbin = None
        qproj_insts = None
        actual_projections = {}
        projection_xclbins = {}
        projection_insts = {}
    elif run_qkv:
        projection_aie_mlirs = {
            "q_proj": args.qproj_aie_mlir,
            "k_proj": args.kproj_aie_mlir,
            "v_proj": args.vproj_aie_mlir,
        }
        projection_xclbins = {}
        projection_insts = {}
        for proj_name, aie_mlir in projection_aie_mlirs.items():
            projection_object = compile_projection_object(
                ggml_type=info["projection_weights"][proj_name]["ggml_type"],
                work_dir=args.work_dir / proj_name,
                peano_install_dir=peano_install_dir,
                output_tile_rows=args.output_tile_rows,
                blocks_per_row=args.blocks_per_row,
                hidden_size=hidden.shape[1],
            )
            _, projection_xclbins[proj_name], projection_insts[proj_name] = compile_runtime(
                aie_mlir=aie_mlir,
                work_dir=args.work_dir / proj_name,
                instance_name=f"run_{proj_name}",
                peano_install_dir=peano_install_dir,
                link_objects=(projection_object,),
            )
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
            embed_expected=embed_expected,
            norm_expected=expected,
            projection_expected=projection_expected,
            warmup=args.warmup,
            iterations=args.iterations,
            rtol=args.rtol,
            atol=args.atol,
            verbose=args.verbose,
        )
        actual_qproj = actual_projections["q_proj"]
        qproj_xclbin = projection_xclbins["q_proj"]
        qproj_insts = projection_insts["q_proj"]
    else:
        q4k_object = compile_q4k_linear_object(
            work_dir=args.work_dir / "q_proj",
            peano_install_dir=peano_install_dir,
            output_tile_rows=args.output_tile_rows,
            blocks_per_row=args.blocks_per_row,
            hidden_size=hidden.shape[1],
        )
        _, qproj_xclbin, qproj_insts = compile_runtime(
            aie_mlir=args.qproj_aie_mlir,
            work_dir=args.work_dir / "q_proj",
            instance_name="run_q_proj",
            peano_install_dir=peano_install_dir,
            link_objects=(q4k_object,),
        )
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
            qproj_weights=qproj_weights,
            qproj_output=qproj_output,
            embed_expected=embed_expected,
            norm_expected=expected,
            qproj_expected=qproj_expected,
            warmup=args.warmup,
            iterations=args.iterations,
            rtol=args.rtol,
            atol=args.atol,
            verbose=args.verbose,
        )
        actual_projections = {}
        projection_xclbins = {}
        projection_insts = {}

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
    print(f"hidden_first8 {actual_hidden.reshape(-1)[:8].tolist()}")
    print(f"output_first8 {actual_output.reshape(-1)[:8].tolist()}")
    if actual_projections:
        for proj_name, actual in actual_projections.items():
            print(f"{proj_name}_first8 {actual.reshape(-1)[:8].tolist()}")
            print(f"{proj_name}_expected_first8 {first_values(projection_expected[proj_name])}")
    elif actual_qproj is not None:
        print(f"qproj_first8 {actual_qproj.reshape(-1)[:8].tolist()}")
        print(f"qproj_expected_first8 {first_values(qproj_expected)}")
    print(f"expected_first8 {first_values(expected)}")
    print(f"hidden_max_abs {hidden_max_abs:.8g}")
    print(f"max_abs {output_max_abs:.8g}")
    if actual_projections:
        for proj_name, actual in actual_projections.items():
            projection_max_abs = max_abs_rocm(actual, projection_expected[proj_name])
            print(f"{proj_name}_max_abs {projection_max_abs:.8g}")
    elif actual_qproj is not None:
        qproj_max_abs = max_abs_rocm(actual_qproj, qproj_expected)
        print(f"qproj_max_abs {qproj_max_abs:.8g}")
    print(f"allclose True rtol={args.rtol:g} atol={args.atol:g}")
    if latencies_ms:
        print(f"mean_ms {sum(latencies_ms) / len(latencies_ms):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
