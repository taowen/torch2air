from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyxrt as xrt
import torch

from . import reference
from .air_runtime import load_xrt_kernel
from .reference_runtime import check_close_rocm
from .stages.attention import reference_attention_core
from .stages.projection import run_projection_slices
from .stages.rope import HEAD_DIM, reference_norm_rope
from .stages.self_attention import reference_oproj


def run_embed_chunks(
    *,
    embed_kernel: xrt.kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr: xrt.bo,
    bo_packed: xrt.bo,
    bo_scales: xrt.bo,
    bo_hidden: xrt.bo,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    hidden: np.ndarray,
    embed_chunk_rows: int,
) -> None:
    sequence_length, hidden_size = hidden.shape
    if embed_chunk_rows <= 0 or sequence_length % embed_chunk_rows != 0:
        raise ValueError("embed_chunk_rows must be positive and divide sequence length")
    if packed_rows.shape[0] != sequence_length:
        raise ValueError("packed_rows sequence length must match hidden")
    if block_f16_scales.shape[0] != sequence_length:
        raise ValueError("block_f16_scales sequence length must match hidden")

    packed_row_bytes = packed_rows.shape[1] * packed_rows.dtype.itemsize
    scales_row_bytes = (
        block_f16_scales.shape[1] * block_f16_scales.shape[2] * block_f16_scales.dtype.itemsize
    )
    hidden_row_bytes = hidden_size * hidden.dtype.itemsize
    packed_chunk_bytes = embed_chunk_rows * packed_row_bytes
    scales_chunk_bytes = embed_chunk_rows * scales_row_bytes
    hidden_chunk_bytes = embed_chunk_rows * hidden_row_bytes

    for token_offset in range(0, sequence_length, embed_chunk_rows):
        packed_chunk = xrt.bo(bo_packed, packed_chunk_bytes, token_offset * packed_row_bytes)
        scales_chunk = xrt.bo(bo_scales, scales_chunk_bytes, token_offset * scales_row_bytes)
        hidden_chunk = xrt.bo(bo_hidden, hidden_chunk_bytes, token_offset * hidden_row_bytes)
        embed_run = embed_kernel(
            3,
            embed_bo_instr,
            len(embed_instr_v),
            packed_chunk,
            scales_chunk,
            hidden_chunk,
        )
        embed_run.wait()


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
    embed_chunk_rows: int,
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
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            hidden=hidden,
            output=output,
            embed_chunk_rows=embed_chunk_rows,
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
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            hidden=hidden,
            output=output,
            embed_chunk_rows=embed_chunk_rows,
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
    output_tile_rows: int,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    qproj_expected: torch.Tensor,
    embed_chunk_rows: int,
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
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            hidden=hidden,
            norm_output=norm_output,
            qproj_weights=qproj_weights,
            qproj_output=qproj_output,
            output_tile_rows=output_tile_rows,
            embed_chunk_rows=embed_chunk_rows,
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
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    qproj_weights: np.ndarray,
    qproj_output: np.ndarray,
    output_tile_rows: int,
    embed_chunk_rows: int,
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

    run_embed_chunks(
        embed_kernel=embed_kernel,
        embed_instr_v=embed_instr_v,
        embed_bo_instr=embed_bo_instr,
        bo_packed=bo_packed,
        bo_scales=bo_scales,
        bo_hidden=bo_hidden,
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        hidden=hidden,
        embed_chunk_rows=embed_chunk_rows,
    )
    norm_run = norm_kernel(
        3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_norm_output
    )
    norm_run.wait()
    run_projection_slices(
        kernel=qproj_kernel,
        instr_v=qproj_instr_v,
        bo_instr=qproj_bo_instr,
        bo_input=bo_norm_output,
        bo_weights=bo_qproj_weights,
        bo_output=bo_qproj_output,
        input_array=norm_output,
        weights_array=qproj_weights,
        output_array=qproj_output,
        output_tile_rows=output_tile_rows,
    )

    for bo in (bo_hidden, bo_norm_output, bo_qproj_output):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = (
        bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    )
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
    output_tile_rows: int,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    projection_expected: dict[str, torch.Tensor],
    embed_chunk_rows: int,
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
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            hidden=hidden,
            norm_output=norm_output,
            projection_outputs=projection_outputs,
            projection_weights=projection_weights,
            output_tile_rows=output_tile_rows,
            embed_chunk_rows=embed_chunk_rows,
            embed_expected=embed_expected,
            norm_expected=norm_expected,
            projection_expected=projection_expected,
        )
        if iteration >= warmup:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_norm, norm_expected, rtol=rtol, atol=atol)
        for proj_name, actual in actual_projections.items():
            check_close_rocm(
                actual, projection_expected[proj_name], rtol=rtol, atol=atol, label=proj_name
            )

    return actual_hidden, actual_norm, actual_projections, latencies_ms


def run_shared_bo_projections_once(
    *,
    embed_kernel: xrt.kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr: xrt.bo,
    norm_kernel: xrt.kernel,
    norm_instr_v: np.ndarray,
    norm_bo_instr: xrt.bo,
    projection_kernels: dict[str, tuple[xrt.kernel, np.ndarray, xrt.bo]],
    bo_packed: xrt.bo,
    bo_scales: xrt.bo,
    bo_hidden: xrt.bo,
    bo_weight: xrt.bo,
    bo_norm_output: xrt.bo,
    bo_projection_weights: dict[str, xrt.bo],
    bo_projection_outputs: dict[str, xrt.bo],
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    projection_outputs: dict[str, np.ndarray],
    projection_weights: dict[str, np.ndarray],
    output_tile_rows: int,
    embed_chunk_rows: int,
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

    run_embed_chunks(
        embed_kernel=embed_kernel,
        embed_instr_v=embed_instr_v,
        embed_bo_instr=embed_bo_instr,
        bo_packed=bo_packed,
        bo_scales=bo_scales,
        bo_hidden=bo_hidden,
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        hidden=hidden,
        embed_chunk_rows=embed_chunk_rows,
    )
    norm_run = norm_kernel(
        3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_norm_output
    )
    norm_run.wait()
    for proj_name, (kernel, instr_v, bo_instr) in projection_kernels.items():
        run_projection_slices(
            kernel=kernel,
            instr_v=instr_v,
            bo_instr=bo_instr,
            bo_input=bo_norm_output,
            bo_weights=bo_projection_weights[proj_name],
            bo_output=bo_projection_outputs[proj_name],
            input_array=norm_output,
            weights_array=projection_weights[proj_name],
            output_array=projection_outputs[proj_name],
            output_tile_rows=output_tile_rows,
        )

    for bo in (bo_hidden, bo_norm_output):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    for bo in bo_projection_outputs.values():
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = (
        bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    )
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


def run_on_npu_projections_rope(
    *,
    embed_xclbin: Path,
    embed_insts: Path,
    norm_xclbin: Path,
    norm_insts: Path,
    projection_xclbins: dict[str, Path],
    projection_insts: dict[str, Path],
    rope_table_xclbin: Path,
    rope_table_insts: Path,
    norm_rope_xclbins: dict[str, Path],
    norm_rope_insts: dict[str, Path],
    attention_xclbins: list[Path],
    attention_insts_paths: list[Path],
    oproj_xclbin: Path | None,
    oproj_insts: Path | None,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    rms_weight: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    projection_weights: dict[str, np.ndarray],
    projection_outputs: dict[str, np.ndarray],
    output_tile_rows: int,
    q_norm_weight: np.ndarray,
    k_norm_weight: np.ndarray,
    start_position: np.ndarray,
    cos_output: np.ndarray,
    sin_output: np.ndarray,
    norm_rope_outputs: dict[str, np.ndarray],
    attention_output: np.ndarray | None,
    oproj_weights: np.ndarray | None,
    oproj_output: np.ndarray | None,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    projection_expected: dict[str, torch.Tensor],
    cos_expected: torch.Tensor,
    sin_expected: torch.Tensor,
    norm_rope_expected: dict[str, torch.Tensor],
    attention_expected: torch.Tensor | None,
    oproj_expected: torch.Tensor | None,
    embed_chunk_rows: int,
    warmup: int,
    iterations: int,
    rtol: float,
    atol: float,
    verbose: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray | None,
    torch.Tensor | None,
    np.ndarray | None,
    torch.Tensor | None,
    list[float],
]:
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
    _, rope_table_kernel, rope_table_instr_v, rope_table_bo_instr = load_xrt_kernel(
        device,
        xclbin=rope_table_xclbin,
        insts=rope_table_insts,
    )
    norm_rope_kernels = {}
    for stage_name, xclbin in norm_rope_xclbins.items():
        _, kernel, instr_v, bo_instr = load_xrt_kernel(
            device,
            xclbin=xclbin,
            insts=norm_rope_insts[stage_name],
        )
        norm_rope_kernels[stage_name] = (kernel, instr_v, bo_instr)
    if len(attention_xclbins) != len(attention_insts_paths):
        raise ValueError("attention_xclbins and attention_insts_paths must have the same length")

    bo_packed = xrt.bo(device, packed_rows.nbytes, xrt.bo.host_only, embed_kernel.group_id(3))
    bo_scales = xrt.bo(device, block_f16_scales.nbytes, xrt.bo.host_only, embed_kernel.group_id(4))
    bo_hidden = xrt.bo(device, hidden.nbytes, xrt.bo.host_only, embed_kernel.group_id(5))
    bo_weight = xrt.bo(device, rms_weight.nbytes, xrt.bo.host_only, norm_kernel.group_id(4))
    bo_norm_output = xrt.bo(device, norm_output.nbytes, xrt.bo.host_only, norm_kernel.group_id(5))
    bo_projection_weights = {}
    bo_projection_outputs = {}
    for proj_name, (kernel, _, _) in projection_kernels.items():
        weights = projection_weights[proj_name]
        output = projection_outputs[proj_name]
        bo_projection_weights[proj_name] = xrt.bo(
            device, weights.nbytes, xrt.bo.host_only, kernel.group_id(4)
        )
        bo_projection_outputs[proj_name] = xrt.bo(
            device, output.nbytes, xrt.bo.host_only, kernel.group_id(5)
        )

    bo_rope_start = xrt.bo(
        device, start_position.nbytes, xrt.bo.host_only, rope_table_kernel.group_id(3)
    )
    bo_cos = xrt.bo(device, cos_output.nbytes, xrt.bo.host_only, rope_table_kernel.group_id(4))
    bo_sin = xrt.bo(device, sin_output.nbytes, xrt.bo.host_only, rope_table_kernel.group_id(5))

    norm_rope_weights = {
        "q_norm_rope": q_norm_weight,
        "k_norm_rope": k_norm_weight,
    }
    bo_norm_rope_weights = {}
    bo_norm_rope_outputs = {}
    for stage_name, (kernel, _, _) in norm_rope_kernels.items():
        weight = norm_rope_weights[stage_name]
        output = norm_rope_outputs[stage_name]
        bo_norm_rope_weights[stage_name] = xrt.bo(
            device, weight.nbytes, xrt.bo.host_only, kernel.group_id(4)
        )
        bo_norm_rope_outputs[stage_name] = xrt.bo(
            device, output.nbytes, xrt.bo.host_only, kernel.group_id(7)
        )

    for bo, array in (
        (bo_packed, packed_rows),
        (bo_scales, block_f16_scales),
        (bo_weight, rms_weight),
        (bo_rope_start, start_position),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for proj_name, bo in bo_projection_weights.items():
        bo.write(projection_weights[proj_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for stage_name, bo in bo_norm_rope_weights.items():
        bo.write(norm_rope_weights[stage_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    actual_hidden = hidden
    actual_norm = norm_output
    actual_projections = projection_outputs
    actual_cos = cos_output
    actual_sin = sin_output
    actual_norm_rope = norm_rope_outputs
    actual_attention = attention_output
    actual_attention_expected = attention_expected
    actual_oproj = oproj_output
    actual_oproj_expected = oproj_expected
    latencies_ms: list[float] = []
    for iteration in range(warmup + iterations):
        hidden.fill(0)
        norm_output.fill(0)
        cos_output.fill(0)
        sin_output.fill(0)
        for output in projection_outputs.values():
            output.fill(0)
        for output in norm_rope_outputs.values():
            output.fill(0)
        if attention_output is not None:
            attention_output.fill(0)
        if oproj_output is not None:
            oproj_output.fill(0)
        start = time.perf_counter()
        (
            actual_hidden,
            actual_norm,
            actual_projections,
            actual_cos,
            actual_sin,
            actual_norm_rope,
            actual_attention,
            actual_oproj,
        ) = run_shared_bo_projections_rope_once(
            device=device,
            embed_kernel=embed_kernel,
            embed_instr_v=embed_instr_v,
            embed_bo_instr=embed_bo_instr,
            norm_kernel=norm_kernel,
            norm_instr_v=norm_instr_v,
            norm_bo_instr=norm_bo_instr,
            projection_kernels=projection_kernels,
            rope_table_kernel=rope_table_kernel,
            rope_table_instr_v=rope_table_instr_v,
            rope_table_bo_instr=rope_table_bo_instr,
            norm_rope_kernels=norm_rope_kernels,
            bo_packed=bo_packed,
            bo_scales=bo_scales,
            bo_hidden=bo_hidden,
            bo_weight=bo_weight,
            bo_norm_output=bo_norm_output,
            bo_projection_weights=bo_projection_weights,
            bo_projection_outputs=bo_projection_outputs,
            bo_rope_start=bo_rope_start,
            bo_cos=bo_cos,
            bo_sin=bo_sin,
            bo_norm_rope_weights=bo_norm_rope_weights,
            bo_norm_rope_outputs=bo_norm_rope_outputs,
            attention_xclbins=attention_xclbins,
            attention_insts_paths=attention_insts_paths,
            oproj_xclbin=oproj_xclbin,
            oproj_insts=oproj_insts,
            packed_rows=packed_rows,
            block_f16_scales=block_f16_scales,
            hidden=hidden,
            norm_output=norm_output,
            projection_outputs=projection_outputs,
            projection_weights=projection_weights,
            output_tile_rows=output_tile_rows,
            embed_chunk_rows=embed_chunk_rows,
            cos_output=cos_output,
            sin_output=sin_output,
            norm_rope_outputs=norm_rope_outputs,
            attention_output=attention_output,
            oproj_weights=oproj_weights,
            oproj_output=oproj_output,
            embed_expected=embed_expected,
            norm_expected=norm_expected,
            projection_expected=projection_expected,
            cos_expected=cos_expected,
            sin_expected=sin_expected,
            norm_rope_expected=norm_rope_expected,
            oproj_expected=oproj_expected,
        )
        if iteration >= warmup:
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        check_close_rocm(actual_hidden, embed_expected, rtol=1e-2, atol=1e-2)
        check_close_rocm(actual_norm, norm_expected, rtol=rtol, atol=atol)
        for proj_name, actual in actual_projections.items():
            check_close_rocm(
                actual, projection_expected[proj_name], rtol=rtol, atol=atol, label=proj_name
            )
        check_close_rocm(actual_cos, cos_expected, rtol=2e-3, atol=2e-3, label="rope_cos")
        check_close_rocm(actual_sin, sin_expected, rtol=2e-3, atol=2e-3, label="rope_sin")
        reference_device = next(reference.get_model().parameters()).device
        norm_rope_inputs = {
            "q_norm_rope": "q_proj",
            "k_norm_rope": "k_proj",
        }
        actual_cos_t = torch.as_tensor(
            np.ascontiguousarray(actual_cos), device=reference_device, dtype=torch.float32
        )
        actual_sin_t = torch.as_tensor(
            np.ascontiguousarray(actual_sin), device=reference_device, dtype=torch.float32
        )
        for stage_name, projection_name in norm_rope_inputs.items():
            actual_projection_t = torch.as_tensor(
                np.ascontiguousarray(actual_projections[projection_name]),
                device=reference_device,
                dtype=torch.float32,
            )
            norm_rope_expected[stage_name] = reference_norm_rope(
                norm_name=stage_name,
                projection=actual_projection_t,
                cos=actual_cos_t,
                sin=actual_sin_t,
                head_count=actual_projection_t.shape[1] // HEAD_DIM,
            )
        for stage_name, actual in actual_norm_rope.items():
            check_close_rocm(
                actual, norm_rope_expected[stage_name], rtol=rtol, atol=atol, label=stage_name
            )
        if actual_attention is not None:
            q_t = torch.as_tensor(
                np.ascontiguousarray(actual_norm_rope["q_norm_rope"]),
                device=reference_device,
                dtype=torch.float32,
            )
            k_t = torch.as_tensor(
                np.ascontiguousarray(actual_norm_rope["k_norm_rope"]),
                device=reference_device,
                dtype=torch.float32,
            )
            v_t = torch.as_tensor(
                np.ascontiguousarray(actual_projections["v_proj"]),
                device=reference_device,
                dtype=torch.float32,
            )
            q_heads = actual_norm_rope["q_norm_rope"].shape[1] // HEAD_DIM
            kv_heads = actual_norm_rope["k_norm_rope"].shape[1] // HEAD_DIM
            actual_attention_expected = reference_attention_core(
                q=q_t,
                k=k_t,
                v=v_t,
                q_heads=q_heads,
                kv_heads=kv_heads,
            )
            check_close_rocm(
                actual_attention,
                actual_attention_expected,
                rtol=rtol,
                atol=atol,
                label="attention_core",
            )
        if actual_oproj is not None:
            if actual_attention is None:
                raise RuntimeError("o_proj requires attention output")
            reference_device = next(reference.get_model().parameters()).device
            attention_t = torch.as_tensor(
                np.ascontiguousarray(actual_attention),
                device=reference_device,
                dtype=torch.float32,
            )
            actual_oproj_expected = reference_oproj(attention_t)
            check_close_rocm(
                actual_oproj, actual_oproj_expected, rtol=rtol, atol=atol, label="o_proj"
            )

    return (
        actual_hidden,
        actual_norm,
        actual_projections,
        actual_cos,
        actual_sin,
        actual_norm_rope,
        actual_attention,
        actual_attention_expected,
        actual_oproj,
        actual_oproj_expected,
        latencies_ms,
    )


def run_attention_xclbins(
    *,
    device: xrt.device,
    attention_xclbins: list[Path],
    attention_insts_paths: list[Path],
    bo_q: xrt.bo,
    bo_k: xrt.bo,
    bo_v: xrt.bo,
    attention_output: np.ndarray,
) -> xrt.bo:
    bo_attention_output: xrt.bo | None = None
    for xclbin, insts in zip(attention_xclbins, attention_insts_paths, strict=True):
        context, kernel, instr_v, bo_instr = load_xrt_kernel(
            device,
            xclbin=xclbin,
            insts=insts,
        )
        if bo_attention_output is None:
            bo_attention_output = xrt.bo(
                device,
                attention_output.nbytes,
                xrt.bo.host_only,
                kernel.group_id(6),
            )
            bo_attention_output.write(attention_output, 0)
            bo_attention_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        attention_run = kernel(
            3,
            bo_instr,
            len(instr_v),
            bo_q,
            bo_k,
            bo_v,
            bo_attention_output,
        )
        attention_run.wait()
        del bo_instr, instr_v, kernel, context
    assert bo_attention_output is not None
    return bo_attention_output


def run_oproj_xclbin(
    *,
    device: xrt.device,
    oproj_xclbin: Path,
    oproj_insts: Path,
    bo_attention_output: xrt.bo,
    attention_output: np.ndarray,
    oproj_weights: np.ndarray,
    oproj_output: np.ndarray,
    output_tile_rows: int,
) -> xrt.bo:
    context, kernel, instr_v, bo_instr = load_xrt_kernel(
        device,
        xclbin=oproj_xclbin,
        insts=oproj_insts,
    )
    bo_oproj_weights = xrt.bo(
        device,
        oproj_weights.nbytes,
        xrt.bo.host_only,
        kernel.group_id(4),
    )
    bo_oproj_output = xrt.bo(
        device,
        oproj_output.nbytes,
        xrt.bo.host_only,
        kernel.group_id(5),
    )
    bo_oproj_weights.write(oproj_weights, 0)
    bo_oproj_weights.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_oproj_output.write(oproj_output, 0)
    bo_oproj_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    run_projection_slices(
        kernel=kernel,
        instr_v=instr_v,
        bo_instr=bo_instr,
        bo_input=bo_attention_output,
        bo_weights=bo_oproj_weights,
        bo_output=bo_oproj_output,
        input_array=attention_output,
        weights_array=oproj_weights,
        output_array=oproj_output,
        output_tile_rows=output_tile_rows,
    )
    del bo_oproj_weights, bo_instr, instr_v, kernel, context
    return bo_oproj_output


def run_shared_bo_projections_rope_once(
    *,
    device: xrt.device,
    embed_kernel: xrt.kernel,
    embed_instr_v: np.ndarray,
    embed_bo_instr: xrt.bo,
    norm_kernel: xrt.kernel,
    norm_instr_v: np.ndarray,
    norm_bo_instr: xrt.bo,
    projection_kernels: dict[str, tuple[xrt.kernel, np.ndarray, xrt.bo]],
    rope_table_kernel: xrt.kernel,
    rope_table_instr_v: np.ndarray,
    rope_table_bo_instr: xrt.bo,
    norm_rope_kernels: dict[str, tuple[xrt.kernel, np.ndarray, xrt.bo]],
    bo_packed: xrt.bo,
    bo_scales: xrt.bo,
    bo_hidden: xrt.bo,
    bo_weight: xrt.bo,
    bo_norm_output: xrt.bo,
    bo_projection_weights: dict[str, xrt.bo],
    bo_projection_outputs: dict[str, xrt.bo],
    bo_rope_start: xrt.bo,
    bo_cos: xrt.bo,
    bo_sin: xrt.bo,
    bo_norm_rope_weights: dict[str, xrt.bo],
    bo_norm_rope_outputs: dict[str, xrt.bo],
    attention_xclbins: list[Path],
    attention_insts_paths: list[Path],
    oproj_xclbin: Path | None,
    oproj_insts: Path | None,
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    hidden: np.ndarray,
    norm_output: np.ndarray,
    projection_outputs: dict[str, np.ndarray],
    projection_weights: dict[str, np.ndarray],
    output_tile_rows: int,
    embed_chunk_rows: int,
    cos_output: np.ndarray,
    sin_output: np.ndarray,
    norm_rope_outputs: dict[str, np.ndarray],
    attention_output: np.ndarray | None,
    oproj_weights: np.ndarray | None,
    oproj_output: np.ndarray | None,
    embed_expected: torch.Tensor,
    norm_expected: torch.Tensor,
    projection_expected: dict[str, torch.Tensor],
    cos_expected: torch.Tensor,
    sin_expected: torch.Tensor,
    norm_rope_expected: dict[str, torch.Tensor],
    oproj_expected: torch.Tensor | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
]:
    for bo, array in (
        (bo_hidden, hidden),
        (bo_norm_output, norm_output),
        (bo_cos, cos_output),
        (bo_sin, sin_output),
    ):
        bo.write(array, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for proj_name, bo in bo_projection_outputs.items():
        bo.write(projection_outputs[proj_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for stage_name, bo in bo_norm_rope_outputs.items():
        bo.write(norm_rope_outputs[stage_name], 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_attention_output: xrt.bo | None = None
    bo_oproj_output: xrt.bo | None = None

    run_embed_chunks(
        embed_kernel=embed_kernel,
        embed_instr_v=embed_instr_v,
        embed_bo_instr=embed_bo_instr,
        bo_packed=bo_packed,
        bo_scales=bo_scales,
        bo_hidden=bo_hidden,
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        hidden=hidden,
        embed_chunk_rows=embed_chunk_rows,
    )
    norm_run = norm_kernel(
        3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_norm_output
    )
    norm_run.wait()
    for proj_name, (kernel, instr_v, bo_instr) in projection_kernels.items():
        run_projection_slices(
            kernel=kernel,
            instr_v=instr_v,
            bo_instr=bo_instr,
            bo_input=bo_norm_output,
            bo_weights=bo_projection_weights[proj_name],
            bo_output=bo_projection_outputs[proj_name],
            input_array=norm_output,
            weights_array=projection_weights[proj_name],
            output_array=projection_outputs[proj_name],
            output_tile_rows=output_tile_rows,
        )
    rope_table_run = rope_table_kernel(
        3,
        rope_table_bo_instr,
        len(rope_table_instr_v),
        bo_rope_start,
        bo_cos,
        bo_sin,
    )
    rope_table_run.wait()
    norm_rope_inputs = {
        "q_norm_rope": "q_proj",
        "k_norm_rope": "k_proj",
    }
    for stage_name, (kernel, instr_v, bo_instr) in norm_rope_kernels.items():
        projection_name = norm_rope_inputs[stage_name]
        norm_rope_run = kernel(
            3,
            bo_instr,
            len(instr_v),
            bo_projection_outputs[projection_name],
            bo_norm_rope_weights[stage_name],
            bo_cos,
            bo_sin,
            bo_norm_rope_outputs[stage_name],
        )
        norm_rope_run.wait()
    if attention_xclbins:
        assert attention_output is not None
        bo_attention_output = run_attention_xclbins(
            device=device,
            attention_xclbins=attention_xclbins,
            attention_insts_paths=attention_insts_paths,
            bo_q=bo_norm_rope_outputs["q_norm_rope"],
            bo_k=bo_norm_rope_outputs["k_norm_rope"],
            bo_v=bo_projection_outputs["v_proj"],
            attention_output=attention_output,
        )
    if oproj_xclbin is not None and oproj_insts is not None:
        assert bo_attention_output is not None
        assert oproj_weights is not None
        assert oproj_output is not None
        assert attention_output is not None
        bo_oproj_output = run_oproj_xclbin(
            device=device,
            oproj_xclbin=oproj_xclbin,
            oproj_insts=oproj_insts,
            bo_attention_output=bo_attention_output,
            attention_output=attention_output,
            oproj_weights=oproj_weights,
            oproj_output=oproj_output,
            output_tile_rows=output_tile_rows,
        )

    for bo in (bo_hidden, bo_norm_output, bo_cos, bo_sin):
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    for bo in bo_projection_outputs.values():
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    for bo in bo_norm_rope_outputs.values():
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    if bo_attention_output is not None:
        bo_attention_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    if bo_oproj_output is not None:
        bo_oproj_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    actual_hidden = (
        bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    )
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
    actual_cos = (
        bo_cos.read(cos_output.nbytes, 0).view(cos_output.dtype).reshape(tuple(cos_expected.shape))
    )
    actual_sin = (
        bo_sin.read(sin_output.nbytes, 0).view(sin_output.dtype).reshape(tuple(sin_expected.shape))
    )
    actual_norm_rope = {}
    for stage_name, bo in bo_norm_rope_outputs.items():
        output = norm_rope_outputs[stage_name]
        actual_norm_rope[stage_name] = (
            bo.read(output.nbytes, 0)
            .view(output.dtype)
            .reshape(tuple(norm_rope_expected[stage_name].shape))
        )
    actual_attention: np.ndarray | None = None
    if bo_attention_output is not None and attention_output is not None:
        actual_attention = (
            bo_attention_output.read(attention_output.nbytes, 0)
            .view(attention_output.dtype)
            .reshape(tuple(attention_output.shape))
        )
    actual_oproj: np.ndarray | None = None
    if bo_oproj_output is not None and oproj_output is not None:
        assert oproj_expected is not None
        actual_oproj = (
            bo_oproj_output.read(oproj_output.nbytes, 0)
            .view(oproj_output.dtype)
            .reshape(tuple(oproj_expected.shape))
        )
    return (
        actual_hidden,
        actual_norm,
        actual_projections,
        actual_cos,
        actual_sin,
        actual_norm_rope,
        actual_attention,
        actual_oproj,
    )


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
    packed_rows: np.ndarray,
    block_f16_scales: np.ndarray,
    hidden: np.ndarray,
    output: np.ndarray,
    embed_chunk_rows: int,
    embed_expected: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    bo_hidden.write(hidden, 0)
    bo_hidden.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_output.write(output, 0)
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run_embed_chunks(
        embed_kernel=embed_kernel,
        embed_instr_v=embed_instr_v,
        embed_bo_instr=embed_bo_instr,
        bo_packed=bo_packed,
        bo_scales=bo_scales,
        bo_hidden=bo_hidden,
        packed_rows=packed_rows,
        block_f16_scales=block_f16_scales,
        hidden=hidden,
        embed_chunk_rows=embed_chunk_rows,
    )
    norm_run = norm_kernel(3, norm_bo_instr, len(norm_instr_v), bo_hidden, bo_weight, bo_output)
    norm_run.wait()

    bo_hidden.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    actual_hidden = (
        bo_hidden.read(hidden.nbytes, 0).view(hidden.dtype).reshape(tuple(embed_expected.shape))
    )
    actual_output = (
        bo_output.read(output.nbytes, 0).view(output.dtype).reshape(tuple(expected.shape))
    )
    return actual_hidden, actual_output
