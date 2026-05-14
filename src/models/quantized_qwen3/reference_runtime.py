from __future__ import annotations

import numpy as np
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


def rocm_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm reference requires torch.cuda.is_available()")
    return torch.device("cuda")


def q4k_block_f16_scales_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint8 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 144:
        raise ValueError(
            f"Expected uint8 Q4_K blocks with shape [N, 144], "
            f"got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    d_bytes = torch.as_tensor(np.array(raw_blocks[:, 0:2], copy=True, order="C"), device=device)
    dmin_bytes = torch.as_tensor(
        np.array(raw_blocks[:, 2:4], copy=True, order="C"),
        device=device,
    )
    d = d_bytes.view(torch.float16).to(torch.float32).reshape(-1)
    dmin = dmin_bytes.view(torch.float16).to(torch.float32).reshape(-1)
    return torch.stack([d, dmin], dim=1)


def q6k_block_f16_scales_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint16 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 105:
        raise ValueError(
            f"Expected uint16 Q6_K blocks with shape [N, 105], "
            f"got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    d_words = torch.as_tensor(np.array(raw_blocks[:, 104:105], copy=True, order="C"), device=device)
    return d_words.view(torch.float16).to(torch.float32).reshape(-1)


def dequantize_q4_k_blocks_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint8 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 144:
        raise ValueError(
            f"Expected uint8 Q4_K blocks with shape [N, 144], "
            f"got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    raw = torch.as_tensor(np.array(raw_blocks, copy=True, order="C"), device=device)
    n_blocks = raw.shape[0]

    block_scales = q4k_block_f16_scales_rocm(raw_blocks)
    d = block_scales[:, 0].reshape(n_blocks, 1, 1)
    dmin = block_scales[:, 1].reshape(n_blocks, 1, 1)

    scale_bytes = raw[:, 4:16].to(torch.int32).reshape(n_blocks, 3, 4)
    d_bytes = scale_bytes[:, 0:1, :]
    m_bytes = scale_bytes[:, 1:2, :]
    md_bytes = scale_bytes[:, 2:3, :]
    scale = torch.cat(
        [
            torch.bitwise_and(d_bytes, 0x3F),
            torch.bitwise_or(
                torch.bitwise_and(md_bytes, 0x0F),
                torch.bitwise_and(torch.bitwise_right_shift(d_bytes, 2), 0x30),
            ),
        ],
        dim=-1,
    ).reshape(n_blocks, 8)
    minimum = torch.cat(
        [
            torch.bitwise_and(m_bytes, 0x3F),
            torch.bitwise_or(
                torch.bitwise_right_shift(md_bytes, 4),
                torch.bitwise_and(torch.bitwise_right_shift(m_bytes, 2), 0x30),
            ),
        ],
        dim=-1,
    ).reshape(n_blocks, 8)

    shifts = torch.tensor([0, 4], device=device, dtype=torch.int32).reshape(1, 1, 2, 1)
    qs = torch.bitwise_right_shift(
        raw[:, 16:].to(torch.int32).reshape(n_blocks, -1, 1, 32),
        shifts,
    )
    qs = torch.bitwise_and(qs, 0x0F).reshape(n_blocks, -1, 32).to(torch.float32)
    return (
        d * scale.to(torch.float32).reshape(n_blocks, 8, 1) * qs
        - dmin * minimum.to(torch.float32).reshape(n_blocks, 8, 1)
    ).reshape(n_blocks, 256)


def dequantize_q6_k_blocks_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint16 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 105:
        raise ValueError(
            f"Expected uint16 Q6_K blocks with shape [N, 105], "
            f"got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    raw_bytes = raw_blocks.view(np.uint8).reshape(raw_blocks.shape[0], 210)
    raw = torch.as_tensor(np.array(raw_bytes, copy=True, order="C"), device=device)
    n_blocks = raw.shape[0]
    ql = raw[:, 0:128].to(torch.int32)
    qh = raw[:, 128:192].to(torch.int32)
    scales = raw[:, 192:208].to(torch.int8).to(torch.int32)
    d = q6k_block_f16_scales_rocm(raw_blocks).reshape(n_blocks, 1)
    output = torch.empty((n_blocks, 256), device=device, dtype=torch.float32)
    lanes = torch.arange(32, device=device, dtype=torch.long)
    scale_lane = lanes // 16

    for half_index in range(2):
        ql_base = half_index * 64
        qh_base = half_index * 32
        scale_base = half_index * 8
        output_base = half_index * 128
        ql0 = ql[:, ql_base : ql_base + 32]
        ql1 = ql[:, ql_base + 32 : ql_base + 64]
        qh_part = qh[:, qh_base : qh_base + 32]

        q1 = torch.bitwise_or(
            torch.bitwise_and(ql0, 0x0F),
            torch.bitwise_left_shift(torch.bitwise_and(qh_part, 0x03), 4),
        ).to(torch.float32) - 32.0
        q2 = torch.bitwise_or(
            torch.bitwise_and(ql1, 0x0F),
            torch.bitwise_left_shift(
                torch.bitwise_and(torch.bitwise_right_shift(qh_part, 2), 0x03),
                4,
            ),
        ).to(torch.float32) - 32.0
        q3 = torch.bitwise_or(
            torch.bitwise_right_shift(ql0, 4),
            torch.bitwise_left_shift(
                torch.bitwise_and(torch.bitwise_right_shift(qh_part, 4), 0x03),
                4,
            ),
        ).to(torch.float32) - 32.0
        q4 = torch.bitwise_or(
            torch.bitwise_right_shift(ql1, 4),
            torch.bitwise_left_shift(torch.bitwise_right_shift(qh_part, 6), 4),
        ).to(torch.float32) - 32.0

        s1 = scales[:, scale_base + scale_lane + 0].to(torch.float32)
        s2 = scales[:, scale_base + scale_lane + 2].to(torch.float32)
        s3 = scales[:, scale_base + scale_lane + 4].to(torch.float32)
        s4 = scales[:, scale_base + scale_lane + 6].to(torch.float32)
        output[:, output_base : output_base + 32] = d * s1 * q1
        output[:, output_base + 32 : output_base + 64] = d * s2 * q2
        output[:, output_base + 64 : output_base + 96] = d * s3 * q3
        output[:, output_base + 96 : output_base + 128] = d * s4 * q4

    return output


def q4k_embedding_module_rocm(
    *,
    num_embeddings: int,
    token_ids: list[int],
    raw_blocks: np.ndarray,
) -> torch.nn.Embedding:
    rows = _q4k_embedding_rows_rocm(token_ids=token_ids, raw_blocks=raw_blocks)
    device = rocm_device()
    module = torch.nn.Embedding(
        num_embeddings,
        int(rows.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    indices = torch.tensor(token_ids, device=device, dtype=torch.long)
    with torch.no_grad():
        module.weight[indices, :] = rows
    module.eval()
    return module


def run_embedding_module_rocm(
    module: torch.nn.Embedding,
    *,
    token_ids: list[int],
) -> torch.Tensor:
    input_ids = torch.tensor(token_ids, device=rocm_device(), dtype=torch.long).reshape(1, -1)
    with torch.no_grad():
        return module(input_ids).reshape(len(token_ids), int(module.weight.shape[1]))


def qwen3_rms_norm_module_rocm(
    *,
    hidden_size: int,
    weight: np.ndarray,
    eps: float,
) -> Qwen3RMSNorm:
    module = Qwen3RMSNorm(hidden_size, eps=eps).to(device=rocm_device(), dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(torch.as_tensor(np.ascontiguousarray(weight), device=rocm_device()))
    module.eval()
    return module


def run_input_layernorm_module_rocm(
    module: Qwen3RMSNorm,
    *,
    hidden: torch.Tensor,
) -> torch.Tensor:
    hidden_t = hidden.to(device=rocm_device(), dtype=torch.float32).reshape(
        1,
        int(hidden.shape[0]),
        int(hidden.shape[1]),
    )
    with torch.no_grad():
        return module(hidden_t).reshape(int(hidden.shape[0]), int(hidden.shape[1]))


def check_close_rocm(
    actual: np.ndarray,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    label: str = "NPU output",
) -> None:
    actual_t = torch.as_tensor(
        np.ascontiguousarray(actual),
        device=expected.device,
        dtype=torch.float32,
    )
    expected_t = expected.to(torch.float32)
    if torch.allclose(actual_t, expected_t, rtol=rtol, atol=atol):
        return
    diff = torch.abs(actual_t - expected_t)
    flat_idx = int(torch.argmax(diff).item())
    index = np.unravel_index(flat_idx, tuple(diff.shape))
    raise AssertionError(
        f"{label} mismatch: "
        f"max_abs={float(diff.reshape(-1)[flat_idx].item())} "
        f"index={index} "
        f"actual={float(actual_t.reshape(-1)[flat_idx].item())} "
        f"expected={float(expected_t.reshape(-1)[flat_idx].item())}"
    )


def max_abs_rocm(actual: np.ndarray, expected: torch.Tensor) -> float:
    actual_t = torch.as_tensor(
        np.ascontiguousarray(actual),
        device=expected.device,
        dtype=torch.float32,
    )
    return float(torch.max(torch.abs(actual_t - expected.to(torch.float32))).item())


def first_values(tensor: torch.Tensor, count: int = 8) -> list[float]:
    return [float(value) for value in tensor.reshape(-1)[:count].detach().cpu().tolist()]


def _q4k_embedding_rows_rocm(
    *,
    token_ids: list[int],
    raw_blocks: np.ndarray,
) -> torch.Tensor:
    if not token_ids:
        raise ValueError("expected at least one token id")
    if raw_blocks.shape[0] % len(token_ids) != 0:
        raise ValueError("Q4_K block count must be divisible by token count")
    blocks_per_row = raw_blocks.shape[0] // len(token_ids)
    return dequantize_q4_k_blocks_rocm(raw_blocks).reshape(len(token_ids), blocks_per_row * 256)
