from __future__ import annotations

import numpy as np
import torch


def rocm_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch ROCm reference requires torch.cuda.is_available()")
    return torch.device("cuda")


def q4k_block_f16_scales_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint8 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 144:
        raise ValueError(
            f"Expected uint8 Q4_K blocks with shape [N, 144], got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    d_bytes = torch.as_tensor(np.array(raw_blocks[:, 0:2], copy=True, order="C"), device=device)
    dmin_bytes = torch.as_tensor(np.array(raw_blocks[:, 2:4], copy=True, order="C"), device=device)
    d = d_bytes.view(torch.float16).to(torch.float32).reshape(-1)
    dmin = dmin_bytes.view(torch.float16).to(torch.float32).reshape(-1)
    return torch.stack([d, dmin], dim=1)


def q6k_block_f16_scales_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint16 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 105:
        raise ValueError(
            f"Expected uint16 Q6_K blocks with shape [N, 105], got {raw_blocks.shape} {raw_blocks.dtype}"
        )
    device = rocm_device()
    d_bytes = np.array(raw_blocks[:, 104:105], copy=True, order="C").view(np.uint8).reshape(-1, 2)
    return torch.as_tensor(d_bytes, device=device).view(torch.float16).to(torch.float32).reshape(-1)


def dequantize_q4_k_blocks_rocm(raw_blocks: np.ndarray) -> torch.Tensor:
    if raw_blocks.dtype != np.uint8 or raw_blocks.ndim != 2 or raw_blocks.shape[1] != 144:
        raise ValueError(
            f"Expected uint8 Q4_K blocks with shape [N, 144], got {raw_blocks.shape} {raw_blocks.dtype}"
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


def rmsnorm_rocm(hidden: torch.Tensor, weight: np.ndarray, eps: float) -> torch.Tensor:
    weight_t = torch.as_tensor(weight, device=hidden.device, dtype=torch.float32)
    hidden_t = hidden.to(torch.float32)
    variance = torch.mean(hidden_t * hidden_t, dim=-1, keepdim=True)
    return hidden_t * torch.rsqrt(variance + eps) * weight_t.reshape(1, -1)


def check_close_rocm(
    actual: np.ndarray,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    label: str = "NPU output",
) -> None:
    actual_t = torch.as_tensor(np.ascontiguousarray(actual), device=expected.device, dtype=torch.float32)
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
    actual_t = torch.as_tensor(np.ascontiguousarray(actual), device=expected.device, dtype=torch.float32)
    return float(torch.max(torch.abs(actual_t - expected.to(torch.float32))).item())


def first_values(tensor: torch.Tensor, count: int = 8) -> list[float]:
    return [float(value) for value in tensor.reshape(-1)[:count].detach().cpu().tolist()]
