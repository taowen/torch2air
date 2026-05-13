"""PyTorch modules used as the source and reference for Qwen3 AIR kernels."""

from __future__ import annotations

from math import prod

import torch

from models.quantized_qwen3.kernels import Qwen3Kernel


class Qwen3LinearTile(torch.nn.Module):
    """Dequantized linear tile for the first Qwen3 AIR path.

    The right-hand operand is already transposed into linalg.matmul form
    `[in_features, out_features]`. Q4_K/Q6_K unpacking is intentionally kept
    outside this first tile.
    """

    def forward(self, activations: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return activations @ weight_t


def make_torch_inputs(kernel: Qwen3Kernel) -> tuple[torch.Tensor, torch.Tensor]:
    lhs_size = prod(kernel.lhs_shape)
    rhs_size = prod(kernel.rhs_shape)
    lhs = torch.arange(lhs_size, dtype=kernel.dtype).reshape(kernel.lhs_shape).remainder(17)
    rhs = torch.arange(rhs_size, dtype=kernel.dtype).reshape(kernel.rhs_shape).remainder(13)
    if kernel.dtype.is_floating_point:
        lhs = lhs / 17.0
        rhs = rhs / 13.0
    lhs = lhs.contiguous()
    rhs = rhs.contiguous()
    return lhs, rhs


def reference_output(kernel: Qwen3Kernel) -> torch.Tensor:
    model = Qwen3LinearTile().eval()
    lhs, rhs = make_torch_inputs(kernel)
    with torch.no_grad():
        return model(lhs, rhs).contiguous()
