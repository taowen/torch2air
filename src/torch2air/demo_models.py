from __future__ import annotations

import torch


class MatmulModel(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs @ rhs


def make_matmul_inputs(
    *,
    m: int = 32,
    k: int = 32,
    n: int = 32,
    dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    lhs = torch.arange(m * k, dtype=dtype).reshape(m, k) % 17
    rhs = torch.arange(k * n, dtype=dtype).reshape(k, n) % 13
    return lhs.contiguous(), rhs.contiguous()
