# Quantized Qwen3

First-pass AIR export experiments for the standalone `Qwen/Qwen3-0.6B` text
model used by `torch2vk`'s `models.quantized_qwen3` package.

This package mirrors the `torch2vk` model layout, but starts with a single
NPU4-safe kernel tile:

- `prefill_q_proj_tile_f32`: `32x1024 @ 1024x128 -> 32x128`

The kernel uses dequantized `f32` inputs for now. Export and verification both
use the PyTorch module in `modules.py`: the same module is lowered to MLIR and
run directly in PyTorch ROCm to produce the reference output. The runner is only
a development smoke test. It loads the VMFB through the IREE Python runtime and
compares the IREE output against the PyTorch ROCm reference with torch
operations. Q4_K/Q6_K GGUF weight loading and fused dequantization are the next
layer above this scaffold.
