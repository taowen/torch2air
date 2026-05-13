# Quantized Qwen3

First-pass AIR export experiments for the standalone `Qwen/Qwen3-0.6B` text
model used by `torch2vk`'s `models.quantized_qwen3` package.

This package mirrors the `torch2vk` model layout, but starts with a single
NPU4-safe kernel tile:

- `prefill_q_proj_tile_i32`: `32x1024 @ 1024x2048 -> 32x2048`

The kernel uses dequantized `i32` inputs for now. Export and verification both
use the PyTorch module in `modules.py`: the same module is lowered to MLIR and
run directly in PyTorch to produce the reference output. The runner is only a
development smoke test. It writes PyTorch-generated inputs to `.npy` because
`iree-run-module` accepts that file format for command-line tensor I/O; the
output is converted back to a torch tensor and compared with the PyTorch
reference. Deployment should call the VMFB through the IREE runtime API with
application-owned buffers instead. Q4_K/Q6_K GGUF weight loading and fused
dequantization are the next layer above this scaffold.
