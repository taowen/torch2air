# Export Aten Linear Kernel

问题：已经验证过的 AIR kernel，怎样接回正式 `torch2air.export`，同时不重新包装
PyTorch 导出的对象？

正式路径保持一层映射：

```text
torch.export ExportedProgram
  -> generated Python export
  -> op-specific AIR builder
  -> AIR external-kernel schedule or AIR compute schedule
```

不要把 PyTorch 导出的对象再包一层 graph。kernel builder 只消费生成的
`builder.define_tensor(...)` 和 `builder.emit_kernel("aten.linear.default", ...)`，
让 generated export 成为 single source of truth。

验证记录使用 GGUF Q4_K projection，decode 配置：

```text
sequence_length = 1
output_rows per xclbin = 64
output_tile_rows per AIE tile = 16
herd = 1 x 4
weight tile ABI = memref<16x152xi32, 2>
```

完整 projection 由 host 连续跑 output chunks。真实 NPU 对拍结果：

```text
q_proj  output=2048 chunks=32 max_abs=2.8610229e-06 mean_ms=435.735
k_proj  output=1024 chunks=16 max_abs=2.8610229e-06 mean_ms=220.341
```

实现细节：

- external function 名字由 op-specific builder 固定。
- link object 由 compile helper 放到 `aiecc` 工作目录。
- `compile_runtime` 必须检查 `xclbin` 和 `insts.bin` 是否真的生成；`aiecc` 可能返回 0
  但缺少 xclbin。

当前性能仍由 correctness-first dot body 主导。这个 recipe 证明正式路径接通，不代表
projection 已经达到可接受吞吐。
