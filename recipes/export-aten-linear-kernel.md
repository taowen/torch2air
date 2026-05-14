# Export Aten Linear Kernel

问题：已经验证过的 AIR kernel，怎样接回正式 `torch2air.export`，同时不重新包装 PyTorch
导出的对象？

## 做法

保持一层映射：

```text
torch.export ExportedProgram
  -> generated Python export
  -> op-specific AIR builder
  -> AIR schedule + optional external tile body
```

generated export 只定义 tensor metadata 并发出原始 aten op：

```python
builder.define_tensor("input", shape=(1, 1, 1024), dtype="float32")
builder.define_tensor("weight", shape=(2048, 1024), dtype="float32")
builder.emit_kernel("aten.linear.default", output="output", inputs=("input", "weight"))
```

op-specific builder 再决定：

- public ABI；
- fixed shape；
- herd 尺寸；
- L3/L1 DMA；
- external tile function 名字和 link object。

## 规则

- 不要把 `ExportedProgram` 再包装成 torch2air graph。
- 不要写 `graph.json` 或 manifest 作为中间层。
- external object 由 compile helper 放到 `aiecc` 工作目录。
- `compile_runtime` 必须显式检查 `xclbin` 和 `insts.bin` 是否生成。
- prefill/decode 这类 shape 差异用不同固定 bucket 表达，不在 AIR public ABI 里引入动态
  memref。

## 适用边界

这个模式适合一个 exported aten op 对应一个本地 AIR kernel 的场景。复合 op 如果会拆成太多
launch，应先设计 fused stage kernel，而不是把内部每个 aten 都直接变成 production stage。
