# Export Aten Sequence

问题：一个模型片段导出后由多个 aten op 组成，应该怎样接到 AIR，而不是先手写一个 fused
语义 kernel？

## 做法

让 generated export 保留原始 op 顺序：

```text
define_tensor(...)
emit_kernel("aten.pow.Tensor_Scalar", ...)
emit_kernel("aten.mean.dim", ...)
emit_kernel("aten.add.Tensor", ...)
...
mark_output(...)
```

AIR builder 负责把每个 `emit_kernel` 映射到本地 op emitter。`reshape`、`to`、`unsqueeze`
这类只改变 view 或 dtype policy 的节点可以作为 alias 处理；真正产生数值的 op 才分配 L1
临时 buffer 并发出计算。

## 规则

- 不把 `ExportedProgram` 包成新的 graph 对象。
- 不新增 `graph.json`、manifest 或 registry 作为中间层。
- 不把一串 aten op 改名成新的 fused op；即使最终落到同一个 AIR module，源码结构也要保持
  op-by-op emitter。
- 每新增一种 aten op，先用真实 NPU 对拍通过，再考虑复用到别的 stage。

## 适用边界

RoPE 这类由 `pow/mean/add/rsqrt/mul/slice/neg/cat` 组成的片段适合先按 export sequence
落地。等完整 attention 路径正确后，再根据真实性能数据决定是否做 fusion。
