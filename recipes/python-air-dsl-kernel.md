# Python AIR DSL Kernel

## 问题

`.py` kernel 有两种完全不同的含义：

- Python 代码拼接 MLIR 字符串；
- Python 代码调用 MLIR-AIR DSL 构造 IR。

前者只是把手写 MLIR 包了一层 Python，不利于后续把 aten op 映射成真正的
torch2air kernel。

## 做法

正式 kernel 用 MLIR-AIR Python API 表达结构：

```python
@module_builder
def build():
    @FuncOp.from_py_func(input_type, output_type, name="run_stage")
    def run_stage(input_l3, output_l3):
        @launch(operands=[input_l3, output_l3])
        def launch_body(input_arg, output_arg):
            @segment(name="seg", operands=[input_arg, output_arg])
            def segment_body(input_seg, output_seg):
                @herd(name="stage", sizes=[1, 4], operands=[input_seg, output_seg])
                def herd_body(_tx, tile_j, _sx, _sy, input_tile, output_tile):
                    ...
```

L3/L1 交接用 `dma_memcpy_nd`，L1 buffer 用 `AllocOp`，tile 内标量计算用
`arith`、`memref.load`、`memref.store`。

## 结论

- `.air.mlir` 仍然会生成，但只是 `.cache` 下的调试和编译输入。
- 源码里不要维护 MLIR 字符串 renderer。
- 一个 aten op 对应一个 Python DSL kernel，builder 只负责从导出对象传入
  tensor metadata 和 aten 调用。

## 不要

- 不要把 `.py` 文件写成大段 `lines.append("...mlir...")`。
- 不要保留只转发到 `builder.emit_kernel` 的 wrapper 层。
