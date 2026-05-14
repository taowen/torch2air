# External Kernel Tile ABI

问题：Python AIR 负责 tiling 和 DMA，tile compute body 放到 AIE external kernel 时，
MLIR 和 object file 应该怎么对接？

稳定写法：

```mlir
func.func private @tile_body(
    memref<1x1024xf32, 2 : i32>,
    memref<16xi32, 2 : i32>,
    memref<1x16xf32, 2 : i32>)
    attributes {link_with = "tile_body.o", llvm.emit_c_interface}
```

AIR herd 内只做 L3/L1 搬运和调用：

```text
L3 input -> L1 memref<..., 2>
func.call @tile_body(...)
L1 output -> L3 output
```

`link_with` 的 object file 要放在 `aiecc` 的工作目录里。external C++ 函数使用扁平
pointer ABI：

```c++
extern "C" void tile_body(float *hidden, int32_t *weight, float *output);
```

关键 lowering 经验：

- `air-to-aie` 会把 external func declaration 复制进 AIE core。
- `air-to-std` 后必须跑 `symbol-dce`，移除 top-level private declaration。
- 不跑 `symbol-dce` 时，`aiecc` 会看到两个同名 `func.func private @tile_body`，报
  redefinition。

验证记录：

```text
hidden  memref<1x1024xf32, 2>
weight  memref<16xi32, 2>
output  memref<1x16xf32, 2>
max_abs 2.3841858e-07
mean_ms 0.675
```

这个记录只验证 external kernel ABI 和 link 流程。具体算子可以替换 tensor shape、
object file 名字和 tile body 函数名。
