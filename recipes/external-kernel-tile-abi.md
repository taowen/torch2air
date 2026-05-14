# External Kernel Tile ABI

问题：Python AIR 负责 tiling 和 DMA，tile compute body 放到 AIE external kernel 时，MLIR
和 object file 应该怎么对接？

## MLIR 形态

在 AIR module 里声明 private function：

```mlir
func.func private @tile_body(
    memref<1x1024xf32, 2 : i32>,
    memref<16x152xi32, 2 : i32>,
    memref<1x16xf32, 2 : i32>)
    attributes {link_with = "tile_body.o", llvm.emit_c_interface}
```

AIR herd 内只表达调度：

```text
L3 input  -> L1 input
L3 weight -> L1 weight
func.call @tile_body(...)
L1 output -> L3 output
```

## C++ ABI

external C++ 函数使用扁平 pointer ABI：

```c++
extern "C" void tile_body(float *input, uint32_t *weight, float *output);
```

object file 必须放在 `aiecc` 工作目录，文件名要和 `link_with` 一致。

## Lowering 规则

- `air-to-aie` 会把 external func declaration 复制进 AIE core。
- `air-to-std` 后跑 `symbol-dce`，避免 top-level private declaration 和 core 内 declaration
  重名。
- Python AIR 仍然拥有 stage boundary、tile shape、memory spaces 和 DMA；`.cc` 文件只写
  tile-local compute body。
