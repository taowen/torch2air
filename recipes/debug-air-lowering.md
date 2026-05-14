# Debug AIR Lowering

## 问题

NPU 问题通常不是一上来就改数学 kernel 能解决。常见失败：

- routing 或 xclbin 编译失败；
- xclbin 能跑但数值错；
- standalone stage 能过，pipeline 串起来后状态污染。

## 步骤

第一层看 AIR graph 和 pass IR：

```bash
AIR_DUMP_DEP_GRAPHS=1 AIR_VERIFY_EACH=1 <run command>
AIR_PRINT_IR_ROOT=.cache/debug-air/ir AIR_VERIFY_EACH=1 <run command>
```

检查 `air.channel.put/get` 是否成对、loop 和 DMA 是否被 pass 改成意外形态。

第二层看 AIE lowering：

```text
air_project/input_with_addresses.mlir
air_project/*_core_*.ld.script
```

检查：

- `aie.flow` 方向是否正确；
- `aie.shim_dma_allocation` 是否冲突；
- input task 是 `dma_free_task` 还是 `dma_await_task`；
- stack 和 L1 buffer 地址是否太近。

第三层把完整 pipeline 缩成 standalone stage，在真实输入上和 PyTorch ROCm 对拍。

第四层必要时临时让 external kernel 写回中间量，例如 score、max、softmax weight、row sum。
确认后删除 debug 输出。

## 结论

- score 对但 output 错，常见原因是 DMA 等待或状态 buffer 生命周期，而不是 dot product。
- 能跑但数值错时，回读中间 BO 或 output debug slot 比猜 tile size 更快。
- stack 太小会表现为 L1 状态被静默污染。

## 不要

- 不要 catch 后静默吞异常。
- 不要只看最终 output。
- 不要把临时 debug kernel 提交进正式路径。
