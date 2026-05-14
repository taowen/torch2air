# Debug AIR Lowering

问题：AIR/AIE 编译失败、xclbin 能跑但数值错、或多 stage 串起来后状态污染时，应该按什么顺序定位？

## 分层定位

第一层看 AIR graph 和 pass IR：

```bash
AIR_DUMP_DEP_GRAPHS=1 AIR_VERIFY_EACH=1 <run command>
AIR_PRINT_IR_ROOT=.cache/debug-air/ir AIR_VERIFY_EACH=1 <run command>
```

检查 `air.channel.put/get` 是否成对，loop 和 DMA 是否被 pass 改成意外形态。

第二层看 AIE lowering 产物：

```text
input_with_addresses.mlir
*_core_*.ld.script
```

重点查：

- `aie.flow` 方向是否符合 producer/consumer 关系；
- shim DMA channel 是否冲突；
- input task 是 `dma_free_task` 还是需要 `dma_await_task`；
- stack、L1 buffer、DMA BD 和 lock 的地址是否互相挤压；
- bank-aware allocation warning 是否对应后续数值异常。

第三层把 pipeline 缩成 standalone stage，用同一份真实输入对拍。score 或中间量对但 output
错时，优先查 DMA 等待和 L1 状态生命周期，而不是先改数学近似。

第四层必要时临时让 external kernel 写回中间量，例如 score、row max、row sum、scale 或
output debug slot。确认后删除 debug ABI。

## 结论

- 能编译不代表运行正确；最终以真实 NPU 对拍为准。
- 输出后半段变 0、重复旧行、或像复用旧 accumulator 时，优先看 L1 地址和 runtime DMA
  ordering。
- stack 太小或 L1 bank 分配失败，可能表现为静默状态污染。
- 不要 catch 后静默吞异常，不要只看最终 output，也不要把 debug output 留在正式 ABI。
