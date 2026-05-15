# Debug AIR Lowering

问题：AIR/AIE 编译失败、xclbin 能跑但数值错、或多 stage 串起来后状态污染时，应该按什么顺序定位？

## 分层定位

第零层先确认不是环境问题：

```bash
source scripts/npu-common.sh
check_npu_device
command -v aircc
aircc --help | head
```

`aircc` 必须是真实 binary，不能是 `.venv/bin/aircc` 这种坏 wrapper。使用
Makefile 时还要确认 Python shim 没有改变 build cwd，否则 `link_with` 的 `.o` 会放在
一个目录、backend 在另一个目录找。

第一层用官方风格拆成可打印、可编译、可运行的入口：

```bash
make print
make run COMPILE_MODE=compile-only
make run
```

没有 Makefile 时也保留同等入口：`--print-module-only`、`--compile-mode compile-only`、
`compile-and-run`。先让同类最小算子在真实 NPU 上 PASS，再调自己的变体。

第二层看 AIR graph 和 pass IR：

```bash
AIR_DUMP_DEP_GRAPHS=1 AIR_VERIFY_EACH=1 <run command>
AIR_PRINT_IR_ROOT=.cache/debug-air/ir AIR_VERIFY_EACH=1 <run command>
```

检查 `air.channel.put/get` 是否成对，loop 和 DMA 是否被 pass 改成意外形态。

第三层看 AIE lowering 产物：

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

第四层检查最终 xclbin，而不是只看 `aiecc/` 临时目录：

```bash
/opt/xilinx/xrt/bin/xclbinutil \
  --dump-section AIE_PARTITION:JSON:/tmp/aie_partition.json \
  --input run.xclbin

rg -n 'uuid|file_name' /tmp/aie_partition.json
```

multi-launch lowering 可能生成 `main.pdi`、`rms_seg.pdi`、`rope_seg.pdi` 多个 PDI，但最终
xclbin 只剩最后写入的 `main.pdi`。`insts.bin` 里有 `aiex.configure @segment` 不代表
segment PDI 已经被打包。

`aiecc --aie-generate-xclbin` 依赖 `xclbinutil`。运行编译前确认：

```bash
command -v xclbinutil
```

如果 `PATH` 没有 `/opt/xilinx/xrt/bin`，可能只生成 `insts.bin`，不生成 xclbin。

第五层把 pipeline 缩成 standalone stage，用同一份真实输入对拍。score 或中间量对但
output 错时，优先查 DMA 等待和 L1 状态生命周期，而不是先改数学近似。

第六层做最小探针，按顺序打开能力：

```text
L3->L1->L3 copy
动态 side input copy
一个标量/向量算术
一个 external kernel call
完整 tile body
```

如果 copy 过、external 过，但 scalar AIR 组合失败，说明问题在 IR lowering 或 L1
读写时序，不要继续扩大正式 kernel。

第七层必要时临时让 external kernel 写回中间量，例如 score、row max、row sum、scale
或 output debug slot。确认后删除 debug ABI。

RoPE/head-major 输出可以直接看每个 head 是否写回：临时脚本打印每个 head 的非零数、
最大误差和前几个值即可，不要把这种 debug ABI 留在正式路径里。

后半 head 为 0、只有 head 0/1 非零、或者命令 timeout，优先查 PDI packaging 和 runtime
sequence drain。

## 结论

- 能编译不代表运行正确；最终以真实 NPU 对拍为准。
- 官方 demo 的调试入口是 `print`、`compile-only`、`run`、`profile`、`verify` 这几层，
  自己的 kernel 也应该保留同等层次。
- 组合 pipeline 要暴露稳定中间边界做对拍，例如 normed、q/k/v、q_roped/k_roped、output。
- 输出后半段变 0、重复旧行、或像复用旧 accumulator 时，优先看 L1 地址和 runtime DMA
  ordering。
- multi-launch 还要额外检查最终 xclbin 的 AIE_PARTITION。`.npu.mlir` 和 `insts.bin`
  正确，不代表 compute PDI 已经加载。
- stack 太小或 L1 bank 分配失败，可能表现为静默状态污染。
- 不要 catch 后静默吞异常，不要只看最终 output，也不要把 debug output 留在正式 ABI。
