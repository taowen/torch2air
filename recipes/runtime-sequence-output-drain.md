# Runtime Sequence Output Drain

## 问题

single herd 里同时写多个 host-visible L3 output，例如：

```text
input -> compute A -> normed BO
      -> compute B -> output BO
```

如果 `normed` 和 `output` 都是 public function 参数，AIR lowering 会给它们生成独立
S2MM channel。runtime sequence 可能先 drain 完一个 output BO，再 drain 另一个 output BO。

## 现象

典型表现：

```text
normed 只写前几个 tile/head
output 全 0 或只写前几个 tile/head
```

这不是数学误差。先看中间 BO 的 nonzero 分布，而不是只看最终 max diff。

`q_norm_rope` 的 single-herd 融合实验里，RMSNorm 和 RoPE 放在同一个 herd，`normed` 和
`output` 都作为 public BO 写回。真实 NPU 上 `normed` 只推进到 head 0/1，`output` 全 0。
这说明 core 被写回/等待顺序卡住，而不是 RoPE 数学或 bf16 external kernel 本身算错。

## 机制

core 每次循环通常需要：

```text
input DMA -> compute -> release normed writeback
          -> compute -> release output writeback
```

如果 runtime sequence 把 `normed` 的多个 `aiex.dma_await_task` 全排在 `output` 的
`aiex.dma_await_task` 前面，core 会被后一个 output buffer 的 backpressure 卡住。
于是前一个 BO 只推进少数 tile，后一个 BO 没有机会写回。

## 检查

在 `*.npu.mlir` 或 `input_with_addresses.mlir` 里查：

```text
aie.shim_dma_allocation @air_channel_normed(..., S2MM, ...)
aie.shim_dma_allocation @air_channel_output(..., S2MM, ...)
aie.runtime_sequence @...
  aiex.dma_configure_task_for @air_channel_normed
  ...
  aiex.dma_configure_task_for @air_channel_output
```

如果一个 S2MM channel 的 16 个 get 全在另一个 S2MM channel 前面，就不要假设它们会和
core loop 自动交错。

同一个 S2MM channel 里 item 太细也会放大 backpressure。prefill RoPE/RMSNorm 如果按
`token * head` 发 128 个 host-visible item，可能只写前几个 head-tile；把 tile 粒度提升成
一整个 token row，可以把 public channel item 降到 `S` 个。

还要查 public function ABI：只要中间量作为 public memref 参数出现，它就会变成 host-visible
BO 和 runtime sequence 的 drain 对象。源码里两个 `dma_memcpy_nd` 相邻，不代表 host drain
会按这个顺序逐 tile 交错。

## 做法

- 正式 pipeline 尽量只保留一个 host-visible final output。
- 需要 stage 间交接时，优先用 tile-to-tile channel，不要先写 public BO 再读回来。
- 需要 debug 中间量时，把 debug BO 当临时 ABI，用完删除；不要把 debug ABI 当生产数据通路。
- 如果必须保留 host-visible 中间 BO，把 stage 拆成不同 host runtime boundary。每个 boundary
  用自己的 xclbin/insts，host 显式串起来。

## 不要

- 不要在同一个 herd loop 里长期保留多个大 L3 写回 BO。
- 不要只凭 AIR graph 认为两个 L3 写回会按源码顺序 drain。
- 不要把前几个 head 正确解释成数值误差；这通常是 DMA/writeback 进度问题。
- 不要用“single herd 里多写一个 debug BO”来绕过 multi-launch PDI packaging 问题；它会把
  问题换成 S2MM drain/backpressure 问题。
