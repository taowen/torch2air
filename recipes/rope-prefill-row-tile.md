# RoPE Prefill Row Tile

## 问题

decode `S=1` 的 RoPE/RMSNorm 可以按 head 切成多列 herd：

```text
herd = [1, 4]
每个 tile 处理一个 head_dim=128
```

但 prefill `S>1` 如果继续让一个 herd tile 在 token/head 两层 loop 里反复做
`L3 -> L1 -> external -> L3`，会产生大量 host-visible channel item。真实 NPU 上常见两种
失败：

- lowering 后 segment 里残留 nested `air.wait_all async`，`air-to-aie` 不能 legalize；
- 即使用更完整的 AIR pass pipeline 通过 lowering，runtime 也可能只推进前几个 head-tile，
  后面的 `normed/output` 为 0。

## 做法

prefill 用 row-parallel tile，而不是 head-tile 粒度：

```text
decode S=1:
  herd = [1, min(head_count, 4)]
  L1 tile = head_dim
  external kernel = one head

prefill S>1:
  herd = [min(S, 4), 1]
  每个 tile 处理 sequence_length / herd_rows 个 token row
  L1 tile = hidden_size
  external kernel = one full row, 在 C++ 里循环所有 heads
```

RMSNorm 的 prefill tile 形态：

```text
source[row, 0:hidden] -> L1 row
weight[0:head_dim]   -> L1 weight
bf16_rms_norm_heads_tile(row, weight, row_out)
row_out              -> normed[row, 0:hidden]
```

RoPE 的 prefill tile 形态：

```text
normed[row, 0:hidden] -> L1 row
rope_lut[row]         -> L1 lut
bf16_rope_heads_tile(row, lut, row_out)
row_out               -> output[row, 0:hidden]
```

## Lowering

`S=1` 保留短 pipeline，避免 aggressive channel fusion 在多列 head shard 上制造 fan-in：

```text
air-dependency
air-dma-to-channel
air-place-herds
air-to-aie
```

`S>1` 需要更接近官方 `aircc` 的 multi-row pipeline，至少包含：

```text
air-dependency
air-hoist-dma-in-accum-pattern
air-broadcast-detection
air-specialize-dma-broadcast
air-dma-to-channel
air-dependency-canonicalize
air-isolate-async-dma-loop-nests
air-fuse-channels
func.func(air-split-l2-memref)
air-isolate-async-dma-loop-nests
func.func(air-fuse-alloc-dealloc)
func.func(air-shrink-memref-sizes-by-access)
func.func(air-opt-memtile-dma-bds)
air-place-herds
air-to-aie
```

不要默认打开 ping-pong transform；先用最小能 clean `air.wait_all` 的 pipeline。ping-pong 会改写
tile loop 和 buffer 生命周期，应该单独验证。

## 检查

- `*.channel.mlir` 里 segment body 不应残留只包含 `air.wait_all` 的空 loop。
- prefill 的 launch-side source/output channel item 数应接近 `S`，不要是 `S * head_count`。
- 对拍先看 `normed`，再看 RoPE output；RMS 阶段已经只写前几个 tile 时，不要继续改 RoPE 数学。
- q/k 要分别跑，因为 `HEAD_COUNT` 不同；external object 要按当前 stage 编译。

## 不要

- 不要把 prefill 的 token/head 双层 loop 直接搬进一个 public L3 writeback channel。
- 不要用 `4x4` 同时切 token 和 head；容易先撞 shim DMA channel 数量。
- 不要把 `air.wait_all` lowering 错误当成唯一问题；它修掉后仍可能暴露 runtime drain
  backpressure。
