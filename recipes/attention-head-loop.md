# Attention Head Loop

问题：一个 attention xclbin 要处理多个 head 时，head 维度应该展开成多份 tile 程序，
还是放进运行时循环？

结论：host 侧可以静态展开 head 来生成常量 DMA offset；AIE tile 侧应该用
`scf.for %head` 运行时循环消费 channel。不要把每个 head 的 compute body 复制进同一个
tile 程序。

原因：

- head 静态展开会线性放大 tile program memory。
- head 的 compute body 不依赖 head id，只依赖按顺序收到的 Q/K/V tile。
- host 静态 offset 能避开动态 DMA offset 在 runtime lowering 中变成非预期整数类型。

推荐形态：

```mlir
// host side: generate constant offsets per head, then put/get channel tiles
%q_col_0 = arith.constant 0 : index
%q_col_1 = arith.constant 128 : index

// tile side: one program body, loop over heads
%heads_done = scf.for %head = %c0 to %q_heads step %c1
    iter_args(%token = %alloc_done) -> (!air.async.token) {
  %done = scf.for %q_block = %c0 to %seq step %q_tile
      iter_args(%q_token = %token) -> (!air.async.token) {
    // ChannelGet Q, ChannelGet K/V, call external tile body, ChannelPut output.
    scf.yield %output_put : !air.async.token
  }
  scf.yield %done : !air.async.token
}
```

已验证边界：

```text
16 Q heads / 8 KV heads / 16 tokens: real NPU PASS
16 Q heads / 8 KV heads / 64 tokens: real NPU PASS
static head body replication at 12 heads: tile program memory overflow
```
