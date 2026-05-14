# Vectorized Quantized Dot

问题：packed Q4_K/Q6_K 权重的 external tile kernel 怎样避免纯标量 dot body？

## 做法

把 bit unpack 和 vector dot 分开：

```text
packed bytes/words -> 16-lane float weight scratch
hidden L1 slice     -> aie::vector<float, 16>
dot accumulation    -> aie::accum<accfloat, 16>
final scalar        -> aie::reduce_add(acc.to_vector<float>())
```

Q4_K 可以按 32-value subblock 处理，每个 subblock 拆成两个 16-lane vector chunk。

Q6_K 可以按 GGML `ql/qh/scales/d` layout 处理，每个 128-value half-block 拆成
`q1/q2/q3/q4` 四组 16-lane vector chunk。

## 规则

- 先保持 AIR schedule 和 weight ABI 不变，只替换 tile-local dot body。
- unpack 可以先写到 tile-local 16-lane scratch，再用 `aie::load_v` 进入 vector。
- accumulation 用 `aie::accum<accfloat, 16>`，不要回到每 lane scalar 累加。
- 性能进一步优化时，再考虑 packed-byte vector unpack 和 compact weight ABI。

## 边界

这种写法提升的是 tile-local dot body。host chunk 数、herd 并行度和 L1 footprint 仍由
Python AIR schedule 和 packed weight ABI 决定。
