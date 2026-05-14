# Herd Scalar Accumulator

## 问题

在 AIE herd 内用 `scf.for iter_args` 做标量累加，AIR/AIE 编译可以通过，但真实 NPU
运行时可能得到错误状态。典型表现是 reduction 结果保持初值，后续输出只反映了
非 reduction 分支的计算。

## 做法

tile 内标量状态放在 L1 memref：

```python
sum_l1 = AllocOp(MemRefType.get([1], f32, memory_space=l1_space), [], [])
store(arith.constant(f32, 0.0), sum_l1, [idx(0)])

for dim_i in range_(hidden_size):
    acc = load(sum_l1, [idx(0)])
    value = load(hidden_l1, [idx(0), dim_i])
    store(arith.addf(acc, arith.mulf(value, value)), sum_l1, [idx(0)])
    yield_([])

sum_squares = load(sum_l1, [idx(0)])
```

## 结论

- herd 内跨 loop 的标量状态优先用 L1 memref 表达。
- `scf.for iter_args` 可以用于 AIR graph 编排 token，但不要先用于核心数值累加。
- 数值表现为只使用初始值时，优先怀疑 reduction 状态没有传出来。
