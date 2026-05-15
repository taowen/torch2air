# Host Stage Boundary

## 问题

两个 stage 之间的中间量必须是显式 BO，例如：

```text
launch 1: source + norm_weight -> normed
launch 2: normed + rope_lut -> output
```

`normed` 既要能单独对拍，也要作为下一阶段输入。

## 做法

把它建模成两个 host runtime boundary，而不是一个 public function 里的两个
`air.launch`：

```text
xclbin A / insts A: source + norm_weight -> normed
xclbin B / insts B: normed + rope_lut -> output

host:
  run A
  validate or reuse normed BO
  run B
```

每个 xclbin 内只保留一个 compute segment 和一个 final public output。

## 适用场景

- 中间 BO 是稳定 ABI，需要被 host 保存、复用或对拍。
- multi-launch lowering 生成多个 `aie.device/PDI`，但当前 packaging 路径不能可靠合成一个
  可运行 xclbin。
- single herd 融合后出现多个 public S2MM 写回，runtime sequence drain 顺序导致只写前几个
  tile/head。

## 不适用

如果中间量只是同一 xclbin 内的 producer/consumer handoff，不要拆 host boundary。优先用
同一个 `air.segment` 内的 AIR channel 传 L1 tile，让 host 只看到最终 output。

## 检查

- 每个 stage 的最终 xclbin `AIE_PARTITION` 只需要包含自己的 compute PDI。
- host 对拍先检查 stage A 的 `normed`，再检查 stage B 的 `output`。
- 不要用一个 `func(...)[i]` 的返回下标来掩盖 ABI 不清晰；stage 函数的输入输出应该和
  runtime boundary 一一对应。
