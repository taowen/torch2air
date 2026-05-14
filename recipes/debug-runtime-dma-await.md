# Debug Runtime DMA Await

问题：standalone kernel 的 dot product 或 score 已经对齐，但完整 NPU 输出缺一段、
变成 0，或像复用了旧 accumulator，应该先查什么？

优先查 runtime sequence 里的输入 DMA 是否真的等待完成。复杂 kernel 会在同一条 FIFO
或同一个 L1 tile 上反复投递输入 tile；如果 host runtime 只 free task，不 await task，
external kernel 可能在最后一个输入 tile 到达前就开始消费。

定位顺序：

- 先把完整 pipeline 缩成 standalone kernel，用同一份真实输入复现。
- 临时让 external kernel 写回 `score0..3`、`tile_max`、`weight0..3` 或输出前几个元素。
- 如果 score 对但 output 少了一段贡献，检查 `airrt-to-npu` 后的 runtime sequence。
- 在 `aiecc/input_with_addresses.mlir` 里确认 input task、lock、DMA BD 和 L1 buffer 地址。

稳定经验：

- 对反复消费同一路输入 tile 的 kernel，输入 task 需要显式 await。
- 这个问题不应该靠改数学近似或扩大 tolerance 解决。
- debug ABI 只留在 `.cache` 实验里，确认后正式 kernel 仍保持简单 stage ABI。
