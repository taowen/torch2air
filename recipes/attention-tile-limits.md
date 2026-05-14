# Attention Tile Limits

问题：attention 的 Q tile 和 K/V tile 应该先选多大？

结论：当前先用 `KEY_TILE_ROWS=8`，`QUERY_TILE_ROWS` 不超过 32。更大的 Q tile
会撞 64 KiB tile-local memory；更大的 K/V tile 已经观察到真实 NPU 数值错误。

已验证结果：

```text
Q tile rows 4, 8, 16: real NPU PASS, 16-token 性能几乎不变
Q tile rows 32: real NPU PASS, 64-token 性能几乎不变
Q tile rows 64: compile FAIL, local buffers exceed tile memory

K/V tile rows 4: real NPU PASS
K/V tile rows 8: real NPU PASS, 16-token 略快
K/V tile rows 16: run FAIL, output contains stale zero values
```

L1 估算：

```text
stack: 4096 bytes
Q buffer: QUERY_TILE_ROWS * 128 * sizeof(f32)
O buffer: QUERY_TILE_ROWS * 128 * sizeof(f32)
K buffer: KEY_TILE_ROWS * 128 * sizeof(f32)
V buffer: KEY_TILE_ROWS * 128 * sizeof(f32)
row state: 2 * QUERY_TILE_ROWS * sizeof(f32)
```

规则：

- `SEQUENCE_LENGTH` 必须能整除两个 tile rows。
- 可编译不代表可用，K/V tile 需要真实 NPU 对拍确认。
- 如果只增大 Q tile 但 external kernel 仍然逐 row 计算，速度不会显著变化。
