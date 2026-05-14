# Static Token Buckets

问题：NPU graph 需要固定 shape，但 prefill token 数是动态的。应该怎样把动态 token
长度落到固定 AIR/NPU graph？

稳定做法是使用固定 token bucket。host 把长序列拆成多个 bucket launch，tail 不足时
pad/mask：

```text
public input:   memref<BUCKETx...>
public weight:  fixed tile weight
public output:  memref<BUCKETx...>
herd:           token lanes x feature lanes
per tile:       small fixed number of tokens
```

验证通过的 Q4_K projection bucket 是 `BUCKET=8`、`herd=4x1`，每个 AIE tile 只连续处理
2 个 token。weight tile 在每个 tile 上搬一次，input 按 token 搬入 L1，调用同一个
1-token external tile body。

真实 NPU 结果：

| token ids | max_abs | mean_ms |
| --- | ---: | ---: |
| `0..7` | `5.9604645e-07` | `27.306` |
| `8..15` | `1.2516975e-06` | `27.174` |

不要把太多 token 串在一个 tile 里：

- `S=8, 1x1 herd` 会在后半 token 出现 0。
- `S=16, 4x1 herd` 会在第三轮 token 出现 0。
- `S=16, 4x2 herd` 把每 tile 限制到 2 token，但 `air-to-aie` 报 shim DMA channels 不够。

结论：prefill 先使用固定 `S=8` graph。更长 context 由 host 分成多个 `S=8` launch；
tail 不足 8 个 token 时 pad/mask，不在 AIR 里引入动态 memref。
