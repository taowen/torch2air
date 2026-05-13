# torch2air Architecture

`torch2air` should be the AIR backend sibling of `torch2vk`.

The frontend shape is intentionally simple:

```text
PyTorch nn.Module
  -> torch.export / FX graph
  -> aten op or small-pattern lookup table
  -> AIR variant templates
  -> mlir-air tools
```

The main difference from `torch2vk` is the backend artifact:

```text
torch2vk:  aten node -> ShaderVariant -> GLSL / Vulkan dispatch
torch2air: aten node -> AirVariant    -> tiled MLIR-AIR / AIR runtime dispatch
```

This project should not be a second generic graph compiler. It should export
real PyTorch modules, inspect the exported aten graph, and replace each known op
or fused pattern with an AIR implementation.

## Repository Shape

Planned layout:

```text
torch2air/
  mlir-air/                         # future git submodule: Xilinx/mlir-air
  docs/
    architecture.md
  src/torch2air/
    export/
      __init__.py
      graph.py                      # torch.export / FX graph normalization
      registry.py                   # aten target -> AirVariant factory
      air_codegen.py                # dispatch + AIR file emission
      templates/                    # textual MLIR-AIR templates
    weights/
      gguf.py                       # exact packed weight metadata/readers
    runtime/
      xrt.py                        # host-side AIR/XRT loading
  models/
    quantized_qwen3/
      export.py
      generated/
  scripts/
    build-air-tools.sh
    export-quantized-qwen3.sh
    run-quantized-qwen3-air.sh
```

Only this architecture document exists after the repository reset. The layout
above is the target shape for the next implementation pass.

## Export API

The model-level export should stay close to the existing `torch2vk` pattern:

```python
export_one(
    "run_text_layer",
    text_model.layers[0],
    args=(torch.zeros(1, prompt_len, hidden_size, device="meta"),),
    kwargs={
        "position_embeddings": (
            torch.zeros(1, prompt_len, head_dim, device="meta"),
            torch.zeros(1, prompt_len, head_dim, device="meta"),
        ),
        "past_key_values": None,
        "attention_mask": None,
    },
    weight_prefix="model.layers.0.",
    kv_inject=KVCacheInjectHint(phase="prefill", max_seq_len=max_seq),
    reference_tensors="model_tensors().text_layers[layer_idx]",
    reference_name="qwen3.prefill.layer.{layer_idx}",
    export_registry=Q4_K_M_REGISTRY,
    weight_quantization=quantized_weights,
    shape_exprs=layer_shape_exprs,
)
```

`torch.export` is used to discover the aten graph, shapes, names, and call
boundaries. It is not expected to express packed GGUF math exactly. Packed
weight handling belongs to the AIR variant selected from the registry.

## Registry

`torch2air.export.registry` should mirror `torch2vk.export.registry`.

Conceptual API:

```python
@dataclass(frozen=True, slots=True)
class AirBinding:
    target: str
    factory: Callable[[torch.fx.Node, AirContext], AirVariant | None]


class AirRegistry:
    def resolve(self, node: torch.fx.Node, context: AirContext) -> AirVariant | None:
        ...
```

Default registry:

```python
DEFAULT_REGISTRY = AirRegistry(
    [
        AirBinding("aten.linear.default", make_linear_variant),
        AirBinding("aten.mul.Tensor", make_elementwise_variant),
        AirBinding("aten.add.Tensor", make_elementwise_variant),
        AirBinding("aten.mean.dim", make_reduce_variant),
        AirBinding("aten.rsqrt.default", make_elementwise_variant),
        AirBinding("aten.silu.default", make_elementwise_variant),
        AirBinding("aten.scaled_dot_product_attention.default", make_sdpa_variant),
        AirBinding("aten.embedding.default", make_embedding_variant),
    ]
)
```

Quantized registries override only the ops whose implementation changes:

```python
Q4_K_M_REGISTRY = AirRegistry(
    [
        AirBinding("aten.linear.default", make_q4_k_m_linear_variant),
        AirBinding("aten.embedding.default", make_q4_k_m_embedding_variant),
        # Remaining bindings can reuse DEFAULT_REGISTRY factories.
    ]
)
```

This is the important simplification: `torch2air` does not need a separate
"model IR" and "logical linalg IR" layer before AIR. The exported aten graph is
the control structure, and the registry maps nodes or small patterns to backend
variants.

## AirVariant

`AirVariant` is the AIR equivalent of `ShaderVariant`.

It should carry:

- `name`: stable generated artifact name
- `target`: matched aten target or fused pattern name
- `inputs` and `outputs`: buffer contracts
- `weight_bindings`: disk weight names and packed formats
- `template`: textual MLIR-AIR template path or renderer
- `tile_config`: tile sizes, herd shape, memory spaces, and buffering mode
- `runtime_entry`: symbol name used by the host runner

Example:

```python
AirVariant(
    name="q4_k_m_linear_1x4096x4096_tile8",
    target="aten.linear.default",
    inputs=[AirBuffer("x", "1xSx4096xf16")],
    outputs=[AirBuffer("y", "1xSx4096xf16")],
    weight_bindings=[
        PackedWeight("weight", "model.layers.0.self_attn.q_proj.weight", "Q4_K_M"),
    ],
    template="templates/q4_k_m_linear.air.mlir.j2",
    tile_config=TileConfig(m_tile=1, n_tile=8, k_tile=1024, cols=8),
    runtime_entry="q4_k_m_linear",
)
```

The AIR template is allowed to contain explicit tiling, `air.herd`, local
buffers, and DMA operations. That is equivalent to a `torch2vk` shader carrying
its own workgroup layout and memory-access strategy.

## Generated Files

For one exported function, generate a small manifest and the AIR artifacts:

```text
models/quantized_qwen3/generated/
  run_text_layer.graph.json
  run_text_layer.manifest.json
  run_text_layer_000_q4_k_m_linear.tiled.mlir
  run_text_layer_000_q4_k_m_linear.air.mlir
  run_text_layer_001_rmsnorm.tiled.mlir
  run_text_layer_001_rmsnorm.air.mlir
```

`.graph.json` records the normalized aten graph for debugging.

`.manifest.json` records inputs, outputs, weights, AIR files, and runtime entry
symbols.

`.tiled.mlir` / `.air.mlir` are the backend artifacts. In simple cases the
variant can emit `.air.mlir` directly. When a helper pass is useful, it can emit
`.tiled.mlir` first and run `air-opt`.

## Tiling Policy

Tiling should live in each AIR variant, not in a global optimizer.

For example, Q4_K linear can use an AIR template that already knows:

```text
activation tile: f16 [sequence_tile, k_tile]
weight tile:     packed Q4_K_M bytes for [n_tile, k_tile]
output tile:     f16 [sequence_tile, n_tile]
herd shape:      columns/rows selected for the target NPU
copy schedule:   global -> local, compute, local -> global
```

This is still the same style as `torch2vk`: the backend implementation owns the
low-level schedule. The registry only decides which implementation a node gets.

## Quantized Qwen3 First

The first useful target should be `models/quantized_qwen3`.

Initial scope:

- Export one real `nn.Module` layer with `torch.export`.
- Load real GGUF metadata and packed weight names.
- Use `Q4_K_M_REGISTRY` for text layer linear/embedding ops.
- Emit AIR variants that consume packed bytes on the NPU path.
- Compare against PyTorch ROCm at the module boundary.

No host-side dequantization should be part of the AIR execution path. Host code
may package buffers and run reference PyTorch ROCm, but the NPU kernel should see
the same packed weight format that is on disk.

## Compile And Run

The project should use MLIR-AIR tools directly:

```bash
scripts/export-quantized-qwen3.sh
scripts/compile-air.sh models/quantized_qwen3/generated/run_text_layer.manifest.json
scripts/run-quantized-qwen3-air.sh
```

Expected toolchain pieces come from the future `mlir-air/` submodule and its
dependencies:

- `air-opt`
- `air-translate`
- `aircc.py`
- XRT/AIR runtime pieces needed by the selected board

IREE `vmfb`, IREE HAL dispatch formation, and IREE encoding dialect are not the
core path for this project.

## Non-Goals

- Do not build a full PyTorch-to-AIR automatic optimizer.
- Do not maintain a large hand-written model description language.
- Do not require every op to pass through linalg first.
- Do not dequantize GGUF weights on the host for NPU execution.
- Do not make `iree-amd-aie` the project root.

The practical model is: export PyTorch, match aten, emit AIR.
