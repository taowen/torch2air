# torch2air Architecture

`torch2air` is the AIR backend workspace for PyTorch-exported models. Keep it
close to how `torch2vk` actually works: model export scripts call a local
`export_one(...)` helper several times with concrete modules and arguments.
That helper uses `torch.export`, walks the exported PyTorch object directly, and
emits the small Python file consumed by the next compile/run step.

Do not build an intermediate graph format, registry system, manifest layer, or
runtime abstraction until a concrete model export proves that one is needed.

## Shape

The intended flow is:

```text
models.quantized_qwen3.export
  -> export_one(name, module, args, kwargs, ...)
  -> torch.export.export(...)
  -> iterate exported_program.graph_module.graph.nodes
  -> generate a small Python AIR kernel file
  -> compile that Python kernel to AIR MLIR / AIE / xclbin
  -> run with existing XRT tooling
```

`torch.export.ExportedProgram` is the source of truth. Do not wrap it in a
torch2air graph object. If code needs nodes, it should iterate:

```python
program = torch.export.export(module, args, kwargs=kwargs, strict=False)
for node in program.graph_module.graph.nodes:
    ...
```

## Repository Layout

Current project layout should stay small:

```text
src/
  torch2air/
    export/
      program.py            # direct ExportedProgram traversal and renderer
      builder.py            # tiny generated-export builder protocol
      air_dsl.py            # narrow AIR Python DSL helpers
      q4k_embedding.py      # op-specific Python AIR builder
      q4k_linear.py         # Q4_K linear Python AIR builder
      q6k_linear.py         # Q6_K linear Python AIR builder
      kernels/              # narrow external tile compute bodies
    weights/
      gguf.py               # GGUF tensor metadata and packed reads
  models/
    quantized_qwen3/
      export.py             # model-specific export_one(...) calls
      reference_runtime.py  # small ROCm check/debug helpers
      run_embed_tokens.py   # embed_tokens XRT execution and reference check
      run_input_layernorm.py
      run_linear.py         # Q4_K/Q6_K linear external-kernel XRT execution
      generated/            # ignored generated files
scripts/
  install-air-tools.sh
  install-python-deps.sh
  export-quantized-qwen3.sh
  run-quantized-qwen3-npu.sh
```

Avoid adding `torch2air.export.graph`, `torch2air.export.registry`, or
`torch2air.runtime.xrt` unless implementation pressure makes the missing module
obvious.

## Export Style

Each model owns its export script. The script may define a local helper like:

```python
def export_one(
    name: str,
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | None] | None = None,
    *,
    weight_prefix: str = "",
    shape_exprs: dict[int, str] | None = None,
) -> None:
    program = torch.export.export(module, args, kwargs=kwargs, strict=False)
    for node in program.graph_module.graph.nodes:
        # Inspect aten target, tensor_meta, args, kwargs directly.
        ...
    # Render the generated Python AIR kernel directly from this context.
```

Then the model export calls it explicitly:

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
    shape_exprs={prompt_len: "sequence_length"},
)
```

This keeps decisions local and visible. If Qwen3 needs special handling for a
specific aten op, add that handling near the model exporter first. Extract a
shared helper only after the second real model needs the same code.

## Generated Files

Generate only files that are consumed by the next step. Do not write
`graph.json` or `export_manifest.json` just because they are convenient debug
containers.

For early Qwen3 work, acceptable generated files are concrete implementation
artifacts, for example:

```text
src/models/quantized_qwen3/generated/
  run_q_proj.py
```

The generated model artifact is a small Python function that defines tensors
and emits the exported aten operator directly. It is not wrapped into a
torch2air graph object. The runtime runner chooses the concrete AIR builder for
that operator, so one exported aten op maps to one local kernel implementation.

Some kernels need the official AIR external-kernel style earlier than the
pure-MLIR body is practical. `quantized_qwen3` attention projections follow
that pattern with `q4k_linear_tile` and `q6k_linear_tile`: the Python AIR
builder emits launch/segment/herd, L3<->L1 DMA, and private `func.func`
declarations carrying `link_with = "q4k_linear.o"` or
`link_with = "q6k_linear.o"`. The Peano objects own only the tile-local compute
body.

Reusable operator bodies should be real Python AIR DSL kernels. The model
package chooses a stage and passes concrete Qwen3 shapes, tensor names, and
module metadata; it should not own a growing library of generic operator
templates.

Do not put model-stage tiling in ad hoc `.cc` files. Native code is acceptable
only as a narrow external tile compute body, kept under
the operator implementation, with the Python AIR DSL still expressing the stage
boundary, tile shape, memory spaces, and DMA.

If a debug dump is useful while developing, make it opt-in and keep it out of
the default export path.

## Weights

GGUF packed weights are real inputs to the AIR path. `torch2air.weights.gguf`
owns reading GGUF metadata and packed byte ranges. The export path should record
or reference GGUF tensor names only where needed by generated code.

Do not host-dequantize quantized weight values for NPU execution. The NPU path
should consume packed GGUF blocks and decode quantized values inside the
generated tile body.

The current Q4_K/Q6_K runners have one temporary exception: block-level f16
scale values are decoded on the host into small f32 side inputs
(`block_f16_scales` for `embed_tokens`, appended sidecar words for attention
projection kernels). The NPU still reads real packed Q4_K/Q6_K quantized values
and decodes subblock scales/mins and q nibbles/bits in the tile body. The Q6_K
path also temporarily widens GGUF halfwords to i32 words for the external kernel
ABI. Remove these exceptions once f16 scalar conversion and compact i16 L1 DMA
are validated.

Host-side full dequantization is allowed only for debug checks or focused tests.
For `quantized_qwen3` runners, expected tensors, `allclose`, and max-abs
reporting are computed with torch tensors on the ROCm device. NumPy is only
used to slice GGUF bytes and pass host buffers to XRT.

## Operator Handoff

Default stage boundaries use explicit memref arguments, so the next stage can
consume the previous stage's output buffer directly. Do not insert graph JSON,
manifest objects, `.npy` files, or host-side repacking between operators.

For adjacent Qwen3 stages, keep the model export simple and official-looking:
generate each operator as its own Python AIR kernel, then connect the stages
with existing AIR/XRT mechanisms. The current production runners validate one
operator at a time against PyTorch ROCm. Multi-operator execution should follow
the torch2vk-style shape: separate xclbins may hand off through shared
`pyxrt.bo` device buffers, without introducing a torch2air runtime object.

The current verified projection shape is decode `S=1` with `output_rows=64`
and fixed prefill bucket `S=8` with `output_rows=16`. Longer prefill should be
host-split into fixed buckets until a larger bucket has real NPU evidence.

When two adjacent operators can be fused without changing the schedule shape,
keep that work behind real NPU verification. Full hidden RMSNorm should not
replace the separate stages until it works for all four Q4_K blocks on real NPU
hardware.

## Runtime Boundary

Do not recreate pyxrt concepts in torch2air.

These already exist and should be used directly through MLIR-AIR or pyxrt:

- `pyxrt.device`
- `pyxrt.xclbin`
- `pyxrt.hw_context`
- `pyxrt.kernel`
- `pyxrt.bo`
- `pyxrt.run`
- `air.backend.xrt.XRTBackend`
- `air.backend.xrt_runner.XRTRunner`

Model code may call existing AIR/XRT helpers, but torch2air should not introduce
new wrapper classes for device, kernel, buffer object, hardware context, or run
state.

## First Model

The first target is `models.quantized_qwen3`.

Initial useful milestone:

- Use this repository's `.venv`; do not use `torch2vk/.venv`.
- Load the Qwen3 PyTorch module on meta tensors.
- Call local `export_one(...)` for one small boundary first.
- Traverse the returned `ExportedProgram` directly.
- Render one concrete Python AIR kernel from the traversed nodes.
- The runner compiles that Python kernel to AIR MLIR, lowers it, and runs it
  with existing AIR/XRT APIs.
- Keep quantized packed weight handling tied to real GGUF tensor metadata.

Only broaden from there once the simple path produces a real artifact that the
next compile/run step can consume.

## Non-Goals

- No custom torch2air graph IR.
- No default graph JSON dump.
- No export manifest layer.
- No AirRegistry/AirVariant abstraction.
- No pyxrt runtime wrapper.
- No host-side quantized weight dequantization in the NPU execution path.
- No IREE `vmfb` path for this project.

The working rule is: direct `export_one(...)`, direct PyTorch graph traversal,
small Python AIR kernels, and existing AIR/XRT runtime APIs.
