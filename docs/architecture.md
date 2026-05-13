# torch2air Architecture

`torch2air` is the AIR backend workspace for PyTorch-exported models. Keep it
close to how `torch2vk` actually works: model export scripts call a local
`export_one(...)` helper several times with concrete modules and arguments.
That helper uses `torch.export`, walks the exported PyTorch object directly, and
renders files from templates.

Do not build an intermediate graph format, registry system, manifest layer, or
runtime abstraction until a concrete model export proves that one is needed.

## Shape

The intended flow is:

```text
models.quantized_qwen3.export
  -> export_one(name, module, args, kwargs, ...)
  -> torch.export.export(...)
  -> iterate exported_program.graph_module.graph.nodes
  -> render Jinja templates for tiled pre-AIR MLIR
  -> lower with air-opt / aircc
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
      templates.py          # small Jinja helpers only
      kernels/              # reusable tiled MLIR kernel templates
    weights/
      gguf.py               # GGUF tensor metadata and packed reads
  models/
    quantized_qwen3/
      export.py             # model-specific export_one(...) calls
      quantization.py       # model-specific quantized tensor naming
      reference.py          # PyTorch ROCm reference, when implemented
      run.py                # thin CLI dispatch for model runners
      run_embed_tokens.py   # embed_tokens XRT execution and reference check
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
    kwargs: dict[str, object] | None = None,
    *,
    weight_prefix: str = "",
    shape_exprs: dict[int, str] | None = None,
) -> None:
    program = torch.export.export(module, args, kwargs=kwargs, strict=False)
    for node in program.graph_module.graph.nodes:
        # Inspect aten target, tensor_meta, args, kwargs directly.
        ...
    # Render generated tiled MLIR directly from this context.
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
  run_text_layer.mlir
```

The generated model artifact is tiled MLIR, not hand-written AIR dialect and
not a Python AIR builder. It should look like the spike fixtures:
`scf.parallel`, `memref.subview`, `memref.alloc(..., 2)`, `memref.copy`, and
tile-local `linalg`/`arith`/`scf` work. `air-opt` is responsible for lowering
that into `air.launch`, `air.herd`, `air.dma_memcpy_nd`, channels, and AIE IR.

Reusable operator bodies belong under `torch2air.export.kernels`. The model
package chooses a stage and passes concrete Qwen3 shapes, tensor names, and
module metadata; it should not own a growing library of generic operator
templates.

Do not put model-stage compute in ad hoc `.cc` files. A separate native kernel
should only appear after a measured reason makes that boundary necessary.

If a debug dump is useful while developing, make it opt-in and keep it out of
the default export path.

## Weights

GGUF packed weights are real inputs to the AIR path. `torch2air.weights.gguf`
owns reading GGUF metadata and packed byte ranges. The export path should record
or reference GGUF tensor names only where needed by generated code.

Do not host-dequantize Q4_K weight values for NPU execution. The NPU path should
consume the packed blocks and decode the quantized values inside the generated
tile body.

The current `quantized_qwen3.embed_tokens` runner has one temporary exception:
the GGUF Q4_K block-level f16 values `d` and `dmin` are decoded on the host into
a small `block_f16_scales` f32 input. The NPU still reads real packed Q4_K
blocks and decodes the per-subblock scale/min bytes and q4 nibbles. This keeps
the first tiled MLIR stage runnable while the AIE f16 scalar conversion issue is
isolated. Remove this exception once f16-to-f32 lowering is validated.

Host-side full dequantization is allowed only for reference checks or tests.

## Operator Handoff

Default stage boundaries use explicit memref arguments, so the next stage can
consume the previous stage's output buffer directly. Do not insert graph JSON,
manifest objects, `.npy` files, or host-side repacking between operators.

For adjacent Qwen3 stages, keep the model export simple and official-looking:
generate each operator as its own tiled MLIR stage, then connect the stages with
the existing AIR/XRT mechanisms. The current full-hidden
`embed_tokens -> input_layernorm` path runs two stage xclbins with one shared
`pyxrt.bo` for the hidden-state memref. That avoids a host-side intermediate
copy while staying inside pyxrt's device, kernel, BO, and run concepts.

The exporter also generates a stitched AIR module in the same style as
MLIR-AIR's upstream llama multi-launch examples: independent stage bodies are
renamed and connected by an arg map. On this Python 3.12 environment, pyxrt does
not expose the ELF loader used by those examples. The xclbin `aircc` path for
the stitched module currently overflows AIE program memory for the full Q4_K
embedding body, so the runnable path remains separate xclbins with a shared BO
until the ELF binding or a smaller embedded body is available.

When two adjacent operators can be fused without changing the schedule shape,
keep that as an explicit experiment. The current
`embed_tokens_input_layernorm` spike proves L1 handoff for one Q4_K block. Full
hidden RMSNorm should not replace the separate stages until it works for all
four Q4_K blocks on real NPU hardware.

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
- Render one concrete file from the traversed nodes.
- That file is a tiled `.mlir` file consumed by `air-opt`.
- Keep Q4_K packed weight handling tied to real GGUF tensor metadata.

Only broaden from there once the simple path produces a real artifact that the
next compile/run step can consume.

## Non-Goals

- No custom torch2air graph IR.
- No default graph JSON dump.
- No export manifest layer.
- No AirRegistry/AirVariant abstraction.
- No pyxrt runtime wrapper.
- No host-side Q4_K dequantization in the NPU execution path.
- No IREE `vmfb` path for this project.

The working rule is: direct `export_one(...)`, direct PyTorch graph traversal,
simple templates, and existing AIR/XRT runtime APIs.
