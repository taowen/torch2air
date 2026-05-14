from __future__ import annotations

from torch2air.export.builder import AirBuilder, KernelAttr


def aten_embedding_default(
    builder: AirBuilder,
    *,
    weight: str,
    indices: str,
    output: str,
) -> None:
    builder.emit_kernel("aten.embedding.default", output=output, inputs=(weight, indices))


def aten_linear_default(
    builder: AirBuilder,
    *,
    source: str,
    weight: str,
    bias: str | None,
    output: str,
) -> None:
    inputs = (source, weight) if bias is None else (source, weight, bias)
    builder.emit_kernel("aten.linear.default", output=output, inputs=inputs)


def aten_matmul_default(builder: AirBuilder, *, lhs: str, rhs: str, output: str) -> None:
    builder.emit_kernel("aten.matmul.default", output=output, inputs=(lhs, rhs))


def aten_cat_default(
    builder: AirBuilder,
    *,
    tensors: tuple[str, ...],
    output: str,
    dim: str,
) -> None:
    builder.emit_kernel(
        "aten.cat.default",
        output=output,
        inputs=tensors,
        attrs=(KernelAttr("dim", dim),),
    )


def aten_slice_tensor(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    dim: str,
    start: str,
    end: str,
    step: str,
) -> None:
    builder.emit_kernel(
        "aten.slice.Tensor",
        output=output,
        inputs=(source,),
        attrs=(
            KernelAttr("dim", dim),
            KernelAttr("start", start),
            KernelAttr("end", end),
            KernelAttr("step", step),
        ),
    )


def aten_select_int(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    dim: str,
    index: str,
) -> None:
    builder.emit_kernel(
        "aten.select.int",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dim", dim), KernelAttr("index", index)),
    )


def aten_repeat_interleave_self_int(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    repeats: str,
    dim: str,
) -> None:
    builder.emit_kernel(
        "aten.repeat_interleave.self_int",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("repeats", repeats), KernelAttr("dim", dim)),
    )
