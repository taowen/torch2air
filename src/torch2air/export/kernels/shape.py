from __future__ import annotations

from torch2air.export.builder import AirBuilder, KernelAttr


def alias(builder: AirBuilder, *, source: str, output: str, target: str) -> None:
    builder.emit_kernel(
        target,
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("alias", "true"),),
    )


def aten_view_default(builder: AirBuilder, *, source: str, output: str, shape: str) -> None:
    builder.emit_kernel(
        "aten.view.default",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("shape", shape),),
    )


def aten_reshape_default(builder: AirBuilder, *, source: str, output: str, shape: str) -> None:
    builder.emit_kernel(
        "aten.reshape.default",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("shape", shape),),
    )


def aten_unsqueeze_default(builder: AirBuilder, *, source: str, output: str, dim: str) -> None:
    builder.emit_kernel(
        "aten.unsqueeze.default",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dim", dim),),
    )


def aten_permute_default(builder: AirBuilder, *, source: str, output: str, dims: str) -> None:
    builder.emit_kernel(
        "aten.permute.default",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dims", dims),),
    )


def aten_transpose_int(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    dim0: str,
    dim1: str,
) -> None:
    builder.emit_kernel(
        "aten.transpose.int",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dim0", dim0), KernelAttr("dim1", dim1)),
    )
