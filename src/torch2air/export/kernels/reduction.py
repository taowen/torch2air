from __future__ import annotations

from torch2air.export.builder import AirBuilder, KernelAttr


def aten_mean_dim(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    dims: str,
    keepdim: str,
) -> None:
    builder.emit_kernel(
        "aten.mean.dim",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dims", dims), KernelAttr("keepdim", keepdim)),
    )


def aten_sum_dim_int_list(
    builder: AirBuilder,
    *,
    source: str,
    output: str,
    dims: str,
    keepdim: str,
) -> None:
    builder.emit_kernel(
        "aten.sum.dim_IntList",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("dims", dims), KernelAttr("keepdim", keepdim)),
    )
