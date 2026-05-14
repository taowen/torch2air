from __future__ import annotations

from torch2air.export.builder import AirBuilder, KernelAttr


def aten_add_tensor(builder: AirBuilder, *, lhs: str, rhs: str, output: str) -> None:
    builder.emit_kernel("aten.add.Tensor", output=output, inputs=(lhs, rhs))


def aten_sub_tensor(builder: AirBuilder, *, lhs: str, rhs: str, output: str) -> None:
    builder.emit_kernel("aten.sub.Tensor", output=output, inputs=(lhs, rhs))


def aten_mul_tensor(builder: AirBuilder, *, lhs: str, rhs: str, output: str) -> None:
    builder.emit_kernel("aten.mul.Tensor", output=output, inputs=(lhs, rhs))


def aten_div_tensor(builder: AirBuilder, *, lhs: str, rhs: str, output: str) -> None:
    builder.emit_kernel("aten.div.Tensor", output=output, inputs=(lhs, rhs))


def aten_neg_default(builder: AirBuilder, *, source: str, output: str) -> None:
    builder.emit_kernel("aten.neg.default", output=output, inputs=(source,))


def aten_rsqrt_default(builder: AirBuilder, *, source: str, output: str) -> None:
    builder.emit_kernel("aten.rsqrt.default", output=output, inputs=(source,))


def aten_sin_default(builder: AirBuilder, *, source: str, output: str) -> None:
    builder.emit_kernel("aten.sin.default", output=output, inputs=(source,))


def aten_cos_default(builder: AirBuilder, *, source: str, output: str) -> None:
    builder.emit_kernel("aten.cos.default", output=output, inputs=(source,))


def aten_pow_tensor_scalar(builder: AirBuilder, *, source: str, output: str, exponent: str) -> None:
    builder.emit_kernel(
        "aten.pow.Tensor_Scalar",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("exponent", exponent),),
    )


def aten_silu_default(builder: AirBuilder, *, source: str, output: str) -> None:
    builder.emit_kernel("aten.silu.default", output=output, inputs=(source,))


def aten_gelu_default(builder: AirBuilder, *, source: str, output: str, approximate: str) -> None:
    builder.emit_kernel(
        "aten.gelu.default",
        output=output,
        inputs=(source,),
        attrs=(KernelAttr("approximate", approximate),),
    )
