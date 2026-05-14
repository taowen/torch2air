from __future__ import annotations

from torch2air.export.builder import AirBuilder, KernelAttr


def aten_scaled_dot_product_attention_default(
    builder: AirBuilder,
    *,
    query: str,
    key: str,
    value: str,
    output: str,
    is_causal: str,
) -> None:
    builder.emit_kernel(
        "aten.scaled_dot_product_attention.default",
        output=output,
        inputs=(query, key, value),
        attrs=(KernelAttr("is_causal", is_causal),),
    )
