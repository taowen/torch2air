from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from air.dialects import arith
from air.dialects.air import T, herd, launch, segment
from air._mlir_libs._mlir.ir import Value

type RegionBody = Callable[..., None]
type RegionDecorator = Callable[[RegionBody], RegionBody]


def air_launch(*, operands: Sequence[Value]) -> RegionDecorator:
    return cast(RegionDecorator, launch(operands=operands))


def air_segment(*, name: str, operands: Sequence[Value]) -> RegionDecorator:
    return cast(RegionDecorator, segment(name=name, operands=operands))


def air_herd(*, name: str, sizes: Sequence[int], operands: Sequence[Value]) -> RegionDecorator:
    return cast(RegionDecorator, herd(name=name, sizes=sizes, operands=operands))


def idx(value: int) -> Value:
    return arith.ConstantOp(T.index(), value)
