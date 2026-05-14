from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class KernelAttr:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str


class AirBuilder(Protocol):
    def define_tensor(self, name: str, *, shape: tuple[int, ...], dtype: str) -> None: ...

    def mark_output(self, name: str) -> None: ...

    def emit_kernel(
        self,
        target: str,
        *,
        output: str,
        inputs: tuple[str, ...],
        attrs: tuple[KernelAttr, ...] = (),
    ) -> None: ...


class TextAirBuilder:
    def __init__(self) -> None:
        self.tensors: dict[str, TensorInfo] = {}
        self.outputs: list[str] = []
        self._lines: list[str] = []

    def define_tensor(self, name: str, *, shape: tuple[int, ...], dtype: str) -> None:
        self.tensors[name] = TensorInfo(name=name, shape=shape, dtype=dtype)
        shape_text = "x".join(str(dim) for dim in shape)
        self._lines.append(f"# tensor {name}: {shape_text} {dtype}")

    def mark_output(self, name: str) -> None:
        self.outputs.append(name)
        self._lines.append(f"# output {name}")

    def emit_kernel(
        self,
        target: str,
        *,
        output: str,
        inputs: tuple[str, ...],
        attrs: tuple[KernelAttr, ...] = (),
    ) -> None:
        input_text = ", ".join(inputs)
        attr_text = ""
        if attrs:
            pairs = ", ".join(f"{attr.name}={attr.value}" for attr in attrs)
            attr_text = f" [{pairs}]"
        self._lines.append(f"{output} = {target}({input_text}){attr_text}")

    def render(self) -> str:
        return "\n".join(self._lines)
