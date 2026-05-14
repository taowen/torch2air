from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class KernelAttr:
    name: str
    value: str


class AirBuilder(Protocol):
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
        self._lines: list[str] = []

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
