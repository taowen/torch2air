from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import torch
from torch.fx.passes.shape_prop import ShapeProp

from torch2air.mlir_text import render_linalg_matmul, tensor_type_from_torch


class Frontend(str, Enum):
    AUTO = "auto"
    FX = "fx"
    TORCH_MLIR = "torch-mlir"


@dataclass(frozen=True)
class ExportedProgram:
    mlir: str
    frontend: str
    function_name: str


def export_model_to_linalg_mlir(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    function_name: str = "forward",
    frontend: Frontend | str = Frontend.AUTO,
) -> ExportedProgram:
    frontend = Frontend(frontend)
    if frontend in {Frontend.AUTO, Frontend.TORCH_MLIR}:
        try:
            return _export_with_torch_mlir(model, sample_inputs, function_name=function_name)
        except Exception:
            if frontend == Frontend.TORCH_MLIR:
                raise

    return _export_matmul_with_fx(model, sample_inputs, function_name=function_name)


def write_exported_mlir(exported: ExportedProgram, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(exported.mlir, encoding="utf-8")


def _export_with_torch_mlir(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    function_name: str,
) -> ExportedProgram:
    import torch_mlir  # type: ignore[import-not-found]

    output_type = getattr(torch_mlir.OutputType, "LINALG_ON_TENSORS")
    module = torch_mlir.compile(model, sample_inputs, output_type=output_type)
    mlir = str(module)
    if f"@{function_name}" not in mlir and "@forward" in mlir and function_name != "forward":
        mlir = mlir.replace("@forward", f"@{function_name}")
    return ExportedProgram(mlir=mlir, frontend="torch-mlir", function_name=function_name)


def _export_matmul_with_fx(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    *,
    function_name: str,
) -> ExportedProgram:
    if len(sample_inputs) != 2:
        raise ValueError("FX fallback only supports a two-input matmul model")

    model = model.eval()
    with torch.no_grad():
        graph_module = torch.fx.symbolic_trace(model)
        ShapeProp(graph_module).propagate(*sample_inputs)

    matmul_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _is_matmul_target(node.target)
    ]
    if len(matmul_nodes) != 1:
        raise ValueError(
            "FX fallback supports exactly one matmul-like call; "
            f"found {len(matmul_nodes)} in {graph_module.graph}"
        )

    matmul_node = matmul_nodes[0]
    lhs_node, rhs_node = _node_args(matmul_node.args, 2)
    lhs_meta = lhs_node.meta["tensor_meta"]
    rhs_meta = rhs_node.meta["tensor_meta"]
    result_meta = matmul_node.meta["tensor_meta"]

    mlir = render_linalg_matmul(
        function_name=function_name,
        lhs=tensor_type_from_torch_meta(lhs_meta),
        rhs=tensor_type_from_torch_meta(rhs_meta),
        result=tensor_type_from_torch_meta(result_meta),
    )
    return ExportedProgram(mlir=mlir, frontend="fx", function_name=function_name)


def tensor_type_from_torch_meta(meta: object):
    fake = torch.empty(tuple(int(dim) for dim in meta.shape), dtype=meta.dtype)
    return tensor_type_from_torch(fake)


def _is_matmul_target(target: object) -> bool:
    matmul_targets = {
        operator.matmul,
        torch.matmul,
        torch.mm,
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
    }
    return target in matmul_targets


def _node_args(args: Iterable[object], count: int):
    values = tuple(args)
    if len(values) != count:
        raise ValueError(f"expected {count} node args, got {len(values)}")
    return values
