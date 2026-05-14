from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import cast

from torch2air.export.builder import AirBuilder
from torch2air.export.input_layernorm import InputLayerNormAirBuilder
from torch2air.export.q4k_embedding import Q4KEmbeddingAirBuilder
from torch2air.export.q4k_linear import Q4KLinearAirBuilder
from torch2air.export.q6k_linear import Q6KLinearAirBuilder


AIR_OPT_DIAGNOSTIC_FLAGS = (
    "--mlir-print-op-on-diagnostic=false",
    "--mlir-disable-diagnostic-notes",
)
Q4K_LINEAR_KERNEL_SOURCE = Path(__file__).resolve().parents[1] / "export" / "kernels" / "q4k_linear.cc"
Q4K_LINEAR_LINK_OBJECT = "q4k_linear.o"
Q6K_LINEAR_KERNEL_SOURCE = Path(__file__).resolve().parents[1] / "export" / "kernels" / "q6k_linear.cc"
Q6K_LINEAR_LINK_OBJECT = "q6k_linear.o"


def compile_q4k_embedding_python_kernel(
    *,
    kernel_py: Path,
    function_name: str,
    work_dir: Path,
    instance_name: str,
) -> tuple[Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    builder = Q4KEmbeddingAirBuilder(function_name=instance_name)
    load_kernel_function(kernel_py, function_name)(builder)
    source_mlir = work_dir / f"{instance_name}.air.mlir"
    source_mlir.write_text(builder.render_air())
    aie_mlir = lower_scf_air_to_aie(
        source_mlir=source_mlir,
        work_dir=work_dir,
        stem=instance_name,
        herd_rows=1,
        herd_cols=_embedding_herd_cols(builder),
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=instance_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts


def compile_input_layernorm_python_kernel(
    *,
    kernel_py: Path,
    function_name: str,
    work_dir: Path,
    instance_name: str,
    eps: float,
) -> tuple[Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    builder = InputLayerNormAirBuilder(function_name=instance_name, eps=eps)
    load_kernel_function(kernel_py, function_name)(builder)
    source_mlir = work_dir / f"{instance_name}.air.mlir"
    source_mlir.write_text(builder.render_air())
    aie_mlir = lower_scf_air_to_aie(
        source_mlir=source_mlir,
        work_dir=work_dir,
        stem=instance_name,
        herd_rows=_input_layernorm_herd_rows(builder),
        herd_cols=1,
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=instance_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts


def compile_q4k_linear_python_kernel(
    *,
    kernel_py: Path,
    function_name: str,
    work_dir: Path,
    instance_name: str,
    output_features: int,
    output_tile_rows: int = 16,
) -> tuple[Path, Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    builder = Q4KLinearAirBuilder(
        function_name=instance_name,
        output_features=output_features,
        output_tile_rows=output_tile_rows,
    )
    load_kernel_function(kernel_py, function_name)(builder)
    _, hidden_size, _ = builder.linear_shape()
    source_mlir = work_dir / f"{instance_name}.air.mlir"
    source_mlir.write_text(builder.render_air())
    object_file = compile_q4k_linear_object(
        work_dir=work_dir,
        peano_install_dir=peano,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    aie_mlir = lower_scf_air_to_aie(
        source_mlir=source_mlir,
        work_dir=work_dir,
        stem=instance_name,
        herd_rows=builder.herd_rows(),
        herd_cols=builder.herd_cols(),
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=instance_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts, object_file


def compile_q6k_linear_python_kernel(
    *,
    kernel_py: Path,
    function_name: str,
    work_dir: Path,
    instance_name: str,
    output_features: int,
    output_tile_rows: int = 16,
) -> tuple[Path, Path, Path, Path, Path]:
    _, _, peano = prepend_air_tool_paths()
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    builder = Q6KLinearAirBuilder(
        function_name=instance_name,
        output_features=output_features,
        output_tile_rows=output_tile_rows,
    )
    load_kernel_function(kernel_py, function_name)(builder)
    _, hidden_size, _ = builder.linear_shape()
    source_mlir = work_dir / f"{instance_name}.air.mlir"
    source_mlir.write_text(builder.render_air())
    object_file = compile_q6k_linear_object(
        work_dir=work_dir,
        peano_install_dir=peano,
        hidden_size=hidden_size,
        output_tile_rows=output_tile_rows,
    )
    aie_mlir = lower_scf_air_to_aie(
        source_mlir=source_mlir,
        work_dir=work_dir,
        stem=instance_name,
        herd_rows=builder.herd_rows(),
        herd_cols=builder.herd_cols(),
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=instance_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts, object_file


def compile_q4k_linear_object(
    *,
    work_dir: Path,
    peano_install_dir: Path,
    hidden_size: int,
    output_tile_rows: int,
) -> Path:
    object_file = work_dir / Q4K_LINEAR_LINK_OBJECT
    aieopt_dir = Path(installed_tool("aie-opt", "MLIR_AIE_INSTALL_DIR")).resolve().parent.parent
    target_triple = f"{os.environ.get('AIE_TARGET', 'aie2p')}-none-unknown-elf"
    subprocess.run(
        [
            str(peano_install_dir / "bin" / "clang++"),
            "-O2",
            "-std=c++20",
            f"--target={target_triple}",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-Wno-unused-command-line-argument",
            "-DNDEBUG",
            f"-I{aieopt_dir / 'include'}",
            f"-DHIDDEN_SIZE={hidden_size}",
            f"-DOUTPUT_TILE_ROWS={output_tile_rows}",
            f"-DBLOCKS_PER_ROW={hidden_size // 256}",
            "-c",
            str(Q4K_LINEAR_KERNEL_SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
        cwd=work_dir,
    )
    return object_file


def compile_q6k_linear_object(
    *,
    work_dir: Path,
    peano_install_dir: Path,
    hidden_size: int,
    output_tile_rows: int,
) -> Path:
    object_file = work_dir / Q6K_LINEAR_LINK_OBJECT
    aieopt_dir = Path(installed_tool("aie-opt", "MLIR_AIE_INSTALL_DIR")).resolve().parent.parent
    target_triple = f"{os.environ.get('AIE_TARGET', 'aie2p')}-none-unknown-elf"
    subprocess.run(
        [
            str(peano_install_dir / "bin" / "clang++"),
            "-O2",
            "-std=c++20",
            f"--target={target_triple}",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-Wno-empty-body",
            "-Wno-unused-command-line-argument",
            "-DNDEBUG",
            f"-I{aieopt_dir / 'include'}",
            f"-DHIDDEN_SIZE={hidden_size}",
            f"-DOUTPUT_TILE_ROWS={output_tile_rows}",
            f"-DBLOCKS_PER_ROW={hidden_size // 256}",
            "-c",
            str(Q6K_LINEAR_KERNEL_SOURCE),
            "-o",
            str(object_file),
        ],
        check=True,
        cwd=work_dir,
    )
    return object_file


def prepend_air_tool_paths() -> tuple[Path, Path, Path]:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    mlir_air = Path(os.environ.get("MLIR_AIR_INSTALL_DIR", site_packages / "mlir_air"))
    mlir_aie = Path(os.environ.get("MLIR_AIE_INSTALL_DIR", site_packages / "mlir_aie"))
    peano = Path(os.environ.get("PEANO_INSTALL_DIR", site_packages / "llvm-aie"))
    paths = [mlir_air / "bin", mlir_aie / "bin", peano / "bin"]
    os.environ["PATH"] = ":".join(str(path) for path in paths) + ":" + os.environ.get("PATH", "")
    os.environ["MLIR_AIR_INSTALL_DIR"] = str(mlir_air)
    os.environ["MLIR_AIE_INSTALL_DIR"] = str(mlir_aie)
    os.environ["PEANO_INSTALL_DIR"] = str(peano)
    os.environ.setdefault("XRT_HACK_UNSECURE_LOADING_XCLBIN", "1")
    return mlir_air, mlir_aie, peano


def lower_scf_air_to_aie(
    *,
    source_mlir: Path,
    work_dir: Path,
    stem: str,
    herd_rows: int,
    herd_cols: int,
) -> Path:
    air_opt = Path(os.environ["MLIR_AIR_INSTALL_DIR"]) / "bin" / "air-opt"
    dma_mlir = work_dir / f"{stem}.dma.mlir"
    channel_mlir = work_dir / f"{stem}.channel.mlir"
    aie_mlir = work_dir / f"{stem}.aie.mlir"
    run_air_opt(
        [
            str(air_opt),
            str(source_mlir),
            *AIR_OPT_DIAGNOSTIC_FLAGS,
            "--air-par-to-launch=depth=0 has-air-segment=true",
            "--air-par-to-herd=depth=0",
            "--scf-forall-to-for",
            "--air-copy-to-dma",
            "--canonicalize",
            "--cse",
            "-o",
            str(dma_mlir),
        ],
    )
    run_air_opt(
        [
            str(air_opt),
            str(dma_mlir),
            *AIR_OPT_DIAGNOSTIC_FLAGS,
            "--air-dependency",
            "--air-dma-to-channel",
            "--canonicalize",
            "--cse",
            f"--air-place-herds=num-rows={herd_rows} num-cols={herd_cols} "
            "row-anchor=2 col-anchor=0",
            "-o",
            str(channel_mlir),
        ],
        suppress_success_diagnostics=True,
    )
    run_air_opt(
        [
            str(air_opt),
            str(channel_mlir),
            *AIR_OPT_DIAGNOSTIC_FLAGS,
            "--air-to-aie=device=npu2_4col row-offset=2 col-offset=0 stack-size=4096 "
            "emit-while-loop=true",
            "--canonicalize",
            "--cse",
            "-o",
            str(aie_mlir),
        ],
    )
    return aie_mlir


def compile_runtime(
    *,
    aie_mlir: Path,
    work_dir: Path,
    instance_name: str,
    peano_install_dir: str,
) -> tuple[Path, Path, Path]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    aiecc_dir = work_dir / "aiecc"
    shutil.rmtree(aiecc_dir, ignore_errors=True)
    aiecc_dir.mkdir(parents=True, exist_ok=True)

    npu_mlir = work_dir / f"{instance_name}.npu.mlir"
    xclbin = work_dir / f"{instance_name}.xclbin"
    insts = work_dir / f"{instance_name}.insts.bin"
    air_opt = installed_tool("air-opt", "MLIR_AIR_INSTALL_DIR")
    aiecc = installed_tool("aiecc", "MLIR_AIE_INSTALL_DIR")

    run_air_opt(
        [
            air_opt,
            str(aie_mlir),
            *AIR_OPT_DIAGNOSTIC_FLAGS,
            "--air-to-std",
            "--symbol-dce",
            "--airrt-to-npu",
            "--canonicalize",
            "-o",
            str(npu_mlir),
        ],
    )
    subprocess.run(
        [
            aiecc,
            "--no-aiesim",
            "--no-xchesscc",
            "--no-xbridge",
            "--no-compile-host",
            f"--tmpdir={aiecc_dir}",
            "--aie-generate-xclbin",
            f"--xclbin-name={xclbin}",
            "--aie-generate-npu-insts",
            f"--npu-insts-name={insts}",
            f"--xclbin-instance-name={instance_name}",
            f"--peano={peano_install_dir}",
            "-O",
            "0",
            str(npu_mlir),
        ],
        check=True,
        cwd=work_dir,
    )
    if not xclbin.exists():
        raise RuntimeError(f"aiecc did not generate xclbin: {xclbin}")
    if not insts.exists():
        raise RuntimeError(f"aiecc did not generate NPU instructions: {insts}")
    return npu_mlir, xclbin, insts


def run_air_opt(args: list[str], *, suppress_success_diagnostics: bool = False) -> None:
    if not suppress_success_diagnostics:
        subprocess.run(args, check=True)
        return

    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    completed.check_returncode()


def installed_tool(name: str, install_env: str) -> str:
    install_dir = os.environ.get(install_env)
    if install_dir:
        candidate = Path(install_dir) / "bin" / name
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} is not on PATH and {install_env} is not set")
    return found


def load_kernel_function(
    kernel_py: Path,
    function_name: str,
) -> Callable[[AirBuilder], None]:
    spec = importlib.util.spec_from_file_location(function_name, kernel_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load Python kernel: {kernel_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    if not callable(function):
        raise RuntimeError(f"{kernel_py} does not define callable {function_name}")
    return cast(Callable[[AirBuilder], None], function)


def _embedding_herd_cols(builder: Q4KEmbeddingAirBuilder) -> int:
    if builder.embedding_output is None:
        raise ValueError("kernel did not emit an embedding output")
    output = builder.tensors[builder.embedding_output]
    if len(output.shape) != 3:
        raise ValueError(f"expected rank-3 embedding output, got {output.shape}")
    hidden_size = output.shape[2]
    if hidden_size % 256 != 0:
        raise ValueError(f"Q4_K hidden size must be divisible by 256, got {hidden_size}")
    return hidden_size // 256


def _input_layernorm_herd_rows(builder: InputLayerNormAirBuilder) -> int:
    if not builder.outputs:
        raise ValueError("kernel did not mark an output")
    output = builder.tensors[builder.outputs[-1]]
    if len(output.shape) != 3:
        raise ValueError(f"expected rank-3 input_layernorm output, got {output.shape}")
    return min(output.shape[1], 4)
