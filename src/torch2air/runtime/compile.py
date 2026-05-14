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
from torch2air.export.q4k_embedding import Q4KEmbeddingAirBuilder


AIR_OPT_DIAGNOSTIC_FLAGS = (
    "--mlir-print-op-on-diagnostic=false",
    "--mlir-disable-diagnostic-notes",
)


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
        herd_cols=_embedding_herd_cols(builder),
    )
    _, xclbin, insts = compile_runtime(
        aie_mlir=aie_mlir,
        work_dir=work_dir,
        instance_name=instance_name,
        peano_install_dir=str(peano),
    )
    return source_mlir, aie_mlir, xclbin, insts


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
            f"--air-place-herds=num-rows=1 num-cols={herd_cols} row-anchor=2 col-anchor=0",
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
