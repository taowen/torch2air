from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyxrt as xrt


def compile_runtime(
    *,
    aie_mlir: Path,
    work_dir: Path,
    instance_name: str,
    peano_install_dir: str,
    link_objects: tuple[Path, ...] = (),
) -> tuple[Path, Path, Path]:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    aiecc_dir = work_dir / "aiecc"
    shutil.rmtree(aiecc_dir, ignore_errors=True)
    aiecc_dir.mkdir(parents=True, exist_ok=True)
    if link_objects:
        for object_path in link_objects:
            target_path = work_dir / object_path.name
            if object_path.resolve() != target_path.resolve():
                shutil.copy2(object_path, target_path)
        for object_dir in (work_dir / "air_project", aiecc_dir / "air_project"):
            object_dir.mkdir(parents=True, exist_ok=True)
            for object_path in link_objects:
                shutil.copy2(object_path, object_dir / object_path.name)

    npu_mlir = work_dir / f"{instance_name}.npu.mlir"
    xclbin = work_dir / f"{instance_name}.xclbin"
    insts = work_dir / f"{instance_name}.insts.bin"
    air_opt = installed_tool("air-opt", "MLIR_AIR_INSTALL_DIR")
    aiecc = installed_tool("aiecc", "MLIR_AIE_INSTALL_DIR")

    subprocess.run(
        [
            air_opt,
            str(aie_mlir),
            "--air-to-std",
            "--airrt-to-npu",
            "--canonicalize",
            "-o",
            str(npu_mlir),
        ],
        check=True,
    )
    if link_objects:
        _dedupe_private_func_declarations(npu_mlir)
    if instance_name == "run_attention_core":
        _await_attention_input_dma(npu_mlir)
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


def compile_external_kernel_object(
    *,
    source: Path,
    object_name: str,
    work_dir: Path,
    peano_install_dir: str,
    defines: dict[str, int | str],
) -> Path:
    object_path = work_dir / object_name
    object_path.parent.mkdir(parents=True, exist_ok=True)
    aie_opt = installed_tool("aie-opt", "MLIR_AIE_INSTALL_DIR")
    include_dir = Path(aie_opt).resolve().parent.parent / "include"
    warning_flags = [
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-unused-command-line-argument",
    ]
    cmd = [
        str(Path(peano_install_dir) / "bin" / "clang++"),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        *warning_flags,
        "-DNDEBUG",
        "-I",
        str(include_dir),
        *(f"-D{name}={value}" for name, value in defines.items()),
        "-c",
        str(source),
        "-o",
        str(object_path),
    ]
    subprocess.run(cmd, check=True)
    return object_path


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


def load_xrt_kernel(
    device: xrt.device,
    *,
    xclbin: Path,
    insts: Path,
) -> tuple[xrt.hw_context, xrt.kernel, np.ndarray, xrt.bo]:
    loaded_xclbin = xrt.xclbin(str(xclbin))
    device.register_xclbin(loaded_xclbin)
    context = xrt.hw_context(device, loaded_xclbin.get_uuid())
    kernel_name = [
        kernel.get_name()
        for kernel in loaded_xclbin.get_kernels()
        if "MLIR_AIE" in kernel.get_name()
    ][0]
    kernel = xrt.kernel(context, kernel_name)
    instr_v = np.frombuffer(insts.read_bytes(), dtype=np.uint32)
    bo_instr = xrt.bo(
        device,
        len(instr_v) * 4,
        xrt.bo.cacheable,
        kernel.group_id(1),
    )
    bo_instr.write(instr_v, 0)
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return context, kernel, instr_v, bo_instr


def trace_npu(message: str) -> None:
    if os.environ.get("TORCH2AIR_TRACE_NPU") == "1":
        print(f"trace_npu {message}", flush=True)


def _dedupe_private_func_declarations(path: Path) -> None:
    seen: set[str] = set()
    output: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("aie.device("):
            seen.clear()
        if stripped.startswith("func.func private @"):
            symbol = stripped.split("@", 1)[1].split("(", 1)[0]
            if symbol in seen:
                continue
            seen.add(symbol)
        output.append(line)
    path.write_text("\n".join(output) + "\n")


def _await_attention_input_dma(path: Path) -> None:
    input_channels = ("@air_attention_q", "@air_attention_kv")
    awaiting_tasks: set[str] = set()
    current_task: str | None = None
    output: list[str] = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if (
            " = aiex.dma_configure_task_for " in line
            and any(channel in line for channel in input_channels)
        ):
            current_task = stripped.split(" = ", 1)[0]
            awaiting_tasks.add(current_task)
            output.append(line)
            continue
        if current_task is not None and stripped == "}":
            output.append(f"{line} {{issue_token = true}}")
            current_task = None
            continue
        if stripped.startswith("aiex.dma_free_task("):
            task = stripped.removeprefix("aiex.dma_free_task(").removesuffix(")")
            if task in awaiting_tasks:
                output.append(line.replace("aiex.dma_free_task", "aiex.dma_await_task"))
                continue
        output.append(line)

    path.write_text("\n".join(output) + "\n")
