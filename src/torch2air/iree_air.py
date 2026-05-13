from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from os import environ
from pathlib import Path


@dataclass(frozen=True)
class IreeAirConfig:
    iree_compile: str = "iree-compile"
    target_backend: str = "amd-aie"
    target_device: str = field(
        default_factory=lambda: environ.get("TORCH2AIR_TARGET_DEVICE", "npu4")
    )
    device_hal: str = field(
        default_factory=lambda: environ.get("TORCH2AIR_DEVICE_HAL", "xrt-lite")
    )
    tile_pipeline: str = field(
        default_factory=lambda: environ.get("TORCH2AIR_TILE_PIPELINE", "pack-peel")
    )
    lower_to_aie_pipeline: str = field(
        default_factory=lambda: environ.get("TORCH2AIR_LOWER_TO_AIE_PIPELINE", "air")
    )
    air_dump_pass: str = "air-dma-to-channel"
    peano_install_dir: Path | None = None
    vitis_install_dir: Path | None = None
    extra_compile_flags: tuple[str, ...] = ("--iree-hal-memoization=false",)


def lower_linalg_to_air(
    source_mlir: Path,
    air_mlir: Path,
    *,
    vmfb: Path | None = None,
    config: IreeAirConfig | None = None,
    vmfb_config: IreeAirConfig | None = None,
) -> None:
    config = config or IreeAirConfig()
    source_mlir = source_mlir.resolve()
    air_mlir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="torch2air-dump-") as temp_dir:
        dump_dir = Path(temp_dir) / "air-dump"
        dump_dir.mkdir()
        dump_error: RuntimeError | None = None
        compile_to_dump = _compile_command(
            source_mlir,
            output=None,
            config=config,
            compile_to_executable_targets=True,
            dump_dir=dump_dir,
        )
        # The AIR dump is emitted before later AIR-to-AIE placement. On NPU4 that
        # later placement may fail while the AIR artifact is still useful.
        try:
            _run(compile_to_dump)
        except RuntimeError as exc:
            dump_error = exc
        try:
            dumped_air = _find_air_dump(dump_dir)
        except RuntimeError:
            if dump_error is not None:
                raise dump_error
            raise
        shutil.copyfile(dumped_air, air_mlir)

    if vmfb is not None:
        vmfb.parent.mkdir(parents=True, exist_ok=True)
        _run(_compile_command(source_mlir, output=vmfb, config=vmfb_config or config))


def _compile_command(
    source_mlir: Path,
    *,
    output: Path | None,
    config: IreeAirConfig,
    compile_to_executable_targets: bool = False,
    dump_dir: Path | None = None,
) -> list[str]:
    command = [
        config.iree_compile,
        str(source_mlir),
        f"--iree-hal-target-backends={config.target_backend}",
        f"--iree-amdaie-target-device={config.target_device}",
        f"--iree-amdaie-device-hal={config.device_hal}",
        f"--iree-amdaie-lower-to-aie-pipeline={config.lower_to_aie_pipeline}",
        f"--iree-amdaie-tile-pipeline={config.tile_pipeline}",
    ]
    if compile_to_executable_targets:
        command.append("--compile-to=executable-targets")
    if dump_dir is not None:
        command.extend(
            [
                "--mlir-disable-threading",
                f"--mlir-print-ir-after={config.air_dump_pass}",
                f"--mlir-print-ir-tree-dir={dump_dir}",
                "--mlir-print-ir-module-scope",
            ]
        )
    if output is not None:
        command.extend(["-o", str(output)])
    peano_dir = _resolve_tool_dir(
        source_mlir,
        explicit=config.peano_install_dir,
        env_names=("PEANO_INSTALL_DIR", "IREE_AMD_AIE_PEANO_INSTALL_DIR"),
        repo_relative="llvm-aie",
        required_file="bin/opt",
    )
    if peano_dir is not None:
        command.append(f"--iree-amd-aie-peano-install-dir={peano_dir}")
    vitis_dir = _resolve_tool_dir(
        source_mlir,
        explicit=config.vitis_install_dir,
        env_names=("VITIS_INSTALL_DIR", "IREE_AMD_AIE_VITIS_INSTALL_DIR"),
        repo_relative=None,
        required_file=None,
    )
    if vitis_dir is not None:
        command.append(f"--iree-amd-aie-vitis-install-dir={vitis_dir}")
    command.extend(config.extra_compile_flags)
    return command


def _resolve_tool_dir(
    source_mlir: Path,
    *,
    explicit: Path | None,
    env_names: tuple[str, ...],
    repo_relative: str | None,
    required_file: str | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for env_name in env_names:
        if environ.get(env_name):
            candidates.append(Path(environ[env_name]))
    if repo_relative is not None:
        for parent in source_mlir.resolve().parents:
            candidates.append(parent / repo_relative)

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if not candidate.exists():
            continue
        if required_file is not None and not (candidate / required_file).exists():
            continue
        return candidate
    return None


def _find_air_dump(dump_dir: Path) -> Path:
    candidates: list[Path] = []
    for path in dump_dir.rglob("*.mlir"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "air.launch" in text or "air.herd" in text or "air.channel" in text:
            candidates.append(path)
    if not candidates:
        raise RuntimeError(f"no AIR dump found under {dump_dir}")
    return sorted(candidates, key=lambda item: item.stat().st_size)[-1]


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(f"tool not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        joined = " ".join(command)
        details = "\n".join(
            part
            for part in [
                f"command failed with exit code {exc.returncode}: {joined}",
                exc.stdout,
                exc.stderr,
            ]
            if part
        )
        raise RuntimeError(details) from exc
