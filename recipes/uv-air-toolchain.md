# uv AIR Toolchain

## 问题

`uv run` 会把 `.venv/bin` 放在 `PATH` 最前面。当前 `.venv/bin/aircc` 是失效 wrapper，
容易报：

```text
ImportError: cannot import name 'main' from air.compiler.aircc.main
```

## 做法

启动 AIR Python builder 或 `XRTRunner` 前，把 wheel 里的真实工具路径放到 `PATH` 前面：

```python
site_packages = Path(sysconfig.get_paths()["purelib"])
mlir_air = site_packages / "mlir_air"
mlir_aie = site_packages / "mlir_aie"
peano = site_packages / "llvm-aie"
os.environ["PATH"] = (
    f"{mlir_air / 'bin'}:{mlir_aie / 'bin'}:{peano / 'bin'}:"
    f"{os.environ.get('PATH', '')}"
)
os.environ["MLIR_AIR_INSTALL_DIR"] = str(mlir_air)
os.environ["MLIR_AIE_INSTALL_DIR"] = str(mlir_aie)
os.environ["PEANO_INSTALL_DIR"] = str(peano)
```

## 检查

```bash
source scripts/npu-common.sh
check_npu_device
```

期望看到当前仓库 `.venv` 里的 `pyxrt`，以及 `RyzenAI-npu4` 设备。

## 不要

- 不要使用其他项目的 `.venv`。
- 不要直接相信 `uv run aircc` 命中的 binary。
- 不要在函数内部临时 import 工具链模块。
