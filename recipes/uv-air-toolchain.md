# uv AIR Toolchain

## 问题

`uv run` 会把 `.venv/bin` 放在 `PATH` 最前面。当前 `.venv/bin/aircc` 是失效 wrapper，
容易报：

```text
ImportError: cannot import name 'main' from air.compiler.aircc.main
```

## 做法

启动 AIR Python builder 或 `XRTRunner` 前，确认 `aircc` 命中 wheel 里的真实 binary：

```bash
source scripts/air-env.sh
command -v aircc
aircc --help | head
```

`command -v aircc` 应该指向 `site-packages/mlir_air/bin/aircc`。如果指向
`.venv/bin/aircc`，删掉这个 wrapper；它会导入已经退休的 Python driver。

用 `uv` 包一层 `python3` 给官方 Makefile 调用时，保持调用方当前目录：

```bash
exec uv --project "$ROOT_DIR" run --no-sync python "$@"
```

不要先 `cd "$ROOT_DIR"` 再执行 Python。AIR backend 会在当前目录创建 `air_project/`
并查找 `link_with` 的 `.o` 文件；改变 cwd 会让外部 kernel 链接失败。

## 检查

```bash
source scripts/npu-common.sh
check_npu_device
```

期望看到当前仓库 `.venv` 里的 `pyxrt`，以及 `RyzenAI-npu4` 设备。

## 不要

- 不要使用其他项目的 `.venv`。
- 不要直接相信 `uv run aircc` 命中的 binary。
- 不要让 Python shim 改变官方 Makefile 的 build cwd。
- 不要在函数内部临时 import 工具链模块。
