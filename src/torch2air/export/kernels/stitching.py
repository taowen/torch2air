from __future__ import annotations

import re


_RETURN_RE = re.compile(r"^\s*return(\s|$|//|loc\()")


def extract_between_func_and_return(mlir_text: str) -> str:
    lines = mlir_text.splitlines()
    body_start = -1
    for line_index, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = line_index + 1
            break
    if body_start < 0:
        raise ValueError("MLIR module does not contain a public func.func")

    body_end = -1
    for line_index in range(len(lines) - 1, body_start, -1):
        if _RETURN_RE.match(lines[line_index]):
            body_end = line_index
            break
    if body_end < 0:
        raise ValueError("MLIR function does not contain a return terminator")
    return "\n".join(lines[body_start:body_end])


def extract_affine_maps(mlir_text: str) -> list[str]:
    return [line for line in mlir_text.splitlines() if line.startswith("#map")]


def rename_all(text: str, prefix: str) -> str:
    for name in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)

    for name in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)

    for name in sorted(set(re.findall(r"%\d+", text)), key=lambda value: int(value[1:]), reverse=True):
        text = re.sub(re.escape(name) + r"(?!\d)", f"%{prefix}_n{name[1:]}", text)

    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text


def fix_launch_func_args(text: str, prefix: str, arg_map: dict[int, int]) -> str:
    for original_index, combined_index in arg_map.items():
        old_ref = f"%{prefix}_arg{original_index}"
        new_ref = f"%arg{combined_index}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


def stitch_quantized_qwen3_embed_norm(
    *,
    embed_dma_mlir: str,
    norm_dma_mlir: str,
    sequence_length: int,
    blocks_per_row: int,
    function_name: str = "run_pipeline_embed_norm",
) -> str:
    hidden_size = blocks_per_row * 256
    row_words = blocks_per_row * 36

    embed_body = rename_all(extract_between_func_and_return(embed_dma_mlir), "emb")
    embed_body = fix_launch_func_args(embed_body, "emb", {0: 0, 1: 1, 2: 3})

    norm_body = rename_all(extract_between_func_and_return(norm_dma_mlir), "norm")
    norm_body = fix_launch_func_args(norm_body, "norm", {0: 3, 1: 2, 2: 4})

    maps = [
        *[rename_all(line, "emb") for line in extract_affine_maps(embed_dma_mlir)],
        *[rename_all(line, "norm") for line in extract_affine_maps(norm_dma_mlir)],
    ]
    maps_text = "\n".join(maps)
    if maps_text:
        maps_text += "\n"

    return f"""{maps_text}module {{
  func.func @{function_name}(
      %arg0: memref<{sequence_length}x{row_words}xi32>,
      %arg1: memref<{sequence_length}x{blocks_per_row}x2xf32>,
      %arg2: memref<{hidden_size}xf32>,
      %arg3: memref<{sequence_length}x{hidden_size}xf32>,
      %arg4: memref<{sequence_length}x{hidden_size}xf32>) {{
{embed_body}
{norm_body}
    return
  }}
}}
"""
