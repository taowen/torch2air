from __future__ import annotations

import json
import pprint
from collections.abc import Mapping, Sequence
from pathlib import Path


type TemplateValue = (
    str
    | int
    | float
    | bool
    | None
    | Sequence[TemplateValue]
    | Mapping[str, TemplateValue]
    | Mapping[int, TemplateValue]
)


def render_template(
    template_dir: str | Path | list[str | Path],
    template_name: str,
    **context: TemplateValue,
) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    loader_path: str | list[str]
    if isinstance(template_dir, list):
        loader_path = [str(Path(path)) for path in template_dir]
    else:
        loader_path = str(Path(template_dir))
    env = Environment(
        autoescape=False,
        keep_trailing_newline=True,
        loader=FileSystemLoader(loader_path),
        lstrip_blocks=True,
        trim_blocks=True,
        undefined=StrictUndefined,
    )
    env.filters["tojson"] = lambda value: json.dumps(value, sort_keys=True)
    env.filters["pyrepr"] = lambda value: pprint.pformat(value, sort_dicts=True, width=120)
    return env.get_template(template_name).render(**context)


def render_to_file(
    template_dir: str | Path | list[str | Path],
    template_name: str,
    output_path: str | Path,
    **context: TemplateValue,
) -> None:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(render_template(template_dir, template_name, **context))
