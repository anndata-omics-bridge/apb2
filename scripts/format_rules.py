"""Lay out apb2's packaged rules.json files: an object or list on one line where it fits, else
one item per line.

``make format`` rewrites the files after an edit; ``--check`` lists the files out of layout and
exits 1, as the rule layout test runs it. apb-export lays out its export rules the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cyclopts import App

from apb2.parserV2.vendor_parse_rules.loader import PACKAGED

app = App(name="format-rules", help=__doc__)

WIDTH = 100
"""The widest line an object or list stays on."""
_INDENT = 2

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


def format_rule(document: Json) -> str:
    """A rule document as its file holds it."""
    return _render(document, 0, 0) + "\n"


def _render(value: Json, indent: int, start: int) -> str:
    """One value whose first character sits in column ``start``, nested ``indent`` deep."""
    compact = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if not isinstance(value, dict | list) or not value or start + len(compact) < WIDTH:
        return compact
    pad = " " * (indent + _INDENT)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            prefix = f"{pad}{json.dumps(key, ensure_ascii=False)}: "
            lines.append(prefix + _render(item, indent + _INDENT, len(prefix)))
        return "{\n" + ",\n".join(lines) + "\n" + " " * indent + "}"
    items = [pad + _render(item, indent + _INDENT, len(pad)) for item in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


@app.default
def main(*, check: bool = False) -> None:
    """Rewrite every packaged rules.json in its layout, or with --check only list the others."""
    stale: list[Path] = []
    for path in PACKAGED:
        text = path.read_text(encoding="utf-8")
        formatted = format_rule(json.loads(text))
        if formatted == text:
            continue
        stale.append(path)
        if not check:
            path.write_text(formatted, encoding="utf-8")
    for path in stale:
        print(path)
    if check and stale:
        sys.exit(1)


if __name__ == "__main__":
    app()
