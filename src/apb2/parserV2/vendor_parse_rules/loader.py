"""``load_rule_document(path)``: the entry point of this package. Nothing else here is called.

This folder deserializes rules.json, and that is all it does. Load a file and you have the
whole of it. ``PACKAGED`` is the list of files Parser V2 ships, as paths: data, not a function.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import cast

from apb2.parserV2.vendor_parse_rules.document import (
    JsonDict,
    RuleDocument,
    make_rule_document,
)


def _packaged() -> tuple[Path, ...]:
    root = Path(str(resources.files("apb2.parserV2.vendor_parse_rules.documents")))
    return tuple(sorted(set(root.glob("*/rules.json")) | set(root.glob("*/v*/rules.json"))))


PACKAGED: tuple[Path, ...] = _packaged()
"""Every schema-0.3 rules.json Parser V2 ships, in stable path order."""


def load_rule_document(path: Path) -> RuleDocument:
    """Read one rules.json. The single entry point of this package.

    Raises when the file is not a readable schema-0.3 document, so a ``RuleDocument`` in hand
    means the shell parsed; a level's blocks are validated when that level is composed, which
    is the only place they can be — a level is validated as a whole rule.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: document must be a JSON object")
    return make_rule_document(path, cast("JsonDict", raw))
