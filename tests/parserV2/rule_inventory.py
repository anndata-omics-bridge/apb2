"""The packaged rule inventory Parser V2 ships."""

from __future__ import annotations

from pathlib import Path

from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel

EXPECTED_DOCUMENT_COUNT = 16
EXPECTED_LEVEL_COUNT = 27


def document_key(path: Path) -> str:
    """Identify one packaged document by its path below ``documents/``."""
    parts = path.parts
    root = len(parts) - 1 - parts[::-1].index("documents")
    return "/".join(parts[root + 1 : -1])


def packaged_levels() -> tuple[tuple[str, QuantificationLevel], ...]:
    """Every ``(document key, level)`` Parser V2 declares."""
    return tuple(
        (document_key(path), level)
        for path in PACKAGED
        for level in load_rule_document(path).levels
    )
