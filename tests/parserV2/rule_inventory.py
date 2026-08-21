"""The packaged rule inventory both generations must agree on.

The unchanged schema-0.2 package is Parser V2's parity oracle, so the set of documents
and effective levels is stated once, here, and every migration test compares against it
rather than against a number written into an assertion.
"""

from __future__ import annotations

from pathlib import Path

from apb2.vendor_parse_rules.model import QuantificationLevel
from apb2.vendor_parse_rules.rules import PACKAGED, load_document

EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_LEVEL_COUNT = 19


def document_key(path: Path) -> str:
    """Identify one packaged document by its path below ``documents/``."""
    parts = path.parts
    root = len(parts) - 1 - parts[::-1].index("documents")
    return "/".join(parts[root + 1 : -1])


def legacy_levels() -> tuple[tuple[str, QuantificationLevel], ...]:
    """Every ``(document key, level)`` the unchanged schema-0.2 package declares."""
    return tuple(
        (document_key(path), level) for path in PACKAGED for level in load_document(path).levels
    )
