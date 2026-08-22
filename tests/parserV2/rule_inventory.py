"""The packaged rule inventory Parser V2 and the external APB oracle must agree on."""

from __future__ import annotations

from pathlib import Path

from anndata_proteomics.vendor_quant_rules._discovery import iter_packaged_documents
from anndata_proteomics.vendor_quant_rules.loader import load_rule_document
from anndata_proteomics.vendor_quant_rules.schema.components import QuantificationLevel

EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_LEVEL_COUNT = 19


def document_key(path: Path) -> str:
    """Identify one packaged document by its path below ``documents/``."""
    parts = path.parts
    root = len(parts) - 1 - parts[::-1].index("documents")
    return "/".join(parts[root + 1 : -1])


def oracle_levels() -> tuple[tuple[str, QuantificationLevel], ...]:
    """Every ``(document key, level)`` the external APB oracle declares."""
    return tuple(
        (document_key(path), level)
        for path in iter_packaged_documents()
        for level in load_rule_document(path).levels
    )
