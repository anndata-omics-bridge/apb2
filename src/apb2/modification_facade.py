"""Public read-only access to APB2's canonical modification vocabulary."""

from __future__ import annotations

from collections.abc import Mapping

from apb2.parserV2.vendor_params.parsers.shared.unimod import UNIMOD_REGISTRY


def canonical_modification_names() -> Mapping[str, str]:
    """Return canonical modification names keyed by Unimod accession."""
    return UNIMOD_REGISTRY.names_by_accession()
