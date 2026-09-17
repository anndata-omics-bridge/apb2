"""Join AlphaDIA's quantitative matrix with its precursor metadata."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

_KEY = "mod_seq_charge_hash"
_IDENTITY = ("sequence", "charge", "mods", "mod_sites", "genes", "decoy")
_ANNOTATIONS = (*_IDENTITY, "proteins", "pg", "pg_master", "channel")


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Identify matrix and precursor inputs by their columns, including renamed files."""
    selected: dict[str, str] = {}
    for name, columns in headers.items():
        if _KEY not in columns:
            continue
        role = "precursors" if set(_IDENTITY) <= set(columns) else "matrix"
        if role in selected:
            raise ValueError(f"AlphaDIA has multiple {role} inputs: {selected[role]}, {name}")
        selected[role] = name
    return selected


def join(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Enrich authoritative matrix values, returning the long table described by the rule.

    The secondary table supplies only feature metadata, never substitute intensities.
    A precursor-only input already has the prepared long layout.
    """
    if "precursors" not in tables:
        raise ValueError("AlphaDIA matrix requires its precursor metadata companion")
    precursors = tables["precursors"]
    if "matrix" not in tables:
        return precursors
    matrix = tables["matrix"]
    metadata = precursors.select(_KEY, *(c for c in _ANNOTATIONS if c in precursors.columns))
    metadata = metadata.unique(maintain_order=True)
    if metadata[_KEY].null_count() or metadata[_KEY].is_duplicated().any():
        raise ValueError("AlphaDIA precursor metadata has missing hashes or conflicting identities")
    if matrix[_KEY].null_count() or matrix[_KEY].is_duplicated().any():
        raise ValueError("AlphaDIA matrix has missing or duplicate precursor hashes")
    missing = matrix.select(_KEY).join(metadata.select(_KEY), on=_KEY, how="anti")
    if missing.height:
        raise ValueError(f"AlphaDIA matrix has {missing.height} hashes without precursor metadata")
    samples = [column for column in matrix.columns if column != _KEY]
    if not samples:
        raise ValueError("AlphaDIA matrix has no quantitative sample columns")
    return matrix.unpivot(on=samples, index=_KEY, variable_name="run", value_name="intensity").join(
        metadata, on=_KEY, how="left", validate="m:1", maintain_order="left"
    )
