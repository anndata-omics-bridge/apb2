"""Bring MetaMorpheus (FlashLFQ) peaks close to ion level: drop ambiguous and decoy peaks."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

_MAPPED = "Full Sequences Mapped"
_SIGNATURE = (_MAPPED, "Full Sequence", "Peak intensity")


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Identify the AllQuantifiedPeaks table by its FlashLFQ columns, including renamed files."""
    selected = [name for name, columns in headers.items() if set(_SIGNATURE) <= set(columns)]
    if len(selected) > 1:
        raise ValueError(f"MetaMorpheus has multiple peak inputs: {', '.join(selected)}")
    return {"peaks": selected[0]} if selected else {}


def join(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Keep target peaks mapped to exactly one peptidoform, with numeric intensities.

    FlashLFQ writes an ambiguous peak as ``SEQA|SEQB``; sequence parsing would otherwise
    concatenate it into a peptide that does not exist. Decoy peptides and FlashLFQ's random-RT
    match-between-runs peaks are decoys, not measurements. Prepared tables are read as text, and
    the rule takes the max of repeated peaks of one ion and run, which needs native numbers.
    """
    return (
        tables["peaks"]
        .filter(
            pl.col(_MAPPED) == "1",
            pl.col("Decoy Peptide") == "False",
            pl.col("Random RT") == "False",
        )
        .with_columns(pl.col("Peak intensity").cast(pl.Float64, strict=True))
    )
