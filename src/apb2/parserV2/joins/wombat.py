"""Bring a WOMBAT-P peptide table to its levels: drop rows that name several peptides."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
from loguru import logger

_SEQUENCE = "modified_peptide"
_SIGNATURE = (_SEQUENCE, "protein_group")
_SEPARATOR = "|"
_EXAMPLE_LIMIT = 3


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Identify the WOMBAT-P table by its peptide and protein-group columns."""
    selected = [name for name, columns in headers.items() if set(_SIGNATURE) <= set(columns)]
    if len(selected) > 1:
        raise ValueError(f"WOMBAT has multiple peptide inputs: {', '.join(selected)}")
    return {"peptides": selected[0]} if selected else {}


def join(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Keep the rows that name one peptide; log how many named several.

    FlashLFQ writes a peak it assigns to several peptides as ``SEQA|SEQB``; such a row is no
    single ion, and sequence parsing would otherwise glue its parts into a peptide that does
    not exist.
    """
    frame = tables["peptides"]
    ambiguous = pl.col(_SEQUENCE).str.contains(_SEPARATOR, literal=True).fill_null(False)
    dropped = frame.filter(ambiguous)
    if dropped.height:
        logger.warning(
            "wombat dropped {} of {} rows naming several peptides, examples={}",
            dropped.height,
            frame.height,
            dropped.get_column(_SEQUENCE).head(_EXAMPLE_LIMIT).to_list(),
        )
    return frame.filter(~ambiguous)
