"""Give MSAngel's peptide ions their protein set's columns: one row per ion and protein set."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

_IONS = "Quantified peptide ions"
_PROTEIN_SETS = "Protein sets"
_KEY = "protein_set_id"

SHEETS = (_IONS, _PROTEIN_SETS)
"""The workbook sheets this preparation reads."""

KEY_COLUMNS = (_KEY,)
"""Columns ``join`` reads besides the rules' own: the protein set of each row."""


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Bind the ion sheet, which MSAngel's raw abundances mark, and its protein sets if joinable.

    ProlineStudio writes the same ion sheet without ``raw_abundance_`` columns; its rule reads
    that sheet directly. Protein sets join only when both sheets carry ``protein_set_id``.
    """
    ions = headers.get(_IONS, ())
    if not any(column.startswith("raw_abundance_") for column in ions):
        return {}
    if _KEY in ions and _KEY in headers.get(_PROTEIN_SETS, ()):
        return {"ion": _IONS, "protein": _PROTEIN_SETS}
    return {"ion": _IONS}


def join(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Add each ion row's protein-set columns; ions without protein sets are only namespaced.

    The ion sheet lists a shared peptide once per protein set, so a row is one ion in one
    protein set and stays one: the ion rule picks an ion's protein set by its score, and the
    protein level reads every set's own columns from the rows of its ions.
    """
    ions = _prefix(tables["ion"], "ion")
    proteins = tables.get("protein")
    if proteins is None:
        return ions
    if proteins[_KEY].null_count() or proteins[_KEY].is_duplicated().any():
        raise ValueError(f"MSAngel {_PROTEIN_SETS!r} has missing or duplicate {_KEY}")
    joined = ions.join(
        _prefix(proteins, "protein"),
        left_on=f"ion.{_KEY}",
        right_on=f"protein.{_KEY}",
        how="left",
        coalesce=False,
        validate="m:1",
        maintain_order="left",
    )
    unmatched = joined.filter(
        pl.col(f"ion.{_KEY}").is_not_null() & pl.col(f"protein.{_KEY}").is_null()
    )
    if unmatched.height:
        raise ValueError(
            f"MSAngel: {unmatched.height} ion rows name a protein set {_PROTEIN_SETS!r} lacks, "
            f"e.g. {unmatched.get_column(f'ion.{_KEY}').head(3).to_list()}"
        )
    return joined


def _prefix(frame: pl.DataFrame, role: str) -> pl.DataFrame:
    return frame.rename({column: f"{role}.{column}" for column in frame.columns})
