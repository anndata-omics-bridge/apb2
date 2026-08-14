"""The ``[fragments]`` block: explode packed parallel-list columns, one row per fragment.

DIA-NN-style reports pack per-fragment values as delimiter-joined lists inside each
precursor row (``Fragment.Info`` plus parallel ``Fragment.Quant.*`` lists, aligned by
index, often terminated by a trailing delimiter). ``exploder_for(rule, …)`` reads
``label_strategy`` once; each exploder derives its own trim set at construction — the
explode multiplies the row count ~12x, so only the columns the rule reads survive it.
"""

from __future__ import annotations

import math
from typing import override

import pandas as pd

from apb2.vendor_parse_rules.model import (
    ColumnLabeledFragments,
    Fragments,
    LongRule,
    PositionalFragments,
    WideRule,
)
from apb2.vendor_parse_rules.runtime import recognition_for


def _split_packed(value: object, delimiter: str) -> list[str]:
    """Split one packed cell into tokens, dropping a trailing empty terminator."""
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    ):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.rstrip(delimiter)  # drop DIA-NN's trailing-delimiter terminator
    if not text:
        return []
    return [token.strip() for token in text.split(delimiter)]


def _positions(tokens: list[str]) -> list[int]:
    """Positions for one token list, exploded alongside the values."""
    return list(range(len(tokens)))


def _columns_read_by(rule: LongRule | WideRule, modification_sources: frozenset[str]) -> set[str]:
    """Every raw column the rule reads: the only ones worth multiplying ~12x."""
    needed: set[str] = set(modification_sources)
    for _axis, group in recognition_for(rule).column_groups():
        needed.update(group.select.values())
        needed.update(group.optional_select.values())
        for column in group.computed:
            needed.update(column.inputs)
    needed.update(layer.source for layer in rule.layers)
    return needed


class _PackedExplode:
    """Shared mechanics of both exploders: trim, split, explode, coerce."""

    def __init__(
        self,
        fragments: PositionalFragments | ColumnLabeledFragments,
        rule: LongRule | WideRule,
        modification_sources: frozenset[str],
    ) -> None:
        self.value_columns = tuple(fragments.value_columns)
        self.delimiter = fragments.delimiter
        self.label_output = fragments.label_output
        self._needed = frozenset(_columns_read_by(rule, modification_sources)) | set(
            self.packed_columns()
        )

    def packed_columns(self) -> tuple[str, ...]:
        return self.value_columns

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim to the read set and split each packed column into token lists."""
        keep = [column for column in df.columns if column in self._needed]
        packed = self.packed_columns()
        missing = [column for column in packed if column not in df.columns]
        if missing:
            raise KeyError(
                f"[fragments] references column(s) missing from the input: {missing}; "
                f"available: {list(df.columns)[:10]}…"
            )
        work = df[keep].copy()
        for column in packed:
            work[column] = work[column].map(lambda value: _split_packed(value, self.delimiter))
        return work

    def _coerce_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce exploded values to numeric so they ride the frame as float64."""
        for column in self.value_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df


class PositionalExplode(_PackedExplode):
    """Fan packed fragment values out, labelling them ``frag_0``, ``frag_1``, … by index."""

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._split(df)
        # A parallel index list per precursor, exploded alongside the values.
        first = self.value_columns[0]
        work["_frag_pos"] = work[first].map(_positions)
        work = work.explode([*self.value_columns, "_frag_pos"], ignore_index=True)
        work = work.dropna(subset=[first]).reset_index(drop=True)
        work[self.label_output] = [f"frag_{int(pos)}" for pos in work["_frag_pos"]]
        work = work.drop(columns=["_frag_pos"])
        return self._coerce_values(work)


class ColumnLabeledExplode(_PackedExplode):
    """Fan packed fragment values out, taking each label from a packed label column."""

    def __init__(
        self,
        fragments: ColumnLabeledFragments,
        rule: LongRule | WideRule,
        modification_sources: frozenset[str],
    ) -> None:
        self.label_column = fragments.label_column
        super().__init__(fragments, rule, modification_sources)

    @override
    def packed_columns(self) -> tuple[str, ...]:
        return (self.label_column, *self.value_columns)

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self._split(df)
        # Multi-column explode keeps the lists aligned and raises if their lengths differ.
        work = work.explode(list(self.packed_columns()), ignore_index=True)
        # Precursors with no fragments explode to a NaN row; drop them.
        work = work.dropna(subset=[self.label_column]).reset_index(drop=True)
        work[self.label_output] = work[self.label_column].astype(str).str.split("/").str[0]
        # Drop the packed label column so the (now ~12x longer) frame does not carry a
        # redundant long string column.
        work = work.drop(columns=[self.label_column])
        return self._coerce_values(work)


class NoFragments:
    """The rule declares no fragment expansion, so the frame passes through untouched."""

    def packed_columns(self) -> tuple[str, ...]:
        return ()

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


type FragmentExploder = PositionalExplode | ColumnLabeledExplode | NoFragments


def exploder_for(
    rule: LongRule | WideRule,
    modification_sources: frozenset[str],
) -> FragmentExploder:
    """Read the rule's ``label_strategy`` once, and return the exploder it names."""
    fragments: Fragments | None = rule.fragments
    if fragments is None:
        return NoFragments()
    if isinstance(fragments, ColumnLabeledFragments):
        return ColumnLabeledExplode(fragments, rule, modification_sources)
    return PositionalExplode(fragments, rule, modification_sources)
