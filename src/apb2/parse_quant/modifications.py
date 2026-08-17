"""Normalize modified sequences on a frame: the runtime of the ``[modifications]`` block.

Appliers memoize on unique source values: normalization is a pure function of the source
columns, so a column with a million rows and fifty thousand distinct sequences tokenizes
fifty thousand times, not a million. Which applier runs, and whether one runs at all, is
decided once by ``selectors.applier_for``; each applier is constructed with the columns it
reads and writes plus a finished engine rule, and never sees a rule declaration.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

import pandas as pd

from apb2.modifications.apply_rules import (
    ModificationRule,
    SiteListRule,
    apply_rule,
    apply_site_list,
)
from apb2.modifications.model import ModifiedSequence


def _normalize_once_per_distinct[K: Hashable](
    keys: Iterable[K], normalize: Callable[[K], ModifiedSequence]
) -> list[ModifiedSequence]:
    """Normalize every distinct key once and replay the result per row.

    Normalization is a pure function of the source values, so a column of a million rows
    with fifty thousand distinct sequences tokenizes fifty thousand times.
    """
    memo: dict[K, ModifiedSequence] = {}
    results: list[ModifiedSequence] = []
    for key in keys:
        if key not in memo:
            memo[key] = normalize(key)
        results.append(memo[key])
    return results


class SequenceColumns:
    """The frame-side mechanics both appliers hold: check the sources, write the outputs.

    A collaborator, never a base class: each applier owns one of these and calls it.
    ``write`` adds three columns in place and returns the same frame: the rule's
    ``output_column`` (ProForma string), ``stripped_sequence`` (amino-acid-only), and
    ``unknown_mod_tokens`` (unresolved vendor tokens per row).
    """

    def __init__(
        self, *, output_column: str, sources: tuple[str, ...], outputs: frozenset[str]
    ) -> None:
        self.output_column = output_column
        self.sources = sources
        self.declared = frozenset(sources) | outputs

    def require(self, df: pd.DataFrame) -> None:
        missing = [column for column in self.sources if column not in df.columns]
        if missing:
            raise KeyError(
                f"[modifications] needs column(s) {missing} not found in DataFrame; "
                f"available: {list(df.columns)[:10]}…"
            )

    def write(self, df: pd.DataFrame, results: list[ModifiedSequence]) -> pd.DataFrame:
        df[self.output_column] = [r.proforma_sequence for r in results]
        df["stripped_sequence"] = [r.stripped_sequence for r in results]
        df["unknown_mod_tokens"] = [r.unknown_tokens for r in results]
        return df


class TokenRegexApplier:
    """Normalize inline modification tokens (``PEPM[15.9949]TIDE``) with one regex."""

    def __init__(self, columns: SequenceColumns, rule: ModificationRule) -> None:
        self.columns = columns
        self.sources = columns.sources
        self._rule = rule

    def source_columns(self) -> frozenset[str]:
        return self.columns.declared

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns.require(df)
        results = _normalize_once_per_distinct(
            df[self.sources[0]].astype(str), lambda sequence: apply_rule(sequence, self._rule)
        )
        return self.columns.write(df, results)


class SiteListApplier:
    """Normalize parallel name/site columns beside a bare sequence (alphabase layout)."""

    def __init__(self, columns: SequenceColumns, rule: SiteListRule) -> None:
        self.columns = columns
        self.sources = columns.sources
        self._rule = rule

    def source_columns(self) -> frozenset[str]:
        return self.columns.declared

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns.require(df)
        sequence_column, modification_column, site_column = self.sources
        rows = zip(
            df[sequence_column].astype(str),
            df[modification_column].fillna("").astype(str),
            df[site_column].fillna("").astype(str),
            strict=True,
        )
        results = _normalize_once_per_distinct(rows, lambda key: apply_site_list(*key, self._rule))
        return self.columns.write(df, results)


class NoModifications:
    """No modification normalization runs: none is declared, or none is read."""

    sources: tuple[str, ...] = ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def source_columns(self) -> frozenset[str]:
        return frozenset()


type ModificationApplier = TokenRegexApplier | SiteListApplier | NoModifications
