"""``ModificationApplier``: normalize modified sequences on a frame.

The contract, its three implementations, and the factory that selects between them are all
here — the dispatch is on which settings record arrived, and both settings types are this
package's own, so nothing about a rule is needed to choose. What stays in
``configure_parse`` is only the rule reading: whether a block is declared, whether anything
consumes its output, and building the settings from the declaration.

Appliers memoize on unique source values: normalization is a pure function of the source
columns, so a column with a million rows and fifty thousand distinct sequences tokenizes
fifty thousand times, not a million.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Protocol

import pandas as pd

from apb2.parse_quant.modifications.modified_sequence import ModifiedSequence
from apb2.parse_quant.modifications.normalize_sequence import (
    SiteListSettings,
    TokenRegexSettings,
    normalize_site_list,
    normalize_token_regex,
)


class ModificationApplier(Protocol):
    """Add the normalized modification columns to a frame, and declare what it touches.

    ``sources`` are the raw vendor columns that must be present; ``source_columns()`` adds
    the names it writes, which together are what the read plan must project.
    """

    sources: tuple[str, ...]

    def source_columns(self) -> frozenset[str]: ...

    def apply(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


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

    def __init__(self, columns: SequenceColumns, settings: TokenRegexSettings) -> None:
        self.columns = columns
        self.sources = columns.sources
        self._settings = settings

    def source_columns(self) -> frozenset[str]:
        return self.columns.declared

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns.require(df)
        results = _normalize_once_per_distinct(
            df[self.sources[0]].astype(str),
            lambda sequence: normalize_token_regex(sequence, self._settings),
        )
        return self.columns.write(df, results)


class SiteListApplier:
    """Normalize parallel name/site columns beside a bare sequence (alphabase layout)."""

    def __init__(self, columns: SequenceColumns, settings: SiteListSettings) -> None:
        self.columns = columns
        self.sources = columns.sources
        self._settings = settings

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
        results = _normalize_once_per_distinct(
            rows, lambda key: normalize_site_list(*key, self._settings)
        )
        return self.columns.write(df, results)


class NoModifications:
    """No modification normalization runs: none is declared, or nothing reads its output.

    Constructed by ``configure_parse.modifications_for``, not by ``applier_for`` below —
    both of those are questions about the rule, and neither is a kind of settings.
    """

    sources: tuple[str, ...] = ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def source_columns(self) -> frozenset[str]:
        return frozenset()


_IMPLEMENTS: tuple[type[ModificationApplier], ...] = (
    TokenRegexApplier,
    SiteListApplier,
    NoModifications,
)
"""Every class claiming the protocol, wherever it is constructed — pyright checks each one
against it here, at its definition site. Not a list of what ``applier_for`` returns:
``NoModifications`` is constructed by ``configure_parse.modifications_for`` instead.
"""


def applier_for(
    columns: SequenceColumns, settings: TokenRegexSettings | SiteListSettings
) -> ModificationApplier:
    """Return the applier that normalizes the kind of settings it was given.

    Two of the three implementations, because only two are a *kind of settings*. The
    identity applier answers "no block is declared" and "nothing reads its output", which
    are questions about the rule; giving this signature a third settings type meaning
    "none" would be ``| None`` wearing a class.
    """
    if isinstance(settings, SiteListSettings):
        return SiteListApplier(columns, settings)
    return TokenRegexApplier(columns, settings)
