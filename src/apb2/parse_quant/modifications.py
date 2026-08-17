"""The ``[modifications]`` block at runtime: normalize modified sequences on a frame.

``applier_for(rule)`` reads the ``parser`` selector once and decides whether
normalization runs at all — is a block declared, does any compute read its output —
instead of on every table. Appliers memoize on unique source values: normalization is a
pure function of the source columns, so a column with a million rows and fifty thousand
distinct sequences tokenizes fifty thousand times, not a million.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

import pandas as pd

from apb2.modifications.apply_rules import (
    MapEntry,
    ModificationRule,
    SiteListRule,
    apply_rule,
    apply_site_list,
)
from apb2.modifications.model import ModifiedSequence
from apb2.modifications.unimod_registry import resolve
from apb2.vendor_parse_rules.model import (
    LongRule,
    Modifications,
    ProformaSequence,
    SiteListModifications,
    StrippedSequence,
    TokenRegexModifications,
    WideRule,
    modification_outputs,
)


def modification_sources(modifications: Modifications) -> tuple[str, ...]:
    """The raw vendor columns one modifications declaration reads."""
    if isinstance(modifications, SiteListModifications):
        return (
            modifications.sequence_column,
            modifications.modification_column,
            modifications.site_column,
        )
    return (modifications.source_column,)


def _map_entries(mods: Modifications) -> tuple[MapEntry, ...]:
    """Fill ``name``, ``target``, ``position``, ``mass_delta`` from the bundled registry.

    Raises ``KeyError`` if an entry references an unknown accession.
    """
    entries: list[MapEntry] = []
    for e in mods.map:
        record = resolve(e.accession)
        entries.append(
            MapEntry(
                token=e.token,
                name=record.name,
                accession=record.accession,
                target=tuple(record.target),
                position=record.position,
                mass_delta=record.mass_delta,
            )
        )
    return tuple(entries)


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

    def __init__(self, modifications: Modifications) -> None:
        self.output_column = modifications.output_column
        self.sources = modification_sources(modifications)
        self.declared = frozenset(self.sources) | modification_outputs(modifications)

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
    """Normalize inline modification tokens (``PEPM[15.9949]TIDE``) with one regex.

    Construction resolves the Unimod map, so an unknown accession fails before any table
    is read.
    """

    def __init__(self, modifications: TokenRegexModifications) -> None:
        self.columns = SequenceColumns(modifications)
        self.sources = self.columns.sources
        self._rule = ModificationRule(
            token_pattern=modifications.token_pattern,
            token_position=modifications.token_position,
            case_sensitive=modifications.case_sensitive,
            unknown_policy=modifications.unknown_policy,
            entries=_map_entries(modifications),
        )

    def source_columns(self) -> frozenset[str]:
        return self.columns.declared

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.columns.require(df)
        results = _normalize_once_per_distinct(
            df[self.sources[0]].astype(str), lambda sequence: apply_rule(sequence, self._rule)
        )
        return self.columns.write(df, results)


class SiteListApplier:
    """Normalize parallel name/site columns beside a bare sequence (alphabase layout).

    Construction resolves the Unimod map, so an unknown accession fails before any table
    is read.
    """

    def __init__(self, modifications: SiteListModifications) -> None:
        self.columns = SequenceColumns(modifications)
        self.sources = self.columns.sources
        self._rule = SiteListRule(
            delimiter=modifications.delimiter,
            site_base=modifications.site_base,
            case_sensitive=modifications.case_sensitive,
            unknown_policy=modifications.unknown_policy,
            entries=_map_entries(modifications),
        )

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


def applier_for(rule: LongRule | WideRule) -> ModificationApplier:
    """Read the ``parser`` selector once; return the applier it names, or the identity.

    The absence questions — is a block declared, does anything read its output — are
    asked once when the applier is built instead of on every table.
    """
    modifications = rule.modifications
    if modifications is None:
        return NoModifications()
    consumed = any(
        isinstance(column, ProformaSequence | StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return NoModifications()
    if isinstance(modifications, SiteListModifications):
        return SiteListApplier(modifications)
    return TokenRegexApplier(modifications)
