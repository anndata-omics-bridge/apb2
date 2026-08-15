"""The ``[modifications]`` block at runtime: normalize modified sequences on a frame.

``applier_for(rule)`` reads the ``parser`` selector once and decides whether
normalization runs at all — is a block declared, does any compute read its output —
instead of on every table. Appliers memoize on unique source values: normalization is a
pure function of the source columns, so a column with a million rows and fifty thousand
distinct sequences tokenizes fifty thousand times, not a million.
"""

from __future__ import annotations

from typing import override

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


class _NormalizeSequences:
    """Shared applier mechanics: column checks, memoized results, output assignment.

    Construction resolves the Unimod map, so an unknown accession fails before any table
    is read. ``apply`` adds three columns in place and returns the same frame: the rule's
    ``output_column`` (ProForma string), ``stripped_sequence`` (amino-acid-only), and
    ``unknown_mod_tokens`` (unresolved vendor tokens per row).
    """

    def __init__(self, modifications: Modifications) -> None:
        self.output_column = modifications.output_column
        self.sources = modification_sources(modifications)
        self._outputs = modification_outputs(modifications)

    def source_columns(self) -> frozenset[str]:
        return frozenset(self.sources) | self._outputs

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.sources if column not in df.columns]
        if missing:
            raise KeyError(
                f"[modifications] needs column(s) {missing} not found in DataFrame; "
                f"available: {list(df.columns)[:10]}…"
            )
        results = self._results(df)
        df[self.output_column] = [r.proforma_sequence for r in results]
        df["stripped_sequence"] = [r.stripped_sequence for r in results]
        df["unknown_mod_tokens"] = [r.unknown_tokens for r in results]
        return df

    def _results(self, df: pd.DataFrame) -> list[ModifiedSequence]:
        raise NotImplementedError


class TokenRegexApplier(_NormalizeSequences):
    """Normalize inline modification tokens (``PEPM[15.9949]TIDE``) with one regex."""

    def __init__(self, modifications: TokenRegexModifications) -> None:
        super().__init__(modifications)
        self._rule = ModificationRule(
            token_pattern=modifications.token_pattern,
            token_position=modifications.token_position,
            case_sensitive=modifications.case_sensitive,
            unknown_policy=modifications.unknown_policy,
            entries=_map_entries(modifications),
        )

    @override
    def _results(self, df: pd.DataFrame) -> list[ModifiedSequence]:
        memo: dict[str, ModifiedSequence] = {}
        out: list[ModifiedSequence] = []
        for sequence in df[self.sources[0]].astype(str):
            if sequence not in memo:
                memo[sequence] = apply_rule(sequence, self._rule)
            out.append(memo[sequence])
        return out


class SiteListApplier(_NormalizeSequences):
    """Normalize parallel name/site columns beside a bare sequence (alphabase layout)."""

    def __init__(self, modifications: SiteListModifications) -> None:
        super().__init__(modifications)
        self._rule = SiteListRule(
            delimiter=modifications.delimiter,
            site_base=modifications.site_base,
            case_sensitive=modifications.case_sensitive,
            unknown_policy=modifications.unknown_policy,
            entries=_map_entries(modifications),
        )

    @override
    def _results(self, df: pd.DataFrame) -> list[ModifiedSequence]:
        sequence_column, modification_column, site_column = self.sources
        memo: dict[tuple[str, str, str], ModifiedSequence] = {}
        out: list[ModifiedSequence] = []
        rows = zip(
            df[sequence_column].astype(str),
            df[modification_column].fillna("").astype(str),
            df[site_column].fillna("").astype(str),
            strict=True,
        )
        for key in rows:
            if key not in memo:
                memo[key] = apply_site_list(*key, self._rule)
            out.append(memo[key])
        return out


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
