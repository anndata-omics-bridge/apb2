"""The ``[modifications]`` block at runtime: normalize modified sequences on a frame.

``applier_for(rule)`` decides once whether normalization runs — is a block declared,
does any compute read its output — instead of on every table. The applier memoizes on
unique source values: normalization is a pure function of the source columns, so a
column with a million rows and fifty thousand distinct sequences tokenizes fifty
thousand times, not a million.
"""

from __future__ import annotations

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
    WideRule,
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


class ApplyModifications:
    """Normalize a vendor modified-sequence column, memoized on unique source values.

    Adds three columns in place and returns the same frame: the rule's
    ``output_column`` (ProForma string), ``stripped_sequence`` (amino-acid-only), and
    ``unknown_mod_tokens`` (unresolved vendor tokens per row). Construction resolves the
    Unimod map, so an unknown accession fails before any table is read.
    """

    def __init__(self, modifications: Modifications) -> None:
        self.output_column = modifications.output_column
        self.sources = modification_sources(modifications)
        entries = _map_entries(modifications)
        if isinstance(modifications, SiteListModifications):
            self._site_rule: SiteListRule | None = SiteListRule(
                delimiter=modifications.delimiter,
                site_base=modifications.site_base,
                case_sensitive=modifications.case_sensitive,
                unknown_policy=modifications.unknown_policy,
                entries=entries,
            )
            self._token_rule: ModificationRule | None = None
        else:
            self._site_rule = None
            self._token_rule = ModificationRule(
                token_pattern=modifications.token_pattern,
                token_position=modifications.token_position,
                case_sensitive=modifications.case_sensitive,
                unknown_policy=modifications.unknown_policy,
                entries=entries,
            )

    def source_columns(self) -> frozenset[str]:
        return frozenset({*self.sources, self.output_column, "stripped_sequence"})

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
        memo: dict[object, ModifiedSequence] = {}
        if self._site_rule is not None:
            sequence_column, modification_column, site_column = self.sources
            rows = zip(
                df[sequence_column].astype(str),
                df[modification_column].fillna("").astype(str),
                df[site_column].fillna("").astype(str),
                strict=True,
            )
            out: list[ModifiedSequence] = []
            for sequence, modifications, sites in rows:
                key = (sequence, modifications, sites)
                if key not in memo:
                    memo[key] = apply_site_list(sequence, modifications, sites, self._site_rule)
                out.append(memo[key])
            return out
        assert self._token_rule is not None
        token_rule = self._token_rule
        out = []
        for sequence in df[self.sources[0]].astype(str):
            if sequence not in memo:
                memo[sequence] = apply_rule(sequence, token_rule)
            out.append(memo[sequence])
        return out


class NoModifications:
    """No modification normalization runs: none is declared, or none is read."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def source_columns(self) -> frozenset[str]:
        return frozenset()


type ModificationApplier = ApplyModifications | NoModifications


def applier_for(rule: LongRule | WideRule) -> ModificationApplier:
    """Decide once whether modification normalization runs for this rule.

    Both conditions are absence questions — is a block declared, does anything read its
    output — asked once when the applier is built instead of on every table.
    """
    if rule.modifications is None:
        return NoModifications()
    consumed = any(
        isinstance(column, ProformaSequence | StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return NoModifications()
    return ApplyModifications(rule.modifications)
