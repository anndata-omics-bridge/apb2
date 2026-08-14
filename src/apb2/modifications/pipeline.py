"""Apply a [modifications] rule to a DataFrame, adding normalized columns."""

from __future__ import annotations

import pandas as pd

from apb2.modifications.apply_rules import (
    MapEntry,
    ModificationRule,
    SiteListRule,
    apply_rule,
    apply_site_list,
)
from apb2.modifications.unimod_registry import resolve
from apb2.vendor_parse_rules.model import (
    Modifications,
    SiteListModifications,
    TokenRegexModifications,
)


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


def _to_runtime_rule(mods: TokenRegexModifications) -> ModificationRule:
    """Convert the validated token_regex model into the runtime dataclass."""
    return ModificationRule(
        token_pattern=mods.token_pattern,
        token_position=mods.token_position,
        case_sensitive=mods.case_sensitive,
        unknown_policy=mods.unknown_policy,
        entries=_map_entries(mods),
    )


def _to_site_list_rule(mods: SiteListModifications) -> SiteListRule:
    """Convert the validated site_list model into the runtime dataclass."""
    return SiteListRule(
        delimiter=mods.delimiter,
        site_base=mods.site_base,
        case_sensitive=mods.case_sensitive,
        unknown_policy=mods.unknown_policy,
        entries=_map_entries(mods),
    )


def _require_columns(df: pd.DataFrame, mods: Modifications) -> None:
    missing = [column for column in mods.source_columns if column not in df.columns]
    if missing:
        raise KeyError(
            f"[modifications] parser={mods.parser!r} needs column(s) {missing} "
            f"not found in DataFrame; available: {list(df.columns)[:10]}…"
        )


def apply_modifications(df: pd.DataFrame, mods: Modifications) -> pd.DataFrame:
    """Add normalized modification columns to ``df`` based on ``mods``.

    Adds (and returns the same frame for convenience):
    - ``mods.output_column`` (default ``"proforma_sequence"``): ProForma string
    - ``"stripped_sequence"``: amino-acid-only sequence
    - ``"unknown_mod_tokens"``: list of unresolved vendor tokens per row

    The source columns are left untouched. Dispatches on ``mods.parser``; see
    ``modifications.schema`` for the two parsers and ``apply_rules`` for their runtimes.
    """
    _require_columns(df, mods)

    if mods.parser == "site_list":
        rule = _to_site_list_rule(mods)
        results = [
            apply_site_list(sequence, modifications, sites, rule)
            for sequence, modifications, sites in zip(
                df[mods.sequence_column].astype(str),
                df[mods.modification_column].fillna("").astype(str),
                df[mods.site_column].fillna("").astype(str),
                strict=True,
            )
        ]
    else:
        runtime = _to_runtime_rule(mods)
        results = [apply_rule(s, runtime) for s in df[mods.source_column].astype(str)]

    df[mods.output_column] = [r.proforma_sequence for r in results]
    df["stripped_sequence"] = [r.stripped_sequence for r in results]
    df["unknown_mod_tokens"] = [r.unknown_tokens for r in results]
    return df
