"""Join MaxQuant's experiment-level wide exports; give run-level evidence its protein group."""

from __future__ import annotations

import re
from collections.abc import Mapping

import polars as pl

_FILES = {
    "modificationSpecificPeptides.txt": "peptidoform",
    "peptides.txt": "peptide",
    "proteinGroups.txt": "protein",
}
_MEASUREMENT = re.compile(r"^(Intensity|LFQ intensity|iBAQ) (.+)$")


def _roles(columns: tuple[str, ...]) -> list[str]:
    present = set(columns)
    if "id" not in present:
        return []
    candidates: list[str] = []
    if {"Raw file", "Modified sequence", "Charge", "Intensity"} <= present:
        candidates.append("evidence")
    if any(_MEASUREMENT.match(column) for column in columns):
        if {"Protein IDs", "Peptide IDs"} <= present:
            candidates.append("protein")
        elif "Mod. peptide IDs" in present:
            candidates.append("peptide")
        elif {"Sequence", "Modifications", "Evidence IDs"} <= present:
            candidates.append("peptidoform")
    return candidates


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Bind higher-level exports; evidence belongs to its own preparation."""
    selected: dict[str, str] = {}
    for name, columns in headers.items():
        candidates = _roles(columns)
        declared = _FILES.get(name)
        if declared is not None and declared not in candidates:
            raise ValueError(f"MaxQuant {name} lacks the columns for {declared}")
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f"MaxQuant input {name} has ambiguous roles: {candidates}")
        role = candidates[0]
        if role == "evidence":
            continue
        if role in selected:
            raise ValueError(f"MaxQuant has multiple {role} inputs: {selected[role]}, {name}")
        selected[role] = name
    return selected


def _prefix(frame: pl.DataFrame, role: str) -> pl.DataFrame:
    return frame.rename({column: f"{role}.{column}" for column in frame.columns})


def _links(frame: pl.DataFrame, role: str) -> pl.DataFrame:
    """Explode only ID references, not wide metadata or quantitative sample rows."""
    key = "Evidence IDs"
    if key not in frame.columns:
        raise ValueError(f"MaxQuant {role} requires Evidence IDs to join related tables")
    return (
        frame.select(
            pl.col("id").cast(pl.String).alias(f"_link_{role}"),
            pl.col(key).cast(pl.String).str.split(";").alias("evidence.id"),
        )
        .explode("evidence.id", empty_as_null=True)
        .with_columns(pl.col("evidence.id").str.strip_chars().replace("", None))
    )


def _relationships(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Return one row per related feature-ID tuple, keeping features without references."""
    links = pl.DataFrame(schema={"evidence.id": pl.String})
    for role, frame in tables.items():
        links = links.join(
            _links(frame, role),
            on="evidence.id",
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )
    # Many evidence IDs describe the same feature relationship. Collapse the ID-only
    # relation before adding experiments and large annotation strings; otherwise a long
    # list of IDs is copied once per ID, per experiment, per linked feature.
    return links.drop("evidence.id").unique(maintain_order=True)


KEY_COLUMNS = ("id", "Evidence IDs")
"""Columns ``join_wide`` reads besides the rules' own: row IDs and evidence references."""


def join_wide(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Join higher-level exports by their foreign keys, keeping every table wide.

    Each row is one relationship between features of different tables, with every table's
    per-experiment columns side by side, so rows grow with relationships, not with features
    times experiments. A feature repeated across relationships repeats identical cells, which
    the rules' keep-first policy resolves. One table is only namespaced.
    """
    if not tables:
        raise ValueError("MaxQuant preparation needs at least one quantitative table")
    if "evidence" in tables:
        raise ValueError("MaxQuant evidence has its own preparation, not this join")
    for role, frame in tables.items():
        if frame["id"].null_count() or frame["id"].is_duplicated().any():
            raise ValueError(f"MaxQuant {role} has missing or duplicate row IDs")
    if len(tables) == 1:
        role, frame = next(iter(tables.items()))
        return _prefix(frame, role)
    links = _relationships(tables)
    joined = links
    for role, frame in tables.items():
        joined = joined.join(
            _prefix(frame, role).with_columns(pl.col(f"{role}.id").cast(pl.String)),
            left_on=f"_link_{role}",
            right_on=f"{role}.id",
            how="left",
            coalesce=False,
            validate="m:1",
            maintain_order="left",
        )
    return joined.drop(links.columns)


_RAZOR_COLUMNS = ("Leading razor protein", "Leading razor proteins")

EVIDENCE_KEY_COLUMNS = ("id", "Protein group IDs", *_RAZOR_COLUMNS, "Protein IDs")
"""Columns ``join_evidence`` reads besides the rules' own: row and group IDs, razor protein."""


def identify_evidence(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Bind evidence and, when present, proteinGroups, whose groups evidence rows reference."""
    selected: dict[str, str] = {}
    for name, columns in headers.items():
        candidates = _roles(columns)
        if name == "evidence.txt" and candidates != ["evidence"]:
            raise ValueError(f"MaxQuant {name} lacks the columns for evidence")
        if candidates not in (["evidence"], ["protein"]):
            continue
        role = candidates[0]
        if role in selected:
            raise ValueError(f"MaxQuant has multiple {role} inputs: {selected[role]}, {name}")
        selected[role] = name
    return selected if "evidence" in selected else {}


def join_evidence(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Keep every evidence row and add its protein group's ``Protein IDs`` when bound.

    ``Protein group IDs`` references every group of a shared peptide; MaxQuant assigns the
    peptide to the group holding its leading razor protein, and so does this lookup. Evidence
    alone is only namespaced: there is no protein table to agree with.
    """
    evidence = tables["evidence"]
    if evidence["id"].null_count() or evidence["id"].is_duplicated().any():
        raise ValueError("MaxQuant evidence has missing or duplicate row IDs")
    prepared = _prefix(evidence, "evidence")
    protein = tables.get("protein")
    if protein is None:
        return prepared
    if protein["id"].null_count() or protein["id"].is_duplicated().any():
        raise ValueError("MaxQuant protein has missing or duplicate row IDs")
    razor = next((column for column in _RAZOR_COLUMNS if column in evidence.columns), None)
    if razor is None:
        raise ValueError("MaxQuant evidence names no leading razor protein to pick its group")
    groups = (
        evidence.select(
            pl.col("id").alias("evidence.id"),
            pl.col(razor).alias("_razor"),
            pl.col("Protein group IDs").str.split(";").alias("_group"),
        )
        .explode("_group", empty_as_null=True)
        .join(
            protein.select(
                pl.col("id").alias("_group"), pl.col("Protein IDs").alias("protein.Protein IDs")
            ),
            on="_group",
            how="inner",
        )
        .filter(pl.col("protein.Protein IDs").str.split(";").list.contains(pl.col("_razor")))
        .select("evidence.id", "protein.Protein IDs")
    )
    matches = groups.group_by("evidence.id").len()
    unmatched = evidence.height - matches.height
    ambiguous = matches.filter(pl.col("len") > 1).height
    if unmatched or ambiguous:
        raise ValueError(
            f"MaxQuant evidence: {unmatched} rows reference no protein group holding their "
            f"leading razor protein, {ambiguous} reference several"
        )
    return prepared.join(
        groups, on="evidence.id", how="left", validate="1:1", maintain_order="left"
    )
