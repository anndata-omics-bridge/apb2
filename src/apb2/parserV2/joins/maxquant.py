"""Join MaxQuant's experiment-level wide exports; evidence is a separate run-level table."""

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
    """Bind higher-level exports; evidence belongs to its own direct-input rule."""
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


def _per_sample_columns(columns: list[str], samples: tuple[str, ...]) -> set[str]:
    """Return columns that repeat one prefix across samples, such as ``Experiment <sample>``.

    A prefix seen with a single sample is kept: an annotation whose name merely ends in a
    sample name, such as ``Protein names`` beside a sample called ``names``, is not repeated.
    """
    names = set(samples)
    by_prefix: dict[str, list[str]] = {}
    for column in columns:
        for index, character in enumerate(column):
            if character == " " and column[index + 1 :] in names:
                by_prefix.setdefault(column[:index], []).append(column)
                break
    return {column for group in by_prefix.values() if len(group) > 1 for column in group}


def _wide_to_long(frame: pl.DataFrame, role: str) -> pl.DataFrame:
    matches = {column: _MEASUREMENT.fullmatch(column) for column in frame.columns}
    measurements = {column: match.groups() for column, match in matches.items() if match}
    if not measurements:
        raise ValueError(f"MaxQuant {role} has no per-sample quantitative columns")
    if frame["id"].null_count() or frame["id"].is_duplicated().any():
        raise ValueError(f"MaxQuant {role} has missing or duplicate row IDs")
    samples = tuple(dict.fromkeys(sample for _metric, sample in measurements.values()))
    metrics = tuple(dict.fromkeys(metric for metric, _sample in measurements.values()))
    # Overall totals are not per-sample measurements and must not shadow them. Other per-sample
    # columns, such as "Experiment <sample>", describe one sample rather than the feature; every
    # sample's rows would otherwise carry every sample's copy, multiplying the table by the
    # sample count. The remaining feature annotations are shared by all samples' rows.
    per_sample = _per_sample_columns(frame.columns, samples)
    annotations = frame.drop(
        [column for column in frame.columns if column in measurements or column in per_sample]
    ).drop(*metrics, strict=False)
    by_sample: list[pl.DataFrame] = []
    for sample in samples:
        quantitative = frame.select(
            pl.lit(sample).alias("sample"),
            *[
                (
                    pl.col(f"{metric} {sample}")
                    if f"{metric} {sample}" in measurements
                    else pl.lit(None, dtype=pl.String)
                ).alias(metric)
                for metric in metrics
            ],
        )
        by_sample.append(annotations.hstack(quantitative))
    return _prefix(pl.concat(by_sample, how="vertical_relaxed"), role)


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


def join(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Join higher-level exports by foreign-key references and experiment, never evidence."""
    if not tables:
        raise ValueError("MaxQuant preparation needs at least one quantitative table")
    if "evidence" in tables:
        raise ValueError("MaxQuant evidence is parsed directly, not joined")
    prepared = {role: _wide_to_long(frame, role) for role, frame in tables.items()}
    if len(prepared) == 1:
        role, frame = next(iter(prepared.items()))
        return frame.with_columns(pl.col(f"{role}.sample").alias("Experiment"))
    links = _relationships(tables)
    experiments = pl.concat(
        frame.select(pl.col(f"{role}.sample").alias("Experiment")).unique(maintain_order=True)
        for role, frame in prepared.items()
    ).unique(maintain_order=True)
    joined = links.join(experiments, how="cross")
    for role, frame in prepared.items():
        joined = joined.join(
            frame.with_columns(pl.col(f"{role}.id").cast(pl.String)),
            left_on=[f"_link_{role}", "Experiment"],
            right_on=[f"{role}.id", f"{role}.sample"],
            how="left",
            coalesce=False,
            validate="m:1",
            maintain_order="left",
        )
    return joined.drop(links.columns)


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
        raise ValueError("MaxQuant evidence is parsed directly, not joined")
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
