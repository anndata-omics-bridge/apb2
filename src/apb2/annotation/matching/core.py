"""Polars-backed annotation-table validation and observation-key matching."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

import polars as pl

from apb2.annotation.data.model import (
    AnnotationCoverage,
    AnnotationError,
    AnnotationMatches,
    AnnotationOrigin,
    AnnotationTable,
    KeyCorrection,
    LevelAnnotationMatch,
)
from apb2.parserV2.parse_quant.data.parsed import (
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)

_EXAMPLE_LIMIT = 5

type Key = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KeyPairing:
    """Annotation-row assignments and fuzzy evidence in observation order."""

    annotation_rows: tuple[int | None, ...]
    corrections: tuple[KeyCorrection, ...]
    near_misses: Mapping[str, tuple[tuple[str, float], ...]]


class AnnotationMatching(Protocol):
    """Match observation identities to annotation row identities."""

    def pair(
        self,
        observations: Sequence[Key],
        identifiers: Sequence[Sequence[Key]],
        /,
    ) -> KeyPairing:
        """Return at most one annotation row for each observation identity."""
        ...


@dataclass(frozen=True, slots=True)
class ExactAnnotationMatching:
    """Reserve only identifiers whose text is exactly equal."""

    def pair(
        self,
        observations: Sequence[Key],
        identifiers: Sequence[Sequence[Key]],
        /,
    ) -> KeyPairing:
        lookup = {
            identifier: row
            for row, row_identifiers in enumerate(identifiers)
            for identifier in row_identifiers
        }
        rows = tuple(lookup.get(observation) for observation in observations)
        return KeyPairing(annotation_rows=rows, corrections=(), near_misses={})


@dataclass(frozen=True, slots=True)
class FuzzyAnnotationMatching:
    """Reserve exact pairs, then accept unambiguous mutual token-wise best matches."""

    cutoff: float
    margin: float
    near_miss_limit: int

    def pair(
        self,
        observations: Sequence[Key],
        identifiers: Sequence[Sequence[Key]],
        /,
    ) -> KeyPairing:
        exact = ExactAnnotationMatching().pair(observations, identifiers)
        assigned = list(exact.annotation_rows)
        used_rows = {row for row in assigned if row is not None}
        unmatched_observations = [index for index, row in enumerate(assigned) if row is None]
        unmatched_rows = [row for row in range(len(identifiers)) if row not in used_rows]
        scores = {
            (obs_index, row): max(
                _similarity(observations[obs_index], identifier) for identifier in identifiers[row]
            )
            for obs_index in unmatched_observations
            for row in unmatched_rows
        }
        corrections: list[KeyCorrection] = []
        for obs_index in unmatched_observations:
            ranked_rows = sorted(
                unmatched_rows,
                key=lambda row: (-scores[(obs_index, row)], row),
            )
            if not ranked_rows:
                continue
            row = ranked_rows[0]
            score = scores[(obs_index, row)]
            obs_runner_up = scores[(obs_index, ranked_rows[1])] if len(ranked_rows) > 1 else 0.0
            ranked_observations = sorted(
                unmatched_observations,
                key=lambda candidate: (-scores[(candidate, row)], candidate),
            )
            row_runner_up = (
                scores[(ranked_observations[1], row)] if len(ranked_observations) > 1 else 0.0
            )
            if (
                ranked_observations[0] == obs_index
                and score >= self.cutoff
                and score - obs_runner_up >= self.margin
                and score - row_runner_up >= self.margin
            ):
                assigned[obs_index] = row
                corrections.append(
                    KeyCorrection(
                        observed=_display_key(observations[obs_index]),
                        expected=_display_key(identifiers[row][0]),
                        score=score,
                    )
                )
        near_misses = {
            _display_key(observations[obs_index]): tuple(
                (_display_key(identifiers[row][0]), scores[(obs_index, row)])
                for row in sorted(
                    unmatched_rows,
                    key=lambda candidate: (-scores[(obs_index, candidate)], candidate),
                )[: self.near_miss_limit]
            )
            for obs_index in unmatched_observations
            if assigned[obs_index] is None
        }
        return KeyPairing(
            annotation_rows=tuple(assigned),
            corrections=tuple(corrections),
            near_misses=near_misses,
        )


def make_annotation_table(
    frame: pl.DataFrame,
    key_columns: tuple[str, ...],
    alias_columns: tuple[str, ...],
    origin: AnnotationOrigin,
    /,
) -> AnnotationTable:
    """Validate identifiers and sanitize only metadata-column names."""
    if not key_columns:
        raise AnnotationError("annotation must declare at least one key column")
    missing = [name for name in (*key_columns, *alias_columns) if name not in frame.columns]
    if missing:
        raise AnnotationError(
            f"annotation is missing identifier column(s) {missing}; present={frame.columns}"
        )
    primary = frame.select(list(key_columns))
    if primary.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise AnnotationError("annotation contains a null primary identifier")
    metadata = [name for name in frame.columns if name not in {*key_columns, *alias_columns}]
    sanitized = _sanitize_columns(metadata)
    if set(sanitized).intersection(key_columns):
        raise AnnotationError("sanitized annotation metadata collides with an identifier column")
    renamed = frame.rename(dict(zip(metadata, sanitized, strict=True)))
    table = AnnotationTable(
        frame=renamed,
        key_columns=key_columns,
        alias_columns=alias_columns,
        origin=origin,
    )
    _annotation_identifiers(table)
    return table


def match_annotation(
    table: AnnotationTable,
    parsed: ParsedLevels,
    matching: Mapping[ParsedLevelName, AnnotationMatching],
    /,
) -> AnnotationMatches:
    """Match one annotation table independently to every parsed level."""
    identifiers = _annotation_identifiers(table)
    levels: dict[ParsedLevelName, LevelAnnotationMatch] = {
        name: _match_level(table, level, identifiers, matching[name])
        for name, level in parsed.levels.items()
    }
    return AnnotationMatches(levels=levels)


def annotation_matching_for(level: ParsedLevel, /) -> AnnotationMatching:
    """Construct the matcher declared by persisted parse provenance, or exact matching."""
    declaration = level.uns.get("sample_annotation_matching")
    if declaration is None:
        return ExactAnnotationMatching()
    if not isinstance(declaration, dict):
        raise AnnotationError("persisted sample_annotation_matching must be an object")
    mode = declaration.get("mode")
    if mode != "fuzzy":
        raise AnnotationError(f"unsupported persisted annotation matching mode {mode!r}")
    cutoff_value = declaration.get("cutoff")
    margin_value = declaration.get("margin")
    limit_value = declaration.get("near_miss_limit")
    if (
        isinstance(cutoff_value, bool)
        or not isinstance(cutoff_value, int | float)
        or isinstance(margin_value, bool)
        or not isinstance(margin_value, int | float)
        or isinstance(limit_value, bool)
        or not isinstance(limit_value, int)
    ):
        raise AnnotationError(
            f"invalid persisted fuzzy annotation matching declaration: {declaration}"
        )
    cutoff = float(cutoff_value)
    margin = float(margin_value)
    near_miss_limit = limit_value
    if not 0.0 <= cutoff <= 1.0 or not 0.0 <= margin <= 1.0 or near_miss_limit < 1:
        raise AnnotationError(
            f"invalid persisted fuzzy annotation matching declaration: {declaration}"
        )
    return FuzzyAnnotationMatching(
        cutoff=cutoff,
        margin=margin,
        near_miss_limit=near_miss_limit,
    )


def _match_level(
    table: AnnotationTable,
    level: ParsedLevel,
    identifiers: tuple[tuple[Key, ...], ...],
    matching: AnnotationMatching,
) -> LevelAnnotationMatch:
    if len(level.obs.key_columns) != len(table.key_columns):
        raise AnnotationError(
            "annotation key arity does not match the observation axis: "
            f"annotation={table.key_columns}, obs={level.obs.key_columns}"
        )
    observations = tuple(
        tuple(str(value) for value in row)
        for row in level.obs.frame.select(list(level.obs.key_columns)).iter_rows()
    )
    pairing = matching.pair(observations, identifiers)
    metadata_columns = [
        name
        for name in table.frame.columns
        if name not in {*table.key_columns, *table.alias_columns}
    ]
    collisions = [name for name in metadata_columns if name in level.obs.frame.columns]
    if collisions:
        raise AnnotationError(f"annotation columns already present in obs: {collisions}")
    aligned = _aligned_metadata(table.frame, metadata_columns, pairing.annotation_rows)
    matched = pl.Series(
        "matched_annotation",
        [row is not None for row in pairing.annotation_rows],
        dtype=pl.Boolean,
    )
    used_rows = {row for row in pairing.annotation_rows if row is not None}
    quant_only = tuple(
        _display_key(key)
        for key, row in zip(observations, pairing.annotation_rows, strict=True)
        if row is None
    )
    annotation_only = tuple(
        _display_key(row_identifiers[0])
        for index, row_identifiers in enumerate(identifiers)
        if index not in used_rows
    )
    coverage = AnnotationCoverage(
        observation_count=len(observations),
        annotation_count=table.frame.height,
        matched_observation_count=int(matched.sum()),
        quant_only_count=len(quant_only),
        annotation_only_count=len(annotation_only),
        quant_only_examples=quant_only[:_EXAMPLE_LIMIT],
        annotation_only_examples=annotation_only[:_EXAMPLE_LIMIT],
        near_misses=pairing.near_misses,
    )
    return LevelAnnotationMatch(
        aligned=aligned,
        matched_rows=matched,
        coverage=coverage,
        corrections=pairing.corrections,
    )


def _annotation_identifiers(table: AnnotationTable) -> tuple[tuple[Key, ...], ...]:
    result: list[tuple[Key, ...]] = []
    owner: dict[Key, int] = {}
    for row_index, row in enumerate(table.frame.iter_rows(named=True)):
        identifiers = [tuple(str(row[name]) for name in table.key_columns)]
        if table.alias_columns and len(table.key_columns) != 1:
            raise AnnotationError("aliases are supported only for a single-column key")
        for name in table.alias_columns:
            value = row[name]
            values = value if isinstance(value, list) else [value]
            identifiers.extend((str(item),) for item in values if item is not None)
        unique = tuple(dict.fromkeys(identifiers))
        for identifier in unique:
            previous = owner.setdefault(identifier, row_index)
            if previous != row_index:
                raise AnnotationError(
                    f"duplicate annotation identifier {_display_key(identifier)!r} "
                    f"in rows {previous} and {row_index}"
                )
        result.append(unique)
    return tuple(result)


def _aligned_metadata(
    frame: pl.DataFrame,
    columns: list[str],
    rows: tuple[int | None, ...],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            name: pl.Series(
                name,
                [None if row is None else frame.item(row, name) for row in rows],
                dtype=frame.schema[name],
                strict=False,
            )
            for name in columns
        }
    )


def _similarity(left: Key, right: Key) -> float:
    return SequenceMatcher(None, _normalized(left), _normalized(right), autojunk=False).ratio()


def _normalized(key: Key) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", " ".join(key).lower()))


def _display_key(key: Key) -> str:
    return " | ".join(key)


def _sanitize_columns(names: list[str]) -> list[str]:
    sanitized = [_sanitize_name(name) for name in names]
    groups: dict[str, list[str]] = {}
    for original, clean in zip(names, sanitized, strict=True):
        groups.setdefault(clean, []).append(original)
    collisions = {clean: values for clean, values in groups.items() if len(set(values)) > 1}
    if collisions:
        raise AnnotationError(f"column-name collision after sanitization: {collisions}")
    return sanitized


def _sanitize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    clean = re.sub(r"[^A-Za-z0-9_.]", "_", ascii_name)
    return re.sub(r"_+", "_", clean).strip("_.") or "col"
