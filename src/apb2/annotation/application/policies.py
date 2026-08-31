"""Runtime behaviors for attaching annotation and selecting observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from apb2.annotation.data.model import (
    AnnotationError,
    AnnotationFileOrigin,
    AnnotationKind,
    AnnotationMatches,
    AnnotationOrigin,
    AnnotationResult,
    LevelAnnotationMatch,
    LevelAnnotationReport,
)
from apb2.parserV2.parse_quant.data.layer_columns import observation_labels
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)


class ObservationSelection(Protocol):
    """Compute one validated Boolean selection per observation."""

    def validate(self, match: LevelAnnotationMatch, /) -> None:
        """Reject evidence from which this selection cannot be computed."""
        ...

    def selected_rows(self, match: LevelAnnotationMatch, /) -> pl.Series:
        """Return one non-null Boolean per observation."""
        ...


class AnnotationApplication(Protocol):
    """Validate and apply one observation-retention behavior."""

    def validate(self, matches: AnnotationMatches, /) -> None:
        """Reject matching evidence that cannot produce this application."""
        ...

    def apply(self, parsed: ParsedLevels, matches: AnnotationMatches, /) -> AnnotationResult:
        """Attach annotation and return a new storage-neutral result."""
        ...


@dataclass(frozen=True, slots=True)
class MatchedAnnotationSelection:
    """Select observations matched to any annotation row."""

    def validate(self, match: LevelAnnotationMatch, /) -> None:
        del match

    def selected_rows(self, match: LevelAnnotationMatch, /) -> pl.Series:
        return match.matched_rows


@dataclass(frozen=True, slots=True)
class BooleanAnnotationSelection:
    """Select matched observations whose annotation field is true."""

    column: str

    def validate(self, match: LevelAnnotationMatch, /) -> None:
        if self.column not in match.aligned.columns:
            raise AnnotationError(
                f"annotation selection column {self.column!r} is absent; "
                f"available={match.aligned.columns}"
            )
        values = match.aligned.get_column(self.column)
        if values.dtype != pl.Boolean:
            raise AnnotationError(
                f"annotation selection column {self.column!r} must be Boolean, got {values.dtype}"
            )
        if values.filter(match.matched_rows).null_count():
            raise AnnotationError(
                f"annotation selection column {self.column!r} contains null matched values"
            )

    def selected_rows(self, match: LevelAnnotationMatch, /) -> pl.Series:
        return match.aligned.get_column(self.column).fill_null(False)


@dataclass(frozen=True, slots=True)
class AllAnnotationSelections:
    """Intersect several independently reusable observation selections."""

    selections: tuple[ObservationSelection, ...]

    def validate(self, match: LevelAnnotationMatch, /) -> None:
        for selection in self.selections:
            selection.validate(match)

    def selected_rows(self, match: LevelAnnotationMatch, /) -> pl.Series:
        selected = pl.Series("selected", [True] * len(match.matched_rows), dtype=pl.Boolean)
        for selection in self.selections:
            selected = selected & selection.selected_rows(match)
        return selected.rename("selected")


@dataclass(frozen=True, slots=True)
class KeepUnmatchedAnnotation:
    """Attach null metadata to unmatched observations and retain every observation."""

    def validate(self, matches: AnnotationMatches, /) -> None:
        _require_any_match(matches)

    def apply(self, parsed: ParsedLevels, matches: AnnotationMatches, /) -> AnnotationResult:
        return _apply(parsed, matches, selections=None)


@dataclass(frozen=True, slots=True)
class RequireCompleteAnnotation:
    """Require every observation to match before attaching annotation."""

    def validate(self, matches: AnnotationMatches, /) -> None:
        _require_any_match(matches)
        incomplete = {
            name: match.coverage
            for name, match in matches.levels.items()
            if match.coverage.quant_only_count
        }
        if incomplete:
            details = "; ".join(
                f"{name}: {coverage.quant_only_count} unmatched "
                f"{list(coverage.quant_only_examples)}, near_misses={dict(coverage.near_misses)}"
                for name, coverage in incomplete.items()
            )
            raise AnnotationError(f"complete sample annotation required; {details}")

    def apply(self, parsed: ParsedLevels, matches: AnnotationMatches, /) -> AnnotationResult:
        return _apply(parsed, matches, selections=None)


@dataclass(frozen=True, slots=True)
class SelectAnnotatedObservations:
    """Attach annotation and consistently retain only selected observations."""

    selection: ObservationSelection

    def validate(self, matches: AnnotationMatches, /) -> None:
        _require_any_match(matches)
        for match in matches.levels.values():
            self.selection.validate(match)

    def apply(self, parsed: ParsedLevels, matches: AnnotationMatches, /) -> AnnotationResult:
        selections: dict[ParsedLevelName, pl.Series] = {
            name: self.selection.selected_rows(match) for name, match in matches.levels.items()
        }
        return _apply(parsed, matches, selections=selections)


def record_annotation_provenance(
    result: AnnotationResult,
    kind: AnnotationKind,
    origin: AnnotationOrigin,
    /,
) -> AnnotationResult:
    """Record JSON-compatible annotation evidence on the shared and level values."""
    source = str(origin.path) if isinstance(origin, AnnotationFileOrigin) else None
    summary: dict[str, JsonValue] = {
        name: _report_json(report) for name, report in result.reports.items()
    }
    record: dict[str, JsonValue] = {
        "kind": kind.value,
        "source": source,
        "levels": summary,
    }
    result.parsed.uns["sample_annotation"] = record
    for name, level in result.parsed.levels.items():
        report = _report_json(result.reports[name])
        level_record: dict[str, JsonValue] = {
            "kind": kind.value,
            "source": source,
            **report,
        }
        level.uns["sample_annotation"] = level_record
    return result


def _apply(
    parsed: ParsedLevels,
    matches: AnnotationMatches,
    selections: Mapping[ParsedLevelName, pl.Series] | None,
) -> AnnotationResult:
    levels: dict[ParsedLevelName, ParsedLevel] = {}
    reports: dict[ParsedLevelName, LevelAnnotationReport] = {}
    for name, level in parsed.levels.items():
        match = matches.levels[name]
        selected = (
            pl.Series("selected", [True] * level.obs.frame.height, dtype=pl.Boolean)
            if selections is None
            else selections[name]
        )
        levels[name] = _annotated_level(level, match, selected)
        reports[name] = LevelAnnotationReport(
            coverage=match.coverage,
            corrections=match.corrections,
            columns_added=tuple(match.aligned.columns),
        )
    return AnnotationResult(
        parsed=ParsedLevels(levels=levels, uns=dict(parsed.uns)),
        reports=reports,
    )


def _require_any_match(matches: AnnotationMatches) -> None:
    empty = [
        name
        for name, match in matches.levels.items()
        if match.coverage.matched_observation_count == 0
    ]
    if empty:
        raise AnnotationError(
            f"sample annotation matched no observations for level(s) {empty}; "
            "no dataset-bound annotation was constructed"
        )


def _annotated_level(
    level: ParsedLevel,
    match: LevelAnnotationMatch,
    selected: pl.Series,
) -> ParsedLevel:
    annotated_obs = level.obs.frame.hstack(match.aligned.get_columns()).filter(selected)
    kept = [index for index, value in enumerate(selected) if value]
    return ParsedLevel(
        obs=ObsFinal(frame=annotated_obs, key_columns=level.obs.key_columns),
        var=level.var,
        primary_layer_name=level.primary_layer_name,
        uns=dict(level.uns),
        layers={name: _subset_layer(layer, kept) for name, layer in level.layers.items()},
        obsm={name: frame.filter(selected) for name, frame in level.obsm.items()},
        varm=dict(level.varm),
        obsp={name: _subset_pairwise(frame, kept) for name, frame in level.obsp.items()},
        varp=dict(level.varp),
    )


def _subset_layer(layer: FinalLayerTable, kept: list[int]) -> FinalLayerTable:
    key_count = len(layer.var_key_columns)
    value_columns = layer.values.columns[key_count:]
    selected_names = [value_columns[index] for index in kept]
    values = layer.values.select([*layer.var_key_columns, *selected_names])
    replacement = observation_labels(len(kept), layer.var_key_columns)
    values = values.rename(dict(zip(selected_names, replacement, strict=True)))
    return FinalLayerTable(
        layer_name=layer.layer_name,
        var_key_columns=layer.var_key_columns,
        values=values,
        role=layer.role,
    )


def _subset_pairwise(frame: pl.DataFrame, kept: list[int]) -> pl.DataFrame:
    mapping = {old: new for new, old in enumerate(kept)}
    if not mapping or frame.is_empty():
        return frame.head(0)
    return frame.filter(pl.col("row").is_in(kept) & pl.col("column").is_in(kept)).with_columns(
        pl.col("row").replace_strict(mapping),
        pl.col("column").replace_strict(mapping),
    )


def _report_json(report: LevelAnnotationReport) -> dict[str, JsonValue]:
    coverage = report.coverage
    near_misses: dict[str, JsonValue] = {
        key: [[candidate, score] for candidate, score in candidates]
        for key, candidates in coverage.near_misses.items()
    }
    corrections: list[JsonValue] = [
        {
            "observed": correction.observed,
            "expected": correction.expected,
            "score": correction.score,
        }
        for correction in report.corrections
    ]
    return {
        "observation_count": coverage.observation_count,
        "annotation_count": coverage.annotation_count,
        "matched_observation_count": coverage.matched_observation_count,
        "quant_only_count": coverage.quant_only_count,
        "annotation_only_count": coverage.annotation_only_count,
        "quant_only_examples": list(coverage.quant_only_examples),
        "annotation_only_examples": list(coverage.annotation_only_examples),
        "near_misses": near_misses,
        "corrections": corrections,
        "columns_added": list(report.columns_added),
    }
