"""Innermost values exchanged by the sample-annotation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import ParsedLevelName, ParsedLevels


class AnnotationError(ValueError):
    """A sample annotation cannot be recognized, parsed, matched, or applied."""


class AnnotationKind(StrEnum):
    """Supported scientific sample-annotation conventions."""

    PROLFQUAPP = "prolfquapp"
    PROTEOBENCH = "proteobench"


@dataclass(frozen=True, slots=True)
class AnnotationFileOrigin:
    """The resolved file from which an annotation was loaded."""

    path: Path


@dataclass(frozen=True, slots=True)
class InMemoryAnnotationOrigin:
    """Marker for a programmatically supplied Polars annotation."""


type AnnotationOrigin = AnnotationFileOrigin | InMemoryAnnotationOrigin

IN_MEMORY_ANNOTATION = InMemoryAnnotationOrigin()


@dataclass(frozen=True, slots=True)
class LoadedAnnotationSource:
    """One physically decoded source plus any convention evidence from its envelope."""

    frame: pl.DataFrame
    origin: AnnotationOrigin
    convention_hint: AnnotationKind | None
    key_field_hint: str | None = None


@dataclass(frozen=True, slots=True)
class AnnotationTable:
    """Validated sample rows and every identifier column accepted for matching."""

    frame: pl.DataFrame
    key_columns: tuple[str, ...]
    alias_columns: tuple[str, ...]
    origin: AnnotationOrigin


@dataclass(frozen=True, slots=True)
class AnnotationCoverage:
    """Counts and bounded mismatch evidence for one quantification level."""

    observation_count: int
    annotation_count: int
    matched_observation_count: int
    quant_only_count: int
    annotation_only_count: int
    quant_only_examples: tuple[str, ...]
    annotation_only_examples: tuple[str, ...]
    near_misses: Mapping[str, tuple[tuple[str, float], ...]]


@dataclass(frozen=True, slots=True)
class KeyCorrection:
    """One accepted fuzzy observation-to-annotation identifier correction."""

    observed: str
    expected: str
    score: float


@dataclass(frozen=True, slots=True)
class LevelAnnotationMatch:
    """Annotation aligned to one observation axis plus its matching evidence."""

    aligned: pl.DataFrame
    matched_rows: pl.Series
    coverage: AnnotationCoverage
    corrections: tuple[KeyCorrection, ...]


@dataclass(frozen=True, slots=True)
class AnnotationMatches:
    """Independent matching evidence for every parsed quantification level."""

    levels: Mapping[ParsedLevelName, LevelAnnotationMatch]


@dataclass(frozen=True, slots=True)
class LevelAnnotationReport:
    """Persistable evidence produced by applying one level's annotation."""

    coverage: AnnotationCoverage
    corrections: tuple[KeyCorrection, ...]
    columns_added: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnnotationResult:
    """A newly annotated storage-neutral result and its per-level reports."""

    parsed: ParsedLevels
    reports: Mapping[ParsedLevelName, LevelAnnotationReport]
