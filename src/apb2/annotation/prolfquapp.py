"""prolfquapp source interpretation and dataset-bound annotation behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from apb2.annotation.application.policies import (
    AnnotationApplication,
    KeepUnmatchedAnnotation,
    record_annotation_provenance,
)
from apb2.annotation.data.model import (
    AnnotationError,
    AnnotationKind,
    AnnotationMatches,
    AnnotationResult,
    AnnotationTable,
    LoadedAnnotationSource,
)
from apb2.annotation.matching.core import (
    annotation_matching_for,
    make_annotation_table,
    match_annotation,
)
from apb2.parserV2.parse_quant.data.parsed import ParsedLevels


def _default_key_pattern() -> re.Pattern[str]:
    return re.compile(r"^channel|^Relative|^raw|^file|^run", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ProlfquappAnnotationParameters:
    """User-selected prolfquapp interpretation and application behavior."""

    application: AnnotationApplication = field(default_factory=KeepUnmatchedAnnotation)
    key_pattern: re.Pattern[str] = field(default_factory=_default_key_pattern)


@dataclass(frozen=True, slots=True)
class ProlfquappAnnotationParser:
    """A prolfquapp parser bound to one already loaded source."""

    source: LoadedAnnotationSource
    parameters: ProlfquappAnnotationParameters

    def parse(self, parsed: ParsedLevels, /) -> ProlfquappAnnotation:
        """Validate, match, and construct only an applicable dataset annotation."""
        table = _table(self.source, self.parameters)
        matches = match_annotation(
            table,
            parsed,
            {name: annotation_matching_for(level) for name, level in parsed.levels.items()},
        )
        self.parameters.application.validate(matches)
        return ProlfquappAnnotation(
            table=table,
            parsed=parsed,
            matches=matches,
            application=self.parameters.application,
        )


@dataclass(frozen=True, slots=True)
class ProlfquappAnnotation:
    """A validated prolfquapp annotation bound to one parsed dataset."""

    table: AnnotationTable
    parsed: ParsedLevels
    matches: AnnotationMatches
    application: AnnotationApplication

    def annotate(self) -> AnnotationResult:
        """Report asymmetric diagnostics and apply the prevalidated behavior."""
        for name, match in self.matches.levels.items():
            coverage = match.coverage
            if coverage.annotation_only_count:
                logger.warning(
                    "level={} annotation rows absent from quantification: count={} examples={}",
                    name,
                    coverage.annotation_only_count,
                    list(coverage.annotation_only_examples),
                )
            if coverage.quant_only_count:
                logger.info(
                    "level={} quantification rows without annotation: count={} examples={}",
                    name,
                    coverage.quant_only_count,
                    list(coverage.quant_only_examples),
                )
            if match.corrections:
                logger.info(
                    "level={} accepted fuzzy annotation corrections={}", name, match.corrections
                )
        result = self.application.apply(self.parsed, self.matches)
        return record_annotation_provenance(
            result,
            AnnotationKind.PROLFQUAPP,
            self.table.origin,
        )


def prolfquapp_signature(
    source: LoadedAnnotationSource,
    parameters: ProlfquappAnnotationParameters,
    /,
) -> bool:
    """Return whether the source envelope and columns identify prolfquapp input."""
    if source.convention_hint is not None:
        return source.convention_hint is AnnotationKind.PROLFQUAPP
    return bool(_primary_candidates(source, parameters))


def _table(
    source: LoadedAnnotationSource,
    parameters: ProlfquappAnnotationParameters,
) -> AnnotationTable:
    candidates = _primary_candidates(source, parameters)
    if len(candidates) != 1:
        raise AnnotationError(
            "prolfquapp annotation requires exactly one identifier column matching "
            f"{parameters.key_pattern.pattern!r}; candidates={candidates}"
        )
    primary = candidates[0]
    aliases = tuple(
        name for name in (f"{primary}_alias", f"{primary}_aliases") if name in source.frame.columns
    )
    return make_annotation_table(
        source.frame,
        (primary,),
        aliases,
        source.origin,
    )


def _primary_candidates(
    source: LoadedAnnotationSource,
    parameters: ProlfquappAnnotationParameters,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in source.frame.columns
        if parameters.key_pattern.search(name)
        and not name.endswith("_alias")
        and not name.endswith("_aliases")
    )
