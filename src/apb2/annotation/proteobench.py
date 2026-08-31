"""ProteoBench source interpretation and strict dataset-bound annotation behavior."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from apb2.annotation.application.policies import (
    RequireCompleteAnnotation,
    record_annotation_provenance,
)
from apb2.annotation.data.model import (
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


@dataclass(frozen=True, slots=True)
class ProteobenchAnnotationParameters:
    """The identifier field ProteoBench sample records use."""

    key_field: str = "raw_file"


@dataclass(frozen=True, slots=True)
class ProteobenchAnnotationParser:
    """A ProteoBench parser bound to one already loaded module annotation."""

    source: LoadedAnnotationSource
    parameters: ProteobenchAnnotationParameters

    def parse(self, parsed: ParsedLevels, /) -> ProteobenchAnnotation:
        """Validate and match the source, rejecting incomplete dataset coverage."""
        key_field = self.source.key_field_hint or self.parameters.key_field
        aliases = tuple(
            name
            for name in (f"{key_field}_alias", f"{key_field}_aliases")
            if name in self.source.frame.columns
        )
        table = make_annotation_table(
            self.source.frame,
            (key_field,),
            aliases,
            self.source.origin,
        )
        matches = match_annotation(
            table,
            parsed,
            {name: annotation_matching_for(level) for name, level in parsed.levels.items()},
        )
        RequireCompleteAnnotation().validate(matches)
        return ProteobenchAnnotation(table=table, parsed=parsed, matches=matches)


@dataclass(frozen=True, slots=True)
class ProteobenchAnnotation:
    """A complete-coverage ProteoBench annotation bound to one parsed dataset."""

    table: AnnotationTable
    parsed: ParsedLevels
    matches: AnnotationMatches

    def annotate(self) -> AnnotationResult:
        """Attach the already validated complete annotation."""
        for name, match in self.matches.levels.items():
            if match.coverage.annotation_only_count:
                logger.info(
                    "level={} unused ProteoBench annotation rows: count={} examples={}",
                    name,
                    match.coverage.annotation_only_count,
                    list(match.coverage.annotation_only_examples),
                )
        result = RequireCompleteAnnotation().apply(self.parsed, self.matches)
        return record_annotation_provenance(
            result,
            AnnotationKind.PROTEOBENCH,
            self.table.origin,
        )


def proteobench_signature(
    source: LoadedAnnotationSource,
    parameters: ProteobenchAnnotationParameters,
    /,
) -> bool:
    """Return whether source shape and fields identify a ProteoBench annotation."""
    if source.convention_hint is not None:
        return source.convention_hint is AnnotationKind.PROTEOBENCH
    key_field = source.key_field_hint or parameters.key_field
    return {key_field, "sample_name", "condition"}.issubset(source.frame.columns)
