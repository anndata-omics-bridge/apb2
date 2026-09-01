"""Result-I/O facade for the storage-neutral sample-annotation workflow."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from apb2.annotation.application.policies import (
    AllAnnotationSelections,
    BooleanAnnotationSelection,
    KeepUnmatchedAnnotation,
    MatchedAnnotationSelection,
    ObservationSelection,
    RequireCompleteAnnotation,
    SelectAnnotatedObservations,
)
from apb2.annotation.compiler import AnnotationCompiler
from apb2.annotation.data.model import AnnotationError, AnnotationResult
from apb2.annotation.prolfquapp import ProlfquappAnnotationParameters
from apb2.parserV2.parse_quant.io.errors import ResultIOError
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels


class UnmatchedObservations(StrEnum):
    """Direct CLI choices for unmatched prolfquapp observations."""

    KEEP = "keep"
    ERROR = "error"
    DROP = "drop"


class AnnotationWorkflowError(ValueError):
    """An expected annotation or parsed-result boundary failure."""


def annotate_result(
    source: Path,
    annotation_source: Path,
    target: Path,
    /,
    *,
    unmatched: UnmatchedObservations | None = None,
    include: str | None = None,
) -> AnnotationResult:
    """Read, compile, annotate, and persist one APB2 result."""
    try:
        if source.resolve() == target.resolve():
            raise AnnotationError("annotation output must differ from its input")
        application = _application(unmatched, include)
        compiler = AnnotationCompiler(
            prolfquapp=ProlfquappAnnotationParameters(application=application),
        )
        parsed = read_parsed_levels(source)
        annotation = compiler.compile(annotation_source).parse(parsed)
        result = annotation.annotate()
        write_parsed_levels(result.parsed, target)
        return result
    except (AnnotationError, ResultIOError) as error:
        raise AnnotationWorkflowError(str(error)) from error


def _application(
    unmatched: UnmatchedObservations | None,
    include: str | None,
) -> KeepUnmatchedAnnotation | RequireCompleteAnnotation | SelectAnnotatedObservations:
    choice = unmatched or UnmatchedObservations.KEEP
    if choice is UnmatchedObservations.KEEP:
        if include is not None:
            raise AnnotationError("--include requires --unmatched drop")
        return KeepUnmatchedAnnotation()
    if choice is UnmatchedObservations.ERROR:
        if include is not None:
            raise AnnotationError("--include requires --unmatched drop")
        return RequireCompleteAnnotation()
    selections: list[ObservationSelection] = [MatchedAnnotationSelection()]
    if include is not None:
        selections.append(BooleanAnnotationSelection(include))
    return SelectAnnotatedObservations(AllAnnotationSelections(tuple(selections)))
