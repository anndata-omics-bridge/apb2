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
from apb2.annotation.compiler import AnnotationCompiler, DetectAnnotation, RequireAnnotation
from apb2.annotation.data.model import AnnotationError, AnnotationKind, AnnotationResult
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
    annotation_type: AnnotationKind | None = None,
    unmatched: UnmatchedObservations | None = None,
    include: str | None = None,
) -> AnnotationResult:
    """Read, compile, annotate, and persist one APB2 result."""
    try:
        if source.resolve() == target.resolve():
            raise AnnotationError("annotation output must differ from its input")
        if annotation_type is AnnotationKind.PROTEOBENCH and (
            unmatched is not None or include is not None
        ):
            raise AnnotationError(
                "ProteoBench annotation is strict and accepts no retention options"
            )
        if annotation_type is None and (unmatched is not None or include is not None):
            raise AnnotationError("retention options require --type prolfquapp")
        application = _application(unmatched, include)
        recognition = (
            DetectAnnotation() if annotation_type is None else RequireAnnotation(annotation_type)
        )
        compiler = AnnotationCompiler(
            recognition=recognition,
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
