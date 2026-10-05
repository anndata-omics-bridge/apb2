"""The `apb2 annotate` workflow: read a result, attach a sample table, write the result."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from apb2.api import (
    AnnotationCompiler,
    AnnotationError,
    AnnotationResult,
    ResultIOError,
    read_parsed_levels,
    write_parsed_levels,
)


class UnmatchedObservations(StrEnum):
    """Direct CLI choices for unmatched prolfquapp or SDRF observations."""

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
        compiler = AnnotationCompiler(
            (unmatched or UnmatchedObservations.KEEP).value,
            include,
        )
        parsed = read_parsed_levels(source)
        annotation = compiler.compile(annotation_source).parse(parsed)
        result = annotation.annotate()
        write_parsed_levels(result.parsed, target)
        return result
    except (AnnotationError, ResultIOError) as error:
        raise AnnotationWorkflowError(str(error)) from error
