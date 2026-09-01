"""Physical decoding of delimited sample-annotation sources into Polars values."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from apb2.annotation.data.model import (
    IN_MEMORY_ANNOTATION,
    AnnotationError,
    AnnotationFileOrigin,
    LoadedAnnotationSource,
)

_DELIMITERS = {".csv": ",", ".tsv": "\t"}


def load_annotation_file(path: Path, /) -> LoadedAnnotationSource:
    """Decode one supported annotation file exactly once."""
    source = path.expanduser().resolve()
    if not source.exists():
        raise AnnotationError(f"annotation source does not exist: {source}")
    if not source.is_file():
        raise AnnotationError(f"annotation source is not a file: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix in _DELIMITERS:
            frame = pl.read_csv(source, separator=_DELIMITERS[suffix])
            loaded = LoadedAnnotationSource(frame=frame, origin=AnnotationFileOrigin(source))
        else:
            raise AnnotationError(
                f"unsupported annotation suffix {suffix or '<none>'!r}; expected .csv or .tsv"
            )
    except (OSError, pl.exceptions.PolarsError) as error:
        raise AnnotationError(f"cannot decode annotation source {source}: {error}") from error
    if loaded.frame.is_empty():
        raise AnnotationError("annotation table must contain at least one sample row")
    return loaded


def load_annotation_frame(frame: pl.DataFrame, /) -> LoadedAnnotationSource:
    """Bind an in-memory Polars annotation without copying or guessing its convention."""
    if frame.is_empty():
        raise AnnotationError("annotation table must contain at least one sample row")
    return LoadedAnnotationSource(
        frame=frame,
        origin=IN_MEMORY_ANNOTATION,
    )
