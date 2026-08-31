"""Physical decoding of sample-annotation sources into Polars values."""

from __future__ import annotations

import tomllib
from pathlib import Path

import polars as pl

from apb2.annotation.data.model import (
    IN_MEMORY_ANNOTATION,
    AnnotationError,
    AnnotationFileOrigin,
    AnnotationKind,
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
            loaded = LoadedAnnotationSource(
                frame=frame,
                origin=AnnotationFileOrigin(source),
                convention_hint=AnnotationKind.PROLFQUAPP,
            )
        elif suffix == ".toml":
            loaded = _load_toml(source)
        else:
            raise AnnotationError(
                f"unsupported annotation suffix {suffix or '<none>'!r}; "
                "expected .csv, .tsv, or .toml"
            )
    except (OSError, pl.exceptions.PolarsError, tomllib.TOMLDecodeError) as error:
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
        convention_hint=None,
    )


def _load_toml(source: Path) -> LoadedAnnotationSource:
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    samples = document.get("samples")
    key_field = "raw_file"
    if samples is None:
        obs = document.get("obs")
        if isinstance(obs, dict):
            samples = obs.get("samples")
            declared = obs.get("key_field", key_field)
            if isinstance(declared, str):
                key_field = declared
    if not isinstance(samples, list) or not samples:
        raise AnnotationError(
            f"annotation TOML has no [[samples]] or [[obs.samples]] records: {source}"
        )
    if not all(isinstance(record, dict) for record in samples):
        raise AnnotationError(f"annotation TOML sample records must be tables: {source}")
    return LoadedAnnotationSource(
        frame=pl.from_dicts(samples, strict=False),
        origin=AnnotationFileOrigin(source),
        convention_hint=AnnotationKind.PROTEOBENCH,
        key_field_hint=key_field,
    )
