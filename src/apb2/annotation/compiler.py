"""Compile one delimited annotation source into a source-bound parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from apb2.annotation.contracts import AnnotationParser
from apb2.annotation.data.model import AnnotationError
from apb2.annotation.prolfquapp import (
    ProlfquappAnnotationParameters,
    ProlfquappAnnotationParser,
    prolfquapp_signature,
)
from apb2.annotation.source.load import load_annotation_file, load_annotation_frame


@dataclass(frozen=True, slots=True)
class AnnotationCompiler:
    """User-configured parser for generic delimited observation annotations."""

    prolfquapp: ProlfquappAnnotationParameters = field(
        default_factory=ProlfquappAnnotationParameters
    )

    def compile(self, source: Path | pl.DataFrame, /) -> AnnotationParser:
        """Load once, verify the tabular convention, and return its bound parser."""
        loaded = (
            load_annotation_file(source)
            if isinstance(source, Path)
            else load_annotation_frame(source)
        )
        if not prolfquapp_signature(loaded, self.prolfquapp):
            raise AnnotationError("annotation table has no supported observation key")
        return ProlfquappAnnotationParser(source=loaded, parameters=self.prolfquapp)
