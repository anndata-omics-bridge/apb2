"""Compile one annotation source into its verified source-bound parser."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl

from apb2.annotation.contracts import AnnotationParser
from apb2.annotation.data.model import AnnotationError, AnnotationKind, LoadedAnnotationSource
from apb2.annotation.prolfquapp import (
    ProlfquappAnnotationParameters,
    ProlfquappAnnotationParser,
    prolfquapp_signature,
)
from apb2.annotation.proteobench import (
    ProteobenchAnnotationParameters,
    ProteobenchAnnotationParser,
    proteobench_signature,
)
from apb2.annotation.source.load import load_annotation_file, load_annotation_frame


class AnnotationRecognition(Protocol):
    """Choose a parser from the convention signatures satisfied by one source."""

    def choose(
        self,
        candidates: Mapping[AnnotationKind, AnnotationParser],
        /,
    ) -> AnnotationParser:
        """Return one candidate or raise a recognition error."""
        ...


@dataclass(frozen=True, slots=True)
class DetectAnnotation:
    """Require automatic convention recognition to produce exactly one candidate."""

    def choose(
        self,
        candidates: Mapping[AnnotationKind, AnnotationParser],
        /,
    ) -> AnnotationParser:
        if len(candidates) == 1:
            return next(iter(candidates.values()))
        if not candidates:
            raise AnnotationError("annotation convention is unsupported")
        raise AnnotationError(
            "annotation convention is ambiguous; specify one of "
            f"{[kind.value for kind in candidates]}"
        )


@dataclass(frozen=True, slots=True)
class RequireAnnotation:
    """Treat a caller-supplied convention as an assertion to verify."""

    expected: AnnotationKind

    def choose(
        self,
        candidates: Mapping[AnnotationKind, AnnotationParser],
        /,
    ) -> AnnotationParser:
        try:
            return candidates[self.expected]
        except KeyError as error:
            raise AnnotationError(
                f"annotation source is not valid {self.expected.value} input"
            ) from error


@dataclass(frozen=True, slots=True)
class AnnotationCompiler:
    """User-configured convention recognition and parser composition boundary."""

    recognition: AnnotationRecognition = field(default_factory=DetectAnnotation)
    prolfquapp: ProlfquappAnnotationParameters = field(
        default_factory=ProlfquappAnnotationParameters
    )
    proteobench: ProteobenchAnnotationParameters = field(
        default_factory=ProteobenchAnnotationParameters
    )

    def compile(self, source: Path | pl.DataFrame, /) -> AnnotationParser:
        """Load once, recognize or verify the convention, and return its bound parser."""
        loaded = (
            load_annotation_file(source)
            if isinstance(source, Path)
            else load_annotation_frame(source)
        )
        return self.recognition.choose(self._candidates(loaded))

    def _candidates(
        self,
        source: LoadedAnnotationSource,
    ) -> Mapping[AnnotationKind, AnnotationParser]:
        candidates: dict[AnnotationKind, AnnotationParser] = {}
        if prolfquapp_signature(source, self.prolfquapp):
            candidates[AnnotationKind.PROLFQUAPP] = ProlfquappAnnotationParser(
                source=source,
                parameters=self.prolfquapp,
            )
        if proteobench_signature(source, self.proteobench):
            candidates[AnnotationKind.PROTEOBENCH] = ProteobenchAnnotationParser(
                source=source,
                parameters=self.proteobench,
            )
        return candidates
