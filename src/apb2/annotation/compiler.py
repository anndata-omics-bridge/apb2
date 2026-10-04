"""Compile one delimited annotation source into a source-bound parser."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from apb2.annotation.application.policies import (
    AllAnnotationSelections,
    AnnotationApplication,
    BooleanAnnotationSelection,
    KeepUnmatchedAnnotation,
    MatchedAnnotationSelection,
    ObservationSelection,
    RequireCompleteAnnotation,
    SelectAnnotatedObservations,
)
from apb2.annotation.contracts import AnnotationParser
from apb2.annotation.data.model import AnnotationError
from apb2.annotation.prolfquapp import ProlfquappAnnotationParser, prolfquapp_signature
from apb2.annotation.sdrf import SdrfAnnotationParser, SdrfSource, sdrf_signature
from apb2.annotation.source.load import load_annotation_file, load_annotation_frame


class AnnotationCompiler:
    """User-configured parser for SDRF and generic delimited observation annotations."""

    __slots__ = ("_application",)

    def __init__(
        self,
        unmatched: Literal["keep", "error", "drop"] = "keep",
        include: str | None = None,
    ) -> None:
        """Choose what happens to observations the annotation does not cover.

        Args:
            unmatched: ``keep`` them with null metadata, raise (``error``), or ``drop`` them.
            include: A Boolean annotation column that further selects observations; requires
                ``unmatched="drop"``.

        Raises:
            AnnotationError: ``include`` is given without ``unmatched="drop"``.
        """
        self._application = _application(unmatched, include)

    def compile(self, source: Path | pl.DataFrame) -> AnnotationParser:
        """Load once, verify the tabular convention, and return its bound parser.

        SDRF headers take precedence; other tables must carry a prolfquapp observation key.
        """
        loaded = (
            load_annotation_file(source)
            if isinstance(source, Path)
            else load_annotation_frame(source)
        )
        if sdrf_signature(loaded):
            return SdrfAnnotationParser(source=SdrfSource(loaded), application=self._application)
        if not prolfquapp_signature(loaded):
            raise AnnotationError("annotation table has no supported observation key")
        return ProlfquappAnnotationParser(source=loaded, application=self._application)


def _application(
    unmatched: Literal["keep", "error", "drop"], include: str | None
) -> AnnotationApplication:
    if unmatched == "drop":
        selections: list[ObservationSelection] = [MatchedAnnotationSelection()]
        if include is not None:
            selections.append(BooleanAnnotationSelection(include))
        return SelectAnnotatedObservations(AllAnnotationSelections(tuple(selections)))
    if include is not None:
        raise AnnotationError("include requires unmatched='drop'")
    return KeepUnmatchedAnnotation() if unmatched == "keep" else RequireCompleteAnnotation()
