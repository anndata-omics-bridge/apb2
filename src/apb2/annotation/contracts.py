"""Small public capabilities returned by sample-annotation compilation."""

from __future__ import annotations

from typing import Protocol

from apb2.annotation.data.model import AnnotationMatches, AnnotationResult
from apb2.parserV2.parse_quant.data.parsed import ParsedLevels


class Annotation(Protocol):
    """A validated sample annotation bound to exactly one parsed dataset."""

    @property
    def matches(self) -> AnnotationMatches:
        """Return completed per-level matching evidence."""
        ...

    def annotate(self) -> AnnotationResult:
        """Apply this annotation to the dataset against which it was validated."""
        ...


class AnnotationParser(Protocol):
    """A convention parser bound to one already loaded annotation source."""

    def parse(self, parsed: ParsedLevels, /) -> Annotation:
        """Validate and match the source, then construct a dataset-bound annotation."""
        ...
