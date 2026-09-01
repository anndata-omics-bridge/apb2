"""Public capabilities for explicitly composed observation-annotation interpreters."""

from apb2.annotation.application.policies import (
    AnnotationApplication,
    RequireCompleteAnnotation,
    record_annotation_provenance,
)
from apb2.annotation.data.model import (
    AnnotationError,
    AnnotationFileOrigin,
    AnnotationMatches,
    AnnotationResult,
)
from apb2.annotation.matching.core import (
    annotation_matching_for,
    make_annotation_table,
    match_annotation,
)

__all__ = [
    "AnnotationApplication",
    "AnnotationError",
    "AnnotationFileOrigin",
    "AnnotationMatches",
    "AnnotationResult",
    "RequireCompleteAnnotation",
    "annotation_matching_for",
    "make_annotation_table",
    "match_annotation",
    "record_annotation_provenance",
]
