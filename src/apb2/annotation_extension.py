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
from apb2.annotation.sdrf import SdrfSource
from apb2.annotation.source.load import load_annotation_file

__all__ = [
    "AnnotationApplication",
    "AnnotationError",
    "AnnotationFileOrigin",
    "AnnotationMatches",
    "AnnotationResult",
    "RequireCompleteAnnotation",
    "SdrfSource",
    "annotation_matching_for",
    "load_annotation_file",
    "make_annotation_table",
    "match_annotation",
    "record_annotation_provenance",
]
