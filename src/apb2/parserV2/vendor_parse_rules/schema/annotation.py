"""Optional sample-annotation matching policy declared by a vendor rule."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase


class FuzzySampleMatching(ModelBase):
    """Unambiguous mutual-best token matching after exact identifiers are reserved."""

    mode: Literal["fuzzy"]
    cutoff: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    near_miss_limit: int = Field(ge=1, le=50)


class SampleAnnotation(ModelBase):
    """Document-level policy used when later matching sample annotations."""

    matching: FuzzySampleMatching
