"""Physical input facts authored by one vendor rule document."""

from __future__ import annotations

from pydantic import Field

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase, TableShape
from apb2.parserV2.vendor_parse_rules.schema.base_formats import (
    DetectedDelimiter,
    DetectedNumberFormat,
    SupportedExtension,
)


class Input(ModelBase):
    """One table's shape, actual extensions, folder name, and exceptional detection."""

    shape: TableShape
    extensions: list[SupportedExtension] = Field(min_length=1)
    file_name: str | None = Field(default=None, min_length=1)
    delimiter: DetectedDelimiter | None = None
    numbers: DetectedNumberFormat | None = None
