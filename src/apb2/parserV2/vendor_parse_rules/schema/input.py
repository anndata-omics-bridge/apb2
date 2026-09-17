"""Physical input facts authored by one vendor rule document."""

from __future__ import annotations

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase, TableShape
from apb2.parserV2.vendor_parse_rules.schema.base_formats import (
    DetectedDelimiter,
    DetectedEncoding,
    DetectedNumberFormat,
    SupportedExtension,
)


class Input(ModelBase):
    """One table's shape, actual extensions, folder name, and exceptional detection."""

    shape: TableShape
    extensions: list[SupportedExtension] = Field(min_length=1)
    file_name: str | None = Field(default=None, min_length=1)
    sheet_name: str | None = Field(default=None, min_length=1)
    delimiter: DetectedDelimiter | None = None
    numbers: DetectedNumberFormat | None = None
    encoding: DetectedEncoding | None = None

    @model_validator(mode="after")
    def _workbook_consistency(self) -> Input:
        """Keep workbook and delimited declarations mutually exclusive."""
        if self.sheet_name is not None and any(
            value is not None for value in (self.delimiter, self.numbers, self.encoding)
        ):
            raise ValueError("workbook input cannot declare text-format detection")
        if ".xlsx" in self.extensions and self.sheet_name is None:
            raise ValueError(".xlsx input requires sheet_name")
        if self.sheet_name is not None and ".parquet" in self.extensions:
            raise ValueError("workbook input cannot declare Parquet extensions")
        return self
