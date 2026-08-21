"""Packed-fragment declarations for long fragment-level sources."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase

type Delimiter = Annotated[str, Field(min_length=1, max_length=1)]


class PositionalFragments(ModelBase):
    """Packed fragment values whose labels are synthesized by token position."""

    label_strategy: Literal["positional"]
    value_columns: list[str] = Field(min_length=1)
    delimiter: Delimiter = ";"
    label_output: str = "fragment_label"


class ColumnLabeledFragments(ModelBase):
    """Packed fragment values with labels stored in a parallel packed column."""

    label_strategy: Literal["column"]
    value_columns: list[str] = Field(min_length=1)
    label_column: str
    delimiter: Delimiter = ";"
    label_output: str = "fragment_label"

    @model_validator(mode="after")
    def _label_is_not_a_value_column(self) -> ColumnLabeledFragments:
        if self.label_column in self.value_columns:
            raise ValueError("fragments.label_column must not also be a packed value column")
        return self


type Fragments = Annotated[
    PositionalFragments | ColumnLabeledFragments,
    Field(discriminator="label_strategy"),
]
