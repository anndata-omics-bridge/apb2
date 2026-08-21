"""Measurement, duplicate, layer, and value-pattern declarations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Discriminator, Field, Tag, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import DuplicateMode, ModelBase


class Duplicates(ModelBase):
    """How repeated raw measurement cells are resolved."""

    mode: DuplicateMode = "error"


class NoValuePattern(ModelBase):
    """The layer already contains scalar values."""

    mode: Literal["none"] = "none"


class RegexValuePattern(ModelBase):
    """Extract one numeric capture group from each structured layer value."""

    mode: Literal["regex"] = "regex"
    pattern: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pattern(self) -> RegexValuePattern:
        try:
            compiled = re.compile(self.pattern)
        except re.error as error:
            raise ValueError(f"value_pattern is not a valid regex: {error}") from error
        if compiled.groups != 1:
            raise ValueError(
                f"value_pattern must have exactly one capture group, found {compiled.groups}"
            )
        return self


type ValuePattern = Annotated[
    NoValuePattern | RegexValuePattern,
    Field(discriminator="mode"),
]


class NumericLayer(ModelBase):
    """A quantitative layer encoded numerically only at an output boundary."""

    encoding_mode: Literal["numeric"] = "numeric"
    name: str
    source: str
    missing_values: list[float] = Field(default_factory=list)
    value_pattern: ValuePattern = Field(default_factory=NoValuePattern)
    required: bool = False


class FactorLayer(ModelBase):
    """A categorical layer encoded through its declared category map."""

    encoding_mode: Literal["factor"]
    name: str
    source: str
    categories: dict[str, int] = Field(min_length=1)
    required: bool = False


def _layer_encoding(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("encoding_mode", "numeric"))
    return str(getattr(value, "encoding_mode", "numeric"))


type Layer = Annotated[
    Annotated[NumericLayer, Tag("numeric")] | Annotated[FactorLayer, Tag("factor")],
    Discriminator(_layer_encoding),
]


class Measurements(ModelBase):
    """Named measurements, their primary layer, and raw duplicate policy."""

    primary_layer: str
    duplicates: Duplicates = Field(default_factory=Duplicates)
    layers: list[Layer] = Field(min_length=1)


def layer_required(primary_layer: str, layer: Layer) -> bool:
    """Whether a layer is primary or explicitly required."""
    return layer.required or layer.name == primary_layer
