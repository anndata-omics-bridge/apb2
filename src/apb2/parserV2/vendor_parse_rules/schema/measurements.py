"""Measurement, duplicate, layer, and value-pattern declarations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Discriminator, Field, Tag, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import DuplicateMode, ModelBase
from apb2.parserV2.vendor_parse_rules.schema.roles import SemanticRole

_MISSING_BOUND = r"^<=\s*(-?\d+(?:\.\d+)?)$"
type MissingBound = Annotated[str, Field(pattern=_MISSING_BOUND)]
"""A ``<=`` threshold: every number at or below it is missing, e.g. ``"<=0"``."""


class Duplicates(ModelBase):
    """How repeated raw measurement cells are resolved.

    ``keep_best`` keeps, in each cell, the repeated row whose ``by`` layer is ``best``, and
    takes every layer's value from that one row, so one feature's intensity, score and
    q-value describe the same identification, such as one PSM.
    """

    mode: DuplicateMode = "error"
    by: str | None = Field(
        default=None, description="keep_best: the numeric layer that ranks repeated rows"
    )
    best: Literal["highest", "lowest"] = Field(
        default="highest", description="keep_best: which value of the 'by' layer wins"
    )

    @model_validator(mode="after")
    def _ranking_only_for_keep_best(self) -> Duplicates:
        if (self.mode == "keep_best") != (self.by is not None):
            raise ValueError("'by' names the ranking layer exactly when mode is keep_best")
        if self.mode != "keep_best" and self.best != "highest":
            raise ValueError("'best' applies only to keep_best")
        return self


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
    type: Literal["number", "integer"] = "number"
    name: str
    source: str
    missing_values: list[float | MissingBound] = Field(default_factory=list)
    value_pattern: ValuePattern = Field(default_factory=NoValuePattern)
    required: bool = False
    roles: list[SemanticRole] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_one_threshold(self) -> NumericLayer:
        thresholds = [value for value in self.missing_values if isinstance(value, str)]
        if len(thresholds) > 1:
            raise ValueError(f"missing_values declares more than one threshold: {thresholds}")
        return self

    @property
    def missing_sentinels(self) -> tuple[float, ...]:
        """The exact numbers declared missing."""
        return tuple(value for value in self.missing_values if isinstance(value, float))

    @property
    def missing_at_or_below(self) -> float | None:
        """The declared ``<=`` threshold, or ``None`` without one."""
        for value in self.missing_values:
            if isinstance(value, str) and (match := re.match(_MISSING_BOUND, value)):
                return float(match.group(1))
        return None


class FactorLayer(ModelBase):
    """A categorical layer encoded through its declared category map."""

    encoding_mode: Literal["factor"]
    name: str
    source: str
    categories: dict[str, int] = Field(min_length=1)
    required: bool = False
    roles: list[SemanticRole] = Field(default_factory=list)


def _layer_encoding(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("encoding_mode", "numeric"))
    return str(getattr(value, "encoding_mode", "numeric"))


type Layer = Annotated[
    Annotated[NumericLayer, Tag("numeric")] | Annotated[FactorLayer, Tag("factor")],
    Discriminator(_layer_encoding),
]


class Measurements(ModelBase):
    """Named measurements, their primary layer, and raw duplicate policy.

    ``sample_layer`` names the wide layer whose header captures are the sample names;
    omitted, the primary layer supplies them.
    """

    primary_layer: str
    sample_layer: str | None = None
    duplicates: Duplicates = Field(default_factory=Duplicates)
    layers: list[Layer] = Field(min_length=1)

    @model_validator(mode="after")
    def _ranking_layer_is_declared(self) -> Measurements:
        ranking = self.duplicates.by
        if ranking is not None and ranking not in {layer.name for layer in self.layers}:
            raise ValueError(f"duplicates rank by {ranking!r}, which names no declared layer")
        return self


def layer_required(measurements: Measurements, layer: Layer) -> bool:
    """Whether a layer is primary, supplies the sample names, or is explicitly required."""
    return layer.required or layer.name in {measurements.primary_layer, measurements.sample_layer}
