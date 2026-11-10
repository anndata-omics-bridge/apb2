"""Measurement, repeated-cell, layer, and value-pattern declarations."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import (
    DuplicateMode,
    ModelBase,
    Note,
    VersionPattern,
)
from apb2.parserV2.vendor_parse_rules.schema.roles import SemanticRole

_MISSING_BOUND = r"^<=\s*(-?\d+(?:\.\d+)?)$"
type MissingBound = Annotated[str, Field(pattern=_MISSING_BOUND)]
"""A ``<=`` threshold: every number at or below it is missing, e.g. ``"<=0"``."""


class RepeatedCells(ModelBase):
    """How repeated raw measurement cells are resolved: one row for every layer, or all rows.

    ``error`` stops at a second row for a cell, whatever it holds. ``same`` takes the value
    the rows share, for rows that are copies, such as a protein value repeated on each
    precursor row, and stops where their values differ. ``sum`` adds up every layer over the
    repeated rows, so it suits only levels whose layers are all additive.
    ``keep_best`` keeps, in each cell, the repeated row whose ``by`` layer is ``best``, and
    takes every layer's value from that one row, so one feature's intensity, score and
    q-value describe the same identification, such as one PSM. Layers listed in ``sum`` are
    added up over the repeated rows instead, such as quantities split across several of them.
    """

    mode: DuplicateMode
    by: str | None = Field(
        default=None, description="keep_best: the numeric layer that ranks repeated rows"
    )
    best: Literal["highest", "lowest"] | None = Field(
        default=None, description="keep_best: which value of the 'by' layer wins"
    )
    sum: list[str] = Field(
        default_factory=list,
        description="keep_best: layers added up over the repeated rows, not taken from the "
        "winning row",
    )

    @model_validator(mode="after")
    def _ranking_only_for_keep_best(self) -> RepeatedCells:
        keep_best = self.mode == "keep_best"
        if keep_best != (self.by is not None):
            raise ValueError("'by' names the ranking layer exactly when mode is keep_best")
        if keep_best != (self.best is not None):
            raise ValueError("'best' names the winning value exactly when mode is keep_best")
        if not keep_best and self.sum:
            raise ValueError("'sum' applies only to keep_best")
        if self.by in self.sum:
            raise ValueError(f"keep_best ranks by {self.by!r}, so it cannot also sum it")
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
    """A quantitative layer encoded numerically only at an output boundary.

    ``encoding_mode``, ``type`` and ``required`` come from the layer or its table's
    ``defaults``.
    """

    encoding_mode: Literal["numeric"]
    type: Literal["number", "integer"]
    name: str
    source: str
    version: VersionPattern = None
    missing_values: list[float | MissingBound] = Field(default_factory=list)
    missing_tokens: list[str] = Field(default_factory=list)
    """Text the vendor writes for a missing value, such as ``-`` or ``NA``; such cells are
    missing, not unreadable."""
    value_pattern: ValuePattern = Field(default_factory=NoValuePattern)
    required: bool
    roles: list[SemanticRole] = Field(default_factory=list)
    note: Note = None

    @model_validator(mode="after")
    def _validate_one_threshold(self) -> NumericLayer:
        thresholds = [value for value in self.missing_values if isinstance(value, str)]
        if len(thresholds) > 1:
            raise ValueError(f"missing_values declares more than one threshold: {thresholds}")
        if any(not token.strip() for token in self.missing_tokens):
            raise ValueError("missing_tokens declares blank text, which is already missing")
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
    version: VersionPattern = None
    categories: dict[str, int] = Field(min_length=1)
    required: bool
    roles: list[SemanticRole] = Field(default_factory=list)
    note: Note = None


type Layer = Annotated[NumericLayer | FactorLayer, Field(discriminator="encoding_mode")]


class Measurements(ModelBase):
    """Named measurements, their primary layer, and the policy for repeated raw cells.

    ``sample_layer`` names the wide layer whose header captures are the sample names;
    omitted, the primary layer supplies them.
    """

    primary_layer: str
    sample_layer: str | None = None
    repeated_cells: RepeatedCells
    layers: list[Layer] = Field(min_length=1)

    @model_validator(mode="after")
    def _ranking_layer_is_declared(self) -> Measurements:
        ranking = self.repeated_cells.by
        if ranking is not None and ranking not in {layer.name for layer in self.layers}:
            raise ValueError(f"repeated_cells rank by {ranking!r}, which names no declared layer")
        return self


def layer_required(measurements: Measurements, layer: Layer) -> bool:
    """Whether a layer is primary, supplies the sample names, or is explicitly required."""
    return layer.required or layer.name in {measurements.primary_layer, measurements.sample_layer}
