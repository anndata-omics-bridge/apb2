"""Measurement parameters: duplicate presence, canonical values, and validation.

Raw presence and final value parsing are two questions about the same authored layer.
Presence decides whether a raw scalar claims a duplicate cell. Value parsing decides the
canonical scalar stored in ``ParsedLevels`` after duplicate resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

type DuplicateMode = Literal["error", "keep_first", "aggregate"]
"""How several raw scalars claiming one measurement cell become one scalar."""

type NumericType = Literal["number", "integer"]
"""Logical numeric type declared for a measurement layer."""


# ---------------------------------------------------------------------------- raw presence


@dataclass(frozen=True, slots=True)
class NullOnlyRawValuePresenceConfig:
    """Only null claims nothing: factors and native numeric layers without sentinels."""

    kind: Literal["null_only"]
    layer_name: str


@dataclass(frozen=True, slots=True)
class PlainNumericRawValuePresenceConfig:
    """Null, blank text, and the declared missing values claim nothing."""

    kind: Literal["plain_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    number_format: NumericTextFormat


@dataclass(frozen=True, slots=True)
class RegexNumericRawValuePresenceConfig:
    """As plain numeric, but the comparable number is one capture of a structured token."""

    kind: Literal["regex_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumericTextFormat


type RawValuePresenceConfig = (
    NullOnlyRawValuePresenceConfig
    | PlainNumericRawValuePresenceConfig
    | RegexNumericRawValuePresenceConfig
)


# --------------------------------------------------------------------- canonical layer values


@dataclass(frozen=True, slots=True)
class PlainNumericLayerConfig:
    """Directly parseable scalars; declared missing values become missing."""

    kind: Literal["plain_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    number_format: NumericTextFormat
    type: NumericType = "number"


@dataclass(frozen=True, slots=True)
class RegexNumericLayerConfig:
    """One numeric capture per structured token, then plain numeric conversion."""

    kind: Literal["regex_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumericTextFormat
    type: NumericType = "number"


@dataclass(frozen=True, slots=True)
class FactorLayerConfig:
    """Declared category labels become their codes; null and unknown labels become -1."""

    kind: Literal["factor"]
    layer_name: str
    categories: tuple[tuple[str, int], ...]


type LayerValueConfig = PlainNumericLayerConfig | RegexNumericLayerConfig | FactorLayerConfig


@dataclass(frozen=True, slots=True)
class LayerContractConfig:
    """The occupancy policy a canonical layer set must satisfy.

    A layer is suspicious only below ``empty_ratio`` while a sibling reaches
    ``populated_ratio``: without a populated sibling, occupancy cannot tell an empty
    experiment from a parse failure.
    """

    primary_layer_name: str
    required_names: tuple[str, ...]
    empty_ratio: float
    populated_ratio: float
