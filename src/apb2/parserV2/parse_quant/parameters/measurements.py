"""Measurement parameters: duplicate mode, raw presence, and AnnData serialization.

Raw presence and AnnData encoding are two projections of the same authored layer, kept
apart on purpose. Presence answers only "does this raw scalar claim its cell" and is needed
by every parse whose duplicate policy must skip a declared sentinel. Encoding answers "what
does this scalar become in a dense float matrix" and is constructed only when the caller
asked for AnnData — a Parquet compile builds no encoder at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from apb2.parserV2.parse_quant.parameters.source import NumericTextFormat

type DuplicateMode = Literal["error", "keep_first", "aggregate"]
"""How several raw scalars claiming one measurement cell become one scalar."""


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


# ------------------------------------------------------------------------- AnnData encoding


@dataclass(frozen=True, slots=True)
class PlainNumericAnnDataEncodingConfig:
    """Directly parseable scalars; declared missing values become missing."""

    kind: Literal["plain_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    number_format: NumericTextFormat


@dataclass(frozen=True, slots=True)
class RegexNumericAnnDataEncodingConfig:
    """One numeric capture per structured token, then plain numeric conversion."""

    kind: Literal["regex_numeric"]
    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumericTextFormat


@dataclass(frozen=True, slots=True)
class FactorAnnDataEncodingConfig:
    """Declared category labels become their codes; null and unknown labels become -1."""

    kind: Literal["factor"]
    layer_name: str
    categories: tuple[tuple[str, int], ...]


type AnnDataLayerEncodingConfig = (
    PlainNumericAnnDataEncodingConfig
    | RegexNumericAnnDataEncodingConfig
    | FactorAnnDataEncodingConfig
)


@dataclass(frozen=True, slots=True)
class AnnDataLayerContractConfig:
    """The occupancy policy an encoded layer set must satisfy.

    A layer is suspicious only below ``empty_ratio`` while a sibling reaches
    ``populated_ratio``: without a populated sibling, occupancy cannot tell an empty
    experiment from a parse failure.
    """

    primary_layer_name: str
    required_names: tuple[str, ...]
    empty_ratio: float
    populated_ratio: float


@dataclass(frozen=True, slots=True)
class AnnDataSerializationConfig:
    """Everything AnnData-only about one level; routed to ``AnnDataWriter`` construction."""

    layer_encodings: tuple[AnnDataLayerEncodingConfig, ...]
    layer_contract: AnnDataLayerContractConfig
