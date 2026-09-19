"""Measurement parameters: duplicate presence, canonical values, and validation.

Raw presence and final value parsing are two questions about the same authored layer.
Presence decides whether a raw scalar claims a duplicate cell. Value parsing decides the
canonical scalar stored in ``ParsedLevels`` after duplicate resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
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


# --------------------------------------------------------------- projected declarations


@dataclass(frozen=True, slots=True)
class PlainNumericLayerDeclaration:
    """A layer whose cells are directly parseable numbers."""

    missing_values: tuple[float, ...]
    type: NumericType = "number"

    def raw_presence_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> RawValuePresenceConfig:
        """Build the duplicate-presence rule for this layer."""
        if not self.missing_values:
            return NullOnlyRawValuePresenceConfig(kind="null_only", layer_name=layer_name)
        return PlainNumericRawValuePresenceConfig(
            kind="plain_numeric",
            layer_name=layer_name,
            missing_values=self.missing_values,
            number_format=numbers,
        )

    def canonical_value_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> LayerValueConfig:
        """Build the canonical scalar conversion for this layer."""
        return PlainNumericLayerConfig(
            kind="plain_numeric",
            layer_name=layer_name,
            missing_values=self.missing_values,
            number_format=numbers,
            type=self.type,
        )

    def supports_native_numeric_read(self) -> bool:
        """Whether an aggregating parser may read this source as numeric."""
        return True


@dataclass(frozen=True, slots=True)
class RegexNumericLayerDeclaration:
    """A numeric layer extracted from one capture of a structured cell."""

    missing_values: tuple[float, ...]
    pattern: str
    type: NumericType = "number"

    def raw_presence_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> RawValuePresenceConfig:
        """Build the duplicate-presence rule for this layer."""
        return RegexNumericRawValuePresenceConfig(
            kind="regex_numeric",
            layer_name=layer_name,
            missing_values=self.missing_values,
            pattern=self.pattern,
            number_format=numbers,
        )

    def canonical_value_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> LayerValueConfig:
        """Build the canonical scalar conversion for this layer."""
        return RegexNumericLayerConfig(
            kind="regex_numeric",
            layer_name=layer_name,
            missing_values=self.missing_values,
            pattern=self.pattern,
            number_format=numbers,
            type=self.type,
        )

    def supports_native_numeric_read(self) -> bool:
        """Whether an aggregating parser may read this source as numeric."""
        return False


@dataclass(frozen=True, slots=True)
class FactorLayerDeclaration:
    """A layer whose category labels have declared integer codes."""

    categories: tuple[tuple[str, int], ...]

    def raw_presence_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> RawValuePresenceConfig:
        """Build the duplicate-presence rule for this layer."""
        del numbers
        return NullOnlyRawValuePresenceConfig(kind="null_only", layer_name=layer_name)

    def canonical_value_config(
        self, layer_name: str, numbers: NumericTextFormat
    ) -> LayerValueConfig:
        """Build the canonical category conversion for this layer."""
        del numbers
        return FactorLayerConfig(kind="factor", layer_name=layer_name, categories=self.categories)

    def supports_native_numeric_read(self) -> bool:
        """Whether an aggregating parser may read this source as numeric."""
        return False


type LayerValueDeclaration = (
    PlainNumericLayerDeclaration | RegexNumericLayerDeclaration | FactorLayerDeclaration
)


@dataclass(frozen=True, slots=True)
class WorkingMeasurementLayer:
    """One named measurement and its canonical value declaration."""

    name: str
    source: str
    value: LayerValueDeclaration
    roles: tuple[str, ...] = ()

    def raw_presence_config(self, numbers: NumericTextFormat) -> RawValuePresenceConfig:
        """Build this layer's duplicate-presence configuration."""
        return self.value.raw_presence_config(self.name, numbers)

    def canonical_value_config(self, numbers: NumericTextFormat) -> LayerValueConfig:
        """Build this layer's canonical scalar configuration."""
        return self.value.canonical_value_config(self.name, numbers)

    def supports_native_numeric_read(self) -> bool:
        """Whether an aggregating parser may read this layer as numeric."""
        return self.value.supports_native_numeric_read()


class WorkingMeasurements:
    """A valid ordered measurement collection with one required primary layer."""

    __slots__ = ("_duplicate_mode", "_layers", "_primary_layer_name", "_required_names")
    _duplicate_mode: DuplicateMode
    _layers: tuple[WorkingMeasurementLayer, ...]
    _primary_layer_name: str
    _required_names: frozenset[str]

    def __init__(
        self,
        primary_layer_name: str,
        duplicate_mode: DuplicateMode,
        layers: Iterable[WorkingMeasurementLayer],
        required_names: Iterable[str],
    ) -> None:
        """Establish unique layers, one existing primary, and consistent required views."""
        ordered_layers = tuple(layers)
        names = tuple(layer.name for layer in ordered_layers)
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"measurement layer names must be unique; duplicated: {duplicates}")
        if primary_layer_name not in set(names):
            raise ValueError(
                f"primary measurement layer {primary_layer_name!r} is not among {list(names)}"
            )

        required_sequence = tuple(required_names)
        duplicated_required = sorted(
            name for name in set(required_sequence) if required_sequence.count(name) > 1
        )
        if duplicated_required:
            raise ValueError(
                f"required measurement names must be unique; duplicated: {duplicated_required}"
            )
        required = frozenset(required_sequence)
        unknown = sorted(required - set(names))
        if unknown:
            raise ValueError(f"required measurement layers are not declared: {unknown}")
        if primary_layer_name not in required:
            raise ValueError(f"primary measurement layer {primary_layer_name!r} must be required")

        self._primary_layer_name = primary_layer_name
        self._duplicate_mode = duplicate_mode
        self._layers = ordered_layers
        self._required_names = required

    @property
    def primary_layer_name(self) -> str:
        """Return the layer that defines occupancy and the wide observation axis."""
        return self._primary_layer_name

    @property
    def duplicate_mode(self) -> DuplicateMode:
        """Return how duplicate raw cells are resolved."""
        return self._duplicate_mode

    @property
    def authored_order(self) -> tuple[str, ...]:
        """Return layer names in declaration order."""
        return tuple(layer.name for layer in self._layers)

    @property
    def required_layers(self) -> tuple[WorkingMeasurementLayer, ...]:
        """Return required layers without changing declaration order."""
        return tuple(layer for layer in self._layers if layer.name in self._required_names)

    @property
    def optional_layers(self) -> tuple[WorkingMeasurementLayer, ...]:
        """Return optional layers without changing declaration order."""
        return tuple(layer for layer in self._layers if layer.name not in self._required_names)

    def authored_layers(self) -> tuple[WorkingMeasurementLayer, ...]:
        """Return every layer in declaration order."""
        return self._layers

    def required_sources(self) -> tuple[str, ...]:
        """Return source declarations whose absence makes the level incompatible."""
        return tuple(layer.source for layer in self.required_layers)
