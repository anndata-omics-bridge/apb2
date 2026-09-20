"""Measurement parameters: duplicate presence, canonical values, and validation.

Raw presence and final value parsing are two questions about the same authored layer.
Presence decides whether a raw scalar claims a duplicate cell. Value parsing decides the
canonical scalar stored in ``ParsedLevels`` after duplicate resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

type DuplicateMode = Literal["error", "keep_first", "aggregate"]
"""How several raw scalars claiming one measurement cell become one scalar."""

type NumericType = Literal["number", "integer"]
"""Logical numeric type declared for a measurement layer."""


# --------------------------------------------------------------- projected declarations


@dataclass(frozen=True, slots=True)
class PlainNumericLayerDeclaration:
    """A layer whose cells are directly parseable numbers."""

    missing_values: tuple[float, ...]
    type: NumericType = "number"
    kind: Literal["plain_numeric"] = field(default="plain_numeric", init=False)


@dataclass(frozen=True, slots=True)
class RegexNumericLayerDeclaration:
    """A numeric layer extracted from one capture of a structured cell."""

    missing_values: tuple[float, ...]
    pattern: str
    type: NumericType = "number"
    kind: Literal["regex_numeric"] = field(default="regex_numeric", init=False)


@dataclass(frozen=True, slots=True)
class FactorLayerDeclaration:
    """A layer whose category labels have declared integer codes."""

    categories: tuple[tuple[str, int], ...]
    kind: Literal["factor"] = field(default="factor", init=False)


type LayerValueDeclaration = (
    PlainNumericLayerDeclaration | RegexNumericLayerDeclaration | FactorLayerDeclaration
)


@dataclass(frozen=True, slots=True)
class LayerValueConfig:
    """One retained layer and the declaration configuring its value parser."""

    layer_name: str
    value: LayerValueDeclaration


@dataclass(frozen=True, slots=True)
class WorkingMeasurementLayer:
    """One named measurement and its canonical value declaration."""

    name: str
    source: str
    value: LayerValueDeclaration
    roles: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkingMeasurements:
    """Validated measurements, in authored order; the facade makes the primary required."""

    primary_layer_name: str
    duplicate_mode: DuplicateMode
    layers: tuple[WorkingMeasurementLayer, ...]
    required_names: frozenset[str]

    @property
    def required_layers(self) -> tuple[WorkingMeasurementLayer, ...]:
        """Return required layers without changing declaration order."""
        return tuple(layer for layer in self.layers if layer.name in self.required_names)
