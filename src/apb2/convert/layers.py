"""How a rule's layer declaration becomes values in a matrix cell.

``Layer`` is a document: it describes what a rule file may say, and it says it with a
mode flag plus three fields that belong to only one mode. Its validator spends three
rejections enforcing that — ``categories`` required for factor, ``missing_values`` and
``value_pattern`` forbidden for it — which is the written admission that the document
describes more than one thing.

The document keeps the flag and keeps the validator, because a person can write that
combination in a rule file and must be told. This module is where the flag is read **once**
and becomes a type. Past this point the illegal combinations are not rejected, they are
unrepresentable: ``FactorCoercion`` has no ``missing_values`` field to set.

Splitting numeric into two types rather than giving one a ``pattern: str | None`` is the
same rule applied one level down: an optional field whose presence selects behaviour is a
discriminator wearing ``| None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from apb2.convert.factors import encode_factor
from apb2.convert.numeric import coerce_numeric, coerce_regex_numeric
from apb2.vendor_parse_rules.model import Layer, RegexValuePattern


@dataclass(frozen=True, slots=True)
class FactorCoercion:
    """Map declared category strings to integer codes."""

    categories: Mapping[str, int]

    def coerce(self, series: pd.Series) -> pd.Series:
        return encode_factor(series, self.categories)


@dataclass(frozen=True, slots=True)
class PlainNumericCoercion:
    """Read directly parseable scalar values, blanking the declared missing ones."""

    missing_values: tuple[float, ...]

    def coerce(self, series: pd.Series) -> pd.Series:
        return coerce_numeric(series, self.missing_values)


@dataclass(frozen=True, slots=True)
class RegexNumericCoercion:
    """Extract one numeric capture group per cell before coercing, for structured values."""

    missing_values: tuple[float, ...]
    pattern: str

    def coerce(self, series: pd.Series) -> pd.Series:
        return coerce_regex_numeric(series, self.missing_values, self.pattern)


type LayerCoercion = FactorCoercion | PlainNumericCoercion | RegexNumericCoercion


def make_layer_coercion(layer: Layer) -> LayerCoercion:
    """Read a layer document's encoding flag once, and return the type it names."""
    if layer.encoding_mode == "factor":
        return FactorCoercion(dict(layer.categories))
    missing_values = tuple(layer.missing_values)
    if isinstance(layer.value_pattern, RegexValuePattern):
        return RegexNumericCoercion(missing_values, layer.value_pattern.pattern)
    return PlainNumericCoercion(missing_values)


@dataclass(frozen=True, slots=True)
class LayerPlan:
    """One layer's identity, its source, and how its values are read.

    ``required`` folds in the rule-level question ``layer_required`` used to ask — a layer is
    required if it says so or if it is the axis ``x_layer`` — so a conversion never needs the
    rule to decide whether an absent source is a skip or an error.
    """

    name: str
    source: str
    required: bool
    coercion: LayerCoercion

    def coerce(self, series: pd.Series) -> pd.Series:
        return self.coercion.coerce(series)


def make_layer_plan(layer: Layer, x_layer: str) -> LayerPlan:
    """Resolve one layer declaration against the axis that decides whether it is required."""
    return LayerPlan(
        name=layer.name,
        source=layer.source,
        required=layer.required or layer.name == x_layer,
        coercion=make_layer_coercion(layer),
    )
