"""How one layer's values become numbers in a matrix cell: the ``layers[]`` runtime.

``selectors.coercion_for`` reads the ``encoding_mode`` flag **once** and it becomes a type
here. Past that point the illegal combinations are not rejected, they are unrepresentable:
a ``FactorCoercion`` has no ``missing_values`` field to set. Splitting numeric into two
types rather than giving one a ``pattern: str | None`` is the same rule one level down: an
optional field whose presence selects behaviour is a discriminator wearing ``| None``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_UNKNOWN_FACTOR_CODE = -1
# A layer this empty is almost always a rule defect (wrong source column, or a structured
# vendor string coerced as a bare number) rather than a genuinely unobserved measurement.
_ALL_MISSING_THRESHOLD = 0.999


class FactorCoercion:
    """Map declared category strings to integer codes; unknown values (and NaN) to -1."""

    def __init__(self, categories: Mapping[str, int]) -> None:
        self.categories = dict(categories)

    def coerce(self, series: pd.Series) -> pd.Series:
        return series.map(self.categories).fillna(_UNKNOWN_FACTOR_CODE).astype("int64")


class PlainNumericCoercion:
    """Read directly parseable scalar values, blanking the declared missing ones."""

    def __init__(self, missing_values: tuple[float, ...]) -> None:
        self.missing_values = missing_values

    def coerce(self, series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        values = values.mask(values.isin(self.missing_values))
        # Layers are float64 end to end. `to_numeric` returns a nullable dtype for
        # nullable input, and pandas 2.3 `bfill(axis=1)` misbehaves on a single-column
        # nullable frame — it fills down the column instead of across.
        return values.astype("float64")


class RegexNumericCoercion:
    """Extract one numeric capture group per cell before coercing, for structured values."""

    def __init__(self, missing_values: tuple[float, ...], pattern: str) -> None:
        self.missing_values = missing_values
        self.pattern = pattern

    def coerce(self, series: pd.Series) -> pd.Series:
        extracted = series.astype("string").str.extract(self.pattern, expand=False)
        return PlainNumericCoercion(self.missing_values).coerce(extracted)


type LayerCoercion = FactorCoercion | PlainNumericCoercion | RegexNumericCoercion


class LayerPlan:
    """One layer's identity, its source, and how its values are read.

    ``required`` folds in the rule-level question — a layer is required if it says so or
    if it is the axis ``x_layer`` — so a conversion never needs the rule to decide whether
    an absent source is a skip or an error.
    """

    def __init__(self, *, name: str, source: str, required: bool, coercion: LayerCoercion) -> None:
        self.name = name
        self.source = source
        self.required = required
        self.coercion = coercion

    def coerce(self, series: pd.Series) -> pd.Series:
        return self.coercion.coerce(series)


def warn_if_all_missing(matrix: NDArray[np.float64], layer_name: str) -> None:
    """Warn when a captured layer is (near-)entirely NaN.

    A layer whose source columns were found but which holds no values at all points at a
    rule defect; without this it reaches the output as a silently empty matrix.
    """
    if matrix.size == 0:
        return
    missing = float(np.isnan(matrix).mean())
    if missing >= _ALL_MISSING_THRESHOLD:
        logger.warning(
            "layer %r is %.1f%% missing after numeric coercion; its source columns "
            "matched but hold no usable numbers — check the rule's 'value_pattern' "
            "and 'missing_values'",
            layer_name,
            100 * missing,
        )
