"""Normalize vendor-reported numeric layer values."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from apb2.convert._pieces import DenseLayerMatrix

logger = logging.getLogger(__name__)

# A layer this empty is almost always a rule defect (wrong source column, or a
# structured vendor string coerced as if it were a bare number) rather than a
# genuinely unobserved measurement.
_ALL_MISSING_THRESHOLD = 0.999


def coerce_numeric(series: pd.Series, missing_values: Sequence[float]) -> pd.Series:
    """Coerce plain numeric values and replace declared missing values with NaN."""
    values = pd.to_numeric(series, errors="coerce")
    values = values.mask(values.isin(missing_values))
    # Layers are float64 end to end (see `_gather_layer_matrix`). `to_numeric` returns a
    # nullable dtype for nullable input, and pandas 2.3 `bfill(axis=1)` misbehaves on a
    # single-column nullable frame — it fills down the column instead of across.
    return values.astype("float64")


def coerce_regex_numeric(
    series: pd.Series,
    missing_values: Sequence[float],
    value_pattern: str,
) -> pd.Series:
    """Extract one regex capture from structured cells and coerce it to numeric."""
    extracted = series.astype("string").str.extract(value_pattern, expand=False)
    return coerce_numeric(extracted, missing_values)


def warn_if_all_missing(matrix: DenseLayerMatrix, layer_name: str) -> None:
    """Warn when a captured layer is (near-)entirely NaN.

    A layer whose source columns were found but which holds no values at all points at
    a rule defect; without this it reaches the output as a silently empty matrix.
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
