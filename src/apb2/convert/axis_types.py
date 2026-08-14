"""Strict logical typing for selected observation and feature columns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from apb2.vendor_parse_rules.model import AxisColumnType

type AxisName = Literal["obs", "var"]


@dataclass(frozen=True, slots=True)
class AxisColumnContext:
    """Identity and logical contract for one selected axis column."""

    axis: AxisName
    output_name: str
    source_name: str
    logical_type: AxisColumnType


def _raise_invalid(
    values: pd.Series,
    invalid: pd.Series,
    context: AxisColumnContext,
) -> None:
    """Raise one bounded, contextual error when a coercion mask contains failures."""
    count = int(invalid.sum())
    if not count:
        return
    examples = values.loc[invalid].astype("string").drop_duplicates().head(5).tolist()
    raise ValueError(
        f"cannot convert {context.axis} column {context.output_name!r} from vendor "
        f"source {context.source_name!r} to logical type {context.logical_type!r}: "
        f"{count} invalid non-missing value(s); examples={examples}"
    )


def coerce_string(values: pd.Series, context: AxisColumnContext) -> pd.Series:
    """Return nullable strings without changing identifier text."""
    del context
    return values.astype("string")


def coerce_number(values: pd.Series, context: AxisColumnContext) -> pd.Series:
    """Return float64 values and reject every invalid non-missing token."""
    parsed = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = pd.Series(np.isfinite(parsed.to_numpy()), index=values.index)
    invalid = values.notna() & (parsed.isna() | ~finite)
    _raise_invalid(values, invalid, context)
    return parsed


def coerce_integer(values: pd.Series, context: AxisColumnContext) -> pd.Series:
    """Return nullable Int64 values after exact integrality and range validation."""
    parsed = pd.to_numeric(values, errors="coerce")
    present = parsed.notna()
    invalid = values.notna() & ~present
    invalid |= (present & parsed.mod(1).ne(0)).fillna(False)
    limits = np.iinfo(np.int64)
    invalid |= (present & (parsed.lt(limits.min) | parsed.gt(limits.max))).fillna(False)
    _raise_invalid(values, invalid, context)
    return parsed.astype("Int64")


def coerce_boolean(values: pd.Series, context: AxisColumnContext) -> pd.Series:
    """Return nullable booleans from the exact canonical boolean spellings."""
    normalized = values.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {
            "false": False,
            "true": True,
            "0": False,
            "0.0": False,
            "1": True,
            "1.0": True,
        }
    )
    invalid = values.notna() & parsed.isna()
    _raise_invalid(values, invalid, context)
    return parsed.astype("boolean")


type AxisColumnCoercer = Callable[[pd.Series, AxisColumnContext], pd.Series]

_COERCERS: Mapping[AxisColumnType, AxisColumnCoercer] = {
    "string": coerce_string,
    "integer": coerce_integer,
    "number": coerce_number,
    "boolean": coerce_boolean,
}


def coerce_axis_column(values: pd.Series, context: AxisColumnContext) -> pd.Series:
    """Apply the selected column's exact logical coercer."""
    return _COERCERS[context.logical_type](values, context)
