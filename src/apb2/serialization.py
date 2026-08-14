"""Concrete recursive JSON values and boundary serialization."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def to_json_compatible(value: object) -> JsonValue:
    """Copy supported NumPy and HDF5 values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"cannot serialize {type(value).__name__} as JSON")
