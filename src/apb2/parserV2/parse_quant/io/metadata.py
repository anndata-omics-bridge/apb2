"""Versioned metadata and physical-name helpers shared by the result adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal, cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    FinalLayerRole,
    JsonValue,
    MeasurementLayerRole,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError

NAMESPACE = "apb"
PARSE_NAMESPACE = "parse"
RESULT_NAMESPACE = "result"
RESULT_FORMAT = "apb2-parsed-levels"
RESULT_FORMAT_VERSION = "1"
MATRIX_PROJECTED_KEY = "matrix_values_projected"

PARQUET_FORMAT = "apb2-parsed-levels-parquet"
PARQUET_FORMAT_VERSION = "2"
PARQUET_MANIFEST_NAME = "manifest.json"
PARQUET_LEVELS_DIRECTORY = "levels"

_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")
_SIMPLE_DTYPES: Mapping[str, pl.DataType | type[pl.DataType]] = {
    str(dtype): dtype
    for dtype in (
        pl.Null,
        pl.Boolean,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.String,
        pl.Binary,
        pl.Date,
        pl.Time,
        pl.Categorical,
    )
}
_LAYER_ROLES_BY_NAME: Mapping[str, FinalLayerRole] = {
    "measurement": MeasurementLayerRole(),
    "auxiliary": AuxiliaryLayerRole(),
}


def layer_role_from_metadata(
    metadata: Mapping[str, object],
    context: str,
    /,
) -> FinalLayerRole:
    """Restore one layer role, defaulting metadata written before roles to measurement."""
    if "role" not in metadata:
        return MeasurementLayerRole()
    value = metadata["role"]
    if not isinstance(value, str):
        raise InvalidResultError(f"{context} role is not text")
    try:
        return _LAYER_ROLES_BY_NAME[value]
    except KeyError as error:
        raise InvalidResultError(f"{context} has unknown role {value!r}") from error


def safe_names(names: Iterable[str], /, *, prefix: str, suffix: str) -> dict[str, str]:
    """Map logical names to unique physical names without trusting them as paths or SQL."""
    taken: set[str] = set()
    result: dict[str, str] = {}
    for index, name in enumerate(names):
        stem = _UNSAFE.sub("_", name).strip("._-") or f"{prefix}_{index}"
        candidate = f"{stem}{suffix}"
        serial = 0
        while candidate in taken:
            serial += 1
            candidate = f"{stem}_{serial}{suffix}"
        taken.add(candidate)
        result[name] = candidate
    return result


def table_metadata(frame: pl.DataFrame, file_name: str, /) -> dict[str, JsonValue]:
    """Record one table's physical name, logical columns, and Polars dtypes."""
    return {
        "file": file_name,
        "columns": list(frame.columns),
        "schema": [_dtype_metadata(frame.schema[name]) for name in frame.columns],
    }


def restore_table_schema(
    frame: pl.DataFrame,
    metadata: Mapping[str, object],
    /,
) -> pl.DataFrame:
    """Restore declared logical dtypes after a backend's physical type translation."""
    columns = _string_list(metadata.get("columns"), "table columns")
    if frame.columns != columns:
        raise InvalidResultError(
            f"stored columns {frame.columns} differ from manifest columns {columns}"
        )
    raw_schema = metadata.get("schema")
    if not isinstance(raw_schema, list) or len(raw_schema) != len(columns):
        raise InvalidResultError("table manifest has no complete logical schema")
    expressions: list[pl.Expr] = []
    for name, raw_dtype in zip(columns, raw_schema, strict=True):
        if not isinstance(raw_dtype, dict):
            raise InvalidResultError(f"the dtype for column {name!r} is not an object")
        expressions.append(pl.col(name).cast(_dtype_from(raw_dtype), strict=True).alias(name))
    return frame.select(expressions)


def _dtype_metadata(dtype: pl.DataType) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {"name": str(dtype)}
    if isinstance(dtype, pl.Enum):
        metadata["enum_categories"] = cast(list[JsonValue], dtype.categories.to_list())
    return metadata


def _dtype_from(metadata: Mapping[str, object]) -> pl.DataType | type[pl.DataType]:
    name = metadata.get("name")
    if not isinstance(name, str):
        raise InvalidResultError("a logical dtype has no text name")
    if name in _SIMPLE_DTYPES:
        return _SIMPLE_DTYPES[name]
    categories = metadata.get("enum_categories")
    if isinstance(categories, list) and all(isinstance(item, str) for item in categories):
        return pl.Enum(cast(list[str], categories))
    match = re.fullmatch(r"Datetime\(time_unit='(ns|us|ms)', time_zone=(None|'[^']*')\)", name)
    if match:
        zone = match.group(2)
        return pl.Datetime(_time_unit(match.group(1)), None if zone == "None" else zone[1:-1])
    match = re.fullmatch(r"Duration\(time_unit='(ns|us|ms)'\)", name)
    if match:
        return pl.Duration(_time_unit(match.group(1)))
    match = re.fullmatch(r"Decimal\(precision=(None|\d+), scale=(\d+)\)", name)
    if match:
        precision = None if match.group(1) == "None" else int(match.group(1))
        return pl.Decimal(precision, int(match.group(2)))
    raise InvalidResultError(f"unsupported logical Polars dtype {name!r}")


def _time_unit(value: str) -> Literal["ns", "us", "ms"]:
    if value == "ns" or value == "us" or value == "ms":
        return value
    raise InvalidResultError(f"unsupported time unit {value!r}")


def object_mapping(value: object, role: str, /) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InvalidResultError(f"{role} is not an object")
    return cast(dict[str, object], value)


def string_value(value: object, role: str, /) -> str:
    if not isinstance(value, str):
        raise InvalidResultError(f"{role} is not text")
    return value


def string_list(value: object, role: str, /) -> list[str]:
    return _string_list(value, role)


def _string_list(value: object, role: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidResultError(f"{role} is not a list of text values")
    return cast(list[str], value)
