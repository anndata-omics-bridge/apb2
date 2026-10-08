"""Versioned metadata and physical-name helpers shared by the result adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal, cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerRole,
    FinalLayerSemantics,
    JsonValue,
    LevelHierarchy,
    MeasurementLayerRole,
    ParsedLevel,
    ParsedLevels,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError

NAMESPACE = "apb"
PARSE_NAMESPACE = "parse"
ROLES_NAMESPACE = "roles"
STORAGE_NAMESPACE = "storage"
RESULT_FORMAT = "apb2-parsed-levels"
RESULT_FORMAT_VERSION = "6"

PARQUET_FORMAT = "apb2-parsed-levels-parquet"
PARQUET_FORMAT_VERSION = "7"
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


def shared_scope(
    parse: Mapping[str, JsonValue],
    metadata: Mapping[str, JsonValue],
    hierarchy: LevelHierarchy | None = None,
    /,
) -> dict[str, JsonValue]:
    """Compose one shared APB scope without merging it into a level."""
    if {PARSE_NAMESPACE, ROLES_NAMESPACE, STORAGE_NAMESPACE, "hierarchy"}.intersection(metadata):
        raise InvalidResultError("root extension metadata uses a reserved APB section")
    result: dict[str, JsonValue] = {PARSE_NAMESPACE: dict(parse), **dict(metadata)}
    if hierarchy is not None:
        result["hierarchy"] = hierarchy.as_json()
    return result


def level_scope(parsed: ParsedLevel, /) -> dict[str, JsonValue]:
    """Compose one level scope, nesting semantic roles beside parse evidence."""
    parse = dict(parsed.uns)
    roles: dict[str, JsonValue] = {}
    if parsed.var.roles:
        roles["columns"] = dict(parsed.var.roles)
    layers: dict[str, JsonValue] = {}
    for name, layer in parsed.layers.items():
        for role in layer.semantic_roles:
            members = layers.setdefault(role, [])
            assert isinstance(members, list)
            members.append(name)
    if layers:
        roles["layers"] = layers
    collisions = {PARSE_NAMESPACE, ROLES_NAMESPACE, STORAGE_NAMESPACE, "hierarchy"}.intersection(
        parsed.metadata
    )
    if collisions:
        raise InvalidResultError(f"level extension metadata uses reserved section(s) {collisions}")
    result: dict[str, JsonValue] = {PARSE_NAMESPACE: parse}
    if roles:
        result[ROLES_NAMESPACE] = roles
    result.update(parsed.metadata)
    return result


def collection_shared_scope(parsed: ParsedLevels, /) -> dict[str, JsonValue]:
    """Compose root scientific metadata, including non-level objects exactly once."""
    result = shared_scope(parsed.uns, parsed.metadata, parsed.hierarchy)
    reserved = {"annotation_tables", "feature_relations"}.intersection(result)
    if reserved:
        raise InvalidResultError(f"shared metadata uses reserved section(s) {reserved}")
    if parsed.annotation_tables:
        result["annotation_tables"] = cast(
            JsonValue,
            {
                name: {
                    "key_columns": list(table.key_columns),
                    "metadata": dict(table.metadata),
                }
                for name, table in parsed.annotation_tables.items()
            },
        )
    if parsed.feature_relations:
        result["feature_relations"] = cast(
            JsonValue,
            {
                name: {
                    "annotation_table": relation.annotation_table,
                    "target_level": relation.target_level,
                    "metadata": dict(relation.metadata),
                }
                for name, relation in parsed.feature_relations.items()
            },
        )
    return result


def unpack_shared_scope(
    value: object, role: str, /
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Split one persisted shared scope into the storage-neutral result fields."""
    scope = object_mapping(value, role)
    parse = cast(
        dict[str, JsonValue],
        dict(object_mapping(scope.get(PARSE_NAMESPACE, {}), f"{role} parse section")),
    )
    metadata = cast(
        dict[str, JsonValue],
        {name: item for name, item in scope.items() if name != PARSE_NAMESPACE},
    )
    return parse, metadata


def unpack_level_scope(
    value: object, role: str, /
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Split one persisted level scope and restore its semantic role maps."""
    scope = object_mapping(value, role)
    parse = cast(
        dict[str, JsonValue],
        dict(object_mapping(scope.get(PARSE_NAMESPACE, {}), f"{role} parse section")),
    )
    roles = object_mapping(scope.get(ROLES_NAMESPACE, {}), f"{role} roles section")
    if "columns" in roles:
        parse["column_roles"] = cast(JsonValue, roles["columns"])
    if "layers" in roles:
        parse["layer_roles"] = cast(JsonValue, roles["layers"])
    metadata = cast(
        dict[str, JsonValue],
        {
            name: item
            for name, item in scope.items()
            if name not in {PARSE_NAMESPACE, ROLES_NAMESPACE}
        },
    )
    return parse, metadata


def layer_role_from_metadata(
    metadata: Mapping[str, object],
    context: str,
    /,
) -> FinalLayerRole:
    """Restore one layer role from a complete current-format descriptor."""
    if "role" not in metadata:
        raise InvalidResultError(f"{context} has no role")
    value = metadata["role"]
    if not isinstance(value, str):
        raise InvalidResultError(f"{context} role is not text")
    try:
        return _LAYER_ROLES_BY_NAME[value]
    except KeyError as error:
        raise InvalidResultError(f"{context} has unknown role {value!r}") from error


def layer_semantics_metadata(semantics: FinalLayerSemantics, /) -> dict[str, JsonValue]:
    """Project canonical layer semantics into backend metadata."""
    if isinstance(semantics, QuantitativeLayerSemantics):
        return {"kind": "quantitative", "logical_type": semantics.logical_type}
    return {
        "kind": "categorical",
        "categories": cast(list[JsonValue], [list(item) for item in semantics.categories]),
        "missing_code": semantics.missing_code,
    }


def layer_semantics_from_metadata(
    value: object,
    context: str,
    /,
) -> FinalLayerSemantics:
    """Restore canonical layer semantics from one backend descriptor."""
    metadata = object_mapping(value, f"{context} semantics")
    kind = string_value(metadata.get("kind"), f"{context} semantics kind")
    if kind == "quantitative":
        logical_type = string_value(metadata.get("logical_type"), f"{context} quantitative type")
        if logical_type not in {"number", "integer"}:
            raise InvalidResultError(f"{context} has unknown quantitative type {logical_type!r}")
        return QuantitativeLayerSemantics(
            logical_type=cast(Literal["number", "integer"], logical_type)
        )
    if kind != "categorical":
        raise InvalidResultError(f"{context} has unknown semantics kind {kind!r}")
    raw_categories = metadata.get("categories")
    if not isinstance(raw_categories, list):
        raise InvalidResultError(f"{context} categorical semantics has no categories list")
    categories: list[tuple[str, int]] = []
    for item in raw_categories:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
        ):
            raise InvalidResultError(f"{context} has an invalid categorical entry")
        categories.append((item[0], item[1]))
    missing_code = metadata.get("missing_code")
    if not isinstance(missing_code, int) or isinstance(missing_code, bool):
        raise InvalidResultError(f"{context} categorical missing code is not an integer")
    return CategoricalLayerSemantics(
        categories=tuple(categories),
        missing_code=missing_code,
    )


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
    return {"file": file_name, **logical_table_metadata(frame)}


def logical_table_metadata(frame: pl.DataFrame, /) -> dict[str, JsonValue]:
    """Record the logical columns and dtypes needed for an exact table round-trip."""
    return {
        "columns": list(frame.columns),
        "schema": [_dtype_metadata(frame.schema[name]) for name in frame.columns],
    }


def column_descriptions(frame: pl.DataFrame, /) -> list[dict[str, JsonValue]]:
    """Describe logical columns without exposing table values or physical storage names."""
    return (
        frame.null_count()
        .unpivot(variable_name="name", value_name="null_count")
        .with_columns(
            pl.Series(
                "dtype", [_public_dtype_name(dtype) for dtype in frame.dtypes], dtype=pl.String
            )
        )
        .to_dicts()
    )


def _public_dtype_name(dtype: pl.DataType, /) -> str:
    """Name a dtype without embedding an Enum's complete category vocabulary."""
    description = str(dtype)
    return str(dtype.base_type()) if "Enum(categories=" in description else description


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


def restore_level_roles(parsed: ParsedLevel) -> ParsedLevel:
    """Hydrate semantic declarations at the physical read boundary."""
    columns = object_mapping(parsed.uns.pop("column_roles", {}), "column roles")
    parsed.var.roles = {
        role: string_value(column, f"column role {role!r}") for role, column in columns.items()
    }
    layers = object_mapping(parsed.uns.pop("layer_roles", {}), "layer roles")
    for role, raw_names in layers.items():
        names = string_list(raw_names, f"layer role {role!r}")
        if len(names) != len(set(names)) or set(names) - set(parsed.layers):
            raise InvalidResultError(f"layer role {role!r} names duplicate or absent layers")
        for name in names:
            parsed.layers[name].semantic_roles += (role,)
    return parsed


def read_hierarchy(metadata: dict[str, JsonValue]) -> LevelHierarchy | None:
    """Restore the typed hierarchy from an adapter-owned shared metadata mapping."""
    value = metadata.pop("hierarchy", None)
    return None if value is None else LevelHierarchy.from_json(value)
