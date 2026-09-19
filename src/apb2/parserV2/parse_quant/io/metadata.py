"""Versioned metadata and physical-name helpers shared by the result adapters."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Literal, cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerRole,
    FinalLayerSemantics,
    JsonValue,
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
RESULT_FORMAT_VERSION = "4"

PARQUET_FORMAT = "apb2-parsed-levels-parquet"
PARQUET_FORMAT_VERSION = "5"
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
    parse: Mapping[str, JsonValue], metadata: Mapping[str, JsonValue], /
) -> dict[str, JsonValue]:
    """Compose one shared APB scope without merging it into a level."""
    if {PARSE_NAMESPACE, ROLES_NAMESPACE, STORAGE_NAMESPACE}.intersection(metadata):
        raise InvalidResultError("root extension metadata uses a reserved APB section")
    return {PARSE_NAMESPACE: dict(parse), **dict(metadata)}


def level_scope(parsed: ParsedLevel, /) -> dict[str, JsonValue]:
    """Compose one level scope, nesting semantic roles beside parse evidence."""
    parse = dict(parsed.uns)
    if isinstance(parse.get("rule_json"), str):
        for repeated in ("schema_version", "software_name", "shape", "quantification_level"):
            parse.pop(repeated, None)
    roles: dict[str, JsonValue] = {}
    columns = parse.pop("column_roles", None)
    layers = parse.pop("layer_roles", None)
    if columns is not None:
        roles["columns"] = columns
    if layers is not None:
        roles["layers"] = layers
    collisions = {PARSE_NAMESPACE, ROLES_NAMESPACE, STORAGE_NAMESPACE}.intersection(parsed.metadata)
    if collisions:
        raise InvalidResultError(f"level extension metadata uses reserved section(s) {collisions}")
    result: dict[str, JsonValue] = {PARSE_NAMESPACE: parse}
    if roles:
        result[ROLES_NAMESPACE] = roles
    result.update(parsed.metadata)
    return result


def collection_shared_scope(parsed: ParsedLevels, /) -> dict[str, JsonValue]:
    """Compose root scientific metadata, including non-level objects exactly once."""
    result = shared_scope(parsed.uns, parsed.metadata)
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


def compose_metadata(
    root: Mapping[str, JsonValue], level: Mapping[str, JsonValue], /
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Combine disjoint metadata and retain only ownership paths for reconstruction."""
    return _merge_metadata(root, level, ()), {
        "root": _metadata_paths(root),
        "level": _metadata_paths(level),
    }


def _merge_metadata(
    root: Mapping[str, JsonValue], level: Mapping[str, JsonValue], path: tuple[str, ...]
) -> dict[str, JsonValue]:
    result = deepcopy(dict(root))
    for key, value in level.items():
        if key not in result:
            result[key] = deepcopy(value)
            continue
        previous = result[key]
        if isinstance(previous, dict) and isinstance(value, dict):
            result[key] = _merge_metadata(previous, value, (*path, key))
            continue
        raise InvalidResultError(f"conflicting APB metadata at {(*path, key)!r}")
    return result


def _metadata_paths(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    values: list[JsonValue] = []
    empty_objects: list[JsonValue] = []

    def visit(mapping: Mapping[str, JsonValue], path: tuple[str, ...]) -> None:
        if not mapping:
            empty_objects.append(list(path))
        for key, item in mapping.items():
            if isinstance(item, dict):
                visit(item, (*path, key))
            else:
                values.append([*path, key])

    visit(value, ())
    values.sort(key=lambda path: tuple(cast(list[str], path)))
    empty_objects.sort(key=lambda path: tuple(cast(list[str], path)))
    return {"values": values, "empty_objects": empty_objects}


def split_metadata(
    namespace: Mapping[str, JsonValue], ownership: object, /
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Restore root and level contributions, rejecting incomplete ownership records."""
    owners = object_mapping(ownership, "APB metadata ownership")
    if set(owners) != {"root", "level"}:
        raise InvalidResultError("APB metadata ownership must declare root and level")
    root = _owned_metadata(namespace, owners["root"])
    level = _owned_metadata(namespace, owners["level"])
    merged, paths = compose_metadata(root, level)
    if merged != namespace or paths != owners:
        raise InvalidResultError("APB metadata ownership does not describe the namespace exactly")
    return root, level


def _owned_metadata(namespace: Mapping[str, JsonValue], descriptor: object) -> dict[str, JsonValue]:
    record = object_mapping(descriptor, "metadata paths")
    if set(record) != {"values", "empty_objects"}:
        raise InvalidResultError("metadata paths must declare values and empty_objects")
    result: dict[str, JsonValue] = {}
    for kind in ("values", "empty_objects"):
        entries = record[kind]
        if not isinstance(entries, list):
            raise InvalidResultError("metadata ownership paths must be lists")
        for entry in cast(list[object], entries):
            path = string_list(entry, "metadata ownership path")
            value = _value_at_path(namespace, path)
            if kind == "empty_objects":
                if not isinstance(value, dict):
                    raise InvalidResultError(f"metadata object path is not an object: {path!r}")
                value = {}
            elif isinstance(value, dict) or not path:
                raise InvalidResultError(f"metadata value path is not a leaf: {path!r}")
            _insert_owned_value(result, path, value)
    return result


def _value_at_path(namespace: Mapping[str, JsonValue], path: list[str]) -> JsonValue:
    value: JsonValue = dict(namespace)
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise InvalidResultError(f"metadata ownership names absent path {path!r}")
        value = value[key]
    return value


def _insert_owned_value(target: dict[str, JsonValue], path: list[str], value: JsonValue) -> None:
    if not path:
        return
    for key in path[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise InvalidResultError(f"overlapping metadata ownership paths: {path!r}")
        target = child
    if path[-1] in target:
        raise InvalidResultError(f"duplicate metadata ownership path: {path!r}")
    target[path[-1]] = deepcopy(value)


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
    return [
        {
            "name": name,
            "dtype": _public_dtype_name(frame.schema[name]),
            "null_count": frame.get_column(name).null_count(),
        }
        for name in frame.columns
    ]


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
