"""Read an APB2 version-2 Parquet result dataset into its Polars value."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    AnnotationTable,
    FeatureRelation,
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.metadata import (
    PARQUET_FORMAT,
    PARQUET_FORMAT_VERSION,
    PARQUET_LEVELS_DIRECTORY,
    PARQUET_MANIFEST_NAME,
    layer_role_from_metadata,
    object_mapping,
    restore_table_schema,
    string_list,
    string_value,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels

FORMAT = PARQUET_FORMAT
FORMAT_VERSION = PARQUET_FORMAT_VERSION
MANIFEST_NAME = PARQUET_MANIFEST_NAME
LEVELS_DIRECTORY = PARQUET_LEVELS_DIRECTORY


@dataclass(frozen=True, slots=True)
class ParquetReader:
    """Read only APB2 result datasets carrying the declared manifest and version."""

    def read(self, source: Path, /) -> ParsedLevels:
        if not source.is_dir():
            raise InvalidResultError(f"{source} is not an APB2 Parquet result directory")
        manifest_path = source / MANIFEST_NAME
        try:
            manifest = object_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")), "Parquet manifest"
            )
        except (OSError, json.JSONDecodeError) as error:
            raise InvalidResultError(
                f"cannot read APB2 Parquet manifest {manifest_path}: {error}"
            ) from error
        if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
            raise InvalidResultError(f"{source} is not {FORMAT} version {FORMAT_VERSION}")
        order = string_list(manifest.get("level_order"), "level order")
        level_entries = object_mapping(manifest.get("levels"), "levels")
        levels: dict[ParsedLevelName, ParsedLevel] = {}
        for name in order:
            if name not in LEVEL_ORDER:
                raise InvalidResultError(f"unknown quantification level {name!r}")
            level = _read_level(source, object_mapping(level_entries.get(name), f"level {name!r}"))
            levels[name] = level
        if set(order) != set(level_entries):
            raise InvalidResultError("level order and level metadata name different levels")
        uns = cast(dict[str, JsonValue], dict(object_mapping(manifest.get("uns"), "shared uns")))
        metadata = cast(
            dict[str, JsonValue],
            dict(object_mapping(manifest.get("metadata", {}), "shared metadata")),
        )
        annotation_tables = _read_annotation_tables(source, manifest)
        feature_relations = _read_feature_relations(source, manifest)
        parsed = ParsedLevels(
            levels=levels,
            uns=uns,
            metadata=metadata,
            annotation_tables=annotation_tables,
            feature_relations=feature_relations,
        )
        validate_parsed_levels(parsed)
        return parsed


def _read_level(source: Path, metadata: dict[str, object] | object) -> ParsedLevel:
    level = object_mapping(metadata, "level metadata")
    physical = _physical_name(level.get("directory"), "level directory")
    directory = source / LEVELS_DIRECTORY / physical
    obs_metadata = object_mapping(level.get("obs"), "obs metadata")
    var_metadata = object_mapping(level.get("var"), "var metadata")
    obs = ObsFinal(
        frame=_read_table(directory, obs_metadata),
        key_columns=tuple(string_list(obs_metadata.get("key_columns"), "obs key columns")),
    )
    var = VarFinal(
        frame=_read_table(directory, var_metadata),
        key_columns=tuple(string_list(var_metadata.get("key_columns"), "var key columns")),
    )
    layers = _read_layers(directory, level)
    return ParsedLevel(
        obs=obs,
        var=var,
        primary_layer_name=string_value(level.get("primary_layer"), "primary layer"),
        layers=layers,
        obsm=_read_named_frames(directory / "obsm", level, "obsm"),
        varm=_read_named_frames(directory / "varm", level, "varm"),
        obsp=_read_named_frames(directory / "obsp", level, "obsp"),
        varp=_read_named_frames(directory / "varp", level, "varp"),
        uns=cast(dict[str, JsonValue], dict(object_mapping(level.get("uns"), "level uns"))),
        metadata=cast(
            dict[str, JsonValue],
            dict(object_mapping(level.get("metadata", {}), "level metadata sections")),
        ),
    )


def _read_layers(directory: Path, level: object) -> dict[str, FinalLayerTable]:
    metadata = object_mapping(level, "level metadata")
    order = string_list(metadata.get("layer_order"), "layer order")
    entries = object_mapping(metadata.get("layers"), "layers")
    result: dict[str, FinalLayerTable] = {}
    for name in order:
        entry = object_mapping(entries.get(name), f"layer {name!r}")
        result[name] = FinalLayerTable(
            layer_name=name,
            var_key_columns=tuple(
                string_list(entry.get("var_key_columns"), f"layer {name!r} var keys")
            ),
            values=_read_table(directory / "layers", entry),
            role=layer_role_from_metadata(entry, f"layer {name!r}"),
        )
    if set(order) != set(entries):
        raise InvalidResultError("layer order and layer metadata name different layers")
    return result


def _read_named_frames(
    directory: Path,
    level: object,
    slot: str,
) -> dict[str, pl.DataFrame]:
    metadata = object_mapping(level, "level metadata")
    order = string_list(metadata.get(f"{slot}_order"), f"{slot} order")
    entries = object_mapping(metadata.get(slot), slot)
    result = {
        name: _read_table(directory, object_mapping(entries.get(name), f"{slot}[{name!r}]"))
        for name in order
    }
    if set(order) != set(entries):
        raise InvalidResultError(f"{slot} order and metadata name different values")
    return result


def _read_annotation_tables(
    source: Path,
    manifest: Mapping[str, object],
) -> dict[str, AnnotationTable]:
    order = string_list(manifest.get("annotation_table_order", []), "annotation table order")
    entries = object_mapping(manifest.get("annotation_tables", {}), "annotation tables")
    result: dict[str, AnnotationTable] = {}
    for name in order:
        entry = object_mapping(entries.get(name), f"annotation table {name!r}")
        result[name] = AnnotationTable(
            frame=_read_table(source / "annotation_tables", entry),
            key_columns=tuple(
                string_list(entry.get("key_columns"), f"annotation table {name!r} keys")
            ),
            metadata=cast(
                dict[str, JsonValue],
                dict(object_mapping(entry.get("metadata", {}), "annotation table metadata")),
            ),
        )
    if set(order) != set(entries):
        raise InvalidResultError("annotation table order and metadata name different tables")
    return result


def _read_feature_relations(
    source: Path,
    manifest: Mapping[str, object],
) -> dict[str, FeatureRelation]:
    order = string_list(manifest.get("feature_relation_order", []), "feature relation order")
    entries = object_mapping(manifest.get("feature_relations", {}), "feature relations")
    result: dict[str, FeatureRelation] = {}
    for name in order:
        entry = object_mapping(entries.get(name), f"feature relation {name!r}")
        target_level = string_value(
            entry.get("target_level"), f"feature relation {name!r} target level"
        )
        if target_level not in LEVEL_ORDER:
            raise InvalidResultError(
                f"feature relation {name!r} has unknown target level {target_level!r}"
            )
        result[name] = FeatureRelation(
            annotation_table=string_value(
                entry.get("annotation_table"),
                f"feature relation {name!r} annotation table",
            ),
            target_level=target_level,
            coordinates=_read_table(source / "feature_relations", entry),
            metadata=cast(
                dict[str, JsonValue],
                dict(object_mapping(entry.get("metadata", {}), "feature relation metadata")),
            ),
        )
    if set(order) != set(entries):
        raise InvalidResultError("feature relation order and metadata name different relations")
    return result


def _read_table(directory: Path, metadata: object) -> pl.DataFrame:
    table = object_mapping(metadata, "table metadata")
    file_name = _physical_name(table.get("file"), "table file")
    try:
        frame = pl.read_parquet(directory / file_name)
    except OSError as error:
        raise InvalidResultError(
            f"cannot read result table {directory / file_name}: {error}"
        ) from error
    return restore_table_schema(frame, table)


def _physical_name(value: object, role: str) -> str:
    name = string_value(value, role)
    if Path(name).name != name or name in {"", ".", ".."}:
        raise InvalidResultError(f"unsafe {role}: {name!r}")
    return name
