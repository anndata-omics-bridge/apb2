"""Read an APB2 version-2 Parquet result dataset into its Polars value."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow.parquet as pq

from apb2.parserV2.parse_quant.data.layer_blocks import READ_BLOCK_BYTES, LayerBlocks
from apb2.parserV2.parse_quant.data.parking import LayerParking
from apb2.parserV2.parse_quant.data.parsed import (
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
    json_object,
    layer_role_from_metadata,
    layer_semantics_from_metadata,
    object_mapping,
    read_hierarchy,
    relations_among,
    restore_level_roles,
    restore_table_schema,
    string_list,
    string_value,
    unpack_level_scope,
    unpack_shared_scope,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels

FORMAT = PARQUET_FORMAT
FORMAT_VERSION = PARQUET_FORMAT_VERSION
MANIFEST_NAME = PARQUET_MANIFEST_NAME
LEVELS_DIRECTORY = PARQUET_LEVELS_DIRECTORY


class ParquetReader:
    """Read only APB2 result datasets carrying the declared manifest and version."""

    __slots__ = ()

    def read(self, source: Path, level: ParsedLevelName | None = None, /) -> ParsedLevels:
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
        if level is not None and level not in order:
            raise InvalidResultError(
                f"Parquet result {source} has no level {level!r}; it has {order}"
            )
        levels: dict[ParsedLevelName, ParsedLevel] = {}
        for name in order:
            if not name:
                raise InvalidResultError(f"unknown quantification level {name!r}")
            if level is not None and name != level:
                continue
            levels[name] = _read_level(
                source, object_mapping(level_entries.get(name), f"level {name!r}")
            )
            LayerParking().park(levels[name])
        if set(order) != set(level_entries):
            raise InvalidResultError("level order and level metadata name different levels")
        uns, metadata = unpack_shared_scope(manifest.get("apb"), "root APB metadata")
        annotation_tables = _read_annotation_tables(source, manifest)
        feature_relations = _read_feature_relations(source, manifest)
        parsed = ParsedLevels(
            hierarchy=read_hierarchy(metadata),
            levels=levels,
            uns=uns,
            metadata=metadata,
            annotation_tables=annotation_tables,
            feature_relations=relations_among(feature_relations, {*levels, *annotation_tables}),
            export_uns=json_object(manifest.get("export_uns"), "root export_uns"),
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
    uns, extension_metadata = unpack_level_scope(level.get("apb"), "level APB metadata")
    return restore_level_roles(
        ParsedLevel(
            obs=obs,
            var=var,
            primary_layer_name=string_value(level.get("primary_layer"), "primary layer"),
            layers=layers,
            obsm=_read_named_frames(directory / "obsm", level, "obsm"),
            varm=_read_named_frames(directory / "varm", level, "varm"),
            obsp=_read_named_frames(directory / "obsp", level, "obsp"),
            varp=_read_named_frames(directory / "varp", level, "varp"),
            uns=uns,
            metadata=extension_metadata,
            export_uns=json_object(level.get("export_uns"), "level export_uns"),
        )
    )


def _read_layers(directory: Path, level: object) -> dict[str, FinalLayerTable]:
    metadata = object_mapping(level, "level metadata")
    order = string_list(metadata.get("layer_order"), "layer order")
    entries = object_mapping(metadata.get("layers"), "layers")
    result: dict[str, FinalLayerTable] = {}
    for name in order:
        entry = object_mapping(entries.get(name), f"layer {name!r}")
        result[name] = LayerBlocks.stored(
            name,
            _read_batches(directory / "layers", entry),
            labels=tuple(string_list(entry.get("columns"), f"layer {name!r} columns")),
            role=layer_role_from_metadata(entry, f"layer {name!r}"),
            semantics=layer_semantics_from_metadata(entry.get("semantics"), f"layer {name!r}"),
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
        result[name] = FeatureRelation(
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


def _read_batches(directory: Path, metadata: object) -> Iterator[pl.DataFrame]:
    """One layer table a batch of rows at a time, each with its recorded schema."""
    table = object_mapping(metadata, "table metadata")
    path = directory / _physical_name(table.get("file"), "table file")
    columns = string_list(table.get("columns"), "table columns")
    try:
        file = pq.ParquetFile(path)
    except OSError as error:
        raise InvalidResultError(f"cannot read result table {path}: {error}") from error
    rows = max(READ_BLOCK_BYTES // max(len(columns) * 8, 1), 1)
    for batch in file.iter_batches(batch_size=rows):
        yield restore_table_schema(cast(pl.DataFrame, pl.from_arrow(batch)), table)


def _physical_name(value: object, role: str) -> str:
    name = string_value(value, role)
    if Path(name).name != name or name in {"", ".", ".."}:
        raise InvalidResultError(f"unsafe {role}: {name!r}")
    return name
