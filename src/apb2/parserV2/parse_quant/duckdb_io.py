"""Lossless APB2 result reading and writing in one DuckDB database."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import duckdb
import polars as pl

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.errors import InvalidResultError
from apb2.parserV2.parse_quant.result_metadata import (
    object_mapping,
    restore_table_schema,
    string_list,
    string_value,
    table_metadata,
)
from apb2.parserV2.parse_quant.result_validation import validate_parsed_levels

FORMAT = "apb2-parsed-levels-duckdb"
FORMAT_VERSION = "1"
METADATA_TABLE = "apb2_result_metadata"
_REGISTERED_FRAME = "apb2_incoming_frame"
_PHYSICAL_TABLE = re.compile(r"data_[0-9]{6}")


@dataclass(slots=True)
class _TableWriter:
    connection: duckdb.DuckDBPyConnection
    serial: int = 0

    def write(self, frame: pl.DataFrame, /) -> dict[str, JsonValue]:
        name = f"data_{self.serial:06d}"
        self.serial += 1
        self.connection.register(_REGISTERED_FRAME, frame)
        try:
            self.connection.execute(f"CREATE TABLE {name} AS SELECT * FROM {_REGISTERED_FRAME}")
        finally:
            self.connection.unregister(_REGISTERED_FRAME)
        return table_metadata(frame, name)


@dataclass(frozen=True, slots=True)
class DuckDBWriter:
    """Stage a complete database beside the destination, then replace it atomically."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
            staged = Path(scratch) / target.name
            with duckdb.connect(str(staged)) as connection:
                tables = _TableWriter(connection)
                levels = cast(
                    dict[str, JsonValue],
                    {
                        name: self._write_level(level, tables)
                        for name, level in parsed.levels.items()
                    },
                )
                manifest: dict[str, JsonValue] = {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "level_order": list(parsed.levels),
                    "levels": levels,
                    "uns": dict(parsed.uns),
                }
                payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
                connection.execute(
                    f"CREATE TABLE {METADATA_TABLE} (manifest_json VARCHAR NOT NULL)"
                )
                connection.execute(
                    f"INSERT INTO {METADATA_TABLE} VALUES (?)",
                    [payload],
                )
                connection.execute("CHECKPOINT")
            if target.exists() and not target.is_file():
                raise InvalidResultError(f"{target} exists and is not a file")
            staged.replace(target)

    @staticmethod
    def _write_level(parsed: ParsedLevel, tables: _TableWriter) -> dict[str, JsonValue]:
        return {
            "primary_layer": parsed.primary_layer_name,
            "obs": {**tables.write(parsed.obs.frame), "key_columns": list(parsed.obs.key_columns)},
            "var": {**tables.write(parsed.var.frame), "key_columns": list(parsed.var.key_columns)},
            "layer_order": list(parsed.layers),
            "layers": {
                name: {
                    **tables.write(layer.values),
                    "var_key_columns": list(layer.var_key_columns),
                }
                for name, layer in parsed.layers.items()
            },
            "obsm_order": list(parsed.obsm),
            "obsm": {name: tables.write(frame) for name, frame in parsed.obsm.items()},
            "varm_order": list(parsed.varm),
            "varm": {name: tables.write(frame) for name, frame in parsed.varm.items()},
            "obsp_order": list(parsed.obsp),
            "obsp": {name: tables.write(frame) for name, frame in parsed.obsp.items()},
            "varp_order": list(parsed.varp),
            "varp": {name: tables.write(frame) for name, frame in parsed.varp.items()},
            "uns": dict(parsed.uns),
        }


@dataclass(frozen=True, slots=True)
class DuckDBReader:
    """Read one APB2-authored result database and close it before returning."""

    def read(self, source: Path, /) -> ParsedLevels:
        if not source.is_file():
            raise InvalidResultError(f"{source} is not an APB2 DuckDB result file")
        try:
            with duckdb.connect(str(source), read_only=True) as connection:
                rows = connection.execute(f"SELECT manifest_json FROM {METADATA_TABLE}").fetchall()
                if len(rows) != 1 or not isinstance(rows[0][0], str):
                    raise InvalidResultError("DuckDB result has no single metadata record")
                manifest = object_mapping(json.loads(rows[0][0]), "DuckDB manifest")
                result = self._read_manifest(connection, manifest)
        except (duckdb.Error, json.JSONDecodeError) as error:
            raise InvalidResultError(f"cannot read APB2 DuckDB result {source}: {error}") from error
        validate_parsed_levels(result)
        return result

    def _read_manifest(
        self,
        connection: duckdb.DuckDBPyConnection,
        manifest: dict[str, object] | object,
    ) -> ParsedLevels:
        root = object_mapping(manifest, "DuckDB manifest")
        if root.get("format") != FORMAT or root.get("format_version") != FORMAT_VERSION:
            raise InvalidResultError(f"database is not {FORMAT} version {FORMAT_VERSION}")
        order = string_list(root.get("level_order"), "level order")
        entries = object_mapping(root.get("levels"), "levels")
        levels: dict[ParsedLevelName, ParsedLevel] = {}
        for name in order:
            if name not in LEVEL_ORDER:
                raise InvalidResultError(f"unknown quantification level {name!r}")
            levels[name] = self._read_level(
                connection,
                object_mapping(entries.get(name), f"level {name!r}"),
            )
        if set(order) != set(entries):
            raise InvalidResultError("level order and level metadata name different levels")
        return ParsedLevels(
            levels=levels,
            uns=cast(dict[str, JsonValue], dict(object_mapping(root.get("uns"), "shared uns"))),
        )

    def _read_level(
        self,
        connection: duckdb.DuckDBPyConnection,
        metadata: dict[str, object] | object,
    ) -> ParsedLevel:
        level = object_mapping(metadata, "level metadata")
        obs_metadata = object_mapping(level.get("obs"), "obs metadata")
        var_metadata = object_mapping(level.get("var"), "var metadata")
        return ParsedLevel(
            obs=ObsFinal(
                frame=self._read_table(connection, obs_metadata),
                key_columns=tuple(string_list(obs_metadata.get("key_columns"), "obs key columns")),
            ),
            var=VarFinal(
                frame=self._read_table(connection, var_metadata),
                key_columns=tuple(string_list(var_metadata.get("key_columns"), "var key columns")),
            ),
            primary_layer_name=string_value(level.get("primary_layer"), "primary layer"),
            layers=self._read_layers(connection, level),
            obsm=self._read_named(connection, level, "obsm"),
            varm=self._read_named(connection, level, "varm"),
            obsp=self._read_named(connection, level, "obsp"),
            varp=self._read_named(connection, level, "varp"),
            uns=cast(dict[str, JsonValue], dict(object_mapping(level.get("uns"), "level uns"))),
        )

    def _read_layers(
        self,
        connection: duckdb.DuckDBPyConnection,
        level: object,
    ) -> dict[str, FinalLayerTable]:
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
                values=self._read_table(connection, entry),
            )
        if set(order) != set(entries):
            raise InvalidResultError("layer order and layer metadata name different layers")
        return result

    def _read_named(
        self,
        connection: duckdb.DuckDBPyConnection,
        level: object,
        slot: str,
    ) -> dict[str, pl.DataFrame]:
        metadata = object_mapping(level, "level metadata")
        order = string_list(metadata.get(f"{slot}_order"), f"{slot} order")
        entries = object_mapping(metadata.get(slot), slot)
        result = {
            name: self._read_table(
                connection,
                object_mapping(entries.get(name), f"{slot}[{name!r}]"),
            )
            for name in order
        }
        if set(order) != set(entries):
            raise InvalidResultError(f"{slot} order and metadata name different values")
        return result

    @staticmethod
    def _read_table(
        connection: duckdb.DuckDBPyConnection,
        metadata: object,
    ) -> pl.DataFrame:
        table = object_mapping(metadata, "table metadata")
        name = string_value(table.get("file"), "physical table")
        if not _PHYSICAL_TABLE.fullmatch(name):
            raise InvalidResultError(f"unsafe physical DuckDB table name {name!r}")
        return restore_table_schema(connection.table(name).pl(), table)
