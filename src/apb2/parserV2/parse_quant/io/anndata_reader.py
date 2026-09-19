"""Read APB2-authored h5ad and h5mu results into storage-neutral Polars values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import anndata
import mudata
import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from scipy import sparse

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    AnnotationTable,
    FeatureRelation,
    FinalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
    QuantitativeLayerSemantics,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.metadata import (
    NAMESPACE,
    PARSE_NAMESPACE,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    ROLES_NAMESPACE,
    STORAGE_NAMESPACE,
    layer_role_from_metadata,
    layer_semantics_from_metadata,
    object_mapping,
    restore_table_schema,
    split_metadata,
    string_list,
    string_value,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels


class H5adReader:
    """Read one APB2 h5ad envelope as a one-level result collection."""

    __slots__ = ()

    def read(self, source: Path, /) -> ParsedLevels:
        try:
            stored = anndata.read_h5ad(source)
        except (OSError, ValueError) as error:
            raise InvalidResultError(f"cannot read h5ad result {source}: {error}") from error
        metadata = _result_metadata(stored)
        root_scope, level_scope = split_metadata(
            _scientific_namespace(stored), metadata.get("metadata_ownership")
        )
        level_name, level = _read_level(stored, metadata, level_scope)
        shared_uns, shared_metadata = _shared_scope(root_scope)
        parsed = ParsedLevels(
            levels={level_name: level},
            uns=shared_uns,
            metadata=shared_metadata,
        )
        validate_parsed_levels(parsed)
        return parsed


class H5muReader:
    """Read one APB2 h5mu envelope and all modalities in declared level order."""

    __slots__ = ()

    def read(self, source: Path, /) -> ParsedLevels:
        try:
            stored = mudata.read_h5mu(source)
        except (OSError, ValueError) as error:
            raise InvalidResultError(f"cannot read h5mu result {source}: {error}") from error
        metadata = _result_metadata(stored)
        level_entries = _ordered_entries(metadata.get("levels"), "h5mu levels")
        levels: dict[ParsedLevelName, ParsedLevel] = {}
        level_names: dict[str, str] = {}
        for entry in level_entries:
            name = string_value(entry.get("name"), "h5mu level name")
            physical_name = string_value(entry.get("physical_name"), "h5mu level physical name")
            if name not in LEVEL_ORDER or physical_name not in stored.mod:
                raise InvalidResultError(f"h5mu declares unavailable level {name!r}")
            modality = cast(AnnData, stored[physical_name])
            level_name, level = _read_level(
                modality, _result_metadata(modality), _scientific_namespace(modality)
            )
            if level_name != name:
                raise InvalidResultError(
                    f"h5mu modality {name!r} contains metadata for {level_name!r}"
                )
            levels[level_name] = level
            level_names[name] = physical_name
        shared_uns, shared_metadata = _shared_scope(_scientific_namespace(stored))
        annotation_tables, annotation_names = _annotation_tables(stored, metadata, shared_metadata)
        expected_modalities = set(level_names.values()).union(annotation_names.values())
        if expected_modalities != set(stored.mod):
            raise InvalidResultError("h5mu level order and modalities name different levels")
        parsed = ParsedLevels(
            levels=levels,
            uns=shared_uns,
            metadata=_shared_extensions(shared_metadata),
            annotation_tables=annotation_tables,
            feature_relations=_feature_relations(
                stored, metadata, shared_metadata, annotation_names, level_names
            ),
        )
        validate_parsed_levels(parsed)
        return parsed


def _read_level(
    stored: AnnData, metadata: Mapping[str, object], scope: dict[str, JsonValue]
) -> tuple[ParsedLevelName, ParsedLevel]:
    level_name = string_value(metadata.get("level"), "quantification level")
    if level_name not in LEVEL_ORDER:
        raise InvalidResultError(f"unknown quantification level {level_name!r}")
    obs_frame = restore_table_schema(
        _axis_frame(cast(pd.DataFrame, stored.obs)),
        object_mapping(metadata.get("obs"), "obs storage metadata"),
    )
    var_frame = restore_table_schema(
        _axis_frame(cast(pd.DataFrame, stored.var)),
        object_mapping(metadata.get("var"), "var storage metadata"),
    )
    obs = ObsFinal(
        frame=obs_frame,
        key_columns=tuple(string_list(metadata.get("obs_key_columns"), "obs key columns")),
    )
    var = VarFinal(
        frame=var_frame,
        key_columns=tuple(string_list(metadata.get("var_key_columns"), "var key columns")),
    )
    layers = _layers(stored, var.frame, metadata)
    uns, level_metadata = _level_scope(scope)
    primary = _primary_layer_name(metadata)
    return level_name, ParsedLevel(
        obs=obs,
        var=var,
        primary_layer_name=primary,
        layers=layers,
        obsm=_aligned_frames(stored.obsm, metadata, "obsm"),
        varm=_aligned_frames(stored.varm, metadata, "varm"),
        obsp=_pairwise_frames(stored.obsp, metadata, "obsp"),
        varp=_pairwise_frames(stored.varp, metadata, "varp"),
        uns=uns,
        metadata=level_metadata,
    )


def _layers(
    stored: AnnData,
    var: pl.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, FinalLayerTable]:
    entries = _ordered_entries(metadata.get("layers"), "layer metadata")
    keys = tuple(string_list(metadata.get("var_key_columns"), "var key columns"))
    result: dict[str, FinalLayerTable] = {}
    for entry in entries:
        name = string_value(entry.get("name"), "logical layer name")
        location = string_value(entry.get("location"), f"layer {name!r} location")
        value_columns = string_list(entry.get("value_columns"), f"layer {name!r} value columns")
        if location == "X":
            matrix = _dense(stored.X)
        elif location == "layers":
            physical_name = string_value(
                entry.get("physical_name"), f"physical layer name for {name!r}"
            )
            if physical_name not in stored.layers:
                raise InvalidResultError(f"h5 result has no declared layer {name!r}")
            matrix = _dense(stored.layers[physical_name])
        else:
            raise InvalidResultError(f"layer {name!r} has unknown location {location!r}")
        if matrix.shape != (stored.n_obs, stored.n_vars):
            raise InvalidResultError(
                f"layer {name!r} has shape {matrix.shape}; expected {(stored.n_obs, stored.n_vars)}"
            )
        semantics = layer_semantics_from_metadata(entry.get("semantics"), f"layer {name!r}")
        values = _canonical_layer_values(
            pl.DataFrame(matrix.T, schema=value_columns, orient="row"),
            semantics,
        )
        result[name] = FinalLayerTable(
            layer_name=name,
            var_key_columns=keys,
            values=pl.concat([var.select(keys), values], how="horizontal_extend"),
            role=layer_role_from_metadata(entry, f"layer {name!r}"),
            semantics=semantics,
        )
    return result


def _canonical_layer_values(
    values: pl.DataFrame,
    semantics: FinalLayerSemantics,
    /,
) -> pl.DataFrame:
    """Restore the canonical scalar representation from an HDF5 matrix."""
    if isinstance(semantics, QuantitativeLayerSemantics):
        if semantics.logical_type == "number":
            return values
        return values.select(
            [_integer_column(values, name, missing=None).alias(name) for name in values.columns]
        )
    return values.select(
        [
            _integer_column(values, name, missing=semantics.missing_code).alias(name)
            for name in values.columns
        ]
    )


def _integer_column(
    values: pl.DataFrame,
    name: str,
    /,
    *,
    missing: int | None,
) -> pl.Expr:
    expression = pl.col(name)
    if values.schema[name].is_float():
        expression = pl.when(expression.is_nan()).then(missing).otherwise(expression)
    if missing is not None:
        expression = expression.fill_null(missing)
    return expression.cast(pl.Int64, strict=True)


def _primary_layer_name(metadata: Mapping[str, object]) -> str:
    entries = _ordered_entries(metadata.get("layers"), "layer metadata")
    primary = [
        string_value(entry.get("name"), "logical layer name")
        for entry in entries
        if entry.get("location") == "X"
    ]
    if len(primary) != 1:
        raise InvalidResultError("h5 result must declare exactly one layer stored in X")
    return primary[0]


def _axis_frame(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.reset_index(drop=True), include_index=False)


def _aligned_frames(
    stored: Mapping[str, object],
    metadata: Mapping[str, object],
    slot: str,
) -> dict[str, pl.DataFrame]:
    result: dict[str, pl.DataFrame] = {}
    for entry in _ordered_entries(metadata.get(slot), f"{slot} storage metadata"):
        name = string_value(entry.get("name"), f"logical name for {slot}")
        physical_name = string_value(
            entry.get("physical_name"), f"physical name for {slot}[{name!r}]"
        )
        if physical_name not in stored:
            raise InvalidResultError(f"h5 result has no declared {slot}[{name!r}]")
        value = stored[physical_name]
        if isinstance(value, pd.DataFrame):
            frame = pl.from_pandas(value.reset_index(drop=True), include_index=False)
        else:
            frame = pl.DataFrame(np.asarray(value))
        result[name] = restore_table_schema(frame, entry)
    return result


def _pairwise_frames(
    stored: Mapping[str, object],
    metadata: Mapping[str, object],
    slot: str,
) -> dict[str, pl.DataFrame]:
    result: dict[str, pl.DataFrame] = {}
    for entry in _ordered_entries(metadata.get(slot), f"{slot} storage metadata"):
        name = string_value(entry.get("name"), f"logical name for {slot}")
        physical_name = string_value(
            entry.get("physical_name"), f"physical name for {slot}[{name!r}]"
        )
        if physical_name not in stored:
            raise InvalidResultError(f"h5 result has no declared {slot}[{name!r}]")
        result[name] = restore_table_schema(_coordinate_frame(stored[physical_name]), entry)
    return result


def _annotation_tables(
    stored: mudata.MuData,
    metadata: Mapping[str, object],
    shared_metadata: Mapping[str, JsonValue],
) -> tuple[dict[str, AnnotationTable], dict[str, str]]:
    entries = _ordered_entries(metadata.get("annotation_tables", []), "annotation tables")
    scientific = object_mapping(
        shared_metadata.get("annotation_tables", {}), "annotation table metadata"
    )
    result: dict[str, AnnotationTable] = {}
    physical_names: dict[str, str] = {}
    for entry in entries:
        name = string_value(entry.get("name"), "annotation table name")
        details = object_mapping(scientific.get(name), f"annotation table {name!r} metadata")
        physical_name = string_value(
            entry.get("physical_name"), f"annotation table {name!r} physical name"
        )
        if physical_name not in stored.mod:
            raise InvalidResultError(f"h5mu has no declared annotation table {name!r}")
        modality = cast(AnnData, stored[physical_name])
        frame = restore_table_schema(
            _axis_frame(cast(pd.DataFrame, modality.var)),
            entry,
        )
        result[name] = AnnotationTable(
            frame=frame,
            key_columns=tuple(
                string_list(details.get("key_columns"), f"annotation table {name!r} keys")
            ),
            metadata=cast(
                dict[str, JsonValue],
                dict(object_mapping(details.get("metadata", {}), "annotation table metadata")),
            ),
        )
        physical_names[name] = physical_name
    if {string_value(entry.get("name"), "annotation table name") for entry in entries} != set(
        scientific
    ):
        raise InvalidResultError("annotation table order and metadata name different tables")
    return result, physical_names


def _feature_relations(
    stored: mudata.MuData,
    metadata: Mapping[str, object],
    shared_metadata: Mapping[str, JsonValue],
    annotation_names: Mapping[str, str],
    level_names: Mapping[str, str],
) -> dict[str, FeatureRelation]:
    entries = _ordered_entries(metadata.get("feature_relations", []), "feature relations")
    scientific = object_mapping(
        shared_metadata.get("feature_relations", {}), "feature relation metadata"
    )
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for modality_name, modality in stored.mod.items():
        offsets[modality_name] = (offset, modality.n_vars)
        offset += modality.n_vars
    result: dict[str, FeatureRelation] = {}
    for storage_entry in entries:
        name = string_value(storage_entry.get("name"), "feature relation name")
        entry = object_mapping(scientific.get(name), f"feature relation {name!r}")
        physical_name = string_value(
            storage_entry.get("physical_name"), f"feature relation {name!r} physical name"
        )
        if physical_name not in stored.varp:
            raise InvalidResultError(f"h5mu has no declared feature relation {name!r}")
        annotation_table = string_value(
            entry.get("annotation_table"), f"feature relation {name!r} annotation table"
        )
        target_level = string_value(
            entry.get("target_level"), f"feature relation {name!r} target level"
        )
        if annotation_table not in annotation_names:
            raise InvalidResultError(
                f"feature relation {name!r} has unknown annotation table {annotation_table!r}"
            )
        if target_level not in LEVEL_ORDER or target_level not in level_names:
            raise InvalidResultError(
                f"feature relation {name!r} has unknown target level {target_level!r}"
            )
        coordinates = _coordinate_frame(stored.varp[physical_name])
        source_offset, source_size = offsets[annotation_names[annotation_table]]
        target_offset, target_size = offsets[level_names[target_level]]
        rows = coordinates.get_column("row")
        columns = coordinates.get_column("column")
        source_valid = (rows >= source_offset) & (rows < source_offset + source_size)
        target_valid = (columns >= target_offset) & (columns < target_offset + target_size)
        if not (source_valid & target_valid).all():
            raise InvalidResultError(
                f"feature relation {name!r} has values outside its declared modality blocks"
            )
        restored_coordinates = restore_table_schema(
            pl.DataFrame(
                {
                    "row": rows - source_offset,
                    "column": columns - target_offset,
                    "value": coordinates.get_column("value"),
                }
            ),
            storage_entry,
        )
        result[name] = FeatureRelation(
            annotation_table=annotation_table,
            target_level=target_level,
            coordinates=restored_coordinates,
            metadata=cast(
                dict[str, JsonValue],
                dict(object_mapping(entry.get("metadata", {}), "feature relation metadata")),
            ),
        )
    if {string_value(entry.get("name"), "feature relation name") for entry in entries} != set(
        scientific
    ):
        raise InvalidResultError("feature relation order and metadata name different relations")
    return result


def _coordinate_frame(value: object) -> pl.DataFrame:
    if sparse.issparse(value):
        coordinates = cast(sparse.csr_matrix, value).tocoo()
        return pl.DataFrame(
            {
                "row": pl.Series("row", coordinates.row, dtype=pl.Int64),
                "column": pl.Series("column", coordinates.col, dtype=pl.Int64),
                "value": coordinates.data,
            }
        )
    dense = np.asarray(value)
    row, column = np.nonzero(dense)
    return pl.DataFrame(
        {
            "row": pl.Series("row", row, dtype=pl.Int64),
            "column": pl.Series("column", column, dtype=pl.Int64),
            "value": dense[row, column],
        }
    )


def _dense(value: object) -> np.ndarray:
    if sparse.issparse(value):
        return cast(np.ndarray, cast(sparse.csr_matrix, value).toarray())
    return np.asarray(value)


def _result_metadata(stored: AnnData | mudata.MuData) -> Mapping[str, object]:
    namespace = object_mapping(stored.uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")
    raw = namespace.get(STORAGE_NAMESPACE)
    if not isinstance(raw, str):
        raise InvalidResultError("h5 result has no APB2 storage descriptor")
    try:
        metadata = object_mapping(json.loads(raw), "APB2 storage descriptor")
    except json.JSONDecodeError as error:
        raise InvalidResultError(f"invalid APB2 storage descriptor: {error}") from error
    if metadata.get("format") != RESULT_FORMAT:
        raise InvalidResultError(f"h5 object is not an {RESULT_FORMAT} result")
    if metadata.get("format_version") != RESULT_FORMAT_VERSION:
        raise InvalidResultError(
            f"unsupported APB2 h5 result version {metadata.get('format_version')!r}"
        )
    return metadata


def _scientific_namespace(stored: AnnData | mudata.MuData) -> dict[str, JsonValue]:
    namespace = object_mapping(stored.uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")
    return _json_object(
        {key: value for key, value in namespace.items() if key != STORAGE_NAMESPACE},
        "APB metadata",
    )


def _shared_scope(
    scope: Mapping[str, JsonValue],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    shared = dict(scope)
    parse = _json_object(shared.pop(PARSE_NAMESPACE, {}), "shared parse metadata")
    return parse, shared


def _level_scope(
    scope: Mapping[str, JsonValue],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    level = dict(scope)
    parse = _json_object(level.pop(PARSE_NAMESPACE, {}), "level parse metadata")
    roles = _json_object(level.pop(ROLES_NAMESPACE, {}), "level roles")
    columns = roles.get("columns")
    layers = roles.get("layers")
    if columns is not None:
        parse["column_roles"] = columns
    if layers is not None:
        parse["layer_roles"] = layers
    return parse, level


def _shared_extensions(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: _json_value(value)
        for key, value in metadata.items()
        if key not in {"annotation_tables", "feature_relations"}
    }


def _ordered_entries(value: object, role: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise InvalidResultError(f"{role} is not an ordered list")
    return [object_mapping(entry, role) for entry in value]


def _json_object(value: object, role: str) -> dict[str, JsonValue]:
    mapping = object_mapping(value, role)
    return {key: _json_value(item) for key, item in mapping.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise InvalidResultError(f"h5 metadata contains unsupported {type(value).__name__}")
