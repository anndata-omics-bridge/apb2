"""Read APB2-authored h5ad and h5mu results into storage-neutral Polars values."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
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
    MATRIX_PROJECTED_KEY,
    NAMESPACE,
    PARSE_NAMESPACE,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    RESULT_NAMESPACE,
    layer_role_from_metadata,
    object_mapping,
    string_list,
    string_value,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels


@dataclass(frozen=True, slots=True)
class H5adReader:
    """Read one APB2 h5ad envelope as a one-level result collection."""

    def read(self, source: Path, /) -> ParsedLevels:
        try:
            stored = anndata.read_h5ad(source)
        except (OSError, ValueError) as error:
            raise InvalidResultError(f"cannot read h5ad result {source}: {error}") from error
        metadata = _result_metadata(stored)
        level_name, level = _read_level(stored, metadata)
        parsed = ParsedLevels(
            levels={level_name: level},
            uns=_json_object(metadata.get("shared_uns"), "shared result provenance"),
            metadata=_stored_extension_metadata(
                metadata,
                "shared_metadata",
                stored,
                "shared extension metadata",
            ),
        )
        validate_parsed_levels(parsed)
        return parsed


@dataclass(frozen=True, slots=True)
class H5muReader:
    """Read one APB2 h5mu envelope and all modalities in declared level order."""

    def read(self, source: Path, /) -> ParsedLevels:
        try:
            stored = mudata.read_h5mu(source)
        except (OSError, ValueError) as error:
            raise InvalidResultError(f"cannot read h5mu result {source}: {error}") from error
        metadata = _result_metadata(stored)
        order = string_list(metadata.get("level_order"), "h5mu level order")
        levels: dict[ParsedLevelName, ParsedLevel] = {}
        for name in order:
            if name not in LEVEL_ORDER or name not in stored.mod:
                raise InvalidResultError(f"h5mu declares unavailable level {name!r}")
            modality = cast(AnnData, stored[name])
            level_name, level = _read_level(modality, _result_metadata(modality))
            if level_name != name:
                raise InvalidResultError(
                    f"h5mu modality {name!r} contains metadata for {level_name!r}"
                )
            levels[level_name] = level
        if set(order) != set(stored.mod):
            raise InvalidResultError("h5mu level order and modalities name different levels")
        parsed = ParsedLevels(
            levels=levels,
            uns=_json_object(metadata.get("shared_uns"), "shared result provenance"),
            metadata=_extension_metadata(stored),
        )
        validate_parsed_levels(parsed)
        return parsed


def _read_level(
    stored: AnnData, metadata: Mapping[str, object]
) -> tuple[ParsedLevelName, ParsedLevel]:
    level_name = string_value(metadata.get("level"), "quantification level")
    if level_name not in LEVEL_ORDER:
        raise InvalidResultError(f"unknown quantification level {level_name!r}")
    obs = ObsFinal(
        frame=_axis_frame(cast(pd.DataFrame, stored.obs)),
        key_columns=tuple(string_list(metadata.get("obs_key_columns"), "obs key columns")),
    )
    var = VarFinal(
        frame=_axis_frame(cast(pd.DataFrame, stored.var)),
        key_columns=tuple(string_list(metadata.get("var_key_columns"), "var key columns")),
    )
    layers = _layers(stored, var.frame, metadata)
    uns = _parse_provenance(stored)
    uns[MATRIX_PROJECTED_KEY] = True
    return level_name, ParsedLevel(
        obs=obs,
        var=var,
        primary_layer_name=string_value(metadata.get("primary_layer"), "primary layer"),
        layers=layers,
        obsm=_aligned_frames(stored.obsm, metadata, "obsm"),
        varm=_aligned_frames(stored.varm, metadata, "varm"),
        obsp=_pairwise_frames(stored.obsp, metadata, "obsp"),
        varp=_pairwise_frames(stored.varp, metadata, "varp"),
        uns=uns,
        metadata=_stored_extension_metadata(
            metadata,
            "level_metadata",
            stored,
            "level extension metadata",
        ),
    )


def _layers(
    stored: AnnData,
    var: pl.DataFrame,
    metadata: Mapping[str, object],
) -> dict[str, FinalLayerTable]:
    order = string_list(metadata.get("layer_order"), "layer order")
    entries = object_mapping(metadata.get("layers"), "layer metadata")
    result: dict[str, FinalLayerTable] = {}
    for name in order:
        entry = object_mapping(entries.get(name), f"layer {name!r}")
        physical_name = string_value(
            entry.get("physical_name"), f"physical layer name for {name!r}"
        )
        if physical_name not in stored.layers:
            raise InvalidResultError(f"h5 result has no declared layer {name!r}")
        keys = tuple(string_list(entry.get("var_key_columns"), f"layer {name!r} var keys"))
        value_columns = string_list(entry.get("value_columns"), f"layer {name!r} value columns")
        matrix = _dense(stored.layers[physical_name])
        if matrix.shape != (stored.n_obs, stored.n_vars):
            raise InvalidResultError(
                f"layer {name!r} has shape {matrix.shape}; expected {(stored.n_obs, stored.n_vars)}"
            )
        values = pl.DataFrame(matrix.T, schema=value_columns, orient="row")
        result[name] = FinalLayerTable(
            layer_name=name,
            var_key_columns=keys,
            values=pl.concat([var.select(keys), values], how="horizontal_extend"),
            role=layer_role_from_metadata(entry, f"layer {name!r}"),
        )
    if set(order) != set(entries):
        raise InvalidResultError("layer order and layer metadata name different layers")
    return result


def _axis_frame(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_pandas(frame.reset_index(drop=True), include_index=False)


def _aligned_frames(
    stored: Mapping[str, object],
    metadata: Mapping[str, object],
    slot: str,
) -> dict[str, pl.DataFrame]:
    order = string_list(metadata.get(f"{slot}_order"), f"{slot} order")
    physical_names = object_mapping(
        metadata.get(f"{slot}_physical_names"), f"{slot} physical names"
    )
    result: dict[str, pl.DataFrame] = {}
    for name in order:
        physical_name = string_value(
            physical_names.get(name), f"physical name for {slot}[{name!r}]"
        )
        if physical_name not in stored:
            raise InvalidResultError(f"h5 result has no declared {slot}[{name!r}]")
        value = stored[physical_name]
        if isinstance(value, pd.DataFrame):
            result[name] = pl.from_pandas(value.reset_index(drop=True), include_index=False)
        else:
            result[name] = pl.DataFrame(np.asarray(value))
    return result


def _pairwise_frames(
    stored: Mapping[str, object],
    metadata: Mapping[str, object],
    slot: str,
) -> dict[str, pl.DataFrame]:
    order = string_list(metadata.get(f"{slot}_order"), f"{slot} order")
    physical_names = object_mapping(
        metadata.get(f"{slot}_physical_names"), f"{slot} physical names"
    )
    result: dict[str, pl.DataFrame] = {}
    for name in order:
        physical_name = string_value(
            physical_names.get(name), f"physical name for {slot}[{name!r}]"
        )
        if physical_name not in stored:
            raise InvalidResultError(f"h5 result has no declared {slot}[{name!r}]")
        value = stored[physical_name]
        if sparse.issparse(value):
            coordinates = cast(sparse.csr_matrix, value).tocoo()
            result[name] = pl.DataFrame(
                {"row": coordinates.row, "column": coordinates.col, "value": coordinates.data}
            )
            continue
        dense = np.asarray(value)
        row, column = np.nonzero(dense)
        result[name] = pl.DataFrame({"row": row, "column": column, "value": dense[row, column]})
    return result


def _dense(value: object) -> np.ndarray:
    if sparse.issparse(value):
        return cast(np.ndarray, cast(sparse.csr_matrix, value).toarray())
    return np.asarray(value)


def _result_metadata(stored: AnnData | mudata.MuData) -> Mapping[str, object]:
    namespace = object_mapping(stored.uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")
    raw = namespace.get(RESULT_NAMESPACE)
    if not isinstance(raw, str):
        raise InvalidResultError("h5 result has no APB2 result envelope")
    try:
        metadata = object_mapping(json.loads(raw), "APB2 result envelope")
    except json.JSONDecodeError as error:
        raise InvalidResultError(f"invalid APB2 result envelope: {error}") from error
    if metadata.get("format") != RESULT_FORMAT:
        raise InvalidResultError(f"h5 object is not an {RESULT_FORMAT} result")
    if metadata.get("format_version") != RESULT_FORMAT_VERSION:
        raise InvalidResultError(
            f"unsupported APB2 h5 result version {metadata.get('format_version')!r}"
        )
    return metadata


def _parse_provenance(stored: AnnData) -> dict[str, JsonValue]:
    namespace = object_mapping(stored.uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")
    return _json_object(namespace.get(PARSE_NAMESPACE), "parse provenance")


def _extension_metadata(stored: AnnData | mudata.MuData) -> dict[str, JsonValue]:
    namespace = object_mapping(stored.uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")
    return {
        key: _json_value(value)
        for key, value in namespace.items()
        if key not in {PARSE_NAMESPACE, RESULT_NAMESPACE}
    }


def _stored_extension_metadata(
    result: Mapping[str, object],
    name: str,
    stored: AnnData,
    role: str,
) -> dict[str, JsonValue]:
    value = result.get(name)
    return _extension_metadata(stored) if value is None else _json_object(value, role)


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
