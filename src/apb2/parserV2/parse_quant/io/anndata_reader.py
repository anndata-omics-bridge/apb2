"""Read APB2-authored h5ad and h5mu results into storage-neutral Polars values.

An AnnData or MuData APB2 did not write, such as an export, is read as levels for its sidecar.

Neither file is read whole. Axes, aligned and pairwise slots and ``uns`` are read element by
element; each matrix is read in slabs of feature columns, of which only the stored cells are
kept, so a dense 79 GB ion level of a 1,236-run report reads back as its 18 GB of cells. A
missing cell of a number layer is NaN in the file, so NaN reads as missing. One level of an
h5mu can be read alone.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from anndata.io import read_elem
from scipy import sparse

from apb2.parserV2.parse_quant.data.layer_blocks import READ_BLOCK_BYTES, LayerBlocks
from apb2.parserV2.parse_quant.data.parking import LayerParking
from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    FeatureRelation,
    FinalLayerRole,
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
    layer_role_from_metadata,
    layer_semantics_from_metadata,
    object_mapping,
    read_hierarchy,
    relations_among,
    restore_level_roles,
    restore_table_schema,
    string_list,
    string_value,
)
from apb2.parserV2.parse_quant.io.uns_json import UnsJsonCodec, python_value
from apb2.parserV2.parse_quant.io.validation import validate_parsed_levels


class H5adReader:
    """Read one APB2 h5ad envelope as a one-level result collection."""

    __slots__ = ()

    def read(self, source: Path, level: ParsedLevelName | None = None, /) -> ParsedLevels:
        with _opened(source, "h5ad") as file:
            stored = _matrixless(file)
            metadata = _result_metadata(stored.uns)
            entries = _ordered_entries(metadata.get("levels"), "h5ad levels")
            if (
                len(entries) != 1
                or metadata.get("annotation_tables")
                or metadata.get("feature_relations")
            ):
                raise InvalidResultError("an h5ad result declares exactly one level and no tables")
            name = string_value(entries[0].get("name"), "h5ad level name")
            if level is not None and level != name:
                raise InvalidResultError(f"h5ad result {source} has level {name!r}, not {level!r}")
            physical_name = string_value(
                entries[0].get("physical_name"), "h5ad level physical name"
            )
            level_part = object_mapping(stored.uns.get(physical_name), f"uns[{physical_name!r}]")
            level_metadata = _result_metadata(level_part)
            level_name, parsed_level = _read_level(
                stored, file, level_metadata, _scientific_namespace(level_part, level_metadata)
            )
            LayerParking().park(parsed_level)
        if level_name != name:
            raise InvalidResultError(f"h5ad level {name!r} contains metadata for {level_name!r}")
        shared_uns, shared_metadata = _shared_scope(_scientific_namespace(stored.uns, metadata))
        parsed = ParsedLevels(
            hierarchy=read_hierarchy(shared_metadata),
            levels={level_name: parsed_level},
            uns=shared_uns,
            metadata=shared_metadata,
            export_uns=_export_uns(stored.uns, metadata),
        )
        validate_parsed_levels(parsed)
        return parsed


class H5muReader:
    """Read one APB2 h5mu envelope: all modalities in declared level order, or one level."""

    __slots__ = ()

    def read(self, source: Path, level: ParsedLevelName | None = None, /) -> ParsedLevels:
        with _opened(source, "h5mu") as file:
            envelope = _MuDataEnvelope.of(file)
            metadata = _result_metadata(envelope.uns)
            level_entries = _ordered_entries(metadata.get("levels"), "h5mu levels")
            names = [string_value(entry.get("name"), "h5mu level name") for entry in level_entries]
            if level is not None and level not in names:
                raise InvalidResultError(
                    f"h5mu result {source} has no level {level!r}; it has {names}"
                )
            levels: dict[ParsedLevelName, ParsedLevel] = {}
            level_names: dict[str, str] = {}
            for name, entry in zip(names, level_entries, strict=True):
                physical_name = string_value(entry.get("physical_name"), "h5mu level physical name")
                if physical_name not in envelope.order:
                    raise InvalidResultError(f"h5mu declares unavailable level {name!r}")
                level_names[name] = physical_name
                if level is not None and name != level:
                    continue
                group = envelope.modality(physical_name)
                modality = _matrixless(group)
                modality_metadata = _result_metadata(modality.uns)
                level_name, parsed_level = _read_level(
                    modality,
                    group,
                    modality_metadata,
                    _scientific_namespace(modality.uns, modality_metadata),
                )
                if level_name != name:
                    raise InvalidResultError(
                        f"h5mu modality {name!r} contains metadata for {level_name!r}"
                    )
                LayerParking().park(parsed_level)
                levels[level_name] = parsed_level
            shared_uns, shared_metadata = _shared_scope(
                _scientific_namespace(envelope.uns, metadata)
            )
            annotation_tables, annotation_names = _annotation_tables(
                envelope, metadata, shared_metadata
            )
            expected_modalities = set(level_names.values()).union(annotation_names.values())
            if expected_modalities != set(envelope.order):
                raise InvalidResultError("h5mu level order and modalities name different levels")
            feature_relations = _feature_relations(
                envelope, metadata, shared_metadata, annotation_names, level_names
            )
        parsed = ParsedLevels(
            hierarchy=read_hierarchy(shared_metadata),
            levels=levels,
            uns=shared_uns,
            metadata=_shared_extensions(shared_metadata),
            annotation_tables=annotation_tables,
            feature_relations=relations_among(feature_relations, {*levels, *annotation_tables}),
            export_uns=_export_uns(envelope.uns, metadata),
        )
        validate_parsed_levels(parsed)
        return parsed


def _opened(source: Path, kind: str) -> h5py.File:
    try:
        return h5py.File(source, "r")
    except OSError as error:
        raise InvalidResultError(f"cannot read {kind} result {source}: {error}") from error


def _matrixless(group: h5py.Group) -> AnnData:
    """One stored AnnData without ``X`` and layers, which stay in the file to be read in slabs."""

    def element(key: str) -> Any:  # noqa: ANN401 - read_elem returns whatever the group encodes
        return read_elem(cast(h5py.Group, group[key])) if key in group else {}

    try:
        return AnnData(
            obs=element("obs"),
            var=element("var"),
            uns=element("uns"),
            obsm=element("obsm"),
            varm=element("varm"),
            obsp=element("obsp"),
            varp=element("varp"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidResultError(f"h5 group {group.name} is not an AnnData: {error}") from error


@dataclass(frozen=True, slots=True)
class _MuDataEnvelope:
    """An h5mu file's root ``uns`` and ``varp``, and its modalities in their stored order."""

    file: h5py.File
    uns: dict[str, object]
    varp: dict[str, object]
    order: tuple[str, ...]

    @classmethod
    def of(cls, file: h5py.File) -> _MuDataEnvelope:
        def element(key: str) -> dict[str, object]:
            return (
                cast(dict[str, object], read_elem(cast(h5py.Group, file[key])))
                if key in file
                else {}
            )

        if "mod" not in file:
            raise InvalidResultError(f"{file.filename} is not an h5mu file")
        modalities = cast(h5py.Group, file["mod"])
        stored_order = modalities.attrs.get("mod-order")
        order = (
            tuple(str(name) for name in cast(list[str], list(stored_order)))
            if stored_order is not None
            else tuple(modalities)
        )
        return cls(file, element("uns"), element("varp"), order)

    def modality(self, name: str) -> h5py.Group:
        return cast(h5py.Group, cast(h5py.Group, self.file["mod"])[name])

    def var_count(self, name: str) -> int:
        """A modality's feature count, from its var index alone."""
        var = cast(h5py.Group, self.modality(name)["var"])
        index = var[str(var.attrs["_index"])]
        # An encoded column (nullable strings, categories) is a group of parallel arrays.
        while isinstance(index, h5py.Group):
            index = next(index[key] for key in ("codes", "values") if key in index)
        return int(cast(h5py.Dataset, index).shape[0])


def _read_level(
    stored: AnnData,
    group: h5py.Group,
    metadata: Mapping[str, object],
    scope: dict[str, JsonValue],
) -> tuple[ParsedLevelName, ParsedLevel]:
    level_name = string_value(metadata.get("level"), "quantification level")
    if not level_name:
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
    layers = _layers(stored, group, metadata)
    uns, level_metadata = _level_scope(scope)
    primary = _primary_layer_name(metadata)
    return level_name, restore_level_roles(
        ParsedLevel(
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
            export_uns=_export_uns(stored.uns, metadata),
        )
    )


def _export_uns(uns: Mapping[str, object], storage: Mapping[str, object]) -> dict[str, JsonValue]:
    """The top-level keys the descriptor lists; keys another tool added are not read."""
    keys = string_list(storage.get("export_uns"), "export_uns keys")
    absent = [key for key in keys if key not in uns]
    if absent:
        raise InvalidResultError(f"uns lacks the recorded export_uns keys {absent}")
    return {key: python_value(uns[key]) for key in keys}


def _layers(
    stored: AnnData,
    group: h5py.Group,
    metadata: Mapping[str, object],
) -> dict[str, FinalLayerTable]:
    entries = _ordered_entries(metadata.get("layers"), "layer metadata")
    shape = (stored.n_obs, stored.n_vars)
    result: dict[str, FinalLayerTable] = {}
    for entry in entries:
        name = string_value(entry.get("name"), "logical layer name")
        location = string_value(entry.get("location"), f"layer {name!r} location")
        value_columns = tuple(
            string_list(entry.get("value_columns"), f"layer {name!r} value columns")
        )
        if location == "X":
            key, parent = "X", group
        elif location == "layers":
            key = string_value(entry.get("physical_name"), f"physical layer name for {name!r}")
            parent = cast(h5py.Group, group["layers"]) if "layers" in group else None
        else:
            raise InvalidResultError(f"layer {name!r} has unknown location {location!r}")
        if parent is None or key not in parent:
            raise InvalidResultError(f"h5 result has no declared layer {name!r}")
        result[name] = _StoredMatrix(
            parent[key],
            shape,
            value_columns,
            layer_semantics_from_metadata(entry.get("semantics"), f"layer {name!r}"),
        ).layer(name, layer_role_from_metadata(entry, f"layer {name!r}"))
    return result


@dataclass(frozen=True, slots=True)
class _StoredMatrix:
    """One stored ``(n_obs, n_var)`` matrix, read as cells one slab of feature columns at a time.

    A sparse matrix another tool wrote is read whole, as before; APB2 writes dense ones.
    """

    element: object
    shape: tuple[int, int]
    value_columns: tuple[str, ...]
    semantics: FinalLayerSemantics

    def layer(self, name: str, role: FinalLayerRole) -> FinalLayerTable:
        return LayerBlocks.stored(
            name, self._blocks(name), labels=self.value_columns, role=role, semantics=self.semantics
        )

    def _blocks(self, name: str) -> Iterator[pl.DataFrame]:
        n_obs, n_var = self.shape
        if isinstance(self.element, h5py.Dataset):
            dataset = self.element
            if dataset.shape != self.shape:
                raise InvalidResultError(
                    f"layer {name!r} has shape {dataset.shape}; expected {self.shape}"
                )
            step = max(READ_BLOCK_BYTES // max(n_obs * 8, 1), 1)
            for start in range(0, max(n_var, 1), step):
                yield self._values(np.asarray(dataset[:, start : start + step]))
            return
        matrix = _dense(read_elem(cast(h5py.Group, self.element)))
        if matrix.shape != self.shape:
            raise InvalidResultError(
                f"layer {name!r} has shape {matrix.shape}; expected {self.shape}"
            )
        yield self._values(matrix)

    def _values(self, matrix: np.ndarray) -> pl.DataFrame:
        return _canonical_layer_values(
            pl.DataFrame(matrix.T, schema=list(self.value_columns), orient="row"), self.semantics
        )


def _canonical_layer_values(
    values: pl.DataFrame,
    semantics: FinalLayerSemantics,
    /,
) -> pl.DataFrame:
    """Restore the canonical scalar representation from an HDF5 matrix."""
    if isinstance(semantics, QuantitativeLayerSemantics):
        if semantics.logical_type == "number":
            # A missing cell is NaN in the file: it cannot be told from a NaN value.
            return values.fill_nan(None)
        return values.fill_nan(None).cast(pl.Int64, strict=True)
    return values.fill_nan(None).fill_null(semantics.missing_code).cast(pl.Int64, strict=True)


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
    stored: _MuDataEnvelope,
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
        if physical_name not in stored.order:
            raise InvalidResultError(f"h5mu has no declared annotation table {name!r}")
        modality = _matrixless(stored.modality(physical_name))
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
    stored: _MuDataEnvelope,
    metadata: Mapping[str, object],
    shared_metadata: Mapping[str, JsonValue],
    annotation_names: Mapping[str, str],
    level_names: Mapping[str, str],
) -> dict[str, FeatureRelation]:
    entries = _ordered_entries(metadata.get("feature_relations", []), "feature relations")
    scientific = object_mapping(
        shared_metadata.get("feature_relations", {}), "feature relation metadata"
    )
    logical = {physical: name for name, physical in {**level_names, **annotation_names}.items()}
    if not entries:
        return {}
    blocks = pl.Series("name", [logical[physical] for physical in stored.order], dtype=pl.String)
    starts = np.cumsum([0, *(stored.var_count(physical) for physical in stored.order)])[:-1]
    result: dict[str, FeatureRelation] = {}
    for storage_entry in entries:
        name = string_value(storage_entry.get("name"), "feature relation name")
        entry = object_mapping(scientific.get(name), f"feature relation {name!r}")
        physical_name = string_value(
            storage_entry.get("physical_name"), f"feature relation {name!r} physical name"
        )
        if physical_name not in stored.varp:
            raise InvalidResultError(f"h5mu has no declared feature relation {name!r}")
        restored_coordinates = restore_table_schema(
            _relation_blocks(_coordinate_frame(stored.varp[physical_name]), blocks, starts),
            storage_entry,
        )
        result[name] = FeatureRelation(
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


def _relation_blocks(
    coordinates: pl.DataFrame, blocks: pl.Series, starts: np.ndarray
) -> pl.DataFrame:
    """Positions on the MuData's var axis as features of the modality each falls in."""
    rows = coordinates.get_column("row").to_numpy()
    columns = coordinates.get_column("column").to_numpy()
    # The last modality starting at or before a position holds it; empty ones start where the
    # next begins.
    source = np.searchsorted(starts, rows, side="right") - 1
    target = np.searchsorted(starts, columns, side="right") - 1
    return pl.DataFrame(
        {
            "source": blocks.gather(source),
            "row": pl.Series(rows - starts[source], dtype=pl.Int64),
            "target": blocks.gather(target),
            "column": pl.Series(columns - starts[target], dtype=pl.Int64),
            "value": coordinates.get_column("value"),
        }
    )


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


def _result_metadata(uns: Mapping[str, object]) -> Mapping[str, object]:
    metadata = UnsJsonCodec().storage(_namespace(uns))
    if metadata.get("format") != RESULT_FORMAT:
        raise InvalidResultError(f"h5 object is not an {RESULT_FORMAT} result")
    if metadata.get("format_version") != RESULT_FORMAT_VERSION:
        raise InvalidResultError(
            f"unsupported APB2 h5 result version {metadata.get('format_version')!r}"
        )
    return metadata


def _scientific_namespace(
    uns: Mapping[str, object], metadata: Mapping[str, object]
) -> dict[str, JsonValue]:
    return UnsJsonCodec().decode(_namespace(uns), metadata)


def _namespace(uns: Mapping[str, object]) -> Mapping[str, object]:
    return object_mapping(uns.get(NAMESPACE), f"uns[{NAMESPACE!r}]")


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
        key: value
        for key, value in metadata.items()
        if key not in {"annotation_tables", "feature_relations"}
    }


def _ordered_entries(value: object, role: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise InvalidResultError(f"{role} is not an ordered list")
    return [object_mapping(entry, role) for entry in value]


def _json_object(value: object, role: str) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], dict(object_mapping(value, role)))
