"""Structural AnnData and MuData serializers for canonical parsed values."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd
import polars as pl
from anndata import AnnData
from mudata import MuData
from scipy import sparse

from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    AnnotationTable,
    FeatureRelation,
    FinalLayerTable,
    JsonValue,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
    QuantitativeLayerSemantics,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.layer_representation import represent_semantics
from apb2.parserV2.parse_quant.io.metadata import (
    NAMESPACE,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    STORAGE_NAMESPACE,
    collection_shared_scope,
    compose_metadata,
    layer_semantics_metadata,
    level_scope,
    logical_table_metadata,
    safe_names,
    shared_scope,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_level, validate_parsed_levels

KEY_SEPARATOR = "_"

LEVEL_VAR_PREFIXES: Mapping[ParsedLevelName, str] = {
    "ion": "ion:",
    "peptidoform": "pfm:",
    "peptide": "pep:",
    "protein": "prt:",
    "fragment": "frg:",
}


class MuDataLevelError(InvalidResultError):
    """The parsed levels cannot form one MuData container."""


def _layer_value_block(layer: FinalLayerTable, /) -> pl.DataFrame:
    """Return observation values without the layer's leading variable keys."""
    return layer.values.select(layer.values.columns[len(layer.var_key_columns) :])


@dataclass(frozen=True, slots=True)
class AnnDataWriter:
    """Structurally serialize one canonical parsed level as ``.h5ad``."""

    def to_anndata(self, parsed: ParsedLevel, /) -> AnnData:
        """Materialize one canonical parsed level without writing it."""
        return self.to_anndata_for_level(parsed, _level_name(parsed), {}, {})

    def to_anndata_for_level(
        self,
        parsed: ParsedLevel,
        level_name: ParsedLevelName,
        shared_uns: Mapping[str, JsonValue],
        shared_metadata: Mapping[str, JsonValue],
        *,
        include_shared: bool = False,
    ) -> AnnData:
        validate_parsed_level(level_name, parsed)
        arrays = {
            name: _layer_value_block(layer).to_numpy().T for name, layer in parsed.layers.items()
        }
        layer_names = safe_names(parsed.layers, prefix="layer", suffix="")
        slot_names = {
            "obsm": safe_names(parsed.obsm, prefix="obsm", suffix=""),
            "varm": safe_names(parsed.varm, prefix="varm", suffix=""),
            "obsp": safe_names(parsed.obsp, prefix="obsp", suffix=""),
            "varp": safe_names(parsed.varp, prefix="varp", suffix=""),
        }
        adata = AnnData(
            X=arrays[parsed.primary_layer_name],
            obs=self._make_axis_frame(parsed.obs.frame, parsed.obs.key_columns),
            var=self._make_axis_frame(parsed.var.frame, parsed.var.key_columns),
            layers={
                layer_names[name]: values
                for name, values in arrays.items()
                if name != parsed.primary_layer_name
            },
        )
        self._write_aligned(parsed, adata, slot_names)
        self._write_pairwise(parsed, adata, slot_names)
        _write_level_namespaces(
            adata,
            level=level_scope(parsed),
            storage=_level_storage_metadata(
                parsed,
                level_name,
                layer_names,
                slot_names,
            ),
            shared=(shared_scope(shared_uns, shared_metadata) if include_shared else None),
        )
        return adata

    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        _write_atomically(target, self.to_anndata(parsed).write_h5ad)

    @staticmethod
    def _write_aligned(
        parsed: ParsedLevel,
        target: AnnData,
        slot_names: Mapping[str, Mapping[str, str]],
    ) -> None:
        for name, frame in parsed.obsm.items():
            target.obsm[slot_names["obsm"][name]] = AnnDataWriter._payload_frame(
                frame, target.obs_names
            )
        for name, frame in parsed.varm.items():
            target.varm[slot_names["varm"][name]] = AnnDataWriter._payload_frame(
                frame, target.var_names
            )

    @staticmethod
    def _write_pairwise(
        parsed: ParsedLevel,
        target: AnnData,
        slot_names: Mapping[str, Mapping[str, str]],
    ) -> None:
        for name, frame in parsed.obsp.items():
            target.obsp[slot_names["obsp"][name]] = AnnDataWriter._sparse_matrix(
                frame, parsed.obs.frame.height
            )
        for name, frame in parsed.varp.items():
            target.varp[slot_names["varp"][name]] = AnnDataWriter._sparse_matrix(
                frame, parsed.var.frame.height
            )

    @staticmethod
    def _payload_frame(frame: pl.DataFrame, index: pd.Index[str]) -> pd.DataFrame:
        payload = pd.DataFrame(
            {name: AnnDataWriter._pandas_column(frame.get_column(name)) for name in frame.columns}
        )
        payload.index = index
        return payload

    @staticmethod
    def _sparse_matrix(frame: pl.DataFrame, axis_size: int) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            sparse.coo_matrix(
                (
                    frame.get_column("value").cast(pl.Float64, strict=True).to_numpy(),
                    (
                        frame.get_column("row").cast(pl.Int64, strict=True).to_numpy(),
                        frame.get_column("column").cast(pl.Int64, strict=True).to_numpy(),
                    ),
                ),
                shape=(axis_size, axis_size),
            )
        )

    @staticmethod
    def _make_axis_frame(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
        """Convert one axis to pandas, keeping every authored key as an ordinary column.

        The per-dtype cases below are a translation table, not a decision: AnnData and HDF5
        accept a specific set of representations, and this is the one place that knows which.
        """
        return _make_axis_frame(frame, key_columns)

    @staticmethod
    def _pandas_column(values: pl.Series) -> pd.Series:
        return _pandas_column(values)

    @staticmethod
    def _storage_index(
        frame: pl.DataFrame,
        columns: Mapping[str, pd.Series],
        key_columns: tuple[str, ...],
    ) -> pd.Index:
        """The one string index AnnData stores, without making it the identity.

        A single string key is already such a string, and the index is built from the column
        itself so the two are the same values in the same dtype — which is what lets AnnData
        store an index that shares its name with a column. Anything else — a number, a
        boolean, several keys — becomes a canonical JSON array of ``[logical type, text]``
        pairs, so an embedded separator, a string ``"1"``, and an integer ``1`` stay
        distinguishable. Parsing never joins or groups on this value.
        """
        return _storage_index(frame, columns, key_columns)


def _make_axis_frame(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
    columns = {name: _pandas_column(frame.get_column(name)) for name in frame.columns}
    table = pd.DataFrame(columns)
    table.index = _storage_index(frame, columns, key_columns)
    return table


def _pandas_column(values: pl.Series) -> pd.Series:
    dtype = values.dtype
    if dtype == pl.Boolean:
        return pd.Series(values.to_list(), dtype="boolean")
    if dtype.is_integer():
        return pd.Series(values.to_list(), dtype="Int64")
    if dtype.is_float():
        return pd.Series(values.to_numpy(), dtype="float64")
    if dtype == pl.Categorical or isinstance(dtype, pl.Enum):
        return values.to_pandas()
    if (dtype == pl.String or dtype == pl.Null) and (values.drop_nulls().n_unique() < values.len()):
        return values.cast(pl.Categorical).to_pandas()
    return pd.Series(values.to_list(), dtype="string")


def _storage_index(
    frame: pl.DataFrame,
    columns: Mapping[str, pd.Series],
    key_columns: tuple[str, ...],
) -> pd.Index:
    if len(key_columns) == 1 and frame.schema[key_columns[0]] == pl.String:
        return pd.Index(columns[key_columns[0]], name=KEY_SEPARATOR.join(key_columns))
    name = KEY_SEPARATOR.join(key_columns)
    while name in columns:
        name += f"{KEY_SEPARATOR}key"
    types = [str(frame.schema[column]) for column in key_columns]
    labels = [
        json.dumps(
            [
                [logical, None if value is None else str(value)]
                for logical, value in zip(types, row, strict=True)
            ],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in frame.select(list(key_columns)).rows()
    ]
    return pd.Index(labels, name=name)


@dataclass(frozen=True, slots=True)
class MuDataWriter:
    """Structurally serialize canonical parsed levels as one MuData file."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        if not parsed.levels:
            raise MuDataLevelError("no parsed levels supplied")
        modalities: dict[str, AnnData] = {}
        writer = AnnDataWriter()
        for level in LEVEL_ORDER:
            if level not in parsed.levels:
                continue
            adata = writer.to_anndata_for_level(
                parsed.levels[level], level, parsed.uns, parsed.metadata
            )
            prefix = LEVEL_VAR_PREFIXES[level]
            adata.var_names = [f"{prefix}{name}" for name in adata.var_names]
            modalities[level] = adata

        annotation_names = {
            name: f"annotation_{physical}"
            for name, physical in safe_names(
                parsed.annotation_tables,
                prefix="table",
                suffix="",
            ).items()
        }
        first = next(iter(modalities.values()))
        for name, table in parsed.annotation_tables.items():
            physical_name = annotation_names[name]
            annotation = _annotation_anndata(table, first, physical_name)
            modalities[physical_name] = annotation

        result = MuData(modalities, axis=0)
        relation_names = {
            name: f"relation_{physical}"
            for name, physical in safe_names(
                parsed.feature_relations,
                prefix="edge",
                suffix="",
            ).items()
        }
        _write_root_feature_relations(
            result,
            parsed.feature_relations,
            annotation_names,
            relation_names,
        )
        _write_namespaces(
            result,
            shared=collection_shared_scope(parsed),
            storage=_collection_storage_metadata(parsed, annotation_names, relation_names),
        )
        _write_atomically(target, result.write_h5mu)


@dataclass(frozen=True, slots=True)
class H5adWriter:
    """Collection-level structural h5ad adapter."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        if parsed.annotation_tables or parsed.feature_relations:
            raise MuDataLevelError("h5ad cannot store annotation tables or feature relations")
        if len(parsed.levels) != 1:
            raise MuDataLevelError(
                f"h5ad requires exactly one parsed level, got {list(parsed.levels)}"
            )
        level_name, level = next(iter(parsed.levels.items()))
        writer = AnnDataWriter()
        _write_atomically(
            target,
            writer.to_anndata_for_level(
                level,
                level_name,
                parsed.uns,
                parsed.metadata,
                include_shared=True,
            ).write_h5ad,
        )


@dataclass(frozen=True, slots=True)
class H5muWriter:
    """Collection-level structural h5mu adapter."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        MuDataWriter().write(parsed, target)


def quantitative_layer_values(parsed: ParsedLevel, layer_name: str, /) -> pl.DataFrame:
    """Return one canonical quantitative value block directly.

    Args:
        parsed: One validated APB2 level.
        layer_name: The logical layer to project.

    Returns:
        A numeric value block with one row per variable and one column per
        observation. Variable-key columns are not included.

    Raises:
        InvalidResultError: The layer is absent or categorical.
    """
    try:
        layer = parsed.layers[layer_name]
    except KeyError as error:
        raise InvalidResultError(f"level has no layer {layer_name!r}") from error
    if not isinstance(layer.semantics, QuantitativeLayerSemantics):
        raise InvalidResultError(f"layer {layer_name!r} is categorical, not quantitative")
    return _layer_value_block(layer)


def represent_layer_values(
    parsed: ParsedLevel,
    layer_name: str,
    /,
    *,
    observation_limit: int,
) -> dict[str, JsonValue]:
    """Describe one canonical layer through its attached scientific semantics."""
    try:
        layer = parsed.layers[layer_name]
    except KeyError as error:
        raise InvalidResultError(f"level has no layer {layer_name!r}") from error
    return represent_semantics(
        layer.semantics,
        _layer_value_block(layer),
        observation_limit=observation_limit,
    )


def _level_name(parsed: ParsedLevel) -> ParsedLevelName:
    value = parsed.uns.get("quantification_level")
    if not isinstance(value, str) or value not in LEVEL_ORDER:
        raise InvalidResultError(
            "an AnnData write requires level provenance in uns['quantification_level']"
        )
    return value


def _level_storage_metadata(
    parsed: ParsedLevel,
    level_name: ParsedLevelName,
    layer_names: Mapping[str, str],
    slot_names: Mapping[str, Mapping[str, str]],
) -> dict[str, JsonValue]:
    layers = cast(
        list[JsonValue],
        [
            {
                "name": name,
                "location": "X" if name == parsed.primary_layer_name else "layers",
                **(
                    {}
                    if name == parsed.primary_layer_name
                    else {"physical_name": layer_names[name]}
                ),
                "value_columns": list(layer.values.columns[len(layer.var_key_columns) :]),
                "role": layer.role.persisted_name(),
                "semantics": layer_semantics_metadata(layer.semantics),
            }
            for name, layer in parsed.layers.items()
        ],
    )
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "level": level_name,
        "obs_key_columns": list(parsed.obs.key_columns),
        "var_key_columns": list(parsed.var.key_columns),
        "obs": logical_table_metadata(parsed.obs.frame),
        "var": logical_table_metadata(parsed.var.frame),
        "layers": layers,
        "obsm": _aligned_storage_entries(parsed.obsm, slot_names["obsm"]),
        "varm": _aligned_storage_entries(parsed.varm, slot_names["varm"]),
        "obsp": _pairwise_storage_entries(parsed.obsp, slot_names["obsp"]),
        "varp": _pairwise_storage_entries(parsed.varp, slot_names["varp"]),
    }


def _aligned_storage_entries(
    frames: Mapping[str, pl.DataFrame], physical_names: Mapping[str, str]
) -> list[JsonValue]:
    return cast(
        list[JsonValue],
        [
            {
                "name": name,
                "physical_name": physical_names[name],
                **logical_table_metadata(frame),
            }
            for name, frame in frames.items()
        ],
    )


def _pairwise_storage_entries(
    frames: Mapping[str, pl.DataFrame], physical_names: Mapping[str, str]
) -> list[JsonValue]:
    return cast(
        list[JsonValue],
        [
            {
                "name": name,
                "physical_name": physical_names[name],
                **logical_table_metadata(frame),
            }
            for name, frame in frames.items()
        ],
    )


def _annotation_anndata(
    table: AnnotationTable,
    reference: AnnData,
    physical_name: str,
) -> AnnData:
    var = _make_axis_frame(table.frame, table.key_columns)
    index_name = "annotation_index"
    while index_name in var.columns:
        index_name = f"_{index_name}"
    var.index = pd.Index(
        [f"ann:{physical_name}:{name}" for name in var.index.astype(str)],
        name=index_name,
    )
    return AnnData(
        X=None,
        obs=cast(pd.DataFrame, reference.obs.copy()),
        var=var,
        shape=(reference.n_obs, table.frame.height),
    )


def _write_root_feature_relations(
    result: MuData,
    relations: Mapping[str, FeatureRelation],
    annotation_names: Mapping[str, str],
    relation_names: Mapping[str, str],
) -> None:
    offsets: dict[str, int] = {}
    offset = 0
    for name, modality in result.mod.items():
        offsets[name] = offset
        offset += modality.n_vars
    for name, relation in relations.items():
        coordinates = relation.coordinates
        source_offset = offsets[annotation_names[relation.annotation_table]]
        target_offset = offsets[relation.target_level]
        matrix = sparse.coo_matrix(
            (
                coordinates.get_column("value").cast(pl.Float64, strict=True).to_numpy(),
                (
                    coordinates.get_column("row").cast(pl.Int64, strict=True).to_numpy()
                    + source_offset,
                    coordinates.get_column("column").cast(pl.Int64, strict=True).to_numpy()
                    + target_offset,
                ),
            ),
            shape=(result.n_vars, result.n_vars),
        )
        result.varp[relation_names[name]] = sparse.csr_matrix(matrix)


def _collection_storage_metadata(
    parsed: ParsedLevels,
    annotation_names: Mapping[str, str],
    relation_names: Mapping[str, str],
) -> dict[str, JsonValue]:
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "levels": cast(
            list[JsonValue],
            [{"name": name, "physical_name": name} for name in parsed.levels],
        ),
        "annotation_tables": cast(
            list[JsonValue],
            [
                {
                    "name": name,
                    "physical_name": annotation_names[name],
                    **logical_table_metadata(table.frame),
                }
                for name, table in parsed.annotation_tables.items()
            ],
        ),
        "feature_relations": cast(
            list[JsonValue],
            [
                {
                    "name": name,
                    "physical_name": relation_names[name],
                    **logical_table_metadata(relation.coordinates),
                }
                for name, relation in parsed.feature_relations.items()
            ],
        ),
    }


def _write_level_namespaces(
    target: AnnData,
    *,
    level: Mapping[str, JsonValue],
    storage: Mapping[str, JsonValue],
    shared: Mapping[str, JsonValue] | None,
) -> None:
    namespace, ownership = compose_metadata(shared or {}, level)
    namespace[STORAGE_NAMESPACE] = json.dumps(
        {**storage, "metadata_ownership": ownership}, ensure_ascii=False, allow_nan=False
    )
    target.uns[NAMESPACE] = namespace


def _write_namespaces(
    target: MuData,
    *,
    shared: Mapping[str, JsonValue],
    storage: Mapping[str, JsonValue],
) -> None:
    """Store canonical shared metadata and a nonduplicating physical descriptor."""
    target.uns[NAMESPACE] = {
        **shared,
        STORAGE_NAMESPACE: json.dumps(storage, ensure_ascii=False, allow_nan=False),
    }


def _write_atomically(target: Path, write: Callable[[Path], None]) -> None:
    """Write beside the destination and replace it only after a complete write."""
    with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
        staged = Path(scratch) / target.name
        write(staged)
        staged.replace(target)
