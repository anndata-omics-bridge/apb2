"""Structural AnnData and MuData serializers for canonical parsed values."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pandas as pd
import polars as pl
import polars.selectors as cs
import pyarrow as pa
from anndata import AnnData
from anndata import settings as anndata_settings
from mudata import MuData
from scipy import sparse

from apb2.parserV2.parse_quant.data.parsed import (
    AnnotationTable,
    FeatureRelation,
    JsonValue,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.layer_representation import represent_semantics
from apb2.parserV2.parse_quant.io.metadata import (
    NAMESPACE,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    collection_shared_scope,
    layer_semantics_metadata,
    level_scope,
    logical_table_metadata,
    safe_names,
    shared_scope,
)
from apb2.parserV2.parse_quant.io.uns_json import UnsJsonCodec
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


class AnnDataWriter:
    """Structurally serialize one canonical parsed level as ``.h5ad``."""

    __slots__ = ()

    def to_anndata_for_level(self, parsed: ParsedLevel, level_name: ParsedLevelName) -> AnnData:
        """Build one level's AnnData with that level's APB part in ``uns["apb"]``."""
        validate_parsed_level(level_name, parsed)
        arrays = {name: layer.values.to_numpy().T for name, layer in parsed.layers.items()}
        layer_names = safe_names(parsed.layers, prefix="layer", suffix="")
        slot_names = {
            "obsm": safe_names(parsed.obsm, prefix="obsm", suffix=""),
            "varm": safe_names(parsed.varm, prefix="varm", suffix=""),
            "obsp": safe_names(parsed.obsp, prefix="obsp", suffix=""),
            "varp": safe_names(parsed.varp, prefix="varp", suffix=""),
        }
        adata = AnnData(
            X=arrays[parsed.primary_layer_name],
            obs=_make_axis_frame(parsed.obs.frame, parsed.obs.key_columns),
            var=_make_axis_frame(parsed.var.frame, parsed.var.key_columns),
            layers={
                layer_names[name]: values
                for name, values in arrays.items()
                if name != parsed.primary_layer_name
            },
        )
        self._write_aligned(parsed, adata, slot_names)
        self._write_pairwise(parsed, adata, slot_names)
        adata.uns[NAMESPACE] = UnsJsonCodec().encode(
            level_scope(parsed),
            _level_storage_metadata(parsed, level_name, layer_names, slot_names),
        )
        return adata

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
        payload = _pandas_frame(frame)
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


def _make_axis_frame(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
    table = _pandas_frame(frame)
    table.index = _storage_index(frame, table, key_columns)
    return table


def _pandas_frame(frame: pl.DataFrame) -> pd.DataFrame:
    """Convert through Arrow once, preserving AnnData's nullable and categorical types."""
    repeated = (
        frame.select(cs.string() | cs.by_dtype(pl.Null))
        .select(pl.all().drop_nulls().n_unique() < pl.len())
        .unpivot()
        .filter(pl.col("value").cast(pl.Boolean))
        .select("variable")
        .to_series()
        .to_list()
    )
    return (
        frame.with_columns(
            cs.integer().cast(pl.Int64),
            cs.float().cast(pl.Float64),
            (cs.binary() | cs.by_dtype(pl.Null)).cast(pl.String),
            (
                ~(
                    cs.integer()
                    | cs.float()
                    | cs.boolean()
                    | cs.categorical()
                    | cs.enum()
                    | cs.string()
                    | cs.binary()
                    | cs.by_dtype(pl.Null)
                )
            ).map_elements(str, return_dtype=pl.String),
        )
        .with_columns(pl.col(repeated).cast(pl.Categorical))
        .to_pandas(
            types_mapper={
                pa.bool_(): pd.BooleanDtype(),
                pa.int64(): pd.Int64Dtype(),
                pa.string(): pd.StringDtype(),
                pa.large_string(): pd.StringDtype(),
            }.get
        )
    )


def _storage_index(
    frame: pl.DataFrame,
    columns: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> pd.Index:
    """Encode typed keys without iterating over table rows; these are storage labels only."""
    if len(key_columns) == 1 and frame.schema[key_columns[0]] == pl.String:
        return pd.Index(columns[key_columns[0]], name=KEY_SEPARATOR.join(key_columns))
    name = KEY_SEPARATOR.join(key_columns)
    while name in columns:
        name += f"{KEY_SEPARATOR}key"
    pairs: list[pl.Expr] = []
    for column, dtype in frame.select(key_columns).schema.items():
        text = pl.col(column).cast(pl.String)
        if dtype == pl.Boolean:
            text = text.replace({"true": "True", "false": "False"})
        elif not (
            isinstance(dtype, (pl.String, pl.Null, pl.Categorical, pl.Enum)) or dtype.is_integer()
        ):
            # Persisted labels use Python's scalar spelling (not Polars' float formatting).
            text = pl.col(column).map_elements(str, return_dtype=pl.String)
        quoted = (
            pl.struct(text.alias("v"))
            .struct.json_encode()
            .str.strip_prefix('{"v":')
            .str.strip_suffix("}")
        )
        pairs.append(pl.concat_str(pl.lit("[" + json.dumps(str(dtype)) + ","), quoted, pl.lit("]")))
    labels = (
        frame.select(pl.concat_str(pl.lit("["), pl.concat_str(pairs, separator=","), pl.lit("]")))
        .to_series()
        .to_numpy()
    )
    return pd.Index(labels, name=name)


class MuDataWriter:
    """Structurally serialize canonical parsed levels as one MuData file."""

    __slots__ = ()

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        if not parsed.levels:
            raise MuDataLevelError("no parsed levels supplied")
        modalities: dict[str, AnnData] = {}
        writer = AnnDataWriter()
        for level in parsed.levels:
            adata = writer.to_anndata_for_level(parsed.levels[level], level)
            prefix = LEVEL_VAR_PREFIXES.get(level, f"{level}:")
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


class H5adWriter:
    """Collection-level structural h5ad adapter."""

    __slots__ = ()

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        if parsed.annotation_tables or parsed.feature_relations:
            raise MuDataLevelError("h5ad cannot store annotation tables or feature relations")
        if len(parsed.levels) != 1:
            raise MuDataLevelError(
                f"h5ad requires exactly one parsed level, got {list(parsed.levels)}"
            )
        level_name, level = next(iter(parsed.levels.items()))
        if level_name == NAMESPACE:
            raise MuDataLevelError(f"an h5ad level cannot be named {NAMESPACE!r}")
        adata = AnnDataWriter().to_anndata_for_level(level, level_name)
        # The same two parts as a MuData: the level's under uns[level], the root's at uns["apb"].
        adata.uns[level_name] = {NAMESPACE: adata.uns.pop(NAMESPACE)}
        adata.uns[NAMESPACE] = UnsJsonCodec().encode(
            shared_scope(parsed.uns, parsed.metadata, parsed.hierarchy),
            _collection_storage_metadata(parsed, {}, {}),
        )
        _write_atomically(target, adata.write_h5ad)


class H5muWriter:
    """Collection-level structural h5mu adapter."""

    __slots__ = ()

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        MuDataWriter().write(parsed, target)


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
        layer.values,
        observation_limit=observation_limit,
    )


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
                "value_columns": list(layer.values.columns),
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


def _write_namespaces(
    target: MuData,
    *,
    shared: Mapping[str, JsonValue],
    storage: Mapping[str, JsonValue],
) -> None:
    """Store canonical shared metadata and a nonduplicating physical descriptor."""
    target.uns[NAMESPACE] = UnsJsonCodec().encode(shared, storage)


def _write_atomically(target: Path, write: Callable[[Path], None]) -> None:
    """Write beside the destination and replace it only after a complete write."""
    with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
        staged = Path(scratch) / target.name
        # Polars string columns become pandas nullable StringArray values here.
        # AnnData's default deliberately rejects them unless the writer opts in.
        with anndata_settings.override(allow_write_nullable_strings=True):
            write(staged)
        staged.replace(target)
