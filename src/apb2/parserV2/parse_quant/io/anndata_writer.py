"""The AnnData result boundary: the only place parsed values stop being vendor scalars.

Everything lossy or backend-specific happens here and only here. Layer text becomes float
codes, missing sentinels become missing, wide frames become dense arrays, Polars becomes
pandas, and a composite identity becomes the one string AnnData will accept as an index. None
of that is parsing, which is why none of it is visible upstream.

The order matters. Encoding runs first, so a failed numeric interpretation is a *visible*
failure rather than a quietly empty column; then the contract check, which can now tell an
empty experiment from a parse that lost its quantities; then allocation, exactly one array per
encoded layer.

``AnnDataLayerEncoder`` and ``AnnDataLayerContractChecker`` are declared here rather than in
the shared contracts module because this writer is their only client.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from loguru import logger
from mudata import MuData
from scipy import sparse

from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation, as_numbers, blank
from apb2.parserV2.parse_quant.data.parsed import (
    LEVEL_ORDER,
    AnnotationTable,
    FeatureRelation,
    FinalLayerTable,
    JsonValue,
    ParsedLevel,
    ParsedLevelName,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.io.errors import (
    AnnDataLayerContractError,
    InvalidResultError,
)
from apb2.parserV2.parse_quant.io.metadata import (
    MATRIX_PROJECTED_KEY,
    NAMESPACE,
    PARSE_NAMESPACE,
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    RESULT_NAMESPACE,
    object_mapping,
    safe_names,
    string_list,
    string_value,
    table_metadata,
)
from apb2.parserV2.parse_quant.io.validation import validate_parsed_level, validate_parsed_levels

KEY_SEPARATOR = "_"
UNKNOWN_FACTOR_CODE = -1
_EXAMPLE_LIMIT = 5
_DERIVED_EMPTY_RATIO = 0.001
_DERIVED_POPULATED_RATIO = 0.5

LEVEL_VAR_PREFIXES: Mapping[ParsedLevelName, str] = {
    "ion": "ion:",
    "peptidoform": "pfm:",
    "peptide": "pep:",
    "protein": "prt:",
    "fragment": "frg:",
}


class MuDataLevelError(InvalidResultError):
    """Parsed levels and their configured AnnData writers do not form one container."""


class AnnDataPlanError(InvalidResultError):
    """Stored parse provenance cannot reconstruct the declared AnnData projection."""


class AnnDataLayerEncoder(Protocol):
    """Encode one layer's value block for AnnData, preserving its shape and column order."""

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame: ...


class AnnDataLayerContractChecker(Protocol):
    """Enforce the required-layer and occupancy policy on the encoded layers."""

    def check(
        self,
        encoded: Mapping[str, pl.DataFrame],
        occupancy_candidates: Mapping[str, pl.DataFrame],
        /,
    ) -> None: ...


def _layer_value_block(layer: FinalLayerTable, /) -> pl.DataFrame:
    """Return observation values without the layer's leading variable keys."""
    return layer.values.select(layer.values.columns[len(layer.var_key_columns) :])


# --------------------------------------------------------------------------------- encoders


def _masked(numbers: pl.Expr, missing_values: tuple[float, ...]) -> pl.Expr:
    """Blank out the values the vendor writes to mean "not measured"."""
    if not missing_values:
        return numbers
    return pl.when(numbers.is_in(list(missing_values))).then(None).otherwise(numbers)


@dataclass(frozen=True, slots=True)
class PlainNumericAnnDataEncoder:
    """Directly parseable scalars become floats; declared missing values become missing.

    A non-blank token this layer cannot hold becomes missing and is reported. It does not
    raise: the vendors these rules were written for write ``-``, ``NA``, and ``False`` in a
    column their own rule calls numeric, and refusing the file would convert nothing at all.
    The encoded-layer contract is what decides whether enough survived to be usable, which is
    the check that can tell an empty experiment from a parse that lost its quantities.
    """

    layer_name: str
    missing_values: tuple[float, ...]
    number_format: NumberNotation

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame:
        columns = tuple(values.columns)
        if not columns:
            return values
        number_labels = self._temporary_labels("_number", len(columns), reserved=columns)
        mask_labels = self._temporary_labels(
            "_unreadable",
            len(columns),
            reserved=(*columns, *number_labels),
        )
        prepared = values.with_columns(
            [
                as_numbers(pl.col(column), values.schema[column], self.number_format).alias(label)
                for column, label in zip(columns, number_labels, strict=True)
            ]
        ).with_columns(
            [
                (
                    ~blank(pl.col(column), values.schema[column]) & pl.col(number_label).is_null()
                ).alias(mask_label)
                for column, number_label, mask_label in zip(
                    columns,
                    number_labels,
                    mask_labels,
                    strict=True,
                )
            ]
        )
        has_unreadable = prepared.select(
            [pl.col(label).any().alias(label) for label in mask_labels]
        ).row(0)
        unreadable: list[str] = []
        for column, mask_label, has_any in zip(
            columns,
            mask_labels,
            has_unreadable,
            strict=True,
        ):
            if has_any:
                unreadable.extend(
                    self._unreadable_examples(
                        prepared.get_column(column),
                        prepared.get_column(mask_label),
                    )
                )
        if unreadable:
            logger.warning(
                f"layer {self.layer_name!r} declares plain numeric values; "
                f"{len(set(unreadable))} distinct unreadable token(s) became missing, "
                f"examples={sorted(set(unreadable))[:_EXAMPLE_LIMIT]}"
            )
        return prepared.select(
            [
                _masked(pl.col(label), self.missing_values).alias(column)
                for column, label in zip(columns, number_labels, strict=True)
            ]
        )

    @staticmethod
    def _temporary_labels(
        prefix: str,
        count: int,
        *,
        reserved: tuple[str, ...],
    ) -> tuple[str, ...]:
        """One collision-free temporary name per observation column."""
        taken = set(reserved)
        while True:
            labels = tuple(f"{prefix}_{index}" for index in range(count))
            if not taken.intersection(labels):
                return labels
            prefix += "_"

    @staticmethod
    def _unreadable_examples(values: pl.Series, invalid: pl.Series) -> list[str]:
        """Distinct unreadable tokens for the bounded warning emitted on the cold path."""
        failed = values.filter(invalid)
        return [str(token) for token in failed.unique(maintain_order=True).head(_EXAMPLE_LIMIT)]


@dataclass(frozen=True, slots=True)
class RegexNumericAnnDataEncoder:
    """One numeric capture per structured token; a token with no such structure is missing."""

    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumberNotation

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame:
        return values.select(
            [
                _masked(
                    as_numbers(
                        pl.col(column).cast(pl.String, strict=False).str.extract(self.pattern, 1),
                        pl.String(),
                        self.number_format,
                    ),
                    self.missing_values,
                ).alias(column)
                for column in values.columns
            ]
        )


@dataclass(frozen=True, slots=True)
class FactorAnnDataEncoder:
    """Declared category labels become their codes; a null or unknown label becomes ``-1``."""

    layer_name: str
    categories: tuple[tuple[str, int], ...]

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame:
        mapping = dict(self.categories)
        return values.select(
            [
                pl.col(column)
                .cast(pl.String, strict=False)
                .replace_strict(mapping, default=UNKNOWN_FACTOR_CODE, return_dtype=pl.Int64)
                .alias(column)
                for column in values.columns
            ]
        )


# -------------------------------------------------------------------------- contract checks


@dataclass(frozen=True, slots=True)
class OccupancyPolicy:
    """Which layers must exist, and how empty is too empty beside a populated sibling."""

    primary_layer_name: str
    required_names: tuple[str, ...]
    empty_ratio: float
    populated_ratio: float


def _occupancy(encoded: pl.DataFrame) -> float:
    """The share of cells that hold a usable number after encoding."""
    cells = encoded.height * encoded.width
    if not cells:
        return 0.0
    usable = sum(
        int((values.is_not_null() & ~values.is_nan().fill_null(value=True)).sum())
        if values.dtype.is_float()
        else int(values.is_not_null().sum())
        for values in encoded.get_columns()
    )
    return usable / cells


def _suspicious(
    encoded: Mapping[str, pl.DataFrame], policy: OccupancyPolicy
) -> tuple[tuple[str, ...], str]:
    """Which layers lost their quantities, and which populated sibling proves it.

    An effectively empty layer beside a populated one means the vendor column was read but
    its values did not survive parsing. Without a populated sibling, occupancy cannot tell
    that from an experiment with nothing in it, and does not invent the distinction.
    """
    ratios = {name: _occupancy(frame) for name, frame in encoded.items()}
    populated = [name for name, ratio in ratios.items() if ratio >= policy.populated_ratio]
    empty = tuple(name for name, ratio in ratios.items() if ratio < policy.empty_ratio)
    if not populated or not empty:
        return (), ""
    return empty, ", ".join(populated[:3])


def _require_declared_layers(encoded: Mapping[str, pl.DataFrame], policy: OccupancyPolicy) -> None:
    missing = [
        name for name in (policy.primary_layer_name, *policy.required_names) if name not in encoded
    ]
    if missing:
        raise AnnDataLayerContractError(
            f"the encoded layers are missing the required name(s) {missing}; present: "
            f"{sorted(encoded)}"
        )


def _contract_message(name: str, reference: str, ratio: float) -> str:
    return (
        f"layer {name!r} is effectively empty ({ratio:.2%}) while {reference} is populated — "
        "the source column was read but its values did not parse; check the vendor number "
        "format and the missing-value sentinels"
    )


@dataclass(frozen=True, slots=True)
class StandardAnnDataLayerContract:
    """The primary layer's emptiness makes the object unusable; another layer's warns."""

    policy: OccupancyPolicy

    def check(
        self,
        encoded: Mapping[str, pl.DataFrame],
        occupancy_candidates: Mapping[str, pl.DataFrame],
        /,
    ) -> None:
        _require_declared_layers(encoded, self.policy)
        empty, reference = _suspicious(occupancy_candidates, self.policy)
        for name in empty:
            message = _contract_message(name, reference, _occupancy(occupancy_candidates[name]))
            if name == self.policy.primary_layer_name:
                raise AnnDataLayerContractError(message)
            logger.warning(message)


@dataclass(frozen=True, slots=True)
class StrictAnnDataLayerContract:
    """Any retained layer that lost its quantities is an error, primary or not."""

    policy: OccupancyPolicy

    def check(
        self,
        encoded: Mapping[str, pl.DataFrame],
        occupancy_candidates: Mapping[str, pl.DataFrame],
        /,
    ) -> None:
        _require_declared_layers(encoded, self.policy)
        empty, reference = _suspicious(occupancy_candidates, self.policy)
        if empty:
            raise AnnDataLayerContractError(
                _contract_message(
                    empty[0],
                    reference,
                    _occupancy(occupancy_candidates[empty[0]]),
                )
            )


# ----------------------------------------------------------------------------- the adapter


@dataclass(frozen=True, slots=True)
class AnnDataWriter:
    """Encode, check, allocate, and write one parsed level as an ``.h5ad`` file."""

    encoders: Mapping[str, AnnDataLayerEncoder]
    contract: AnnDataLayerContractChecker

    def to_anndata(self, parsed: ParsedLevel, /) -> AnnData:
        """Encode and materialize one parsed level without writing it."""
        return self.to_anndata_for_level(parsed, _level_name(parsed), {}, {})

    def to_anndata_for_level(
        self,
        parsed: ParsedLevel,
        level_name: ParsedLevelName,
        shared_uns: Mapping[str, JsonValue],
        shared_metadata: Mapping[str, JsonValue],
    ) -> AnnData:
        validate_parsed_level(level_name, parsed)
        encoded = {
            name: self.encoders[name].encode(_layer_value_block(layer))
            for name, layer in parsed.layers.items()
        }
        occupancy_candidates: dict[str, pl.DataFrame] = {}
        for name, layer in parsed.layers.items():
            occupancy_candidates.update(layer.role.occupancy_candidates(name, encoded[name]))
        self.contract.check(encoded, occupancy_candidates)
        arrays = {
            name: frame.to_numpy().astype(np.float64, copy=False).T
            for name, frame in encoded.items()
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
            layers={layer_names[name]: values for name, values in arrays.items()},
        )
        self._write_aligned(parsed, adata, slot_names)
        self._write_pairwise(parsed, adata, slot_names)
        _write_namespaces(
            adata,
            parse=dict(parsed.uns),
            result=_level_result_metadata(
                parsed,
                level_name,
                shared_uns,
                shared_metadata,
                layer_names,
                slot_names,
            ),
            metadata={**shared_metadata, **parsed.metadata},
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
    """Materialize configured parsed levels as one shared-observation MuData file."""

    level_writers: Mapping[ParsedLevelName, AnnDataWriter]

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        if not parsed.levels:
            raise MuDataLevelError("no parsed levels supplied")
        parsed_names = set(parsed.levels)
        writer_names = set(self.level_writers)
        if parsed_names != writer_names:
            raise MuDataLevelError(
                "parsed levels and configured writers differ: "
                f"parsed={sorted(parsed_names)}, writers={sorted(writer_names)}"
            )

        modalities: dict[str, AnnData] = {}
        for level in LEVEL_ORDER:
            if level not in parsed.levels:
                continue
            adata = self.level_writers[level].to_anndata_for_level(
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
            parse=parsed.uns,
            result=_collection_result_metadata(parsed, annotation_names, relation_names),
            metadata=parsed.metadata,
        )
        _write_atomically(target, result.write_h5mu)


@dataclass(frozen=True, slots=True)
class H5adWriter:
    """Collection-level h5ad adapter configured from the stored resolved plan."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        if parsed.annotation_tables or parsed.feature_relations:
            raise MuDataLevelError("h5ad cannot store annotation tables or feature relations")
        if len(parsed.levels) != 1:
            raise MuDataLevelError(
                f"h5ad requires exactly one parsed level, got {list(parsed.levels)}"
            )
        level_name, level = next(iter(parsed.levels.items()))
        writer = _ann_data_writer_from_stored_plan(level)
        _write_atomically(
            target,
            writer.to_anndata_for_level(
                level,
                level_name,
                parsed.uns,
                parsed.metadata,
            ).write_h5ad,
        )


@dataclass(frozen=True, slots=True)
class H5muWriter:
    """Collection-level h5mu adapter configured from each level's stored plan."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None:
        validate_parsed_levels(parsed)
        writers: dict[ParsedLevelName, AnnDataWriter] = {
            name: _ann_data_writer_from_stored_plan(level) for name, level in parsed.levels.items()
        }
        MuDataWriter(level_writers=writers).write(parsed, target)


def quantitative_layer_values(parsed: ParsedLevel, layer_name: str, /) -> pl.DataFrame:
    """Project one stored layer to its quantitative values through the APB2 encoder.

    Columnar APB2 results deliberately retain vendor scalars. Downstream quantitative tools
    call this boundary instead of guessing how localized numbers, missing sentinels, regex
    captures, or factor encodings should be interpreted.

    Args:
        parsed: One validated APB2 level.
        layer_name: The logical layer to project.

    Returns:
        A Float64-compatible value block with one row per variable and one column per
        observation. Variable-key columns are not included.

    Raises:
        AnnDataPlanError: The layer is absent or its stored encoding cannot be reconstructed.
    """
    try:
        layer = parsed.layers[layer_name]
    except KeyError as error:
        raise AnnDataPlanError(f"level has no layer {layer_name!r}") from error
    writer = _ann_data_writer_from_stored_plan(parsed)
    return writer.encoders[layer_name].encode(_layer_value_block(layer))


def numeric_result_level(parsed: ParsedLevel, /) -> ParsedLevel:
    """Mark a level whose complete layer set is already numeric for future writers.

    Every value column must have a numeric or Null Polars dtype. The returned level owns a
    copied provenance mapping; the input level is not mutated. APB2 writers use the marker to
    apply plain numeric encoders instead of reconstructing vendor encodings.

    Args:
        parsed: Storage-neutral result level to validate and mark.

    Returns:
        A shallow replacement with copied provenance and APB2's numeric marker.

    Raises:
        AnnDataPlanError: Any layer contains a nonnumeric value column.
    """
    for layer in parsed.layers.values():
        _plain_numeric_encoder_for(layer)
    provenance = dict(parsed.uns)
    provenance[MATRIX_PROJECTED_KEY] = True
    return replace(parsed, uns=provenance)


def _level_name(parsed: ParsedLevel) -> ParsedLevelName:
    value = parsed.uns.get("quantification_level")
    if not isinstance(value, str) or value not in LEVEL_ORDER:
        raise AnnDataPlanError(
            "an AnnData write requires level provenance in uns['quantification_level']"
        )
    return value


def _level_result_metadata(
    parsed: ParsedLevel,
    level_name: ParsedLevelName,
    shared_uns: Mapping[str, JsonValue],
    shared_metadata: Mapping[str, JsonValue],
    layer_names: Mapping[str, str],
    slot_names: Mapping[str, Mapping[str, str]],
) -> dict[str, JsonValue]:
    layers = cast(
        dict[str, JsonValue],
        {
            name: {
                "physical_name": layer_names[name],
                "var_key_columns": list(layer.var_key_columns),
                "value_columns": list(layer.values.columns[len(layer.var_key_columns) :]),
                "role": layer.role.persisted_name(),
            }
            for name, layer in parsed.layers.items()
        },
    )
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "level": level_name,
        "shared_uns": dict(shared_uns),
        "shared_metadata": dict(shared_metadata),
        "level_metadata": dict(parsed.metadata),
        "primary_layer": parsed.primary_layer_name,
        "obs_key_columns": list(parsed.obs.key_columns),
        "var_key_columns": list(parsed.var.key_columns),
        "layer_order": list(parsed.layers),
        "layers": layers,
        "obsm_order": list(parsed.obsm),
        "obsm_physical_names": dict(slot_names["obsm"]),
        "varm_order": list(parsed.varm),
        "varm_physical_names": dict(slot_names["varm"]),
        "obsp_order": list(parsed.obsp),
        "obsp_physical_names": dict(slot_names["obsp"]),
        "varp_order": list(parsed.varp),
        "varp_physical_names": dict(slot_names["varp"]),
    }


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


def _collection_result_metadata(
    parsed: ParsedLevels,
    annotation_names: Mapping[str, str],
    relation_names: Mapping[str, str],
) -> dict[str, JsonValue]:
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "level_order": list(parsed.levels),
        "shared_uns": dict(parsed.uns),
        "annotation_table_order": list(parsed.annotation_tables),
        "annotation_tables": cast(
            dict[str, JsonValue],
            {
                name: {
                    **table_metadata(table.frame, annotation_names[name]),
                    "physical_name": annotation_names[name],
                    "key_columns": list(table.key_columns),
                    "metadata": dict(table.metadata),
                }
                for name, table in parsed.annotation_tables.items()
            },
        ),
        "feature_relation_order": list(parsed.feature_relations),
        "feature_relation_physical_names": dict(relation_names),
        "feature_relations": cast(
            dict[str, JsonValue],
            {
                name: {
                    "annotation_table": relation.annotation_table,
                    "target_level": relation.target_level,
                    "metadata": dict(relation.metadata),
                }
                for name, relation in parsed.feature_relations.items()
            },
        ),
    }


def _ann_data_writer_from_stored_plan(parsed: ParsedLevel) -> AnnDataWriter:
    raw_plan = parsed.uns.get("plan_json")
    projected = parsed.uns.get(MATRIX_PROJECTED_KEY) is True
    if not isinstance(raw_plan, str) and not projected:
        raise AnnDataPlanError("AnnData output requires parse provenance key 'plan_json'")
    if not isinstance(raw_plan, str):
        encoders = {
            name: _plain_numeric_encoder_for(layer) for name, layer in parsed.layers.items()
        }
        return AnnDataWriter(
            encoders=encoders,
            contract=StandardAnnDataLayerContract(_derived_occupancy_policy(parsed)),
        )
    try:
        plan = object_mapping(json.loads(raw_plan), "stored resolved plan")
    except json.JSONDecodeError as error:
        raise AnnDataPlanError(f"stored plan_json is invalid JSON: {error}") from error
    ann_data = object_mapping(plan.get("ann_data"), "stored AnnData plan")
    raw_encodings = ann_data.get("layer_encodings")
    if not isinstance(raw_encodings, list):
        raise AnnDataPlanError("stored AnnData plan has no layer_encodings list")
    encoders: dict[str, AnnDataLayerEncoder] = {}
    if projected:
        encoders = {
            name: _plain_numeric_encoder_for(layer) for name, layer in parsed.layers.items()
        }
    else:
        for raw_encoding in raw_encodings:
            encoding = object_mapping(raw_encoding, "stored layer encoding")
            name = string_value(encoding.get("layer_name"), "stored layer name")
            encoders[name] = _encoder_from_stored_config(name, encoding)
    contract = object_mapping(ann_data.get("layer_contract"), "stored layer contract")
    policy = OccupancyPolicy(
        primary_layer_name=string_value(contract.get("primary_layer_name"), "stored primary layer"),
        required_names=tuple(string_list(contract.get("required_names"), "stored required layers")),
        empty_ratio=_float_value(contract.get("empty_ratio"), "stored empty ratio"),
        populated_ratio=_float_value(contract.get("populated_ratio"), "stored populated ratio"),
    )
    for name, layer in parsed.layers.items():
        if name not in encoders:
            encoders[name] = _plain_numeric_encoder_for(layer)
    return AnnDataWriter(
        encoders={name: encoders[name] for name in parsed.layers},
        contract=StandardAnnDataLayerContract(policy),
    )


def _plain_numeric_encoder_for(layer: FinalLayerTable) -> PlainNumericAnnDataEncoder:
    values = _layer_value_block(layer)
    nonnumeric = [
        name for name, dtype in values.schema.items() if dtype != pl.Null and not dtype.is_numeric()
    ]
    if nonnumeric:
        raise AnnDataPlanError(
            f"unplanned layer {layer.layer_name!r} is not already numeric in column(s) {nonnumeric}"
        )
    return PlainNumericAnnDataEncoder(
        layer_name=layer.layer_name,
        missing_values=(),
        number_format=NumberNotation(decimal_mark=".", thousands_marks=()),
    )


def _derived_occupancy_policy(parsed: ParsedLevel) -> OccupancyPolicy:
    return OccupancyPolicy(
        primary_layer_name=parsed.primary_layer_name,
        required_names=(parsed.primary_layer_name,),
        empty_ratio=_DERIVED_EMPTY_RATIO,
        populated_ratio=_DERIVED_POPULATED_RATIO,
    )


def _encoder_from_stored_config(
    layer_name: str,
    config: Mapping[str, object],
) -> AnnDataLayerEncoder:
    kind = string_value(config.get("kind"), f"encoding kind for {layer_name!r}")
    if kind == "factor":
        raw_categories = config.get("categories")
        if not isinstance(raw_categories, list):
            raise AnnDataPlanError(f"factor layer {layer_name!r} has no categories")
        categories: list[tuple[str, int]] = []
        for raw_category in raw_categories:
            if (
                not isinstance(raw_category, list)
                or len(raw_category) != 2
                or not isinstance(raw_category[0], str)
                or not isinstance(raw_category[1], int)
            ):
                raise AnnDataPlanError(f"factor layer {layer_name!r} has an invalid category")
            categories.append((raw_category[0], raw_category[1]))
        return FactorAnnDataEncoder(layer_name=layer_name, categories=tuple(categories))
    missing_values = tuple(
        _float_value(value, f"missing value for {layer_name!r}")
        for value in _value_list(config.get("missing_values"), f"missing values for {layer_name!r}")
    )
    notation = _notation_from_stored_config(
        object_mapping(config.get("number_format"), f"number format for {layer_name!r}")
    )
    if kind == "regex_numeric":
        return RegexNumericAnnDataEncoder(
            layer_name=layer_name,
            missing_values=missing_values,
            pattern=string_value(config.get("pattern"), f"pattern for {layer_name!r}"),
            number_format=notation,
        )
    if kind == "plain_numeric":
        return PlainNumericAnnDataEncoder(
            layer_name=layer_name,
            missing_values=missing_values,
            number_format=notation,
        )
    raise AnnDataPlanError(f"unsupported stored layer encoding kind {kind!r}")


def _notation_from_stored_config(config: Mapping[str, object]) -> NumberNotation:
    return NumberNotation(
        decimal_mark=string_value(config.get("decimal_mark"), "stored decimal mark"),
        thousands_marks=tuple(string_list(config.get("thousands_marks"), "stored thousands marks")),
    )


def _value_list(value: object, role: str) -> list[object]:
    if not isinstance(value, list):
        raise AnnDataPlanError(f"{role} is not a list")
    return cast(list[object], value)


def _float_value(value: object, role: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise AnnDataPlanError(f"{role} is not numeric")
    return float(value)


def _write_namespaces(
    target: AnnData | MuData,
    *,
    parse: Mapping[str, JsonValue],
    result: Mapping[str, JsonValue],
    metadata: Mapping[str, JsonValue],
) -> None:
    """Store independent parse, result, and extension sections under APB's namespace."""
    collisions = {PARSE_NAMESPACE, RESULT_NAMESPACE}.intersection(metadata)
    if collisions:
        raise AnnDataPlanError(f"APB extension metadata uses reserved section(s) {collisions}")
    target.uns[NAMESPACE] = {
        PARSE_NAMESPACE: dict(parse),
        RESULT_NAMESPACE: json.dumps(result, ensure_ascii=False, allow_nan=False),
        **dict(metadata),
    }


def _write_atomically(target: Path, write: Callable[[Path], None]) -> None:
    """Write beside the destination and replace it only after a complete write."""
    with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
        staged = Path(scratch) / target.name
        write(staged)
        staged.replace(target)
