"""The AnnData boundary: the only place a parsed value stops being what the vendor wrote.

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
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

import numpy as np
import pandas as pd
import polars as pl
from anndata import AnnData
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, JsonValue, ParsedLevel
from apb2.parserV2.parse_quant.numeric_text import NumberNotation, as_numbers, blank

NAMESPACE = "anndata_proteomics"
"""The uns namespace apb writes; unchanged while the legacy package is the parity oracle."""

KEY_SEPARATOR = "_"
UNKNOWN_FACTOR_CODE = -1
_EXAMPLE_LIMIT = 5


class AnnDataLayerContractError(ValueError):
    """An encoded layer carries too few values to be a usable quantitative layer."""


class AnnDataLayerEncoder(Protocol):
    """Encode one layer's value block for AnnData, preserving its shape and column order."""

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame: ...


class AnnDataLayerContractChecker(Protocol):
    """Enforce the required-layer and occupancy policy on the encoded layers."""

    def check(self, encoded: Mapping[str, pl.DataFrame], /) -> None: ...


# --------------------------------------------------------------------------------- encoders


def _unreadable_examples(values: pl.Series, numbers: pl.Series) -> list[str]:
    """Non-blank tokens this notation could not read, for the line that reports them."""
    failed = values.filter(~blank(values) & numbers.is_null())
    return [str(token) for token in failed.unique(maintain_order=True).head(_EXAMPLE_LIMIT)]


def _masked(numbers: pl.Series, missing_values: tuple[float, ...]) -> pl.Series:
    """Blank out the values the vendor writes to mean "not measured"."""
    if not missing_values:
        return numbers
    return pl.select(
        pl.when(numbers.is_in(list(missing_values))).then(None).otherwise(numbers)
    ).to_series()


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
        encoded: list[pl.Series] = []
        unreadable: list[str] = []
        for column in values.columns:
            raw = values.get_column(column)
            numbers = as_numbers(raw, self.number_format)
            unreadable.extend(_unreadable_examples(raw, numbers))
            encoded.append(_masked(numbers, self.missing_values).rename(column))
        if unreadable:
            logger.warning(
                f"layer {self.layer_name!r} declares plain numeric values; "
                f"{len(set(unreadable))} distinct unreadable token(s) became missing, "
                f"examples={sorted(set(unreadable))[:_EXAMPLE_LIMIT]}"
            )
        return pl.DataFrame(encoded)


@dataclass(frozen=True, slots=True)
class RegexNumericAnnDataEncoder:
    """One numeric capture per structured token; a token with no such structure is missing."""

    layer_name: str
    missing_values: tuple[float, ...]
    pattern: str
    number_format: NumberNotation

    def encode(self, values: pl.DataFrame, /) -> pl.DataFrame:
        return pl.DataFrame(
            [
                _masked(
                    as_numbers(
                        values.get_column(column)
                        .cast(pl.String, strict=False)
                        .str.extract(self.pattern, 1),
                        self.number_format,
                    ),
                    self.missing_values,
                ).rename(column)
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

    def check(self, encoded: Mapping[str, pl.DataFrame], /) -> None:
        _require_declared_layers(encoded, self.policy)
        empty, reference = _suspicious(encoded, self.policy)
        for name in empty:
            message = _contract_message(name, reference, _occupancy(encoded[name]))
            if name == self.policy.primary_layer_name:
                raise AnnDataLayerContractError(message)
            logger.warning(message)


@dataclass(frozen=True, slots=True)
class StrictAnnDataLayerContract:
    """Any retained layer that lost its quantities is an error, primary or not."""

    policy: OccupancyPolicy

    def check(self, encoded: Mapping[str, pl.DataFrame], /) -> None:
        _require_declared_layers(encoded, self.policy)
        empty, reference = _suspicious(encoded, self.policy)
        if empty:
            raise AnnDataLayerContractError(
                _contract_message(empty[0], reference, _occupancy(encoded[empty[0]]))
            )


# ----------------------------------------------------------------------------- the adapter


@dataclass(frozen=True, slots=True)
class AnnDataWriter:
    """Encode, check, allocate, and write one parsed level as an ``.h5ad`` file."""

    encoders: Mapping[str, AnnDataLayerEncoder]
    contract: AnnDataLayerContractChecker

    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        encoded = {
            name: self.encoders[name].encode(self._value_block(layer))
            for name, layer in parsed.layers.items()
        }
        self.contract.check(encoded)
        arrays = {
            name: frame.to_numpy().astype(np.float64, copy=False).T
            for name, frame in encoded.items()
        }
        adata = AnnData(
            X=arrays[parsed.primary_layer_name],
            obs=self._make_axis_frame(parsed.obs.frame, parsed.obs.key_columns),
            var=self._make_axis_frame(parsed.var.frame, parsed.var.key_columns),
            layers=arrays,
        )
        self._write_namespace(adata, dict(parsed.uns))
        self._write_atomically(target, adata.write_h5ad)

    @staticmethod
    def _value_block(layer: FinalLayerTable) -> pl.DataFrame:
        """One layer's observation columns; its leading var keys are identity, not values."""
        return layer.values.select(layer.values.columns[len(layer.var_key_columns) :])

    @staticmethod
    def _make_axis_frame(frame: pl.DataFrame, key_columns: tuple[str, ...]) -> pd.DataFrame:
        """Convert one axis to pandas, keeping every authored key as an ordinary column.

        The per-dtype cases below are a translation table, not a decision: AnnData and HDF5
        accept a specific set of representations, and this is the one place that knows which.
        """
        columns = {
            name: AnnDataWriter._pandas_column(frame.get_column(name)) for name in frame.columns
        }
        table = pd.DataFrame(columns)
        table.index = AnnDataWriter._storage_index(frame, columns, key_columns)
        return table

    @staticmethod
    def _pandas_column(values: pl.Series) -> pd.Series:
        dtype = values.dtype
        if dtype == pl.Boolean:
            return pd.Series(values.to_list(), dtype="boolean")
        if dtype.is_integer():
            return pd.Series(values.to_list(), dtype="Int64")
        if dtype.is_float():
            return pd.Series(values.to_numpy(), dtype="float64")
        return pd.Series(values.to_list(), dtype="string")

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
        if len(key_columns) == 1 and frame.schema[key_columns[0]] == pl.String:
            return pd.Index(columns[key_columns[0]], name=KEY_SEPARATOR.join(key_columns))
        # A derived index holds different values from the columns it was built from, so it
        # must not borrow one of their names.
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

    @staticmethod
    def _write_namespace(adata: AnnData, namespace: dict[str, JsonValue]) -> None:
        """Store the parse provenance where every apb reader already looks for it."""
        adata.uns[NAMESPACE] = namespace

    @staticmethod
    def _write_atomically(target: Path, write: Callable[[Path], None]) -> None:
        """Write beside the destination and replace it only after a complete write."""
        with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
            staged = Path(scratch) / target.name
            write(staged)
            staged.replace(target)
