"""Axis leaf algorithms: coerce one selected series, or compute one declared column.

Everything here runs on a small axis frame — one row per distinct raw key — never on the
source table, which is why a per-value check can afford to report the tokens that failed.

Each class is configured once and asks nothing afterwards. A coercer knows one logical type;
a computer knows its output name and its exact ordered inputs and receives exactly those
series. Neither reads a rule, a ``how``, or an optionality flag: source resolution pruned the
operations this file cannot run before any of these objects existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_EXAMPLE_LIMIT = 5
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_BOOLEAN_SPELLINGS = {
    "false": False,
    "true": True,
    "0": False,
    "0.0": False,
    "1": True,
    "1.0": True,
}


class AxisCoercionError(ValueError):
    """One selected axis column holds values its declared logical type cannot read."""


def _require_valid(values: pl.Series, invalid: pl.Series, name: str, source: str) -> None:
    """Report the tokens one coercion could not read, bounded and with examples."""
    count = int(invalid.sum())
    if not count:
        return
    examples = (
        values.filter(invalid).cast(pl.String).unique(maintain_order=True).head(_EXAMPLE_LIMIT)
    )
    raise AxisCoercionError(
        f"cannot convert column {name!r} from vendor source {source!r}: "
        f"{count} invalid non-missing value(s); examples={examples.to_list()}"
    )


class StringAxisCoercer:
    """Keep identifier text exactly as the vendor wrote it."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        del name, source
        return values.cast(pl.String)


class NumberAxisCoercer:
    """Read floating-point values, rejecting every invalid non-missing token."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        parsed = values.cast(pl.Float64, strict=False)
        invalid = values.is_not_null() & ~parsed.is_finite().fill_null(value=False)
        _require_valid(values, invalid, name, source)
        return parsed


class IntegerAxisCoercer:
    """Read integers, rejecting fractions and values outside the 64-bit range."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        parsed = values.cast(pl.Float64, strict=False)
        finite = parsed.is_finite().fill_null(value=False)
        integral = (parsed % 1 == 0).fill_null(value=False)
        in_range = ((parsed >= _INT64_MIN) & (parsed <= _INT64_MAX)).fill_null(value=False)
        invalid = values.is_not_null() & ~(finite & integral & in_range)
        _require_valid(values, invalid, name, source)
        return parsed.cast(pl.Int64, strict=False)


class BooleanAxisCoercer:
    """Read the exact canonical boolean spellings, and nothing else."""

    def coerce(self, values: pl.Series, *, name: str, source: str) -> pl.Series:
        parsed = (
            values.cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .replace_strict(
                _BOOLEAN_SPELLINGS,
                default=None,
                return_dtype=pl.Boolean,
            )
        )
        invalid = values.is_not_null() & parsed.is_null()
        _require_valid(values, invalid, name, source)
        return parsed


class ColumnComputationError(ValueError):
    """One computed axis column cannot be materialized from the series it received."""


def _require_arity(name: str, inputs: tuple[str, ...], columns: tuple[pl.Series, ...]) -> None:
    """A computer receives exactly its configured inputs, in order — or it refuses to run."""
    if len(columns) != len(inputs):
        raise ColumnComputationError(
            f"computed column {name!r} declares inputs {list(inputs)} but received "
            f"{len(columns)} series"
        )


@dataclass(frozen=True, slots=True)
class CoalesceColumn:
    """Take the first non-null input value in declaration order."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, columns: tuple[pl.Series, ...], /) -> pl.Series:
        _require_arity(self.name, self.inputs, columns)
        result = columns[0].cast(pl.String)
        for values in columns[1:]:
            result = result.zip_with(result.is_not_null(), values.cast(pl.String))
        return result


@dataclass(frozen=True, slots=True)
class JoinNonemptyColumn:
    """Join the non-empty input values with a separator; nothing present stays null."""

    name: str
    inputs: tuple[str, ...]
    separator: str

    def compute(self, columns: tuple[pl.Series, ...], /) -> pl.Series:
        _require_arity(self.name, self.inputs, columns)
        frame = pl.DataFrame(
            [values.cast(pl.String).alias(f"_{index}") for index, values in enumerate(columns)]
        )
        present = [
            pl.when(pl.col(f"_{index}").is_not_null() & (pl.col(f"_{index}") != ""))
            .then(pl.col(f"_{index}"))
            .otherwise(None)
            for index in range(len(columns))
        ]
        joined = pl.concat_str(present, separator=self.separator, ignore_nulls=True)
        empty = pl.all_horizontal(
            [
                pl.col(f"_{index}").is_null() | (pl.col(f"_{index}") == "")
                for index in range(len(columns))
            ]
        )
        return frame.select(
            pl.when(empty).then(None).otherwise(joined).alias(self.name)
        ).to_series()


@dataclass(frozen=True, slots=True)
class DerivedSequenceColumn:
    """Expose, under its declared name, a column modification normalization already derived.

    One class for the stripped peptide and for the ProForma peptidoform: the difference is
    which derived column the configuration points at, and that is data, not behaviour.
    """

    name: str
    inputs: tuple[str, ...]

    def compute(self, columns: tuple[pl.Series, ...], /) -> pl.Series:
        _require_arity(self.name, self.inputs, columns)
        return columns[0].cast(pl.String)


@dataclass(frozen=True, slots=True)
class ProformaIonColumn:
    """Combine a peptidoform with a positive integer charge."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, columns: tuple[pl.Series, ...], /) -> pl.Series:
        _require_arity(self.name, self.inputs, columns)
        sequences, charges = columns
        if charges.is_null().any():
            raise ColumnComputationError(f"cannot derive {self.name!r} from a missing charge")
        nonpositive = charges <= 0
        if nonpositive.any():
            examples = charges.filter(nonpositive).unique(maintain_order=True).head(_EXAMPLE_LIMIT)
            raise ColumnComputationError(
                f"cannot derive {self.name!r}: charge must be positive; "
                f"examples={examples.to_list()}"
            )
        return sequences.cast(pl.String) + "/" + charges.cast(pl.String)


@dataclass(frozen=True, slots=True)
class ProformaFragmentColumn:
    """Combine a ProForma ion with a fragment label."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, columns: tuple[pl.Series, ...], /) -> pl.Series:
        _require_arity(self.name, self.inputs, columns)
        ion, label = columns
        return ion.cast(pl.String) + "/" + label.cast(pl.String)
