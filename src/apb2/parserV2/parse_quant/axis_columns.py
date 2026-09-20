"""Native Polars expressions for declared axis selections and computations."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.data.numeric_text import NumberNotation, as_numbers
from apb2.parserV2.parse_quant.errors import ColumnComputationError

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


def _require_valid(frame: pl.DataFrame, invalid: pl.Expr, name: str, source: str) -> None:
    """Report the tokens one coercion could not read, bounded and with examples."""
    count = frame.select(invalid.sum()).item()
    if not count:
        return
    examples = (
        frame.select(
            pl.col(source)
            .filter(invalid)
            .cast(pl.String)
            .unique(maintain_order=True)
            .head(_EXAMPLE_LIMIT)
        )
        .to_series()
        .to_list()
    )
    raise AxisCoercionError(
        f"cannot convert column {name!r} from vendor source {source!r}: "
        f"{count} invalid non-missing value(s); examples={examples}"
    )


class StringAxisCoercer:
    """Keep identifier text exactly as the vendor wrote it."""

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        return pl.col(source).cast(pl.String).alias(name)


@dataclass(frozen=True, slots=True)
class NumberAxisCoercer:
    """Read floating-point values, rejecting every invalid non-missing token."""

    notation: NumberNotation

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        parsed = as_numbers(pl.col(source), frame.schema[source], self.notation)
        invalid = pl.col(source).is_not_null() & ~parsed.is_finite().fill_null(value=False)
        _require_valid(frame, invalid, name, source)
        return parsed.alias(name)


@dataclass(frozen=True, slots=True)
class IntegerAxisCoercer:
    """Read integers, rejecting fractions and values outside the 64-bit range."""

    notation: NumberNotation

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        parsed = as_numbers(pl.col(source), frame.schema[source], self.notation)
        finite = parsed.is_finite().fill_null(value=False)
        integral = (parsed % 1 == 0).fill_null(value=False)
        in_range = ((parsed >= _INT64_MIN) & (parsed <= _INT64_MAX)).fill_null(value=False)
        invalid = pl.col(source).is_not_null() & ~(finite & integral & in_range)
        _require_valid(frame, invalid, name, source)
        return parsed.cast(pl.Int64, strict=False).alias(name)


class BooleanAxisCoercer:
    """Read the exact canonical boolean spellings, and nothing else."""

    def coerce(self, frame: pl.DataFrame, *, name: str, source: str) -> pl.Expr:
        parsed = (
            pl.col(source)
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .replace_strict(
                _BOOLEAN_SPELLINGS,
                default=None,
                return_dtype=pl.Boolean,
            )
        )
        invalid = pl.col(source).is_not_null() & parsed.is_null()
        _require_valid(frame, invalid, name, source)
        return parsed.alias(name)


@dataclass(frozen=True, slots=True)
class CoalesceColumn:
    """Take the first non-null input value in declaration order."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
        expression = pl.coalesce(pl.col(self.inputs).cast(pl.String)).alias(self.name)
        return frame.with_columns(expression), ()


@dataclass(frozen=True, slots=True)
class JoinNonemptyColumn:
    """Join the non-empty input values with a separator; nothing present stays null."""

    name: str
    inputs: tuple[str, ...]
    separator: str

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
        present = pl.col(self.inputs).cast(pl.String).replace("", None)
        joined = pl.concat_str(present, separator=self.separator, ignore_nulls=True)
        expression = pl.when(pl.all_horizontal(present.is_null())).then(None).otherwise(joined)
        return frame.with_columns(expression.alias(self.name)), ()


@dataclass(frozen=True, slots=True)
class ProformaIonColumn:
    """Combine a peptidoform with a positive integer charge."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
        sequences, charges = (pl.col(name) for name in self.inputs)
        if frame.select(charges.is_null().any()).item():
            raise ColumnComputationError(f"cannot derive {self.name!r} from a missing charge")
        nonpositive = charges <= 0
        examples = frame.select(
            charges.filter(nonpositive).unique(maintain_order=True).head(_EXAMPLE_LIMIT)
        )
        if examples.height:
            raise ColumnComputationError(
                f"cannot derive {self.name!r}: charge must be positive; "
                f"examples={examples.to_series().to_list()}"
            )
        return frame.with_columns(
            pl.concat_str(sequences, charges, separator="/").alias(self.name)
        ), ()


@dataclass(frozen=True, slots=True)
class ProformaFragmentColumn:
    """Combine a ProForma ion with a fragment label."""

    name: str
    inputs: tuple[str, ...]

    def compute(self, frame: pl.DataFrame, /) -> tuple[pl.DataFrame, tuple[str, ...]]:
        return frame.with_columns(pl.concat_str(self.inputs, separator="/").alias(self.name)), ()
