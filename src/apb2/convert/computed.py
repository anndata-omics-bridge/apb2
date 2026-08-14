"""How one computed-column declaration becomes a column of values.

``ComputedColumn``'s six modes used to be enumerated in three modules across two packages: a
validator in ``vendor_quant_rules/schema/components.py`` rejecting field combinations, a four-arm
per-mode validator in ``vendor_quant_rules/schema/parse_rule.py``, and a five-arm computation in
``converters/assemble.py``. Adding a seventh mode meant editing all three, with nothing
linking them.

Six document types map to **five** computers here, and the mismatch is the point. A factory
is a mapping, not a mirror: ``stripped_sequence`` and ``proforma_sequence`` differ only in
which APB-derived column they read, which is a difference in the file format and not in the
computation.

What does *not* move to construction time is the source-availability check. ``allow_missing``
describes one input file's absent ``optional_select`` columns, so it cannot be resolved when
a rule is built — that branch asks what happened, and stays inside ``compute``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from apb2.vendor_parse_rules.model import (
    Coalesce,
    ComputedColumn,
    JoinNonempty,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
)


def _require_present(name: str, keys: tuple[str, ...], df: pd.DataFrame) -> None:
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"cannot compute column {name!r}; source column(s) missing: {missing}")


@dataclass(frozen=True, slots=True)
class CoalesceColumn:
    """Take the first non-null source value in declaration order."""

    name: str
    sources: tuple[str, ...]

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        present = _present_sources(self.name, self.sources, df, allow_missing)
        result = df[present[0]].astype(object).copy()
        for source in present[1:]:
            result = result.where(result.notna(), df[source].astype(object))
        return result


@dataclass(frozen=True, slots=True)
class JoinNonEmptyColumn:
    """Join the non-empty source values with a separator."""

    name: str
    sources: tuple[str, ...]
    separator: str

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        present = _present_sources(self.name, self.sources, df, allow_missing)
        result = pd.Series("", index=df.index, dtype="string")
        has_value = pd.Series(False, index=df.index)
        for source in present:
            values = df[source].astype("string")
            valid = values.notna() & values.ne("")
            append = has_value & valid
            result = result.mask(append, result + self.separator + values)
            first = ~has_value & valid
            result = result.mask(first, values)
            has_value |= valid
        return result.mask(~has_value, pd.NA)


@dataclass(frozen=True, slots=True)
class DerivedSequenceColumn:
    """Expose a column APB derived earlier in the pipeline under its declared name."""

    name: str
    source_key: str

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        if self.source_key not in df.columns:
            raise ValueError(
                f"cannot compute column {self.name!r}; APB column {self.source_key!r} is missing"
            )
        return df[self.source_key]


@dataclass(frozen=True, slots=True)
class ProformaIonColumn:
    """Combine a string peptidoform and an already-typed positive integer charge."""

    name: str
    sequence_key: str
    charge_key: str

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        _require_present(self.name, (self.sequence_key, self.charge_key), df)
        charges = df[self.charge_key]
        if charges.isna().any():
            raise ValueError("cannot derive proforma_ion from missing charge")
        nonpositive = charges.le(0)
        if nonpositive.any():
            examples = charges.loc[nonpositive].drop_duplicates().head(5).tolist()
            raise ValueError(f"charge must be positive; examples={examples}")
        sequences = df[self.sequence_key].astype("string")
        return sequences + "/" + charges.astype("string")


@dataclass(frozen=True, slots=True)
class ProformaFragmentColumn:
    """Combine a ProForma ion and a fragment label."""

    name: str
    ion_key: str
    label_key: str

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        _require_present(self.name, (self.ion_key, self.label_key), df)
        return pd.Series(
            [
                f"{ion}/{label}"
                for ion, label in zip(df[self.ion_key], df[self.label_key], strict=True)
            ],
            index=df.index,
        )


type ColumnComputer = (
    CoalesceColumn
    | JoinNonEmptyColumn
    | DerivedSequenceColumn
    | ProformaIonColumn
    | ProformaFragmentColumn
)


def _present_sources(
    name: str,
    sources: tuple[str, ...],
    df: pd.DataFrame,
    allow_missing: frozenset[str],
) -> tuple[str, ...]:
    """Drop skipped optional sources, and reject anything else that is absent.

    This is the check that cannot move to construction time: ``allow_missing`` is derived
    from the input file being converted, not from the rule.
    """
    present = tuple(key for key in sources if key not in allow_missing)
    _require_present(name, present, df)
    if not present:
        raise ValueError(
            f"cannot compute column {name!r}; every source column is an "
            f"optional_select absent from this input: {list(sources)}"
        )
    return present


def make_computer(column: ComputedColumn) -> ColumnComputer:
    """Read a computed-column declaration's mode once, and return the computer it names."""
    sources = tuple(column.inputs)
    if isinstance(column, Coalesce):
        return CoalesceColumn(column.name, sources)
    if isinstance(column, JoinNonempty):
        return JoinNonEmptyColumn(column.name, sources, column.separator)
    if isinstance(column, StrippedSequence | ProformaSequence):
        return DerivedSequenceColumn(column.name, column.how)
    if isinstance(column, ProformaIon):
        return ProformaIonColumn(column.name, sources[0], sources[1])
    return ProformaFragmentColumn(column.name, sources[0], sources[1])
