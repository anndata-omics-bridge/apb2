"""The ``columns`` block: select, optional_select, types, and computed columns.

Materialization is split around the pivot, because cost lives on the flat table: the
axis-key closure (the key columns plus everything they are computed from) is prepared on
the flat frame — the pivot cannot group without it — while every remaining declared
column is materialized afterwards on the deduplicated obs/var frames, where a column fix
touches nrObs or nrVars rows instead of nrObs x nrVars.

``computer_for(column)`` reads the ``how`` selector once; each computer is constructed
from its own declaration variant and fails at construction when it cannot be built.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from apb2.modifications.pipeline import ModificationApplier
from apb2.result import ParsedData
from apb2.vendor_parse_rules.model import (
    AxisColumnType,
    Coalesce,
    ColumnGroup,
    ComputedColumn,
    JoinNonempty,
    LongRule,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    StrippedSequence,
    WideRule,
)
from apb2.vendor_parse_rules.runtime import Recognition, group_names

# ------------------------------------------------------------- logical axis-column typing


def _raise_invalid(values: pd.Series, invalid: pd.Series, name: str, source: str) -> None:
    """Raise one bounded, contextual error when a coercion mask contains failures."""
    count = int(invalid.sum())
    if not count:
        return
    examples = values.loc[invalid].astype("string").drop_duplicates().head(5).tolist()
    raise ValueError(
        f"cannot convert column {name!r} from vendor source {source!r}: "
        f"{count} invalid non-missing value(s); examples={examples}"
    )


def _coerce_string(values: pd.Series, name: str, source: str) -> pd.Series:
    """Return nullable strings without changing identifier text."""
    del name, source
    return values.astype("string")


def _coerce_number(values: pd.Series, name: str, source: str) -> pd.Series:
    """Return float64 values and reject every invalid non-missing token."""
    parsed = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = pd.Series(np.isfinite(parsed.to_numpy()), index=values.index)
    invalid = values.notna() & (parsed.isna() | ~finite)
    _raise_invalid(values, invalid, name, source)
    return parsed


def _coerce_integer(values: pd.Series, name: str, source: str) -> pd.Series:
    """Return nullable Int64 values after exact integrality and range validation."""
    parsed = pd.to_numeric(values, errors="coerce")
    present = parsed.notna()
    invalid = values.notna() & ~present
    invalid |= (present & parsed.mod(1).ne(0)).fillna(False)
    limits = np.iinfo(np.int64)
    invalid |= (present & (parsed.lt(limits.min) | parsed.gt(limits.max))).fillna(False)
    _raise_invalid(values, invalid, name, source)
    return parsed.astype("Int64")


def _coerce_boolean(values: pd.Series, name: str, source: str) -> pd.Series:
    """Return nullable booleans from the exact canonical boolean spellings."""
    normalized = values.astype("string").str.strip().str.lower()
    parsed = normalized.map(
        {"false": False, "true": True, "0": False, "0.0": False, "1": True, "1.0": True}
    )
    invalid = values.notna() & parsed.isna()
    _raise_invalid(values, invalid, name, source)
    return parsed.astype("boolean")


type _AxisCoercer = Callable[[pd.Series, str, str], pd.Series]

_AXIS_COERCERS: Mapping[AxisColumnType, _AxisCoercer] = {
    "string": _coerce_string,
    "integer": _coerce_integer,
    "number": _coerce_number,
    "boolean": _coerce_boolean,
}


def _coerce_axis_column(
    values: pd.Series, logical_type: AxisColumnType, name: str, source: str
) -> pd.Series:
    return _AXIS_COERCERS[logical_type](values, name, source)


# ------------------------------------------------------------------------------ computers


def _require_present(name: str, keys: tuple[str, ...], df: pd.DataFrame) -> None:
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"cannot compute column {name!r}; source column(s) missing: {missing}")


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


class CoalesceColumn:
    """Take the first non-null source value in declaration order."""

    def __init__(self, column: Coalesce) -> None:
        self.name = column.name
        self.sources = tuple(column.inputs)

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        present = _present_sources(self.name, self.sources, df, allow_missing)
        result = df[present[0]].astype(object).copy()
        for source in present[1:]:
            result = result.where(result.notna(), df[source].astype(object))
        return result


class JoinNonEmptyColumn:
    """Join the non-empty source values with a separator."""

    def __init__(self, column: JoinNonempty) -> None:
        self.name = column.name
        self.sources = tuple(column.inputs)
        self.separator = column.separator

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


class DerivedSequenceColumn:
    """Expose a column apb2 derived earlier in the pipeline under its declared name."""

    def __init__(self, column: StrippedSequence | ProformaSequence) -> None:
        self.name = column.name
        self.source_key = column.how

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        del allow_missing
        if self.source_key not in df.columns:
            raise ValueError(
                f"cannot compute column {self.name!r}; apb2 column {self.source_key!r} is missing"
            )
        return df[self.source_key]


class ProformaIonColumn:
    """Combine a string peptidoform and an already-typed positive integer charge."""

    def __init__(self, column: ProformaIon) -> None:
        self.name = column.name
        self.sequence_key, self.charge_key = column.inputs

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        del allow_missing
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


class ProformaFragmentColumn:
    """Combine a ProForma ion and a fragment label."""

    def __init__(self, column: ProformaFragment) -> None:
        self.name = column.name
        self.ion_key, self.label_key = column.inputs

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        del allow_missing
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


def computer_for(column: ComputedColumn) -> ColumnComputer:
    """Read a computed-column declaration's ``how`` once; return the computer it names."""
    if isinstance(column, Coalesce):
        return CoalesceColumn(column)
    if isinstance(column, JoinNonempty):
        return JoinNonEmptyColumn(column)
    if isinstance(column, StrippedSequence | ProformaSequence):
        return DerivedSequenceColumn(column)
    if isinstance(column, ProformaIon):
        return ProformaIonColumn(column)
    return ProformaFragmentColumn(column)


# --------------------------------------------------------------- split materialization


class _GroupMaterialization:
    """One axis group's materialization, split into the key closure and the rest."""

    def __init__(self, group: ColumnGroup, keys: list[str]) -> None:
        self.group = group
        self.declared = group_names(group)
        closure = self._key_closure(group, keys)
        self.key_select = [(n, s) for n, s in group.select.items() if n in closure]
        self.key_optional = [(n, s) for n, s in group.optional_select.items() if n in closure]
        self.key_computers = [computer_for(c) for c in group.computed if c.name in closure]
        self.rest_select = [(n, s) for n, s in group.select.items() if n not in closure]
        self.rest_optional = [(n, s) for n, s in group.optional_select.items() if n not in closure]
        self.rest_computers = [computer_for(c) for c in group.computed if c.name not in closure]
        self.keys_need_modifications = any(
            isinstance(c, DerivedSequenceColumn) for c in self.key_computers
        )
        self.rest_needs_modifications = any(
            isinstance(c, DerivedSequenceColumn) for c in self.rest_computers
        )

    @staticmethod
    def _key_closure(group: ColumnGroup, keys: list[str]) -> set[str]:
        """Every declared column the axis keys are computed from, keys included."""
        computed_inputs = {c.name: set(c.inputs) for c in group.computed}
        closure = set(keys)
        changed = True
        while changed:
            changed = False
            for name, inputs in computed_inputs.items():
                if name in closure and not inputs.issubset(closure):
                    closure |= inputs
                    changed = True
        return closure

    def _type_of(self, name: str) -> AxisColumnType:
        return self.group.types.get(name, "string")

    def _materialize(
        self,
        frame: pd.DataFrame,
        select: list[tuple[str, str]],
        optional: list[tuple[str, str]],
        computers: list[ColumnComputer],
        skipped: set[str],
    ) -> None:
        """One materialization pass: required selects, optional selects, computed columns."""
        for name, source in select:
            if source not in frame.columns:
                raise ValueError(f"cannot select column {name!r}; source {source!r} is missing")
            frame[name] = _coerce_axis_column(frame[source], self._type_of(name), name, source)
        for name, source in optional:
            if source in frame.columns:
                frame[name] = _coerce_axis_column(frame[source], self._type_of(name), name, source)
            else:
                skipped.add(name)
        for computer in computers:
            frame[computer.name] = computer.compute(frame, frozenset(skipped)).astype("string")

    def materialize_keys(self, table: pd.DataFrame, applier: ModificationApplier) -> None:
        """Materialize the key closure on the flat frame, before the pivot.

        The applier runs first when a key is computed from a modification output; it
        reads only raw vendor columns, so the order against the selects is free.
        """
        if self.keys_need_modifications:
            applier.apply(table)
        self._materialize(table, self.key_select, self.key_optional, self.key_computers, set())

    def finish(
        self,
        frame: pd.DataFrame,
        applier: ModificationApplier,
        *,
        modifications_applied: bool,
    ) -> pd.DataFrame:
        """Materialize every remaining declared column on the deduplicated axis frame.

        Key-phase optional skips are reconstructed from the frame: a skipped optional is
        exactly one absent from the carried axis frame, since every materialized key
        optional is carried with the declared columns.
        """
        skipped = {name for name, _source in self.key_optional if name not in frame.columns}
        if self.rest_needs_modifications and not modifications_applied:
            applier.apply(frame)
        self._materialize(frame, self.rest_select, self.rest_optional, self.rest_computers, skipped)
        present = [name for name in self.declared if name in frame.columns]
        return frame[present]


class ColumnMaterialization:
    """Materialize declared columns: key closure on the flat table, the rest per axis."""

    def __init__(
        self,
        rule: LongRule | WideRule,
        recognition: Recognition,
        applier: ModificationApplier,
    ) -> None:
        self.applier = applier
        keys_by_axis = {"obs": rule.axis.obs_keys, "var": rule.axis.var_keys}
        self.groups = {
            axis: _GroupMaterialization(group, keys_by_axis[axis])
            for axis, group in recognition.column_groups()
        }
        # A static fact of the rule, never sniffed from frame columns: a vendor file that
        # itself carries a column named like a modification output must not skip the run.
        self.modifications_applied_in_prepare = any(
            group.keys_need_modifications for group in self.groups.values()
        )

    def prepare_keys(self, table: pd.DataFrame) -> pd.DataFrame:
        """Materialize every axis-key closure on the flat frame, before the pivot.

        Mutates and returns ``table``: the pipeline is linear and the reader/exploder
        hand over ownership of the frame.
        """
        for group in self.groups.values():
            group.materialize_keys(table, self.applier)
        return table

    def finish(self, result: ParsedData) -> ParsedData:
        """Materialize every remaining declared column on the small axis frames.

        Mutates the axis frames in place; the conversion built them fresh.
        """
        frames = {"obs": result.obs, "var": result.var}
        finished = {
            axis: group.finish(
                frames[axis],
                self.applier,
                modifications_applied=self.modifications_applied_in_prepare,
            )
            for axis, group in self.groups.items()
        }
        return replace(
            result,
            obs=finished.get("obs", result.obs),
            var=finished.get("var", result.var),
        )
