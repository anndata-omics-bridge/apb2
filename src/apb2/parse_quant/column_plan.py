"""``ColumnPlan``: the ``columns`` block — select, optional_select, types, computed.

Materialization is split around the pivot, because cost lives on the flat table: the
axis-key closure (the key columns plus everything they are computed from) is prepared on
the flat frame — the pivot cannot group without it — while every remaining declared
column is materialized afterwards on the deduplicated obs/var frames, where a column fix
touches nrObs or nrVars rows instead of nrObs x nrVars.

``selectors.column_plan_for`` performs the split and reads the ``how`` and ``types``
selectors once; what arrives here is two finished passes per axis — the columns to select,
the computers to run, and the coercion each declared column's logical type named.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from apb2.parse_quant.modifications import ModificationApplier
from apb2.parse_quant.parse_strategy import ColumnPlan
from apb2.parse_quant.result import ParsedData

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


type AxisCoercer = Callable[[pd.Series, str, str], pd.Series]

AXIS_COERCERS: Mapping[str, AxisCoercer] = {
    "string": _coerce_string,
    "integer": _coerce_integer,
    "number": _coerce_number,
    "boolean": _coerce_boolean,
}
"""One coercion per logical axis-column type; ``selectors.coercer_for`` reads this table."""


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

    def __init__(self, *, name: str, sources: tuple[str, ...]) -> None:
        self.name = name
        self.sources = sources

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        present = _present_sources(self.name, self.sources, df, allow_missing)
        result = df[present[0]].astype(object).copy()
        for source in present[1:]:
            result = result.where(result.notna(), df[source].astype(object))
        return result


class JoinNonEmptyColumn:
    """Join the non-empty source values with a separator."""

    def __init__(self, *, name: str, sources: tuple[str, ...], separator: str) -> None:
        self.name = name
        self.sources = sources
        self.separator = separator

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

    def __init__(self, *, name: str, source_key: str) -> None:
        self.name = name
        self.source_key = source_key

    def compute(self, df: pd.DataFrame, allow_missing: frozenset[str]) -> pd.Series:
        del allow_missing
        if self.source_key not in df.columns:
            raise ValueError(
                f"cannot compute column {self.name!r}; apb2 column {self.source_key!r} is missing"
            )
        return df[self.source_key]


class ProformaIonColumn:
    """Combine a string peptidoform and an already-typed positive integer charge."""

    def __init__(self, *, name: str, sequence_key: str, charge_key: str) -> None:
        self.name = name
        self.sequence_key = sequence_key
        self.charge_key = charge_key

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

    def __init__(self, *, name: str, ion_key: str, label_key: str) -> None:
        self.name = name
        self.ion_key = ion_key
        self.label_key = label_key

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


# --------------------------------------------------------------- split materialization


class MaterializationPass:
    """One pass over one frame: required selects, optional selects, computed columns.

    ``coercers`` holds the coercion each selected column's declared logical type named, so
    a pass never asks what type a column is — it looks up what to do with it.
    """

    def __init__(
        self,
        *,
        select: tuple[tuple[str, str], ...],
        optional: tuple[tuple[str, str], ...],
        coercers: Mapping[str, AxisCoercer],
        computers: tuple[ColumnComputer, ...],
    ) -> None:
        self.select = select
        self.optional = optional
        self.coercers = coercers
        self.computers = computers
        self.needs_modifications = any(
            isinstance(computer, DerivedSequenceColumn) for computer in computers
        )

    def run(self, frame: pd.DataFrame, skipped: set[str]) -> None:
        """Materialize this pass's columns onto ``frame``, recording optional skips."""
        for name, source in self.select:
            if source not in frame.columns:
                raise ValueError(f"cannot select column {name!r}; source {source!r} is missing")
            frame[name] = self.coercers[name](frame[source], name, source)
        for name, source in self.optional:
            if source in frame.columns:
                frame[name] = self.coercers[name](frame[source], name, source)
            else:
                skipped.add(name)
        for computer in self.computers:
            frame[computer.name] = computer.compute(frame, frozenset(skipped)).astype("string")


class AxisMaterialization:
    """One axis group's materialization, split into the key closure and the rest."""

    def __init__(
        self,
        *,
        declared: tuple[str, ...],
        keys: MaterializationPass,
        rest: MaterializationPass,
    ) -> None:
        self.declared = declared
        self.keys = keys
        self.rest = rest

    def materialize_keys(self, table: pd.DataFrame, applier: ModificationApplier) -> None:
        """Materialize the key closure on the flat frame, before the pivot.

        The applier runs first when a key is computed from a modification output; it
        reads only raw vendor columns, so the order against the selects is free.
        """
        if self.keys.needs_modifications:
            applier.apply(table)
        self.keys.run(table, set())

    def finish(
        self,
        frame: pd.DataFrame,
        applier: ModificationApplier,
        *,
        modifications_applied: bool,
    ) -> pd.DataFrame:
        """Materialize every remaining declared column on the deduplicated axis frame.

        Key-phase optional skips are reconstructed from raw SOURCE presence — sources are
        always carried, so source-in-frame is exactly source-was-in-input. Declared-name
        presence would be defeated by a vendor column bearing the declared name.
        """
        skipped = {name for name, source in self.keys.optional if source not in frame.columns}
        if self.rest.needs_modifications and not modifications_applied:
            applier.apply(frame)
        self.rest.run(frame, skipped)
        present = [name for name in self.declared if name in frame.columns and name not in skipped]
        return frame[present]


class ColumnMaterialization:
    """Materialize declared columns: key closure on the flat table, the rest per axis."""

    def __init__(
        self,
        *,
        groups: Mapping[str, AxisMaterialization],
        applier: ModificationApplier,
    ) -> None:
        self.groups = groups
        self.applier = applier
        # A static fact of the rule, never sniffed from frame columns: a vendor file that
        # itself carries a column named like a modification output must not skip the run.
        self.modifications_applied_in_prepare = any(
            group.keys.needs_modifications for group in groups.values()
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


_IMPLEMENTS: type[ColumnPlan] = ColumnMaterialization
"""Pyright checks the class against the protocol here, at its definition site."""
