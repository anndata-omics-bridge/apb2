"""Column materialization as a configured runtime strategy.

Construction extracts each axis group's selections, logical types, and computers from the
rule; the strategy itself holds only those plain values. Missing required sources and
absent optional sources are input facts judged per table — that stays here, per file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from apb2.convert.axis_types import (
    AxisColumnContext,
    AxisName,
    coerce_axis_column,
)
from apb2.convert.computed import ColumnComputer, make_computer
from apb2.vendor_parse_rules.model import (
    AxisColumnType,
    ColumnGroup,
    LongRule,
    WideRule,
)


@dataclass(frozen=True, slots=True)
class AxisColumns:
    """One axis group's complete materialization inputs, no schema retained."""

    axis: AxisName
    select: tuple[tuple[str, str], ...]
    optional: tuple[tuple[str, str], ...]
    types: Mapping[str, AxisColumnType]
    computers: tuple[ColumnComputer, ...]

    def materialize(self, table: pd.DataFrame) -> None:
        for name, source in self.select:
            if source not in table.columns:
                raise ValueError(f"cannot select column {name!r}; source {source!r} is missing")
            table[name] = self._coerced(table[source], name, source)
        skipped: set[str] = set()
        for name, source in self.optional:
            if source in table.columns:
                table[name] = self._coerced(table[source], name, source)
            else:
                skipped.add(name)
        for computer in self.computers:
            table[computer.name] = computer.compute(table, frozenset(skipped)).astype("string")

    def _coerced(self, values: pd.Series, name: str, source: str) -> pd.Series:
        return coerce_axis_column(
            values,
            AxisColumnContext(
                axis=self.axis,
                output_name=name,
                source_name=source,
                logical_type=self.types[name],
            ),
        )


@dataclass(frozen=True, slots=True)
class MaterializeColumns:
    """Materialize every declared axis group's selected, typed, and computed columns.

    A long rule contributes obs and var groups; a wide rule contributes var only — its
    observation axis comes from layer regex captures, not from table columns.
    """

    groups: tuple[AxisColumns, ...]

    def materialize(self, table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        for group in self.groups:
            group.materialize(out)
        return out


def make_column_plan(rule: LongRule | WideRule) -> MaterializeColumns:
    """Extract every declared axis group's materialization inputs from one rule."""
    return MaterializeColumns(
        groups=tuple(_axis_columns(group, axis) for axis, group in rule.named_column_groups())
    )


def _axis_columns(group: ColumnGroup, axis: AxisName) -> AxisColumns:
    selected = {**group.select, **group.optional_select}
    return AxisColumns(
        axis=axis,
        select=tuple(group.select.items()),
        optional=tuple(group.optional_select.items()),
        types={name: group.type_for(name) for name in selected},
        computers=tuple(make_computer(column) for column in group.computed),
    )
