"""The four contracts a parse runs on, and the context that runs them in order.

Each Protocol is one pipeline step, and each is implemented by the module named after it.
They are declared here rather than beside their implementations because ``Parser`` is their
only consumer: typing its fields structurally is what keeps the context from importing a
single concrete strategy, and what lets a test inject four fakes to assert the order.

``Parser`` holds only completed runtime behavior and concrete values — no rule model, no
unresolved policy, no storage backend — and neither it nor any injected strategy branches
on which rule variant, vendor, layout, format, dialect policy, or mode was selected.
Filling these four fields is ``configure_parse``'s job, one layer up, where the rule and
the search-parameter evidence live.

The pipeline runs conversion-first: after the read and the fragment explode, only the
axis-key closure is materialized on the flat table (the pivot cannot group without it);
every remaining declared column is materialized afterwards on the deduplicated axis
frames, where a column fix touches nrObs or nrVars rows instead of nrObs x nrVars.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol

import pandas as pd

from apb2.parse_quant.result import ParsedData
from apb2.serialization import JsonValue


class BoundInputReader(Protocol):
    """Read the bound source into one assembled table."""

    def read(self) -> pd.DataFrame: ...


class FragmentExploder(Protocol):
    """Expand packed per-fragment values into rows, and declare the columns they came from."""

    def packed_columns(self) -> tuple[str, ...]: ...

    def explode(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


class ColumnPlan(Protocol):
    """Materialize declared columns: key closure before the pivot, the rest after."""

    def prepare_keys(self, table: pd.DataFrame, /) -> pd.DataFrame: ...

    def finish(self, result: ParsedData, /) -> ParsedData: ...


class TableConversion(Protocol):
    """Convert the key-prepared table into one backend-neutral result."""

    def parse(self, table: pd.DataFrame, /) -> ParsedData: ...


class Parser:
    """One quantification level's completed strategy graph.

    ``level`` is an output identity used for naming and provenance, never a discriminator
    consulted by any strategy — which is why it is a plain name here and a schema literal
    on the rule side. Every field is unconditional: a rule that declares no fragments
    receives the identity exploder.
    """

    def __init__(
        self,
        *,
        level: str,
        input: BoundInputReader,
        fragments: FragmentExploder,
        columns: ColumnPlan,
        conversion: TableConversion,
        provenance: Mapping[str, JsonValue],
    ) -> None:
        self.level = level
        self.input = input
        self.fragments = fragments
        self.columns = columns
        self.conversion = conversion
        self.provenance = provenance

    def parse(self) -> ParsedData:
        """Run the plan once and return one backend-neutral result."""
        table = self.input.read()
        table = self.fragments.explode(table)
        table = self.columns.prepare_keys(table)
        result = self.conversion.parse(table)
        result = self.columns.finish(result)
        return replace(result, uns={**result.uns, **self.provenance})
