"""Fully configured parser orchestration.

``Parser`` holds only completed runtime behavior and concrete values. It retains no rule
model, no unresolved policy, and no storage backend, and neither it nor any injected
strategy branches on which rule variant, vendor, layout, format, dialect policy, or mode
was selected — registries finish that selection during construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from apb2.result import ParsedData
from apb2.vendor_parse_rules.model import QuantificationLevel


class BoundInputReader(Protocol):
    """Read the bound source into one assembled table."""

    def read(self) -> pd.DataFrame: ...


class ModificationApplier(Protocol):
    """Normalize modified-sequence content in the assembled table."""

    def apply(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


class FragmentExploder(Protocol):
    """Expand packed per-fragment values into rows."""

    def explode(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


class ColumnPlan(Protocol):
    """Materialize selected, computed, and typed columns."""

    def materialize(self, table: pd.DataFrame, /) -> pd.DataFrame: ...


class TableConversion(Protocol):
    """Convert the materialized table into one backend-neutral result."""

    def parse(self, table: pd.DataFrame, /) -> ParsedData: ...


class OutputMetadata(Protocol):
    """Attach the precomputed output provenance to the result."""

    def attach(self, result: ParsedData, /) -> ParsedData: ...


@dataclass(frozen=True, slots=True)
class Parser:
    """One quantification level's completed strategy graph.

    ``level`` is an output identity used for naming and provenance, never a discriminator
    consulted by any strategy. Every field is unconditional: a rule that declares no
    modifications or fragments receives the identity appliers from ``identity.py``.
    """

    level: QuantificationLevel
    input: BoundInputReader
    modifications: ModificationApplier
    fragments: FragmentExploder
    columns: ColumnPlan
    conversion: TableConversion
    metadata: OutputMetadata

    def parse(self) -> ParsedData:
        """Run the plan once and return one backend-neutral result."""
        table = self.input.read()
        table = self.modifications.apply(table)
        table = self.fragments.explode(table)
        table = self.columns.materialize(table)
        result = self.conversion.parse(table)
        return self.metadata.attach(result)
