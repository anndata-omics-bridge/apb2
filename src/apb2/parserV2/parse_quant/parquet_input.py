"""Parquet: read the physical schema, then one level's exact projection.

Nothing to detect. A Parquet file states its own column names and types, which is why this
module has no dialect resolution and why the reader overrides no dtype: overriding the
physical schema would discard the very typing that makes Parquet worth reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from apb2.parserV2.parse_quant.parameters.source import (
    FrameSourceEvidence,
    InputFile,
    InputTable,
    LevelReadPlan,
    LevelSourceTable,
)


def schema_evidence(path: Path) -> FrameSourceEvidence:
    """Read the physical schema, in file order.

    No header predicate, because there is nothing to choose between: a Parquet file has one
    reading. Whether this level can use those columns is source resolution's answer.
    """
    schema = pl.read_parquet_schema(path)
    return FrameSourceEvidence(columns=tuple(schema), dtypes=tuple(schema.items()))


@dataclass(frozen=True, slots=True)
class ParquetInputReader:
    """One Parquet file, its column names, and one level's exact projection."""

    path: Path
    columns: tuple[str, ...]
    plan: LevelReadPlan

    def read(self) -> LevelSourceTable:
        """Read the projected columns, preserving their physical types and order."""
        frame = (
            pl.scan_parquet(self.path)
            .select(list(self.plan.projected_columns))
            .with_columns(*self.plan.casts())
            .collect()
        )
        table = InputTable(InputFile(self.path, "table", "parquet"), self.columns, frame.height)
        return LevelSourceTable(frame, (table,))
