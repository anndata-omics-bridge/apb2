"""Project one level from the already prepared, shared input frame."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.parameters.resolved import ResolvedLevelPlan


@dataclass(frozen=True, slots=True)
class PreparedInputReader:
    """A level projection, not another physical read or preparation."""

    frame: pl.DataFrame
    plan: ResolvedLevelPlan

    def read(self) -> LevelSourceTable:
        """Keep native prepared types and the rule's declared source closure."""
        frame = self.frame.select(self.plan.read.projected_columns)
        # Full joins retain measurements absent from another level. Only an entirely
        # absent identity is skipped; partially malformed identities still fail parsing.
        for axis in (self.plan.obs, self.plan.var):
            frame = frame.filter(
                pl.any_horizontal(
                    pl.col(key).is_not_null() for key in axis.source.keys.raw_key_columns
                )
            )
        return LevelSourceTable(frame)
