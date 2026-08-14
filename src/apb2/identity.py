"""Identity strategies for plan steps a rule does not declare.

A rule without fragments gets this instead of a ``| None`` field: the absent case behaves
uniformly, so no consumer ever asks whether the step was configured. The modification
step's identity member is reused from the legacy runtime (``preprocess.NoModifications``),
whose one-argument ``apply`` already satisfies the V2 protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class NoFragments:
    """Leave the table unchanged when the rule declares no fragments."""

    def packed_columns(self) -> tuple[str, ...]:
        """No packed source columns: nothing is exploded."""
        return ()

    def explode(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return the table unchanged."""
        return table
