"""The one value a bound input reader returns and a source decomposer accepts.

A projected physical table and nothing else: the columns one level's read plan asked for,
in that order, with no axis identity yet. Everything that gives a row meaning — which
columns are keys, which are measurements — is decided by the decomposer that receives it.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(slots=True)
class LevelSourceTable:
    """One level's projected physical rows."""

    frame: pl.DataFrame
    # pl.DataFrame({
    #     "sequence": ["PEPMIDE", "OTHER"],
    #     "mods": ["Oxidation@M", None],
    #     "mod_sites": ["4", None],
    #     "charge": ["2", "3"],
    #     "run_A": [100.0, 50.0],
    #     "run_B": [120.0, 60.0],
    # })
