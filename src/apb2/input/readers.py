"""Concrete bound readers: one path, one resolved dialect, one compiled projection."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from apb2.dialect import DelimitedDialect
from apb2.input.plan import ReadPlan


@dataclass(frozen=True, slots=True)
class DelimitedTableReader:
    """Read one delimited file with its concrete dialect and exact projection."""

    path: Path
    dialect: DelimitedDialect
    plan: ReadPlan

    def read(self) -> pd.DataFrame:
        dtypes: Mapping[Hashable, str] = dict.fromkeys(sorted(self.plan.string_sources), "string")
        return pd.read_csv(
            self.path,
            sep=self.dialect.delimiter,
            encoding="utf-8-sig",
            usecols=list(self.plan.columns),
            dtype=dtypes,
            **self.dialect.numbers.read_csv_options(),
        )


@dataclass(frozen=True, slots=True)
class ParquetTableReader:
    """Read one Parquet file's projected columns; its schema already types them."""

    path: Path
    plan: ReadPlan

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path, columns=list(self.plan.columns))
