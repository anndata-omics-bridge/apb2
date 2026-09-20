"""Excel workbook input: inspect and read one explicitly named sheet.

Workbook rules identify a sheet rather than guessing one. The file suffix remains a hint only:
fixture stores and upload systems commonly preserve a generic ``.txt`` name while retaining the
original XLSX bytes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs
from fastexcel import FastExcelError

from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    ExcelFormatContract,
    ExcelSourceEvidence,
    LevelReadPlan,
    NumericTextFormat,
)

_DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())


def sheet_header(path: Path, sheet_name: str) -> tuple[str, ...]:
    """Inspect the named sheet without materializing its data rows in Python."""
    try:
        frame = pl.read_excel(
            path,
            sheet_name=sheet_name,
            read_options={"n_rows": 0},
            drop_empty_cols=False,
            raise_if_empty=False,
        )
    except (FastExcelError, OSError, ValueError, pl.exceptions.PolarsError) as error:
        raise IncompatibleSourceError(
            f"{path}: cannot read workbook sheet {sheet_name!r}: {error}"
        ) from error
    if not frame.width:
        raise IncompatibleSourceError(f"{path}: workbook sheet {sheet_name!r} is empty")
    return tuple(frame.columns)


def schema_evidence(
    path: Path,
    contract: ExcelFormatContract,
    accepts: Callable[[tuple[str, ...]], bool],
) -> ExcelSourceEvidence:
    """Inspect the declared sheet and require its header to satisfy the level."""
    columns = sheet_header(path, contract.sheet_name)
    if not accepts(columns):
        raise IncompatibleSourceError(
            f"{path}: workbook sheet {contract.sheet_name!r} does not carry the required columns"
        )
    return ExcelSourceEvidence(
        columns=columns,
        sheet_name=contract.sheet_name,
        number_format=_DOT,
    )


@dataclass(frozen=True, slots=True)
class ExcelInputReader:
    """One workbook sheet and one level's exact projection."""

    path: Path
    evidence: ExcelSourceEvidence
    plan: LevelReadPlan

    def read(self) -> LevelSourceTable:
        """Read only projected columns and apply the already resolved dtypes."""
        projected = pl.read_excel(
            self.path,
            sheet_name=self.evidence.sheet_name,
            columns=self.plan.projected_columns,
            infer_schema_length=None,
            drop_empty_rows=False,
            drop_empty_cols=False,
            raise_if_empty=False,
        )
        # Calamine's previous Python reader exposed Excel numbers as floats and empty
        # cells as empty strings. Preserve those axis semantics using whole-frame casts.
        projected = projected.with_columns(cs.integer().cast(pl.Float64)).with_columns(
            pl.col(self.plan.text_sources).cast(pl.String).fill_null(""),
            pl.col(self.plan.native_numeric_sources).cast(pl.Float64),
        )
        return LevelSourceTable(frame=projected)


def make_excel_reader(
    path: Path,
    evidence: ExcelSourceEvidence,
    plan: LevelReadPlan,
) -> ExcelInputReader:
    """Construct the reader one resolved workbook source and level plan describe."""
    return ExcelInputReader(path=path, evidence=evidence, plan=plan)
