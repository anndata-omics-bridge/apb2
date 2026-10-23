"""Excel workbook input: inspect and read one explicitly named sheet.

Workbook rules identify a sheet rather than guessing one. The file suffix remains a hint only:
fixture stores and upload systems commonly preserve a generic ``.txt`` name while retaining the
original XLSX bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from zipfile import is_zipfile

import fastexcel
import polars as pl
import polars.selectors as cs
from fastexcel import FastExcelError

from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    ExcelFormatContract,
    ExcelSourceEvidence,
    InputFile,
    InputTable,
    LevelReadPlan,
    LevelSourceTable,
    NumericTextFormat,
)

_DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())


def sheet_header(path: Path, sheet_name: str) -> tuple[str, ...]:
    """Inspect the named sheet without materializing its data rows in Python."""
    if not is_zipfile(path):
        raise IncompatibleSourceError(f"{path}: not a ZIP-based Excel workbook")
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


def sheet_names(path: Path) -> tuple[str, ...]:
    """The sheets of a ZIP-based workbook in file order; none for any other file."""
    if not is_zipfile(path):
        return ()
    try:
        return tuple(fastexcel.read_excel(path).sheet_names)
    except (FastExcelError, OSError) as error:
        raise IncompatibleSourceError(f"{path}: cannot read workbook: {error}") from error


def read_sheet(path: Path, sheet_name: str, columns: Sequence[str] | None) -> pl.DataFrame:
    """Read one sheet's cells as text, as a preparation reads every table it joins.

    ``columns`` limits the read to the named columns; ``None`` reads all of them.
    """
    return pl.read_excel(
        path,
        sheet_name=sheet_name,
        columns=columns,
        infer_schema_length=0,
        drop_empty_rows=False,
        drop_empty_cols=False,
        raise_if_empty=False,
    )


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
        # Excel stores every number as a float; an integer-looking cell is one too.
        projected = projected.with_columns(cs.integer().cast(pl.Float64)).with_columns(
            pl.col(self.plan.text_sources).cast(pl.String),
            pl.col(self.plan.native_numeric_sources).cast(pl.Float64),
        )
        table = InputTable(
            InputFile(self.path, "table", "xlsx"), self.evidence.columns, projected.height
        )
        return LevelSourceTable(projected, (table,))
