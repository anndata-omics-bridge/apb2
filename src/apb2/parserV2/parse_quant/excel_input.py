"""Excel workbook input: inspect and read one explicitly named sheet.

Workbook rules identify a sheet rather than guessing one. The file suffix remains a hint only:
fixture stores and upload systems commonly preserve a generic ``.txt`` name while retaining the
original XLSX bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from python_calamine import CalamineError, load_workbook

from apb2.parserV2.parse_quant.data.source import LevelSourceTable
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    ExcelFormatContract,
    ExcelSourceEvidence,
    LevelReadPlan,
    NumericTextFormat,
)

_DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())


def _sheet_rows(path: Path, sheet_name: str) -> Sequence[Sequence[object]]:
    """Load one declared sheet through the workbook reader."""
    try:
        workbook = load_workbook(path)
        try:
            return workbook.get_sheet_by_name(sheet_name).to_python()
        finally:
            workbook.close()
    except (CalamineError, OSError, ValueError) as error:
        raise IncompatibleSourceError(
            f"{path}: cannot read workbook sheet {sheet_name!r}: {error}"
        ) from error


def sheet_header(path: Path, sheet_name: str) -> tuple[str, ...]:
    """Read a named sheet's header."""
    rows = _sheet_rows(path, sheet_name)
    if not rows:
        raise IncompatibleSourceError(f"{path}: workbook sheet {sheet_name!r} is empty")
    return tuple(str(name) for name in rows[0])


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
        rows = _sheet_rows(self.path, self.evidence.sheet_name)
        header = tuple(str(name) for name in rows[0])
        positions = {name: index for index, name in enumerate(header)}
        projected = pl.DataFrame(
            {
                name: _column_values(rows[1:], positions[name])
                for name in self.plan.projected_columns
            },
            strict=False,
        )
        expressions = [
            pl.col(name).cast(pl.String, strict=True).alias(name) for name in self.plan.text_sources
        ]
        expressions.extend(
            pl.col(name).cast(pl.Float64, strict=True).alias(name)
            for name in self.plan.native_numeric_sources
        )
        if expressions:
            projected = projected.with_columns(expressions)
        return LevelSourceTable(frame=projected)


def _column_values(rows: Sequence[Sequence[object]], index: int) -> list[object | None]:
    """Read one rectangular column, treating absent trailing cells as missing."""
    return [row[index] if index < len(row) else None for row in rows]


def make_excel_reader(
    path: Path,
    evidence: ExcelSourceEvidence,
    plan: LevelReadPlan,
) -> ExcelInputReader:
    """Construct the reader one resolved workbook source and level plan describe."""
    return ExcelInputReader(path=path, evidence=evidence, plan=plan)
