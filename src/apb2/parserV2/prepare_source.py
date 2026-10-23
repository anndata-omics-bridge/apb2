"""Compose physical input binding, tool-specific joins, and a shared prepared source."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Literal, Self

import polars as pl
from loguru import logger

from apb2.parserV2.joins import alphadia, maxquant, metamorpheus, msangel, wombat
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.excel_input import read_sheet, sheet_header, sheet_names
from apb2.parserV2.parse_quant.operations import WorkingParseConfiguration
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    FrameSourceEvidence,
    InputFile,
    InputTable,
    SingleFile,
)
from apb2.parserV2.parse_quant.parameters.vendor_file import VendorFile
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.parse_rule_facade import ParseRuleFacade

type Identify = Callable[[Mapping[str, tuple[str, ...]]], dict[str, str]]
type Join = Callable[[Mapping[str, pl.DataFrame]], pl.DataFrame]
type Wanted = Callable[[str], bool]
"""Whether a rule can read one namespaced prepared column, such as ``peptide.Sequence``."""


@dataclass(frozen=True, slots=True)
class _Preparation:
    """One tool's join: how it binds its inputs, how it joins them, and what it reads."""

    identify: Identify
    join: Join
    keys: tuple[str, ...] | None = None
    """Columns the join reads itself; with them, only those and the rules' columns are read.

    ``None`` reads every column.
    """
    sheets: tuple[str, ...] = ()
    """The workbook sheets the join reads; a join naming none reads delimited text files."""


_PREPARATIONS: dict[str, _Preparation] = {
    "alphadia": _Preparation(alphadia.identify, alphadia.join),
    "maxquant_wide": _Preparation(maxquant.identify, maxquant.join_wide, maxquant.KEY_COLUMNS),
    "maxquant_evidence": _Preparation(
        maxquant.identify_evidence, maxquant.join_evidence, maxquant.EVIDENCE_KEY_COLUMNS
    ),
    "metamorpheus": _Preparation(metamorpheus.identify, metamorpheus.join),
    "msangel": _Preparation(msangel.identify, msangel.join, msangel.KEY_COLUMNS, msangel.SHEETS),
    "wombat": _Preparation(wombat.identify, wombat.join),
}


class InputPreparationError(ConversionError):
    """Recognized vendor inputs could not produce a valid joined table."""


@dataclass(frozen=True, slots=True)
class PreparedTable:
    """One preparation result shared by detection and every requested level."""

    path: Path
    frame: pl.DataFrame
    how: str
    input_tables: tuple[InputTable, ...]
    duration_seconds: float

    def header(self, working: WorkingParseConfiguration) -> tuple[str, ...]:
        """The joined columns; a level's own predicate decides whether it can use them."""
        return tuple(self.frame.columns)

    def parser(self, facade: ParseRuleFacade, checks: Literal["standard", "strict"]) -> Parser:
        """Resolve one level against the joined schema and read it from the joined frame."""
        evidence = FrameSourceEvidence(tuple(self.frame.columns), tuple(self.frame.schema.items()))
        strategy = facade.resolve_source(evidence, checks=checks)
        # A wide level's observation keys come from header captures, not prepared columns,
        # so only key columns the level reads can mark a row as absent.
        projected = set(strategy.read.projected_columns)
        keys = tuple(
            columns
            for columns in (strategy.obs.keys.raw_key_columns, strategy.var.keys.raw_key_columns)
            if set(columns) <= projected
        )
        preparation = {
            "how": self.how,
            "duration_seconds": self.duration_seconds,
            "estimated_size_bytes": self.frame.estimated_size(),
        }
        strategy = replace(
            strategy, provenance={**strategy.provenance, "input_preparation": preparation}
        )
        reader = PreparedInputReader(self.frame, strategy.read, keys, self.input_tables)
        return Parser(reader, strategy)


@dataclass(frozen=True, slots=True)
class _DelimitedTable:
    """One delimited vendor table, plain or gzipped, named as its tool wrote it."""

    file: VendorFile

    @property
    def name(self) -> str:
        return self.file.name

    @property
    def path(self) -> Path:
        return self.file.path

    @property
    def format(self) -> str:
        return "csv" if self._separator == "," else "tsv"

    @property
    def _separator(self) -> str:
        """Vendor tables are tab-separated, except a ``.csv`` such as WOMBAT-P's peptides."""
        return "," if self.file.suffix == ".csv" else "\t"

    def header(self) -> tuple[str, ...]:
        with self.file.open_text("utf-8-sig") as handle:
            return tuple(next(csv.reader(handle, delimiter=self._separator), ()))

    def read(self, columns: list[str] | None) -> pl.DataFrame:
        return pl.read_csv(
            self.file.path,
            separator=self._separator,
            infer_schema=False,
            null_values=[""],
            columns=columns,
        )


@dataclass(frozen=True, slots=True)
class _WorkbookSheet:
    """One named sheet of an Excel workbook, read as text like a delimited table."""

    path: Path
    name: str

    @property
    def format(self) -> str:
        return "xlsx"

    def header(self) -> tuple[str, ...]:
        return sheet_header(self.path, self.name)

    def read(self, columns: list[str] | None) -> pl.DataFrame:
        return read_sheet(self.path, self.name, columns)


type _Table = _DelimitedTable | _WorkbookSheet


def _tables(source: SingleFile | Folder, sheets: tuple[str, ...]) -> dict[str, _Table]:
    """Each candidate table keyed by its name: a delimited file, or a named workbook sheet.

    A preparation that names sheets reads only workbooks, and only those sheets; every other
    preparation reads delimited text.
    """
    files = source.tables(None)
    if not sheets:
        return {file.name: _DelimitedTable(file) for file in files if file.suffix in _DELIMITED}
    tables: dict[str, _Table] = {}
    for file in files:
        present = sheet_names(file.path)
        tables.update(
            {sheet: _WorkbookSheet(file.path, sheet) for sheet in sheets if sheet in present}
        )
    return tables


_DELIMITED = frozenset({".tsv", ".txt", ".csv"})


def _projection(
    header: tuple[str, ...], role: str, keys: tuple[str, ...] | None, wanted: Wanted | None
) -> list[str] | None:
    """Name the columns of one table to read, or ``None`` to read all of them."""
    if keys is None or wanted is None:
        return None
    return [column for column in header if column in keys or wanted(f"{role}.{column}")]


@dataclass(frozen=True, slots=True)
class PreparationInputs:
    """One preparation's physical tables, each identified once from its header."""

    how: str
    tables: Mapping[str, tuple[_Table, tuple[str, ...]]]
    """Each role's table and the header that identified it; empty when no input matches."""

    @classmethod
    def identify(cls, source: SingleFile | Folder, how: str) -> Self:
        """Bind this preparation's tables from bounded headers, without reading quantities."""
        preparation = _PREPARATIONS[how]
        try:
            tables = _tables(source, preparation.sheets)
            headers = {name: table.header() for name, table in tables.items()}
            roles = preparation.identify(headers)
        except (UnicodeError, csv.Error, IncompatibleSourceError):
            return cls(how, {})
        except ValueError as error:
            raise InputPreparationError(f"{how} preparation: {error}") from error
        return cls(how, {role: (tables[name], headers[name]) for role, name in roles.items()})

    def prepare(self, path: Path, wanted: Wanted | None = None) -> PreparedTable:
        """Read and join the bound tables once, for every level the preparation serves.

        ``wanted`` limits reading to the columns the rules can use, for joins that name their
        own key columns.
        """
        started = perf_counter()
        preparation = _PREPARATIONS[self.how]
        frames: dict[str, pl.DataFrame] = {}
        tables: list[InputTable] = []
        try:
            for role, (table, header) in self.tables.items():
                columns = _projection(header, role, preparation.keys, wanted)
                with logged_step("prepare.read", how=self.how, table=table.name):
                    frames[role] = table.read(columns)
                logger.info(
                    "prepare.read table={} rows={} columns={}",
                    table.name,
                    frames[role].height,
                    frames[role].width,
                )
                tables.append(
                    InputTable(
                        InputFile(table.path, role, table.format), header, frames[role].height
                    )
                )
            with logged_step("prepare.join", how=self.how, tables=",".join(sorted(frames))):
                frame = preparation.join(frames)
            logger.info(
                "prepare.join how={} rows={} columns={}", self.how, frame.height, frame.width
            )
        except (ValueError, pl.exceptions.PolarsError) as error:
            raise InputPreparationError(f"{self.how} preparation: {error}") from error
        return PreparedTable(
            path=path,
            frame=frame,
            how=self.how,
            input_tables=tuple(tables),
            duration_seconds=perf_counter() - started,
        )
