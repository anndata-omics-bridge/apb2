"""Compose physical input binding, tool-specific joins, and a shared prepared source."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Self

import polars as pl
from loguru import logger

from apb2.parserV2.joins import alphadia, maxquant, metamorpheus, wombat
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFile,
    InputTable,
    PreparedTable,
    SingleFile,
)
from apb2.parserV2.parse_quant.parameters.vendor_file import VendorFile, make_vendor_file

type Identify = Callable[[Mapping[str, tuple[str, ...]]], dict[str, str]]
type Join = Callable[[Mapping[str, pl.DataFrame]], pl.DataFrame]
type Wanted = Callable[[str], bool]
"""Whether a rule can read one namespaced prepared column, such as ``peptide.Sequence``."""

# The third entry names the columns a join reads itself. A join that declares them reads only
# those plus the columns its rules can use; ``None`` reads every column.
_PREPARATIONS: dict[str, tuple[Identify, Join, tuple[str, ...] | None]] = {
    "alphadia": (alphadia.identify, alphadia.join, None),
    "maxquant_wide": (maxquant.identify, maxquant.join_wide, maxquant.KEY_COLUMNS),
    "maxquant_evidence": (
        maxquant.identify_evidence,
        maxquant.join_evidence,
        maxquant.EVIDENCE_KEY_COLUMNS,
    ),
    "metamorpheus": (metamorpheus.identify, metamorpheus.join, None),
    "wombat": (wombat.identify, wombat.join, None),
}


class InputPreparationError(ConversionError):
    """Recognized vendor inputs could not produce a valid joined table."""


def _files(source: SingleFile | Folder) -> dict[str, VendorFile]:
    """Each candidate table keyed by the name its tool wrote, plain or gzipped."""
    if isinstance(source, Folder):
        files = (make_vendor_file(path) for path in sorted(source.path.iterdir()) if path.is_file())
        return {file.name: file for file in files if file.suffix in {".tsv", ".txt"}}
    file = make_vendor_file(source.path)
    return {file.name: file}


def _separator(file: VendorFile) -> str:
    """Vendor tables are tab-separated, except a ``.csv`` such as WOMBAT-P's peptide table."""
    return "," if file.suffix == ".csv" else "\t"


def _header(file: VendorFile) -> tuple[str, ...]:
    with file.open_text("utf-8-sig") as handle:
        return tuple(next(csv.reader(handle, delimiter=_separator(file)), ()))


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
    tables: Mapping[str, tuple[VendorFile, tuple[str, ...]]]
    """Each role's file and the header that identified it; empty when no input matches."""

    @classmethod
    def identify(cls, source: SingleFile | Folder, how: str) -> Self:
        """Bind this preparation's tables from bounded headers, without reading quantities."""
        identify, _join, _keys = _PREPARATIONS[how]
        try:
            files = _files(source)
            headers = {name: _header(file) for name, file in files.items()}
            roles = identify(headers)
        except (UnicodeError, csv.Error):
            return cls(how, {})
        except ValueError as error:
            raise InputPreparationError(f"{how} preparation: {error}") from error
        return cls(how, {role: (files[name], headers[name]) for role, name in roles.items()})

    def prepare(self, path: Path, wanted: Wanted | None = None) -> PreparedTable:
        """Read and join the bound tables once, for every level the preparation serves.

        ``wanted`` limits reading to the columns the rules can use, for joins that name their
        own key columns.
        """
        started = perf_counter()
        _identify, join, keys = _PREPARATIONS[self.how]
        frames: dict[str, pl.DataFrame] = {}
        tables: list[InputTable] = []
        try:
            for role, (file, header) in self.tables.items():
                columns = _projection(header, role, keys, wanted)
                with logged_step("prepare.read", how=self.how, table=file.name):
                    frames[role] = pl.read_csv(
                        file.path,
                        separator=_separator(file),
                        infer_schema=False,
                        null_values=[""],
                        columns=columns,
                    )
                logger.info(
                    "prepare.read table={} rows={} columns={}",
                    file.name,
                    frames[role].height,
                    frames[role].width,
                )
                file_format = "csv" if _separator(file) == "," else "tsv"
                tables.append(
                    InputTable(InputFile(file.path, role, file_format), header, frames[role].height)
                )
            with logged_step("prepare.join", how=self.how, tables=",".join(sorted(frames))):
                frame = join(frames)
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
