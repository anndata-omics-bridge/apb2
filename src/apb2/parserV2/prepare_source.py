"""Compose physical input binding, tool-specific joins, and a shared prepared source."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import polars as pl
from loguru import logger

from apb2.parserV2.joins import alphadia, maxquant, metamorpheus, wombat
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFiles,
    InputSource,
    PreparedTable,
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


def _files(source: InputSource) -> dict[str, VendorFile]:
    """Each candidate table keyed by the name its tool wrote, plain or gzipped."""
    if isinstance(source, InputFiles):
        return {name: make_vendor_file(path) for name, path in source.files.items()}
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


def _bindings(source: InputSource, identify: Identify) -> dict[str, VendorFile]:
    files = _files(source)
    selected = identify({name: _header(file) for name, file in files.items()})
    return {role: files[name] for role, name in selected.items()}


def _projection(
    file: VendorFile, role: str, keys: tuple[str, ...] | None, wanted: Wanted | None
) -> list[str] | None:
    """Name the columns of one table to read, or ``None`` to read all of them."""
    if keys is None or wanted is None:
        return None
    return [column for column in _header(file) if column in keys or wanted(f"{role}.{column}")]


def preparation_paths(source: InputSource, how: str) -> tuple[Path, ...]:
    """Identify this table group's physical inputs without reading its quantitative rows."""
    identify, _join, _keys = _PREPARATIONS[how]
    try:
        return tuple(file.path for file in _bindings(source, identify).values())
    except (UnicodeError, csv.Error):
        return ()
    except ValueError as error:
        raise InputPreparationError(f"{how} preparation: {error}") from error


def recognizes_preparation(source: InputSource, how: str) -> bool:
    """Recognize a tool from bounded headers without reading quantities or joining."""
    return bool(preparation_paths(source, how))


def prepare_source(
    source: InputSource, how: str | None, wanted: Wanted | None = None
) -> InputSource:
    """Prepare once; already prepared sources are passed to all level compilers unchanged.

    ``wanted`` limits reading to the columns the rules can use, for joins that name their own
    key columns.
    """
    if how is None:
        return source
    if isinstance(source, PreparedTable):
        if source.how != how:
            raise InputPreparationError(f"source prepared with {source.how}, rule requests {how}")
        return source
    started = perf_counter()
    identify, join, keys = _PREPARATIONS[how]
    try:
        files = _bindings(source, identify)
        if not files:
            raise IncompatibleSourceError(f"{source.path} has no recognizable {how} inputs")
        frames: dict[str, pl.DataFrame] = {}
        for role, file in files.items():
            columns = _projection(file, role, keys, wanted)
            with logged_step("prepare.read", how=how, table=file.name):
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
        with logged_step("prepare.join", how=how, tables=",".join(sorted(frames))):
            frame = join(frames)
        logger.info("prepare.join how={} rows={} columns={}", how, frame.height, frame.width)
    except IncompatibleSourceError:
        raise
    except (ValueError, pl.exceptions.PolarsError) as error:
        raise InputPreparationError(f"{how} preparation: {error}") from error
    return PreparedTable(
        path=source.path,
        frame=frame,
        how=how,
        source_paths=tuple(file.path for file in files.values()),
        duration_seconds=perf_counter() - started,
    )
