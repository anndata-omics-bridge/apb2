"""Compose physical input binding, tool-specific joins, and a shared prepared source."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import polars as pl
from loguru import logger

from apb2.parserV2.joins import alphadia, maxquant, metamorpheus
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFiles,
    InputSource,
    PreparedTable,
)

type Identify = Callable[[Mapping[str, tuple[str, ...]]], dict[str, str]]
type Join = Callable[[Mapping[str, pl.DataFrame]], pl.DataFrame]
type Wanted = Callable[[str], bool]
"""Whether a rule can read one namespaced prepared column, such as ``peptide.Sequence``."""

# The third entry names the columns a join reads itself. A join that declares them reads only
# those plus the columns its rules can use; ``None`` reads every column.
_PREPARATIONS: dict[str, tuple[Identify, Join, tuple[str, ...] | None]] = {
    "alphadia": (alphadia.identify, alphadia.join, None),
    "maxquant": (maxquant.identify, maxquant.join, None),
    "maxquant_wide": (maxquant.identify, maxquant.join_wide, maxquant.KEY_COLUMNS),
    "maxquant_evidence": (
        maxquant.identify_evidence,
        maxquant.join_evidence,
        maxquant.EVIDENCE_KEY_COLUMNS,
    ),
    "metamorpheus": (metamorpheus.identify, metamorpheus.join, None),
}


class InputPreparationError(ConversionError):
    """Recognized vendor inputs could not produce a valid joined table."""


def _files(source: InputSource) -> dict[str, Path]:
    if isinstance(source, InputFiles):
        return dict(source.files)
    if isinstance(source, Folder):
        return {
            path.name: path
            for path in sorted(source.path.iterdir())
            if path.is_file() and path.suffix.lower() in {".tsv", ".txt"}
        }
    return {source.path.name: source.path}


def _bindings(source: InputSource, identify: Identify) -> dict[str, Path]:
    paths = _files(source)
    headers: dict[str, tuple[str, ...]] = {}
    for name, path in paths.items():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            headers[name] = tuple(next(csv.reader(handle, delimiter="\t"), ()))
    selected = identify(headers)
    return {role: paths[name] for role, name in selected.items()}


def _projection(
    path: Path, role: str, keys: tuple[str, ...] | None, wanted: Wanted | None
) -> list[str] | None:
    """Name the columns of one table to read, or ``None`` to read all of them."""
    if keys is None or wanted is None:
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), [])
    return [column for column in header if column in keys or wanted(f"{role}.{column}")]


def preparation_paths(source: InputSource, how: str) -> tuple[Path, ...]:
    """Identify this table group's physical inputs without reading its quantitative rows."""
    identify, _join, _keys = _PREPARATIONS[how]
    try:
        return tuple(_bindings(source, identify).values())
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
        paths = _bindings(source, identify)
        if not paths:
            raise IncompatibleSourceError(f"{source.path} has no recognizable {how} inputs")
        frames: dict[str, pl.DataFrame] = {}
        for role, path in paths.items():
            columns = _projection(path, role, keys, wanted)
            with logged_step("prepare.read", how=how, table=path.name):
                frames[role] = pl.read_csv(
                    path, separator="\t", infer_schema=False, null_values=[""], columns=columns
                )
            logger.info(
                "prepare.read table={} rows={} columns={}",
                path.name,
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
        source_paths=tuple(paths.values()),
        duration_seconds=perf_counter() - started,
    )
