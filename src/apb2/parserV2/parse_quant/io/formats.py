"""Format-selected result readers, writers, and the storage-only reformat workflow."""

from __future__ import annotations

from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Protocol

from apb2.parserV2.parse_quant.data.parsed import (
    ParsedLevelName,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.io.anndata_reader import H5adReader, H5muReader
from apb2.parserV2.parse_quant.io.anndata_writer import H5adWriter, H5muWriter
from apb2.parserV2.parse_quant.io.duckdb import DuckDBReader, DuckDBWriter
from apb2.parserV2.parse_quant.io.errors import UnsupportedResultFormatError
from apb2.parserV2.parse_quant.io.json_representation import write_result_with_representation
from apb2.parserV2.parse_quant.io.parquet_reader import ParquetReader
from apb2.parserV2.parse_quant.io.parquet_writer import ParquetLevelsWriter


class ParsedLevelsReader(Protocol):
    """Read one physical APB2 result, or one level of it, into its storage-neutral value."""

    def read(self, source: Path, level: ParsedLevelName | None = None, /) -> ParsedLevels: ...


class ParsedLevelsWriter(Protocol):
    """Persist one storage-neutral APB2 result."""

    def write(self, parsed: ParsedLevels, target: Path, /) -> None: ...


class ResultFormat(StrEnum):
    H5AD = "h5ad"
    H5MU = "h5mu"
    PARQUET = "parquet"
    DUCKDB = "duckdb"


_READERS: dict[ResultFormat, ParsedLevelsReader] = {
    ResultFormat.H5AD: H5adReader(),
    ResultFormat.H5MU: H5muReader(),
    ResultFormat.PARQUET: ParquetReader(),
    ResultFormat.DUCKDB: DuckDBReader(),
}

_WRITERS: dict[ResultFormat, ParsedLevelsWriter] = {
    ResultFormat.H5AD: H5adWriter(),
    ResultFormat.H5MU: H5muWriter(),
    ResultFormat.PARQUET: ParquetLevelsWriter(),
    ResultFormat.DUCKDB: DuckDBWriter(),
}

_FORMATS_BY_SUFFIX: dict[str, ResultFormat] = {
    ".h5ad": ResultFormat.H5AD,
    ".h5mu": ResultFormat.H5MU,
    ".parquet": ResultFormat.PARQUET,
    ".duckdb": ResultFormat.DUCKDB,
}


def reader_for(result_format: ResultFormat, /) -> ParsedLevelsReader:
    """Select the immutable reader registered for one explicit format."""
    return _READERS[result_format]


def writer_for(result_format: ResultFormat, /) -> ParsedLevelsWriter:
    """Select the immutable writer registered for one explicit format."""
    return _WRITERS[result_format]


def result_format_for(path: Path, /) -> ResultFormat:
    """Interpret the result suffix used by the path-inferred convenience API."""
    try:
        return _FORMATS_BY_SUFFIX[path.suffix.lower()]
    except KeyError as error:
        raise UnsupportedResultFormatError(
            f"unsupported result suffix {path.suffix!r}; expected one of "
            f"{sorted(_FORMATS_BY_SUFFIX)}"
        ) from error


def read_parsed_levels(source: Path, level: ParsedLevelName | None = None) -> ParsedLevels:
    """Read a result, or one of its levels, after inferring its format from the source path.

    Args:
        source: An h5ad, h5mu, Parquet or DuckDB result.
        level: The one level to read; ``None`` reads every level. Annotation tables are read
            either way; a feature relation only when every level it links is read.

    Returns:
        The stored result, or only the named level of it.

    Raises:
        ResultIOError: The result cannot be read, or has no such level.
    """
    return reader_for(result_format_for(source)).read(source, level)


def write_parsed_levels(parsed: ParsedLevels, target: Path, var_name_prefix: bool = True) -> None:
    """Write a result and its compact APB JSON representation sidecar.

    ``var_name_prefix`` applies to ``.h5mu``: ``False`` keeps each modality's var index as
    written, named after a single String key, instead of prefixing it with the level
    (``pfi:``, ``prt:``). The other formats prefix nothing.
    """
    result_format = result_format_for(target)
    writer = (
        H5muWriter(var_name_prefix)
        if result_format is ResultFormat.H5MU
        else writer_for(result_format)
    )
    with logged_step("write", format=result_format, path=target):
        write_result_with_representation(
            parsed,
            target,
            partial(writer.write, parsed, target),
        )
