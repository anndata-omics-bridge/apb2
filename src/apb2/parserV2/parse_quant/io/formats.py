"""Format-selected result readers, writers, and the storage-only reformat workflow."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol

from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import ParsedLevels
from apb2.parserV2.parse_quant.io.anndata_reader import H5adReader, H5muReader
from apb2.parserV2.parse_quant.io.anndata_writer import H5adWriter, H5muWriter
from apb2.parserV2.parse_quant.io.duckdb import DuckDBReader, DuckDBWriter
from apb2.parserV2.parse_quant.io.errors import UnsupportedResultFormatError
from apb2.parserV2.parse_quant.io.parquet_reader import ParquetReader
from apb2.parserV2.parse_quant.io.parquet_writer import ParquetLevelsWriter


class ParsedLevelsReader(Protocol):
    """Read one physical APB2 result into its storage-neutral value."""

    def read(self, source: Path, /) -> ParsedLevels: ...


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


def read_parsed_levels(source: Path, /) -> ParsedLevels:
    """Read a result after inferring its format from the source path."""
    return reader_for(result_format_for(source)).read(source)


def write_parsed_levels(parsed: ParsedLevels, target: Path, /) -> None:
    """Write a result after inferring its format from the destination path."""
    writer_for(result_format_for(target)).write(parsed, target)


def reformat(source: Path, target: Path, /) -> None:
    """Read one APB2 result and write the same value through another format adapter."""
    input_format = result_format_for(source)
    output_format = result_format_for(target)
    parsed = reader_for(input_format).read(source)
    for level, value in parsed.levels.items():
        logger.info(
            "level={} shape=({}, {}) layers={}",
            level,
            value.obs.frame.height,
            value.var.frame.height,
            list(value.layers),
        )
    writer_for(output_format).write(parsed, target)
    logger.info("reformatted {} -> {}", source, target)
