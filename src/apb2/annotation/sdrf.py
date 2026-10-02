"""SDRF-Proteomics source interpretation and dataset-bound annotation behavior."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from apb2.annotation.application.policies import (
    AnnotationApplication,
    KeepUnmatchedAnnotation,
    record_annotation_provenance,
)
from apb2.annotation.data.model import (
    AnnotationError,
    AnnotationMatches,
    AnnotationResult,
    AnnotationTable,
    LoadedAnnotationSource,
)
from apb2.annotation.matching.core import (
    annotation_matching_for,
    make_annotation_table,
    match_annotation,
    normalize_mass_spec_basename,
)
from apb2.parserV2.parse_quant.data.parsed import JsonValue, ParsedLevels

_SOURCE_NAME = "source name"
_DATA_FILE = "comment[data file]"
_LABEL = "comment[label]"
_LABEL_FREE_NAME = "label free sample"
_LABEL_FREE_ACCESSION = "MS:1002038"
_DATA_FILE_BASENAME = "__sdrf_data_file_basename"


@dataclass(frozen=True, slots=True)
class SdrfAnnotationParameters:
    """User-selected SDRF application behavior."""

    application: AnnotationApplication = field(default_factory=KeepUnmatchedAnnotation)


class SdrfSource:
    """One loaded label-free annotation source that follows the SDRF-Proteomics columns.

    Header lookup ignores case and surrounding whitespace. Repeated headers, such as several
    ``characteristics[spiked compound]`` columns, remain separately addressable in file order.
    """

    __slots__ = ("_columns", "_source")

    def __init__(self, source: LoadedAnnotationSource, /) -> None:
        if not sdrf_signature(source):
            raise AnnotationError(
                f"SDRF annotation requires {_SOURCE_NAME!r} and {_DATA_FILE!r} columns; "
                f"headers={list(source.headers)}"
            )
        columns: dict[str, list[str]] = {}
        for header, column in zip(source.headers, source.frame.columns, strict=True):
            columns.setdefault(_header_key(header), []).append(column)
        self._source = source
        self._columns = {header: tuple(names) for header, names in columns.items()}
        self._require_label_free()

    @property
    def source(self) -> LoadedAnnotationSource:
        """Return the loaded source this SDRF view reads."""
        return self._source

    def columns(self, header: str, /) -> tuple[str, ...]:
        """Return the frame column of every occurrence of one SDRF header, in file order."""
        return self._columns.get(_header_key(header), ())

    def column(self, header: str, /) -> str:
        """Return the frame column of a header that must occur exactly once."""
        found = self.columns(header)
        if len(found) != 1:
            raise AnnotationError(
                f"SDRF header {header!r} must occur exactly once; found {len(found)}"
            )
        return found[0]

    def data_file_basenames(self) -> pl.Series:
        """Return each data file without directories or mass-spectrometry extension.

        Vendor tables usually name a run by this basename rather than by the file name that
        ``comment[data file]`` records.
        """
        values = self._source.frame.get_column(self.column(_DATA_FILE)).cast(pl.String)
        return pl.Series(
            _DATA_FILE_BASENAME,
            [
                None if value is None else normalize_mass_spec_basename((value,))[0]
                for value in values.to_list()
            ],
            dtype=pl.String,
        )

    def table(self) -> AnnotationTable:
        """Key rows by data file and accept the data-file basename as an exact alias."""
        frame = self._source.frame.with_columns(self.data_file_basenames())
        return make_annotation_table(
            frame,
            (self.column(_DATA_FILE),),
            (_DATA_FILE_BASENAME,),
            self._source.origin,
        )

    def _require_label_free(self) -> None:
        for column in self.columns(_LABEL):
            labels = self._source.frame.get_column(column).cast(pl.String).to_list()
            multiplexed = sorted(
                {label for label in labels if label is None or not _is_label_free(label)},
                key=str,
            )
            if multiplexed:
                raise AnnotationError(
                    "SDRF annotation supports label-free rows only; multiplexed rows need "
                    f"run-and-channel observation keys; labels={multiplexed}"
                )


@dataclass(frozen=True, slots=True)
class SdrfAnnotationParser:
    """An SDRF parser bound to one already loaded source."""

    source: SdrfSource
    parameters: SdrfAnnotationParameters

    def parse(self, parsed: ParsedLevels, /) -> SdrfAnnotation:
        """Validate, match, and construct only an applicable dataset annotation."""
        table = self.source.table()
        matches = match_annotation(
            table,
            parsed,
            {name: annotation_matching_for(level) for name, level in parsed.levels.items()},
        )
        self.parameters.application.validate(matches)
        return SdrfAnnotation(
            table=table,
            parsed=parsed,
            matches=matches,
            application=self.parameters.application,
            columns=_column_map(self.source.source, table),
        )


@dataclass(frozen=True, slots=True)
class SdrfAnnotation:
    """A validated SDRF annotation bound to one parsed dataset."""

    table: AnnotationTable
    parsed: ParsedLevels
    matches: AnnotationMatches
    application: AnnotationApplication
    columns: tuple[tuple[str, str], ...]

    def annotate(self) -> AnnotationResult:
        """Apply the prevalidated behavior and record each verbatim header's obs column."""
        result = self.application.apply(self.parsed, self.matches)
        mapping: list[JsonValue] = [
            {"header": header, "column": column} for header, column in self.columns
        ]
        return record_annotation_provenance(
            result,
            "sdrf",
            self.table.origin,
            metadata={"columns": mapping},
        )


def sdrf_signature(source: LoadedAnnotationSource, /) -> bool:
    """Return whether the source headers identify SDRF-Proteomics input."""
    headers = {_header_key(header) for header in source.headers}
    return {_SOURCE_NAME, _DATA_FILE} <= headers


def _column_map(
    source: LoadedAnnotationSource,
    table: AnnotationTable,
) -> tuple[tuple[str, str], ...]:
    identifiers = {*table.key_columns, *table.alias_columns}
    return tuple(
        (header, projected)
        for header, projected in zip(source.headers, table.frame.columns, strict=False)
        if projected not in identifiers
    )


def _is_label_free(label: str) -> bool:
    fields = {
        key.strip().upper(): value.strip()
        for key, separator, value in (part.partition("=") for part in label.split(";"))
        if separator
    }
    name = fields.get("NT", label.strip())
    return name.lower() == _LABEL_FREE_NAME or fields.get("AC", "").upper() == _LABEL_FREE_ACCESSION


def _header_key(header: str) -> str:
    return header.strip().lower()
