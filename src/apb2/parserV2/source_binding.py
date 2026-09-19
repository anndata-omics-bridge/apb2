"""Bind physical vendor tables and inspect the evidence exposed by them."""

from __future__ import annotations

from pathlib import Path

from apb2.parserV2.parse_quant import delimited_input, excel_input, parquet_input
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFile,
    ExcelFormatContract,
    ExcelSourceEvidence,
    Folder,
    FrameSourceEvidence,
    InputContract,
    InputSource,
    LevelReadPlan,
    ParquetFormatContract,
    PhysicalFormatContract,
    SingleFile,
    SourceEvidence,
)


class BoundTable:
    """One caller source bound to one declared physical table interpretation."""

    __slots__ = ("_format", "_path", "_source")

    def __init__(self, source: InputSource, contract: InputContract) -> None:
        """Resolve the source path and its one unambiguous declared format."""
        if isinstance(source, SingleFile | DelimitedFile):
            path = source.path
        elif isinstance(source, Folder):
            if contract.file_name is None:
                raise IncompatibleSourceError(
                    f"{source.path} is a folder, but this rule declares no file_name"
                )
            path = source.path / contract.file_name
            if not path.is_file():
                raise IncompatibleSourceError(
                    f"{source.path} does not contain the declared file {contract.file_name!r}"
                )
        else:
            raise IncompatibleSourceError("multiple/prepared inputs require a preparation rule")

        suffix = path.suffix.lower()
        claiming = tuple(declared for declared in contract.formats if suffix in declared.extensions)
        if not claiming and len(contract.formats) == 1:
            selected = contract.formats[0]
        elif not claiming:
            extensions = sorted(
                extension for declared in contract.formats for extension in declared.extensions
            )
            raise IncompatibleSourceError(
                f"{path} has extension {suffix!r}, which no declared format accepts; "
                f"declared: {extensions}"
            )
        elif len(claiming) > 1:
            raise AmbiguousDialectError(
                f"{path} extension {suffix!r} is claimed by several declared formats"
            )
        else:
            selected = next(iter(claiming))

        self._source = source
        self._path = path
        self._format = selected

    @property
    def path(self) -> Path:
        """Concrete physical table path."""
        return self._path

    @property
    def format(self) -> PhysicalFormatContract:
        """Declared physical interpretation selected for the table."""
        return self._format

    def evidence(self, accepts: delimited_input.HeaderPredicate) -> SourceEvidence:
        """Observe physical evidence, including data-row number formatting when needed."""
        if isinstance(self._format, ParquetFormatContract):
            return parquet_input.schema_evidence(self._path)
        if isinstance(self._format, ExcelFormatContract):
            return excel_input.schema_evidence(self._path, self._format, accepts)
        if isinstance(self._source, DelimitedFile):
            return delimited_input.stated_evidence(self._source, self._format, accepts)
        return delimited_input.detected_evidence(self._path, self._format, accepts)

    def recognition_evidence(self, accepts: delimited_input.HeaderPredicate) -> SourceEvidence:
        """Observe only schema/header evidence for packaged-rule recognition."""
        if isinstance(self._format, ParquetFormatContract):
            return parquet_input.schema_evidence(self._path)
        if isinstance(self._format, ExcelFormatContract):
            return excel_input.schema_evidence(self._path, self._format, accepts)
        if isinstance(self._source, DelimitedFile):
            return delimited_input.stated_evidence(self._source, self._format, accepts)
        return delimited_input.detected_header_evidence(self._path, self._format, accepts)

    def reader(
        self,
        evidence: SourceEvidence,
        plan: LevelReadPlan,
    ) -> (
        delimited_input.DelimitedInputReader
        | excel_input.ExcelInputReader
        | parquet_input.ParquetInputReader
    ):
        """Construct the reader for observed evidence and one resolved level plan."""
        if isinstance(evidence, FrameSourceEvidence):
            return parquet_input.make_parquet_reader(self._path, plan)
        if isinstance(evidence, ExcelSourceEvidence):
            return excel_input.make_excel_reader(self._path, evidence, plan)
        return delimited_input.make_delimited_reader(self._path, evidence, plan)
