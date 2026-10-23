"""Bind physical vendor tables and inspect the evidence exposed by them."""

from __future__ import annotations

from pathlib import Path

from apb2.parserV2.parse_quant import delimited_input, excel_input, parquet_input
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
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

    __slots__ = ("_file", "_format")

    def __init__(self, source: InputSource, contract: InputContract) -> None:
        """Resolve the source file and its one unambiguous declared format."""
        if isinstance(source, SingleFile):
            file = source.file
        elif isinstance(source, Folder):
            if contract.file_name is None:
                raise IncompatibleSourceError(
                    f"{source.path} is a folder, but this rule declares no file_name"
                )
            member = source.member(contract.file_name)
            if member is None:
                raise IncompatibleSourceError(
                    f"{source.path} does not contain the declared file {contract.file_name!r}"
                )
            file = member
        else:
            raise IncompatibleSourceError("multiple/prepared inputs require a preparation rule")

        path = file.path
        suffix = file.suffix
        extensions = {
            extension for declared in contract.formats for extension in declared.extensions
        }
        if suffix == ".parquet" and ".parquet" not in extensions:
            raise IncompatibleSourceError(f"{path} is Parquet but the rule is delimited")
        if suffix != ".parquet" and extensions == {".parquet"}:
            raise IncompatibleSourceError(f"{path} is not a Parquet file")
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

        self._file = file
        self._format = selected

    @property
    def path(self) -> Path:
        """Concrete physical table path."""
        return self._file.path

    @property
    def format(self) -> PhysicalFormatContract:
        """Declared physical interpretation selected for the table."""
        return self._format

    def evidence(self, accepts: delimited_input.HeaderPredicate) -> SourceEvidence:
        """Observe physical evidence, including data-row number formatting when needed."""
        if isinstance(self._format, ParquetFormatContract):
            return parquet_input.schema_evidence(self._file.path)
        if isinstance(self._format, ExcelFormatContract):
            return excel_input.schema_evidence(self._file.path, self._format, accepts)
        return delimited_input.detected_evidence(self._file, self._format, accepts)

    def header(self, accepts: delimited_input.HeaderPredicate) -> tuple[str, ...]:
        """Observe only the header, without inspecting data rows, for rule recognition."""
        if isinstance(self._format, DelimitedFormatContract):
            return delimited_input.detected_header(self._file, self._format, accepts)
        return self.evidence(accepts).columns

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
            return parquet_input.ParquetInputReader(self._file.path, evidence.columns, plan)
        if isinstance(evidence, ExcelSourceEvidence):
            return excel_input.ExcelInputReader(self._file.path, evidence, plan)
        return delimited_input.DelimitedInputReader(self._file, evidence, plan)
