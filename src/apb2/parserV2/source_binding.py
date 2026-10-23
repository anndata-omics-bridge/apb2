"""Bind physical vendor tables and inspect the evidence exposed by them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from apb2.parserV2.parse_quant import delimited_input, excel_input, parquet_input
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.operations import WorkingParseConfiguration
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    ExcelFormatContract,
    ExcelSourceEvidence,
    FrameSourceEvidence,
    InputContract,
    LevelReadPlan,
    ParquetFormatContract,
    PhysicalFormatContract,
    SourceEvidence,
)
from apb2.parserV2.parse_quant.parameters.vendor_file import GzipFile, VendorFile
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_rule_facade import ParseRuleFacade


class BoundTable:
    """One physical table file bound to one declared interpretation."""

    __slots__ = ("_file", "_format")

    def __init__(self, file: VendorFile, contract: InputContract) -> None:
        """Resolve the file's one unambiguous declared format."""
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
        if isinstance(file, GzipFile) and not isinstance(selected, DelimitedFormatContract):
            raise IncompatibleSourceError(
                f"{path}: decompress gzipped Parquet or Excel input first"
            )

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


@dataclass(frozen=True, slots=True)
class PhysicalTable:
    """One vendor table file, bound per level to that level's declared format."""

    file: VendorFile

    @property
    def path(self) -> Path:
        """The table file on disk."""
        return self.file.path

    def header(self, working: WorkingParseConfiguration) -> tuple[str, ...]:
        """The header the level accepts, read without inspecting data rows."""
        return BoundTable(self.file, working.input).header(working.accepts_header)

    def parser(self, facade: ParseRuleFacade, checks: Literal["standard", "strict"]) -> Parser:
        """Resolve one level against this file's evidence and bind the reader it needs."""
        working = facade.working_parameters
        bound = BoundTable(self.file, working.input)
        evidence = bound.evidence(working.accepts_header)
        strategy = facade.resolve_source(evidence, checks=checks)
        return Parser(bound.reader(evidence, strategy.read), strategy)
