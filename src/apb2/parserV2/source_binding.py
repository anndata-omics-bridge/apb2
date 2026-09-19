"""Bind physical vendor tables and inspect the evidence exposed by them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from apb2.parserV2.parse_quant import delimited_input, excel_input, parquet_input
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.axis import (
    EmbeddedSiteListModificationConfig,
    SiteListModificationConfig,
)
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
from apb2.parserV2.parse_quant.parameters.working import (
    ColumnLabeledFragmentLayout,
    PositionalFragmentLayout,
    WideSourceLayout,
    WorkingParseConfiguration,
)


@dataclass(frozen=True, slots=True)
class BoundTable:
    """One concrete file and the declared physical interpretation selected for it."""

    path: Path
    format: PhysicalFormatContract


def bind_source(source: InputSource, contract: InputContract) -> BoundTable:
    """Resolve one complete caller-supplied source into one concrete table."""
    return _bind_path(_path_of(source, contract), contract)


def source_evidence(
    source: InputSource,
    bound: BoundTable,
    accepts: delimited_input.HeaderPredicate,
) -> SourceEvidence:
    """Observe the physical evidence one bound table exposes before a full read."""
    if isinstance(bound.format, ParquetFormatContract):
        return parquet_input.schema_evidence(bound.path)
    if isinstance(bound.format, ExcelFormatContract):
        return excel_input.schema_evidence(bound.path, bound.format, accepts)
    if isinstance(source, DelimitedFile):
        return delimited_input.stated_evidence(source, bound.format, accepts)
    return delimited_input.detected_evidence(bound.path, bound.format, accepts)


def source_recognition_evidence(
    source: InputSource,
    bound: BoundTable,
    accepts: delimited_input.HeaderPredicate,
) -> SourceEvidence:
    """Observe only schema/header evidence for packaged-rule recognition."""
    if isinstance(bound.format, ParquetFormatContract):
        return parquet_input.schema_evidence(bound.path)
    if isinstance(bound.format, ExcelFormatContract):
        return excel_input.schema_evidence(bound.path, bound.format, accepts)
    if isinstance(source, DelimitedFile):
        return delimited_input.stated_evidence(source, bound.format, accepts)
    return delimited_input.detected_header_evidence(bound.path, bound.format, accepts)


def make_reader(
    bound: BoundTable,
    evidence: SourceEvidence,
    plan: LevelReadPlan,
) -> (
    delimited_input.DelimitedInputReader
    | excel_input.ExcelInputReader
    | parquet_input.ParquetInputReader
):
    """Construct the reader selected by the bound source and observed evidence."""
    if isinstance(evidence, FrameSourceEvidence):
        return parquet_input.make_parquet_reader(bound.path, plan)
    if isinstance(evidence, ExcelSourceEvidence):
        return excel_input.make_excel_reader(bound.path, evidence, plan)
    return delimited_input.make_delimited_reader(bound.path, evidence, plan)


def header_predicate(
    working: WorkingParseConfiguration,
) -> delimited_input.HeaderPredicate:
    """Return the minimum header predicate required by one level declaration."""
    exact = frozenset(
        {
            *(
                selection.source
                for axis in (working.obs, working.var)
                for selection in axis.columns.required_selections
            ),
            *_modification_sources(working),
            *_packed_sources(working),
        }
    )
    required_layers = tuple(layer.source for layer in working.measurements.required_layers)
    wide = isinstance(working.source_layout, WideSourceLayout)

    def accepts(header: tuple[str, ...]) -> bool:
        present = frozenset(header)
        if not exact <= present:
            return False
        if not wide:
            return all(source in present for source in required_layers)
        return all(
            any(re.compile(pattern).match(name) for name in header) for pattern in required_layers
        )

    return accepts


def _modification_sources(working: WorkingParseConfiguration) -> tuple[str, ...]:
    return tuple(
        column
        for config in working.modifications
        for column in (
            (config.sequence_column, config.modification_column, config.site_column)
            if isinstance(config, SiteListModificationConfig)
            else (
                (config.sequence_column, config.modification_column)
                if isinstance(config, EmbeddedSiteListModificationConfig)
                else (config.source_column,)
            )
        )
    )


def _packed_sources(working: WorkingParseConfiguration) -> tuple[str, ...]:
    layout = working.source_layout
    if isinstance(layout, ColumnLabeledFragmentLayout):
        return (layout.label_source, *layout.packed_value_sources)
    if isinstance(layout, PositionalFragmentLayout):
        return layout.packed_value_sources
    return ()


def _path_of(source: InputSource, contract: InputContract) -> Path:
    if isinstance(source, SingleFile | DelimitedFile):
        return source.path
    if isinstance(source, Folder):
        return _folder_path(source, contract)
    raise IncompatibleSourceError("multiple/prepared inputs require a preparation rule")


def _folder_path(source: Folder, contract: InputContract) -> Path:
    if contract.file_name is None:
        raise IncompatibleSourceError(
            f"{source.path} is a folder, but this rule declares no file_name"
        )
    path = source.path / contract.file_name
    if not path.is_file():
        raise IncompatibleSourceError(
            f"{source.path} does not contain the declared file {contract.file_name!r}"
        )
    return path


def _bind_path(path: Path, contract: InputContract) -> BoundTable:
    suffix = path.suffix.lower()
    claiming = [declared for declared in contract.formats if suffix in set(declared.extensions)]
    if not claiming and len(contract.formats) == 1:
        return BoundTable(path=path, format=contract.formats[0])
    if not claiming:
        declared = sorted(extension for entry in contract.formats for extension in entry.extensions)
        raise IncompatibleSourceError(
            f"{path} has extension {suffix!r}, which no declared format accepts; declared: "
            f"{declared}"
        )
    if len(claiming) > 1:
        raise AmbiguousDialectError(
            f"{path} extension {suffix!r} is claimed by several declared formats"
        )
    return BoundTable(path=path, format=claiming[0])
