"""File-to-file conversion workflow owned by the ``apb2`` command."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import ValidationError

from apb2.parserV2.compile import (
    ExplicitRuleCompiler,
    ParseRuleCompiler,
)
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    LevelSelection,
    RuleDetectionError,
    search_parameter_evidence,
    software_slug,
)
from apb2.parserV2.parse_quant.axis_columns import AxisCoercionError, ColumnComputationError
from apb2.parserV2.parse_quant.data.layer_columns import StorageLabelError
from apb2.parserV2.parse_quant.data.parsed import ParsedLevel, ParsedLevels
from apb2.parserV2.parse_quant.duplicates import AggregateTypeError, DuplicateCellError
from apb2.parserV2.parse_quant.errors import (
    AmbiguousDialectError,
    IncompatibleSourceError,
)
from apb2.parserV2.parse_quant.fragments import PackedLengthError
from apb2.parserV2.parse_quant.io import formats
from apb2.parserV2.parse_quant.io.errors import ResultIOError
from apb2.parserV2.parse_quant.modifications import (
    PackedSiteMismatchError,
    UnknownModificationError,
)
from apb2.parserV2.parse_quant.observation_groups import group_observations
from apb2.parserV2.parse_quant.parameters.source import Folder, InputSource, SingleFile
from apb2.parserV2.parse_quant.parser import (
    AxisShapeError,
    CanonicalKeyCollisionError,
    ParserCollection,
)
from apb2.parserV2.prepare_source import InputPreparationError
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel

type AnnDataChecks = Literal["standard", "strict"]
ReformatError = ResultIOError


def reformat_result(source: Path, target: Path, /) -> None:
    """Run the storage-only result workflow used by the command line."""
    formats.reformat(source, target)


class ConversionError(ValueError):
    """An expected input, selection, parsing, or writing failure."""


@dataclass(frozen=True, slots=True)
class LevelConversionSummary:
    """CLI-facing dimensions of one converted quantification level."""

    level: QuantificationLevel
    observation_count: int
    variable_count: int
    layer_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConversionSummary:
    """Small CLI-facing summary of one completed single- or multi-level conversion."""

    software: str
    version: str | None
    levels: tuple[LevelConversionSummary, ...]
    outputs: tuple[Path, ...]


_EXPECTED_CONVERSION_FAILURES = (
    InputPreparationError,
    AggregateTypeError,
    AmbiguousDialectError,
    AxisCoercionError,
    AxisShapeError,
    CanonicalKeyCollisionError,
    ColumnComputationError,
    DuplicateCellError,
    IncompatibleSourceError,
    json.JSONDecodeError,
    OSError,
    PackedLengthError,
    PackedSiteMismatchError,
    ParamsError,
    RuleDetectionError,
    RuleNotApplicable,
    ResultIOError,
    StorageLabelError,
    UnknownModificationError,
    ValidationError,
)


def convert_from_rule_config(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    rule_config: Path,
    parameters_path: Path | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Convert one level using an explicitly supplied schema-0.7 rule document."""
    try:
        document, parameters, evidence = _explicit_conversion_inputs(
            rule_config=rule_config,
            parameters_path=parameters_path,
            parameters_software=parameters_software,
        )
        compiler = ExplicitRuleCompiler(
            document=document,
            source=_input_source(data),
            requested_levels=(level,),
            parameter_evidence=evidence,
            checks=checks,
        )
        parsed, outputs = _parse_and_write(
            output,
            compiler.compile(),
            compiler.selections,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=software_slug(document.software_name),
            version=parameters.software_version if parameters is not None else None,
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_all_from_rule_config(
    *,
    data: Path,
    output: Path,
    rule_config: Path,
    parameters_path: Path | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Convert every compatible level of an explicit schema-0.7 document."""
    try:
        document, parameters, evidence = _explicit_conversion_inputs(
            rule_config=rule_config,
            parameters_path=parameters_path,
            parameters_software=parameters_software,
        )
        compiler = ExplicitRuleCompiler(
            document=document,
            source=_input_source(data),
            requested_levels=document.levels,
            parameter_evidence=evidence,
            checks=checks,
        )
        parsed, outputs = _parse_and_write(
            output,
            compiler.compile(),
            compiler.selections,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=software_slug(document.software_name),
            version=parameters.software_version if parameters is not None else None,
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_from_packaged_rules(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    parameters_path: Path,
    software: str | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Detect a packaged document from the source and parameter file, then convert it."""
    try:
        compiler = ParseRuleCompiler(
            data,
            parameters_path,
            requested_levels=(level,),
            checks=checks,
            software=software,
            parameters_software=parameters_software,
        )
        parsed, outputs = _parse_and_write(
            output,
            compiler.compile(),
            compiler.detection.levels,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=compiler.detection.software,
            version=compiler.detection.version,
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_all_from_packaged_rules(
    *,
    data: Path,
    output: Path,
    parameters_path: Path,
    software: str | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Detect and convert every compatible packaged level from one file or folder."""
    try:
        compiler = ParseRuleCompiler(
            data,
            parameters_path,
            requested_levels=LEVELS,
            checks=checks,
            software=software,
            parameters_software=parameters_software,
        )
        parsed, outputs = _parse_and_write(
            output,
            compiler.compile(),
            compiler.detection.levels,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=compiler.detection.software,
            version=compiler.detection.version,
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def _explicit_conversion_inputs(
    *,
    rule_config: Path,
    parameters_path: Path | None,
    parameters_software: str | None,
) -> tuple[RuleDocument, Parameters | None, SearchParameterEvidence]:
    """Load one explicit document and its optional parameter evidence once."""
    document = load_rule_document(rule_config)
    if parameters_path is None:
        return document, None, UNKNOWN_SEARCH_PARAMETERS
    parameters = parse_params(
        parameters_path,
        software=parameters_software or software_slug(document.software_name),
    )
    return document, parameters, search_parameter_evidence(parameters)


def _input_source(data: Path) -> InputSource:
    """Bind a canonical vendor-result folder or one direct input file."""
    return Folder(path=data) if data.is_dir() else SingleFile(path=data)


def _parse_and_write(
    output: Path,
    parser: ParserCollection,
    selections: tuple[LevelSelection, ...],
) -> tuple[ParsedLevels, tuple[Path, ...]]:
    """Parse selected levels, align compatible observations, and write each resolution."""
    for selection in selections:
        logger.info("level={} source={}", selection.level, selection.source_path)
    combined = parser.parse()
    groups = group_observations(combined)
    outputs = _group_output_paths(groups, output)
    for group, target in zip(groups, outputs, strict=True):
        formats.write_parsed_levels(group, target)
    combined.levels = {
        name: group.levels[name] for name in LEVELS for group in groups if name in group.levels
    }
    return combined, outputs


def _group_output_paths(groups: tuple[ParsedLevels, ...], output: Path) -> tuple[Path, ...]:
    """Name split artifacts before writing; never leave a misleading combined result."""
    if len(groups) == 1:
        return (output,)
    qualifiers = [
        re.sub(
            r"[^a-z0-9_]+", "_", "_".join(next(iter(group.levels.values())).obs.key_columns).lower()
        )
        for group in groups
    ]
    if len(set(qualifiers)) != len(qualifiers) or not all(qualifiers):
        raise ConversionError("observation keys do not produce distinct output filenames")
    outputs = tuple(output.with_name(f"{output.stem}.{key}{output.suffix}") for key in qualifiers)
    if output.exists():
        raise ConversionError(f"split conversion would leave an existing combined target: {output}")
    return outputs


def _conversion_summary(
    levels: Mapping[QuantificationLevel, ParsedLevel],
    *,
    software: str,
    version: str | None,
    outputs: tuple[Path, ...],
) -> ConversionSummary:
    return ConversionSummary(
        software=software,
        version=version,
        outputs=outputs,
        levels=tuple(
            LevelConversionSummary(
                level=level,
                observation_count=parsed.obs.frame.height,
                variable_count=parsed.var.frame.height,
                layer_names=tuple(parsed.layers),
            )
            for level, parsed in levels.items()
        ),
    )
