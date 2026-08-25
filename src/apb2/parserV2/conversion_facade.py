"""Application workflows for one Parser V2 source-to-AnnData conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import ValidationError

from apb2.parserV2.compile import (
    AnnDataOutput,
    ParseRuleCompiler,
    compile_mudata_parsers,
)
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    DetectedRuleDocument,
    RuleDetectionError,
    detect_rule_document,
    guess_software,
    search_parameter_evidence,
    software_slug,
)
from apb2.parserV2.parse_quant.anndata_writer import (
    AnnDataLayerContractError,
    MuDataLevelError,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.axis_columns import AxisCoercionError, ColumnComputationError
from apb2.parserV2.parse_quant.data.layer_columns import StorageLabelError
from apb2.parserV2.parse_quant.data.parsed import JsonValue, ParsedLevel
from apb2.parserV2.parse_quant.duplicates import AggregateTypeError, DuplicateCellError
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.fragments import PackedLengthError
from apb2.parserV2.parse_quant.modifications import (
    PackedSiteMismatchError,
    UnknownModificationError,
)
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_quant.parser import AxisShapeError, CanonicalKeyCollisionError
from apb2.parserV2.parse_rule_facade import PRODUCER, ParseRuleFacade
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel

type AnnDataChecks = Literal["standard", "strict"]
type RuleSelectionMethod = Literal["software_version", "columns", "rule_config"]


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


_EXPECTED_CONVERSION_FAILURES = (
    AggregateTypeError,
    AmbiguousDialectError,
    AnnDataLayerContractError,
    AxisCoercionError,
    AxisShapeError,
    CanonicalKeyCollisionError,
    ColumnComputationError,
    DuplicateCellError,
    IncompatibleSourceError,
    json.JSONDecodeError,
    MuDataLevelError,
    OSError,
    PackedLengthError,
    PackedSiteMismatchError,
    ParamsError,
    RuleDetectionError,
    RuleNotApplicable,
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
    """Convert one level using an explicitly supplied schema-0.3 rule document."""
    try:
        document, parameters, evidence = _explicit_conversion_inputs(
            rule_config=rule_config,
            parameters_path=parameters_path,
            parameters_software=parameters_software,
        )
        parsed = _parse_and_write(
            data=data,
            level=level,
            output=output,
            document=document,
            evidence=evidence,
            checks=checks,
            selection_method="rule_config",
            parameters=parameters,
            parameters_path=parameters_path,
        )
        return _conversion_summary(
            {level: parsed},
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
    """Convert every compatible level of an explicit schema-0.3 document to MuData."""
    try:
        document, parameters, evidence = _explicit_conversion_inputs(
            rule_config=rule_config,
            parameters_path=parameters_path,
            parameters_software=parameters_software,
        )
        parsed = _parse_all_and_write(
            data=data,
            output=output,
            document=document,
            evidence=evidence,
            checks=checks,
            selection_method="rule_config",
            parameters=parameters,
            parameters_path=parameters_path,
        )
        return _conversion_summary(
            parsed.levels,
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
        detected, parameters, method = _packaged_conversion_inputs(
            data=data,
            parameters_path=parameters_path,
            software=software,
            parameters_software=parameters_software,
        )
        parsed = _parse_and_write(
            data=data,
            level=level,
            output=output,
            document=detected.document,
            evidence=search_parameter_evidence(parameters),
            checks=checks,
            selection_method=method,
            parameters=parameters,
            parameters_path=parameters_path,
        )
        return _conversion_summary(
            {level: parsed},
            software=detected.software,
            version=detected.version,
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
    """Detect one packaged document and convert every compatible level to MuData."""
    try:
        detected, parameters, method = _packaged_conversion_inputs(
            data=data,
            parameters_path=parameters_path,
            software=software,
            parameters_software=parameters_software,
        )
        parsed = _parse_all_and_write(
            data=data,
            output=output,
            document=detected.document,
            evidence=search_parameter_evidence(parameters),
            checks=checks,
            selection_method=method,
            parameters=parameters,
            parameters_path=parameters_path,
        )
        return _conversion_summary(
            parsed.levels,
            software=detected.software,
            version=detected.version,
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


def _packaged_conversion_inputs(
    *,
    data: Path,
    parameters_path: Path,
    software: str | None,
    parameters_software: str | None,
) -> tuple[DetectedRuleDocument, Parameters, RuleSelectionMethod]:
    """Parse parameters and detect one packaged rule document once."""
    source = SingleFile(path=data)
    parser_slug = parameters_software or software or guess_software(source)
    if parser_slug is None:
        raise ConversionError(
            f"could not auto-detect the vendor for {data}; pass --software SLUG "
            "or --rule-config PATH"
        )
    parameters = parse_params(parameters_path, software=parser_slug)
    detected = detect_rule_document(parameters, source)
    if software is not None and detected.software != software:
        raise ConversionError(
            f"--software {software!r} does not match the detected vendor {detected.software!r}"
        )
    method: RuleSelectionMethod = "software_version" if detected.version is not None else "columns"
    return detected, parameters, method


def _parse_and_write(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    document: RuleDocument,
    evidence: SearchParameterEvidence,
    checks: AnnDataChecks,
    selection_method: RuleSelectionMethod,
    parameters: Parameters | None,
    parameters_path: Path | None,
) -> ParsedLevel:
    facade = ParseRuleFacade(document, level, evidence)
    parser = ParseRuleCompiler(facade=facade, output=AnnDataOutput(checks=checks)).compile(
        SingleFile(path=data)
    )
    parsed = parser.parse()
    parsed.uns.update(_shared_parse_provenance(selection_method, parameters, parameters_path))
    parser.convert(parsed, output)
    return parsed


def _parse_all_and_write(
    *,
    data: Path,
    output: Path,
    document: RuleDocument,
    evidence: SearchParameterEvidence,
    checks: AnnDataChecks,
    selection_method: RuleSelectionMethod,
    parameters: Parameters | None,
    parameters_path: Path | None,
) -> ParsedLevels:
    """Parse every compatible level independently, then write their storage composition."""
    parsers, writer = compile_mudata_parsers(
        document=document,
        levels=document.levels,
        parameter_evidence=evidence,
        source=SingleFile(path=data),
        checks=checks,
    )
    shared = _shared_parse_provenance(selection_method, parameters, parameters_path)
    levels: dict[QuantificationLevel, ParsedLevel] = {}
    for parser in parsers:
        parsed = parser.parse()
        parsed.uns.update(shared)
        level = cast(QuantificationLevel, parser.level)
        levels[level] = parsed
    combined = ParsedLevels(
        levels=levels,
        uns={
            "produced_by": PRODUCER,
            **shared,
            "quantification_levels": [str(level) for level in levels],
        },
    )
    writer.write(combined, output)
    return combined


def _shared_parse_provenance(
    selection_method: RuleSelectionMethod,
    parameters: Parameters | None,
    parameters_path: Path | None,
) -> dict[str, JsonValue]:
    """Shared selection and parameter facts written at one or several levels."""
    provenance: dict[str, JsonValue] = {"rule_selection_method": selection_method}
    if parameters is None:
        return provenance
    if parameters_path is None:
        raise ValueError("parsed search parameters require their source path")
    provenance.update(
        {
            "search_parameters_version_status": (
                "missing" if parameters.software_version is None else "present"
            ),
            "search_parameters_path": str(parameters_path),
            "search_parameters": json.dumps(parameters.model_dump(mode="json")),
        }
    )
    return provenance


def _conversion_summary(
    levels: Mapping[QuantificationLevel, ParsedLevel],
    *,
    software: str,
    version: str | None,
) -> ConversionSummary:
    return ConversionSummary(
        software=software,
        version=version,
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
