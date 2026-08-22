"""Application workflows for one Parser V2 source-to-AnnData conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from apb2.parserV2.compile import AnnDataOutput, ParseRuleCompiler
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    detect_rule_document,
    guess_software,
    search_parameter_evidence,
    software_slug,
)
from apb2.parserV2.parse_quant.data.parsed import ParsedLevel
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.search_parameters.model import Parameters
from apb2.parserV2.search_parameters.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import RuleDocument, SearchParameterEvidence
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel

type AnnDataChecks = Literal["standard", "strict"]
type RuleSelectionMethod = Literal["software_version", "columns", "rule_config"]


class SoftwareGuessError(ValueError):
    """The source header does not identify one parameter-parser vendor."""


class SoftwareMismatchError(ValueError):
    """The explicitly requested software conflicts with the detected rule document."""


@dataclass(frozen=True, slots=True)
class ConversionResult:
    """The written parse and the software evidence selected for it."""

    parsed: ParsedLevel
    software: str
    version: str | None


def convert_from_rule_config(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    rule_config: Path,
    parameters_path: Path | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionResult:
    """Convert one level using an explicitly supplied schema-0.3 rule document."""
    document = load_rule_document(rule_config)
    parameters = (
        parse_params(
            parameters_path,
            software=parameters_software or software_slug(document.software_name),
        )
        if parameters_path is not None
        else None
    )
    evidence = (
        search_parameter_evidence(parameters)
        if parameters is not None
        else UNKNOWN_SEARCH_PARAMETERS
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
    return ConversionResult(
        parsed=parsed,
        software=software_slug(document.software_name),
        version=parameters.software_version if parameters is not None else None,
    )


def convert_from_packaged_rules(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    parameters_path: Path,
    software: str | None,
    parameters_software: str | None,
    checks: AnnDataChecks,
) -> ConversionResult:
    """Detect a packaged document from the source and parameter file, then convert it."""
    source = SingleFile(path=data)
    parser_slug = parameters_software or software or guess_software(source)
    if parser_slug is None:
        raise SoftwareGuessError(
            f"could not auto-detect the vendor for {data}; pass --software SLUG "
            "or --rule-config PATH"
        )
    parameters = parse_params(parameters_path, software=parser_slug)
    detected = detect_rule_document(parameters, source)
    if software is not None and detected.software != software:
        raise SoftwareMismatchError(
            f"--software {software!r} does not match the detected vendor {detected.software!r}"
        )
    method: RuleSelectionMethod = "software_version" if detected.version is not None else "columns"
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
    return ConversionResult(
        parsed=parsed,
        software=detected.software,
        version=detected.version,
    )


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
    parsed.uns["rule_selection_method"] = selection_method
    if parameters is not None:
        parsed.uns.update(
            {
                "search_parameters_version_status": (
                    "missing" if parameters.software_version is None else "present"
                ),
                "search_parameters_path": str(parameters_path),
                "search_parameters": json.dumps(parameters.model_dump(mode="json")),
            }
        )
    parser.convert(parsed, output)
    return parsed
