"""Public compiler objects for packaged and explicit parse rules."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from apb2.parserV2.detect_document import (
    DetectedRuleSet,
    LevelSelection,
    RuleUnavailableError,
    detect_rule_documents,
    guess_software,
    search_parameter_evidence,
    select_document_levels,
    software_slug,
)
from apb2.parserV2.parse_quant.parameters.source import Folder, InputSource, SingleFile
from apb2.parserV2.parse_quant.parser import ParserCollection
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.parser_factory import compile_level
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import RuleDocument, SearchParameterEvidence
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel

type ValidationChecks = Literal["standard", "strict"]


class ParseRuleCompiler:
    """Resolve vendor inputs once and compile their complete parser collection."""

    __slots__ = ("_checks", "_detection", "_parameter_evidence", "_parameters")
    _checks: ValidationChecks

    def __init__(
        self,
        data: Path,
        parameters_path: Path,
        *,
        requested_levels: Iterable[QuantificationLevel] | None = None,
        checks: ValidationChecks = "standard",
        software: str | None = None,
        parameters_software: str | None = None,
    ) -> None:
        """Resolve the source, parameters, vendor rules, and selected levels.

        Args:
            data: Vendor result table or canonical multi-file result directory.
            parameters_path: Vendor search-parameter file.
            requested_levels: Quantification levels to compile.
            checks: Canonical layer validation level applied during parsing.
            software: Optional vendor slug to select and verify.
            parameters_software: Optional independent parameter-parser slug.

        Raises:
            ValueError: The level request is empty or duplicated.
            RuleUnavailableError: The source does not identify the requested vendor rules.
        """
        levels = _validated_levels(LEVELS if requested_levels is None else requested_levels)
        source = _input_source(data)
        requested_software = None if software is None else software_slug(software)
        parameter_parser = parameters_software or requested_software or guess_software(source)
        if parameter_parser is None:
            raise RuleUnavailableError(
                f"could not auto-detect the vendor for {data}; pass software= or use an "
                "explicit rule document"
            )
        parameters = parse_params(parameters_path, software=parameter_parser)
        detection = detect_rule_documents(parameters, source, levels)
        if requested_software is not None and detection.software != requested_software:
            raise RuleUnavailableError(
                f"software {requested_software!r} does not match the detected vendor "
                f"{detection.software!r}"
            )
        self._parameters = parameters
        self._detection = detection
        self._parameter_evidence = search_parameter_evidence(parameters)
        self._checks = checks

    @property
    def parameters(self) -> Parameters:
        """Typed vendor parameters used for rule detection."""
        return self._parameters

    @property
    def detection(self) -> DetectedRuleSet:
        """Resolved software metadata and selected rule levels."""
        return self._detection

    def compile(self) -> ParserCollection:
        """Compile every detected selection into one collection parser."""
        return _compile_selections(
            self._detection.levels,
            self._parameter_evidence,
            self._checks,
        )


class ExplicitRuleCompiler:
    """Compile caller-supplied rule documents without vendor auto-detection."""

    __slots__ = ("_checks", "_parameter_evidence", "_selections")
    _checks: ValidationChecks

    def __init__(
        self,
        document: RuleDocument,
        source: InputSource,
        requested_levels: Iterable[QuantificationLevel],
        parameter_evidence: SearchParameterEvidence,
        *,
        checks: ValidationChecks = "standard",
    ) -> None:
        levels = _validated_levels(requested_levels)
        selections = select_document_levels(document, source, levels, parameter_evidence)
        if not selections:
            names = {
                level: document.declared(level).input.file_name
                for level in levels
                if level in document.levels
            }
            raise RuleUnavailableError(
                f"requested levels {list(levels)} are unavailable from {source.path}; "
                f"expected named tables: {names}"
            )
        self._selections = selections
        self._parameter_evidence = parameter_evidence
        self._checks = checks

    @property
    def selections(self) -> tuple[LevelSelection, ...]:
        """Resolved rule and physical source for each selected level."""
        return self._selections

    def compile(self) -> ParserCollection:
        """Compile every explicit selection into one collection parser."""
        return _compile_selections(
            self._selections,
            self._parameter_evidence,
            self._checks,
        )


def _compile_selections(
    selections: tuple[LevelSelection, ...],
    parameter_evidence: SearchParameterEvidence,
    checks: ValidationChecks,
) -> ParserCollection:
    parsers = tuple(
        compile_level(
            ParseRuleFacade(selection.document, selection.level, parameter_evidence),
            selection.source,
            checks,
        )
        for selection in selections
    )
    return ParserCollection(parsers)


def _validated_levels(
    requested_levels: Iterable[QuantificationLevel],
) -> tuple[QuantificationLevel, ...]:
    levels = tuple(requested_levels)
    if not levels:
        raise ValueError("at least one quantification level is required")
    duplicates = tuple(level for level in LEVELS if levels.count(level) > 1)
    if duplicates:
        raise ValueError(f"duplicate quantification levels: {duplicates}")
    requested = set(levels)
    return tuple(level for level in LEVELS if level in requested)


def _input_source(data: Path) -> InputSource:
    return Folder(path=data) if data.is_dir() else SingleFile(path=data)


__all__ = [
    "ExplicitRuleCompiler",
    "ParseRuleCompiler",
]
