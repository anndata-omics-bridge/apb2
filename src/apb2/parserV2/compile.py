"""Public compiler objects for packaged and explicit parse rules."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError

from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    DetectedRuleSet,
    LevelSelection,
    RuleUnavailableError,
    detect_rule_documents,
    detect_software_rules,
    guess_software,
    search_parameter_evidence,
    select_document_levels,
    software_slug,
)
from apb2.parserV2.parse_quant.data.errors import ConversionError
from apb2.parserV2.parse_quant.parameters.source import Folder, InputSource, SingleFile
from apb2.parserV2.parse_quant.parser import ParserCollection
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel

type ValidationChecks = Literal["standard", "strict"]


class ParseRuleCompiler:
    """Resolve vendor inputs once and compile their complete parser collection."""

    __slots__ = ("_detection", "_parameters")

    def __init__(
        self,
        data: Path,
        parameters_path: Path | None,
        requested_levels: Iterable[QuantificationLevel] | None = None,
        checks: ValidationChecks = "standard",
        software: str | None = None,
    ) -> None:
        """Resolve the source, parameters, vendor rules, and selected levels.

        Args:
            data: Vendor result table or canonical multi-file result directory.
            parameters_path: Vendor search-parameter file, or ``None`` for a packaged
                parameter-free rule selected by ``software``.
            requested_levels: Quantification levels to compile.
            checks: Canonical layer validation level applied during parsing.
            software: Parameter-file software, or the parameter-free rule to select;
                limits result rules to that vendor and its declared quantification software.

        Raises:
            ValueError: The level request is empty or duplicated.
            ConversionError: The parameter file is unreadable, or the source does not
                identify the requested vendor rules.
        """
        levels = _validated_levels(LEVELS if requested_levels is None else requested_levels)
        source = _input_source(data)
        if parameters_path is None:
            if software is None:
                raise RuleUnavailableError(
                    "parameter-free packaged conversion requires --software (software= in Python)"
                )
            parameters = None
            vendors = frozenset({software_slug(software)})
        else:
            parameter_software = (
                guess_software(source) if software is None else software_slug(software)
            )
            parameters = _read_parameters(parameters_path, parameter_software)
            vendors = frozenset(
                software_slug(name)
                for name in (parameter_software, parameters.quantification_software)
                if name is not None
            )
        detection = detect_rule_documents(
            parameters, source, levels, vendors=vendors, checks=checks
        )
        self._parameters = parameters
        self._detection = detection

    @classmethod
    def from_software(
        cls,
        data: Path,
        software: str,
        requested_levels: Iterable[QuantificationLevel] | None = None,
        checks: ValidationChecks = "standard",
    ) -> Self:
        """Compile a known result producer's export without a parameter file.

        Column matching must identify one rule per requested level without search
        settings. Ambiguous versions or missing scientific evidence raise an error.
        ``software`` names the result producer, not a compound workflow's parameter
        software. The detected version is ``None`` and ``parameters`` is unavailable.
        """
        levels = _validated_levels(LEVELS if requested_levels is None else requested_levels)
        compiler = cls.__new__(cls)
        compiler._detection = detect_software_rules(
            _input_source(data), levels, software=software, checks=checks
        )
        compiler._parameters = None
        return compiler

    @classmethod
    def from_rule(
        cls,
        data: Path,
        rule: Path,
        parameters_path: Path | None = None,
        requested_levels: Iterable[QuantificationLevel] | None = None,
        software: str | None = None,
        checks: ValidationChecks = "standard",
    ) -> Self:
        """Compile a caller's schema-0.8 rule file instead of a detected packaged one.

        ``requested_levels`` defaults to every level the rule declares. A parameter file adds
        search-parameter evidence and the detected version; ``software`` selects its grammar
        when the rule's software name does not.

        Raises:
            ConversionError: The rule or parameter file is unreadable, or the source offers
                none of the requested levels.
        """
        try:
            document = load_rule_document(rule)
        except (json.JSONDecodeError, ValidationError) as error:
            raise ConversionError(f"cannot read rule file {rule}: {error}") from error
        parameters = (
            None
            if parameters_path is None
            else _read_parameters(
                parameters_path,
                software_slug(document.software_name if software is None else software),
            )
        )
        try:
            selections = ExplicitRuleCompiler(
                document,
                _input_source(data),
                document.levels if requested_levels is None else requested_levels,
                UNKNOWN_SEARCH_PARAMETERS
                if parameters is None
                else search_parameter_evidence(parameters),
                checks=checks,
            ).selections
        except RuleNotApplicable as error:
            raise ConversionError(str(error)) from error
        compiler = cls.__new__(cls)
        compiler._detection = DetectedRuleSet(
            software=software_slug(document.software_name),
            version=None if parameters is None else parameters.software_version,
            levels=selections,
        )
        compiler._parameters = parameters
        return compiler

    @property
    def parameters(self) -> Parameters:
        """Typed vendor parameters used for rule detection, when supplied."""
        if self._parameters is None:
            raise ValueError("this conversion has no vendor parameter file")
        return self._parameters

    @property
    def detection(self) -> DetectedRuleSet:
        """Resolved software metadata and selected rule levels."""
        return self._detection

    def compile(self) -> ParserCollection:
        """Compile every detected selection into one collection parser."""
        return ParserCollection(tuple(selection.parser for selection in self._detection.levels))


class ExplicitRuleCompiler:
    """Compile caller-supplied rule documents without vendor auto-detection."""

    __slots__ = ("_selections",)

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
        selections = select_document_levels(
            document, source, levels, parameter_evidence, checks=checks
        )
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

    @property
    def selections(self) -> tuple[LevelSelection, ...]:
        """Resolved rule and physical source for each selected level."""
        return self._selections

    def compile(self) -> ParserCollection:
        """Compile every explicit selection into one collection parser."""
        return ParserCollection(tuple(selection.parser for selection in self._selections))


def _validated_levels(
    requested_levels: Iterable[QuantificationLevel],
) -> tuple[QuantificationLevel, ...]:
    levels = tuple(requested_levels)
    if not levels:
        raise ValueError("at least one quantification level is required")
    duplicates = tuple(level for level in dict.fromkeys(levels) if levels.count(level) > 1)
    if duplicates:
        raise ValueError(f"duplicate quantification levels: {duplicates}")
    return levels


def _read_parameters(path: Path, software: str) -> Parameters:
    try:
        return parse_params(path, software=software)
    except (ParamsError, json.JSONDecodeError, ValidationError) as error:
        raise ConversionError(
            f"cannot parse {path} as {software}: {error}; "
            "use --software (software= in Python) to select the parameter-file grammar"
        ) from error


def _input_source(data: Path) -> InputSource:
    return Folder(path=data) if data.is_dir() else SingleFile(path=data)


__all__ = [
    "ExplicitRuleCompiler",
    "ParseRuleCompiler",
]
