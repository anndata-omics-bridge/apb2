"""Public compiler objects for packaged and explicit parse rules."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from loguru import logger
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
from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.data.step_log import logged_step
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFile,
    InputFileRecord,
    InputSource,
    SingleFile,
)
from apb2.parserV2.parse_quant.parser import ParserCollection
from apb2.parserV2.search_parameters.model import Parameters
from apb2.parserV2.vendor_params.parsers.shared.errors import ParamsError
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel

type ValidationChecks = Literal["standard", "strict"]


@dataclass(frozen=True, slots=True)
class ParameterFile:
    """A vendor parameter file: its ``input_files`` record and its typed parameters."""

    record: InputFileRecord
    parameters: Parameters

    @classmethod
    def read(cls, path: Path, software: str) -> Self:
        """Parse the file in ``software``'s parameter grammar and record the file.

        Raises:
            ConversionError: The bytes do not parse in ``software``'s parameter grammar.
        """
        try:
            parameters = parse_params(io.BytesIO(path.read_bytes()), software=software)
        except (ParamsError, json.JSONDecodeError, ValidationError) as error:
            raise ConversionError(
                f"cannot parse {path} as {software}: {error}; "
                "use --software (software= in Python) to select the parameter-file grammar"
            ) from error
        file_format = path.suffix.removeprefix(".").lower()
        return cls(InputFile(path, "parameter_file", file_format).record(), parameters)

    def provenance(self) -> dict[str, JsonValue]:
        """The root parse provenance that records the typed parameters."""
        return {"search_parameters": {"parameters": self.parameters.model_dump(mode="json")}}


@dataclass(frozen=True, slots=True)
class ParseRuleCompiler:
    """Resolved vendor inputs; build one with ``detect``, ``from_software`` or ``from_rule``."""

    detection: DetectedRuleSet
    parameter_file: ParameterFile | None

    @classmethod
    def detect(
        cls,
        data: Path,
        parameters_path: Path | None,
        requested_levels: Iterable[QuantificationLevel] | None = None,
        checks: ValidationChecks = "standard",
        software: str | None = None,
    ) -> Self:
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
            parameter_file = None
            parameters = None
            vendors = frozenset({software_slug(software)})
        else:
            parameter_software = (
                guess_software(source) if software is None else software_slug(software)
            )
            parameter_file = ParameterFile.read(parameters_path, parameter_software)
            parameters = parameter_file.parameters
            vendors = frozenset(
                software_slug(name)
                for name in (parameter_software, parameters.quantification_software)
                if name is not None
            )
        with logged_step("detect", source=data, levels=",".join(levels)):
            detection = detect_rule_documents(
                parameters, source, levels, vendors=vendors, checks=checks
            )
        _log_detection(detection)
        return cls(detection, parameter_file)

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
        with logged_step("detect", source=data, software=software, levels=",".join(levels)):
            detection = detect_software_rules(
                _input_source(data), levels, software=software, checks=checks
            )
        _log_detection(detection)
        return cls(detection, None)

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
        """Compile a caller's schema-0.9 rule file instead of a detected packaged one.

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
        parameter_file = (
            None
            if parameters_path is None
            else ParameterFile.read(
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
                if parameter_file is None
                else search_parameter_evidence(parameter_file.parameters),
                checks=checks,
            ).selections
        except RuleNotApplicable as error:
            raise ConversionError(str(error)) from error
        detection = DetectedRuleSet(
            software=software_slug(document.software_name),
            version=None if parameter_file is None else parameter_file.parameters.software_version,
            levels=selections,
        )
        return cls(detection, parameter_file)

    @property
    def parameters(self) -> Parameters:
        """Typed vendor parameters used for rule detection, when supplied."""
        if self.parameter_file is None:
            raise ValueError("this conversion has no vendor parameter file")
        return self.parameter_file.parameters

    def compile(self) -> ParserCollection:
        """Compile every detected selection into one collection parser."""
        parsers = tuple(selection.parser for selection in self.detection.levels)
        if self.parameter_file is None:
            return ParserCollection(parsers, {})
        return ParserCollection(
            parsers, self.parameter_file.provenance(), (self.parameter_file.record,)
        )


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
                level: document.table_input(level).file_name
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
        return ParserCollection(tuple(selection.parser for selection in self._selections), {})


def _log_detection(detection: DetectedRuleSet) -> None:
    """Name the packaged rule document each detected level will be parsed with."""
    for selection in detection.levels:
        rule = "/".join(selection.document.path.parts[-3:])
        logger.info("detected level={} rule={}", selection.level, rule)


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


def _input_source(data: Path) -> SingleFile | Folder:
    return Folder(path=data) if data.is_dir() else SingleFile(path=data)


__all__ = [
    "ExplicitRuleCompiler",
    "ParseRuleCompiler",
]
