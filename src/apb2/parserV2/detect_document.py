"""Identify the packaged Parser V2 rule document supported by one physical source."""

from __future__ import annotations

import re
from dataclasses import dataclass

from apb2.parserV2.compile import bind_source, header_predicate, source_recognition_evidence
from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import InputSource
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.search_parameters.model import Parameters
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document


class RuleDetectionError(ValueError):
    """The supplied evidence does not identify exactly one packaged rule document."""


class RuleUnavailableError(RuleDetectionError):
    """No packaged rule document accepts the source and parameter evidence."""


class AmbiguousRuleError(RuleDetectionError):
    """Several packaged rule documents accept the same evidence."""


@dataclass(frozen=True, slots=True)
class DetectedRuleDocument:
    """The one packaged document identified by source and parameter evidence."""

    document: RuleDocument
    software: str
    version: str | None


UNKNOWN_SEARCH_PARAMETERS = SearchParameterEvidence(
    acquisition_method="unknown",
    combine_charge_states=None,
)
"""Evidence supplied to an explicit rule when the caller has no parameter file."""


def search_parameter_evidence(parameters: Parameters) -> SearchParameterEvidence:
    """Project the two search-parameter fields schema 0.3 is allowed to inspect."""
    return SearchParameterEvidence(
        acquisition_method=parameters.acquisition_method,
        combine_charge_states=parameters.combine_charge_states,
    )


def software_slug(software_name: str) -> str:
    """Map a catalog name such as ``DIA-NN`` to its parser/rule folder slug."""
    return re.sub(r"[^a-z0-9]", "", software_name.lower())


def guess_software(source: InputSource) -> str | None:
    """Return the unique vendor slug whose declared levels accept this source."""
    slugs = {
        software_slug(document.software_name)
        for document in _packaged_documents()
        if _declared_source_matches(document, source)
    }
    return next(iter(slugs)) if len(slugs) == 1 else None


def detect_rule_document(
    parameters: Parameters,
    source: InputSource,
) -> DetectedRuleDocument:
    """Identify exactly one packaged document without reading the full source table."""
    matches: list[DetectedRuleDocument] = []
    evidence = search_parameter_evidence(parameters)
    for document in _packaged_documents():
        version = _version_for(parameters, software_slug(document.software_name))
        if version is not None and not _pattern_admits(document.software_version_pattern, version):
            continue
        if not _parameterized_source_matches(document, evidence, source):
            continue
        matches.append(
            DetectedRuleDocument(
                document=document,
                software=software_slug(document.software_name),
                version=version,
            )
        )
    if not matches:
        raise RuleUnavailableError(
            "no packaged rules.json matches the source and parameter evidence; "
            "pass --rule-config PATH for an unpackaged format"
        )
    if len(matches) > 1:
        paths = sorted(str(match.document.path) for match in matches)
        raise AmbiguousRuleError(f"evidence matches several packaged documents: {paths}")
    return matches[0]


def _packaged_documents() -> tuple[RuleDocument, ...]:
    return tuple(load_rule_document(rule_path) for rule_path in PACKAGED)


def _declared_source_matches(document: RuleDocument, source: InputSource) -> bool:
    for level in document.levels:
        facade = ParseRuleFacade.from_declared_rule(document, level)
        if _source_matches(facade, source):
            return True
    return False


def _parameterized_source_matches(
    document: RuleDocument,
    parameters: SearchParameterEvidence,
    source: InputSource,
) -> bool:
    for level in document.levels:
        try:
            facade = ParseRuleFacade(document, level, parameters)
        except RuleNotApplicable:
            continue
        if _source_matches(facade, source):
            return True
    return False


def _source_matches(facade: ParseRuleFacade, source: InputSource) -> bool:
    working = facade.working_parameters
    extensions = {
        extension
        for physical_format in working.input.formats
        for extension in physical_format.extensions
    }
    suffix = source.path.suffix.lower()
    if suffix == ".parquet" and ".parquet" not in extensions:
        return False
    if suffix != ".parquet" and extensions == {".parquet"}:
        return False
    try:
        bound = bind_source(source, working.input)
        evidence = source_recognition_evidence(source, bound, header_predicate(working))
        facade.resolve_source(evidence)
    except IncompatibleSourceError:
        return False
    return True


def _version_for(parameters: Parameters, rule_slug: str) -> str | None:
    candidates = (
        (parameters.software_name, parameters.software_version),
        (parameters.quantification_software, parameters.quantification_software_version),
    )
    for software_name, version in candidates:
        if software_name is not None and software_slug(software_name) == rule_slug:
            return version
    if not any(software_name for software_name, _version in candidates):
        return parameters.software_version
    return None


def _pattern_admits(pattern: str, version: str) -> bool:
    try:
        return re.search(pattern, version) is not None
    except re.error as error:
        raise ValueError(f"invalid software_version_pattern regex {pattern!r}") from error
