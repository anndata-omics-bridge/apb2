"""Identify the packaged Parser V2 rules supported by one physical source or folder."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from apb2.parserV2.parse_quant.errors import IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFiles,
    InputSource,
    PreparedTable,
    SingleFile,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.parser_factory import compile_level
from apb2.parserV2.prepare_source import (
    InputPreparationError,
    preparation_paths,
    prepare_source,
    recognizes_preparation,
)
from apb2.parserV2.source_binding import BoundTable
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel


class RuleDetectionError(ValueError):
    """The supplied evidence does not identify one unambiguous packaged selection."""


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


@dataclass(frozen=True, slots=True)
class LevelSelection:
    """One resolved rule, level, and concrete source ready for compilation."""

    level: QuantificationLevel
    document: RuleDocument
    source_path: Path
    source: InputSource
    parser: Parser = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DetectedRuleSet:
    """The ordered compatible levels selected from one vendor source."""

    software: str
    version: str | None
    levels: tuple[LevelSelection, ...]


@dataclass(frozen=True, slots=True)
class _TableDetection:
    """Compatible levels and rejected evidence observed for one physical table."""

    matches: tuple[LevelSelection, ...]
    incompatibilities: tuple[str, ...]
    named_tables: tuple[Path, ...]


UNKNOWN_SEARCH_PARAMETERS = SearchParameterEvidence(
    acquisition_method="unknown",
    combine_charge_states=None,
)
"""Evidence supplied to an explicit rule when the caller has no parameter file."""


def search_parameter_evidence(parameters: Parameters) -> SearchParameterEvidence:
    """Project the two search-parameter fields schema 0.8 is allowed to inspect."""
    return SearchParameterEvidence(
        acquisition_method=parameters.acquisition_method,
        combine_charge_states=parameters.combine_charge_states,
    )


def software_slug(software_name: str) -> str:
    """Map a catalog name such as ``DIA-NN`` to its parser/rule folder slug."""
    return re.sub(r"[^a-z0-9]", "", software_name.lower())


def guess_software(source: InputSource) -> str:
    """Return the unique vendor slug whose declared levels accept this source."""
    slugs = {
        software_slug(document.software_name)
        for document in _packaged_documents(parameter_file="required")
        if (_document_matches(document, source))
    }
    if len(slugs) > 1:
        raise AmbiguousRuleError(
            f"source matches several packaged vendors: {sorted(slugs)}; pass --software"
        )
    if not slugs:
        raise RuleUnavailableError(
            f"could not recognize the vendor for {source.path}; pass --software or --rule-config"
        )
    return next(iter(slugs))


def detect_rule_document(
    parameters: Parameters,
    source: InputSource,
) -> DetectedRuleDocument:
    """Identify exactly one packaged document without reading the full source table."""
    detected = detect_rule_documents(parameters, source, LEVELS)
    documents = {selection.document.path: selection.document for selection in detected.levels}
    if len(documents) > 1:
        paths = sorted(str(path) for path in documents)
        raise AmbiguousRuleError(f"evidence matches several packaged documents: {paths}")
    return DetectedRuleDocument(
        document=next(iter(documents.values())),
        software=detected.software,
        version=detected.version,
    )


def detect_rule_documents(
    parameters: Parameters | None,
    source: InputSource,
    levels: Iterable[QuantificationLevel],
    *,
    vendors: frozenset[str] | None = None,
    checks: Literal["standard", "strict"] = "standard",
) -> DetectedRuleSet:
    """Identify one packaged rule per compatible requested level.

    A folder is a bundle of independent tables. Each rule still binds exactly one table, and
    each level remains independently compiled. Missing named tables are unavailable; a named
    table that exists but satisfies none of its document's requested levels is invalid input.
    When supplied, vendor slugs restrict candidates before any physical source probing.
    """
    requested = _requested_levels(levels)
    if not requested:
        raise RuleUnavailableError("no quantification levels were requested")

    matches: dict[QuantificationLevel, list[LevelSelection]] = {level: [] for level in requested}
    expected_names: dict[QuantificationLevel, set[str]] = {level: set() for level in requested}
    evidence = _detection_evidence(parameters)
    parameter_file = "none" if parameters is None else "required"
    for document in _packaged_documents(vendors, parameter_file=parameter_file):
        version = _detection_version(parameters, software_slug(document.software_name))
        if version is not None and not _pattern_admits(document.software_version_pattern, version):
            continue
        for level in requested:
            if level in document.levels:
                name = document.declared(level).input.file_name
                if name is not None:
                    expected_names[level].add(name)
        for selection in select_document_levels(
            document, source, requested, evidence, checks=checks
        ):
            matches[selection.level].append(selection)

    selected = _unique_level_matches(matches, requested)
    if not selected:
        scope = f"; candidate vendors: {sorted(vendors)}" if vendors is not None else ""
        raise RuleUnavailableError(_unavailable_message(source, requested, expected_names) + scope)

    matched_vendors = {software_slug(selection.document.software_name) for selection in selected}
    if len(matched_vendors) > 1:
        raise AmbiguousRuleError(
            f"source levels match several packaged vendors: {sorted(matched_vendors)}"
        )
    software = next(iter(matched_vendors))
    return DetectedRuleSet(
        software=software,
        version=_detection_version(parameters, software),
        levels=tuple(selected),
    )


def detect_software_rules(
    source: InputSource,
    levels: Iterable[QuantificationLevel],
    *,
    software: str,
    checks: Literal["standard", "strict"] = "standard",
) -> DetectedRuleSet:
    """Select a known producer's rules from file columns without parameter evidence."""
    vendor = software_slug(software)
    if not vendor:
        raise ValueError("software must name the result producer")
    requested = _requested_levels(levels)
    matches: dict[QuantificationLevel, list[LevelSelection]] = {level: [] for level in requested}
    for document in _packaged_documents(frozenset({vendor})):
        for selection in select_document_levels(document, source, requested, None, checks=checks):
            resolved = _software_only_selection(selection, checks)
            if resolved is not None:
                matches[selection.level].append(resolved)
    selected = _unique_level_matches(matches, requested)
    if not selected:
        raise RuleUnavailableError(
            f"no matching packaged rule for software {vendor!r}, levels {list(requested)}, "
            f"source {source.path}"
        )
    return DetectedRuleSet(software=vendor, version=None, levels=tuple(selected))


def _software_only_selection(
    selection: LevelSelection, checks: Literal["standard", "strict"]
) -> LevelSelection | None:
    """Apply authored column signatures and any inferable search-parameter override."""
    document = selection.document
    declared = document.declared(selection.level)
    columns = set(_selection_columns(selection))
    software_only = declared.input.software_only
    if software_only is not None and not set(software_only.required_columns) <= columns:
        return None
    if software_only is not None and set(software_only.forbidden_columns) & columns:
        return None
    acquisition: Literal["DDA", "DIA", "unknown"] = "unknown"
    if software_only is not None:
        acquisition = software_only.acquisition_method_otherwise
        if any(column in columns for column in software_only.acquisition_method_if_any):
            acquisition = "DDA"
    evidence = SearchParameterEvidence(acquisition_method=acquisition, combine_charge_states=None)
    rule = declared.declaration
    needed = set(rule.requires_search_parameters)
    for override in rule.search_parameter_overrides:
        needed.update(override.when_search_parameters)
    unresolved = needed - ({"acquisition_method"} if acquisition != "unknown" else set())
    if unresolved:
        raise RuleUnavailableError(
            f"{document.path}: level {selection.level!r} requires search-parameter "
            f"evidence for {sorted(unresolved)}; software and columns cannot resolve this rule"
        )
    try:
        facade = ParseRuleFacade(document, selection.level, evidence)
    except RuleNotApplicable:
        return None
    return replace(selection, parser=compile_level(facade, selection.source, checks))


def _selection_columns(selection: LevelSelection) -> tuple[str, ...]:
    """Read only the selected table's header for software-only evidence."""
    if isinstance(selection.source, PreparedTable):
        return tuple(selection.source.frame.columns)
    facade = ParseRuleFacade.from_declared_rule(selection.document, selection.level)
    bound = BoundTable(selection.source, facade.working_parameters.input)
    return bound.recognition_evidence(facade.working_parameters.accepts_header).columns


def select_document_levels(
    document: RuleDocument,
    source: InputSource,
    levels: Iterable[QuantificationLevel],
    evidence: SearchParameterEvidence | None,
    *,
    checks: Literal["standard", "strict"] = "standard",
) -> tuple[LevelSelection, ...]:
    """Select table groups before compiling, for packaged and explicit documents alike.

    A recognized direct filename selects its table even when the requested level belongs
    elsewhere. Renamed files must identify one table through header evidence. Within that
    table, unavailable optional levels remain skippable. Folder validity is checked per
    present table, so a successful sibling cannot conceal a malformed table.
    ``None`` probes declared rules; the software-only caller must reject any matched
    search-parameter dependency before returning those selections.
    """
    requested = _requested_levels(levels)
    tables = document.table_levels
    if not isinstance(source, Folder | InputFiles):
        named = tuple(
            table
            for table in tables
            if document.declared(table[0]).input.file_name == source.path.name
        )
        tables = named or tables
    accepted: list[_TableDetection] = []
    consumed: set[Path] = set()
    for table in tables:
        selected: tuple[QuantificationLevel, ...] = tuple(
            level for level in requested if level in table
        )
        candidates, paths = _table_sources(document, table, source, selected)
        consumed.update(paths)
        table_matches: list[LevelSelection] = []
        for candidate in candidates:
            observed = _detect_table_levels(document, table, evidence, candidate, checks)
            consumed.update(match.source_path for match in observed.matches)
            if selected and _present_table_is_incompatible(observed):
                raise RuleUnavailableError(
                    f"declared table(s) {sorted(str(path) for path in observed.named_tables)} "
                    "are present but incompatible with "
                    f"{document.path}: {list(observed.incompatibilities)}"
                )
            table_matches.extend(match for match in observed.matches if match.level in selected)
        if len({match.source_path for match in table_matches}) > 1:
            raise AmbiguousRuleError(
                f"multiple inputs match table levels {table}: "
                f"{sorted(str(match.source_path) for match in table_matches)}"
            )
        if table_matches:
            accepted.append(_TableDetection(tuple(table_matches), (), ()))
    if isinstance(source, InputFiles) and consumed:
        unused = set(source.files.values()) - consumed
        if unused:
            raise InputPreparationError(
                f"unrecognized explicit companion inputs: {sorted(str(path) for path in unused)}"
            )
    if not isinstance(source, Folder | InputFiles) and len(accepted) > 1:
        candidates = [
            [
                (match.level, document.declared(match.level).input.file_name)
                for match in item.matches
            ]
            for item in accepted
        ]
        raise AmbiguousRuleError(
            f"{source.path} matches several table groups in {document.path}: {candidates}; "
            "request a level or use a declared filename"
        )
    by_level = {match.level: match for item in accepted for match in item.matches}
    return tuple(by_level[level] for level in requested if level in by_level)


def _table_sources(
    document: RuleDocument,
    table: tuple[QuantificationLevel, ...],
    source: InputSource,
    selected: tuple[QuantificationLevel, ...],
) -> tuple[tuple[InputSource, ...], tuple[Path, ...]]:
    """Bind one group, reading quantitative rows only for requested prepared groups."""
    declaration = document.declared(table[0])
    preparation = declaration.preparation
    if preparation is not None:
        if isinstance(source, PreparedTable):
            return ((source,), source.source_paths) if source.how == preparation else ((), ())
        paths = preparation_paths(source, preparation)
        if not paths or not selected:
            return (), paths
        prepared = prepare_source(
            InputFiles(source.path, {path.name: path for path in paths}), preparation
        )
        return (prepared,), paths
    filename = declaration.input.file_name
    declared_names = {document.declared(level).input.file_name for level in document.levels}
    candidates = tuple(
        candidate
        for candidate in _direct_table_sources(source, filename)
        if candidate.path.name == filename or candidate.path.name not in declared_names
    )
    return candidates, ()


def _direct_table_sources(source: InputSource, filename: str | None) -> tuple[InputSource, ...]:
    """Bind direct groups independently, including renamed files in explicit bundles."""
    if isinstance(source, InputFiles):
        return tuple(SingleFile(path) for path in source.files.values())
    if isinstance(source, Folder):
        if filename is not None and (source.path / filename).is_file():
            return (source,)
        return tuple(
            SingleFile(path)
            for path in sorted(source.path.iterdir())
            if path.is_file()
            and path.suffix.lower() in {".txt", ".tsv", ".csv", ".parquet", ".xlsx"}
        )
    return (source,)


def _detect_table_levels(
    document: RuleDocument,
    requested: tuple[QuantificationLevel, ...],
    evidence: SearchParameterEvidence | None,
    source: InputSource,
    checks: Literal["standard", "strict"],
) -> _TableDetection:
    """Inspect requested levels that share one physical input."""
    matches: list[LevelSelection] = []
    incompatibilities: list[str] = []
    named_tables: set[Path] = set()
    for level in requested:
        if level not in document.levels:
            continue
        try:
            facade = (
                ParseRuleFacade.from_declared_rule(document, level)
                if evidence is None
                else ParseRuleFacade(document, level, evidence)
            )
        except RuleNotApplicable:
            continue
        declared_name = facade.working_parameters.input.file_name
        if declared_name is not None:
            if isinstance(source, Folder):
                named_tables.add(source.path / declared_name)
            elif source.path.name == declared_name:
                named_tables.add(source.path)
        try:
            parser = compile_level(facade, source, checks)
            source_path = (
                source.path / declared_name
                if isinstance(source, Folder) and declared_name is not None
                else source.path
            )
        except IncompatibleSourceError as error:
            incompatibilities.append(f"{level}: {error}")
            continue
        matches.append(
            LevelSelection(
                level=level,
                document=document,
                source_path=source_path,
                source=SingleFile(source_path) if isinstance(source, Folder) else source,
                parser=parser,
            )
        )
    return _TableDetection(
        matches=tuple(matches),
        incompatibilities=tuple(incompatibilities),
        named_tables=tuple(sorted(named_tables)),
    )


def _present_table_is_incompatible(observed: _TableDetection) -> bool:
    """Whether a named table exists but no requested level accepts its evidence."""
    return (
        not observed.matches
        and bool(observed.incompatibilities)
        and any(path.is_file() for path in observed.named_tables)
    )


def _unique_level_matches(
    matches: dict[QuantificationLevel, list[LevelSelection]],
    requested: tuple[QuantificationLevel, ...],
) -> list[LevelSelection]:
    """Select at most one document per level and retain canonical requested order."""
    selected: list[LevelSelection] = []
    for level in requested:
        candidates = matches[level]
        if len(candidates) > 1:
            paths = sorted(str(candidate.document.path) for candidate in candidates)
            raise AmbiguousRuleError(f"level {level!r} matches several packaged documents: {paths}")
        selected.extend(candidates)
    return selected


def _packaged_documents(
    vendors: frozenset[str] | None = None,
    *,
    parameter_file: Literal["required", "none"] | None = None,
) -> tuple[RuleDocument, ...]:
    documents = (load_rule_document(rule_path) for rule_path in PACKAGED)
    return tuple(
        document
        for document in documents
        if (vendors is None or software_slug(document.software_name) in vendors)
        and (parameter_file is None or document.parameter_file == parameter_file)
    )


def _document_matches(document: RuleDocument, source: InputSource) -> bool:
    for table in document.table_levels:
        preparation = document.declared(table[0]).preparation
        if preparation is not None and recognizes_preparation(source, preparation):
            return True
    return any(
        _source_matches(ParseRuleFacade.from_declared_rule(document, level), candidate)
        for level in document.levels
        if document.declared(level).preparation is None
        for candidate in _direct_table_sources(source, document.declared(level).input.file_name)
    )


def _source_matches(facade: ParseRuleFacade, source: InputSource) -> bool:
    try:
        working = facade.working_parameters
        if isinstance(source, PreparedTable):
            return working.accepts_header(tuple(source.frame.columns))
        bound = BoundTable(source, working.input)
        observed = bound.recognition_evidence(working.accepts_header)
        return working.accepts_header(observed.columns)
    except IncompatibleSourceError:
        return False


def _requested_levels(
    levels: Iterable[QuantificationLevel],
) -> tuple[QuantificationLevel, ...]:
    """Return unique requested levels in canonical parsing order."""
    requested = set(levels)
    return tuple(level for level in LEVELS if level in requested)


def _unavailable_message(
    source: InputSource,
    requested: tuple[QuantificationLevel, ...],
    expected_names: dict[QuantificationLevel, set[str]],
) -> str:
    """Describe an unavailable requested level, including exact folder names when known."""
    if len(requested) == 1:
        level = requested[0]
        names = sorted(expected_names[level])
        expected = f"; expected named table(s): {names}" if names else ""
        return f"requested level {level!r} is unavailable from {source.path}{expected}"
    names = sorted({name for values in expected_names.values() for name in values})
    expected = f"; expected named table(s): {names}" if names else ""
    return (
        f"no requested level is available from {source.path}{expected}; "
        "pass --rule-config PATH for an unpackaged format"
    )


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


def _detection_evidence(parameters: Parameters | None) -> SearchParameterEvidence:
    """Use unknown search settings only for a declared parameter-free rule."""
    return (
        UNKNOWN_SEARCH_PARAMETERS if parameters is None else search_parameter_evidence(parameters)
    )


def _detection_version(parameters: Parameters | None, rule_slug: str) -> str | None:
    """Parameter-free documents have no software-version evidence to check or report."""
    return None if parameters is None else _version_for(parameters, rule_slug)


def _pattern_admits(pattern: str, version: str) -> bool:
    try:
        return re.search(pattern, version) is not None
    except re.error as error:
        raise ValueError(f"invalid software_version_pattern regex {pattern!r}") from error
