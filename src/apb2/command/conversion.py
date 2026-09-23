"""File-to-file conversion workflow owned by the ``apb2`` command."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
    CanonicalKeyCollisionError,
    LevelParseTimings,
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
class PhaseTiming:
    """One internal conversion phase, independent of subprocess runtime."""

    name: str
    seconds: float


@dataclass(frozen=True, slots=True)
class ConversionTimings:
    """Tool-owned phase and level timings for an optional JSON artifact."""

    phases: tuple[PhaseTiming, ...]
    levels: tuple[LevelParseTimings, ...]


class _TimingRecorder:
    """Accumulate measured phases without changing the conversion result format."""

    def __init__(self) -> None:
        self.phases: list[PhaseTiming] = []
        self.levels: tuple[LevelParseTimings, ...] = ()

    @contextmanager
    def phase(self, name: str) -> Generator[None]:
        """Record and log a phase even when its operation fails."""
        started = perf_counter()
        try:
            yield
        finally:
            self.record(name, perf_counter() - started)

    def record(self, name: str, seconds: float) -> None:
        """Record an independently measured phase and keep the human log."""
        self.phases.append(PhaseTiming(name=name, seconds=seconds))
        logger.info("conversion phase={} seconds={:.3f}", name, seconds)

    def snapshot(self) -> ConversionTimings:
        """Freeze the measurements for a successful conversion."""
        return ConversionTimings(phases=tuple(self.phases), levels=self.levels)


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
    timings: ConversionTimings


_EXPECTED_CONVERSION_FAILURES = (
    InputPreparationError,
    AggregateTypeError,
    AmbiguousDialectError,
    AxisCoercionError,
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
    software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Convert one level using an explicitly supplied schema-0.8 rule document."""
    timings = _TimingRecorder()
    try:
        with timings.phase("compile"):
            document, parameters, evidence = _explicit_conversion_inputs(
                rule_config=rule_config,
                parameters_path=parameters_path,
                software=software,
            )
            compiler = ExplicitRuleCompiler(
                document=document,
                source=_input_source(data),
                requested_levels=(level,),
                parameter_evidence=evidence,
                checks=checks,
            )
            parser = compiler.compile()
        parsed, outputs = _parse_and_write(
            output,
            parser,
            compiler.selections,
            timings,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=software_slug(document.software_name),
            version=parameters.software_version if parameters is not None else None,
            timings=timings.snapshot(),
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_all_from_rule_config(
    *,
    data: Path,
    output: Path,
    rule_config: Path,
    parameters_path: Path | None,
    software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Convert every compatible level of an explicit schema-0.8 document."""
    timings = _TimingRecorder()
    try:
        with timings.phase("compile"):
            document, parameters, evidence = _explicit_conversion_inputs(
                rule_config=rule_config,
                parameters_path=parameters_path,
                software=software,
            )
            compiler = ExplicitRuleCompiler(
                document=document,
                source=_input_source(data),
                requested_levels=document.levels,
                parameter_evidence=evidence,
                checks=checks,
            )
            parser = compiler.compile()
        parsed, outputs = _parse_and_write(
            output,
            parser,
            compiler.selections,
            timings,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=software_slug(document.software_name),
            version=parameters.software_version if parameters is not None else None,
            timings=timings.snapshot(),
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_from_packaged_rules(
    *,
    data: Path,
    level: QuantificationLevel,
    output: Path,
    parameters_path: Path | None,
    software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Detect a packaged document from source and optional parameters, then convert it."""
    timings = _TimingRecorder()
    try:
        with timings.phase("compile"):
            compiler = _packaged_compiler(
                data, parameters_path, software, requested_levels=(level,), checks=checks
            )
            parser = compiler.compile()
        parsed, outputs = _parse_and_write(
            output,
            parser,
            compiler.detection.levels,
            timings,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=compiler.detection.software,
            version=compiler.detection.version,
            timings=timings.snapshot(),
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def convert_all_from_packaged_rules(
    *,
    data: Path,
    output: Path,
    parameters_path: Path | None,
    software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Detect and convert every compatible packaged level from one file or folder."""
    timings = _TimingRecorder()
    try:
        with timings.phase("compile"):
            compiler = _packaged_compiler(
                data, parameters_path, software, requested_levels=LEVELS, checks=checks
            )
            parser = compiler.compile()
        parsed, outputs = _parse_and_write(
            output,
            parser,
            compiler.detection.levels,
            timings,
        )
        return _conversion_summary(
            parsed.levels,
            outputs=outputs,
            software=compiler.detection.software,
            version=compiler.detection.version,
            timings=timings.snapshot(),
        )
    except _EXPECTED_CONVERSION_FAILURES as error:
        raise ConversionError(str(error)) from error


def _packaged_compiler(
    data: Path,
    parameters_path: Path | None,
    software: str | None,
    *,
    requested_levels: tuple[QuantificationLevel, ...],
    checks: AnnDataChecks,
) -> ParseRuleCompiler:
    """Choose the parameter-backed or column-backed compiler at the command boundary."""
    if parameters_path is None and software is not None:
        return ParseRuleCompiler.from_software(
            data, software=software, requested_levels=requested_levels, checks=checks
        )
    return ParseRuleCompiler(
        data,
        parameters_path,
        requested_levels=requested_levels,
        checks=checks,
        software=software,
    )


def _explicit_conversion_inputs(
    *,
    rule_config: Path,
    parameters_path: Path | None,
    software: str | None,
) -> tuple[RuleDocument, Parameters | None, SearchParameterEvidence]:
    """Load one explicit document and its optional parameter evidence once."""
    document = load_rule_document(rule_config)
    if parameters_path is None:
        return document, None, UNKNOWN_SEARCH_PARAMETERS
    parameters = parse_params(
        parameters_path,
        software=software_slug(document.software_name if software is None else software),
    )
    return document, parameters, search_parameter_evidence(parameters)


def _input_source(data: Path) -> InputSource:
    """Bind a canonical vendor-result folder or one direct input file."""
    return Folder(path=data) if data.is_dir() else SingleFile(path=data)


def _parse_and_write(
    output: Path,
    parser: ParserCollection,
    selections: tuple[LevelSelection, ...],
    timings: _TimingRecorder,
) -> tuple[ParsedLevels, tuple[Path, ...]]:
    """Time bound reads, parsing/alignment, and writing without rereading any source."""
    for selection in selections:
        logger.info("level={} source={}", selection.level, selection.source_path)
    started = perf_counter()
    combined, level_timings = parser.parse_with_timings()
    groups = group_observations(combined)
    outputs = _group_output_paths(groups, output)
    read_seconds = sum(timing.read_seconds for timing in level_timings)
    parse_seconds = perf_counter() - started - read_seconds
    for timing in level_timings:
        logger.info(
            "conversion level={} read_seconds={:.3f} parse_seconds={:.3f}",
            timing.level,
            timing.read_seconds,
            timing.parse_seconds,
        )
    timings.levels = level_timings
    timings.record("read", read_seconds)
    timings.record("parse", parse_seconds)
    with timings.phase("write"):
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
    timings: ConversionTimings,
) -> ConversionSummary:
    return ConversionSummary(
        software=software,
        version=version,
        outputs=outputs,
        timings=timings,
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


def write_conversion_timings(timings: ConversionTimings, target: Path) -> Path:
    """Atomically publish optional tool timings without changing APB result metadata."""
    if target.exists():
        raise ConversionError(f"timing output already exists: {target}")
    document = {
        "format": "apb-tool-timings",
        "format_version": 1,
        "tool": "apb2",
        "operation": "convert",
        "phases": [{"name": phase.name, "seconds": phase.seconds} for phase in timings.phases],
        "levels": [
            {
                "level": level.level,
                "read_seconds": level.read_seconds,
                "parse_seconds": level.parse_seconds,
            }
            for level in timings.levels
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise ConversionError(f"timing output already exists: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
