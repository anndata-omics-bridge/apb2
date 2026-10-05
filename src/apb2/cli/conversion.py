"""File-to-file conversion workflow owned by the ``apb2`` command."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from loguru import logger

from apb2.api import (
    ConversionError,
    LevelParseTimings,
    ParsedLevels,
    ParseRuleCompiler,
    QuantificationLevel,
    read_parsed_levels,
    write_parsed_levels,
)

type AnnDataChecks = Literal["standard", "strict"]


def reformat_result(source: Path, target: Path) -> None:
    """Read one APB2 result and write the same value through the target's format."""
    parsed = read_parsed_levels(source)
    for level, value in parsed.levels.items():
        logger.info(
            "level={} shape=({}, {}) layers={}",
            level,
            value.obs.frame.height,
            value.var.frame.height,
            list(value.layers),
        )
    write_parsed_levels(parsed, target)
    logger.info("reformatted {} -> {}", source, target)


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
    return _convert(
        lambda: ParseRuleCompiler.from_rule(
            data, rule_config, parameters_path, (level,), software, checks
        ),
        output,
    )


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
    return _convert(
        lambda: ParseRuleCompiler.from_rule(
            data, rule_config, parameters_path, None, software, checks
        ),
        output,
    )


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
    return _convert(
        lambda: _packaged_compiler(data, parameters_path, software, (level,), checks), output
    )


def convert_all_from_packaged_rules(
    *,
    data: Path,
    output: Path,
    parameters_path: Path | None,
    software: str | None,
    checks: AnnDataChecks,
) -> ConversionSummary:
    """Detect and convert every compatible packaged level from one file or folder."""
    return _convert(
        lambda: _packaged_compiler(data, parameters_path, software, None, checks), output
    )


def _packaged_compiler(
    data: Path,
    parameters_path: Path | None,
    software: str | None,
    requested_levels: tuple[QuantificationLevel, ...] | None,
    checks: AnnDataChecks,
) -> ParseRuleCompiler:
    """Choose the parameter-backed or column-backed compiler at the command boundary."""
    if parameters_path is None and software is not None:
        return ParseRuleCompiler.from_software(data, software, requested_levels, checks)
    return ParseRuleCompiler(data, parameters_path, requested_levels, checks, software)


def _convert(build: Callable[[], ParseRuleCompiler], output: Path) -> ConversionSummary:
    """Time compilation, bound reads, parsing/alignment and writing without rereading."""
    timings = _TimingRecorder()
    with timings.phase("compile"):
        compiler = build()
        parser = compiler.compile()
    for selection in compiler.detection.levels:
        logger.info("level={} source={}", selection.level, selection.source_path)
    started = perf_counter()
    combined, level_timings = parser.parse_with_timings()
    groups = combined.observation_groups()
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
            write_parsed_levels(group, target)
    written = {
        name: next(group.levels[name] for group in groups if name in group.levels)
        for name in combined.levels
    }
    return ConversionSummary(
        software=compiler.detection.software,
        version=compiler.detection.version,
        outputs=outputs,
        timings=timings.snapshot(),
        levels=tuple(
            LevelConversionSummary(
                level=name,
                observation_count=level.obs.frame.height,
                variable_count=level.var.frame.height,
                layer_names=tuple(level.layers),
            )
            for name, level in written.items()
        ),
    )


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
