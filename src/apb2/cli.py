"""The ``apb2 convert`` command: a thin adapter over Parser V2 workflows."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger

from apb2 import annotation_facade
from apb2.command import conversion

app = App(name="apb2", help="Rules-driven vendor-result conversion", help_on_error=True)


@dataclass(frozen=True, slots=True)
class ConvertCliOptions:
    """Flat Cyclopts option group for ``apb2 convert``."""

    params: Path | None = None
    rule_config: Path | None = None
    software: str | None = None
    params_software: str | None = None
    output: Path | None = None
    storage_format: Annotated[
        Literal["hdf5", "parquet", "duckdb"],
        Parameter(name="--format"),
    ] = "hdf5"
    strict: bool = False


DEFAULT_CONVERT_CLI_OPTIONS = ConvertCliOptions()

_SINGLE_LEVEL_SUFFIX = {"hdf5": ".h5ad", "parquet": ".parquet", "duckdb": ".duckdb"}
_MULTI_LEVEL_SUFFIX = {"hdf5": ".h5mu", "parquet": ".parquet", "duckdb": ".duckdb"}


@app.command
def convert(
    data: Path,
    level: conversion.QuantificationLevel | None = None,
    options: Annotated[ConvertCliOptions, Parameter(name="*")] = DEFAULT_CONVERT_CLI_OPTIONS,
) -> int:
    """Convert one vendor table or result directory.

    An explicit LEVEL selects one level; omitting it converts every compatible level.
    A directory supplies a multi-file vendor result together. Levels with incompatible
    observation identities are written separately with observation-key suffixes; one-to-one
    aliases are aligned.
    --params is the vendor parameter file and is required unless --rule-config is given.
    --software disambiguates packaged rule detection. --params-software selects the
    parameter parser independently for compound workflows. --rule-config selects an
    explicit schema-0.8 document. --format selects hdf5, parquet, or duckdb. --output is a
    basename to which apb2 appends the selected suffix; the name may contain dots, it simply
    must not already carry that suffix. --strict promotes layer-contract warnings to errors.
    """
    suffixes = _MULTI_LEVEL_SUFFIX if level is None else _SINGLE_LEVEL_SUFFIX
    output_suffix = suffixes[options.storage_format]
    # Only the extension this command appends is refused, and only to stop a doubled
    # suffix. A dotted basename is a legal name — ``ion.apb2`` beside ``ion`` is how a caller
    # comparing two converters spells the pair — and rejecting it is not this check's job.
    if options.output is not None and options.output.suffix == output_suffix:
        logger.error(
            "--output must not already end in {}; apb2 appends it, got {}",
            output_suffix,
            options.output,
        )
        return 2
    output = (
        data.with_suffix(output_suffix)
        if options.output is None
        else Path(f"{options.output}{output_suffix}")
    )
    checks = "strict" if options.strict else "standard"
    try:
        if options.rule_config is not None:
            if level is None:
                result = conversion.convert_all_from_rule_config(
                    data=data,
                    output=output,
                    rule_config=options.rule_config,
                    parameters_path=options.params,
                    parameters_software=options.params_software,
                    checks=checks,
                )
            else:
                result = conversion.convert_from_rule_config(
                    data=data,
                    level=level,
                    output=output,
                    rule_config=options.rule_config,
                    parameters_path=options.params,
                    parameters_software=options.params_software,
                    checks=checks,
                )
        else:
            if options.params is None:
                logger.error("pass --params (it gives the software version) or --rule-config PATH")
                return 1
            if level is None:
                result = conversion.convert_all_from_packaged_rules(
                    data=data,
                    output=output,
                    parameters_path=options.params,
                    software=options.software,
                    parameters_software=options.params_software,
                    checks=checks,
                )
            else:
                result = conversion.convert_from_packaged_rules(
                    data=data,
                    level=level,
                    output=output,
                    parameters_path=options.params,
                    software=options.software,
                    parameters_software=options.params_software,
                    checks=checks,
                )
            logger.info(
                "vendor={} software_version={}",
                result.software,
                result.version or "missing",
            )
    except conversion.ConversionError as error:
        logger.error(str(error))
        return 1
    _log_result(result)
    return 0


@app.command
def reformat(source: Path, target: Path) -> int:
    """Convert one APB2-authored result between h5ad, h5mu, Parquet, and DuckDB."""
    try:
        conversion.reformat_result(source, target)
    except (conversion.ReformatError, OSError) as error:
        logger.error(str(error))
        return 1
    return 0


@app.command
def annotate(
    source: Path,
    annotation: Path,
    target: Path,
    unmatched: annotation_facade.UnmatchedObservations | None = None,
    include: str | None = None,
) -> int:
    """Attach a delimited sample table to an APB2 result.

    --unmatched selects keep, error, or drop behavior; --include COLUMN further filters
    drop mode by one Boolean annotation field.
    """
    try:
        result = annotation_facade.annotate_result(
            source,
            annotation,
            target,
            unmatched=unmatched,
            include=include,
        )
    except (
        annotation_facade.AnnotationWorkflowError,
        OSError,
    ) as error:
        logger.error(str(error))
        return 1
    for level, report in result.reports.items():
        logger.info(
            "level={} matched={}/{} annotation_only={} columns_added={}",
            level,
            report.coverage.matched_observation_count,
            report.coverage.observation_count,
            report.coverage.annotation_only_count,
            list(report.columns_added),
        )
    logger.info("wrote {}", target)
    return 0


def _log_result(result: conversion.ConversionSummary) -> None:
    for level in result.levels:
        logger.info(
            "level={} shape=({}, {}) layers={}",
            level.level,
            level.observation_count,
            level.variable_count,
            list(level.layer_names),
        )
    for output in result.outputs:
        logger.info("wrote {}", output)


def main() -> int:
    """Console-script entry point."""
    result = app()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
