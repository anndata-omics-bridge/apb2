"""The ``apb2 convert`` command: a thin adapter over Parser V2."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger
from pydantic import ValidationError

from apb2.parserV2.conversion import (
    ConversionResult,
    SoftwareGuessError,
    SoftwareMismatchError,
    convert_from_packaged_rules,
    convert_from_rule_config,
)
from apb2.parserV2.detect_document import RuleDetectionError
from apb2.parserV2.parse_quant.anndata_writer import AnnDataLayerContractError
from apb2.parserV2.parse_quant.axis_columns import AxisCoercionError, ColumnComputationError
from apb2.parserV2.parse_quant.data.layer_columns import StorageLabelError
from apb2.parserV2.parse_quant.duplicates import AggregateTypeError, DuplicateCellError
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.fragments import PackedLengthError
from apb2.parserV2.parse_quant.modifications import (
    PackedSiteMismatchError,
    UnknownModificationError,
)
from apb2.parserV2.parse_quant.parser import AxisShapeError, CanonicalKeyCollisionError
from apb2.parserV2.search_parameters.model import ParamsError
from apb2.parserV2.vendor_parse_rules.document import RuleNotApplicable
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel

app = App(name="apb2", help="Rules-driven vendor-table conversion", help_on_error=True)

_EXPECTED_FAILURES = (
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
    OSError,
    PackedLengthError,
    PackedSiteMismatchError,
    ParamsError,
    RuleDetectionError,
    RuleNotApplicable,
    SoftwareGuessError,
    SoftwareMismatchError,
    StorageLabelError,
    UnknownModificationError,
    ValidationError,
)


@dataclass(frozen=True, slots=True)
class ConvertCliOptions:
    """Flat Cyclopts option group for ``apb2 convert``."""

    params: Path | None = None
    rule_config: Path | None = None
    software: str | None = None
    params_software: str | None = None
    output: Path | None = None
    strict: bool = False


DEFAULT_CONVERT_CLI_OPTIONS = ConvertCliOptions()


@app.command
def convert(
    data: Path,
    level: QuantificationLevel,
    options: Annotated[ConvertCliOptions, Parameter(name="*")] = DEFAULT_CONVERT_CLI_OPTIONS,
) -> int:
    """Convert one quantification level of a vendor file to AnnData.

    --params is the vendor parameter file and is required unless --rule-config is given.
    --software disambiguates packaged rule detection. --params-software selects the
    parameter parser independently for compound workflows. --rule-config selects an
    explicit schema-0.3 document. --output is an extensionless basename; apb2 appends
    .h5ad. --strict promotes layer-contract warnings to errors.
    """
    if options.output is not None and options.output.suffix:
        logger.error(
            "--output must be an extensionless basename, got {}; apb2 appends .h5ad",
            options.output,
        )
        return 2
    output = data.with_suffix(".h5ad") if options.output is None else Path(f"{options.output}.h5ad")
    checks = "strict" if options.strict else "standard"
    try:
        if options.rule_config is not None:
            result = convert_from_rule_config(
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
            result = convert_from_packaged_rules(
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
    except _EXPECTED_FAILURES as error:
        logger.error(str(error))
        return 1
    _log_result(output, result)
    return 0


def _log_result(output: Path, result: ConversionResult) -> None:
    parsed = result.parsed
    logger.info(
        "wrote {}  shape=({}, {})  layers={}",
        output,
        parsed.obs.frame.height,
        parsed.var.frame.height,
        list(parsed.layers),
    )


def main() -> int:
    """Console-script entry point."""
    result = app()
    return int(result) if result is not None else 0


if __name__ == "__main__":
    sys.exit(main())
