"""apb2 CLI dispatcher.

Subcommands:
- convert <data> LEVEL       convert one quantification level of a vendor file to .h5ad
- export-schema              regenerate the packaged JSON Schema artifact
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger

from apb2.errors import IncompatibleSourceError
from apb2.input import format_for
from apb2.output import as_anndata, update_namespace
from apb2.parse_strategy import make_parse_strategy
from apb2.sources import SingleFile
from apb2.vendor_params.model import Parameters
from apb2.vendor_params.registry import parse_params
from apb2.vendor_parse_rules.documents import export as schema_export
from apb2.vendor_parse_rules.documents.select import (
    AmbiguousRuleError,
    DetectedSoftware,
    RuleUnavailableError,
    guess_software,
    software_slug,
)
from apb2.vendor_parse_rules.model import (
    LongRule,
    QuantificationLevel,
    WideRule,
    compose_rule,
    load_document,
)

app = App(name="apb2", help="apb2 CLI: rules-driven vendor-table conversion", help_on_error=True)

_SelectionMethod = Literal["software_version", "columns", "rule_config"]


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
    """Convert one quantification level of a vendor file to an AnnData (.h5ad).

    --params is the vendor parameter file and is required unless --rule-config is given.
    The vendor is detected from the parameter values and the column headers; --software
    (the rule folder slug, e.g. "diann") checks and disambiguates that detection. For
    compound workflows such as FragPipe with DIA-NN output, --params-software selects the
    parameter parser independently (e.g. "fragpipe"). --rule-config selects an explicit
    software-version document; LEVEL chooses one section. --output is an extensionless
    basename; apb2 appends .h5ad. Without --output, the result is written next to the
    input using the input stem. --strict promotes layer-contract warnings to errors.
    """
    if options.output is not None and options.output.suffix:
        logger.error(
            f"--output must be an extensionless basename, got {options.output}; apb2 appends .h5ad"
        )
        return 2
    output = data.with_suffix(".h5ad") if options.output is None else Path(f"{options.output}.h5ad")
    headers = tuple(format_for(data).columns())
    if options.rule_config is not None:
        return _convert_from_rule_config(data, level, output, options, options.rule_config)
    return _convert_from_packaged_rules(data, level, headers, output, options)


def _convert_from_rule_config(
    data: Path,
    level: QuantificationLevel,
    output: Path,
    options: ConvertCliOptions,
    rule_config: Path,
) -> int:
    """Compose one level of an explicit rule document and execute it."""
    document = load_document(rule_config)
    if level not in document.levels:
        logger.error(f"{rule_config} has no level {level!r}; available: {list(document.levels)}")
        return 1
    rule = compose_rule(document, level)
    if options.params is not None:
        parameters = parse_params(
            options.params,
            software=options.params_software or software_slug(document.software_name),
        )
        return _execute(data, output, rule, "rule_config", parameters, options.params, options)
    return _execute(data, output, rule, "rule_config", None, None, options)


def _convert_from_packaged_rules(
    data: Path,
    level: QuantificationLevel,
    headers: tuple[str, ...],
    output: Path,
    options: ConvertCliOptions,
) -> int:
    """Detect the packaged document from the evidence and execute one level of it."""
    if options.params is None:
        logger.error("pass --params (it gives the software version) or --rule-config PATH")
        return 1
    parser_slug = options.params_software or options.software or guess_software(headers)
    if parser_slug is None:
        logger.error(
            f"could not auto-detect the vendor for {data}; pass --software SLUG "
            "or --rule-config PATH"
        )
        return 1
    parameters = parse_params(options.params, software=parser_slug)
    try:
        detected = DetectedSoftware(parameters, headers)
    except (RuleUnavailableError, AmbiguousRuleError) as error:
        logger.error(str(error))
        return 1
    if options.software is not None and detected.software != options.software:
        logger.error(
            f"--software {options.software!r} does not match the detected vendor "
            f"{detected.software!r}"
        )
        return 1
    logger.info("vendor={} software_version={}", detected.software, detected.version or "missing")
    document = load_document(detected.get_rule_path())
    if level not in document.levels:
        logger.error(f"{document.path} has no level {level!r}; available: {list(document.levels)}")
        return 1
    rule = compose_rule(document, level)
    method: _SelectionMethod = "software_version" if detected.version is not None else "columns"
    return _execute(data, output, rule, method, parameters, options.params, options)


def _execute(
    data: Path,
    output: Path,
    rule: LongRule | WideRule,
    method: _SelectionMethod,
    parameters: Parameters | None,
    parameters_path: Path | None,
    options: ConvertCliOptions,
) -> int:
    """Parse one composed rule and write it atomically with its provenance."""
    try:
        strategy = make_parse_strategy(rule, SingleFile(data), parameters, strict=options.strict)
    except IncompatibleSourceError as error:
        logger.error(str(error))
        return 1
    parsed = strategy.parse()
    adata = as_anndata(parsed)
    update_namespace(adata, {"rule_selection_method": method})
    if parameters is not None:
        update_namespace(
            adata,
            {
                "search_parameters_version_status": (
                    "missing" if parameters.software_version is None else "present"
                ),
                "search_parameters_path": str(parameters_path),
                "search_parameters": json.dumps(parameters.model_dump(mode="json")),
            },
        )
    _write_atomically(output, adata.write_h5ad)
    logger.info(f"wrote {output}  shape={adata.shape}  layers={list(parsed.layers)}")
    return 0


def _write_atomically(output: Path, writer: Callable[[Path], None]) -> None:
    """Write beside the destination and replace it only after a complete write."""
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as folder:
        temporary = Path(folder) / output.name
        writer(temporary)
        temporary.replace(output)


@app.command(name="export-schema")
def export_schema_cmd() -> int:
    """Regenerate the packaged JSON Schema artifact from the pydantic models."""
    schema_export.main()
    return 0


def main() -> int:
    """Console-script entry point."""
    rc = app()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
