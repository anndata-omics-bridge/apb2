"""MSAngel JSON workflow parameter parser."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import IO, cast

from apb2.parserV2.vendor_params.parsers.shared.common import (
    homogenize_paren_mods,
    modifications,
    read_text,
    tolerance_unit,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    ParamsError,
    Probability,
    SearchedModification,
)


def _mods(value: str, mod_type: ModType) -> list[SearchedModification]:
    tokens = (
        token for part in value.split(",") for token in homogenize_paren_mods(part.strip(), {})
    )
    return modifications(tokens, mod_type)


def _operation_values(
    operations: list[object],
) -> tuple[tuple[str, dict[str, object]] | None, Probability | None, int | None]:
    """Extract the one search form and validation values from the operation list."""
    search: tuple[str, dict[str, object]] | None = None
    psm_fdr: Probability | None = None
    min_length: int | None = None
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        engines = operation.get("searchEnginesWithForms")
        if isinstance(engines, list) and len(engines) == 1:
            name, configuration = engines[0]
            search = (str(name), cast("dict[str, object]", configuration))
        validation = operation.get("validationConfig")
        if isinstance(validation, dict):
            validation_values = cast("dict[str, object]", validation)
            if validation_values.get("psmExpectedFdr") is not None:
                psm_fdr = Probability(value=float(str(validation_values["psmExpectedFdr"])) / 100)
            for raw_filter in cast("list[object]", validation_values.get("psmFilters", [])):
                if isinstance(raw_filter, dict) and raw_filter.get("parameter") == "PEP_SEQ_LENGTH":
                    min_length = int(float(str(cast("dict[str, object]", raw_filter)["threshold"])))
    return search, psm_fdr, min_length


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse the single-search-engine MSAngel workflow used for quantification."""
    payload = json.loads(read_text(source))
    if not isinstance(payload, dict) or not isinstance(payload.get("msAngelVersion"), str):
        raise ParamsError("not an MSAngel parameter file")
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ParamsError("MSAngel operations are missing")
    search, psm_fdr, min_length = _operation_values(operations)
    if search is None:
        raise ParamsError("MSAngel requires exactly one search engine")
    engine, configuration = search
    if engine != "Mascot":
        raise ParamsError(f"unsupported MSAngel search engine: {engine!r}")
    parameter_map = configuration.get("paramMap")
    if not isinstance(parameter_map, dict):
        raise ParamsError("MSAngel Mascot parameter map is missing")
    values = cast("dict[str, object]", parameter_map)
    charges = [int(value) for value in re.findall(r"(\d+)\+", str(values.get("CHARGE", "")))]
    return Parameters(
        software_name="MSAngel",
        software_version=str(payload["msAngelVersion"]),
        quantification_software="MSAngel",
        quantification_software_version=str(payload["msAngelVersion"]),
        acquisition_method="DDA",
        search_engine=engine,
        ident_fdr_psm=psm_fdr,
        enable_match_between_runs=True,
        precursor_mass_tolerance=MassTolerance(
            mode="absolute",
            value=float(str(values["TOL"])),
            unit=tolerance_unit(str(values["TOLU"])),
        ),
        fragment_mass_tolerance=MassTolerance(
            mode="absolute",
            value=float(str(values["ITOL"])),
            unit=tolerance_unit(str(values["ITOLU"])),
        ),
        enzyme=str(values["CLE"]),
        allowed_miscleavages=int(float(str(values["PFA"]))),
        min_peptide_length=min_length,
        fixed_mods=_mods(str(values.get("MODS", "")), ModType.fixed),
        variable_mods=_mods(str(values.get("IT_MODS", "")), ModType.variable),
        min_precursor_charge=min(charges) if charges else None,
        max_precursor_charge=max(charges) if charges else None,
    )
