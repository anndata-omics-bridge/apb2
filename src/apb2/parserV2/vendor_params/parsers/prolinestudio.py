"""ProlineStudio workbook parameter parser."""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd

from apb2.parserV2.vendor_params.parsers.shared.common import (
    homogenize_paren_mods,
    modifications,
    tolerance_from_text,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    ModType,
    Parameters,
    ParamsError,
    Probability,
    SearchedModification,
)


def _unique(frame: pd.DataFrame, name: str) -> object:
    row = cast("pd.Series[Any]", frame.loc[name])
    values = list(dict.fromkeys(row.dropna().tolist()))
    if len(values) != 1:
        raise ParamsError(f"ProlineStudio parameter {name!r} is not constant: {values!r}")
    return values[0]


def _mods(value: object, mod_type: ModType) -> list[SearchedModification]:
    tokens = (
        token for part in str(value).split(";") for token in homogenize_paren_mods(part.strip(), {})
    )
    return modifications(tokens, mod_type)


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse ProlineStudio's parameter sheets from its result workbook."""
    try:
        search = pd.read_excel(
            source,
            sheet_name="Search settings and infos",
            header=None,
            index_col=0,
            engine="calamine",
        )
    except (OSError, ValueError) as error:
        raise ParamsError(f"not a ProlineStudio workbook: {error}") from error
    required = {
        "software_name",
        "software_version",
        "enzymes",
        "max_missed_cleavages",
        "fixed_ptms",
        "variable_ptms",
        "peptide_charge_states",
        "peptide_mass_error_tolerance",
        "fragment_mass_error_tolerance",
    }
    if not required <= set(search.index.astype(str)):
        raise ParamsError("not a ProlineStudio parameter workbook")
    charges = [
        int(value)
        for value in re.findall(r"(\d+)\+", str(_unique(search, "peptide_charge_states")))
    ]
    imports = pd.read_excel(
        source,
        sheet_name="Import and filters",
        header=None,
        index_col=0,
        engine="calamine",
    )
    psm_fdr = float(str(_unique(imports, "psm_filter_expected_fdr"))) / 100.0
    length_match = re.search(r"\[threshold_value=([0-9]+)]", str(_unique(imports, "psm_filter_2")))
    quant = pd.read_excel(
        source,
        sheet_name="Quant config",
        header=None,
        index_col=0,
        engine="calamine",
    )
    version: str | None = None
    try:
        statistics = pd.read_excel(
            source,
            sheet_name="Dataset statistics and infos",
            header=None,
            index_col=0,
            engine="calamine",
        )
        version = str(_unique(statistics, "version"))
    except ValueError:
        pass
    precursor = str(_unique(search, "peptide_mass_error_tolerance"))
    fragment = str(_unique(search, "fragment_mass_error_tolerance"))
    return Parameters(
        software_name="ProlineStudio",
        software_version=version,
        quantification_software="ProlineStudio",
        quantification_software_version=version,
        acquisition_method="DDA",
        search_engine=str(_unique(search, "software_name")),
        search_engine_version=str(_unique(search, "software_version")),
        ident_fdr_psm=Probability(value=psm_fdr),
        enable_match_between_runs=any(
            "cross assignment" in str(value).lower() for value in quant.index
        ),
        precursor_mass_tolerance=tolerance_from_text(precursor),
        fragment_mass_tolerance=tolerance_from_text(fragment),
        enzyme=str(_unique(search, "enzymes")),
        allowed_miscleavages=int(float(str(_unique(search, "max_missed_cleavages")))),
        min_peptide_length=int(length_match.group(1)) if length_match else None,
        fixed_mods=_mods(_unique(search, "fixed_ptms"), ModType.fixed),
        variable_mods=_mods(_unique(search, "variable_ptms"), ModType.variable),
        min_precursor_charge=min(charges),
        max_precursor_charge=max(charges),
    )
