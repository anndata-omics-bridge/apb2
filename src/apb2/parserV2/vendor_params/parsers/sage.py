"""Sage parameter-file parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.common import (
    modifications,
    read_text,
    symmetric_tolerance,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    SearchedModification,
)

# Sage uses "[" for N-terminal and "]" for C-terminal modifications.
RESIDUE_MAP = {"[": "Protein N-term", "]": "Protein C-term", "^": "N-term", "$": "C-term"}


def _tolerance(declared: dict[str, list[float]]) -> MassTolerance:
    """Read Sage's ``{unit: [lower, upper]}`` tolerance."""
    for unit, bounds in declared.items():
        lower, upper = bounds
        return symmetric_tolerance(lower, upper, unit)
    raise ValueError(f"Sage tolerance declares no unit: {declared!r}")


def _static_mods(mods: dict[str, float], mod_type: ModType) -> list[SearchedModification]:
    """Resolve Sage ``static_mods`` ({residue: mass})."""
    return modifications(
        (f"{RESIDUE_MAP.get(residue, residue)}[{mass}]" for residue, mass in mods.items()),
        mod_type,
    )


def _variable_mods(mods: dict[str, list[float]], mod_type: ModType) -> list[SearchedModification]:
    """Resolve Sage ``variable_mods`` ({residue: [masses]})."""
    return modifications(
        (
            f"{RESIDUE_MAP.get(residue, residue)}[{mass}]"
            for residue, masses in mods.items()
            for mass in masses
        ),
        mod_type,
    )


def _enzyme(declared: dict[str, object]) -> str:
    """Name the protease Sage's cleavage rule describes.

    ``restrict`` names the residue that blocks cleavage when it follows the cleavage site, so a
    null or absent ``restrict`` means nothing blocks it: cleaving after K and R regardless of a
    following proline is Trypsin/P, and blocking on proline is Trypsin.
    """
    cleave_at = str(declared["cleave_at"])
    if cleave_at not in ("KR", "RK"):
        return cleave_at
    return "Trypsin" if declared.get("restrict") == "P" else "Trypsin/P"


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse a Sage JSON parameter file into a :class:`Parameters` record.

    Accepts a filesystem path or an open file-like object (bytes or text).
    Field mapping mirrors ProteoBench's ``proteobench.io.params.sage.extract_params``
    so existing expected-output CSVs are reproduced unchanged.
    """
    data = json.loads(read_text(source))
    database = data["database"]
    enzyme = database["enzyme"]
    max_len = enzyme["max_len"]

    # `quant.lfq_settings.combine_charge_states` decides whether `lfq.tsv` is ion- or
    # peptidoform-level: Sage's DOCS.md gives it default `true` ("Combine all charge states
    # for quantification"), and a combined row is written with `charge = -1`. Absent when the
    # run had no LFQ stage, in which case there is no charge-state decision to report.
    lfq_settings = data.get("quant", {}).get("lfq_settings") or {}
    combine_charge_states = (
        lfq_settings.get("combine_charge_states", True) if lfq_settings else None
    )

    return Parameters(
        software_name="Sage",
        software_version=data["version"],
        combine_charge_states=combine_charge_states,
        search_engine="Sage",
        search_engine_version=data["version"],
        enzyme=_enzyme(enzyme),
        semi_enzymatic=bool(enzyme.get("semi_enzymatic")),
        allowed_miscleavages=enzyme["missed_cleavages"],
        fixed_mods=_static_mods(database["static_mods"], ModType.fixed),
        variable_mods=_variable_mods(database["variable_mods"], ModType.variable),
        precursor_mass_tolerance=_tolerance(data["precursor_tol"]),
        fragment_mass_tolerance=_tolerance(data["fragment_tol"]),
        min_peptide_length=int(enzyme["min_len"]),
        max_peptide_length=int(max_len) if max_len is not None else None,
        max_mods=int(database["max_variable_mods"]),
        min_precursor_charge=int(data["precursor_charge"][0]),
        max_precursor_charge=int(data["precursor_charge"][1]),
        enable_match_between_runs=True,
    )
