"""Sage parameter-file parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from apb2.modifications import unimod_registry
from apb2.vendor_params.model import Parameters
from apb2.vendor_params.parsers._common import (
    format_tolerance_range,
    read_text,
)

MASS_TOLERANCE = 0.001

# Sage uses "[" for N-terminal and "]" for C-terminal modifications.
RESIDUE_MAP = {"[": "Protein N-term", "]": "Protein C-term", "^": "N-term", "$": "C-term"}


def _lookup_mod_name(mass: float) -> str:
    """Return a modification name for a mass shift within tolerance, else the raw mass."""
    result = unimod_registry.find_by_mass(mass, tolerance=MASS_TOLERANCE)
    if isinstance(result, unimod_registry.UnimodMatch):
        return result.entry.name
    return str(mass)


def _parse_static_mods(mods: dict[str, float]) -> str:
    """Render Sage ``static_mods`` ({residue: mass}) as a ProForma-like string."""
    results = []
    for residue, mass in mods.items():
        res = RESIDUE_MAP.get(residue, residue)
        results.append(f"{res}[{_lookup_mod_name(mass)}]")
    return ", ".join(results)


def _parse_variable_mods(mods: dict[str, list[float]]) -> str:
    """Render Sage ``variable_mods`` ({residue: [masses]}) as a ProForma-like string."""
    results = []
    for residue, masses in mods.items():
        res = RESIDUE_MAP.get(residue, residue)
        for mass in masses:
            results.append(f"{res}[{_lookup_mod_name(mass)}]")
    return ", ".join(results)


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse a Sage JSON parameter file into a :class:`Parameters` record.

    Accepts a filesystem path or an open file-like object (bytes or text).
    Field mapping mirrors ProteoBench's ``proteobench.io.params.sage.extract_params``
    so existing expected-output CSVs are reproduced unchanged.
    """
    data = json.loads(read_text(source))

    enzyme = data["database"]["enzyme"]["cleave_at"]
    if enzyme in ("KR", "RK"):
        if "restrict" not in data["database"]["enzyme"]:
            enzyme = "Trypsin/P"
        elif data["database"]["enzyme"]["restrict"] == "P":
            enzyme = "Trypsin"
        # restrict present but not "P" (e.g. null) → keep raw KR/RK

    semi = data["database"]["enzyme"].get("semi_enzymatic")
    if semi is None or semi is False:
        semi_enzymatic = False
    elif semi is True:
        semi_enzymatic = True
    else:
        raise ValueError(f"unknown semi_enzymatic value: {semi!r}")

    max_len = data["database"]["enzyme"]["max_len"]

    # `quant.lfq_settings.combine_charge_states` decides whether `lfq.tsv` is ion- or
    # peptidoform-level: Sage's DOCS.md gives it default `true` ("Combine all charge states
    # for quantification"), and a combined row is written with `charge = -1`. Absent when the
    # run had no LFQ stage, in which case there is no charge-state decision to report.
    lfq_settings = data.get("quant", {}).get("lfq_settings") or {}
    combine_charge_states = (
        lfq_settings.get("combine_charge_states", True) if lfq_settings else None
    )

    return Parameters.model_validate(
        {
            "software_name": "Sage",
            "software_version": data["version"],
            "combine_charge_states": combine_charge_states,
            "search_engine": "Sage",
            "search_engine_version": data["version"],
            "enzyme": enzyme,
            "semi_enzymatic": semi_enzymatic,
            "allowed_miscleavages": data["database"]["enzyme"]["missed_cleavages"],
            "fixed_mods": _parse_static_mods(data["database"]["static_mods"]),
            "variable_mods": _parse_variable_mods(data["database"]["variable_mods"]),
            "precursor_mass_tolerance": format_tolerance_range(data["precursor_tol"]),
            "fragment_mass_tolerance": format_tolerance_range(data["fragment_tol"]),
            "min_peptide_length": int(data["database"]["enzyme"]["min_len"]),
            "max_peptide_length": int(max_len) if max_len is not None else None,
            "max_mods": int(data["database"]["max_variable_mods"]),
            "min_precursor_charge": int(data["precursor_charge"][0]),
            "max_precursor_charge": int(data["precursor_charge"][1]),
            "enable_match_between_runs": True,
        }
    )
