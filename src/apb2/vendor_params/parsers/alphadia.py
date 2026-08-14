"""AlphaDIA parameter parser.

AlphaDIA ships a *run log*, not a config file: ANSI-coloured, timestamped, with the
resolved config rendered as an indented tree and overridden entries annotated
``[user defined, default: X]``::

    0:00:00.015201 PROGRESS: version: 1.10.3
    0:00:00.093070 INFO: │   ├──enzyme: trypsin/p [user defined, default: trypsin]
    0:00:00.093094 INFO: │   ├──fixed_modifications: Carbamidomethyl@C

Field mapping mirrors ``proteobench.io.params.alphadia`` so the two agree on what a
submission was searched with.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import IO

from apb2.vendor_params.model import MassTolerance, Parameters, ParamsError
from apb2.vendor_params.parsers._common import read_lines

_ANSI_RE = re.compile(r"(\x9b|\x1b\[)[0-?]*[ -/]*[@-~]")
_TIMESTAMP_RE = re.compile(r"(\d+ days?,\s*)?(\d+):\d{2}:\d{2}\.?\d*")
_LEVEL_RE = re.compile(r"(PROGRESS|INFO|WARNING|ERROR|CRITICAL|DEBUG):")
_TREE_RE = re.compile(r"^\s*(├──|└──|│)\s*|\s*(├──|└──|│)\s*")
# An overridden entry renders as "value [user defined, default: other]"; the
# effective value is the one before the bracket.
_ANNOTATION_RE = re.compile(r"\s*[\[(]\s*user defined.*$", re.IGNORECASE)

# The version line AlphaDIA prints at startup. The config tree also contains a
# bare `version:` key (the schema version, "1" or "None"), so the software
# version must be anchored on the PROGRESS prefix rather than the key alone.
_VERSION_RE = re.compile(r"PROGRESS:\s*version:\s*(?P<version>\S+)")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

_MODIFICATION_DELIMITER = ";"


def _clean(line: str) -> str:
    """Strip ANSI codes, timestamp, log level, and tree drawing characters."""
    for pattern in (_ANSI_RE, _TIMESTAMP_RE, _LEVEL_RE, _TREE_RE):
        line = pattern.sub("", line)
    return line.strip()


def _effective_value(value: str) -> str:
    """Drop the ``[user defined, default: …]`` annotation, keeping the applied value."""
    return _ANNOTATION_RE.sub("", value).strip()


def _homogenize_mods(mod_string: str) -> str:
    """Render AlphaDIA ``Name@Residue`` tokens in APB's ProForma-like notation.

    ``Oxidation@M`` -> ``M[Oxidation]``;
    ``Acetyl@Protein_N-term`` -> ``Protein N-term[Acetyl]``.
    """
    out: list[str] = []
    for token in mod_string.split(_MODIFICATION_DELIMITER):
        token = token.strip()
        if not token:
            continue
        if "@" in token:
            name, residue = token.split("@", 1)
            out.append(f"{residue.replace('_', ' ')}[{name}]")
        else:
            out.append(token)
    return ", ".join(out)


def _nested_range(lines: list[str], start: int) -> tuple[int, int] | None:
    """Read the two-line ``min``/``max`` block that follows a bare range key.

    ``precursor_len:`` and friends print their bounds on the following indented
    lines rather than inline.
    """
    found: list[int] = []
    for raw in lines[start + 1 :]:
        cleaned = _effective_value(_clean(raw))
        if not cleaned:
            continue
        match = re.search(r"-?\d+", cleaned)
        if match is None:
            break
        found.append(int(match.group(0)))
        if len(found) == 2:
            return found[0], found[1]
    return None


def _ppm(value: str) -> MassTolerance:
    """AlphaDIA reports both search tolerances in ppm.

    A tolerance of ``0`` means AlphaDIA calibrated it from the data rather than
    searching at zero width, so it routes through :meth:`MassTolerance.parse` to
    reach the same automatic-calibration record other vendors produce.
    """
    text = value.strip()
    if not _FLOAT_RE.fullmatch(text):
        raise ValueError(f"AlphaDIA ppm tolerance must be numeric, got {value!r}")
    numeric = abs(float(text))
    if numeric == 0:
        return MassTolerance(mode="automatic", label="Automatic calibration")
    return MassTolerance(mode="absolute", value=numeric, unit="ppm")


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse an AlphaDIA run log into a :class:`Parameters` record.

    Raises :class:`ParamsError` when the file carries no AlphaDIA version banner,
    so a mis-routed parameter file degrades cleanly rather than yielding an empty
    record.
    """
    lines = read_lines(source)

    scalars: dict[str, str] = {}
    ranges: dict[str, tuple[int, int]] = {}
    version: str | None = None

    for index, raw in enumerate(lines):
        if version is None:
            banner = _VERSION_RE.search(_ANSI_RE.sub("", raw))
            if banner is not None:
                version = banner.group("version")

        cleaned = _clean(raw)
        if ":" not in cleaned:
            continue
        key, _, value = cleaned.partition(":")
        key, value = key.strip(), _effective_value(value)

        if not value and key in {"precursor_len", "precursor_charge"}:
            bounds = _nested_range(lines, index)
            if bounds is not None and key not in ranges:
                ranges[key] = bounds
            continue
        # The config tree is printed once; later log chatter reuses these words in
        # prose, so first occurrence wins.
        if value and key not in scalars:
            scalars[key] = value

    if version is None:
        raise ParamsError("no AlphaDIA version banner found; not an AlphaDIA run log")

    fdr = scalars.get("fdr")
    length = ranges.get("precursor_len")
    charge = ranges.get("precursor_charge")
    mbr = scalars.get("mbr_step_enabled")
    precursor_tolerance = scalars.get("target_ms1_tolerance")
    fragment_tolerance = scalars.get("target_ms2_tolerance")
    fixed_modifications = scalars.get("fixed_modifications")
    variable_modifications = scalars.get("variable_modifications")

    return Parameters.model_validate(
        {
            "software_name": "AlphaDIA",
            "software_version": version,
            "search_engine": "AlphaDIA",
            "search_engine_version": version,
            "acquisition_method": "DIA",
            "ident_fdr_psm": fdr,
            "ident_fdr_protein": fdr,
            "enable_match_between_runs": mbr.strip() == "True" if mbr is not None else None,
            "precursor_mass_tolerance": (
                None if precursor_tolerance is None else _ppm(precursor_tolerance)
            ),
            "fragment_mass_tolerance": (
                None if fragment_tolerance is None else _ppm(fragment_tolerance)
            ),
            "enzyme": scalars.get("enzyme"),
            "allowed_miscleavages": scalars.get("missed_cleavages"),
            "min_peptide_length": length[0] if length else None,
            "max_peptide_length": length[1] if length else None,
            "fixed_mods": (
                [] if fixed_modifications is None else _homogenize_mods(fixed_modifications)
            ),
            "variable_mods": (
                [] if variable_modifications is None else _homogenize_mods(variable_modifications)
            ),
            "max_mods": scalars.get("max_var_mod_num"),
            "min_precursor_charge": charge[0] if charge else None,
            "max_precursor_charge": charge[1] if charge else None,
            "quantification_method": "DirectLFQ",
            "predictors_library": "AlphaPeptDeep",
        }
    )
