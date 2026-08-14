"""DIA-NN log/cfg parameter-file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TypedDict

from packaging.version import VERSION_PATTERN, Version

from apb2.vendor_params.model import AcquisitionMethod, MassTolerance, Parameters, ParamsError
from apb2.vendor_params.parsers._common import read_lines

_Source = Path | IO[bytes] | IO[str]

MODIFICATION_MAPPING = {
    # Command-line short forms
    "unimod4": "C[Carbamidomethyl]",
    # Descriptive forms
    "Carbamidomethyl (C)": "C[Carbamidomethyl]",
    "Cysteine carbamidomethylation": "C[Carbamidomethyl]",
    "Oxidation (M)": "M[Oxidation]",
    "Acetyl": "N-term[Acetyl]",
    # UniMod short forms (from cfg-extracted log text)
    "UniMod:4": "C[Carbamidomethyl]",
    "UniMod:35": "M[Oxidation]",
    "UniMod:1": "N-term[Acetyl]",
    "UniMod:21": "S[Phospho], T[Phospho], Y[Phospho]",
    "UniMod:121": "K[GG]",
    # UniMod full forms with slash separators (from command-line parsing)
    "UniMod:35/15.994915/M": "M[Oxidation]",
    "UniMod:1/42.010565/*n": "N-term[Acetyl]",
    "UniMod:21/79.966331/STY": "STY[Phospho]",
    "UniMod:121/114.042927/K": "K[GG]",
    # UniMod full forms with comma separators (alternative notation)
    "UniMod:1,42.010565,*n": "N-term[Acetyl]",
    "UniMod:21,79.966331,STY": "STY[Phospho]",
    "UniMod:121,114.042927,K": "K[GG]",
}

_FRAGMENT_TOL = r"Optimised mass accuracy: (\d*\.?\d+) ppm"
_PRECURSOR_TOL = r"Recommended MS1 mass accuracy setting: (\d*\.?\d+) ppm"
_SOFTWARE_VERSION = r"DIA-NN\s(.*?)\s\(Data-Independent Acquisition by Neural Networks\)"
_SCAN_WINDOW = r"Scan window radius set to (\d+)"
_FDR = r"Output will be filtered at (\d+\.\d+) FDR"
_MIN_PEP_LEN = r"Min peptide length set to (\d+)"
_MAX_PEP_LEN = r"Max peptide length set to (\d+)"
_MIN_Z = r"Min precursor charge set to (\d+)"
_MAX_Z = r"Max precursor charge set to (\d+)"
_MIN_MZ_PREC = r"Min precursor m/z set to (\d+)"
_MAX_MZ_PREC = r"Max precursor m/z set to (\d+)"
_MIN_MZ_FRAG = r"Min fragment m/z set to (\d+)"
_MAX_MZ_FRAG = r"Max fragment m/z set to (\d+)"
_CLEAVAGE = r"In silico digest will involve cuts at (.*)"
_CLEAVAGE_EXC = r"But excluding cuts at (.*)"
_MISSED_CLEAVAGES = r"Maximum number of missed cleavages set to (\d+)"
_MAX_MODS = r"Maximum number of variable modifications set to (\d+)"
_FIXED_MODS_1 = r"(.*) enabled as a fixed modification"
_FIXED_MODS_2 = r"Modification (.*) with mass delta \d+\.*\d* at .+ will be considered as fixed"
_VAR_MODS = r"Modification (.*) with mass delta \d+\.*\d* at .+ will be considered as variable"
_QUANT_MODE = r"(.*?) quantification mode"
_PROTEIN_INFERENCE = r"Implicit protein grouping: (.*);"
_NORMALISATION_DISABLED = r"(Normalisation disabled)"
_MBR_FLAG = r"(MBR enabled)|(reanalyse them)"
_DDA_LOG_MARKER = "All runs will be analysed as DDA runs"
_DIANN_VERSION_CORE_RE = re.compile(rf"^(?:{VERSION_PATTERN})$", re.IGNORECASE | re.VERBOSE)
_DIANN_UNIMOD_CHANGE_VERSION = Version("1.8")

_PROT_INF_MAP = {"isoform IDs": "Isoforms", "protein names": "Protein_names", "genes": "Genes"}
_CommandValue = bool | list[str]


@dataclass(frozen=True, slots=True)
class ParsedDiannVersion:
    """A DIA-NN version that can drive version-specific syntax."""

    value: Version


@dataclass(frozen=True, slots=True)
class MissingDiannVersion:
    """A command-line-only DIA-NN record contains no version banner."""


type DiannVersionEvidence = ParsedDiannVersion | MissingDiannVersion

MISSING_DIANN_VERSION = MissingDiannVersion()


class DiannImplicitDefaults(TypedDict):
    """DIA-NN defaults that are present even when the log omits them."""

    min_precursor_charge: int
    max_precursor_charge: int
    min_peptide_length: int
    max_peptide_length: int
    min_fragment_mz: int
    max_fragment_mz: int
    min_precursor_mz: int
    max_precursor_mz: int


class DiannParameterData(TypedDict, total=False):
    """Precisely typed values assembled from DIA-NN parser stages."""

    software_name: str
    software_version: str | None
    acquisition_method: AcquisitionMethod
    search_engine: str
    search_engine_version: str | None
    ident_fdr_psm: float | None
    ident_fdr_protein: float | None
    enable_match_between_runs: bool
    precursor_mass_tolerance: str | float | MassTolerance | None
    fragment_mass_tolerance: str | float | MassTolerance | None
    enzyme: str
    allowed_miscleavages: int | None
    min_peptide_length: int | None
    max_peptide_length: int | None
    fixed_mods: str | None
    variable_mods: str | None
    max_mods: int | None
    min_precursor_charge: int | None
    max_precursor_charge: int | None
    min_precursor_mz: str | int | None
    max_precursor_mz: str | int | None
    min_fragment_mz: str | int | None
    max_fragment_mz: str | int | None
    quantification_method: str
    protein_inference: str
    abundance_normalization_ions: str
    predictors_library: str | None
    scan_window: int | None


# DIA-NN built-in defaults, reported when the log/cfg omits the corresponding
# setting. These mirror DIA-NN's own built-in defaults and are version-sensitive:
# re-verify against DIA-NN release notes when bumping supported versions.
_DIANN_IMPLICIT_DEFAULTS: DiannImplicitDefaults = {
    "min_precursor_charge": 1,
    "max_precursor_charge": 4,
    "min_peptide_length": 7,
    "max_peptide_length": 30,
    "min_fragment_mz": 200,
    "max_fragment_mz": 1800,
    "min_precursor_mz": 300,
    "max_precursor_mz": 1800,
}


def _find_cmdline(lines: list[str]) -> str | None:
    for line in lines:
        if "diann" in line and "--" in line:
            return line.strip()
    return None


def _parse_diann_version(software_version: str) -> ParsedDiannVersion:
    """Validate the version token used by DIA-NN's version-specific syntax."""
    normalized = software_version.strip()
    if not normalized:
        raise ParamsError("DIA-NN version banner contains no version value")
    head = normalized.split(maxsplit=1)[0]
    if not _DIANN_VERSION_CORE_RE.fullmatch(head):
        raise ParamsError(f"invalid DIA-NN version string: {software_version!r}")
    return ParsedDiannVersion(Version(head))


def _parse_cmdline(
    cmd: str,
    version: DiannVersionEvidence,
) -> dict[str, _CommandValue]:
    settings: dict[str, _CommandValue] = {}
    var_mods: list[str] = []
    fixed_mods: list[str] = []
    for parts in (s.split() for s in cmd.split(" --")):
        if not parts:  # empty token (e.g. an empty command line) — nothing to parse
            continue
        key, values = parts[0], parts[1:]
        if key.startswith("unimod"):
            _append_unimod(key, parts, version, fixed_mods, var_mods)
            continue
        if len(parts) == 1:
            settings[key] = True
        elif key == "var-mod":
            var_mods.append("".join(values).replace(",", "/"))
        else:
            settings[key] = values

    settings["var-mod"] = var_mods
    if "mod" not in settings:
        settings["mod"] = fixed_mods
    return settings


def _missing_cmdline_settings() -> dict[str, _CommandValue]:
    """Return the explicit DIA-NN settings state when no command line was logged."""
    return {"var-mod": [], "mod": []}


def _append_unimod(
    key: str,
    parts: list[str],
    version: DiannVersionEvidence,
    fixed_mods: list[str],
    variable_mods: list[str],
) -> None:
    """Apply DIA-NN's version-specific command-line UniMod shorthand."""
    if len(parts) != 1:
        raise ValueError(f"invalid `unimod` format: {parts}")
    if isinstance(version, MissingDiannVersion):
        if key == "unimod4":
            # Carbamidomethyl is fixed on both sides of the DIA-NN 1.8 syntax change.
            fixed_mods.append(key)
            return
        raise ParamsError(
            f"DIA-NN version is required to classify version-sensitive option --{key}"
        )
    below_1_8 = version.value < _DIANN_UNIMOD_CHANGE_VERSION
    if not below_1_8:
        fixed_mods.append(key)
        return
    if key == "unimod4":
        fixed_mods.append("Carbamidomethyl (C)")
    elif key == "unimod35":
        variable_mods.append("Oxidation (M)")


def _arguments(cmd_dict: dict[str, _CommandValue], name: str) -> list[str] | None:
    """Return arguments for one option, rejecting a value-less flag."""
    value = cmd_dict.get(name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"DIA-NN command-line setting {name!r} must contain arguments")
    return value


def _flag(cmd_dict: dict[str, _CommandValue], name: str) -> bool | None:
    """Return a command-line flag, rejecting unexpected arguments."""
    value = cmd_dict.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        raise TypeError(f"DIA-NN command-line flag {name!r} must not contain arguments")
    return value


def _extract_with_regex(lines: list[str], regex: str, search_all: bool = False) -> str | None:
    container: list[str] = []
    for line in lines:
        match = re.search(regex, line)
        if not match:
            continue
        if not search_all:
            return match.group(1)
        container.append(match.group(1))
    return container[-1] if container else None


def _extract_cfg_text(
    lines: list[str],
    regex: str,
    *,
    search_all: bool = False,
) -> str | None:
    """Extract an optional text value from DIA-NN's free-text cfg block."""
    return _extract_with_regex(lines, regex, search_all=search_all)


def _extract_cfg_int(lines: list[str], regex: str) -> int | None:
    """Extract an optional integer from DIA-NN's free-text cfg block."""
    raw = _extract_with_regex(lines, regex)
    return None if raw is None else int(raw)


def _extract_cfg_float(lines: list[str], regex: str) -> float | None:
    """Extract an optional float from DIA-NN's free-text cfg block."""
    raw = _extract_with_regex(lines, regex)
    return None if raw is None else float(raw)


def _extract_cfg_int_or_default(
    lines: list[str],
    regex: str,
    default: int,
) -> int:
    """Extract an integer or use the named DIA-NN default when absent."""
    value = _extract_cfg_int(lines, regex)
    return default if value is None else value


def _extract_last_cfg_text_or_default(
    lines: list[str],
    regex: str,
    default: str,
) -> str:
    """Extract the final matching cfg value or use the named DIA-NN default."""
    value = _extract_cfg_text(lines, regex, search_all=True)
    return default if value is None else value


def _extract_modifications(lines: list[str], regexes: list[str]) -> str | None:
    joined = "\n".join(lines)
    mods: list[str] = []
    for regex in regexes:
        for match in re.finditer(regex, joined):
            value = match.group(1)
            if not value.endswith("\n"):
                value = value + "\n"
            mods.append(value)
    return ",".join(mods).replace("\n", "") if mods else None


def _protein_inference(cmd_dict: dict[str, _CommandValue]) -> str:
    if "no-prot-inf" in cmd_dict:
        return "Disabled"
    if "pg-level" in cmd_dict:
        values = cmd_dict["pg-level"]
        if not isinstance(values, list) or not values:
            raise ValueError("DIA-NN pg-level requires a value")
        return values[0]
    return "Genes"


def _quantification_strategy(cmd_dict: dict[str, _CommandValue]) -> str:
    if "direct-quant" in cmd_dict:
        return "Legacy"
    if "high-acc" in cmd_dict:
        return "QuantUMS high-accuracy"
    return "QuantUMS high-precision"


def _predictors_library(cmd_dict: dict[str, _CommandValue]) -> str | None:
    if "predictor" in cmd_dict:
        return "{'RT': 'DIANN', 'IM': 'DIANN', 'MS2_int': 'DIANN'}"
    if "lib" in cmd_dict and not isinstance(cmd_dict["lib"], bool):
        return (
            "{'RT': 'User defined speclib', 'IM': 'User defined speclib', "
            "'MS2_int': 'User defined speclib'}"
        )
    return None


def _normalize_enzyme(enzyme_str: str) -> str:
    if enzyme_str == "K*,R*":
        return "Trypsin/P"
    if enzyme_str == "K*,R*,!P":
        return "Trypsin"
    return enzyme_str


def _defaults() -> DiannParameterData:
    """Static defaults seeded before any log/cmdline/cfg parsing."""
    return {
        "software_name": "DIA-NN",
        "search_engine": "DIA-NN",
        "enable_match_between_runs": False,
        "quantification_method": "QuantUMS high-precision",
        "protein_inference": "Genes",
        **_DIANN_IMPLICIT_DEFAULTS,
    }


def _acquisition_method(
    lines: list[str],
    cmd_dict: dict[str, _CommandValue],
) -> AcquisitionMethod:
    """Return DIA-NN's acquisition mode from its command line or log marker."""
    if "dda" in cmd_dict or any(line.strip() == _DDA_LOG_MARKER for line in lines):
        return "DDA"
    return "DIA"


def _command_identification(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Extract command-line identification and MBR fields."""
    out: DiannParameterData = {}
    q_value = _arguments(cmd_dict, "qvalue")
    if q_value is not None:
        out["ident_fdr_psm"] = float(q_value[0])
    reanalyse = _flag(cmd_dict, "reanalyse")
    if reanalyse is not None:
        out["enable_match_between_runs"] = reanalyse
    return out


def _command_tolerances(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Extract command-line precursor and fragment tolerances."""
    out: DiannParameterData = {}
    precursor = _arguments(cmd_dict, "mass-acc-ms1")
    if precursor is not None:
        out["precursor_mass_tolerance"] = float(precursor[0])
    fragment = _arguments(cmd_dict, "mass-acc")
    if fragment is not None:
        out["fragment_mass_tolerance"] = float(fragment[0])
    return out


def _command_digest(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Extract command-line enzyme, cleavage, and peptide-length fields."""
    enzyme = _arguments(cmd_dict, "cut")
    out: DiannParameterData = {
        "enzyme": "Trypsin/P" if enzyme is None else _normalize_enzyme("".join(enzyme))
    }
    missed_cleavages = _arguments(cmd_dict, "missed-cleavages")
    if missed_cleavages is not None:
        out["allowed_miscleavages"] = int(missed_cleavages[0])
    minimum = _arguments(cmd_dict, "min-pep-len")
    if minimum is not None:
        out["min_peptide_length"] = int(minimum[0])
    maximum = _arguments(cmd_dict, "max-pep-len")
    if maximum is not None:
        out["max_peptide_length"] = int(maximum[0])
    return out


def _command_modifications(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Extract command-line fixed and variable modification fields."""
    out: DiannParameterData = {}
    maximum = _arguments(cmd_dict, "var-mods")
    if maximum is not None:
        out["max_mods"] = int(maximum[0])
    fixed = _arguments(cmd_dict, "mod")
    if fixed is not None:
        out["fixed_mods"] = ",".join(fixed)
    variable = _arguments(cmd_dict, "var-mod")
    if variable is not None:
        out["variable_mods"] = ",".join(variable)
    return out


def _command_ranges(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Extract command-line charge, m/z, and scan-window fields."""
    out: DiannParameterData = {}
    integer_fields = (
        ("min-pr-charge", "min_precursor_charge"),
        ("max-pr-charge", "max_precursor_charge"),
        ("window", "scan_window"),
    )
    for command_name, field_name in integer_fields:
        values = _arguments(cmd_dict, command_name)
        if values is not None:
            out[field_name] = int(values[0])
    text_fields = (
        ("min-fr-mz", "min_fragment_mz"),
        ("max-fr-mz", "max_fragment_mz"),
        ("min-pr-mz", "min_precursor_mz"),
        ("max-pr-mz", "max_precursor_mz"),
    )
    for command_name, field_name in text_fields:
        values = _arguments(cmd_dict, command_name)
        if values is not None:
            out[field_name] = "".join(values)
    return out


def _from_cmdline(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Settings derived from the ``diann --...`` command line.

    Tolerance fields land here as raw numeric values; they are normalized to
    typed :class:`MassTolerance` once in :func:`extract_params`.
    """
    out: DiannParameterData = {
        "quantification_method": _quantification_strategy(cmd_dict),
        "protein_inference": _protein_inference(cmd_dict),
        "predictors_library": _predictors_library(cmd_dict),
        "abundance_normalization_ions": (
            "None" if "no-norm" in cmd_dict else "Cross-run normalization"
        ),
    }
    out.update(_command_identification(cmd_dict))
    out.update(_command_tolerances(cmd_dict))
    out.update(_command_digest(cmd_dict))
    out.update(_command_modifications(cmd_dict))
    out.update(_command_ranges(cmd_dict))
    return out


def _from_log_regex(
    lines: list[str],
    *,
    has_fragment_tolerance: bool,
    has_precursor_tolerance: bool,
) -> DiannParameterData:
    """In-log regex fallbacks: tolerances gap-fill the command line, scan window overrides it."""
    out: DiannParameterData = {}
    if not has_fragment_tolerance:
        out["fragment_mass_tolerance"] = _extract_with_regex(lines, _FRAGMENT_TOL)
    if not has_precursor_tolerance:
        out["precursor_mass_tolerance"] = _extract_with_regex(lines, _PRECURSOR_TOL)
    scan_window = _extract_with_regex(lines, _SCAN_WINDOW)
    out["scan_window"] = int(scan_window) if scan_window is not None else None
    return out


def _from_cfg(lines: list[str]) -> DiannParameterData:
    """Settings re-read from the ``--cfg`` free-text block when a config file was used."""
    cleavage = _extract_cfg_text(lines, _CLEAVAGE)
    cleavage_exclusion = _extract_cfg_text(lines, _CLEAVAGE_EXC)
    cleavage_text = "" if cleavage is None else cleavage
    exclusion_text = "" if cleavage_exclusion is None else cleavage_exclusion
    out: DiannParameterData = {
        "ident_fdr_psm": _extract_cfg_float(lines, _FDR),
        "ident_fdr_protein": None,
        "enable_match_between_runs": bool(re.search(_MBR_FLAG, "".join(lines))),
        "enzyme": _normalize_enzyme(f"{cleavage_text},!{exclusion_text.strip('*')}"),
        "allowed_miscleavages": _extract_cfg_int(lines, _MISSED_CLEAVAGES),
        "min_peptide_length": _extract_cfg_int(lines, _MIN_PEP_LEN),
        "max_peptide_length": _extract_cfg_int(lines, _MAX_PEP_LEN),
        "min_precursor_charge": _extract_cfg_int(lines, _MIN_Z),
        "max_precursor_charge": _extract_cfg_int(lines, _MAX_Z),
        "max_mods": _extract_cfg_int_or_default(lines, _MAX_MODS, 0),
        "quantification_method": _extract_last_cfg_text_or_default(
            lines, _QUANT_MODE, "QuantUMS high-precision"
        ),
        "fixed_mods": _extract_modifications(lines, [_FIXED_MODS_1, _FIXED_MODS_2]),
        "variable_mods": _extract_modifications(lines, [_VAR_MODS]),
        "min_fragment_mz": _extract_cfg_int(lines, _MIN_MZ_FRAG),
        "max_fragment_mz": _extract_cfg_int(lines, _MAX_MZ_FRAG),
        "min_precursor_mz": _extract_cfg_int(lines, _MIN_MZ_PREC),
        "max_precursor_mz": _extract_cfg_int(lines, _MAX_MZ_PREC),
    }
    if re.search(_NORMALISATION_DISABLED, "".join(lines)):
        out["abundance_normalization_ions"] = "None"
    inference = _extract_cfg_text(lines, _PROTEIN_INFERENCE)
    out["protein_inference"] = (
        "Genes" if inference is None else _PROT_INF_MAP.get(inference, "Genes")
    )
    return out


def extract_params(source: _Source) -> Parameters:
    """Parse a DIA-NN log file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.diann.extract_params``. Walks the log,
    finds the ``diann --...`` command line, applies command-line settings
    into precise fields, falls back to in-log regex extraction for
    fragment/precursor tolerances and the scan window, and finally
    re-reads from the ``--cfg`` free-text block when a config file was
    used. Stages merge with explicit precedence:
    defaults < command line < log regex (gap-fill) < cfg block.
    """
    lines = read_lines(source)
    software_version = _extract_with_regex(lines, _SOFTWARE_VERSION)
    cmdline = _find_cmdline(lines)
    if software_version is None and cmdline is None:
        # Neither a DIA-NN version banner nor a `diann --...` command line: this is not a DIA-NN
        # parameter file (e.g. a FragPipe `.workflow` mis-attached to a DIA-NN submission). Reject
        # cleanly instead of parsing unrelated content.
        raise ParamsError(
            "not a DIA-NN parameter file: no 'diann --' command line or DIA-NN version string found"
        )
    cfg_used = cmdline is not None and "--cfg" in cmdline
    version_evidence: DiannVersionEvidence = (
        MISSING_DIANN_VERSION
        if software_version is None
        else _parse_diann_version(software_version)
    )
    cmd_dict = (
        _missing_cmdline_settings()
        if cmdline is None
        else _parse_cmdline(cmdline, version_evidence)
    )

    out = _defaults()
    out["software_version"] = software_version
    out["search_engine_version"] = software_version
    out["acquisition_method"] = _acquisition_method(lines, cmd_dict)
    out.update(_from_cmdline(cmd_dict))
    out.update(
        _from_log_regex(
            lines,
            has_fragment_tolerance="fragment_mass_tolerance" in out,
            has_precursor_tolerance="precursor_mass_tolerance" in out,
        )
    )
    if cfg_used:
        out.update(_from_cfg(lines))

    # Normalize tolerances to typed MassTolerance once. DIA-NN tolerances are
    # always a symmetric ppm half-width; the value comes from either the command
    # line (numeric) or the in-log regex (string), so coerce via float().
    for key in ("fragment_mass_tolerance", "precursor_mass_tolerance"):
        value = out.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str | int | float):
            raise TypeError(f"DIA-NN {key} must be numeric")
        out[key] = MassTolerance(mode="absolute", value=float(value), unit="ppm")

    # Map modification strings to ProForma-like notation.
    for mod_key in ("fixed_mods", "variable_mods"):
        raw = out.get(mod_key)
        if not isinstance(raw, str) or not raw:
            continue
        mapped = [MODIFICATION_MAPPING.get(mod.strip(), mod.strip()) for mod in raw.split(",")]
        out[mod_key] = ", ".join(mapped)

    return Parameters.model_validate(out)
