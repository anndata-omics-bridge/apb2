"""DIA-NN log/cfg parameter-file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TypedDict

from packaging.version import VERSION_PATTERN, Version

from apb2.vendor_params.model import (
    AcquisitionMethod,
    MassTolerance,
    ModType,
    Parameters,
    ParamsError,
    Probability,
    SearchedModification,
)
from apb2.vendor_params.parsers._common import mapped_modifications, read_lines

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

# DIA-NN command-line options that carry one integer, and the field each one sets.
_COUNT_OPTIONS = (
    ("var-mods", "max_mods"),
    ("missed-cleavages", "allowed_miscleavages"),
    ("min-pep-len", "min_peptide_length"),
    ("max-pep-len", "max_peptide_length"),
    ("min-pr-charge", "min_precursor_charge"),
    ("max-pr-charge", "max_precursor_charge"),
    ("window", "scan_window"),
)
_TOLERANCE_OPTIONS = (
    ("mass-acc-ms1", "precursor_mass_tolerance"),
    ("mass-acc", "fragment_mass_tolerance"),
)
_MZ_OPTIONS = (
    ("min-fr-mz", "min_fragment_mz"),
    ("max-fr-mz", "max_fragment_mz"),
    ("min-pr-mz", "min_precursor_mz"),
    ("max-pr-mz", "max_precursor_mz"),
)
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


class DiannParameterData(TypedDict, total=False):
    """Fields assembled from DIA-NN's stages, in the schema's own types.

    DIA-NN is the one parser that merges four evidence sources by precedence — defaults <
    command line < in-log regex < ``--cfg`` block — so a stage's partial record is a real
    intermediate value, not a second declaration of the schema. Each stage converts text
    where it reads it, so every field here holds what :class:`Parameters` declares.
    """

    software_name: str
    software_version: str | None
    acquisition_method: AcquisitionMethod
    search_engine: str
    search_engine_version: str | None
    ident_fdr_psm: Probability | None
    ident_fdr_protein: Probability | None
    enable_match_between_runs: bool
    precursor_mass_tolerance: MassTolerance | None
    fragment_mass_tolerance: MassTolerance | None
    enzyme: str
    allowed_miscleavages: int | None
    min_peptide_length: int | None
    max_peptide_length: int | None
    fixed_mods: list[SearchedModification]
    variable_mods: list[SearchedModification]
    max_mods: int | None
    min_precursor_charge: int | None
    max_precursor_charge: int | None
    min_precursor_mz: float | None
    max_precursor_mz: float | None
    min_fragment_mz: float | None
    max_fragment_mz: float | None
    quantification_method: str
    protein_inference: str
    abundance_normalization_ions: str
    predictors_library: str | None
    scan_window: int | None


# DIA-NN built-in defaults, reported when the log/cfg omits the corresponding
# setting. These mirror DIA-NN's own built-in defaults and are version-sensitive:
# re-verify against DIA-NN release notes when bumping supported versions.
_DIANN_IMPLICIT_DEFAULTS: DiannParameterData = {
    "min_precursor_charge": 1,
    "max_precursor_charge": 4,
    "min_peptide_length": 7,
    "max_peptide_length": 30,
    "min_fragment_mz": 200,
    "max_fragment_mz": 1800,
    "min_precursor_mz": 300,
    "max_precursor_mz": 1800,
}


def _ppm(value: str) -> MassTolerance:
    """DIA-NN states both search tolerances as a symmetric ppm half-width."""
    return MassTolerance(mode="absolute", value=float(value), unit="ppm")


def _probability(value: str | None) -> Probability | None:
    """An FDR the cfg block does not state is absent."""
    return None if value is None else Probability(value=float(value))


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


def _cfg_int(lines: list[str], regex: str) -> int | None:
    """Read an optional integer from DIA-NN's free-text cfg block."""
    raw = _extract_with_regex(lines, regex)
    return None if raw is None else int(raw)


def _extract_modifications(
    lines: list[str],
    regexes: list[str],
    mod_type: ModType,
) -> list[SearchedModification]:
    """Read the modifications DIA-NN announces in its cfg block."""
    joined = "\n".join(lines)
    mods = [
        match.group(1).rstrip("\n") for regex in regexes for match in re.finditer(regex, joined)
    ]
    return mapped_modifications(mods, MODIFICATION_MAPPING, mod_type)


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


def _from_cmdline(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Settings derived from the ``diann --...`` command line.

    An option the command line does not carry is left out of the record entirely, so the
    stage below it in the precedence order keeps whatever it decided.
    """
    enzyme = _arguments(cmd_dict, "cut")
    out: DiannParameterData = {
        "quantification_method": _quantification_strategy(cmd_dict),
        "protein_inference": _protein_inference(cmd_dict),
        "predictors_library": _predictors_library(cmd_dict),
        "abundance_normalization_ions": (
            "None" if "no-norm" in cmd_dict else "Cross-run normalization"
        ),
        "enzyme": "Trypsin/P" if enzyme is None else _normalize_enzyme("".join(enzyme)),
    }
    reanalyse = _flag(cmd_dict, "reanalyse")
    if reanalyse is not None:
        out["enable_match_between_runs"] = reanalyse
    out.update(_from_cmdline_arguments(cmd_dict))
    return out


def _from_cmdline_arguments(cmd_dict: dict[str, _CommandValue]) -> DiannParameterData:
    """Settings named by a command-line option that carries arguments."""
    out: DiannParameterData = {}
    q_value = _arguments(cmd_dict, "qvalue")
    if q_value is not None:
        out["ident_fdr_psm"] = Probability(value=float(q_value[0]))
    fixed = _arguments(cmd_dict, "mod")
    if fixed is not None:
        out["fixed_mods"] = mapped_modifications(fixed, MODIFICATION_MAPPING, ModType.fixed)
    variable = _arguments(cmd_dict, "var-mod")
    if variable is not None:
        out["variable_mods"] = mapped_modifications(
            variable, MODIFICATION_MAPPING, ModType.variable
        )
    for option, tolerance_field in _TOLERANCE_OPTIONS:
        values = _arguments(cmd_dict, option)
        if values is not None:
            out[tolerance_field] = _ppm(values[0])
    for option, count_field in _COUNT_OPTIONS:
        values = _arguments(cmd_dict, option)
        if values is not None:
            out[count_field] = int(values[0])
    for option, mz_field in _MZ_OPTIONS:
        values = _arguments(cmd_dict, option)
        if values is not None:
            out[mz_field] = float("".join(values))
    return out


def _log_tolerance(lines: list[str], regex: str) -> MassTolerance | None:
    """The tolerance DIA-NN announces in its log, when the command line did not set one."""
    value = _extract_with_regex(lines, regex)
    return None if value is None else _ppm(value)


def _from_cfg(lines: list[str]) -> DiannParameterData:
    """Settings re-read from the ``--cfg`` free-text block when a config file was used."""
    cleavage = _extract_with_regex(lines, _CLEAVAGE)
    cleavage_exclusion = _extract_with_regex(lines, _CLEAVAGE_EXC)
    cleavage_text = "" if cleavage is None else cleavage
    exclusion_text = "" if cleavage_exclusion is None else cleavage_exclusion
    out: DiannParameterData = {
        "ident_fdr_psm": _probability(_extract_with_regex(lines, _FDR)),
        "ident_fdr_protein": None,
        "enable_match_between_runs": bool(re.search(_MBR_FLAG, "".join(lines))),
        "enzyme": _normalize_enzyme(f"{cleavage_text},!{exclusion_text.strip('*')}"),
        "allowed_miscleavages": _cfg_int(lines, _MISSED_CLEAVAGES),
        "min_peptide_length": _cfg_int(lines, _MIN_PEP_LEN),
        "max_peptide_length": _cfg_int(lines, _MAX_PEP_LEN),
        "min_precursor_charge": _cfg_int(lines, _MIN_Z),
        "max_precursor_charge": _cfg_int(lines, _MAX_Z),
        "max_mods": _cfg_int(lines, _MAX_MODS) or 0,
        "quantification_method": _extract_with_regex(lines, _QUANT_MODE, search_all=True)
        or "QuantUMS high-precision",
        "fixed_mods": _extract_modifications(lines, [_FIXED_MODS_1, _FIXED_MODS_2], ModType.fixed),
        "variable_mods": _extract_modifications(lines, [_VAR_MODS], ModType.variable),
        "min_fragment_mz": _cfg_int(lines, _MIN_MZ_FRAG),
        "max_fragment_mz": _cfg_int(lines, _MAX_MZ_FRAG),
        "min_precursor_mz": _cfg_int(lines, _MIN_MZ_PREC),
        "max_precursor_mz": _cfg_int(lines, _MAX_MZ_PREC),
    }
    if re.search(_NORMALISATION_DISABLED, "".join(lines)):
        out["abundance_normalization_ions"] = "None"
    inference = _extract_with_regex(lines, _PROTEIN_INFERENCE)
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
    # No logged command line means no options were given, not that they are unknown.
    cmd_dict: dict[str, _CommandValue] = (
        {"var-mod": [], "mod": []} if cmdline is None else _parse_cmdline(cmdline, version_evidence)
    )

    out = _defaults()
    out["software_version"] = software_version
    out["search_engine_version"] = software_version
    out["acquisition_method"] = _acquisition_method(lines, cmd_dict)
    out.update(_from_cmdline(cmd_dict))
    # The log's own tolerance lines gap-fill the command line; the scan window overrides it.
    if "fragment_mass_tolerance" not in out:
        out["fragment_mass_tolerance"] = _log_tolerance(lines, _FRAGMENT_TOL)
    if "precursor_mass_tolerance" not in out:
        out["precursor_mass_tolerance"] = _log_tolerance(lines, _PRECURSOR_TOL)
    scan_window = _extract_with_regex(lines, _SCAN_WINDOW)
    out["scan_window"] = int(scan_window) if scan_window is not None else None
    if cfg_used:
        out.update(_from_cfg(lines))

    return Parameters(**out)
