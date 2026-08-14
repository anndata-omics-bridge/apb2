"""FragPipe ``fragpipe.workflow`` parameter-file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, NamedTuple, TypedDict

from apb2.modifications import unimod_registry
from apb2.vendor_params.model import MassTolerance, Parameters
from apb2.vendor_params.parsers._common import (
    MassModificationMatch,
    lookup_mass_mod,
    read_text,
)


class Parameter(NamedTuple):
    """One parsed FragPipe workflow entry."""

    name: str
    value: str | None
    comment: str | None


@dataclass(frozen=True, slots=True)
class WorkflowVersion:
    """One version string declared by a FragPipe workflow."""

    value: str


@dataclass(frozen=True, slots=True)
class WorkflowVersionUnavailable:
    """A FragPipe workflow did not declare this component version."""


type WorkflowVersionEvidence = WorkflowVersion | WorkflowVersionUnavailable

VERSION_UNAVAILABLE = WorkflowVersionUnavailable()


@dataclass(frozen=True, slots=True)
class ParsedWorkflow:
    """Typed values extracted from one FragPipe workflow document."""

    header: str
    msfragger_version: WorkflowVersionEvidence
    fragpipe_version: WorkflowVersionEvidence
    diann_version: WorkflowVersionEvidence
    records: list[Parameter]


_VERSION_NO_PATTERN = r"MSFragger-(.+)\.jar"
_DIANN_PATH_VERSION = re.compile(r"/diann/([^/]+)/", re.IGNORECASE)

_DIANN_QUANT = {
    1: "Any LC (high accuracy)",
    2: "Any LC (high precision)",
    3: "Robust LC (high accuracy)",
    4: "Robust LC (high precision)",
}

# FragPipe-specific labels that are outside APB's small canonical registry.
_VENDOR_MASS_TO_MOD = {
    4.025107: "Label:2H(4)",
    6.020129: "Label:13C(6)",
    8.014199: "Label:13C(6)15N(2)",
    10.008269: "Label:13C(6)15N(4)",
}
_MASS_TOLERANCE = 0.001


@dataclass(frozen=True, slots=True)
class DiannIdentificationFdrs:
    """DIA-NN reports PSM and protein FDR without a peptide FDR."""

    psm: float
    protein: float


@dataclass(frozen=True, slots=True)
class PhiReportIdentificationFdrs:
    """Philosopher reports PSM, peptide, and protein FDRs."""

    psm: float
    peptide: float
    protein: float


@dataclass(frozen=True, slots=True)
class LabelFreeQuantification:
    """IonQuant label-free quantification settings."""

    match_between_runs: bool


@dataclass(frozen=True, slots=True)
class DiannQuantification:
    """DIA-NN quantification settings."""

    match_between_runs: bool
    method: str


@dataclass(frozen=True, slots=True)
class NoQuantification:
    """The workflow does not declare a quantification stage."""


@dataclass(frozen=True, slots=True)
class BoundedChargeRange:
    """Both precursor-charge bounds are known."""

    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class LowerBoundedChargeRange:
    """Only the default minimum precursor charge is known."""

    minimum: int


@dataclass(frozen=True, slots=True)
class DigestMassRange:
    """MSFragger digest-mass bounds."""

    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class PrecursorMzRange:
    """Both precursor m/z bounds derived from bounded charge and mass."""

    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class MaximumPrecursorMz:
    """The only m/z bound derivable from a lower-bounded charge range."""

    maximum: float


class FragPipeVariantData(TypedDict, total=False):
    """Fields whose presence depends on tagged workflow stages."""

    ident_fdr_peptide: float
    enable_match_between_runs: bool
    quantification_method: str
    max_precursor_charge: int
    min_precursor_mz: float
    max_precursor_mz: float


class FragPipeParameterData(FragPipeVariantData):
    """Precisely typed values accepted by :class:`Parameters`."""

    software_name: str
    software_version: str | None
    quantification_software: str | None
    quantification_software_version: str | None
    search_engine: str
    search_engine_version: str | None
    enzyme: str
    allowed_miscleavages: int
    semi_enzymatic: bool
    fixed_mods: str
    variable_mods: str
    max_mods: int
    min_peptide_length: int
    max_peptide_length: int
    precursor_mass_tolerance: str
    fragment_mass_tolerance: MassTolerance
    ident_fdr_psm: float
    ident_fdr_protein: float
    protein_inference: str | None
    min_precursor_charge: int


def _lookup_mod_name(mass: float, source_token: str) -> str:
    """Resolve a display name, preserving the source mass when it is unknown."""
    canonical = unimod_registry.find_by_mass(mass, tolerance=_MASS_TOLERANCE)
    if isinstance(canonical, unimod_registry.UnimodMatch):
        return canonical.entry.name
    vendor = lookup_mass_mod(mass, _VENDOR_MASS_TO_MOD, tol=_MASS_TOLERANCE)
    if isinstance(vendor, MassModificationMatch):
        return vendor.name
    return source_token.strip()


def _parse_fixed_mods(raw: str) -> str:
    """Parse MSFragger fixed modifications string into ProForma-like format.

    Input format: ``mass,residue_description,active,num_sites`` entries separated by ``; ``.
    Example: ``57.02146,C (cysteine),true,-1``
    """
    if not raw or not raw.strip():
        return ""
    results = []
    for entry in raw.split("; "):
        parts = entry.strip().split(",", 3)
        if len(parts) < 3:
            continue
        mass_str, residue_desc, active = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if active != "true":
            continue
        mass = float(mass_str)
        if abs(mass) < _MASS_TOLERANCE:
            continue
        mod_name = _lookup_mod_name(mass, mass_str)
        residue_match = re.match(r"^([A-Z])\s*\(", residue_desc)
        if residue_match:
            residue = residue_match.group(1)
        elif "N-Term" in residue_desc:
            residue = "N-term"
        elif "C-Term" in residue_desc:
            residue = "C-term"
        else:
            residue = residue_desc
        results.append(f"{residue}[{mod_name}]")
    return ", ".join(results)


def _parse_variable_mods(raw: str) -> str:
    """Parse MSFragger variable modifications string into ProForma-like format.

    Input format: ``mass,residue,active,max_occurrences`` entries separated by ``; ``.
    Special residue notations: ``[^`` = protein N-term, ``nX`` = peptide N-term of residue X.
    """
    if not raw or not raw.strip():
        return ""
    results = []
    for entry in raw.split("; "):
        parts = entry.strip().split(",", 3)
        if len(parts) < 3:
            continue
        mass_str, residue_field, active = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if active != "true":
            continue
        mass = float(mass_str)
        if abs(mass) < _MASS_TOLERANCE:
            continue
        mod_name = _lookup_mod_name(mass, mass_str)
        if residue_field == "[^":
            results.append(f"N-term[{mod_name}]")
        elif residue_field.startswith("n"):
            aa_residues = re.findall(r"n([A-Z])", residue_field)
            if aa_residues:
                for aa in aa_residues:
                    results.append(f"N-term {aa}[{mod_name}]")
            else:
                results.append(f"N-term[{mod_name}]")
        else:
            results.append(f"{residue_field}[{mod_name}]")
    return ", ".join(results)


def _parse_lines(lines: list[str], sep: str = "=") -> list[Parameter]:
    """Parse FragPipe ``key=value # comment`` style lines."""
    out: list[Parameter] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            parts = line.split("#")
            param, comment = parts[0].strip(), parts[1].strip()
        else:
            param, comment = line, None
        kv = param.split(sep, maxsplit=1)
        if len(kv) == 1:
            out.append(Parameter(kv[0].strip(), None, comment))
            continue
        out.append(Parameter(kv[0].strip(), kv[1].strip(), comment))
    return out


def _parse_phi_report_filters(cmd: str) -> tuple[float, float, float]:
    """Read PSM/peptide/protein FDR triplet from a ``phi-report.filter`` value."""
    default = 0.01
    patterns = {
        "psm": r"--psm\s+(\d+\.\d+)",
        "peptide": r"--pep\s+(\d+\.\d+)",
        "protein": r"--prot\s+(\d+\.\d+)",
    }
    psm_match = re.search(patterns["psm"], cmd)
    peptide_match = re.search(patterns["peptide"], cmd)
    protein_match = re.search(patterns["protein"], cmd)
    return (
        float(psm_match.group(1)) if psm_match else default,
        float(peptide_match.group(1)) if peptide_match else default,
        float(protein_match.group(1)) if protein_match else default,
    )


def _read_workflow(
    content: str,
) -> ParsedWorkflow:
    lines = content.splitlines()
    header = lines[0][1:].strip()  # leading '#'
    msfragger_version: WorkflowVersionEvidence = VERSION_UNAVAILABLE
    fragpipe_version: WorkflowVersionEvidence = VERSION_UNAVAILABLE
    diann_version: WorkflowVersionEvidence = VERSION_UNAVAILABLE
    for line in lines[1:]:
        if line.startswith("# MSFragger version"):
            msfragger_version = WorkflowVersion(line.split(" ")[-1].strip())
        elif line.startswith("fragpipe-config.bin-msfragger"):
            path = line.split("=")[-1].strip()
            filename = path.replace("\\", "/").rsplit("/", 1)[-1]
            match = re.search(_VERSION_NO_PATTERN, filename)
            if match:
                msfragger_version = WorkflowVersion(match.group(1))
        if line.startswith("# FragPipe version"):
            fragpipe_version = WorkflowVersion(line.split(" ")[-1].strip())
        elif line.startswith("# DIA-NN version"):
            diann_version = WorkflowVersion(line.removeprefix("# DIA-NN version").strip())
        elif line.startswith(("fragpipe-config.bin-diann", "fragpipe.config.bin-diann")):
            executable = re.sub(
                r"/+",
                "/",
                line.split("=", maxsplit=1)[-1].replace("\\", "/"),
            )
            match = _DIANN_PATH_VERSION.search(executable)
            if match:
                diann_version = WorkflowVersion(match.group(1).replace("_", " "))
    return ParsedWorkflow(
        header=header,
        msfragger_version=msfragger_version,
        fragpipe_version=fragpipe_version,
        diann_version=diann_version,
        records=_parse_lines(lines),
    )


def _resolve_enzyme(primary: str, secondary: str) -> str:
    """Combine the two MSFragger enzyme slots and canonicalize trypsin variants."""
    enzyme = primary
    if secondary != "null":
        enzyme = f"{enzyme}|{secondary}"
    if enzyme == "stricttrypsin":
        return "Trypsin/P"
    if enzyme == "trypsin":
        return "Trypsin"
    return enzyme


def _tolerances(
    precursor_mass_units: str,
    precursor_mass_lower: str,
    precursor_mass_upper: str,
    fragment_mass_units: str,
    fragment_mass_tolerance: str,
) -> tuple[str, MassTolerance]:
    """Return ``(precursor_tolerance, fragment_tolerance)``.

    The precursor range carries independent lower/upper bounds (structurally
    asymmetric), so it stays a bracketed string that ``MassTolerance.parse``
    validates; the symmetric fragment tolerance is built as a typed object.
    """
    precursor_unit = "ppm" if int(precursor_mass_units) else "Da"
    precursor_tol = (
        f"[{precursor_mass_lower} {precursor_unit}, {precursor_mass_upper} {precursor_unit}]"
    )
    fragment_unit = "ppm" if int(fragment_mass_units) else "Da"
    fragment_tol = MassTolerance(
        mode="absolute",
        value=float(fragment_mass_tolerance),
        unit=fragment_unit,
    )
    return precursor_tol, fragment_tol


def _diann_identification_fdrs(q_value: str) -> DiannIdentificationFdrs:
    """Build the two FDR values reported by a DIA-NN identification stage."""
    value = float(q_value)
    return DiannIdentificationFdrs(psm=value, protein=value)


def _phi_report_identification_fdrs(command: str) -> PhiReportIdentificationFdrs:
    """Build the FDR triplet reported by a Philosopher identification stage."""
    psm, peptide, protein = _parse_phi_report_filters(command)
    return PhiReportIdentificationFdrs(psm=psm, peptide=peptide, protein=protein)


def _label_free_quantification(mbr: str) -> LabelFreeQuantification:
    """Build IonQuant label-free quantification settings."""
    return LabelFreeQuantification(match_between_runs=bool(int(mbr)))


def _diann_quantification(command_options: str, strategy: str) -> DiannQuantification:
    """Build DIA-NN quantification settings."""
    return DiannQuantification(
        match_between_runs="--reanalyse" in command_options,
        method=_DIANN_QUANT[int(strategy)],
    )


def _diann_quantification_without_command_options(strategy: str) -> DiannQuantification:
    """Build DIA-NN quantification when the workflow declares no option field."""
    return DiannQuantification(
        match_between_runs=False,
        method=_DIANN_QUANT[int(strategy)],
    )


def _bounded_charge_range(minimum: str, maximum: str) -> BoundedChargeRange:
    """Parse both explicitly overridden precursor-charge bounds."""
    return BoundedChargeRange(minimum=int(minimum), maximum=int(maximum))


def _digest_mass_range(minimum: str, maximum: str) -> DigestMassRange:
    """Extract the concrete digest-mass interval required for m/z derivation."""
    return DigestMassRange(minimum=float(minimum), maximum=float(maximum))


def _precursor_mz_range(
    digest_mass: DigestMassRange,
    charge: BoundedChargeRange,
) -> PrecursorMzRange:
    """Derive both m/z bounds when both charge bounds are known."""
    return PrecursorMzRange(
        minimum=digest_mass.minimum / charge.maximum,
        maximum=digest_mass.maximum / charge.minimum,
    )


def _maximum_precursor_mz(
    digest_mass: DigestMassRange,
    charge: LowerBoundedChargeRange,
) -> MaximumPrecursorMz:
    """Derive only maximum m/z when the maximum charge is unknown."""
    return MaximumPrecursorMz(maximum=digest_mass.maximum / charge.minimum)


def _protein_inference(command_options: str) -> str:
    """Describe the active ProteinProphet configuration."""
    return f"ProteinProphet: {command_options}"


def _workflow_values(records: list[Parameter]) -> dict[str, str]:
    """Narrow parsed workflow records to named text values once."""
    return {record.name: record.value for record in records if record.value is not None}


def _legacy_fragpipe_version(header: str) -> WorkflowVersionEvidence:
    """Parse a FragPipe version from a legacy workflow header."""
    match = re.match(r"FragPipe \((\d+\.\d+.*)\)", header)
    return WorkflowVersion(match.group(1)) if match else VERSION_UNAVAILABLE


def _workflow_identification(
    values: dict[str, str], uses_diann: bool
) -> DiannIdentificationFdrs | PhiReportIdentificationFdrs:
    """Select the workflow's active identification stage."""
    if uses_diann:
        return _diann_identification_fdrs(values["diann.q-value"])
    return _phi_report_identification_fdrs(values["phi-report.filter"])


def _workflow_quantification(
    values: dict[str, str], uses_diann: bool
) -> LabelFreeQuantification | DiannQuantification | NoQuantification:
    """Select the workflow's active quantification stage."""
    if values["quantitation.run-label-free-quant"] == "true":
        return _label_free_quantification(values["ionquant.mbr"])
    if uses_diann:
        strategy = values["diann.quantification-strategy"]
        if "diann.fragpipe.cmd-opts" in values:
            return _diann_quantification(values["diann.fragpipe.cmd-opts"], strategy)
        if "diann.cmd-opts" in values:
            return _diann_quantification(values["diann.cmd-opts"], strategy)
        return _diann_quantification_without_command_options(strategy)
    return NoQuantification()


def _workflow_charge_range(
    values: dict[str, str],
) -> BoundedChargeRange | LowerBoundedChargeRange:
    """Select the explicit or default precursor-charge shape."""
    if values["msfragger.override_charge"] == "true":
        return _bounded_charge_range(
            values["msfragger.misc.fragger.precursor-charge-lo"],
            values["msfragger.misc.fragger.precursor-charge-hi"],
        )
    return LowerBoundedChargeRange(minimum=1)


def _add_variant_parameter_data(
    data: FragPipeParameterData,
    fdrs: DiannIdentificationFdrs | PhiReportIdentificationFdrs,
    quantification: LabelFreeQuantification | DiannQuantification | NoQuantification,
    charge_range: BoundedChargeRange | LowerBoundedChargeRange,
    digest_mass_range: DigestMassRange,
) -> None:
    """Project tagged workflow variants onto optional parameter fields."""
    data["max_precursor_mz"] = _maximum_precursor_mz(
        digest_mass_range,
        LowerBoundedChargeRange(minimum=charge_range.minimum),
    ).maximum
    if isinstance(fdrs, PhiReportIdentificationFdrs):
        data["ident_fdr_peptide"] = fdrs.peptide
    if isinstance(quantification, LabelFreeQuantification | DiannQuantification):
        data["enable_match_between_runs"] = quantification.match_between_runs
    if isinstance(quantification, DiannQuantification):
        data["quantification_method"] = quantification.method
    if isinstance(charge_range, BoundedChargeRange):
        precursor_mz = _precursor_mz_range(digest_mass_range, charge_range)
        data["max_precursor_charge"] = charge_range.maximum
        data["min_precursor_mz"] = precursor_mz.minimum
        data["max_precursor_mz"] = precursor_mz.maximum


def extract_params(source: Path | IO[bytes] | IO[str] | BytesIO) -> Parameters:
    """Parse a FragPipe ``.workflow`` file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.fragger.extract_params``. Narrows workflow
    entries to strings once, then passes exact values to independent derivations.
    """
    content = read_text(source)
    workflow = _read_workflow(content)
    values = _workflow_values(workflow.records)
    fragpipe_version = workflow.fragpipe_version
    if isinstance(fragpipe_version, WorkflowVersionUnavailable):
        fragpipe_version = _legacy_fragpipe_version(workflow.header)

    precursor_tol, fragment_tol = _tolerances(
        values["msfragger.precursor_mass_units"],
        values["msfragger.precursor_mass_lower"],
        values["msfragger.precursor_mass_upper"],
        values["msfragger.fragment_mass_units"],
        values["msfragger.fragment_mass_tolerance"],
    )
    uses_diann = values["diann.run-dia-nn"] == "true"
    fdrs = _workflow_identification(values, uses_diann)
    quantification = _workflow_quantification(values, uses_diann)
    charge_range = _workflow_charge_range(values)
    digest_mass_range = _digest_mass_range(
        values["msfragger.misc.fragger.digest-mass-lo"],
        values["msfragger.misc.fragger.digest-mass-hi"],
    )
    protein_inference = (
        _protein_inference(values["protein-prophet.cmd-opts"])
        if values["protein-prophet.run-protein-prophet"] == "true"
        else None
    )
    parameter_data: FragPipeParameterData = {
        "software_name": "FragPipe",
        "software_version": (
            fragpipe_version.value if isinstance(fragpipe_version, WorkflowVersion) else None
        ),
        "quantification_software": "DIA-NN" if uses_diann else None,
        "quantification_software_version": (
            workflow.diann_version.value
            if uses_diann and isinstance(workflow.diann_version, WorkflowVersion)
            else None
        ),
        "search_engine": "MSFragger",
        "search_engine_version": (
            workflow.msfragger_version.value
            if isinstance(workflow.msfragger_version, WorkflowVersion)
            else None
        ),
        "enzyme": _resolve_enzyme(
            values["msfragger.search_enzyme_name_1"],
            values["msfragger.search_enzyme_name_2"],
        ),
        "allowed_miscleavages": int(values["msfragger.allowed_missed_cleavage_1"]),
        "semi_enzymatic": values["msfragger.num_enzyme_termini"] != "2",
        "fixed_mods": _parse_fixed_mods(values["msfragger.table.fix-mods"]),
        "variable_mods": _parse_variable_mods(values["msfragger.table.var-mods"]),
        "max_mods": int(values["msfragger.max_variable_mods_per_peptide"]),
        "min_peptide_length": int(values["msfragger.digest_min_length"]),
        "max_peptide_length": int(values["msfragger.digest_max_length"]),
        "precursor_mass_tolerance": precursor_tol,
        "fragment_mass_tolerance": fragment_tol,
        "ident_fdr_psm": fdrs.psm,
        "ident_fdr_protein": fdrs.protein,
        "protein_inference": protein_inference,
        "min_precursor_charge": charge_range.minimum,
    }
    _add_variant_parameter_data(
        parameter_data,
        fdrs,
        quantification,
        charge_range,
        digest_mass_range,
    )

    return Parameters.model_validate(parameter_data)
