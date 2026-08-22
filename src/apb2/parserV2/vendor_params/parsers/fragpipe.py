"""FragPipe ``fragpipe.workflow`` parameter-file parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, NamedTuple

from apb2.parserV2.vendor_params.parsers.shared import unimod as unimod_registry
from apb2.parserV2.vendor_params.parsers.shared.common import (
    MassModificationMatch,
    lookup_mass_mod,
    modifications,
    read_text,
    symmetric_tolerance,
    tolerance_unit,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)


class Parameter(NamedTuple):
    """One parsed FragPipe workflow entry."""

    name: str
    value: str | None
    comment: str | None


@dataclass(frozen=True, slots=True)
class ParsedWorkflow:
    """Typed values extracted from one FragPipe workflow document.

    A component version is ``None`` when the workflow does not declare it. That absence is
    what happened while reading the file, not a variant with its own behaviour: every reader
    asks the same question and reports "not stated", so there is nothing for a separate type
    to implement.
    """

    header: str
    msfragger_version: str | None
    fragpipe_version: str | None
    diann_version: str | None
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
class IdentificationFdrs:
    """The FDR thresholds a workflow's identification stage applied.

    ``peptide`` is ``None`` for a DIA-NN identification stage, which reports PSM and protein
    FDR only. That is an absent value rather than a second kind of stage: nothing downstream
    behaves differently, it just has one fewer threshold to record.
    """

    psm: Probability
    peptide: Probability | None
    protein: Probability


@dataclass(frozen=True, slots=True)
class Quantification:
    """The quantification stage a workflow declared, if it declared one.

    ``method`` is named only by DIA-NN; IonQuant's label-free stage reports match-between-runs
    alone. A workflow with no quantification stage yields ``None`` instead of an instance.
    """

    match_between_runs: bool
    method: str | None


@dataclass(frozen=True, slots=True)
class ChargeRange:
    """The precursor-charge bounds MSFragger searched.

    ``maximum`` is ``None`` unless the workflow overrides the charge range, in which case
    MSFragger's own default upper bound applies and the file does not state it.
    """

    minimum: int
    maximum: int | None


def _lookup_mod_name(mass: float, source_token: str) -> str:
    """Resolve a display name, preserving the source mass when it is unknown."""
    canonical = unimod_registry.find_by_mass(mass, tolerance=_MASS_TOLERANCE)
    if isinstance(canonical, unimod_registry.UnimodMatch):
        return canonical.entry.name
    vendor = lookup_mass_mod(mass, _VENDOR_MASS_TO_MOD, tol=_MASS_TOLERANCE)
    if isinstance(vendor, MassModificationMatch):
        return vendor.name
    return source_token.strip()


def _parse_fixed_mods(raw: str) -> list[SearchedModification]:
    """Parse MSFragger fixed modifications string into ProForma-like format.

    Input format: ``mass,residue_description,active,num_sites`` entries separated by ``; ``.
    Example: ``57.02146,C (cysteine),true,-1``
    """
    results: list[str] = []
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
    return modifications(results, ModType.fixed)


def _parse_variable_mods(raw: str) -> list[SearchedModification]:
    """Parse MSFragger variable modifications string into ProForma-like format.

    Input format: ``mass,residue,active,max_occurrences`` entries separated by ``; ``.
    Special residue notations: ``[^`` = protein N-term, ``nX`` = peptide N-term of residue X.
    """
    results: list[str] = []
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
    return modifications(results, ModType.variable)


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
    msfragger_version: str | None = None
    fragpipe_version: str | None = None
    diann_version: str | None = None
    for line in lines[1:]:
        if line.startswith("# MSFragger version"):
            msfragger_version = line.split(" ")[-1].strip()
        elif line.startswith("fragpipe-config.bin-msfragger"):
            path = line.split("=")[-1].strip()
            filename = path.replace("\\", "/").rsplit("/", 1)[-1]
            match = re.search(_VERSION_NO_PATTERN, filename)
            if match:
                msfragger_version = match.group(1)
        if line.startswith("# FragPipe version"):
            fragpipe_version = line.split(" ")[-1].strip()
        elif line.startswith("# DIA-NN version"):
            diann_version = line.removeprefix("# DIA-NN version").strip()
        elif line.startswith(("fragpipe-config.bin-diann", "fragpipe.config.bin-diann")):
            executable = re.sub(
                r"/+",
                "/",
                line.split("=", maxsplit=1)[-1].replace("\\", "/"),
            )
            match = _DIANN_PATH_VERSION.search(executable)
            if match:
                diann_version = match.group(1).replace("_", " ")
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
) -> tuple[MassTolerance, MassTolerance]:
    """Return ``(precursor_tolerance, fragment_tolerance)``.

    MSFragger states the precursor range as independent lower and upper bounds, which
    ``symmetric_tolerance`` reduces to the half-width the schema stores — rejecting a genuinely
    asymmetric range rather than silently keeping one side.
    """
    precursor = symmetric_tolerance(
        float(precursor_mass_lower),
        float(precursor_mass_upper),
        "ppm" if int(precursor_mass_units) else "Da",
    )
    fragment = MassTolerance(
        mode="absolute",
        value=float(fragment_mass_tolerance),
        unit=tolerance_unit("ppm" if int(fragment_mass_units) else "Da"),
    )
    return precursor, fragment


def _workflow_values(records: list[Parameter]) -> dict[str, str]:
    """Narrow parsed workflow records to named text values once."""
    return {record.name: record.value for record in records if record.value is not None}


def _legacy_fragpipe_version(header: str) -> str | None:
    """Read a FragPipe version from a legacy workflow header."""
    match = re.match(r"FragPipe \((\d+\.\d+.*)\)", header)
    return match.group(1) if match else None


def _workflow_identification(values: dict[str, str], uses_diann: bool) -> IdentificationFdrs:
    """Read the FDR thresholds of the workflow's active identification stage."""
    if uses_diann:
        q_value = Probability(value=float(values["diann.q-value"]))
        return IdentificationFdrs(psm=q_value, peptide=None, protein=q_value)
    psm, peptide, protein = _parse_phi_report_filters(values["phi-report.filter"])
    return IdentificationFdrs(
        psm=Probability(value=psm),
        peptide=Probability(value=peptide),
        protein=Probability(value=protein),
    )


def _workflow_quantification(values: dict[str, str], uses_diann: bool) -> Quantification | None:
    """Read the workflow's quantification stage, or ``None`` when it declares none."""
    if values["quantitation.run-label-free-quant"] == "true":
        return Quantification(match_between_runs=bool(int(values["ionquant.mbr"])), method=None)
    if not uses_diann:
        return None
    options = values.get("diann.fragpipe.cmd-opts", values.get("diann.cmd-opts", ""))
    return Quantification(
        match_between_runs="--reanalyse" in options,
        method=_DIANN_QUANT[int(values["diann.quantification-strategy"])],
    )


def _workflow_charge_range(values: dict[str, str]) -> ChargeRange:
    """Read the precursor-charge bounds, which the workflow states only when overridden."""
    if values["msfragger.override_charge"] != "true":
        return ChargeRange(minimum=1, maximum=None)
    return ChargeRange(
        minimum=int(values["msfragger.misc.fragger.precursor-charge-lo"]),
        maximum=int(values["msfragger.misc.fragger.precursor-charge-hi"]),
    )


def extract_params(source: Path | IO[bytes] | IO[str] | BytesIO) -> Parameters:
    """Parse a FragPipe ``.workflow`` file into :class:`Parameters`.

    Mirrors ``proteobench.io.params.fragger.extract_params``. Narrows workflow
    entries to strings once, then passes exact values to independent derivations.
    """
    content = read_text(source)
    workflow = _read_workflow(content)
    values = _workflow_values(workflow.records)
    fragpipe_version = workflow.fragpipe_version or _legacy_fragpipe_version(workflow.header)

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
    digest_mass_lo = float(values["msfragger.misc.fragger.digest-mass-lo"])
    digest_mass_hi = float(values["msfragger.misc.fragger.digest-mass-hi"])
    protein_inference = (
        f"ProteinProphet: {values['protein-prophet.cmd-opts']}"
        if values["protein-prophet.run-protein-prophet"] == "true"
        else None
    )
    return Parameters(
        software_name="FragPipe",
        software_version=fragpipe_version,
        quantification_software="DIA-NN" if uses_diann else None,
        quantification_software_version=workflow.diann_version if uses_diann else None,
        search_engine="MSFragger",
        search_engine_version=workflow.msfragger_version,
        enzyme=_resolve_enzyme(
            values["msfragger.search_enzyme_name_1"],
            values["msfragger.search_enzyme_name_2"],
        ),
        allowed_miscleavages=int(values["msfragger.allowed_missed_cleavage_1"]),
        semi_enzymatic=values["msfragger.num_enzyme_termini"] != "2",
        fixed_mods=_parse_fixed_mods(values["msfragger.table.fix-mods"]),
        variable_mods=_parse_variable_mods(values["msfragger.table.var-mods"]),
        max_mods=int(values["msfragger.max_variable_mods_per_peptide"]),
        min_peptide_length=int(values["msfragger.digest_min_length"]),
        max_peptide_length=int(values["msfragger.digest_max_length"]),
        precursor_mass_tolerance=precursor_tol,
        fragment_mass_tolerance=fragment_tol,
        ident_fdr_psm=fdrs.psm,
        ident_fdr_peptide=fdrs.peptide,
        ident_fdr_protein=fdrs.protein,
        protein_inference=protein_inference,
        enable_match_between_runs=(
            None if quantification is None else quantification.match_between_runs
        ),
        quantification_method=None if quantification is None else quantification.method,
        min_precursor_charge=charge_range.minimum,
        max_precursor_charge=charge_range.maximum,
        # The lightest peptide at the highest charge, which needs an upper charge bound.
        min_precursor_mz=(
            None if charge_range.maximum is None else digest_mass_lo / charge_range.maximum
        ),
        # The heaviest peptide at the lowest charge, which is always known.
        max_precursor_mz=digest_mass_hi / charge_range.minimum,
    )
