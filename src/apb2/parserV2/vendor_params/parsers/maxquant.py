"""MaxQuant ``mqpar.xml`` parameter-file parser."""

from __future__ import annotations

import collections.abc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.common import (
    homogenize_paren_mods,
    modifications,
    split_modifications,
)
from apb2.parserV2.vendor_params.parsers.shared.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)

XmlValue = str | dict[str, "XmlValue"] | list["XmlValue"] | None
FlatValue = str | None
KeyPath = tuple[str, ...]
FlatDocument = dict[KeyPath, list[FlatValue]]

# Fallback mapping for modifications without parenthesized residue specifiers.
_MODIFICATION_MAPPING = {
    "Cys-Cys": "C[Disulfide]",
    "Cysteinyl": "C[Cysteinyl]",
    "Cysteinyl - carbamidomethyl": "C[Cysteinyl + Carbamidomethyl]",
}


def _homogenize_mods(raw_mods: str, mod_type: ModType) -> list[SearchedModification]:
    """Resolve a comma-delimited ``{name} ({residues})`` string into modifications."""
    return modifications(
        (
            token
            for mod in split_modifications(raw_mods)
            for token in homogenize_paren_mods(mod, _MODIFICATION_MAPPING)
        ),
        mod_type,
    )


def _add_record(data: dict[str, XmlValue], tag: str, record: XmlValue) -> dict[str, XmlValue]:
    if tag in data:
        existing = data[tag]
        if isinstance(existing, list):
            existing.append(record)
        else:
            data[tag] = [existing, record]
    else:
        data[tag] = record
    return data


def _read_element(element: ET.Element) -> XmlValue:
    data: dict[str, XmlValue] = {}
    if element.attrib:
        data.update(element.attrib)
    for child in element:
        if len(child) > 1 and child.tag:
            # Each list item wraps grandchild as {grandchild.tag: parsed-value}.
            data[child.tag] = [
                _add_record(
                    {},
                    tag=grand.tag,
                    record=(
                        grand.text.strip()
                        if (grand.text and grand.text.strip())
                        else _read_element(grand)
                    ),
                )
                for grand in child
            ]
        elif child.text and child.text.strip():
            _add_record(data, child.tag, child.text.strip())
        else:
            _add_record(data, child.tag, _read_element(child))
    return data or None


def _read_xml(source: Path | IO[bytes] | IO[str]) -> dict[str, XmlValue]:
    tree = ET.parse(source)
    parsed = _read_element(tree.getroot())
    if not isinstance(parsed, dict):
        raise ValueError("mqpar root did not parse to a mapping")
    return parsed


def _flatten(d: dict[str, XmlValue], parent_key: KeyPath = ()) -> list[tuple[KeyPath, FlatValue]]:
    items: list[tuple[KeyPath, FlatValue]] = []
    for key, value in d.items():
        new_key = (*parent_key, key)
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(_flatten(value, parent_key=new_key))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, collections.abc.MutableMapping):
                    items.extend(_flatten(item, parent_key=new_key))
                elif isinstance(item, str) or item is None:
                    items.append((new_key, item))
        else:
            items.append((new_key, value))
    return items


def _flat_document(record: dict[str, XmlValue]) -> FlatDocument:
    """Group every flattened value under its element key path.

    mqpar repeats element names: one ``<string>`` per raw file, per experiment, and per
    modification. The mapped value is a list so a repeated path keeps every entry in
    document order instead of the last one winning.
    """
    document: FlatDocument = {}
    for key, value in _flatten(record):
        document.setdefault(key, []).append(value)
    return document


def _values_under(document: FlatDocument, *path: str) -> list[FlatValue]:
    """Return every value whose key path starts with ``path``, in document order.

    The prefix is matched element by element, so ``("minPeptideLength",)`` selects the field
    of that name and never ``minPeptideLengthForUnspecificSearch``. Matching a prefix rather
    than a complete path is what reads one field whether MaxQuant wrote entries
    (``variableModifications/string``) or wrote the element empty (``variableModifications``).
    """
    return [
        value for key, values in document.items() if key[: len(path)] == path for value in values
    ]


def _text(value: FlatValue, field: str) -> str:
    """Return one text value; an empty mqpar element carries none."""
    if value is None:
        raise TypeError(f"MaxQuant {field} must contain one text value")
    return value


def _single_text(values: list[FlatValue], field: str) -> str:
    """Return the one text value a scalar mqpar field must declare exactly once."""
    if not values:
        raise KeyError(f"MaxQuant parameters contain no {field} field")
    if len(values) != 1:
        raise TypeError(f"MaxQuant {field} must contain one text value")
    return _text(values[0], field)


def _joined_text(values: list[FlatValue], field: str) -> str:
    """Return zero or more declared entries as a comma-delimited string.

    An empty mqpar element (``<field />`` or whitespace-only) flattens to a single ``None``:
    the search declared no entries, which is valid input rather than a malformed field, so it
    yields the empty string. A wholly absent element is a different fact and stays an error.
    """
    if not values:
        raise KeyError(f"MaxQuant parameters contain no {field} field")
    if len(values) == 1 and values[0] is None:
        return ""
    return ",".join(_text(value, field) for value in values)


def _field(document: FlatDocument, name: str) -> str:
    """Read one top-level mqpar value as text."""
    return _single_text(_values_under(document, name), name)


def _group_field(document: FlatDocument, *path: str) -> str:
    """Read one value from mqpar's single selected parameter group as text."""
    return _single_text(
        _values_under(document, "parameterGroups", "parameterGroup", *path),
        path[0],
    )


def _msms_field(document: FlatDocument, name: str) -> str:
    """Read one value from the MS2 fragmentation entry selected by ``ms2frac`` as text."""
    return _single_text(_values_under(document, "msmsParamsArray", "msmsParams", name), name)


def _tolerance_pair(document: FlatDocument) -> tuple[MassTolerance, MassTolerance]:
    """Build precursor (ppm) and fragment (ppm/Da) tolerances from the flattened mqpar."""
    precursor = MassTolerance(
        mode="absolute",
        value=float(_group_field(document, "mainSearchTol")),
        unit="ppm",
    )
    frag_value = float(_msms_field(document, "MatchTolerance"))
    in_ppm = bool(_msms_field(document, "MatchToleranceInPpm"))
    fragment = MassTolerance(mode="absolute", value=frag_value, unit="ppm" if in_ppm else "Da")
    return precursor, fragment


def _min_peptide_length(document: FlatDocument) -> int:
    """Read the minimum peptide length, tolerating the pre/post-rename key."""
    for field in ("minPepLen", "minPeptideLength"):
        values = _values_under(document, field)
        if values:
            return int(_single_text(values, field))
    raise KeyError("MaxQuant parameters contain no minimum peptide length field")


def _mods_for_version(
    document: FlatDocument,
    version: str,
) -> tuple[list[SearchedModification], list[SearchedModification]]:
    """Resolve fixed/variable modifications, handling the 1.6.0.0 path change."""
    fixed_path: KeyPath = (
        ("parameterGroups", "parameterGroup", "fixedModifications")
        if version > "1.6.0.0"
        else ("fixedModifications",)
    )
    fixed_mods = _joined_text(_values_under(document, *fixed_path), "fixedModifications")

    variable_mods = _joined_text(
        _values_under(document, "parameterGroups", "parameterGroup", "variableModifications"),
        "variableModifications",
    )

    return (
        _homogenize_mods(fixed_mods, ModType.fixed),
        _homogenize_mods(variable_mods, ModType.variable),
    )


def extract_params(
    source: Path | IO[bytes] | IO[str],
    ms2frac: str = "FTMS",
) -> Parameters:
    """Parse a MaxQuant ``mqpar.xml`` into :class:`Parameters`.

    Mirrors ``proteobench.io.params.maxquant.extract_params``: MS2
    fragmentation method must be selected explicitly (``"FTMS"`` by
    default) because mqpar.xml carries one entry per fragmentation
    method.
    """
    record = _read_xml(source)
    msms_params_array = record.get("msmsParamsArray")
    if not isinstance(msms_params_array, list):
        raise ValueError("mqpar msmsParamsArray must be a list")
    selected_params: list[XmlValue] = []
    for entry in msms_params_array:
        if not isinstance(entry, dict):
            raise ValueError("mqpar msmsParamsArray entries must be mappings")
        params = entry.get("msmsParams")
        if not isinstance(params, dict):
            raise ValueError("mqpar msmsParams entry must be a mapping")
        if params.get("Name") == ms2frac:
            selected_params.append(entry)
    record["msmsParamsArray"] = selected_params
    document = _flat_document(record)

    version = _field(document, "maxQuantVersion")
    precursor_tolerance, fragment_tolerance = _tolerance_pair(document)
    enzyme_mode = int(_group_field(document, "enzymeMode"))
    fixed_mods, variable_mods = _mods_for_version(document, version)

    return Parameters(
        software_name="MaxQuant",
        software_version=version,
        search_engine="Andromeda",
        ident_fdr_psm=Probability(value=float(_field(document, "peptideFdr"))),
        ident_fdr_protein=Probability(value=float(_field(document, "proteinFdr"))),
        enable_match_between_runs=_field(document, "matchBetweenRuns").lower() == "true",
        precursor_mass_tolerance=precursor_tolerance,
        fragment_mass_tolerance=fragment_tolerance,
        enzyme=_group_field(document, "enzymes", "string"),
        semi_enzymatic=enzyme_mode != 0,
        allowed_miscleavages=int(_group_field(document, "maxMissedCleavages")),
        min_peptide_length=_min_peptide_length(document),
        fixed_mods=fixed_mods,
        variable_mods=variable_mods,
        max_mods=int(_group_field(document, "maxNmods")),
        max_precursor_charge=int(_group_field(document, "maxCharge")),
    )
