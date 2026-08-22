"""MaxQuant ``mqpar.xml`` parameter-file parser."""

from __future__ import annotations

import collections.abc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

import pandas as pd

from apb2.parserV2.search_parameters.common import (
    homogenize_paren_mods,
    modifications,
    split_modifications,
)
from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)

XmlValue = str | dict[str, "XmlValue"] | list["XmlValue"] | None
FlatValue = str | None
KeyPath = tuple[str | None, ...]

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


def _build_series(record: dict[str, XmlValue], index_length: int = 4) -> pd.Series:
    """Index the flattened document by a fixed-width key path, padding shorter paths."""
    items = _flatten(record)
    if any(len(key) > index_length for key, _ in items):
        raise ValueError(f"mqpar nests deeper than the index width {index_length}")
    keys = [key + (None,) * (index_length - len(key)) for key, _ in items]
    return pd.Series([value for _, value in items], index=pd.MultiIndex.from_tuples(keys))


def _text(value: object, field: str) -> str:
    """Return one scalar text value read from the flattened XML series."""
    if not isinstance(value, str):
        raise TypeError(f"MaxQuant {field} must contain one text value")
    return value


def _joined_text(value: object, field: str) -> str:
    """Return one or more text values as a comma-delimited string."""
    if isinstance(value, str):
        return value
    if not isinstance(value, pd.Series):
        raise TypeError(f"MaxQuant {field} must contain text values")
    return ",".join(_text(item, field) for item in value)


def _field(series: pd.Series, name: str) -> str:
    """Read one top-level mqpar value as text."""
    return _text(series.loc[name].squeeze(), name)


def _group_field(series: pd.Series, *path: str) -> str:
    """Read one value from mqpar's single selected parameter group as text."""
    return _text(
        series.loc[pd.IndexSlice[("parameterGroups", "parameterGroup", *path)]].squeeze(),
        path[0],
    )


def _tolerance_pair(series: pd.Series) -> tuple[MassTolerance, MassTolerance]:
    """Build precursor (ppm) and fragment (ppm/Da) tolerances from the mqpar series."""
    precursor = MassTolerance(
        mode="absolute",
        value=float(_group_field(series, "mainSearchTol")),
        unit="ppm",
    )
    frag_value = float(
        _text(
            series.loc[
                pd.IndexSlice["msmsParamsArray", "msmsParams", "MatchTolerance", :]
            ].squeeze(),
            "MatchTolerance",
        )
    )
    in_ppm = bool(
        series.loc[
            pd.IndexSlice["msmsParamsArray", "msmsParams", "MatchToleranceInPpm", :]
        ].squeeze()
    )
    fragment = MassTolerance(mode="absolute", value=frag_value, unit="ppm" if in_ppm else "Da")
    return precursor, fragment


def _min_peptide_length(series: pd.Series) -> int:
    """Read the minimum peptide length, tolerating the pre/post-rename key."""
    keys = set(series.index.get_level_values(0))
    for field in ("minPepLen", "minPeptideLength"):
        if field in keys:
            return int(_field(series, field))
    raise KeyError("MaxQuant parameters contain no minimum peptide length field")


def _mods_for_version(
    series: pd.Series,
    version: str,
) -> tuple[list[SearchedModification], list[SearchedModification]]:
    """Resolve fixed/variable modifications, handling the 1.6.0.0 path change."""
    fixed_path = (
        pd.IndexSlice["parameterGroups", "parameterGroup", "fixedModifications", :]
        if version > "1.6.0.0"
        else pd.IndexSlice["fixedModifications", :]
    )
    fixed_mods = _joined_text(series.loc[fixed_path].squeeze(), "fixedModifications")

    variable_mods = _joined_text(
        series.loc[
            pd.IndexSlice["parameterGroups", "parameterGroup", "variableModifications", :]
        ].squeeze(),
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
    series = _build_series(record, 4).sort_index()

    version = str(series.loc["maxQuantVersion"].squeeze())
    precursor_tolerance, fragment_tolerance = _tolerance_pair(series)
    enzyme_mode = int(_group_field(series, "enzymeMode"))
    fixed_mods, variable_mods = _mods_for_version(series, version)

    return Parameters(
        software_name="MaxQuant",
        software_version=version,
        search_engine="Andromeda",
        ident_fdr_psm=Probability(value=float(_field(series, "peptideFdr"))),
        ident_fdr_protein=Probability(value=float(_field(series, "proteinFdr"))),
        enable_match_between_runs=_field(series, "matchBetweenRuns").lower() == "true",
        precursor_mass_tolerance=precursor_tolerance,
        fragment_mass_tolerance=fragment_tolerance,
        enzyme=_group_field(series, "enzymes", "string"),
        semi_enzymatic=enzyme_mode != 0,
        allowed_miscleavages=int(_group_field(series, "maxMissedCleavages")),
        min_peptide_length=_min_peptide_length(series),
        fixed_mods=fixed_mods,
        variable_mods=variable_mods,
        max_mods=int(_group_field(series, "maxNmods")),
        max_precursor_charge=int(_group_field(series, "maxCharge")),
    )
