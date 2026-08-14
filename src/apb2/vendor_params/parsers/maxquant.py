"""MaxQuant ``mqpar.xml`` parameter-file parser."""

from __future__ import annotations

import collections.abc
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

import pandas as pd

from apb2.vendor_params.model import MassTolerance, Parameters
from apb2.vendor_params.parsers._common import homogenize_paren_mods

XmlValue = str | dict[str, "XmlValue"] | list["XmlValue"] | None
FlatValue = str | None
KeyPath = tuple[str | None, ...]

# Fallback mapping for modifications without parenthesized residue specifiers.
_MODIFICATION_MAPPING = {
    "Cys-Cys": "C[Disulfide]",
    "Cysteinyl": "C[Cysteinyl]",
    "Cysteinyl - carbamidomethyl": "C[Cysteinyl + Carbamidomethyl]",
}


def _homogenize_mods(raw_mods: str, sep: str = ",") -> str:
    """Parse and homogenize a separator-delimited ``{name} ({residues})`` string."""
    if not raw_mods or not raw_mods.strip():
        return ""
    return ", ".join(
        homogenize_paren_mods(mod, _MODIFICATION_MAPPING)
        for mod in raw_mods.split(sep)
        if mod.strip()
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


def _extend(t: KeyPath, target_length: int) -> KeyPath:
    if len(t) > target_length:
        raise ValueError(f"tuple too long for index width {target_length}: {t!r}")
    return t + (None,) * (target_length - len(t))


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
    items = _flatten(record)
    keys = [_extend(key, index_length) for key, _ in items]
    values = [value for _, value in items]
    idx = pd.MultiIndex.from_tuples(keys)
    return pd.Series(values, index=idx)


def _text(value: object, field: str) -> str:
    """Return one scalar text value read from the flattened XML series."""
    if not isinstance(value, str):
        raise TypeError(f"MaxQuant {field} must contain one text value")
    return value


def _joined_text(value: object, field: str) -> str:
    """Return one or more text values as a comma-delimited string."""
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Series):
        parts: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError(f"MaxQuant {field} entries must be text")
            parts.append(item)
        return ",".join(parts)
    raise TypeError(f"MaxQuant {field} must contain text values")


def _tolerance_pair(series: pd.Series) -> tuple[MassTolerance, MassTolerance]:
    """Build precursor (ppm) and fragment (ppm/Da) tolerances from the mqpar series."""
    prec_value = float(
        _text(
            series.loc[
                pd.IndexSlice["parameterGroups", "parameterGroup", "mainSearchTol", :]
            ].squeeze(),
            "mainSearchTol",
        )
    )
    precursor = MassTolerance(mode="absolute", value=prec_value, unit="ppm")
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
    if "minPepLen" in keys:
        field = "minPepLen"
    elif "minPeptideLength" in keys:
        field = "minPeptideLength"
    else:
        raise KeyError("MaxQuant parameters contain no minimum peptide length field")
    return int(_text(series.loc[field].squeeze(), field))


def _mods_for_version(series: pd.Series, version: str) -> tuple[str, str]:
    """Homogenize fixed/variable modifications, handling the 1.6.0.0 path change."""
    if version > "1.6.0.0":
        fixed_path = pd.IndexSlice["parameterGroups", "parameterGroup", "fixedModifications", :]
    else:
        fixed_path = pd.IndexSlice["fixedModifications", :]
    fixed_mods = _joined_text(series.loc[fixed_path].squeeze(), "fixedModifications")

    variable_mods = _joined_text(
        series.loc[
            pd.IndexSlice["parameterGroups", "parameterGroup", "variableModifications", :]
        ].squeeze(),
        "variableModifications",
    )

    return _homogenize_mods(fixed_mods), _homogenize_mods(variable_mods)


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
    enzyme_mode = int(
        _text(
            series.loc[("parameterGroups", "parameterGroup", "enzymeMode")].squeeze(),
            "enzymeMode",
        )
    )
    fixed_mods, variable_mods = _mods_for_version(series, version)

    return Parameters.model_validate(
        {
            "software_name": "MaxQuant",
            "software_version": version,
            "search_engine": "Andromeda",
            "ident_fdr_psm": float(_text(series.loc["peptideFdr"].squeeze(), "peptideFdr")),
            "ident_fdr_peptide": None,
            "ident_fdr_protein": float(_text(series.loc["proteinFdr"].squeeze(), "proteinFdr")),
            "enable_match_between_runs": (
                _text(
                    series.loc["matchBetweenRuns"].squeeze(),
                    "matchBetweenRuns",
                ).lower()
                == "true"
            ),
            "precursor_mass_tolerance": precursor_tolerance,
            "fragment_mass_tolerance": fragment_tolerance,
            "enzyme": _text(
                series.loc[("parameterGroups", "parameterGroup", "enzymes", "string")].squeeze(),
                "enzyme",
            ),
            "semi_enzymatic": enzyme_mode != 0,
            "allowed_miscleavages": int(
                _text(
                    series.loc[
                        pd.IndexSlice[
                            "parameterGroups",
                            "parameterGroup",
                            "maxMissedCleavages",
                            :,
                        ]
                    ].squeeze(),
                    "maxMissedCleavages",
                )
            ),
            "min_peptide_length": _min_peptide_length(series),
            "max_peptide_length": None,
            "fixed_mods": fixed_mods,
            "variable_mods": variable_mods,
            "max_mods": int(
                _text(
                    series.loc[("parameterGroups", "parameterGroup", "maxNmods")].squeeze(),
                    "maxNmods",
                )
            ),
            "min_precursor_charge": None,
            "max_precursor_charge": int(
                _text(
                    series.loc[
                        pd.IndexSlice["parameterGroups", "parameterGroup", "maxCharge", :]
                    ].squeeze(),
                    "maxCharge",
                )
            ),
        }
    )
