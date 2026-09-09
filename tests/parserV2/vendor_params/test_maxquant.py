"""MaxQuant XML parser equivalence tests."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.maxquant import extract_params
from parserV2.vendor_params import proteobench_params

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"

CASES = [
    ("mqpar1.5.3.30_MBR.xml", "mqpar1.5.3.30_MBR_sel.json"),
    ("mqpar1.5.3.30_noMBR.xml", "mqpar1.5.3.30_noMBR_sel.json"),
    ("mqpar_MQ1.6.3.3_MBR.xml", "mqpar_MQ1.6.3.3_MBR_sel.json"),
    ("mqpar_MQ2.1.3.0_noMBR.xml", "mqpar_MQ2.1.3.0_noMBR_sel.json"),
    ("mqpar_mq2.6.2.0_1mc_MBR.xml", "mqpar_mq2.6.2.0_1mc_MBR_sel.json"),
    # maxDIA writes <variableModificationsFirstSearch> as an empty element pair, the second
    # spelling of "no entries declared" that the empty-modification tests below cover.
    ("mqpar_maxdia.xml", "mqpar_maxdia_sel.json"),
]


@pytest.mark.parametrize(("xml_name", "expected_name"), CASES)
def test_maxquant_matches_proteobench(xml_name: str, expected_name: str) -> None:
    xml_path = PROTEOBENCH_PARAMS / xml_name
    expected_path = PROTEOBENCH_PARAMS / expected_name
    if not xml_path.exists() or not expected_path.exists():
        pytest.skip("ProteoBench fixture missing")
    expected = proteobench_params.expected_json(expected_path)
    params = extract_params(xml_path)

    fields = [
        "software_name",
        "software_version",
        "search_engine",
        "ident_fdr_psm",
        "ident_fdr_protein",
        "enable_match_between_runs",
        "precursor_mass_tolerance",
        "fragment_mass_tolerance",
        "enzyme",
        "semi_enzymatic",
        "allowed_miscleavages",
        "min_peptide_length",
        "fixed_mods",
        "variable_mods",
        "max_mods",
        "max_precursor_charge",
    ]
    mismatches = proteobench_params.compare(params, expected, fields)
    assert not mismatches, "; ".join(mismatches)


_BASE_MQPAR = "mqpar1.5.3.30_MBR.xml"

# MaxQuant writes an unused modification list either self-closed or as an empty element pair.
# Both spellings appear in real files; mqpar_maxdia.xml uses the second for
# variableModificationsFirstSearch.
EMPTY_SPELLINGS = [
    "<{field} />",
    "<{field}>\n</{field}>",
]


def _mqpar_with_empty(field: str, spelling: str) -> io.StringIO:
    """Rewrite the base mqpar so one modification list declares no entries."""
    source = (PROTEOBENCH_PARAMS / _BASE_MQPAR).read_text(encoding="utf-8")
    replaced, count = re.subn(
        rf"<{field}>.*?</{field}>",
        spelling.format(field=field),
        source,
        count=1,
        flags=re.DOTALL,
    )
    assert count == 1, f"base mqpar does not contain a populated <{field}>"
    return io.StringIO(replaced)


@pytest.mark.parametrize("spelling", EMPTY_SPELLINGS)
def test_empty_variable_modifications_parse_as_none_declared(spelling: str) -> None:
    if not (PROTEOBENCH_PARAMS / _BASE_MQPAR).exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(_mqpar_with_empty("variableModifications", spelling))

    assert params.variable_mods == []
    assert [mod.source for mod in params.fixed_mods] == ["C[Carbamidomethyl]"]


@pytest.mark.parametrize("spelling", EMPTY_SPELLINGS)
def test_empty_fixed_modifications_parse_as_none_declared(spelling: str) -> None:
    if not (PROTEOBENCH_PARAMS / _BASE_MQPAR).exists():
        pytest.skip("ProteoBench fixture missing")

    params = extract_params(_mqpar_with_empty("fixedModifications", spelling))

    assert params.fixed_mods == []
    assert [mod.source for mod in params.variable_mods] == [
        "M[Oxidation]",
        "Protein N-term[Acetyl]",
    ]
