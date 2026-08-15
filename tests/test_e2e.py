"""Parity: apb2 matches the legacy conversion on every packaged rule with cached data.

Each side composes its rule from its own document tree — apb2 from ``apb2.vendor_parse_rules.documents``,
legacy from ``anndata_proteomics.vendor_quant_rules`` — mapped by document-relative path.
Feeding one rule object into both verticals would let copied-class dispatch silently
misroute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata_proteomics.converters.assemble import convert_table
from anndata_proteomics.converters.pipeline import (
    string_sources_for_rules as legacy_string_sources,
)
from anndata_proteomics.readers.dispatch import read_table_preserving_strings
from anndata_proteomics.test_data import VendorDataUnavailable, find_test_data_for_version
from anndata_proteomics.vendor_quant_rules.loader import load_rule as load_legacy_rule
from anndata_proteomics.vendor_quant_rules.registry import (
    iter_packaged_rules as iter_legacy_rules,
)

from apb2.parse_quant.parse_strategy import make_parse_strategy
from apb2.parse_quant.sources import SingleFile
from apb2.vendor_params.model import Parameters
from apb2.vendor_params.registry import parse_params
from apb2.vendor_parse_rules.documents.select import packaged_documents, software_slug
from apb2.vendor_parse_rules.model import (
    LongRule,
    QuantificationLevel,
    WideRule,
    compose_rule,
    load_document,
)
from apb2.vendor_parse_rules.runtime import available_for

_APB2_DOCUMENT_PATHS = [document.path for document in packaged_documents()]


def _document_key(path: Path) -> tuple[str, ...]:
    """Return the path parts below the tree's ``documents/`` root."""
    parts = path.parts
    index = len(parts) - 1 - parts[::-1].index("documents")
    return parts[index + 1 :]


_LEGACY_LOCATORS = {
    (_document_key(locator.path), locator.level): locator for locator in iter_legacy_rules()
}

_CASES = [
    pytest.param(path, level, id=f"{path.parent.name}/{level}")
    for path in _APB2_DOCUMENT_PATHS
    for level in load_document(path).levels
]


def _cached_parameters(rule: LongRule | WideRule, data_file: Path) -> Parameters | None:
    """Parse the parameter file cached beside ``data_file``, when there is one."""
    param_paths = sorted(data_file.parent.glob("param_0.*"))
    if not param_paths:
        return None
    slug = software_slug(rule.software_name)
    return parse_params(param_paths[0], software=slug)


@pytest.mark.parametrize(("path", "level"), _CASES)
def test_apb2_matches_legacy_conversion(path: Path, level: QuantificationLevel) -> None:
    rule = compose_rule(load_document(path), level)
    if rule.fragments is not None:
        pytest.skip("fragment level converted on a subset in legacy tests")

    data_file = find_test_data_for_version(rule.software_name, rule.software_version_pattern)
    if isinstance(data_file, VendorDataUnavailable) or not data_file.exists():
        pytest.skip(f"no test data for {rule.software_name!r} {rule.software_version_pattern!r}")
    parameters = _cached_parameters(rule, data_file)
    if not available_for(rule, parameters):
        pytest.skip(f"cached {rule.software_name} file is not {level}-level")

    legacy_locator = _LEGACY_LOCATORS[(_document_key(path), level)]
    legacy_rule = load_legacy_rule(legacy_locator)

    df = read_table_preserving_strings(data_file, legacy_string_sources([legacy_rule]))
    if not legacy_rule.matches_headers(list(df.columns)):
        pytest.skip(f"cached {rule.software_name} file lacks columns for {level}")

    pieces = convert_table(df, legacy_rule)
    parsed = make_parse_strategy(rule, SingleFile(data_file), parameters).parse()

    np.testing.assert_allclose(parsed.X, pieces.X, equal_nan=True)
    pd.testing.assert_frame_equal(parsed.obs, pieces.obs)
    pd.testing.assert_frame_equal(parsed.var, pieces.var)
    assert set(parsed.layers) == set(pieces.layers)
    for name, matrix in pieces.layers.items():
        np.testing.assert_allclose(parsed.layers[name], matrix, equal_nan=True)

    assert parsed.uns["software_name"] == rule.software_name
    assert parsed.uns["quantification_level"] == rule.quantification_level
    assert json.loads(str(parsed.uns["rule_json"])) == rule.model_dump(mode="json")
