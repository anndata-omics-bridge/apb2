"""ProteoBench's Custom ion-upload table needs no search-parameter file."""

from __future__ import annotations

from pathlib import Path

import anndata
import numpy as np
import pytest

from apb2.api import ParseRuleCompiler
from apb2.cli import ConvertCliOptions, convert
from apb2.parserV2.detect_document import RuleUnavailableError
from parserV2.fixtures import committed_sample


def _sample() -> Path:
    source = committed_sample("pb_custom")
    assert source is not None
    return source


def test_pb_custom_hint_selects_parameter_free_packaged_rule() -> None:
    compiler = ParseRuleCompiler(_sample(), None, software="pb_custom", requested_levels=("ion",))

    assert compiler.detection.software == "pbcustom"
    assert compiler.detection.version is None
    assert len(compiler.detection.levels) == 1
    assert compiler.detection.levels[0].document.software_name == "pb_custom"
    with pytest.raises(ValueError, match="no vendor parameter file"):
        _ = compiler.parameters


def test_pb_custom_cli_conversion_without_params(tmp_path: Path) -> None:
    output = tmp_path / "converted"

    assert convert(_sample(), "ion", ConvertCliOptions(software="pb_custom", output=output)) == 0

    result = anndata.read_h5ad(tmp_path / "converted.h5ad")
    assert result.shape == (6, 8493)
    quantities = np.asarray(result.X)
    assert np.isfinite(quantities).any()
    assert np.isnan(quantities).any()
    assert result.var["ProForma_ion"].str.endswith("/2").any()
    assert result.var["Proteins"].str.contains("P52292").any()


def test_parameter_free_packaged_rules_require_a_matching_software_hint() -> None:
    with pytest.raises(RuleUnavailableError, match="requires --software"):
        ParseRuleCompiler(_sample(), None)
    with pytest.raises(RuleUnavailableError, match=r"candidate vendors.*maxquant"):
        ParseRuleCompiler(_sample(), None, software="maxquant")
