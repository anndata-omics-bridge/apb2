"""PEAKS settings are persisted metadata, never conversion-time measurement filters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from apb2.parserV2.conversion_facade import convert_all_from_packaged_rules
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from parserV2.fixtures import committed_sample


@pytest.mark.parametrize("suffix", [".h5ad", ".h5mu", ".parquet", ".duckdb"])
def test_peaks_fdr_metadata_roundtrips_without_filtering(tmp_path: Path, suffix: str) -> None:
    data = committed_sample("peaks")
    assert data is not None
    source_params = Path(__file__).parent / "vendor_params/params/PEAKS_astral_report.txt"
    text = source_params.read_text(encoding="utf-8")
    targets: list[Path] = []
    for index, fdr in enumerate(("1.00", "0.00")):
        parameters = tmp_path / f"settings-{index}.txt"
        parameters.write_text(text.replace("FDR(%): 1.00", f"FDR(%): {fdr}"), encoding="utf-8")
        target = tmp_path / f"converted-{index}{suffix}"
        result = convert_all_from_packaged_rules(
            data=data,
            output=target,
            parameters_path=parameters,
            software=None,
            parameters_software="peaks",
            checks="standard",
        )
        assert result.version is None
        assert [level.level for level in result.levels] == ["ion"]
        targets.append(target)

    first = read_parsed_levels(targets[0]).levels["ion"]
    second = read_parsed_levels(targets[1]).levels["ion"]
    assert first.obs.frame.height == 6
    assert first.var.frame.height > 0
    assert_frame_equal(first.obs.frame, second.obs.frame)
    assert_frame_equal(first.var.frame, second.var.frame)
    assert first.layers.keys() == second.layers.keys()
    for name in first.layers:
        assert_frame_equal(first.layers[name].values, second.layers[name].values)
    for parsed, expected_fdr in ((first, 0.01), (second, 0.0)):
        metadata = parsed.uns["search_parameters"]
        assert isinstance(metadata, str)
        parameters = json.loads(metadata)
        assert parameters["ident_fdr_protein"] == {"value": expected_fdr}
        assert parameters["enable_match_between_runs"] is None
        assert parameters["software_version"] is None
