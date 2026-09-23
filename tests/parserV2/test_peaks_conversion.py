"""PEAKS settings select rules but never filter measurements or enter parsed provenance."""

from __future__ import annotations

from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from apb2.command.conversion import convert_all_from_packaged_rules
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from parserV2.fixtures import committed_sample


@pytest.mark.parametrize("suffix", [".h5ad", ".h5mu", ".parquet", ".duckdb"])
def test_peaks_fdr_does_not_filter_or_enter_parsed_provenance(
    tmp_path: Path,
    suffix: str,
) -> None:
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
            software="peaks",
            checks="standard",
        )
        assert result.version is None
        assert [level.level for level in result.levels] == ["ion"]
        targets.append(target)

    first_result = read_parsed_levels(targets[0])
    second_result = read_parsed_levels(targets[1])
    first = first_result.levels["ion"]
    second = second_result.levels["ion"]
    assert first.obs.frame.height == 6
    assert first.var.frame.height > 0
    assert_frame_equal(first.obs.frame, second.obs.frame)
    assert_frame_equal(first.var.frame, second.var.frame)
    assert first.layers.keys() == second.layers.keys()
    for name in first.layers:
        assert_frame_equal(first.layers[name].values, second.layers[name].values)
    assert first_result.uns == {}
    assert second_result.uns == {}
