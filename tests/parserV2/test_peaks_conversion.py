"""PEAKS settings select rules but never filter measurements or enter parsed provenance."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from apb2.cli.conversion import convert_all_from_packaged_rules
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


def test_peaks_retains_the_minus_10lgp_identification_score(tmp_path: Path) -> None:
    """Every feature row's ``-10LgP`` survives as the numeric var column ``Minus_10LgP``."""
    data = committed_sample("peaks")
    assert data is not None
    target = tmp_path / "converted.parquet"
    convert_all_from_packaged_rules(
        data=data,
        output=target,
        parameters_path=None,
        software="peaks",
        checks="standard",
    )

    var = read_parsed_levels(target).levels["ion"].var.frame
    source = pl.read_csv(data).get_column("-10LgP")
    assert var.schema["Minus_10LgP"] == pl.Float64
    assert_series_equal(
        var.get_column("Minus_10LgP").sort(),
        source.sort(),
        check_names=False,
    )


def test_peaks_samples_come_from_run_mz_columns_not_group_averages(tmp_path: Path) -> None:
    """Runs of any name are samples; ``<group> Normalized Area`` averages have no ``m/z``."""
    runs = ("A1_run", "B1_run")
    per_run = [
        f"{run} {suffix}" for run in runs for suffix in ("m/z", "RT mean", "Normalized Area")
    ]
    header = [
        *("Peptide", "Quality", "Significance", "-10LgP", "m/z", "RT range", "z", "Avg. Area"),
        *per_run,
        *("Sample Profile (Ratio)", "A Normalized Area", "B Normalized Area"),
        *("Group Profile (Ratio)", "RT mean", "Id Count", "Accession", "PTM"),
    ]
    rows = [
        [
            *("PEPTIDEK", "1", "20", "35.5", "450.7", "10.1-10.4", "2", "150"),
            *("450.7", "10.2", "100", "450.7", "10.3", "200"),
            *("1:2", "100", "200", "1:2", "10.25", "2", "P00001", ""),
        ],
        [
            *("M(+15.99)PEPTIDER", "1", "18", "30.1", "380.2", "12.0-12.2", "3", "300"),
            *("380.2", "12.1", "300", "380.2", "12.1", "0"),
            *("1:0", "300", "0", "1:0", "12.1", "1", "P00002", "Oxidation (M)"),
        ],
    ]
    data = tmp_path / "peaks.csv"
    pl.DataFrame(rows, schema=header, orient="row").write_csv(data)
    target = tmp_path / "converted.parquet"
    convert_all_from_packaged_rules(
        data=data,
        output=target,
        parameters_path=None,
        software="peaks",
        checks="standard",
    )

    level = read_parsed_levels(target).levels["ion"]
    assert level.obs.frame.get_column("sample").to_list() == list(runs)
    assert set(level.layers) == {"Normalized_Area", "Sample_Mz", "Sample_RT_Mean"}
    assert level.layers["Normalized_Area"].values.width == len(runs)
