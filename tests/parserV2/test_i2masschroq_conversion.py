"""i2MassChroQ's ``sequence`` column may carry PSI-MOD tokens; the peptide key never does."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from apb2.command.conversion import convert_all_from_packaged_rules
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from parserV2.fixtures import DATA_DIR


def test_i2masschroq_peptide_is_stripped_and_peptidoform_is_the_vendor_proforma(
    tmp_path: Path,
) -> None:
    sequences = (
        "PEPTIDEK",
        "PEPM[MOD:00719]TC[MOD:00397]IDEK",
        "Q[MOD:01160]PEPTIDER",
        "[MOD:00408]-M[MOD:00425]PEPTIDEK",
        "[42]-MPEPTIDER",
    )
    data = tmp_path / "i2masschroq.tsv"
    pl.DataFrame(
        {
            "rawfile": ["run_A.mzML"] * len(sequences),
            "sequence": sequences,
            "ProForma": sequences,
            "charge": [2] * len(sequences),
            "proteins": ["sp|P00001|TEST_HUMAN"] * len(sequences),
            "area": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    ).write_csv(data, separator="\t")
    target = tmp_path / "converted.parquet"
    convert_all_from_packaged_rules(
        data=data,
        output=target,
        parameters_path=DATA_DIR / "i2masschroq" / "param_0..txt",
        software="i2masschroq",
        checks="standard",
    )

    var = read_parsed_levels(target).levels["ion"].var.frame
    assert sorted(var["ProForma_peptidoform"]) == sorted(sequences)
    assert var["ProForma_ion"].to_list() == [f"{s}/2" for s in var["ProForma_peptidoform"]]
    assert dict(zip(var["ProForma_peptidoform"], var["ProForma_peptide"], strict=True)) == {
        "PEPTIDEK": "PEPTIDEK",
        "PEPM[MOD:00719]TC[MOD:00397]IDEK": "PEPMTCIDEK",
        "Q[MOD:01160]PEPTIDER": "QPEPTIDER",
        "[MOD:00408]-M[MOD:00425]PEPTIDEK": "MPEPTIDEK",
        "[42]-MPEPTIDER": "MPEPTIDER",
    }
