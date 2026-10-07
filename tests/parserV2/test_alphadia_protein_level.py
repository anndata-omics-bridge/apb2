"""AlphaDIA 2.x protein-group values belong to a protein level, not to every precursor row."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.testing import assert_series_equal

from apb2.cli.conversion import convert_all_from_rule_config
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from parserV2.fixtures import committed_dir, committed_sample

RULE = Path("src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v2/rules.json")


def test_the_protein_group_q_value_moves_from_ion_to_the_protein_level(tmp_path: Path) -> None:
    """One protein-group q-value per group, quantified by ``pg.intensity``."""
    data = committed_sample("alphadia/v2")
    folder = committed_dir("alphadia/v2")
    assert data is not None
    assert folder is not None
    target = tmp_path / "converted.parquet"
    convert_all_from_rule_config(
        data=data,
        output=target,
        rule_config=RULE,
        parameters_path=folder / "param_0..txt",
        software=None,
        checks="standard",
    )

    parsed = read_parsed_levels(target)
    protein = parsed.levels["protein"]
    source = pl.read_parquet(data).select("pg.name", "pg.qval").unique()
    joined = protein.var.frame.join(source, left_on="Protein_Group", right_on="pg.name")

    assert "Protein_Group_QValue" not in parsed.levels["ion"].var.frame.columns
    assert protein.primary_layer_name == "PG_Intensity"
    assert protein.var.frame.height == source.height
    assert_series_equal(
        joined.get_column("Protein_Group_QValue"),
        joined.get_column("pg.qval"),
        check_names=False,
    )


def test_ion_and_protein_levels_share_the_protein_identity_column(tmp_path: Path) -> None:
    """Ions are assigned to the protein level's own key, so a rollup needs no name mapping."""
    data = committed_sample("alphadia/v2")
    folder = committed_dir("alphadia/v2")
    assert data is not None
    assert folder is not None
    target = tmp_path / "converted.parquet"
    convert_all_from_rule_config(
        data=data,
        output=target,
        rule_config=RULE,
        parameters_path=folder / "param_0..txt",
        software=None,
        checks="standard",
    )

    parsed = read_parsed_levels(target)
    ion, protein = parsed.levels["ion"], parsed.levels["protein"]
    assert ion.var.roles["protein_assignment"] == "Protein_Group"
    assert ion.var.roles["fasta_accessions"] == "Proteins"
    assert protein.var.key_columns == ("Protein_Group",)
    assigned = set(ion.var.frame.get_column("Protein_Group").to_list())
    assert assigned <= set(protein.var.frame.get_column("Protein_Group").to_list())
