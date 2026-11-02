"""MaxQuant's match status: whether a value was identified by MS/MS or by matching between runs."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from apb2.cli.conversion import convert_all_from_rule_config
from apb2.parserV2.parse_quant.data.parsed import CategoricalLayerSemantics
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from parserV2.fixtures import committed_dir, committed_sample

RULE = Path("src/apb2/parserV2/vendor_parse_rules/documents/maxquant_wide/rules.json")


def _factor_layer(
    key: str, level: str, layer: str, tmp_path: Path
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Convert one committed MaxQuant sample; return a factor layer's cells and category codes."""
    data = committed_sample(key)
    folder = committed_dir(key)
    assert data is not None
    assert folder is not None
    params = folder / "param_0..xml"
    target = tmp_path / "converted.parquet"
    convert_all_from_rule_config(
        data=data,
        output=target,
        rule_config=RULE,
        parameters_path=params if params.exists() else None,
        software=None,
        checks="standard",
    )
    table = read_parsed_levels(target).levels[level].layers[layer]
    assert isinstance(table.semantics, CategoricalLayerSemantics)
    return table.cells, dict(table.semantics.categories)


def test_each_identification_type_label_becomes_its_category_code(tmp_path: Path) -> None:
    """Every ``By MS/MS`` and ``By matching`` cell of peptides.txt keeps its label as a code."""
    cells, categories = _factor_layer(
        "maxquant_peptides", "peptide", "Identification_Type", tmp_path
    )
    data = committed_sample("maxquant_peptides")
    assert data is not None
    source = pl.read_csv(data, separator="\t", infer_schema=False, quote_char=None)
    labels = (
        source.select(pl.col(r"^Identification type .+$"))
        .unpivot()
        .get_column("value")
        .drop_nulls()
    )

    assert categories == {"By MS/MS": 0, "By matching": 1}
    for label, code in categories.items():
        assert cells.get_column("value").eq(code).sum() == labels.eq(label).sum() > 0
    assert cells.height == labels.len()


def test_every_evidence_type_is_a_declared_category(tmp_path: Path) -> None:
    """No evidence row falls back to the unknown code, including the undocumented
    ``MULTI-MATCH-MSMS``."""
    cells, categories = _factor_layer("maxquant", "peptidoform_ion", "Type", tmp_path)
    codes = set(cells.get_column("value").to_list())

    assert codes <= set(categories.values())
    assert categories["MULTI-MATCH-MSMS"] in codes
    assert categories["MULTI-MATCH"] in codes
