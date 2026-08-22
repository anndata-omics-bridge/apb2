"""Parser V2 against the unchanged implementation, on the files these rules were written for.

The generic claim is that one implementation covers every vendor. The only way to believe it is
to run both conversions over the same cached export and compare the quantities cell by cell,
which is what most of this module does. The rest checks the boundary the application owns: it
reads its own parameter model and hands Parser V2 the two fields schema 0.3 permits.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata_proteomics.converters._pieces import ConversionPieces
from anndata_proteomics.converters.assemble import convert_table
from anndata_proteomics.converters.pipeline import string_sources_for_rules
from anndata_proteomics.readers.dispatch import read_table_preserving_strings
from anndata_proteomics.vendor_params.model import Parameters as OracleParameters
from anndata_proteomics.vendor_quant_rules.loader import (
    load_parameterized_rule as load_parameterized_oracle_rule,
)
from anndata_proteomics.vendor_quant_rules.loader import load_rule as load_oracle_rule
from anndata_proteomics.vendor_quant_rules.registry import iter_packaged_rules
from anndata_proteomics.vendor_quant_rules.schema.parse_rule import ParseRule as OracleRule

from apb2.parserV2.compile import (
    AnnDataOutput,
    ParquetOutput,
    ParseRuleCompiler,
    compile_parsers,
)
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    search_parameter_evidence,
    software_slug,
)
from apb2.parserV2.parse_quant.data.parsed import ParsedLevel
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_quant.parquet_writer import MANIFEST_NAME
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import (
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2.fixtures import DocumentPair, document_pairs, level_pairs
from parserV2.rule_inventory import document_key

_SEPARATOR = "\x1f"
"""A unit separator, so a joined key cannot be confused with a key containing one."""

_LEVEL_CASES = [
    pytest.param(pair, level, id=f"{pair.key}/{level}") for pair, level in level_pairs()
]
_ORACLE_LOCATORS = {
    (document_key(locator.path), locator.level): locator for locator in iter_packaged_rules()
}


def cached_parameters(pair: DocumentPair) -> Parameters | None:
    """The parameter file cached beside this vendor's export, when there is one."""
    data = pair.data_path()
    if data is None:
        return None
    found = sorted(data.parent.glob("param_0.*"))
    if not found:
        return None
    software = load_rule_document(pair.parser_v2_path).software_name
    return parse_params(found[0], software=software_slug(software))


def evidence_for(pair: DocumentPair) -> SearchParameterEvidence:
    """What the application hands Parser V2: the two permitted fields, or neither."""
    parameters = cached_parameters(pair)
    if parameters is None:
        return UNKNOWN_SEARCH_PARAMETERS
    return search_parameter_evidence(parameters)


def joined_keys(frame: object, columns: tuple[str, ...]) -> list[str]:
    """One address per row, from the authored key columns of either implementation's axis.

    ``object`` because ``AnnData.obs`` is typed as either a pandas frame or its own lazy
    stand-in; what a written and re-read object hands back here is the former.
    """
    assert isinstance(frame, pd.DataFrame)
    return list(frame[list(columns)].astype("string").agg(_SEPARATOR.join, axis=1))


def matrix_frame(
    matrix: object, obs: object, var: object, keys: tuple[tuple[str, ...], ...]
) -> pd.DataFrame:
    """One layer as a frame addressed by its authored keys, so two of them can be aligned."""
    obs_keys, var_keys = keys
    return pd.DataFrame(
        np.asarray(matrix, dtype="float64"),
        index=joined_keys(obs, obs_keys),
        columns=joined_keys(var, var_keys),
    )


def assert_same_quantities(
    label: str, expected: np.ndarray, actual: np.ndarray, *, aggregates: bool
) -> None:
    """The two conversions agree, allowing the one difference the architecture chose.

    A numeric aggregate over a cell with rows but no usable value is ``0.0`` in the
    unchanged implementation and missing here. That was a deliberate decision: a wide layer
    carries values, not a ledger of which physical cells existed, so "summed to nothing" and
    "nothing to sum" cannot be told apart afterwards, and reporting the honest one beats
    manufacturing a zero.
    """
    close = np.isclose(expected, actual, rtol=1e-9, atol=0.0, equal_nan=True)
    if aggregates:
        close |= (expected == 0.0) & np.isnan(actual)
    differing = int((~close).sum())
    assert not differing, f"{label}: {differing}/{expected.size} quantities differ"


def oracle_conversion(
    pair: DocumentPair, level: QuantificationLevel
) -> tuple[ConversionPieces, OracleRule]:
    locator = _ORACLE_LOCATORS[(pair.key, level)]
    parameters = cached_parameters(pair)
    rule = (
        load_parameterized_oracle_rule(
            locator,
            OracleParameters.model_validate(parameters.model_dump(mode="json")),
        )
        if parameters is not None
        else load_oracle_rule(locator)
    )
    data = pair.data_path()
    assert data is not None
    frame = read_table_preserving_strings(data, string_sources_for_rules([rule]))
    return convert_table(frame, rule), rule


def parser_v2_conversion(
    pair: DocumentPair, level: QuantificationLevel
) -> tuple[ParsedLevel, anndata.AnnData]:
    """The parsed level and the object it was written to; layer *order* lives in the former.

    An ``.h5ad`` stores its layers by name, so reading one back says nothing about the order
    the parse produced.
    """
    document = load_rule_document(pair.parser_v2_path)
    data = pair.data_path()
    assert data is not None
    facade = ParseRuleFacade(document, level, evidence_for(pair))
    parser = ParseRuleCompiler(facade=facade, output=AnnDataOutput()).compile(SingleFile(path=data))
    parsed = parser.parse()
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "level.h5ad"
        parser.convert(parsed, target)
        return parsed, anndata.read_h5ad(target)


# ------------------------------------------------------------------------ packaged parity


@pytest.mark.parametrize(("pair", "level"), _LEVEL_CASES)
def test_the_conversion_matches_the_unchanged_implementation(
    pair: DocumentPair, level: QuantificationLevel
) -> None:
    if level == "fragment":
        pytest.skip("a fragment level converts a whole subset; covered by its own tests")
    if pair.data_path() is None:
        pytest.skip(f"no cached export for {pair.key}")
    try:
        oracle, rule = oracle_conversion(pair, level)
    except (RuleNotApplicable, ValueError, KeyError) as reason:
        pytest.skip(f"the unchanged implementation does not convert this fixture: {reason}")

    parsed, stored = parser_v2_conversion(pair, level)

    keys = (tuple(rule.axis.obs_keys), tuple(rule.axis.var_keys))
    aggregates = rule.axis.duplicates.mode == "aggregate"
    assert list(parsed.layers) == list(oracle.layers)
    assert parsed.primary_layer_name == rule.axis.x_layer
    assert set(keys[0]) <= set(stored.obs.columns)
    assert set(keys[1]) <= set(stored.var.columns)
    for name, matrix in oracle.layers.items():
        expected = matrix_frame(matrix, oracle.obs, oracle.var, keys)
        actual = matrix_frame(np.asarray(stored.layers[name]), stored.obs, stored.var, keys)
        assert sorted(expected.index) == sorted(actual.index), name
        assert sorted(expected.columns) == sorted(actual.columns), name
        aligned = actual.loc[expected.index, expected.columns]
        assert_same_quantities(name, expected.to_numpy(), aligned.to_numpy(), aggregates=aggregates)
    primary = matrix_frame(oracle.X, oracle.obs, oracle.var, keys)
    assert_same_quantities(
        "X",
        primary.to_numpy(),
        matrix_frame(np.asarray(stored.layers[rule.axis.x_layer]), stored.obs, stored.var, keys)
        .loc[primary.index, primary.columns]
        .to_numpy(),
        aggregates=aggregates,
    )


def test_the_packaged_fragment_level_parses_its_own_export() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    data = pair.data_path()
    assert data is not None
    facade = ParseRuleFacade(
        load_rule_document(pair.parser_v2_path), "fragment", evidence_for(pair)
    )

    parsed = (
        ParseRuleCompiler(facade=facade, output=ParquetOutput())
        .compile(SingleFile(path=data))
        .parse()
    )

    assert parsed.var.key_columns == ("ProForma_fragment",)
    assert parsed.var.frame.height > parsed.obs.frame.height
    # Every fragment key is one precursor ion plus one label the separator synthesized.
    assert all(
        key.count("/") >= 2
        for key in parsed.var.frame.get_column("ProForma_fragment").head(50).to_list()
    )
    assert list(parsed.layers) == ["Fragment_Quant_Raw", "Fragment_Correlations"]


# -------------------------------------------------------------- one source, several levels


def test_each_level_reads_only_its_own_columns_from_one_shared_source() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    data = pair.data_path()
    assert data is not None

    parsers = compile_parsers(
        document=load_rule_document(pair.parser_v2_path),
        levels=("ion", "protein"),
        parameter_evidence=evidence_for(pair),
        source=SingleFile(path=data),
        output=ParquetOutput(),
    )
    ion, protein = (parser.parse() for parser in parsers)

    assert ion.var.key_columns == ("ProForma_ion",)
    assert protein.var.key_columns == ("Protein_Group",)
    assert ion.var.frame.height > protein.var.frame.height
    assert list(ion.layers) != list(protein.layers)


def test_parsing_once_and_writing_twice_never_reads_again(tmp_path: Path) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v2")
    data = pair.data_path()
    assert data is not None
    document = load_rule_document(pair.parser_v2_path)
    facade = ParseRuleFacade(document, "protein", evidence_for(pair))
    to_parquet = ParseRuleCompiler(facade=facade, output=ParquetOutput()).compile(
        SingleFile(path=data)
    )
    to_anndata = ParseRuleCompiler(
        facade=ParseRuleFacade(document, "protein", evidence_for(pair)),
        output=AnnDataOutput(),
    ).compile(SingleFile(path=data))

    parsed = to_parquet.parse()
    reads: list[str] = []
    _spy_on_reads(to_parquet, reads)
    _spy_on_reads(to_anndata, reads)
    to_parquet.convert(parsed, tmp_path / "protein")
    to_anndata.convert(parsed, tmp_path / "protein.h5ad")

    assert reads == []
    assert (tmp_path / "protein" / MANIFEST_NAME).is_file()
    stored = anndata.read_h5ad(tmp_path / "protein.h5ad")
    assert stored.shape == (parsed.obs.frame.height, parsed.var.frame.height)
    # The same parsed value reached both backends, and Parquet stored it as parsing left
    # it: this source is Parquet, so its measurements were never text to begin with.
    written = pl.read_parquet(tmp_path / "protein" / "layers" / "PG_MaxLFQ.parquet")
    assert written.schema == parsed.layers["PG_MaxLFQ"].values.schema


def _spy_on_reads(parser: object, calls: list[str]) -> None:
    """Replace a compiled parser's reader with one that objects to being used."""

    class Refusing:
        def read(self) -> object:
            calls.append("read")
            raise AssertionError("convert must not read")

    object.__setattr__(parser, "_input", Refusing())


# ------------------------------------------------------------------- the outer boundary


def test_the_application_hands_over_exactly_the_two_permitted_fields() -> None:
    parameters = Parameters(
        software_name="Sage",
        acquisition_method="DDA",
        combine_charge_states=True,
    )

    evidence = search_parameter_evidence(parameters)

    assert evidence == SearchParameterEvidence(acquisition_method="DDA", combine_charge_states=True)
    assert set(vars(SearchParameterEvidence).get("__slots__", ())) == {
        "acquisition_method",
        "combine_charge_states",
    }


def test_no_parameters_read_is_a_different_fact_from_parameters_that_say_nothing() -> None:
    absent = UNKNOWN_SEARCH_PARAMETERS

    assert absent == SearchParameterEvidence(
        acquisition_method="unknown", combine_charge_states=None
    )


def test_a_gate_reached_through_the_outer_boundary_selects_the_level() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "sage")
    document = load_rule_document(pair.parser_v2_path)
    data = pair.data_path()
    assert data is not None
    combined = search_parameter_evidence(
        Parameters(software_name="Sage", acquisition_method="DDA", combine_charge_states=True)
    )
    separate = search_parameter_evidence(
        Parameters(software_name="Sage", acquisition_method="DDA", combine_charge_states=False)
    )

    for evidence, expected in ((combined, "peptidoform"), (separate, "ion")):
        parsers = compile_parsers(
            document=document,
            levels=document.levels,
            parameter_evidence=evidence,
            source=SingleFile(path=data),
            output=ParquetOutput(),
        )
        assert [parser.level for parser in parsers] == [expected]


def test_an_override_reached_through_the_outer_boundary_swaps_the_primary_layer() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v2")
    document = load_rule_document(pair.parser_v2_path)
    dda = search_parameter_evidence(Parameters(software_name="DIA-NN", acquisition_method="DDA"))
    dia = search_parameter_evidence(Parameters(software_name="DIA-NN", acquisition_method="DIA"))

    assert (
        ParseRuleFacade(document, "ion", dda).working_parameters.measurements.primary_layer_name
        == "Ms1_Normalised"
    )
    assert (
        ParseRuleFacade(document, "ion", dia).working_parameters.measurements.primary_layer_name
        == "Precursor_Normalised"
    )


def test_the_provenance_of_a_parsed_level_names_the_rule_it_came_from() -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "alphapept")
    data = pair.data_path()
    assert data is not None
    facade = ParseRuleFacade(load_rule_document(pair.parser_v2_path), "ion", evidence_for(pair))

    parsed: ParsedLevel = (
        ParseRuleCompiler(facade=facade, output=ParquetOutput())
        .compile(SingleFile(path=data))
        .parse()
    )

    assert parsed.uns["software_name"] == "AlphaPept"
    assert parsed.uns["quantification_level"] == "ion"
    assert parsed.uns["schema_version"] == "0.3"
    assert isinstance(parsed.uns["rule_json"], str)
