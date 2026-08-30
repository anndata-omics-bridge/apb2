"""Physical input: which dialect a file is, and exactly which of its columns one level reads.

Two properties matter here. A candidate dialect is judged by the header it exposes, so an
unusable candidate costs one row rather than one table; and every projected column's dtype is
decided before the read, so a lexical token cannot be reinterpreted on the way in.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2 import compile as composition
from apb2.parserV2.parse_quant import delimited_input, parquet_input
from apb2.parserV2.parse_quant.contracts import BoundInputReader
from apb2.parserV2.parse_quant.errors import AmbiguousDialectError, IncompatibleSourceError
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFile,
    DelimitedFormatContract,
    Folder,
    InputContract,
    LevelReadPlan,
    NumericTextFormat,
    ParquetFormatContract,
    ParquetSourceEvidence,
    SingleFile,
)
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from parserV2 import synthetic
from parserV2.fixtures import PackagedDocument, document_pairs

DOT = NumericTextFormat(decimal_mark=".", thousands_marks=())
COMMA = NumericTextFormat(decimal_mark=",", thousands_marks=())
GROUPED = NumericTextFormat(decimal_mark=",", thousands_marks=(".",))

TSV = DelimitedFormatContract(
    extensions=(".tsv",),
    encoding="utf8",
    quote_char='"',
    delimiter_candidates=("\t",),
    number_format_candidates=(DOT,),
)
TEXT = DelimitedFormatContract(
    extensions=(".txt",),
    encoding="utf8",
    quote_char='"',
    delimiter_candidates=("\t", ",", ";"),
    number_format_candidates=(DOT, COMMA),
)
PARQUET = ParquetFormatContract(extensions=(".parquet",))


def accepts_sample_and_feature(header: tuple[str, ...]) -> bool:
    return {"Sample", "Feature"} <= set(header)


def write(path: Path, text: str, *, bom: bool = False) -> Path:
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return path


def plan(*columns: str, numeric: tuple[str, ...] = ()) -> LevelReadPlan:
    return LevelReadPlan(
        projected_columns=columns,
        text_sources=frozenset(name for name in columns if name not in numeric),
        native_numeric_sources=frozenset(numeric),
    )


# --------------------------------------------------------------------------------- dialects


@pytest.mark.parametrize(
    ("delimiter", "name"),
    [("\t", "tab"), (",", "comma"), (";", "semicolon")],
)
def test_a_declared_delimiter_candidate_is_resolved_from_the_header(
    tmp_path: Path, delimiter: str, name: str
) -> None:
    path = write(
        tmp_path / f"{name}.txt",
        delimiter.join(("Sample", "Feature", "Quantity"))
        + "\n"
        + delimiter.join(("A", "F", "1.5"))
        + "\n",
    )

    evidence = delimited_input.detected_evidence(path, TEXT, accepts_sample_and_feature)

    assert evidence.delimiter == delimiter
    assert evidence.columns == ("Sample", "Feature", "Quantity")


def test_a_fixed_delimiter_needs_no_detection_but_still_needs_a_usable_header(
    tmp_path: Path,
) -> None:
    good = write(tmp_path / "good.tsv", "Sample\tFeature\nA\tF\n")
    bad = write(tmp_path / "bad.tsv", "Sample\tOther\nA\tF\n")

    assert delimited_input.detected_evidence(good, TSV, accepts_sample_and_feature).delimiter == (
        "\t"
    )
    with pytest.raises(IncompatibleSourceError, match="no usable header"):
        delimited_input.detected_evidence(bad, TSV, accepts_sample_and_feature)


def test_a_header_usable_under_two_candidates_is_ambiguous_not_guessed(tmp_path: Path) -> None:
    # Both the tab and the semicolon reading expose Sample and Feature.
    path = write(tmp_path / "both.txt", "Sample\tFeature;Sample;Feature\nA\tF;A;F\n")

    with pytest.raises(AmbiguousDialectError, match="several declared delimiters"):
        delimited_input.detected_evidence(
            path, TEXT, lambda header: {"Sample", "Feature"} & set(header) != set()
        )


def test_a_quoted_delimiter_stays_inside_its_field(tmp_path: Path) -> None:
    path = write(
        tmp_path / "quoted.tsv",
        'Sample\tFeature\nA\t"holds\ta tab"\n',
    )

    evidence = delimited_input.detected_evidence(path, TSV, accepts_sample_and_feature)
    frame = delimited_input.make_delimited_reader(path, evidence, plan("Sample", "Feature")).read()

    assert evidence.columns == ("Sample", "Feature")
    assert frame.frame.get_column("Feature").to_list() == ["holds\ta tab"]


def test_a_utf8_bom_does_not_become_part_of_the_first_column_name(tmp_path: Path) -> None:
    path = write(tmp_path / "bom.tsv", "Sample\tFeature\nA\tF\n", bom=True)

    evidence = delimited_input.detected_evidence(path, TSV, accepts_sample_and_feature)

    assert evidence.columns == ("Sample", "Feature")


def test_a_stated_dialect_is_checked_against_the_declaration_and_the_header(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "stated.txt", "Sample;Feature\nA;F\n")
    stated = DelimitedFile(path=path, delimiter=";", encoding="utf8", numbers=DOT)

    evidence = delimited_input.stated_evidence(stated, TEXT, accepts_sample_and_feature)

    assert evidence.delimiter == ";"
    assert evidence.number_format == DOT
    with pytest.raises(IncompatibleSourceError, match="not among the declared candidates"):
        delimited_input.stated_evidence(
            DelimitedFile(path=path, delimiter="|", encoding="utf8", numbers=DOT),
            TEXT,
            accepts_sample_and_feature,
        )
    with pytest.raises(IncompatibleSourceError, match="number format"):
        delimited_input.stated_evidence(
            DelimitedFile(path=path, delimiter=";", encoding="utf8", numbers=GROUPED),
            TEXT,
            accepts_sample_and_feature,
        )


def test_a_stated_dialect_that_hides_the_required_columns_is_incompatible(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "stated.txt", "Sample;Feature\nA;F\n")
    stated = DelimitedFile(path=path, delimiter="\t", encoding="utf8", numbers=DOT)

    with pytest.raises(IncompatibleSourceError, match="does not carry the columns"):
        delimited_input.stated_evidence(stated, TEXT, accepts_sample_and_feature)


# -------------------------------------------------------------------------- number notation


def test_a_comma_decimal_file_is_detected_from_its_own_values(tmp_path: Path) -> None:
    path = write(
        tmp_path / "locale.txt",
        "Sample\tFeature\tQuantity\nA\tF\t1,5\nB\tG\t2,25\n",
    )

    evidence = delimited_input.detected_evidence(path, TEXT, accepts_sample_and_feature)

    assert evidence.number_format == COMMA
    frame = delimited_input.make_delimited_reader(
        path, evidence, plan("Sample", "Feature", "Quantity", numeric=("Quantity",))
    ).read()
    assert frame.frame.get_column("Quantity").to_list() == [1.5, 2.25]


def test_a_three_digit_group_is_never_read_as_decimal_evidence(tmp_path: Path) -> None:
    path = write(
        tmp_path / "grouped.txt",
        "Sample\tFeature\tQuantity\nA\tF\t1,234\nB\tG\t2,000\n",
    )

    evidence = delimited_input.detected_evidence(path, TEXT, accepts_sample_and_feature)

    assert evidence.number_format == DOT


def test_a_comma_delimited_file_cannot_also_carry_comma_decimals(tmp_path: Path) -> None:
    path = write(tmp_path / "commas.txt", "Sample,Feature,Quantity\nA,F,1.5\n")

    evidence = delimited_input.detected_evidence(path, TEXT, accepts_sample_and_feature)

    assert evidence.delimiter == ","
    assert evidence.number_format == DOT


def test_values_readable_under_two_decimal_marks_are_ambiguous(tmp_path: Path) -> None:
    path = write(
        tmp_path / "mixed.txt",
        "Sample\tFeature\tOne\tTwo\nA\tF\t1.5\t2,25\n",
    )

    with pytest.raises(AmbiguousDialectError, match="several declared marks"):
        delimited_input.detected_evidence(path, TEXT, accepts_sample_and_feature)


def test_a_grouped_notation_leaves_its_values_as_text_for_a_later_encoder(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "grouped.tsv", "Sample\tFeature\tQuantity\nA\tF\t100.000.000\n")
    evidence = delimited_input.detected_evidence(path, TSV, accepts_sample_and_feature)

    frame = delimited_input.make_delimited_reader(
        path, evidence, plan("Sample", "Feature", "Quantity")
    ).read()

    assert frame.frame.get_column("Quantity").to_list() == ["100.000.000"]


def test_no_projected_column_is_left_to_inference(tmp_path: Path) -> None:
    """A lexical token survives exactly as written, whatever it looks like."""
    path = write(
        tmp_path / "lexical.tsv",
        "Sample\tFeature\tCharge\tQuantity\nA\tF\t01\t1.5\nB\tG\t1\t2.5\n",
    )
    evidence = delimited_input.detected_evidence(path, TSV, accepts_sample_and_feature)

    frame = delimited_input.make_delimited_reader(
        path,
        evidence,
        plan("Sample", "Feature", "Charge", "Quantity", numeric=("Quantity",)),
    ).read()

    assert frame.frame.get_column("Charge").to_list() == ["01", "1"]
    assert frame.frame.schema["Charge"] == pl.String
    assert frame.frame.schema["Quantity"] == pl.Float64


def test_a_read_returns_the_projection_in_plan_order(tmp_path: Path) -> None:
    path = write(tmp_path / "order.tsv", "Sample\tFeature\tQuantity\nA\tF\t1.5\n")
    evidence = delimited_input.detected_evidence(path, TSV, accepts_sample_and_feature)

    frame = delimited_input.make_delimited_reader(path, evidence, plan("Feature", "Sample")).read()

    assert frame.frame.columns == ["Feature", "Sample"]


# ------------------------------------------------------------------------------ binding


def contract(
    *formats: DelimitedFormatContract | ParquetFormatContract, **kwargs: object
) -> InputContract:
    file_name = kwargs.get("file_name")
    assert file_name is None or isinstance(file_name, str)
    return InputContract(file_name=file_name, formats=formats)


def test_a_single_file_binds_through_its_extension(tmp_path: Path) -> None:
    path = write(tmp_path / "report.tsv", "Sample\tFeature\nA\tF\n")

    bound = composition.bind_source(SingleFile(path=path), contract(TSV, PARQUET))

    assert bound.path == path
    assert bound.format == TSV


def test_one_declared_format_treats_the_extension_as_a_hint(tmp_path: Path) -> None:
    path = write(tmp_path / "generic-name.txt", "Sample\tFeature\nA\tF\n")

    bound = composition.bind_source(SingleFile(path=path), contract(TSV))

    assert bound.format == TSV


def test_an_unknown_extension_is_incompatible_before_anything_is_read(tmp_path: Path) -> None:
    path = write(tmp_path / "report.dat", "Sample\tFeature\nA\tF\n")

    with pytest.raises(IncompatibleSourceError, match="no declared format accepts"):
        composition.bind_source(SingleFile(path=path), contract(TSV, PARQUET))


def test_a_folder_binds_the_one_declared_candidate_it_holds(tmp_path: Path) -> None:
    write(tmp_path / "evidence.txt", "Sample\tFeature\nA\tF\n")
    write(tmp_path / "peptides.txt", "Sample\tFeature\nA\tF\n")
    declared = contract(TEXT, file_name="evidence.txt")

    bound = composition.bind_source(Folder(path=tmp_path), declared)

    assert bound.path == tmp_path / "evidence.txt"


def test_a_folder_holding_none_of_the_declared_candidates_is_incompatible(
    tmp_path: Path,
) -> None:
    write(tmp_path / "peptides.txt", "Sample\tFeature\nA\tF\n")
    declared = contract(TEXT, file_name="evidence.txt")

    with pytest.raises(IncompatibleSourceError, match="does not contain the declared file"):
        composition.bind_source(Folder(path=tmp_path), declared)


def test_a_folder_bound_to_a_rule_declaring_no_candidate_names_is_incompatible(
    tmp_path: Path,
) -> None:
    with pytest.raises(IncompatibleSourceError, match="declares no file_name"):
        composition.bind_source(Folder(path=tmp_path), contract(TEXT))


def test_the_maxquant_document_names_the_table_a_folder_must_resolve(tmp_path: Path) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "maxquant")
    facade = ParseRuleFacade(
        load_rule_document(pair.parser_v2_path),
        "ion",
        synthetic.NO_EVIDENCE,
    )
    write(tmp_path / "evidence.txt", "Raw file\tSequence\n")

    bound = composition.bind_source(Folder(path=tmp_path), facade.working_parameters.input)

    assert bound.path.name == "evidence.txt"


# ------------------------------------------------------------------------------- parquet


def test_parquet_evidence_reports_its_physical_schema_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "report.parquet"
    pl.DataFrame(
        {"Sample": ["A"], "Feature": ["F"], "Quantity": [1.5], "Count": [3]}
    ).write_parquet(path)

    evidence = parquet_input.schema_evidence(path)

    assert evidence.columns == ("Sample", "Feature", "Quantity", "Count")
    assert dict(evidence.dtypes)["Quantity"] == pl.Float64
    assert dict(evidence.dtypes)["Count"] == pl.Int64


def test_a_parquet_read_preserves_physical_types_and_projects_in_plan_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.parquet"
    pl.DataFrame({"Sample": ["A"], "Feature": ["F"], "Count": [3]}).write_parquet(path)

    frame = parquet_input.make_parquet_reader(path, plan("Count", "Sample")).read()

    assert frame.frame.columns == ["Count", "Sample"]
    assert frame.frame.schema["Count"] == pl.Int64


def test_binding_and_evidence_route_a_parquet_source_without_a_dialect(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.parquet"
    pl.DataFrame({"Sample": ["A"], "Feature": ["F"]}).write_parquet(path)
    source = SingleFile(path=path)
    bound = composition.bind_source(source, contract(TSV, PARQUET))

    evidence = composition.source_evidence(source, bound, accepts_sample_and_feature)

    assert isinstance(evidence, ParquetSourceEvidence)
    reader = composition.make_reader(bound, evidence, plan("Sample"))
    assert isinstance(reader, parquet_input.ParquetInputReader)


# ------------------------------------------------------------ one source, several levels


def test_two_levels_of_one_document_read_their_own_projections(tmp_path: Path) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1")
    document = load_rule_document(pair.parser_v2_path)
    header = (
        "Run",
        "Modified.Sequence",
        "Stripped.Sequence",
        "Precursor.Charge",
        "Precursor.Id",
        "Protein.Group",
        "Protein.Ids",
        "Protein.Names",
        "Genes",
        "Precursor.Normalised",
        "PG.MaxLFQ",
    )
    path = write(
        tmp_path / "report.tsv",
        "\t".join(header) + "\n" + "\t".join("1" * 1 for _ in header) + "\n",
    )
    evidence = delimited_input.detected_evidence(path, TSV, lambda columns: "Run" in columns)

    ion = ParseRuleFacade(document, "ion", synthetic.NO_EVIDENCE).resolve_source(evidence)
    protein = ParseRuleFacade(document, "protein", synthetic.NO_EVIDENCE).resolve_source(evidence)

    assert "Modified.Sequence" in ion.read.projected_columns
    assert "Modified.Sequence" not in protein.read.projected_columns
    assert "PG.MaxLFQ" in protein.read.projected_columns
    assert "PG.MaxLFQ" not in ion.read.projected_columns
    for resolved in (ion, protein):
        frame = delimited_input.make_delimited_reader(path, evidence, resolved.read).read()
        assert frame.frame.columns == list(resolved.read.projected_columns)


# ---------------------------------------------------------------------- structural typing


def test_both_readers_satisfy_the_parser_owned_reader_contract(tmp_path: Path) -> None:
    text = write(tmp_path / "report.tsv", "Sample\tFeature\nA\tF\n")
    table = tmp_path / "report.parquet"
    pl.DataFrame({"Sample": ["A"], "Feature": ["F"]}).write_parquet(table)
    evidence = delimited_input.detected_evidence(text, TSV, accepts_sample_and_feature)

    readers: tuple[BoundInputReader, ...] = (
        delimited_input.make_delimited_reader(text, evidence, plan("Sample")),
        parquet_input.make_parquet_reader(table, plan("Sample")),
    )

    for reader in readers:
        assert reader.read().frame.columns == ["Sample"]


@pytest.mark.parametrize("pair", [pytest.param(pair, id=pair.key) for pair in document_pairs()])
def test_every_cached_vendor_export_resolves_to_one_unambiguous_reading(
    pair: PackagedDocument,
) -> None:
    """The declared input policy must accept the files these rules were written for."""
    document = load_rule_document(pair.parser_v2_path)
    path = pair.data_path()
    if path is None:
        pytest.skip(f"no cached export for {pair.key}")
    facade = pair.first_admitted_facade()
    source = SingleFile(path=path)
    bound = composition.bind_source(source, facade.working_parameters.input)

    # A document with several levels shares one binding, so the predicate that decides a
    # dialect is "does any level recognize this header" -- the same question vendor
    # detection asks.
    evidence = composition.source_evidence(source, bound, document.matches)

    assert evidence.columns == pair.header()
    if isinstance(evidence, ParquetSourceEvidence):
        assert len(evidence.dtypes) == len(evidence.columns)
    else:
        assert isinstance(bound.format, DelimitedFormatContract)
        assert evidence.number_format in bound.format.number_format_candidates
