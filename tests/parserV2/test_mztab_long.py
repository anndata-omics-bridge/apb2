"""mzTab files become long tables, one per section and sample axis, with sample metadata."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from apb2.parserV2.joins.mztab import MzTabFormatError, write_long_tables


def _text(*rows: tuple[str, ...]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


_MZTAB_1_0 = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "mzTab-type", "Quantification"),
    ("MTD", "ms_run[1]-location", "file://run_a.mzML"),
    ("MTD", "ms_run[2]-location", "file://run_b.mzML"),
    ("MTD", "assay[1]-quantification_reagent", "[MS, MS:1002038, unlabeled sample, ]"),
    ("MTD", "assay[1]-ms_run_ref", "ms_run[1]"),
    ("MTD", "assay[1]-sample_ref", "sample[1]"),
    ("MTD", "assay[2]-ms_run_ref", "ms_run[2]"),
    ("MTD", "sample[1]-species[1]", "[NEWT, 9606, Homo sapiens (Human), ]"),
    ("MTD", "study_variable[1]-assay_refs", "assay[1],assay[2]"),
    ("MTD", "study_variable[1]-description", "both"),
    ("MTD", "protein_search_engine_score[1]", "[MS, MS:1001171, Mascot:score, ]"),
    ("MTD", "psm_search_engine_score[1]", "[, , local FDR, ]"),
    ("COM", "a comment"),
    (
        "PRH",
        "accession",
        "best_search_engine_score[1]",
        "search_engine_score[1]_ms_run[1]",
        "search_engine_score[1]_ms_run[2]",
        "num_psms_ms_run[1]",
        "num_psms_ms_run[2]",
        "protein_abundance_assay[1]",
        "protein_abundance_assay[2]",
        "protein_abundance_study_variable[1]",
        "opt_global_x",
    ),
    ("PRT", "P1", "50", "50", "null", "3", "0", "100", "null", "100", "a"),
    ("PRT", "P2", "20", "10", "20", "1", "2", "5", "7", "6", "b"),
    (
        "PSH",
        "sequence",
        "search_engine_score[1]",
        "spectra_ref",
        "opt_ms_run[1]_delta",
        "opt_ms_run[2]_delta",
    ),
    ("PSM", "PEPTIDE", "0.01", "ms_run[2]:scan=5", "null", "0.5"),
    ("PSM", "PEPTIDER", "0.02", "ms_run[1]:scan=7|ms_run[2]:scan=9", "0.1", "0.2"),
)

_MZTAB_M = _text(
    ("MTD", "mzTab-version", "2.0.0-M"),
    ("MTD", "ms_run[1]-location", "file:///a.mzML"),
    ("MTD", "assay[1]", "A1"),
    ("MTD", "assay[1]-ms_run_ref", "ms_run[1]"),
    ("MTD", "study_variable[1]", "group"),
    ("MTD", "study_variable[1]-assay_refs", "assay[1]"),
    ("MTD", "id_confidence_measure[1]", "[MS, MS:1002890, fragmentation score, ]"),
    (
        "SMH",
        "SML_ID",
        "abundance_assay[1]",
        "abundance_study_variable[1]",
        "abundance_variation_study_variable[1]",
    ),
    ("SML", "1", "10", "10", "null"),
    ("SFH", "SMF_ID", "abundance_assay[1]", ""),
    ("SMF", "1", "10", ""),
    ("SEH", "SME_ID", "spectra_ref", "id_confidence_measure[1]", "opt_global_x"),
    ("SME", "1", "ms_run[1]:index=3", "0.9", "x", "", ""),
)


def _write(tmp_path: Path, text: str, name: str = "input.mzTab") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _rows(path: Path, *sort: str) -> list[dict[str, str | None]]:
    return pl.read_parquet(path).sort(list(sort)).to_dicts()


def test_assay_table_joins_runs_samples_and_study_variables(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    rows = _rows(out / "PRT.assay.parquet", "accession", "assay")

    assert [(row["accession"], row["assay"], row["protein_abundance"]) for row in rows] == [
        ("P1", "assay[1]", "100"),
        ("P1", "assay[2]", None),
        ("P2", "assay[1]", "5"),
        ("P2", "assay[2]", "7"),
    ]
    assert rows[0]["best_score[MS:1001171]"] == "50"
    assert rows[0]["opt_global_x"] == "a"
    assert rows[0]["ms_run:location"] == "file://run_a.mzML"
    assert rows[0]["sample:species[1]"] == "[NEWT, 9606, Homo sapiens (Human), ]"
    assert rows[0]["study_variable:key"] == "study_variable[1]"
    assert rows[0]["study_variable:description"] == "both"
    assert rows[1]["ms_run:location"] == "file://run_b.mzML"


def test_per_run_and_study_variable_values_get_their_own_tables(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    runs = _rows(out / "PRT.ms_run.parquet", "accession", "ms_run")
    groups = _rows(out / "PRT.study_variable.parquet", "accession")

    assert [(row["ms_run"], row["score[MS:1001171]"], row["num_psms"]) for row in runs] == [
        ("ms_run[1]", "50", "3"),
        ("ms_run[2]", None, "0"),
        ("ms_run[1]", "10", "1"),
        ("ms_run[2]", "20", "2"),
    ]
    assert runs[0]["assay:key"] == "assay[1]"
    assert runs[0]["sample:species[1]"] == "[NEWT, 9606, Homo sapiens (Human), ]"
    assert runs[3]["assay:key"] == "assay[2]"
    assert [(row["accession"], row["protein_abundance"]) for row in groups] == [
        ("P1", "100"),
        ("P2", "6"),
    ]
    assert groups[0]["study_variable:description"] == "both"
    assert groups[0]["assay:key"] == "assay[1]|assay[2]"
    assert groups[0]["ms_run:location"] == "file://run_a.mzML|file://run_b.mzML"
    assert groups[0]["sample:key"] == "sample[1]"


def test_psm_rows_take_their_run_from_spectra_ref(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    rows = _rows(out / "PSM.ms_run.parquet", "sequence", "ms_run")

    assert [(row["sequence"], row["ms_run"], row["opt_delta"]) for row in rows] == [
        ("PEPTIDE", "ms_run[2]", "0.5"),
        ("PEPTIDER", "ms_run[1]", "0.1"),
        ("PEPTIDER", "ms_run[2]", "0.2"),
    ]
    assert rows[0]["score[local FDR]"] == "0.01"
    assert rows[0]["ms_run:location"] == "file://run_b.mzML"


def test_metadata_and_sample_tables_are_written(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    metadata = pl.read_csv(out / "MTD.tsv", separator="\t", infer_schema=False)
    assays = pl.read_csv(out / "samples.assay.tsv", separator="\t", infer_schema=False)

    assert result.version == "1.0.0"
    assert metadata.height == 13
    assert assays["assay"].to_list() == ["assay[1]", "assay[2]"]
    assert {path.name for path in out.iterdir()} == {
        "MTD.tsv",
        "COM.tsv",
        "samples.assay.tsv",
        "samples.ms_run.tsv",
        "samples.study_variable.tsv",
        "PRT.assay.parquet",
        "PRT.ms_run.parquet",
        "PRT.study_variable.parquet",
        "PSM.ms_run.parquet",
        "sections",
    }


def test_sections_are_stored_as_the_file_wrote_them(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    proteins = pl.read_parquet(out / "sections" / "PRT.parquet")
    psms = pl.read_parquet(out / "sections" / "PSM.parquet")

    assert {path.name for path in (out / "sections").iterdir()} == {"PRT.parquet", "PSM.parquet"}
    assert proteins.columns[:3] == [
        "accession",
        "best_search_engine_score[1]",
        "search_engine_score[1]_ms_run[1]",
    ]
    assert proteins["protein_abundance_assay[2]"].to_list() == [None, "7"]
    assert psms.height == 2


def test_mztab_m_sections_and_confidence_scores(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _MZTAB_M), out)

    molecules = _rows(out / "SML.assay.parquet", "SML_ID")
    groups = _rows(out / "SML.study_variable.parquet", "SML_ID")
    features = _rows(out / "SMF.assay.parquet", "SMF_ID")
    evidence = _rows(out / "SME.ms_run.parquet", "SME_ID")

    assert molecules[0]["abundance"] == "10"
    assert molecules[0]["assay:name"] == "A1"
    assert groups[0]["abundance_variation"] is None
    assert groups[0]["study_variable:name"] == "group"
    assert list(features[0]) == [
        "SMF_ID",
        "assay",
        "abundance",
        "assay:name",
        "assay:ms_run_ref",
        "ms_run:key",
        "ms_run:location",
        "study_variable:key",
        "study_variable:name",
    ]
    assert evidence[0]["score[MS:1002890]"] == "0.9"
    assert evidence[0]["ms_run"] == "ms_run[1]"


def test_byte_order_mark_crlf_gzip_and_key_case_are_read(tmp_path: Path) -> None:
    path = tmp_path / "input.mzTab.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        text = _MZTAB_1_0.replace("mzTab-version", "mzTab-Version")
        handle.write("\ufeff" + text.replace("\n", "\r\n"))
    out = tmp_path / "out"

    result = write_long_tables(path, out)

    assert result.version == "1.0.0"
    assert _rows(out / "PRT.assay.parquet", "accession", "assay")[3]["opt_global_x"] == "b"


def test_values_beyond_the_header_are_refused(tmp_path: Path) -> None:
    text = _MZTAB_1_0.replace("\ta\n", "\ta\t5.3\n")

    with pytest.raises(MzTabFormatError, match=r"input.mzTab:16: PRT row has values beyond"):
        write_long_tables(_write(tmp_path, text), tmp_path / "out")


def test_repeated_header_columns_are_refused(tmp_path: Path) -> None:
    text = _text(("PRH", "accession", "accession"), ("PRT", "P1", "P1"))

    with pytest.raises(MzTabFormatError, match="header repeats accession"):
        write_long_tables(_write(tmp_path, text), tmp_path / "out")


def test_rows_before_their_header_are_refused(tmp_path: Path) -> None:
    text = _text(("PRT", "P1"))

    with pytest.raises(MzTabFormatError, match="PRT row before its PRH header"):
        write_long_tables(_write(tmp_path, text), tmp_path / "out")


def test_a_line_no_section_reads_is_refused(tmp_path: Path) -> None:
    text = _MZTAB_1_0 + _text(("XYZ", "a stray line"))

    with pytest.raises(MzTabFormatError, match=r"input.mzTab:21: unknown line prefix 'XYZ'"):
        write_long_tables(_write(tmp_path, text), tmp_path / "out")


def test_comment_lines_are_kept(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = write_long_tables(_write(tmp_path, _MZTAB_1_0), out)

    comments = pl.read_csv(out / "COM.tsv", separator="\t", infer_schema=False)

    assert result.comments == 1
    assert comments.rows() == [("14", "a comment")]


def test_a_value_for_a_run_spectra_ref_does_not_name_is_refused(tmp_path: Path) -> None:
    text = _MZTAB_1_0.replace("ms_run[2]:scan=5\tnull\t0.5", "ms_run[2]:scan=5\t0.3\t0.5")

    with pytest.raises(
        MzTabFormatError,
        match=r"PSM rows hold values for runs their spectra_ref does not name: ms_run\[1\] \(1",
    ):
        write_long_tables(_write(tmp_path, text), tmp_path / "out")


_REFERENCES = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "instrument[1]-name", "[MS, MS:1001911, Q Exactive, ]"),
    ("MTD", "ms_run[1]-location", "file://a.mzML"),
    ("MTD", "ms_run[1]-instrument_ref", "instrument[1]"),
    ("MTD", "ms_run[2]-location", "file://b.mzML"),
    ("MTD", "assay[1]-ms_run_ref", "ms_run[1]"),
    ("MTD", "assay[1]-sample_ref", "sample[1]"),
    ("MTD", "assay[2]-ms_run_ref", "ms_run[1]"),
    ("MTD", "assay[2]-sample_ref", "sample[2]"),
    ("MTD", "assay[3]-ms_run_ref", "ms_run[2]"),
    ("MTD", "assay[3]-sample_ref", "sample[3]"),
    ("MTD", "sample[1]-description", "s1"),
    ("MTD", "sample[2]-description", "s2"),
    ("MTD", "sample[3]-description", "s3"),
    ("MTD", "study_variable[1]-assay_refs", "assay[1],assay[3]"),
    ("MTD", "study_variable[1]-description", "treated"),
    ("MTD", "study_variable[2]-assay_refs", "assay[2]"),
    ("MTD", "study_variable[2]-description", "control"),
    ("PRH", "accession", "protein_abundance_assay[1]"),
    ("PRT", "P1", "1"),
)


def _samples(out: Path, axis: str) -> dict[str, dict[str, str | None]]:
    table = pl.read_csv(out / f"samples.{axis}.tsv", separator="\t", infer_schema=False)
    return {str(row[axis]): row for row in table.to_dicts()}


def test_an_assay_reaches_its_run_instrument_sample_and_study_variable(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _REFERENCES), out)

    assay = _samples(out, "assay")["assay[2]"]

    assert assay["ms_run:key"] == "ms_run[1]"
    assert assay["ms_run:location"] == "file://a.mzML"
    assert assay["instrument:name"] == "[MS, MS:1001911, Q Exactive, ]"
    assert assay["sample:description"] == "s2"
    assert assay["study_variable:description"] == "control"
    assert "assay:key" not in assay


def test_a_run_reaches_its_assays_samples_but_not_their_group_mates(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _REFERENCES), out)

    runs = _samples(out, "ms_run")

    assert runs["ms_run[1]"]["assay:key"] == "assay[1]|assay[2]"
    assert runs["ms_run[1]"]["sample:description"] == "s1|s2"
    assert runs["ms_run[1]"]["study_variable:key"] == "study_variable[1]|study_variable[2]"
    assert runs["ms_run[1]"]["instrument:key"] == "instrument[1]"
    assert runs["ms_run[2]"]["assay:key"] == "assay[3]"
    assert runs["ms_run[2]"]["sample:key"] == "sample[3]"


def test_a_study_variable_resolves_down_to_runs_and_samples(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _REFERENCES), out)

    group = _samples(out, "study_variable")["study_variable[1]"]

    assert group["assay:key"] == "assay[1]|assay[3]"
    assert group["ms_run:location"] == "file://a.mzML|file://b.mzML"
    assert group["study_variable:assay_refs"] == "assay[1],assay[3]"
    assert "ms_run:instrument_ref" not in group
    assert group["sample:description"] == "s1|s3"
    assert group["instrument:key"] == "instrument[1]"


_PSMS = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "ms_run[1]-location", "file://a.mzML"),
    ("MTD", "ms_run[2]-location", "file://b.mzML"),
    ("MTD", "psm_search_engine_score[1]", "[MS, MS:1001330, X!Tandem:expect, ]"),
    (
        "PSH",
        "PSM_ID",
        "sequence",
        "accession",
        "start",
        "end",
        "search_engine_score[1]",
        "spectra_ref",
    ),
    ("PSM", "1", "PEPK", "P1", "10", "14", "0.01", "ms_run[1]:scan=5"),
    ("PSM", "1", "PEPK", "P2", "20", "24", "0.01", "ms_run[1]:scan=5"),
    ("PSM", "1", "PEPK", "P2", "20", "24", "0.01", "ms_run[1]:scan=5"),
    ("PSM", "1", "PEPK", "P1", "40", "44", "0.01", "ms_run[1]:scan=5"),
    ("PSM", "0", "AAAK", "P3", "1", "4", "0.02", "ms_run[1]:scan=9"),
    ("PSM", "0", "AAAK", "P3", "1", "4", "0.03", "ms_run[2]:scan=9"),
)


def test_a_psm_written_once_per_protein_becomes_one_row(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _PSMS), out)

    rows = _rows(out / "PSM.ms_run.parquet", "ms_run", "PSM_ID")

    assert [
        (row["ms_run"], row["PSM_ID"], row["accession"], row["start"], row["score[MS:1001330]"])
        for row in rows
    ] == [
        ("ms_run[1]", "0", "P3", "1", "0.02"),
        ("ms_run[1]", "1", "P1|P2|P1", "10|20|40", "0.01"),
        ("ms_run[2]", "0", "P3", "1", "0.03"),
    ]


_PEPTIDES = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "ms_run[1]-location", "file://a.mzML"),
    ("MTD", "assay[1]-ms_run_ref", "ms_run[1]"),
    (
        "PEH",
        "sequence",
        "accession",
        "unique",
        "modifications",
        "charge",
        "retention_time",
        "peptide_abundance_assay[1]",
    ),
    ("PEP", "GHK", "P1", "0", "null", "2", "778.6", "100"),
    ("PEP", "GHK", "P2", "0", "null", "2", "778.6", "100"),
    ("PEP", "GHK", "P1", "0", "null", "2", "1031.0", "50"),
)


def test_peptide_rows_fold_across_proteins_but_not_across_features(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _PEPTIDES), out)

    rows = _rows(out / "PEP.assay.parquet", "retention_time")

    assert [
        (row["accession"], row["unique"], row["retention_time"], row["peptide_abundance"])
        for row in rows
    ] == [("P1", "0", "1031.0", "50"), ("P1|P2", "0", "778.6", "100")]


_IDENTIFICATION = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "ms_run[1]-location", "file://a.mzML"),
    ("PRH", "accession", "description"),
    ("PRT", "P1", "a protein"),
)


def test_a_section_without_sample_values_gets_no_long_table(tmp_path: Path) -> None:
    out = tmp_path / "out"
    text = _IDENTIFICATION + _text(
        ("PSH", "PSM_ID", "sequence", "spectra_ref"), ("PSM", "1", "PEPK", "ms_run[1]:scan=5")
    )
    result = write_long_tables(_write(tmp_path, text), out)

    assert [(table.section, table.axis) for table in result.long] == [("PSM", "ms_run")]
    assert not (out / "PRT.parquet").exists()
    assert (out / "sections" / "PRT.parquet").exists()


def test_a_file_without_sample_values_is_refused(tmp_path: Path) -> None:
    with pytest.raises(MzTabFormatError, match="no section has per-sample values"):
        write_long_tables(_write(tmp_path, _IDENTIFICATION), tmp_path / "out")


def test_stacked_tables_name_section_columns_and_keep_sample_columns(tmp_path: Path) -> None:
    result = write_long_tables(_write(tmp_path, _MZTAB_1_0), tmp_path / "out")

    stacked = result.stacked().collect()
    proteins = stacked.filter(pl.col("PRT.accession").is_not_null() & pl.col("assay").is_not_null())

    assert {"PRT.accession", "PSM.sequence", "assay", "ms_run", "ms_run:location"} <= set(
        stacked.columns
    )
    assert proteins.height == 4
    assert proteins["ms_run"].null_count() == 4
    assert stacked.filter(pl.col("PSM.sequence").is_not_null())["assay"].null_count() == 3


_PROTEIN_KINDS = _text(
    ("MTD", "mzTab-version", "1.0.0"),
    ("MTD", "ms_run[1]-location", "file://a.mzML"),
    ("MTD", "assay[1]-ms_run_ref", "ms_run[1]"),
    ("MTD", "assay[2]-ms_run_ref", "ms_run[1]"),
    (
        "PRH",
        "accession",
        "ambiguity_members",
        "protein_coverage",
        "protein_abundance_assay[1]",
        "protein_abundance_assay[2]",
        "opt_global_result_type",
    ),
    ("PRT", "P1", "P1,P2", "0.7", "10", "20", "indistinguishable_protein_group"),
    ("PRT", "P1", "null", "0.7", "null", "null", "protein_details"),
    ("PRT", "P2", "null", "0.6", "null", "null", "protein_details"),
    ("PRT", "P3", "null", "0.5", "5", "null", "single_protein"),
)


def test_protein_result_types_become_columns_of_one_row_per_accession(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_long_tables(_write(tmp_path, _PROTEIN_KINDS), out)

    table = pl.read_parquet(out / "PRT.assay.parquet").sort("accession", "assay")
    group = "indistinguishable_protein_group"

    assert "opt_global_result_type" not in table.columns
    assert table.select("accession", "assay").is_unique().all()
    assert table.select(
        "accession",
        "assay",
        f"{group}.ambiguity_members",
        f"{group}.protein_abundance",
        "protein_details.protein_coverage",
        "protein_details.protein_abundance",
        "single_protein.protein_abundance",
    ).rows() == [
        ("P1", "assay[1]", "P1,P2", "10", "0.7", None, None),
        ("P1", "assay[2]", "P1,P2", "20", "0.7", None, None),
        ("P2", "assay[1]", None, None, "0.6", None, None),
        ("P2", "assay[2]", None, None, "0.6", None, None),
        ("P3", "assay[1]", None, None, None, None, "5"),
        ("P3", "assay[2]", None, None, None, None, None),
    ]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            ("PRT", "P2", "null", "0.6", "null", "null", "protein_details"),
            r"repeats accession 'P2' under opt_global_result_type 'protein_details' \(2 rows\)",
        ),
        (
            ("PRT", "P4", "null", "0.1", "1", "null", "null"),
            r"rows without opt_global_result_type: 1",
        ),
    ],
)
def test_protein_rows_a_result_type_cannot_place_are_refused(
    tmp_path: Path, row: tuple[str, ...], message: str
) -> None:
    with pytest.raises(MzTabFormatError, match=message):
        write_long_tables(_write(tmp_path, _PROTEIN_KINDS + _text(row)), tmp_path / "out")
