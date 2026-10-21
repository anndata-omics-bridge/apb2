"""Gzipped vendor tables read exactly like the plain files they compress."""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import cast

import pytest
from polars.testing import assert_frame_equal

from apb2.cli.conversion import convert_all_from_rule_config
from apb2.parserV2.compile import ParseRuleCompiler
from apb2.parserV2.parse_quant import delimited_input
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from apb2.parserV2.parse_quant.parameters.source import (
    DelimitedFormatContract,
    Folder,
    LevelReadPlan,
    NumericTextFormat,
)
from apb2.parserV2.parse_quant.parameters.vendor_file import (
    GzipFile,
    PlainFile,
    make_vendor_file,
)
from parserV2.fixtures import committed_dir, level_pairs
from parserV2.join_fixtures import maxquant_tables

RULES = Path("src/apb2/parserV2/vendor_parse_rules/documents")
MAXQUANT_FILES = {
    "evidence": "evidence.txt",
    "peptidoform": "modificationSpecificPeptides.txt",
    "peptide": "peptides.txt",
    "protein": "proteinGroups.txt",
}


def _record(key: str) -> dict[str, object]:
    folder = committed_dir(key)
    assert folder is not None
    return cast("dict[str, object]", json.loads((folder / "expected.json").read_text()))


def _rule_config(key: str) -> Path:
    return next(pair.parser_v2_path for pair, _level in level_pairs() if pair.sample_key == key)


_GZIPPED_SAMPLES = tuple(
    sorted(
        key
        for key in {pair.sample_key for pair, _level in level_pairs()}
        if committed_dir(key) is not None and str(_record(key)["sample"]).endswith(".gz")
    )
)


def test_the_gz_suffix_alone_chooses_how_a_file_is_read(tmp_path: Path) -> None:
    plain = tmp_path / "report.tsv"
    plain.write_text("Sample\tFeature\n", encoding="utf-8")
    packed = tmp_path / "report.tsv.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes()))

    assert make_vendor_file(plain) == PlainFile(plain)
    assert make_vendor_file(packed) == GzipFile(packed)
    for file in (make_vendor_file(plain), make_vendor_file(packed)):
        assert (file.name, file.suffix) == ("report.tsv", ".tsv")
        with file.open() as stream:
            assert stream.read() == b"Sample\tFeature\n"
        with file.open_text("utf-8") as handle:
            assert handle.read() == "Sample\tFeature\n"


def test_a_declared_folder_table_is_found_plain_or_gzipped(tmp_path: Path) -> None:
    folder = Folder(path=tmp_path)
    (tmp_path / "evidence.txt.gz").write_bytes(gzip.compress(b"Raw file\n"))

    assert folder.member("evidence.txt") == GzipFile(tmp_path / "evidence.txt.gz")
    assert folder.member("peptides.txt") is None


def test_a_gzipped_windows_1252_table_resolves_its_encoding_from_decompressed_bytes(
    tmp_path: Path,
) -> None:
    """The encoding probe and the eager single-byte read both see the text, not gzip bytes."""
    path = tmp_path / "table.tsv.gz"
    path.write_bytes(gzip.compress("Sample\tFeature\nCafé\tF\n".encode("cp1252")))
    contract = DelimitedFormatContract(
        extensions=(".tsv",),
        encoding_candidates=("utf8", "windows-1252"),
        quote_char='"',
        delimiter_candidates=("\t",),
        number_format_candidates=(NumericTextFormat(decimal_mark=".", thousands_marks=()),),
    )
    file = make_vendor_file(path)

    evidence = delimited_input.detected_evidence(
        file, contract, lambda header: {"Sample", "Feature"} <= set(header)
    )
    plan = LevelReadPlan(
        projected_columns=("Sample", "Feature"),
        text_sources=frozenset({"Sample", "Feature"}),
        native_numeric_sources=frozenset(),
    )
    frame = delimited_input.make_delimited_reader(file, evidence, plan).read()

    assert evidence.encoding == "windows-1252"
    assert frame.frame.get_column("Sample").to_list() == ["Café"]


@pytest.mark.parametrize("key", _GZIPPED_SAMPLES)
def test_every_gzipped_committed_sample_converts_without_decompression(
    key: str, tmp_path: Path
) -> None:
    """Each stored ``.gz`` sample converts to the levels recorded for its plain table."""
    folder = committed_dir(key)
    assert folder is not None
    record = _record(key)
    companions = [folder / str(name) for name in cast("list[str]", record.get("companions", []))]
    data = folder / str(record["sample"])
    if companions:
        data = tmp_path / "bundle"
        data.mkdir()
        for stored in (folder / str(record["sample"]), *companions):
            shutil.copy(stored, data / stored.name)
    params = record["params"]

    summary = convert_all_from_rule_config(
        data=data,
        output=tmp_path / "converted.h5mu",
        rule_config=_rule_config(key),
        parameters_path=folder / str(params) if params is not None else None,
        software=None,
        checks="standard",
    )

    produced = {
        level.level: {
            "observations": level.observation_count,
            "variables": level.variable_count,
            "layers": list(level.layer_names),
        }
        for level in summary.levels
    }
    assert produced == record["levels"]


def test_a_gzipped_maxquant_folder_converts_like_its_plain_tables(tmp_path: Path) -> None:
    """Folder members, preparation headers and joins all read the ``.gz`` tables."""
    plain, packed = tmp_path / "plain", tmp_path / "packed"
    plain.mkdir()
    packed.mkdir()
    for role, frame in maxquant_tables().items():
        frame.write_csv(plain / MAXQUANT_FILES[role], separator="\t")
        (packed / f"{MAXQUANT_FILES[role]}.gz").write_bytes(
            gzip.compress((plain / MAXQUANT_FILES[role]).read_bytes())
        )

    outputs = {
        name: convert_all_from_rule_config(
            data=folder,
            output=tmp_path / f"{name}.parquet",
            rule_config=RULES / "maxquant_wide/rules.json",
            parameters_path=None,
            software=None,
            checks="standard",
        ).outputs
        for name, folder in (("plain", plain), ("packed", packed))
    }

    assert len(outputs["packed"]) == len(outputs["plain"]) == 2
    for expected_path, packed_path in zip(outputs["plain"], outputs["packed"], strict=True):
        expected = read_parsed_levels(expected_path).levels
        produced = read_parsed_levels(packed_path).levels
        assert list(produced) == list(expected)
        for name, level in produced.items():
            for layer, values in level.layers.items():
                assert_frame_equal(values.values, expected[name].layers[layer].values)
            assert_frame_equal(level.obs.frame, expected[name].obs.frame)


def test_detection_finds_a_gzipped_table_in_a_folder(tmp_path: Path) -> None:
    """Folder detection and vendor recognition see ``report.txt.gz`` as a ``.txt`` table."""
    folder = committed_dir("diann/v1_8")
    assert folder is not None
    record = _record("diann/v1_8")
    shutil.copy(folder / str(record["sample"]), tmp_path / "report.txt.gz")

    parsed = ParseRuleCompiler.detect(tmp_path, folder / str(record["params"])).compile().parse()

    expected = cast("dict[str, dict[str, object]]", record["levels"])
    assert sorted(parsed.levels) == sorted(expected)
    for name, level in parsed.levels.items():
        assert level.obs.frame.height == expected[name]["observations"]
        assert level.var.frame.height == expected[name]["variables"]
