"""The Parser V2 application boundary used by ``apb2 convert``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Never

import anndata
import mudata
import pytest

from apb2.parserV2 import conversion_facade
from apb2.parserV2 import detect_document as detection_module
from apb2.parserV2.conversion_facade import (
    ConversionError,
    convert_all_from_packaged_rules,
    convert_from_packaged_rules,
)
from apb2.parserV2.detect_document import (
    AmbiguousRuleError,
    detect_rule_document,
    detect_rule_documents,
)
from apb2.parserV2.detect_document import guess_software as guess_packaged_software
from apb2.parserV2.parse_quant import delimited_input
from apb2.parserV2.parse_quant.io import formats
from apb2.parserV2.parse_quant.io.anndata_writer import NAMESPACE, PARSE_NAMESPACE
from apb2.parserV2.parse_quant.io.json_representation import sidecar_path
from apb2.parserV2.parse_quant.parameters.source import Folder, SingleFile
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, QuantificationLevel
from parserV2.fixtures import PackagedDocument, committed_dir, committed_sample, document_pairs
from parserV2.join_fixtures import maxquant_tables


class _StopAfterParserSelection(Exception):
    """End a precedence test immediately after the parameter parser is selected."""


def _diann_v2() -> PackagedDocument:
    return next(pair for pair in document_pairs() if pair.key == "diann/v2")


def _parameter_file(pair: PackagedDocument) -> Path:
    data = pair.required_data_path()
    (parameters,) = sorted(data.parent.glob("param_0.*"))
    return parameters


_MAXQUANT_TABLES: tuple[tuple[QuantificationLevel, str, str], ...] = (
    ("ion", "maxquant", "evidence.txt"),
    (
        "peptidoform",
        "maxquant_modificationspecificpeptides",
        "modificationSpecificPeptides.txt",
    ),
    ("peptide", "maxquant_peptides", "peptides.txt"),
    ("protein", "maxquant_proteingroups", "proteinGroups.txt"),
)


def _maxquant_parameters() -> Path:
    folder = committed_dir("maxquant")
    assert folder is not None
    return folder / "param_0..xml"


def _maxquant_folder(
    destination: Path,
    levels: tuple[QuantificationLevel, ...],
) -> Path:
    destination.mkdir()
    tables = maxquant_tables()
    for level, _fixture, file_name in _MAXQUANT_TABLES:
        if level not in levels:
            continue
        role = "evidence" if level == "ion" else level
        tables[role].write_csv(destination / file_name, separator="\t")
    return destination


def test_packaged_conversion_detects_parses_and_writes_with_provenance(tmp_path: Path) -> None:
    pair = _diann_v2()
    data = pair.required_data_path()
    parameters_path = _parameter_file(pair)
    target = tmp_path / "protein.h5ad"

    result = convert_from_packaged_rules(
        data=data,
        level="protein",
        output=target,
        parameters_path=parameters_path,
        software=None,
        parameters_software=None,
        checks="standard",
    )

    assert result.software == "diann"
    assert target.is_file()
    stored = anndata.read_h5ad(target)
    namespace = stored.uns[NAMESPACE][PARSE_NAMESPACE]
    expected = parse_params(parameters_path, software="diann").model_dump(mode="json")
    assert json.loads(str(namespace["search_parameters"])) == expected
    assert namespace["search_parameters_path"] == str(parameters_path)
    assert namespace["rule_selection_method"] in {"software_version", "columns"}
    representation = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert representation["levels"][0]["name"] == "protein"
    assert representation["levels"][0]["uns"]["search_parameters_path"] == parameters_path.name


def test_packaged_conversion_without_a_level_writes_every_compatible_modality(
    tmp_path: Path,
) -> None:
    pair = _diann_v2()
    data = pair.required_data_path()
    parameters_path = _parameter_file(pair)
    target = tmp_path / "all.h5mu"

    result = convert_all_from_packaged_rules(
        data=data,
        output=target,
        parameters_path=parameters_path,
        software=None,
        parameters_software=None,
        checks="standard",
    )

    stored = mudata.read_h5mu(target)
    assert list(stored.mod) == [summary.level for summary in result.levels]
    assert set(stored.mod) >= {"ion", "protein"}
    namespace = stored.uns[NAMESPACE][PARSE_NAMESPACE]
    assert list(namespace["quantification_levels"]) == list(stored.mod)
    assert namespace["rule_selection_method"] in {"software_version", "columns"}
    assert all(
        modality.uns[NAMESPACE][PARSE_NAMESPACE]["search_parameters_path"] == str(parameters_path)
        for modality in stored.mod.values()
    )
    representation = json.loads(sidecar_path(target).read_text(encoding="utf-8"))
    assert [level["name"] for level in representation["levels"]] == list(stored.mod)


@pytest.mark.parametrize(
    "suffix",
    (".h5mu", ".parquet", ".duckdb"),
)
def test_maxquant_folder_converts_all_tables_in_canonical_order(
    tmp_path: Path,
    suffix: str,
) -> None:
    folder = _maxquant_folder(
        tmp_path / "maxquant",
        ("ion", "peptidoform", "peptide", "protein"),
    )
    target = tmp_path / f"all{suffix}"

    result = convert_all_from_packaged_rules(
        data=folder,
        output=target,
        parameters_path=_maxquant_parameters(),
        software=None,
        parameters_software=None,
        checks="standard",
    )

    expected = ["ion", "peptidoform", "peptide", "protein"]
    assert [summary.level for summary in result.levels] == expected
    assert result.outputs == (
        target.with_name(f"all.raw_file{suffix}"),
        target.with_name(f"all.experiment{suffix}"),
    )
    assert not target.exists()
    assert [
        name for path in result.outputs for name in formats.read_parsed_levels(path).levels
    ] == expected
    for path in result.outputs:
        assert path.is_dir() if suffix == ".parquet" else path.is_file()
        assert sidecar_path(path).is_file()


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        (("ion", "peptide", "protein"), ("ion", "peptide", "protein")),
        (("ion", "peptide"), ("ion", "peptide")),
        (("ion", "protein"), ("ion", "protein")),
        (("ion",), ("ion",)),
    ),
)
def test_maxquant_folder_detection_skips_missing_tables(
    tmp_path: Path,
    available: tuple[QuantificationLevel, ...],
    expected: tuple[QuantificationLevel, ...],
) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", available)
    parameters = parse_params(_maxquant_parameters(), software="maxquant")

    detected = detect_rule_documents(parameters, Folder(path=folder), LEVELS)

    assert tuple(selection.level for selection in detected.levels) == expected


def test_maxquant_folder_explicit_level_decomposes_only_requested_level(tmp_path: Path) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", ("ion", "peptide", "protein"))
    target = tmp_path / "peptide.h5ad"

    result = convert_from_packaged_rules(
        data=folder,
        level="peptide",
        output=target,
        parameters_path=_maxquant_parameters(),
        software=None,
        parameters_software=None,
        checks="standard",
    )

    assert [summary.level for summary in result.levels] == ["peptide"]
    assert list(formats.read_parsed_levels(target).levels) == ["peptide"]


def test_maxquant_folder_missing_explicit_level_names_expected_table(tmp_path: Path) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", ("ion",))
    target = tmp_path / "peptide.h5ad"

    with pytest.raises(ConversionError, match="level 'peptide' is unavailable"):
        convert_from_packaged_rules(
            data=folder,
            level="peptide",
            output=target,
            parameters_path=_maxquant_parameters(),
            software=None,
            parameters_software=None,
            checks="standard",
        )

    assert not target.exists()
    assert not sidecar_path(target).exists()


def test_maxquant_folder_malformed_present_table_aborts_before_write(tmp_path: Path) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", ("ion",))
    (folder / "peptides.txt").write_text("not\ta\tMaxQuant\theader\n", encoding="utf-8")
    target = tmp_path / "all.h5mu"

    with pytest.raises(ConversionError, match=r"peptides\.txt lacks the columns"):
        convert_all_from_packaged_rules(
            data=folder,
            output=target,
            parameters_path=_maxquant_parameters(),
            software=None,
            parameters_software=None,
            checks="standard",
        )

    assert not target.exists()
    assert not sidecar_path(target).exists()


def test_maxquant_folder_per_level_parse_failure_aborts_before_write(tmp_path: Path) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", ("ion", "peptide"))
    peptide_table = folder / "peptides.txt"
    lines = peptide_table.read_text(encoding="utf-8").splitlines()
    peptide_table.write_text("\n".join((*lines, lines[1], "")), encoding="utf-8")
    target = tmp_path / "all.h5mu"

    with pytest.raises(ConversionError, match="duplicate row IDs"):
        convert_all_from_packaged_rules(
            data=folder,
            output=target,
            parameters_path=_maxquant_parameters(),
            software=None,
            parameters_software=None,
            checks="standard",
        )

    assert not target.exists()
    assert not sidecar_path(target).exists()


def test_explicit_rule_config_binds_its_one_document_from_a_folder(tmp_path: Path) -> None:
    folder = _maxquant_folder(tmp_path / "maxquant", ("ion", "peptide"))
    target = tmp_path / "ion.h5ad"
    document = next(pair for pair in document_pairs() if pair.key == "maxquant")

    result = conversion_facade.convert_from_rule_config(
        data=folder,
        level="ion",
        output=target,
        rule_config=document.parser_v2_path,
        parameters_path=None,
        parameters_software=None,
        checks="standard",
    )

    assert [summary.level for summary in result.levels] == ["ion"]
    assert list(formats.read_parsed_levels(target).levels) == ["ion"]


@pytest.mark.parametrize(
    ("parameters_software", "software", "inferred", "expected"),
    (
        ("params-choice", "software-choice", "source-choice", "params-choice"),
        (None, "software-choice", "source-choice", "software-choice"),
        (None, None, "source-choice", "source-choice"),
    ),
)
def test_parameter_parser_selection_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters_software: str | None,
    software: str | None,
    inferred: str,
    expected: str,
) -> None:
    selected: list[str] = []

    def infer(_source: object) -> str:
        return inferred

    def stop_after_selection(_path: object, software: str) -> Parameters:
        selected.append(software)
        raise _StopAfterParserSelection

    monkeypatch.setattr(conversion_facade, "guess_software", infer)
    monkeypatch.setattr(conversion_facade, "parse_params", stop_after_selection)

    with pytest.raises(_StopAfterParserSelection):
        convert_from_packaged_rules(
            data=tmp_path / "source.tsv",
            level="ion",
            output=tmp_path / "out.h5ad",
            parameters_path=tmp_path / "parameters.txt",
            software=software,
            parameters_software=parameters_software,
            checks="standard",
        )

    assert selected == [expected]


def test_expected_subsystem_failure_becomes_one_conversion_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parse_and_write(**_arguments: object) -> Never:
        raise OSError("cannot write target")

    monkeypatch.setattr(conversion_facade, "_parse_document_and_write", fail_parse_and_write)

    with pytest.raises(ConversionError, match="cannot write target"):
        conversion_facade.convert_from_rule_config(
            data=tmp_path / "source.tsv",
            level="ion",
            output=tmp_path / "out.h5ad",
            rule_config=_diann_v2().parser_v2_path,
            parameters_path=None,
            parameters_software=None,
            checks="standard",
        )


def test_unexpected_subsystem_failure_remains_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parse_and_write(**_arguments: object) -> Never:
        raise RuntimeError("implementation defect")

    monkeypatch.setattr(conversion_facade, "_parse_document_and_write", fail_parse_and_write)

    with pytest.raises(RuntimeError, match="implementation defect"):
        conversion_facade.convert_from_rule_config(
            data=tmp_path / "source.tsv",
            level="ion",
            output=tmp_path / "out.h5ad",
            rule_config=_diann_v2().parser_v2_path,
            parameters_path=None,
            parameters_software=None,
            checks="standard",
        )


def test_duplicate_packaged_matches_are_reported_as_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _diann_v2()
    data = pair.required_data_path()
    document = load_rule_document(pair.parser_v2_path)
    parameters = parse_params(_parameter_file(pair), software="diann")

    def duplicate_document() -> tuple[object, ...]:
        return (document, document)

    monkeypatch.setattr(detection_module, "_packaged_documents", duplicate_document)

    with pytest.raises(AmbiguousRuleError, match="several packaged documents"):
        detect_rule_document(parameters, SingleFile(path=data))


@pytest.mark.parametrize(
    ("key", "version"),
    (
        ("diann/v1_7", "1.0.0"),
        ("diann/v1_7", "1.7.16"),
        ("diann/v1_8", "1.8.0"),
        ("diann/v1_8", "1.9.2"),
    ),
)
def test_diann_v1_rule_ranges_are_disjoint(key: str, version: str) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == key)
    fixture_dir = committed_dir(key)
    data = committed_sample(key)
    assert fixture_dir is not None
    assert data is not None
    parameters = parse_params(fixture_dir / "param_0..txt", software="diann").model_copy(
        update={"software_version": version}
    )

    detected = detect_rule_document(parameters, SingleFile(path=data))

    assert detected.document.path == pair.parser_v2_path


@pytest.mark.parametrize("suffix", (".txt", ".tsv"))
def test_diann_v1_9_delimited_extensions_remain_accepted(
    tmp_path: Path,
    suffix: str,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "diann/v1_8")
    fixture_dir = committed_dir("diann/v1_8")
    source = committed_sample("diann/v1_8")
    assert fixture_dir is not None
    assert source is not None
    data = tmp_path / f"input{suffix}"
    shutil.copyfile(source, data)
    parameters = parse_params(fixture_dir / "param_0..txt", software="diann")

    detected = detect_rule_document(parameters, SingleFile(path=data))

    assert detected.document.path == pair.parser_v2_path


def test_diann_v1_9_parquet_produces_available_levels(tmp_path: Path) -> None:
    fixture_dir = committed_dir("diann/v1_8")
    data = committed_sample("diann/v2")
    assert fixture_dir is not None
    assert data is not None
    target = tmp_path / "converted.h5mu"

    result = convert_all_from_packaged_rules(
        data=data,
        output=target,
        parameters_path=fixture_dir / "param_0..txt",
        software=None,
        parameters_software="diann",
        checks="standard",
    )

    assert [level.level for level in result.levels] == ["ion", "protein"]
    assert "Ms1_Normalised" in result.levels[0].layer_names
    assert list(formats.read_parsed_levels(target).levels) == ["ion", "protein"]


def test_spectronaut_v21_selects_its_dedicated_rule() -> None:
    key = "spectronaut/v21"
    pair = next(candidate for candidate in document_pairs() if candidate.key == key)
    fixture_dir = committed_dir(key)
    data = committed_sample(key)
    assert fixture_dir is not None
    assert data is not None
    parameters = parse_params(fixture_dir / "param_0..txt", software="spectronaut")

    detected = detect_rule_document(parameters, SingleFile(path=data))

    assert detected.document.path == pair.parser_v2_path


def test_source_only_rule_recognition_never_inspects_data_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    data = pair.required_data_path()

    def refuse_row_inspection(*_arguments: object) -> Never:
        raise AssertionError("rule recognition may inspect only source metadata and headers")

    monkeypatch.setattr(delimited_input, "_resolved_number_format", refuse_row_inspection)

    assert guess_packaged_software(SingleFile(path=data)) == "spectronaut"
