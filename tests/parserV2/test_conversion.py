"""The Parser V2 application boundary used by ``apb2 convert``."""

from __future__ import annotations

import json
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
from apb2.parserV2.detect_document import AmbiguousRuleError, detect_rule_document
from apb2.parserV2.detect_document import guess_software as guess_packaged_software
from apb2.parserV2.parse_quant import delimited_input
from apb2.parserV2.parse_quant.io.anndata_writer import NAMESPACE, PARSE_NAMESPACE
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from parserV2.fixtures import DocumentPair, document_pairs


class _StopAfterParserSelection(Exception):
    """End a precedence test immediately after the parameter parser is selected."""


def _diann_v2() -> DocumentPair:
    return next(pair for pair in document_pairs() if pair.key == "diann/v2")


def _parameter_file(pair: DocumentPair) -> Path:
    data = pair.data_path()
    assert data is not None
    (parameters,) = sorted(data.parent.glob("param_0.*"))
    return parameters


def test_packaged_conversion_detects_parses_and_writes_with_provenance(tmp_path: Path) -> None:
    pair = _diann_v2()
    data = pair.data_path()
    assert data is not None
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


def test_packaged_conversion_without_a_level_writes_every_compatible_modality(
    tmp_path: Path,
) -> None:
    pair = _diann_v2()
    data = pair.data_path()
    assert data is not None
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

    monkeypatch.setattr(conversion_facade, "_parse_and_write", fail_parse_and_write)

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

    monkeypatch.setattr(conversion_facade, "_parse_and_write", fail_parse_and_write)

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
    data = pair.data_path()
    assert data is not None
    document = load_rule_document(pair.parser_v2_path)
    parameters = parse_params(_parameter_file(pair), software="diann")

    def duplicate_document() -> tuple[object, ...]:
        return (document, document)

    monkeypatch.setattr(detection_module, "_packaged_documents", duplicate_document)

    with pytest.raises(AmbiguousRuleError, match="several packaged documents"):
        detect_rule_document(parameters, SingleFile(path=data))


def test_source_only_rule_recognition_never_inspects_data_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = next(candidate for candidate in document_pairs() if candidate.key == "spectronaut")
    data = pair.data_path()
    assert data is not None

    def refuse_row_inspection(*_arguments: object) -> Never:
        raise AssertionError("rule recognition may inspect only source metadata and headers")

    monkeypatch.setattr(delimited_input, "_resolved_number_format", refuse_row_inspection)

    assert guess_packaged_software(SingleFile(path=data)) == "spectronaut"
