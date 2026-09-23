"""One software hint constrains recognition without conflating producer and quantifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Never

import pytest

from apb2.api import ParseRuleCompiler
from apb2.parserV2 import compile as compilation
from apb2.parserV2 import detect_document as detection
from apb2.parserV2.detect_document import AmbiguousRuleError, RuleUnavailableError
from apb2.parserV2.parse_quant.parameters.source import InputSource
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError
from apb2.parserV2.vendor_params.registry import parse_params
from apb2.parserV2.vendor_parse_rules.document import RuleDocument
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2.fixtures import committed_dir, committed_sample, document_pairs

PARAMS = Path(__file__).parent / "vendor_params" / "params"


def inputs(key: str) -> tuple[Path, Path]:
    folder = committed_dir(key)
    source = committed_sample(key)
    assert folder is not None and source is not None
    record = json.loads((folder / "expected.json").read_text())
    return source, folder / record["params"]


@pytest.mark.parametrize(
    "key",
    [
        "diann/v1_7",
        "diann/v1_8",
        "diann/v2",
        "spectronaut",
        "spectronaut/v21",
        "maxquant",
        "fragpipe",
        "alphapept",
        "peaks",
        "sage",
        "wombat",
        "quantms",
        "i2masschroq",
        "msangel",
        "prolinestudio",
        "alphadia/v1_10",
        "alphadia/v1_12",
        "alphadia/v2",
    ],
)
@pytest.mark.parametrize("hinted", [False, True])
def test_hinted_and_unhinted_vendor_selection(key: str, hinted: bool) -> None:
    source, parameters = inputs(key)
    vendor = key.split("/", 1)[0]
    compiler = ParseRuleCompiler(source, parameters, software=vendor if hinted else None)

    assert compiler.detection.software == vendor
    assert compiler.detection.levels


@pytest.mark.parametrize("suffix", [".txt", ".tsv"])
def test_hint_skips_global_recognition_and_unrelated_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    source, parameters = inputs("diann/v1_8")
    renamed = tmp_path / f"input{suffix}"
    renamed.write_bytes(source.read_bytes())
    parsed_with: list[str] = []
    compiled: list[str] = []
    original_parse = compilation.parse_params
    original_compile = detection.compile_level

    def no_global_scan(_source: InputSource) -> Never:
        raise AssertionError("hinted conversion must not recognize unrelated vendors")

    def record_params(path: Path, software: str) -> Parameters:
        parsed_with.append(software)
        return original_parse(path, software)

    def check_compile(
        facade: ParseRuleFacade,
        source: InputSource,
        checks: Literal["standard", "strict"],
    ) -> Parser:
        assert facade.working_parameters.provenance["software_name"] == "DIA-NN"
        parser = original_compile(facade, source, checks)
        compiled.append(parser.level)
        return parser

    monkeypatch.setattr(compilation, "guess_software", no_global_scan)
    monkeypatch.setattr(compilation, "parse_params", record_params)
    monkeypatch.setattr(detection, "compile_level", check_compile)
    compiler = ParseRuleCompiler(renamed, parameters, software="DIA-NN")
    compiler.compile()
    compiler.compile()

    assert parsed_with == ["diann"]
    assert sorted(compiled) == sorted(selection.level for selection in compiler.detection.levels)


def test_fragpipe_parameters_select_diann_results() -> None:
    source, _parameters = inputs("diann/v1_8")
    compiler = ParseRuleCompiler(source, PARAMS / "fragpipe.workflow", software="FragPipe")

    assert compiler.parameters.software_name == "FragPipe"
    assert compiler.parameters.quantification_software == "DIA-NN"
    assert compiler.detection.software == "diann"
    assert compiler.detection.version == "1.8.2 beta 8"
    assert all("v1_8" in str(item.document.path) for item in compiler.detection.levels)


def test_unhinted_recognition_visits_every_packaged_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    source, parameters = inputs("diann/v1_8")
    visited: set[str] = set()
    original = ParseRuleFacade.from_declared_rule

    def record_document(
        cls: type[ParseRuleFacade], document: RuleDocument, level: QuantificationLevel
    ) -> ParseRuleFacade:
        visited.add(document.software_name)
        return original(document, level)

    monkeypatch.setattr(ParseRuleFacade, "from_declared_rule", classmethod(record_document))
    ParseRuleCompiler(source, parameters)

    assert visited == {
        document.software_name
        for path in detection.PACKAGED
        if (document := load_rule_document(path)).parameter_file == "required"
    }


def test_unhinted_compound_parameters_request_the_grammar() -> None:
    source, _parameters = inputs("diann/v1_8")
    with pytest.raises(ParamsError, match=r"cannot parse .* as diann: .*--software"):
        ParseRuleCompiler(source, PARAMS / "fragpipe.workflow")


def test_mismatched_hint_never_falls_back_to_global_recognition() -> None:
    source, _parameters = inputs("maxquant")
    with pytest.raises(RuleUnavailableError, match=r"candidate vendors: \['diann', 'fragpipe'\]"):
        ParseRuleCompiler(source, PARAMS / "fragpipe.workflow", software="fragpipe")


@pytest.mark.parametrize("hint", [None, "fragpipe"])
def test_ambiguous_vendors_remain_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hint: str | None
) -> None:
    pair = next(pair for pair in document_pairs() if pair.key == "diann/v1_8")
    raw = json.loads(pair.parser_v2_path.read_text())
    raw["software_name"] = "FragPipe"
    raw["software_version_pattern"] = ".*"
    duplicate = tmp_path / "rules.json"
    duplicate.write_text(json.dumps(raw))
    monkeypatch.setattr(detection, "PACKAGED", (pair.parser_v2_path, duplicate))
    source, _parameters = inputs("diann/v1_8")

    with pytest.raises(AmbiguousRuleError, match="several packaged"):
        ParseRuleCompiler(source, PARAMS / "fragpipe.workflow", software=hint)


def test_unrecognized_input_has_an_actionable_error(tmp_path: Path) -> None:
    source = tmp_path / "unknown.tsv"
    source.write_text("unrecognized\tcolumns\n1\t2\n")
    with pytest.raises(RuleUnavailableError, match=r"could not recognize.*--software"):
        ParseRuleCompiler(source, PARAMS / "fragpipe.workflow")


def test_hint_does_not_bypass_version_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    source, parameters = inputs("diann/v1_8")

    def future_parameters(path: Path, software: str) -> Parameters:
        return parse_params(path, software).model_copy(update={"software_version": "99.0"})

    monkeypatch.setattr(compilation, "parse_params", future_parameters)
    with pytest.raises(RuleUnavailableError, match=r"candidate vendors.*diann"):
        ParseRuleCompiler(source, parameters, software="diann")


def test_hint_does_not_bypass_level_checks() -> None:
    source, parameters = inputs("maxquant")
    with pytest.raises(RuleUnavailableError, match="requested level 'protein' is unavailable"):
        ParseRuleCompiler(source, parameters, software="maxquant", requested_levels=("protein",))


def test_hint_preserves_malformed_named_table_diagnostics(tmp_path: Path) -> None:
    _source, parameters = inputs("maxquant")
    source = tmp_path / "evidence.txt"
    source.write_text("wrong\tcolumns\n1\t2\n")
    with pytest.raises(RuleUnavailableError, match="are present but incompatible"):
        ParseRuleCompiler(source, parameters, software="maxquant", requested_levels=("ion",))
