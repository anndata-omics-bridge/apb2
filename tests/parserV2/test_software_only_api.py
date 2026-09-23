"""A known result producer can be parsed without vendor search-parameter files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from polars.testing import assert_frame_equal

from apb2.api import ParseRuleCompiler, QuantificationLevel
from apb2.parserV2 import detect_document as detection
from apb2.parserV2.detect_document import AmbiguousRuleError, RuleUnavailableError
from parserV2.fixtures import committed_dir, committed_sample, document_pairs


def _sample(key: str) -> Path:
    source = committed_sample(key)
    assert source is not None
    return source


def test_software_only_api_preserves_parsed_values_and_rule_provenance() -> None:
    source = _sample("fragpipe")
    folder = committed_dir("fragpipe")
    assert folder is not None
    record = json.loads((folder / "expected.json").read_text())
    with_parameters = ParseRuleCompiler(
        source, folder / record["params"], software="fragpipe", requested_levels=("ion",)
    )
    without_parameters = ParseRuleCompiler.from_software(
        source, software="fragpipe", requested_levels=("ion",)
    )

    expected = with_parameters.compile().parse().levels["ion"]
    actual = without_parameters.compile().parse().levels["ion"]
    assert actual.obs.frame.height == record["levels"]["ion"]["observations"]
    assert actual.var.frame.height == record["levels"]["ion"]["variables"]
    assert_frame_equal(actual.obs.frame, expected.obs.frame)
    assert_frame_equal(actual.var.frame, expected.var.frame)
    assert actual.primary_layer_name == expected.primary_layer_name
    assert actual.uns == expected.uns
    assert actual.layers.keys() == expected.layers.keys()
    for name, layer in actual.layers.items():
        assert_frame_equal(layer.values, expected.layers[name].values)
    assert without_parameters.detection.version is None
    with pytest.raises(ValueError, match="no vendor parameter file"):
        _ = without_parameters.parameters
    # The constructor used by the CLI retains its existing parameter requirement.
    with pytest.raises(RuleUnavailableError):
        ParseRuleCompiler(source, None, software="fragpipe", requested_levels=("ion",))


def test_software_only_api_accepts_pb_custom() -> None:
    parsed = (
        ParseRuleCompiler.from_software(_sample("pb_custom"), software="pb_custom")
        .compile()
        .parse()
    )
    assert set(parsed.levels) == {"ion"}
    assert parsed.levels["ion"].var.frame.height == 8493


@pytest.mark.parametrize("software", ["", "unrecognized", "fragpipe"])
def test_software_hint_must_identify_the_result_producer(software: str) -> None:
    with pytest.raises(ValueError, match="software"):
        ParseRuleCompiler.from_software(_sample("diann/v1_8"), software=software)


def test_overlapping_versions_remain_ambiguous() -> None:
    with pytest.raises(AmbiguousRuleError, match="several packaged documents"):
        ParseRuleCompiler.from_software(
            _sample("diann/v1_8"), software="diann", requested_levels=("ion",)
        )


@pytest.mark.parametrize("level", ["ion", "peptidoform"])
def test_requested_level_does_not_invent_sage_charge_evidence(level: QuantificationLevel) -> None:
    with pytest.raises(RuleUnavailableError, match="combine_charge_states"):
        ParseRuleCompiler.from_software(_sample("sage"), software="sage", requested_levels=(level,))


def test_missing_acquisition_evidence_never_falls_back_to_an_older_diann_rule() -> None:
    with pytest.raises(RuleUnavailableError, match="acquisition_method"):
        ParseRuleCompiler.from_software(
            _sample("diann/v2"), software="diann", requested_levels=("ion",)
        )


def test_only_requested_levels_require_search_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    document = next(pair.parser_v2_path for pair in document_pairs() if pair.key == "diann/v2")
    monkeypatch.setattr(detection, "PACKAGED", (document,))

    parsed = (
        ParseRuleCompiler.from_software(
            _sample("diann/v2"), software="diann", requested_levels=("protein",)
        )
        .compile()
        .parse()
    )
    assert set(parsed.levels) == {"protein"}
    assert parsed.levels["protein"].var.frame.height > 0
    with pytest.raises(RuleUnavailableError, match="acquisition_method"):
        ParseRuleCompiler.from_software(
            _sample("diann/v2"), software="diann", requested_levels=("ion",)
        )
