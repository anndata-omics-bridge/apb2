"""Public parameter-parser registry contract tests."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import assert_type

import pytest

from apb2.parserV2.vendor_params.parsers.maxquant import extract_params as extract_maxquant
from apb2.parserV2.vendor_params.parsers.shared.model import ParamsError
from apb2.parserV2.vendor_params.registry import (
    ParseFn,
    available_software,
    get_parser,
    parse_params,
)

PARAMS = Path(__file__).resolve().parent / "params"
METAMORPHEUS_TOML = PARAMS / "metamorpheus_search_task_config.toml"
METAMORPHEUS_VERSION = PARAMS / "metamorpheus_version_result.txt"
MAXQUANT_XML = PARAMS / "mqpar_MQ2.1.3.0_noMBR.xml"
WOMBAT_YAML = PARAMS / "wombat_params.yaml"


def test_every_registered_parser_has_the_uniform_callable_signature() -> None:
    for software in available_software():
        parser = get_parser(software)
        assert_type(parser, ParseFn)
        parameters = tuple(inspect.signature(parser).parameters.values())
        assert len(parameters) == 1, software
        assert parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY


def test_registry_preserves_single_source_calls() -> None:
    parsed = parse_params(WOMBAT_YAML, "wombat")

    assert parsed.software_name == "Wombat"


def test_registry_accepts_explicit_metamorpheus_source_pair() -> None:
    parsed = parse_params(
        (METAMORPHEUS_TOML, METAMORPHEUS_VERSION),
        "metamorpheus",
    )

    assert parsed.software_name == "MetaMorpheus"
    assert parsed.software_version == "1.1.1"


def test_registry_rejects_incomplete_metamorpheus_input_cleanly() -> None:
    with pytest.raises(
        ParamsError,
        match=r"MetaMorpheus requires exactly 2 parameter sources; received 1",
    ):
        parse_params(METAMORPHEUS_TOML, "metamorpheus")


def test_maxquant_keeps_parser_specific_ms2frac_option() -> None:
    ftms = extract_maxquant(MAXQUANT_XML)
    itms = extract_maxquant(MAXQUANT_XML, ms2frac="ITMS")

    assert ftms.fragment_mass_tolerance is not None
    assert ftms.fragment_mass_tolerance.value == 20
    assert ftms.fragment_mass_tolerance.unit == "ppm"
    assert itms.fragment_mass_tolerance is not None
    assert itms.fragment_mass_tolerance.value == 0.5
