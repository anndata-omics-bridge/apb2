"""Acquisition-method defaults shared by parameter parsers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from apb2.parserV2.search_parameters import (
    alphapept,
    fragpipe,
    maxquant,
    metamorpheus,
    msaid,
    peaks,
    sage,
    spectronaut,
    wombat,
)
from apb2.parserV2.search_parameters.model import Parameters

PROTEOBENCH_PARAMS = Path(__file__).resolve().parent / "params"
Parser = Callable[..., Parameters]

CASES: list[tuple[str, Parser, tuple[Path, ...]]] = [
    ("AlphaPept", alphapept.extract_params, (PROTEOBENCH_PARAMS / "alphapept_0.4.9.yaml",)),
    ("FragPipe", fragpipe.extract_params, (PROTEOBENCH_PARAMS / "fragpipe.workflow",)),
    (
        "MaxQuant",
        maxquant.extract_params,
        (PROTEOBENCH_PARAMS / "mqpar_MQ2.1.3.0_noMBR.xml",),
    ),
    (
        "MetaMorpheus",
        metamorpheus.extract_params,
        (
            PROTEOBENCH_PARAMS / "metamorpheus_search_task_config.toml",
            PROTEOBENCH_PARAMS / "metamorpheus_version_result.txt",
        ),
    ),
    ("MSAID", msaid.extract_params, (PROTEOBENCH_PARAMS / "MSAID_default_params.csv",)),
    ("PEAKS", peaks.extract_params, (PROTEOBENCH_PARAMS / "PEAKS_parameters.txt",)),
    ("Sage", sage.extract_params, (PROTEOBENCH_PARAMS / "sage_parameterfile.json",)),
    (
        "Spectronaut",
        spectronaut.extract_params,
        (PROTEOBENCH_PARAMS / "Spectronaut_static.txt",),
    ),
    ("WOMBAT", wombat.extract_params, (PROTEOBENCH_PARAMS / "wombat_params.yaml",)),
]


@pytest.mark.parametrize(
    ("software", "parser", "sources"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_non_diann_parsers_default_acquisition_method_to_unknown(
    software: str,
    parser: Parser,
    sources: tuple[Path, ...],
) -> None:
    missing = [source for source in sources if not source.exists()]
    if missing:
        pytest.skip(f"{software} fixture missing: {missing}")

    assert parser(*sources).acquisition_method == "unknown"
