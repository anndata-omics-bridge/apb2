"""Contract tests: orchestration order, level ordering, identity steps, dialects."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from apb2 import configure_parse
from apb2.parse_quant.errors import (
    IncompatibleSourceError,
    NoCompatibleLevelError,
)
from apb2.parse_quant.fragment_exploder import NoFragments
from apb2.parse_quant.parse_strategy import Parser
from apb2.parse_quant.result import ParsedData
from apb2.parse_quant.sources import (
    DelimitedDialect,
    GroupedNumbers,
    InputSource,
    SingleFile,
    UngroupedNumbers,
)
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.model import LongRule


def _empty_result(uns: dict[str, str] | None = None) -> ParsedData:
    return ParsedData(
        X=np.zeros((0, 0)),
        obs=pd.DataFrame(),
        var=pd.DataFrame(),
        uns=uns if uns is not None else {},
        layers={},
    )


def test_parse_runs_injected_strategies_in_pipeline_order() -> None:
    calls: list[str] = []

    class Reader:
        def read(self) -> pd.DataFrame:
            calls.append("read")
            return pd.DataFrame({"a": [1]})

    class Fragments:
        def packed_columns(self) -> tuple[str, ...]:
            return ()

        def explode(self, table: pd.DataFrame) -> pd.DataFrame:
            calls.append("explode")
            return table

    class Columns:
        def prepare_keys(self, table: pd.DataFrame) -> pd.DataFrame:
            calls.append("prepare_keys")
            return table

        def finish(self, result: ParsedData) -> ParsedData:
            calls.append("finish")
            return result

    class Conversion:
        def parse(self, table: pd.DataFrame) -> ParsedData:
            calls.append("convert")
            del table
            return _empty_result()

    parser = Parser(
        level="ion",
        input=Reader(),
        fragments=Fragments(),
        columns=Columns(),
        conversion=Conversion(),
        provenance={"apb": "provenance"},
    )

    result = parser.parse()

    assert calls == ["read", "explode", "prepare_keys", "convert", "finish"]
    assert result.uns == {"apb": "provenance"}


def test_identity_fragments_return_the_table_unchanged() -> None:
    table = pd.DataFrame({"a": [1, 2]})
    assert NoFragments().explode(table) is table
    assert NoFragments().packed_columns() == ()


def _rule(level: str) -> LongRule:
    return cast(
        "LongRule",
        SimpleNamespace(quantification_level=level, software_name="fake"),
    )


def _source() -> InputSource:
    return SingleFile(Path("report.tsv"))


def test_make_parsers_orders_parsers_by_quantification_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[str] = []

    def fake_make_parser(
        rule: LongRule,
        source: InputSource,
        parameters: Parameters | None = None,
        *,
        strict: bool = False,
    ) -> Parser:
        del source, parameters, strict
        built.append(rule.quantification_level)
        return cast("Parser", SimpleNamespace(level=rule.quantification_level))

    monkeypatch.setattr(configure_parse, "make_parse_strategy", fake_make_parser)

    parsers = configure_parse.make_parse_strategies(
        [_rule("protein"), _rule("fragment"), _rule("ion")], _source()
    )

    assert built == ["ion", "protein", "fragment"]
    assert [parser.level for parser in parsers] == ["ion", "protein", "fragment"]


def test_make_parsers_skips_incompatible_rules_and_keeps_compatible_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_make_parser(
        rule: LongRule,
        source: InputSource,
        parameters: Parameters | None = None,
        *,
        strict: bool = False,
    ) -> Parser:
        del source, parameters, strict
        if rule.quantification_level == "protein":
            raise IncompatibleSourceError("protein table role is not bound")
        return cast("Parser", SimpleNamespace(level=rule.quantification_level))

    monkeypatch.setattr(configure_parse, "make_parse_strategy", fake_make_parser)

    parsers = configure_parse.make_parse_strategies([_rule("protein"), _rule("ion")], _source())

    assert [parser.level for parser in parsers] == ["ion"]


def test_make_parsers_raises_when_no_rule_is_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_make_parser(
        rule: LongRule,
        source: InputSource,
        parameters: Parameters | None = None,
        *,
        strict: bool = False,
    ) -> Parser:
        del rule, source, parameters, strict
        raise IncompatibleSourceError("nothing is bound")

    monkeypatch.setattr(configure_parse, "make_parse_strategy", fake_make_parser)

    with pytest.raises(NoCompatibleLevelError, match="ion"):
        configure_parse.make_parse_strategies([_rule("ion")], _source())


def test_grouped_numbers_reject_equal_separators() -> None:
    with pytest.raises(ValueError, match="must differ"):
        GroupedNumbers(decimal=",", thousands=",")


def test_numeric_notations_reject_empty_marks() -> None:
    with pytest.raises(ValueError, match="decimal"):
        UngroupedNumbers(decimal="")
    with pytest.raises(ValueError, match="thousands"):
        GroupedNumbers(decimal=".", thousands="")


def test_dialect_rejects_empty_delimiter() -> None:
    with pytest.raises(ValueError, match="delimiter"):
        DelimitedDialect(delimiter="", numbers=UngroupedNumbers(decimal="."))


def test_notations_declare_their_complete_read_options() -> None:
    assert UngroupedNumbers(decimal=".").read_csv_options() == {"decimal": "."}
    assert GroupedNumbers(decimal=".", thousands=",").read_csv_options() == {
        "decimal": ".",
        "thousands": ",",
    }
