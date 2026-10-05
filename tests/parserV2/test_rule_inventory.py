"""The packaged inventory counts: 21 packaged documents, 41 effective levels."""

from __future__ import annotations

from parserV2.rule_inventory import (
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_LEVEL_COUNT,
    packaged_levels,
)


def test_parser_v2_packages_twenty_two_documents_and_forty_two_levels() -> None:
    levels = packaged_levels()

    assert len({key for key, _level in levels}) == EXPECTED_DOCUMENT_COUNT
    assert len(levels) == EXPECTED_LEVEL_COUNT
    assert len(set(levels)) == EXPECTED_LEVEL_COUNT
