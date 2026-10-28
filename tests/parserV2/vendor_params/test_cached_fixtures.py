"""Every cached vendor parameter file still parses to the same record.

The ProteoBench equivalence tests pin twelve curated files field by field. This one is
broader and blunter: it parses every parameter file in the downloaded test-data cache and
compares the whole ``model_dump`` against a committed snapshot, so a change in any parser or
in the schema shows up as a diff rather than as silence. Regenerate deliberately with::

    PYTHONPATH=tests uv run python -m parserV2.vendor_params.snapshot_cached_params

Skipped when the cache is absent, which is the normal state of a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apb2.parserV2.vendor_params.parsers.shared.errors import ParamsError
from apb2.parserV2.vendor_params.registry import (
    ParameterInput,
    get_parser,
    parse_params,
)
from parserV2.fixtures import corpus_root
from parserV2.vendor_params.proteobench_params import apb_software_name

SNAPSHOT = Path(__file__).resolve().parent / "params" / "cached_parameters_snapshot.json"
_ROOT = corpus_root()


def cached_cases() -> list[tuple[str, str, ParameterInput]]:
    """Return ``(case id, parser slug, source)`` for every downloaded parameter file."""
    import csv

    if _ROOT is None:
        return []
    cases: list[tuple[str, str, ParameterInput]] = []
    with (_ROOT / "downloads.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            slug = _slug(row["software_name"])
            source = _ROOT / row["vendor_parameter_file_path"]
            if row.get("status") != "ok" or not _has_parser(slug) or not source.is_file():
                continue
            cases.append((f"{slug}/{row['intermediate_hash']}", slug, source))
    return sorted(cases)


def _slug(software_name: str) -> str:
    """Map a catalog label such as ``FragPipe (DIA-NN quant)`` or ``WOMBAT`` to its parser slug."""
    import re

    return re.sub(r"[^a-z0-9]", "", apb_software_name(software_name.split("(")[0].strip()).lower())


def _has_parser(slug: str) -> bool:
    try:
        get_parser(slug)
    except ParamsError:
        return False
    return True


_CASES = cached_cases()


@pytest.mark.skipif(not _CASES, reason="downloaded test-data cache unavailable")
@pytest.mark.skipif(not SNAPSHOT.exists(), reason="snapshot not generated")
@pytest.mark.parametrize(
    ("case_id", "slug", "source"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_cached_parameter_file_parses_to_the_snapshot(
    case_id: str,
    slug: str,
    source: ParameterInput,
) -> None:
    snapshot: dict[str, object] = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if case_id not in snapshot:
        pytest.skip(f"{case_id} is not in the snapshot")

    assert parse_params(source, slug).model_dump(mode="json") == snapshot[case_id]
