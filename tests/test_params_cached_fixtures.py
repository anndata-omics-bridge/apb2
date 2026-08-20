"""Every cached vendor parameter file still parses to the same record.

The ProteoBench equivalence tests pin twelve curated files field by field. This one is
broader and blunter: it parses every parameter file in the downloaded test-data cache and
compares the whole ``model_dump`` against a committed snapshot, so a change in any parser or
in the schema shows up as a diff rather than as silence. Regenerate deliberately with::

    uv run python -m tests.snapshot_cached_params

Skipped when the cache is absent, which is the normal state of a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apb2.vendor_params.registry import ParameterInput, available_software, parse_params

SNAPSHOT = Path(__file__).resolve().parent / "params" / "cached_parameters_snapshot.json"
CACHE_INDEX = (
    Path(__file__).resolve().parents[2]
    / "apb"
    / "test_data_download"
    / "raw_file_db_downloaded.csv"
)


def cached_cases() -> list[tuple[str, str, ParameterInput]]:
    """Return ``(case id, parser slug, source)`` for every cached parameter file."""
    import csv

    if not CACHE_INDEX.exists():
        return []
    root = CACHE_INDEX.parent / "json_dir"
    cases: list[tuple[str, str, ParameterInput]] = []
    with CACHE_INDEX.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            slug = _slug(row["software_name"])
            if slug not in available_software():
                continue
            directory = root / Path(row["input_file_path"]).parent
            sources = sorted(directory.glob("param_0.*"))
            if not sources:
                continue
            cases.append((f"{slug}/{directory.name[:8]}", slug, sources[0]))
    return sorted(cases)


def _slug(software_name: str) -> str:
    """Map a catalog label such as ``FragPipe (DIA-NN quant)`` to its parser slug."""
    import re

    return re.sub(r"[^a-z0-9]", "", software_name.split("(")[0].strip().lower())


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
