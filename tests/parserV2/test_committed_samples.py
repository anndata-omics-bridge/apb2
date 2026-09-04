"""Every packaged rule converts its committed sample to the recorded expectations.

The artifacts under ``tests/parserV2/data/<rule_key>/`` are real vendor-export excerpts, created
once by the workspace scripts (``apb_studio/scripts/make_apb2_test_samples.py`` and its
``extend_directflq_benchmark`` twin) and append-only thereafter. They make end-to-end conversion
testable everywhere, including CI; the downloaded corpus stays the ground truth where present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from apb2.parserV2.conversion_facade import convert_all_from_rule_config
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED
from parserV2.fixtures import DATA_DIR, committed_dir, committed_sample
from parserV2.rule_inventory import document_key

_KEYS = tuple(sorted(document_key(path) for path in PACKAGED))


def test_every_packaged_document_has_committed_artifacts() -> None:
    """A new rule document must ship its header, sample, and expectations."""
    missing = [key for key in _KEYS if committed_dir(key) is None]
    assert missing == []


@pytest.mark.parametrize("key", _KEYS)
def test_committed_sample_converts_to_the_recorded_expectations(key: str, tmp_path: Path) -> None:
    folder = committed_dir(key)
    assert folder is not None
    record = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    sample = committed_sample(key)
    assert sample is not None
    params = record["params"]

    summary = convert_all_from_rule_config(
        data=sample,
        output=tmp_path / "converted",
        rule_config=_rule_config(key),
        parameters_path=folder / str(params) if params is not None else None,
        parameters_software=None,
        checks="standard",
    )

    produced = {
        level.level: {
            "observations": level.observation_count,
            "variables": level.variable_count,
            "layers": list(level.layer_names),
        }
        for level in summary.levels
    }
    assert produced == cast("dict[str, object]", record["levels"])


def test_committed_folders_all_belong_to_packaged_documents() -> None:
    """No orphaned artifact folder survives a rule document's removal or rename."""
    found = {
        path.parent.relative_to(DATA_DIR).as_posix() for path in DATA_DIR.rglob("expected.json")
    }
    assert found <= set(_KEYS)


def _rule_config(key: str) -> Path:
    return next(path for path in PACKAGED if document_key(path) == key)
