"""Regenerate the cached-parameter snapshot used by ``test_params_cached_fixtures``.

Run from the repository root::

    .venv/bin/python tests/snapshot_cached_params.py
"""

from __future__ import annotations

import json

from cyclopts import App
from loguru import logger

import test_params_cached_fixtures as cached
from apb2.vendor_params.registry import parse_params

app = App(name="snapshot-cached-params", help=__doc__)


@app.default
def regenerate() -> None:
    """Parse every cached parameter file and write the snapshot."""
    cases = cached.cached_cases()
    if not cases:
        logger.error("no cached parameter files found; nothing to snapshot")
        return
    snapshot = {
        case_id: parse_params(source, slug).model_dump(mode="json")
        for case_id, slug, source in cases
    }
    cached.SNAPSHOT.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"wrote {cached.SNAPSHOT} with {len(snapshot)} cases")


if __name__ == "__main__":
    app()
