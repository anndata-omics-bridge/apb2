"""quantms versions-YAML parameter parser."""

from __future__ import annotations

from pathlib import Path
from typing import IO, cast

import yaml

from apb2.parserV2.vendor_params.parsers.shared.common import read_text
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Read the workflow and search-engine identities recorded by quantms."""
    payload = yaml.safe_load(read_text(source))
    if not isinstance(payload, dict) or not isinstance(payload.get("Workflow"), dict):
        raise ParamsError("not a quantms versions file")
    workflow = cast("dict[str, object]", payload["Workflow"])
    version = workflow.get("bigbio/quantms") or workflow.get("nf-core/quantms")
    if not isinstance(version, str):
        raise ParamsError("quantms workflow version is missing")
    engines: list[str] = []
    engine_versions: list[str] = []
    for name, raw in payload.items():
        if not isinstance(name, str) or not name.startswith("SEARCHENGINE"):
            continue
        engine = name.removeprefix("SEARCHENGINE").lower()
        engines.append(engine)
        if isinstance(raw, dict):
            values = cast("dict[str, object]", raw)
            candidate = values.get("Comet") or values.get("msgf_plus")
            if candidate is not None:
                engine_versions.append(str(candidate))
    return Parameters(
        software_name="quantms",
        software_version=version,
        quantification_software="quantms",
        quantification_software_version=version,
        acquisition_method="DDA",
        search_engine=",".join(engines) or None,
        search_engine_version=",".join(engine_versions) or None,
    )
