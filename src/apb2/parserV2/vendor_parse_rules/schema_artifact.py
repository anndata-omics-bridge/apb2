"""Write Parser V2's packaged effective-rule JSON Schema artifact."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from loguru import logger

from apb2.parserV2.vendor_parse_rules.schema.rule import rule_json_schema

DOCUMENT_PACKAGE = "apb2.parserV2.vendor_parse_rules.documents"


def artifact_path() -> Path:
    """Where Parser V2 publishes its ``rule.schema.json``."""
    return Path(str(resources.files(DOCUMENT_PACKAGE))) / "_schema" / "rule.schema.json"


def write_artifact() -> Path:
    """Regenerate Parser V2's ``documents/_schema/rule.schema.json``."""
    output = artifact_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rule_json_schema(), indent=2) + "\n")
    logger.info("wrote {}", output)
    return output
