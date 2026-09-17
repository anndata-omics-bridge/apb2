"""Write Parser V2's authored-document and effective-rule JSON Schema artifacts."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from loguru import logger

from apb2.parserV2.vendor_parse_rules.document import document_json_schema
from apb2.parserV2.vendor_parse_rules.schema.rule import rule_json_schema

DOCUMENT_PACKAGE = "apb2.parserV2.vendor_parse_rules.documents"


def artifact_path() -> Path:
    """Where Parser V2 publishes its ``rule.schema.json``."""
    return Path(str(resources.files(DOCUMENT_PACKAGE))) / "_schema" / "rule.schema.json"


def write_artifact() -> Path:
    """Regenerate both rule-schema artifacts and return the effective-rule schema path."""
    output = artifact_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rule_json_schema(), indent=2) + "\n")
    output.with_name("document.schema.json").write_text(
        json.dumps(document_json_schema(), indent=2) + "\n"
    )
    logger.info("wrote {}", output)
    return output
