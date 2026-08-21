"""Write the packaged effective-rule JSON Schema artifacts from the models.

Not part of getting a rule, so it lives above the rule packages rather than inside one: it
reads the schema each generation's models declare and writes one file next to its documents.
Both generations are published because both are loadable — schema 0.2 for the parity oracle,
schema 0.3 for Parser V2.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from loguru import logger

from apb2.parserV2.vendor_parse_rules.schema_rule import rule_json_schema as parser_v2_schema
from apb2.vendor_parse_rules.model import rule_json_schema

ARTIFACTS = (
    ("apb2.vendor_parse_rules.documents", rule_json_schema),
    ("apb2.parserV2.vendor_parse_rules.documents", parser_v2_schema),
)
"""Each rule generation's document package and the schema its own models declare."""


def artifact_path(package: str) -> Path:
    """Where one rule generation publishes its ``rule.schema.json``."""
    return Path(str(resources.files(package))) / "_schema" / "rule.schema.json"


def write_artifact() -> tuple[Path, ...]:
    """Regenerate every generation's ``documents/_schema/rule.schema.json``."""
    written: list[Path] = []
    for package, schema in ARTIFACTS:
        output = artifact_path(package)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(schema(), indent=2) + "\n")
        logger.info(f"wrote {output}")
        written.append(output)
    return tuple(written)
