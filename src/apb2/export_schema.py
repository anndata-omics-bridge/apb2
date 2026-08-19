"""Write the packaged effective-rule JSON Schema artifact from the models.

Not part of getting a rule, so it lives above ``vendor_parse_rules`` rather than inside it:
it reads the schema the models declare and writes one file next to the documents.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from loguru import logger

from apb2.vendor_parse_rules.model import rule_json_schema


def write_artifact() -> Path:
    """Regenerate ``documents/_schema/rule.schema.json`` and return its path."""
    directory = Path(str(resources.files("apb2.vendor_parse_rules.documents"))) / "_schema"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "rule.schema.json"
    output.write_text(json.dumps(rule_json_schema(), indent=2) + "\n")
    logger.info(f"wrote {output}")
    return output
