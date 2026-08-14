"""Generate the packaged effective-rule JSON Schema artifact from the models."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from apb2.vendor_parse_rules.model import rule_json_schema


def main() -> None:
    output_directory = Path(__file__).resolve().parent / "_schema"
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / "rule.schema.json"
    output.write_text(json.dumps(rule_json_schema(), indent=2) + "\n")
    logger.info(f"wrote {output}")


if __name__ == "__main__":
    main()
