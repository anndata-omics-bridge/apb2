"""Write the parsing-rules bundle that APB Studio's rules viewer reads.

``make rules-bundle`` writes ``build/apb2-rules-bundle/``:

- ``manifest.json`` — bundle format, apb2 version and commit, and the files below;
- ``rules.json`` — one entry per packaged rules.json: software, version pattern, DDA/DIA
  categories from ``catalog.json``, and per table its levels, input policy and preparation;
- ``levels.json`` — one entry per rule and level: each distinct effective declaration, exactly
  as conversion stores it in a level's ``rule_json``, with the search-parameter evidence that
  selects it;
- ``documents/<rule>`` — each packaged rules.json, byte for byte.

The viewer shows these files and composes nothing: ``RuleDocument.rule`` merges each table's
base into its levels and applies the search-parameter overrides here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from importlib import metadata, resources
from pathlib import Path
from typing import Literal

from cyclopts import App

from apb2.parserV2.vendor_parse_rules.document import (
    RuleDocument,
    RuleNotApplicable,
    SearchParameterEvidence,
)
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel

app = App(name="build-rules-bundle", help=__doc__)

BUNDLE_FORMAT = 1
MANIFEST = "manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "build" / "apb2-rules-bundle"
DOCUMENTS = Path(str(resources.files("apb2.parserV2.vendor_parse_rules.documents")))
CATALOG = Path(str(resources.files("apb2.parserV2.vendor_parse_rules"))) / "catalog.json"

# Every evidence a conversion can supply, in the order packaged_rule_declarations tries them,
# so each level's declarations come in its first-selected order.
_ACQUISITION: tuple[Literal["DDA", "DIA", "unknown"], ...] = ("DDA", "DIA", "unknown")
_COMBINE_CHARGE_STATES: tuple[bool | None, ...] = (True, False, None)
EVIDENCES = tuple(
    SearchParameterEvidence(acquisition_method=acquisition, combine_charge_states=combine)
    for acquisition in _ACQUISITION
    for combine in _COMBINE_CHARGE_STATES
)

type Json = dict[str, object]


def _rule_entry(rule: str, document: RuleDocument, categories: list[str]) -> Json:
    """One document as the Rules tab lists it; its levels' declarations are in levels.json."""
    return {
        "rule": rule,
        "software_name": document.software_name,
        "software_version_pattern": document.software_version_pattern,
        "schema_version": document.schema_version,
        "file_version": document.file_version,
        "hierarchy": document.hierarchy,
        "parameter_file": document.parameter_file,
        "categories": categories,
        "levels": list(document.levels),
        "tables": [
            {
                "levels": list(table.levels),
                "input": table.input.model_dump(mode="json", exclude_none=True),
                "preparation": None if table.prepare is None else table.prepare.how,
            }
            for table in document.tables
        ],
    }


def _level_entry(rule: str, document: RuleDocument, level: QuantificationLevel) -> Json:
    """One level's distinct effective declarations, each with the evidence that selects it."""
    selected_by: dict[str, list[Json]] = {}
    for evidence in EVIDENCES:
        try:
            effective = document.rule(level, evidence)
        except RuleNotApplicable:
            continue
        text = json.dumps(effective.declaration.model_dump(mode="json"))
        selected_by.setdefault(text, []).append(
            {
                "acquisition_method": evidence.acquisition_method,
                "combine_charge_states": evidence.combine_charge_states,
            }
        )
    return {
        "rule": rule,
        "level": level,
        "declarations": [
            {"selected_by": evidences, "declaration": json.loads(text)}
            for text, evidences in selected_by.items()
        ],
    }


def _commit() -> str | None:
    """The apb2 checkout's commit, ``-dirty`` when it has local changes; None outside git."""
    try:
        described = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=DOCUMENTS,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return described.stdout.strip()


def _write(path: Path, payload: Json) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def build(output: Path) -> None:
    """Write the bundle into ``output``, replacing a bundle written there before.

    Raises:
        ValueError: ``output`` is a non-empty folder that holds no bundle.
    """
    if output.exists() and any(output.iterdir()):
        if not (output / MANIFEST).is_file():
            raise ValueError(f"{output} is not empty and holds no rules bundle")
        shutil.rmtree(output)
    catalogue = json.loads(CATALOG.read_text(encoding="utf-8"))
    categories = {row["rule"]: row["categories"] for row in catalogue["assignments"]}
    rules: list[Json] = []
    levels: list[Json] = []
    files = ["rules.json", "levels.json"]
    for path in PACKAGED:
        rule = path.relative_to(DOCUMENTS).as_posix()
        document = load_rule_document(path)
        rules.append(_rule_entry(rule, document, categories[rule]))
        levels.extend(_level_entry(rule, document, level) for level in document.levels)
        target = output / "documents" / rule
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files.append(f"documents/{rule}")
    _write(output / "rules.json", {"rules": rules})
    _write(output / "levels.json", {"levels": levels})
    _write(
        output / MANIFEST,
        {
            "bundle_format": BUNDLE_FORMAT,
            "apb2_version": metadata.version("apb2"),
            "apb2_commit": _commit(),
            "files": files,
        },
    )


@app.default
def main(output: Path = DEFAULT_OUTPUT) -> None:
    """Write the parsing-rules bundle into OUTPUT."""
    build(output)


if __name__ == "__main__":
    app()
