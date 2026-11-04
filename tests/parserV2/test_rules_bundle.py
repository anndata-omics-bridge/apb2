"""The parsing-rules bundle: every packaged rule and level, as conversion stores it."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from apb2.api import packaged_rule_declarations

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = Path(str(resources.files("apb2.parserV2.vendor_parse_rules.documents")))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("rules") / "bundle"
    subprocess.run(
        [sys.executable, "-m", "scripts.build_rules_bundle", str(output)], cwd=ROOT, check=True
    )
    return output


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_each_level_holds_the_declarations_conversion_stores(bundle: Path) -> None:
    levels = _read(bundle / "levels.json")["levels"]

    stored = {
        (entry["rule"], entry["level"]): tuple(
            json.dumps(declaration["declaration"]) for declaration in entry["declarations"]
        )
        for entry in levels
    }

    assert len(stored) == len(levels)
    assert stored == packaged_rule_declarations()


def test_the_evidence_names_which_declaration_a_conversion_uses(bundle: Path) -> None:
    levels = _read(bundle / "levels.json")["levels"]
    (ion,) = (
        entry
        for entry in levels
        if entry["rule"] == "diann/v2/rules.json" and entry["level"] == "peptidoform_ion"
    )

    primary = {
        evidence["acquisition_method"]: declaration["declaration"]["measurements"]["primary_layer"]
        for declaration in ion["declarations"]
        for evidence in declaration["selected_by"]
    }

    assert primary == {
        "DDA": "Ms1_Normalised",
        "DIA": "Precursor_Normalised",
        "unknown": "Precursor_Normalised",
    }


def test_documents_are_copied_verbatim_and_listed_in_the_manifest(bundle: Path) -> None:
    rules = _read(bundle / "rules.json")["rules"]
    manifest = _read(bundle / "manifest.json")

    assert manifest["bundle_format"] == 1
    assert {rule["rule"] for rule in rules} == {
        path.relative_to(DOCUMENTS).as_posix() for path in DOCUMENTS.glob("**/rules.json")
    }
    for rule in rules:
        copy = bundle / "documents" / rule["rule"]
        assert copy.read_bytes() == (DOCUMENTS / rule["rule"]).read_bytes()
        assert {level for table in rule["tables"] for level in table["levels"]} == set(
            rule["levels"]
        )
    assert all((bundle / path).is_file() for path in manifest["files"])


def test_a_folder_holding_something_else_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_rules_bundle", str(other)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "holds no rules bundle" in result.stderr
    assert (other / "notes.txt").read_text(encoding="utf-8") == "keep"
