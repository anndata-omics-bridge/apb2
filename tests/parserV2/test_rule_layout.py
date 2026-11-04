"""Every packaged rules.json is in the layout ``make format`` writes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_rules_are_in_layout() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.format_rules", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"run make format; out of layout:\n{result.stdout}"


def test_a_computed_entry_names_its_operation_after_its_inputs() -> None:
    """The formatter orders a computed entry: what it reads, the operation, its settings, roles."""
    entry = {
        "roles": ["fasta_accessions"],
        "separator": ";",
        "how": "join_nonempty",
        "note": "Joined",
        "inputs": ["A", "B"],
        "source": "Protein",
        "name": "Proteins",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; from scripts.format_rules import format_rule; "
            "print(format_rule(json.load(sys.stdin)), end='')",
        ],
        input=json.dumps({"var": [entry]}),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    formatted = json.loads(result.stdout)["var"][0]

    assert list(formatted) == ["name", "source", "inputs", "how", "separator", "roles", "note"]
