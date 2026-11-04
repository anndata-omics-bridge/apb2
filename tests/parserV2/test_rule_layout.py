"""Every packaged rules.json is in the layout ``make format`` writes."""

from __future__ import annotations

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
