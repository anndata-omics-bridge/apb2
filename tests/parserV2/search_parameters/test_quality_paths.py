"""Focused edge-path coverage for the shared parser helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from apb2.parserV2.search_parameters import common as _common
from apb2.parserV2.search_parameters.model import MassTolerance, ModType


class _Unseekable:
    def seekable(self) -> bool:
        return False

    def read(self) -> str:
        return "content"


def test_shared_source_reading_accepts_paths_bytes_and_unseekable_streams(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    path.write_text("path", encoding="utf-8")

    assert _common.read_text(path) == "path"
    assert _common.read_text(BytesIO(b"bytes")) == "bytes"
    assert _common.read_text(cast(Any, _Unseekable())) == "content"


def test_vendor_mass_table_lookup_reports_an_unmatched_mass_explicitly() -> None:
    assert _common.lookup_mass_mod(1.0, {1.0: "known"}) == _common.MassModificationMatch("known")
    assert _common.lookup_mass_mod(2.0, {1.0: "known"}) == _common.UnrecognizedModificationMass(2.0)


def test_paren_modification_tokens_fall_back_to_the_vendor_mapping() -> None:
    assert _common.homogenize_paren_mods("known", {"known": "mapped"}) == ["mapped"]
    assert _common.homogenize_paren_mods("Phospho (STY)", {}) == [
        "S[Phospho]",
        "T[Phospho]",
        "Y[Phospho]",
    ]
    assert _common.homogenize_paren_mods("Acetyl (Protein N-term)", {}) == [
        "Protein N-term[Acetyl]"
    ]


def test_prose_tolerances_are_read_including_the_calibrated_case() -> None:
    assert _common.tolerance_from_text("20 ppm") == MassTolerance(
        mode="absolute", value=20.0, unit="ppm"
    )
    assert _common.tolerance_from_text("0.02Da") == MassTolerance(
        mode="absolute", value=0.02, unit="Da"
    )
    assert _common.tolerance_from_text("20 Th") == MassTolerance(
        mode="absolute", value=20.0, unit="Da"
    )
    assert _common.tolerance_from_text("Dynamic") == _common.automatic_tolerance()
    with pytest.raises(ValueError, match="could not read"):
        _common.tolerance_from_text("not a tolerance")
    with pytest.raises(ValueError, match="must be ppm or Da"):
        _common.tolerance_from_text("20 kg")


def test_a_signed_interval_becomes_a_half_width_and_rejects_asymmetry() -> None:
    assert _common.symmetric_tolerance(-20.0, 20.0, "ppm") == MassTolerance(
        mode="absolute", value=20.0, unit="ppm"
    )
    with pytest.raises(ValueError, match="asymmetric"):
        _common.symmetric_tolerance(-10.0, 30.0, "ppm")


def test_modification_tokens_resolve_and_keep_the_declared_type() -> None:
    resolved = _common.modifications(["M[Oxidation]", "K[Vendor label]"], ModType.variable)

    assert [mod.name for mod in resolved] == ["M[Oxidation]", "K[Vendor label]"]
    assert [mod.accession for mod in resolved] == ["UNIMOD:35", None]
    assert {mod.mod_type for mod in resolved} == {ModType.variable}
