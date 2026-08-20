"""Tests for ``apb2.vendor_params.model.Parameters`` as a storage schema.

``Parameters`` accepts the types its fields declare; the vendor parsers produce those types
and nothing else. ProteoBench's text shapes are read by ``proteobench_params``, the parity
oracle, so the assertions here are about invariants rather than about coercion.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from apb2.vendor_params.model import (
    MassTolerance,
    ModType,
    Parameters,
    Probability,
    SearchedModification,
)


def test_construct_empty_uses_unknown_acquisition_method() -> None:
    dumped = Parameters().model_dump()

    assert dumped["software_name"] is None
    assert dumped["enzyme"] is None
    assert dumped["scan_window"] is None
    assert dumped["acquisition_method"] == "unknown"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Parameters.model_validate({"software_name": "Sage", "vendor_specific_thing": 42})


def test_empty_and_missing_text_becomes_none() -> None:
    """Vendor CSV and XML cells arrive as blanks and pandas ``NaN``; both mean absent."""
    params = Parameters(software_name="Sage", quantification_method="", protein_inference="unknown")

    assert params.quantification_method is None
    assert params.protein_inference is None


def test_modifications_keep_the_type_the_parser_assigned() -> None:
    params = Parameters(
        fixed_mods=[SearchedModification(name="C[Carbamidomethyl]", mod_type=ModType.fixed)],
        variable_mods=[SearchedModification(name="M[Oxidation]", mod_type=ModType.variable)],
    )

    assert [mod.mod_type for mod in params.fixed_mods] == [ModType.fixed]
    assert [mod.mod_type for mod in params.variable_mods] == [ModType.variable]


def test_modification_accession_must_look_like_an_identifier() -> None:
    with pytest.raises(ValidationError, match="UNIMOD:35"):
        SearchedModification(name="M[Oxidation]", accession="oxidation")


@pytest.mark.parametrize("value", ["DDA", "DIA", "unknown"])
def test_acquisition_method_accepts_declared_values(value: str) -> None:
    assert Parameters.model_validate({"acquisition_method": value}).acquisition_method == value


@pytest.mark.parametrize("value", [None, "SWATH", "dda"])
def test_acquisition_method_rejects_values_outside_vocabulary(value: object) -> None:
    with pytest.raises(ValidationError):
        Parameters.model_validate({"acquisition_method": value})


def test_probability_rejects_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Probability(value=1.2)


def test_parameters_reject_negative_mz() -> None:
    with pytest.raises(ValidationError):
        Parameters(min_precursor_mz=-1)


def test_parameters_reject_invalid_charge() -> None:
    with pytest.raises(ValidationError):
        Parameters(min_precursor_charge=0)


def test_parameters_reject_invalid_range_ordering() -> None:
    with pytest.raises(ValidationError):
        Parameters(min_precursor_mz=900, max_precursor_mz=300)


def test_absolute_tolerance_requires_a_value_and_a_unit() -> None:
    with pytest.raises(ValidationError, match="requires value"):
        MassTolerance(mode="absolute", unit="ppm")
    with pytest.raises(ValidationError, match="requires unit"):
        MassTolerance(mode="absolute", value=20)


def test_automatic_tolerance_carries_no_numeric_bounds() -> None:
    with pytest.raises(ValidationError, match="cannot define unit"):
        MassTolerance(mode="automatic", unit="ppm")
    with pytest.raises(ValidationError, match="cannot define numeric"):
        MassTolerance(mode="automatic", value=1)

    automatic = MassTolerance(mode="automatic", label="Automatic calibration")

    assert automatic.value is None
    assert automatic.unit is None
