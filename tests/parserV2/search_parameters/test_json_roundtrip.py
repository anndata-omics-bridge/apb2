"""Search parameters survive the JSON round trip the converter stores them through.

``cli._execute`` persists ``json.dumps(parameters.model_dump(mode="json"))`` into the apb2
namespace, so the schema has to reconstruct itself from exactly that payload.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from apb2.parserV2.search_parameters.model import (
    MassTolerance,
    Parameters,
    Probability,
    SearchedModification,
)


def _round_trip(parameters: Parameters) -> Parameters:
    return Parameters.model_validate(json.loads(json.dumps(parameters.model_dump(mode="json"))))


def test_typed_values_survive_the_round_trip() -> None:
    parameters = Parameters(
        software_name="DIA-NN",
        software_version="2.3.0 Academia ",
        ident_fdr_psm=Probability(value=0.01),
        precursor_mass_tolerance=MassTolerance(mode="absolute", value=15.0, unit="ppm"),
        fragment_mass_tolerance=MassTolerance(mode="automatic", label="Automatic calibration"),
        fixed_mods=[SearchedModification(name="C[Carbamidomethyl]", accession="UNIMOD:4")],
    )

    recovered = _round_trip(parameters)

    assert recovered == parameters
    assert recovered.ident_fdr_psm == Probability(value=0.01)
    assert recovered.precursor_mass_tolerance == MassTolerance(
        mode="absolute", value=15.0, unit="ppm"
    )
    assert recovered.fragment_mass_tolerance == MassTolerance(
        mode="automatic", label="Automatic calibration"
    )


def test_every_field_is_serialized_and_nulls_are_preserved() -> None:
    payload = json.loads(json.dumps(Parameters(software_name="PEAKS").model_dump(mode="json")))

    assert set(payload) == set(Parameters.model_fields)
    assert payload["software_version"] is None
    assert payload["acquisition_method"] == "unknown"


def test_a_payload_from_a_newer_schema_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValidationError):
        Parameters.model_validate(json.loads('{"software_name": "Sage", "vendor_specific": 42}'))


def test_a_payload_from_an_older_schema_validates_against_the_current_one() -> None:
    recovered = Parameters.model_validate(json.loads('{"software_name": "Sage"}'))

    assert recovered.software_name == "Sage"
    assert recovered.acquisition_method == "unknown"
