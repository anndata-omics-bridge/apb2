"""Where the application hands its parsed search parameters to Parser V2.

The only module that knows both vocabularies, which is the point of its existing. Parser V2's
rule package owns the finite condition vocabulary schema 0.3 permits — two fields — and never
sees the parameter model those values were read from. This is the translation, and it is the
whole boundary: everything else a caller needs is Parser V2's own public API.
"""

from __future__ import annotations

from apb2.parserV2.vendor_parse_rules.document import SearchParameterEvidence
from apb2.vendor_params.model import Parameters


def search_parameter_evidence(parameters: Parameters) -> SearchParameterEvidence:
    """The two fields a schema-0.3 gate or override may read, and nothing else."""
    return SearchParameterEvidence(
        acquisition_method=parameters.acquisition_method,
        combine_charge_states=parameters.combine_charge_states,
    )


def unknown_search_parameters() -> SearchParameterEvidence:
    """The evidence a caller has when no parameter file was supplied.

    A distinct function rather than a nullable argument: "no parameters were read" is a
    different fact from "the parameters say this", and a gated level must be able to exclude
    the first as clearly as the second.
    """
    return SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)
