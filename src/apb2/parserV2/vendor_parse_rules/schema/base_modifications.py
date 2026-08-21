"""Vendor modification declarations in the rules.json storage model."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from apb2.parserV2.vendor_parse_rules.schema.base import (
    ModelBase,
    TokenPosition,
    UnknownPolicy,
)


class ModificationMapEntry(ModelBase):
    """One vendor token and the Unimod accession it denotes."""

    token: str
    accession: str


class TokenRegexModifications(ModelBase):
    """Inline modification tokens extracted from a sequence column."""

    parser: Literal["token_regex"]
    source_column: str
    token_pattern: str
    token_position: TokenPosition = "after_residue"
    case_sensitive: bool = False
    unknown_policy: UnknownPolicy = "preserve"
    output_column: str = "proforma_sequence"
    map: list[ModificationMapEntry] = Field(min_length=1)


class SiteListModifications(ModelBase):
    """Parallel modification-name and modification-site columns beside a sequence."""

    parser: Literal["site_list"]
    sequence_column: str
    modification_column: str
    site_column: str
    delimiter: str = ";"
    site_base: int = Field(default=1, ge=0, le=1)
    case_sensitive: bool = False
    unknown_policy: UnknownPolicy = "preserve"
    output_column: str = "proforma_sequence"
    map: list[ModificationMapEntry] = Field(min_length=1)


type Modifications = Annotated[
    TokenRegexModifications | SiteListModifications,
    Field(discriminator="parser"),
]


def modification_outputs(modifications: Modifications) -> frozenset[str]:
    """Names synthesized by modification normalization."""
    return frozenset({modifications.output_column, "stripped_sequence", "unknown_mod_tokens"})
