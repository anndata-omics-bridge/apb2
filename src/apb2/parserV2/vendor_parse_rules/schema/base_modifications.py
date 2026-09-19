"""Named sequence grammars and vendor-token maps, independent of column selection."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase, TokenPosition


class ModificationMapEntry(ModelBase):
    """One vendor token and the Unimod accession it denotes."""

    token: str
    accession: str


class PlainSequenceSyntax(ModelBase):
    """An unmodified sequence whose alphabetic characters are residues."""

    parser: Literal["plain_sequence"]


class TokenRegexSyntax(ModelBase):
    """Inline modification tokens extracted by the declared pattern."""

    parser: Literal["token_regex"]
    token_pattern: str
    token_position: TokenPosition = "after_residue"

    @model_validator(mode="after")
    def _valid_pattern(self) -> TokenRegexSyntax:
        try:
            re.compile(self.token_pattern)
        except re.error as error:
            raise ValueError(f"token_pattern is not a valid regex: {error}") from error
        return self


class SiteListSyntax(ModelBase):
    """Parallel modification-name and modification-site lists."""

    parser: Literal["site_list"]
    delimiter: str = Field(default=";", min_length=1)
    site_base: int = Field(default=1, ge=0, le=1)


class EmbeddedSiteListSyntax(ModelBase):
    """Modification-name list entries that also contain their localization."""

    parser: Literal["embedded_site_list"]
    delimiter: str = Field(default=";", min_length=1)
    entry_pattern: str
    site_base: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def _pattern_captures_token_and_site(self) -> EmbeddedSiteListSyntax:
        try:
            groups = re.compile(self.entry_pattern).groupindex
        except re.error as error:
            raise ValueError(f"entry_pattern is not a valid regex: {error}") from error
        missing = {"token", "site"} - set(groups)
        if missing:
            raise ValueError(f"entry_pattern requires named groups: {sorted(missing)}")
        return self


type SequenceSyntax = Annotated[
    PlainSequenceSyntax | TokenRegexSyntax | SiteListSyntax | EmbeddedSiteListSyntax,
    Field(discriminator="parser"),
]
type ModificationMap = Annotated[list[ModificationMapEntry], Field(min_length=1)]
