"""Named sequence grammars and vendor-token maps, independent of column selection."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase, TokenPosition

_SITES = frozenset({"nterm", "cterm", "residue"})


class ModificationMapEntry(ModelBase):
    """One vendor token and the Unimod accession it denotes."""

    token: str
    accession: str


class PlainSequenceSyntax(ModelBase):
    """An unmodified sequence: residue letters only, apart from terminal ``_``, ``-`` or ``.``."""

    parser: Literal["plain_sequence"]


class TokenRegexSyntax(ModelBase):
    """Inline modification tokens extracted by the declared pattern; the rest must be residues."""

    parser: Literal["token_regex"]
    token_pattern: str = Field(
        description="each capturing group named for its token's site: nterm, cterm or residue"
    )
    token_position: TokenPosition = Field(
        default="after_residue",
        description="whether a 'residue' token follows or precedes its residue",
    )
    marker_pattern: str | None = Field(
        default=None,
        description="vendor text that is neither residue nor modification, removed before "
        "tokenizing, such as AlphaPept's '_decoy' suffix",
    )

    @model_validator(mode="after")
    def _valid_pattern(self) -> TokenRegexSyntax:
        for field, pattern in (
            ("token_pattern", self.token_pattern),
            ("marker_pattern", self.marker_pattern),
        ):
            if pattern is None:
                continue
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"{field} is not a valid regex: {error}") from error
        compiled = re.compile(self.token_pattern)
        names = set(compiled.groupindex)
        if not names or compiled.groups != len(names) or not names <= _SITES:
            raise ValueError(
                f"token_pattern {self.token_pattern!r} must name every capturing group "
                f"after a site, {sorted(_SITES)}"
            )
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
