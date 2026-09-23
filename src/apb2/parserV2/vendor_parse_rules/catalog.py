"""Read the APB2-owned category catalogue for packaged quant-result rules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from pydantic import Field, model_validator

from apb2.parserV2.vendor_parse_rules.document import RuleDocument
from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import LEVELS, ModelBase, QuantificationLevel


@dataclass(frozen=True, slots=True)
class RuleVariant:
    """One packaged rule and its declared software-version range and levels."""

    rule: str
    software_version_pattern: str
    levels: tuple[QuantificationLevel, ...]


@dataclass(frozen=True, slots=True)
class SoftwareRules:
    """One result producer with the matching version-specific rule variants."""

    software_name: str
    variants: tuple[RuleVariant, ...]


class _Assignment(ModelBase):
    """Catalogue categories for one packaged rule path."""

    rule: str = Field(min_length=1)
    categories: tuple[str, ...]
    reason: str | None = None

    @model_validator(mode="after")
    def _unclassified_reason(self) -> _Assignment:
        if not self.categories and not self.reason:
            raise ValueError(f"unclassified rule {self.rule!r} requires a reason")
        if self.categories and self.reason is not None:
            raise ValueError(f"classified rule {self.rule!r} cannot have an unclassified reason")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError(f"duplicate categories for rule {self.rule!r}")
        return self


class _Catalogue(ModelBase):
    """Strict boundary for the separate packaged category configuration."""

    categories: tuple[str, ...] = Field(min_length=1)
    assignments: tuple[_Assignment, ...]

    @model_validator(mode="after")
    def _valid_assignments(self) -> _Catalogue:
        if any(not category.strip() for category in self.categories):
            raise ValueError("category names must be nonempty")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("duplicate category names")
        rules = [assignment.rule for assignment in self.assignments]
        if len(rules) != len(set(rules)):
            raise ValueError("duplicate rule paths")
        unknown = {
            category
            for assignment in self.assignments
            for category in assignment.categories
            if category not in self.categories
        }
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")
        return self


class RuleCatalog:
    """Join category assignments to the current packaged rule inventory."""

    __slots__ = ("_categories", "_rules")

    def __init__(self, source: Path, packaged: tuple[Path, ...] = PACKAGED) -> None:
        catalogue = _Catalogue.model_validate_json(source.read_text(encoding="utf-8"))
        document_root = Path(str(resources.files("apb2.parserV2.vendor_parse_rules.documents")))
        paths = {path.relative_to(document_root).as_posix(): path for path in packaged}
        assigned = {assignment.rule for assignment in catalogue.assignments}
        missing = paths.keys() - assigned
        extra = assigned - paths.keys()
        if missing or extra:
            raise ValueError(
                f"rule catalogue differs from packaged rules: missing={sorted(missing)}, "
                f"unknown={sorted(extra)}"
            )
        self._categories = catalogue.categories
        self._rules: tuple[tuple[_Assignment, RuleDocument], ...] = tuple(
            (assignment, load_rule_document(paths[assignment.rule]))
            for assignment in catalogue.assignments
        )

    def get_rules(
        self, category: str, *, level: QuantificationLevel | None = None
    ) -> list[SoftwareRules]:
        """Return packaged producers and version-specific rules in one category."""
        if category not in self._categories:
            raise ValueError(f"unknown category {category!r}; available: {list(self._categories)}")
        if level is not None and level not in LEVELS:
            raise ValueError(f"unknown quantification level {level!r}; available: {list(LEVELS)}")
        grouped: dict[str, list[RuleVariant]] = defaultdict(list)
        for assignment, document in self._rules:
            if category not in assignment.categories or (
                level is not None and level not in document.levels
            ):
                continue
            grouped[document.software_name].append(
                RuleVariant(
                    rule=assignment.rule,
                    software_version_pattern=document.software_version_pattern,
                    levels=document.levels,
                )
            )
        return [
            SoftwareRules(
                software_name=name,
                variants=tuple(sorted(grouped[name], key=lambda variant: variant.rule)),
            )
            for name in sorted(grouped, key=lambda name: (name.casefold(), name))
        ]


def get_rules(category: str, *, level: QuantificationLevel | None = None) -> list[SoftwareRules]:
    """List packaged quant-result rules by APB2 category, optionally by level.

    Version patterns describe the declared rule range, not a finite list of tested releases.
    The catalogue does not guarantee that an arbitrary upload can be parsed without parameters.
    """
    source = Path(str(resources.files("apb2.parserV2.vendor_parse_rules"))) / "catalog.json"
    return RuleCatalog(source).get_rules(category, level=level)
