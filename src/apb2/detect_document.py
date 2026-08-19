"""Which packaged rules.json does the evidence name? Detection, outside the rules folder.

Choosing a file is not deserializing one, so this sits above ``vendor_parse_rules`` and uses
its door like any other caller: ``packaged_rules()`` for the inventory, then ``get_rule`` (or
``declared_rule`` when there is no evidence yet) and the recognition that comes back with it.

The constructor IS the detection: a ``DetectedSoftware`` existing proves that the parsed
parameter values plus the table headers identified exactly one packaged file. The
quantification level is deliberately not part of it — a level is a section of a document, not
evidence about which document this is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from apb2.errors import AmbiguousRuleError, RuleNotApplicable, RuleUnavailableError
from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.rules import PACKAGED, Document, load_document


def software_slug(software_name: str) -> str:
    """Map a catalog software name such as ``DIA-NN`` to its rule-folder slug."""
    return re.sub(r"[^a-z0-9]", "", software_name.lower())


def guess_software(headers: Iterable[str]) -> str | None:
    """Detect with headers alone: the unique vendor slug they match, or ``None``.

    Weaker than ``DetectedSoftware`` — no version, no gates, may shrug — but enough to pick
    which vendor's parameter parser to run before full detection has parameters. Gates are
    ignored on purpose: this runs when there is no evidence to satisfy them with.
    """
    header_set = frozenset(headers)
    slugs = {
        software_slug(rules.software_name)
        for rules in (load_document(path) for path in PACKAGED)
        if rules.matches(header_set)
    }
    return next(iter(slugs)) if len(slugs) == 1 else None


class DetectedSoftware:
    """One identified vendor: the evidence named exactly one packaged rules.json."""

    __slots__ = ("_path", "software", "version")

    software: str
    version: str | None
    _path: Path

    def __init__(self, parameters: Parameters, headers: Iterable[str]) -> None:
        """Detect the file; fails when the evidence does not name exactly one."""
        header_set = frozenset(headers)
        matches: list[tuple[Document, str | None]] = []
        for path in PACKAGED:
            rules = load_document(path)
            version = _version_for(parameters, software_slug(rules.software_name))
            if version is not None and not _pattern_admits(rules.software_version_pattern, version):
                continue
            if not _any_level_matches(rules, parameters, header_set):
                continue
            matches.append((rules, version))
        if not matches:
            raise RuleUnavailableError(
                "no packaged rules.json matches the table headers and parameter evidence; "
                "pass --rule-config PATH for an unpackaged format"
            )
        if len(matches) > 1:
            paths = sorted(str(rules.path) for rules, _version in matches)
            raise AmbiguousRuleError(f"evidence matches several packaged documents: {paths}")
        rules, version = matches[0]
        self.software = software_slug(rules.software_name)
        self.version = version
        self._path = rules.path

    def get_rule_path(self) -> Path:
        """Return the one packaged rules.json this evidence identified."""
        return self._path


def _version_for(parameters: Parameters, rule_slug: str) -> str | None:
    """The version belonging to the software this slug names, if the evidence has one."""
    candidates = (
        (parameters.software_name, parameters.software_version),
        (parameters.quantification_software, parameters.quantification_software_version),
    )
    for software_name, version in candidates:
        if software_name is not None and software_slug(software_name) == rule_slug:
            return version
    if not any(software_name for software_name, _version in candidates):
        return parameters.software_version
    return None


def _pattern_admits(pattern: str, version: str) -> bool:
    try:
        return re.search(pattern, version) is not None
    except re.error as exc:
        raise ValueError(f"invalid software_version_pattern regex {pattern!r}") from exc


def _any_level_matches(rules: Document, parameters: Parameters, header_set: frozenset[str]) -> bool:
    """Whether any level this evidence admits also recognizes these headers."""
    for level in rules.levels:
        try:
            rule = rules.rule(level, parameters)
        except RuleNotApplicable:
            continue
        if rule.recognition.matches(header_set):
            return True
    return False
