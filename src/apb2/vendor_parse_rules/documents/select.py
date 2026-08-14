"""Detect which packaged rules.json the evidence names.

The constructor IS the detection: a ``DetectedSoftware`` existing proves that the parsed
parameter values plus the table headers identified exactly one packaged document.
``get_rule_path()`` returns that file; loading and composing it stays in ``load.py``,
where the quantification level enters — a level is a section of the document, not part
of detecting it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from importlib import resources
from pathlib import Path

from apb2.vendor_params.model import Parameters
from apb2.vendor_parse_rules.model import Document, compose_rule, load_document
from apb2.vendor_parse_rules.runtime import available_for, recognition_for, resolved_for


class RuleUnavailableError(ValueError):
    """No packaged rules.json satisfies the supplied evidence."""


class AmbiguousRuleError(ValueError):
    """Several packaged rules.json satisfy evidence that must identify exactly one."""


def software_slug(software_name: str) -> str:
    """Map a catalog software name such as ``DIA-NN`` to its rule slug."""
    return re.sub(r"[^a-z0-9]", "", software_name.lower())


def guess_software(headers: Iterable[str]) -> str | None:
    """Detect with headers alone: the unique vendor slug they match, or ``None``.

    Weaker than ``DetectedSoftware`` — no version, no gates, may shrug — but enough to
    pick which vendor's parameter parser to run before full detection has parameters.
    """
    header_set = frozenset(headers)
    slugs = {
        software_slug(document.software_name)
        for document in _packaged_documents()
        if any(
            recognition_for(compose_rule(document, level)).matches(header_set)
            for level in document.levels
        )
    }
    return next(iter(slugs)) if len(slugs) == 1 else None


class DetectedSoftware:
    """One identified vendor: the evidence named exactly one packaged rules.json."""

    __slots__ = ("_path", "software", "version")

    software: str
    version: str | None
    _path: Path

    def __init__(self, parameters: Parameters, headers: Iterable[str]) -> None:
        """Detect the document; fails when the evidence does not name exactly one file."""
        header_set = frozenset(headers)
        matches: list[tuple[Document, str | None]] = []
        for document in _packaged_documents():
            slug = software_slug(document.software_name)
            version = _version_for(parameters, slug)
            if version is not None and not _pattern_admits(
                document.software_version_pattern, version
            ):
                continue
            if not _any_level_matches(document, parameters, header_set):
                continue
            matches.append((document, version))
        if not matches:
            raise RuleUnavailableError(
                "no packaged rules.json matches the table headers and parameter evidence; "
                "pass --rule-config PATH for an unpackaged format"
            )
        if len(matches) > 1:
            paths = sorted(str(document.path) for document, _version in matches)
            raise AmbiguousRuleError(f"evidence matches several packaged documents: {paths}")
        document, version = matches[0]
        self.software = software_slug(document.software_name)
        self.version = version
        self._path = document.path

    def get_rule_path(self) -> Path:
        """Return the one packaged rules.json this evidence identified."""
        return self._path


def _packaged_documents() -> tuple[Document, ...]:
    """Load every packaged document in stable path order."""
    root = Path(str(resources.files("apb2.vendor_parse_rules.documents")))
    paths = sorted(set(root.glob("*/rules.json")) | set(root.glob("*/v*/rules.json")))
    return tuple(load_document(path) for path in paths)


def _version_for(parameters: Parameters, rule_slug: str) -> str | None:
    """The version belonging to the software this slug names, if the evidence has one."""
    candidates = (
        (parameters.software_name, parameters.software_version),
        (
            parameters.quantification_software,
            parameters.quantification_software_version,
        ),
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


def _any_level_matches(
    document: Document,
    parameters: Parameters,
    header_set: frozenset[str],
) -> bool:
    for level in document.levels:
        rule = compose_rule(document, level)
        if not available_for(rule, parameters):
            continue
        if recognition_for(resolved_for(rule, parameters)).matches(header_set):
            return True
    return False
