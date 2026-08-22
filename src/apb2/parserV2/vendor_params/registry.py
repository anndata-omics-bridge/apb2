"""Software-name → parameter-parser dispatch."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.alphadia import extract_params as _alphadia_extract
from apb2.parserV2.vendor_params.parsers.alphapept import extract_params as _alphapept_extract
from apb2.parserV2.vendor_params.parsers.diann import extract_params as _diann_extract
from apb2.parserV2.vendor_params.parsers.fragpipe import extract_params as _fragpipe_extract
from apb2.parserV2.vendor_params.parsers.maxquant import extract_params as _maxquant_extract
from apb2.parserV2.vendor_params.parsers.metamorpheus import (
    extract_params as _metamorpheus_extract,
)
from apb2.parserV2.vendor_params.parsers.msaid import extract_params as _msaid_extract
from apb2.parserV2.vendor_params.parsers.peaks import extract_params as _peaks_extract
from apb2.parserV2.vendor_params.parsers.sage import extract_params as _sage_extract
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters, ParamsError
from apb2.parserV2.vendor_params.parsers.spectronaut import (
    extract_params as _spectronaut_extract,
)
from apb2.parserV2.vendor_params.parsers.wombat import extract_params as _wombat_extract

type ParameterSource = Path | IO[bytes] | IO[str]
type ParameterInput = ParameterSource | tuple[ParameterSource, ...]
type ParseFn = Callable[[ParameterInput], Parameters]
type _SingleSourceParseFn = Callable[[ParameterSource], Parameters]


def _sources(value: ParameterInput) -> tuple[ParameterSource, ...]:
    """Normalize one source or an explicit source tuple without treating paths as sequences."""
    return value if isinstance(value, tuple) else (value,)


def _require_source_count(
    value: ParameterInput,
    *,
    software: str,
    expected: int,
) -> tuple[ParameterSource, ...]:
    sources = _sources(value)
    if len(sources) != expected:
        raise ParamsError(
            f"{software} requires exactly {expected} parameter sources; received {len(sources)}"
        )
    return sources


def _single_source(
    software: str,
    extract: _SingleSourceParseFn,
) -> ParseFn:
    """Adapt one-source vendor implementations to the public registry contract."""

    def parse(value: ParameterInput, /) -> Parameters:
        (source,) = _require_source_count(value, software=software, expected=1)
        return extract(source)

    return parse


def _parse_metamorpheus(value: ParameterInput, /) -> Parameters:
    """Adapt MetaMorpheus's explicit TOML + version-text pair."""
    file_a, file_b = _require_source_count(
        value,
        software="MetaMorpheus",
        expected=2,
    )
    return _metamorpheus_extract(file_a, file_b)


_alphadia_parse = _single_source("AlphaDIA", _alphadia_extract)
_alphapept_parse = _single_source("AlphaPept", _alphapept_extract)
_diann_parse = _single_source("DIA-NN", _diann_extract)
_fragpipe_parse = _single_source("FragPipe", _fragpipe_extract)
_maxquant_parse = _single_source("MaxQuant", _maxquant_extract)
_msaid_parse = _single_source("MSAID", _msaid_extract)
_peaks_parse = _single_source("PEAKS", _peaks_extract)
_sage_parse = _single_source("Sage", _sage_extract)
_spectronaut_parse = _single_source("Spectronaut", _spectronaut_extract)
_wombat_parse = _single_source("Wombat", _wombat_extract)

_REGISTRY: dict[str, ParseFn] = {
    "alphadia": _alphadia_parse,
    "alphapept": _alphapept_parse,
    "dia-nn": _diann_parse,
    "diann": _diann_parse,
    "fragpipe": _fragpipe_parse,
    "maxquant": _maxquant_parse,
    "metamorpheus": _parse_metamorpheus,
    "msaid": _msaid_parse,
    "peaks": _peaks_parse,
    "sage": _sage_parse,
    "spectronaut": _spectronaut_parse,
    "wombat": _wombat_parse,
}


def get_parser(software: str) -> ParseFn:
    """Look up a parser by software name (case-insensitive)."""
    key = software.lower()
    if key not in _REGISTRY:
        raise ParamsError(
            f"no parameter parser registered for {software!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def parse_params(path: ParameterInput, software: str) -> Parameters:
    """Look up a parser and run it on one source or an explicit source tuple."""
    return get_parser(software)(path)


def available_software() -> list[str]:
    return sorted(_REGISTRY)
