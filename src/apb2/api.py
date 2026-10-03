"""The one public APB2 module: vendor parsing, parsed results, and result files."""

from __future__ import annotations

from collections.abc import Mapping

from apb2.annotation.compiler import AnnotationCompiler
from apb2.annotation.data.model import AnnotationError
from apb2.annotation.sdrf import SdrfSource
from apb2.parserV2.compile import ParseRuleCompiler
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ParsedLevel,
    ParsedLevels,
)
from apb2.parserV2.parse_quant.io.errors import ResultIOError
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels
from apb2.parserV2.parse_quant.io.json_representation import sidecar_path
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_params.parsers.shared.unimod import UNIMOD_REGISTRY
from apb2.parserV2.vendor_params.registry import parse_params as parse_search_parameters
from apb2.parserV2.vendor_parse_rules.catalog import RuleVariant, SoftwareRules, get_rules
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel


def canonical_modification_names() -> Mapping[str, str]:
    """Return canonical modification names keyed by Unimod accession."""
    return UNIMOD_REGISTRY.names_by_accession()


__all__ = [
    "AnnotationCompiler",
    "AnnotationError",
    "FinalLayerTable",
    "JsonValue",
    "Parameters",
    "ParseRuleCompiler",
    "ParsedLevel",
    "ParsedLevels",
    "QuantificationLevel",
    "ResultIOError",
    "RuleVariant",
    "SdrfSource",
    "SoftwareRules",
    "canonical_modification_names",
    "get_rules",
    "parse_search_parameters",
    "read_parsed_levels",
    "sidecar_path",
    "write_parsed_levels",
]
