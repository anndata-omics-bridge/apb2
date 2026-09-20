"""Bind a source reader and result writer to one compiled parsing strategy."""

from __future__ import annotations

from typing import Literal

from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.io.formats import ParsedLevelFormatWriter
from apb2.parserV2.parse_quant.parameters.source import (
    FrameSourceEvidence,
    InputSource,
    PreparedTable,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.prepare_source import prepare_source
from apb2.parserV2.source_binding import BoundTable


def compile_level(
    facade: ParseRuleFacade,
    source: InputSource,
    checks: Literal["standard", "strict"],
) -> Parser:
    """Resolve one selected level and bind its fully configured strategy."""
    working = facade.working_parameters
    source = prepare_source(source, working.preparation)
    preparation: dict[str, JsonValue] = {}
    if isinstance(source, PreparedTable):
        evidence = FrameSourceEvidence(
            columns=tuple(source.frame.columns), dtypes=tuple(source.frame.schema.items())
        )
        strategy = facade.resolve_source(evidence, checks=checks)
        input_reader = PreparedInputReader(
            source.frame,
            strategy.read,
            (strategy.obs.keys.raw_key_columns, strategy.var.keys.raw_key_columns),
        )
        preparation = {
            "input_preparation": {
                "how": source.how,
                "sources": [str(path) for path in source.source_paths],
                "duration_seconds": source.duration_seconds,
                "rows": source.frame.height,
                "estimated_size_bytes": source.frame.estimated_size(),
            }
        }
    else:
        bound = BoundTable(source, working.input)
        evidence = bound.evidence(working.accepts_header)
        strategy = facade.resolve_source(evidence, checks=checks)
        input_reader = bound.reader(evidence, strategy.read)
    strategy.provenance.update(preparation)
    return Parser(input_reader, strategy, ParsedLevelFormatWriter())
