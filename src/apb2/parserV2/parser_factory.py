"""Bind a source reader and result writer to one compiled parsing strategy."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from apb2.parserV2.parse_quant.parameters.source import (
    FrameSourceEvidence,
    InputSource,
    PreparedTable,
)
from apb2.parserV2.parse_quant.parser import Parser
from apb2.parserV2.parse_quant.prepared_input import PreparedInputReader
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.source_binding import BoundTable


def compile_level(
    facade: ParseRuleFacade,
    source: InputSource,
    checks: Literal["standard", "strict"],
) -> Parser:
    """Resolve one selected level and bind its fully configured strategy."""
    working = facade.working_parameters
    if isinstance(source, PreparedTable):
        evidence = FrameSourceEvidence(
            columns=tuple(source.frame.columns), dtypes=tuple(source.frame.schema.items())
        )
        strategy = facade.resolve_source(evidence, checks=checks)
        # A wide level's observation keys come from header captures, not prepared columns,
        # so only key columns the level reads can mark a row as absent.
        projected = set(strategy.read.projected_columns)
        input_reader = PreparedInputReader(
            source.frame,
            strategy.read,
            tuple(
                keys
                for keys in (strategy.obs.keys.raw_key_columns, strategy.var.keys.raw_key_columns)
                if set(keys) <= projected
            ),
            source.input_tables,
        )
        preparation = {
            "how": source.how,
            "duration_seconds": source.duration_seconds,
            "estimated_size_bytes": source.frame.estimated_size(),
        }
        strategy = replace(
            strategy, provenance={**strategy.provenance, "input_preparation": preparation}
        )
    else:
        bound = BoundTable(source, working.input)
        evidence = bound.evidence(working.accepts_header)
        strategy = facade.resolve_source(evidence, checks=checks)
        input_reader = bound.reader(evidence, strategy.read)
    return Parser(input_reader, strategy)
