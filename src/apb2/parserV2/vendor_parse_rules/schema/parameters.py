"""Search-parameter gates and primary-measurement overrides."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from apb2.parserV2.vendor_parse_rules.schema.base import ModelBase

type ConditionValue = str | int | float | bool | None
type SearchParameterField = Literal["acquisition_method", "combine_charge_states"]


class SearchParameterOverride(ModelBase):
    """Swap the primary layer when the declared parameter equality holds."""

    when_search_parameters: dict[SearchParameterField, ConditionValue] = Field(min_length=1)
    primary_layer: str
