"""Packaged hierarchy declarations, independent of the parsing runtime."""

from importlib import resources

from pydantic import TypeAdapter

HIERARCHIES = TypeAdapter(dict[str, tuple[tuple[str, str], ...]]).validate_json(
    resources.files("apb2.parserV2.vendor_parse_rules.schema")
    .joinpath("hierarchies.json")
    .read_bytes()
)
