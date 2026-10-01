"""Store APB's JSON metadata in AnnData ``uns`` without HDF5 or NumPy altering it.

AnnData writes every list through ``np.array`` and every key as an HDF5 link name. Lists of
records, nulls or lists cannot be written; mixed scalars are coerced (``[1, "a"]`` becomes
text, ``[True, 1]`` integers); integers beyond 64 bits, NUL characters, and keys that are
empty, ``"."`` or contain ``"/"`` fail or are truncated. Such a value is stored as JSON text
and the storage descriptor records its path, so a reader decodes exactly those values and
leaves ordinary text alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import numpy as np

from apb2.parserV2.parse_quant.data.parsed import JsonValue
from apb2.parserV2.parse_quant.io.errors import InvalidResultError
from apb2.parserV2.parse_quant.io.metadata import STORAGE_NAMESPACE, object_mapping, string_list

JSON_VALUES = "json_values"

_INT64 = range(-(2**63), 2**63)
_NATIVE_ITEM_TYPES = frozenset({str, bool, int, float})


class UnsJsonCodec:
    """Write and read one ``uns["apb"]`` namespace: metadata plus its storage descriptor."""

    __slots__ = ()

    def encode(
        self,
        metadata: Mapping[str, JsonValue],
        storage: Mapping[str, JsonValue],
        /,
    ) -> dict[str, object]:
        """Return the physical namespace, recording every value stored as JSON text."""
        physical, paths = _encoded_sections(metadata)
        physical[STORAGE_NAMESPACE] = json.dumps(
            {**storage, JSON_VALUES: cast(list[JsonValue], paths)},
            ensure_ascii=False,
            allow_nan=False,
        )
        return physical

    def storage(self, namespace: Mapping[str, object], /) -> Mapping[str, object]:
        """Parse the storage descriptor without interpreting its format."""
        raw = namespace.get(STORAGE_NAMESPACE)
        if not isinstance(raw, str):
            raise InvalidResultError("h5 result has no APB2 storage descriptor")
        try:
            return object_mapping(json.loads(raw), "APB2 storage descriptor")
        except json.JSONDecodeError as error:
            raise InvalidResultError(f"invalid APB2 storage descriptor: {error}") from error

    def decode(
        self, namespace: Mapping[str, object], storage: Mapping[str, object], /
    ) -> dict[str, JsonValue]:
        """Restore the metadata ``encode`` received, rejecting invented or missing paths."""
        recorded = storage.get(JSON_VALUES)
        if not isinstance(recorded, list):
            raise InvalidResultError(f"APB2 storage descriptor has no {JSON_VALUES} list")
        paths = [string_list(entry, "JSON value path") for entry in cast(list[object], recorded)]
        result = {
            key: _python_value(value)
            for key, value in namespace.items()
            if key != STORAGE_NAMESPACE
        }
        for path in paths:
            _decode_at(result, path)
        if _encoded_sections(result)[1] != paths:
            raise InvalidResultError("recorded JSON value paths do not describe the metadata")
        return result


def _encoded_sections(
    metadata: Mapping[str, JsonValue],
) -> tuple[dict[str, object], list[list[str]]]:
    invalid = [key for key in metadata if not _is_link_name(key)]
    if invalid:
        raise InvalidResultError(f"APB metadata section(s) {invalid!r} cannot name an HDF5 group")
    paths: list[list[str]] = []
    physical = {key: _encoded(value, [key], paths) for key, value in metadata.items()}
    return physical, sorted(paths)


def _encoded(value: JsonValue, path: list[str], paths: list[list[str]]) -> object:
    if isinstance(value, dict) and all(_is_link_name(key) for key in value):
        return {key: _encoded(item, [*path, key], paths) for key, item in value.items()}
    if _is_native(value):
        return value
    paths.append(path)
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _is_native(value: JsonValue) -> bool:
    if isinstance(value, list):
        return len({type(item) for item in value}) <= 1 and all(
            type(item) in _NATIVE_ITEM_TYPES and _is_native(item) for item in value
        )
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in _INT64
    if isinstance(value, str):
        return "\x00" not in value
    return value is None or isinstance(value, float)


def _is_link_name(key: str) -> bool:
    return key not in {"", "."} and "/" not in key and "\x00" not in key


def _decode_at(root: dict[str, JsonValue], path: list[str]) -> None:
    if not path:
        raise InvalidResultError("a recorded JSON value path is empty")
    parent: JsonValue = root
    for key in path[:-1]:
        parent = parent.get(key) if isinstance(parent, dict) else None
    text = parent.get(path[-1]) if isinstance(parent, dict) else None
    if not isinstance(parent, dict) or not isinstance(text, str):
        raise InvalidResultError(f"recorded JSON value path {path!r} does not name stored text")
    try:
        parent[path[-1]] = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidResultError(f"invalid JSON value at {path!r}: {error}") from error


def _python_value(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return _python_value(value.item())
    if isinstance(value, np.ndarray):
        return [_python_value(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return [_python_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _python_value(item) for key, item in value.items()}
    raise InvalidResultError(f"h5 metadata contains unsupported {type(value).__name__}")
