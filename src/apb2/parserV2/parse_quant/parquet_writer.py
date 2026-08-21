"""Persist one parsed level as a Parquet directory dataset, losing nothing.

This writer is the proof that ``ParsedLevel`` is storage-neutral. It holds no encoder, builds
no matrix, and constructs no pandas or NumPy value: every frame is written as it came out of
parsing, with its own dtypes, its nulls, and its column order intact. A string layer stays a
string layer, and a localized number stays the token the vendor wrote.

``ParsedLevel`` is several tables, so the target is a directory rather than one file, and a
manifest states what each file is: which columns are axis keys, which are positional
observation labels, which layer is primary, and what provenance came with the parse. A layer
name is a vendor string, so it is encoded to a safe file name and mapped explicitly in the
manifest instead of being interpolated into a path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import FinalLayerTable, JsonValue, ParsedLevel

FORMAT = "apb2-parser-v2-parquet"
FORMAT_VERSION = "1"
MANIFEST_NAME = "manifest.json"
OBS_NAME = "obs.parquet"
VAR_NAME = "var.parquet"
LAYERS_DIRECTORY = "layers"

_UNSAFE = re.compile(r"[^0-9A-Za-z._-]+")


class ParquetWriteError(OSError):
    """Persisting a parsed level failed; the previous target was left as it was."""


@dataclass(frozen=True, slots=True)
class ParquetWriter:
    """Write one parsed level as ``obs``, ``var``, one file per layer, and a manifest."""

    def write(self, parsed: ParsedLevel, target: Path, /) -> None:
        """Stage the whole dataset beside the target, then replace it in one step."""
        file_names = self._file_names(tuple(parsed.layers))
        with TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
            staged = Path(scratch) / target.name
            (staged / LAYERS_DIRECTORY).mkdir(parents=True)
            parsed.obs.frame.write_parquet(staged / OBS_NAME)
            parsed.var.frame.write_parquet(staged / VAR_NAME)
            for name, layer in parsed.layers.items():
                layer.values.write_parquet(staged / LAYERS_DIRECTORY / file_names[name])
            (staged / MANIFEST_NAME).write_text(
                json.dumps(self._manifest(parsed, file_names), indent=2) + "\n",
                encoding="utf-8",
            )
            self._replace(staged, target, scratch=Path(scratch))

    @staticmethod
    def _file_names(layer_names: tuple[str, ...]) -> dict[str, str]:
        """Encode each layer name to a distinct safe file name, in authored order."""
        taken: set[str] = set()
        names: dict[str, str] = {}
        for index, name in enumerate(layer_names):
            safe = _UNSAFE.sub("_", name).strip("._-") or f"layer_{index}"
            candidate = f"{safe}.parquet"
            suffix = 0
            while candidate in taken:
                suffix += 1
                candidate = f"{safe}_{suffix}.parquet"
            taken.add(candidate)
            names[name] = candidate
        return names

    @staticmethod
    def _manifest(parsed: ParsedLevel, file_names: dict[str, str]) -> dict[str, JsonValue]:
        """State what each file is, so nothing has to be inferred from the frames."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "primary_layer": parsed.primary_layer_name,
            "obs": {
                "file": OBS_NAME,
                "key_columns": list(parsed.obs.key_columns),
                "columns": list(parsed.obs.frame.columns),
            },
            "var": {
                "file": VAR_NAME,
                "key_columns": list(parsed.var.key_columns),
                "columns": list(parsed.var.frame.columns),
            },
            "layer_order": list(parsed.layers),
            "layers": {
                name: ParquetWriter._layer_manifest(layer, file_names[name])
                for name, layer in parsed.layers.items()
            },
            "uns": dict(parsed.uns),
        }

    @staticmethod
    def _layer_manifest(layer: FinalLayerTable, file_name: str) -> dict[str, JsonValue]:
        keys = len(layer.var_key_columns)
        return {
            "file": f"{LAYERS_DIRECTORY}/{file_name}",
            "var_key_columns": list(layer.var_key_columns),
            "observation_columns": list(layer.values.columns[keys:]),
        }

    @staticmethod
    def _replace(staged: Path, target: Path, *, scratch: Path) -> None:
        """Swap the staged dataset in, moving any previous one out of the way first.

        The previous target is moved *into* the scratch directory rather than deleted, so the
        only recursive removal this writer performs is the one the temporary directory does to
        itself. A failure before the swap leaves the old target exactly as it was.
        """
        if target.exists():
            if not target.is_dir():
                raise ParquetWriteError(
                    f"{target} exists and is not a directory; a Parquet level is a directory "
                    "dataset"
                )
            logger.info(f"replacing the existing Parquet dataset at {target}")
            target.replace(scratch / f"{target.name}.replaced")
        staged.replace(target)
