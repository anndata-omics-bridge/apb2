"""A parsed level's layer cells leave memory for memory-mapped files, so levels do not add up.

Every level of one conversion is parsed before any is written, because observation alignment
needs every level's observations. Kept in memory, the levels' long cells add up: a DIA-NN
report's fragment, ion and protein levels together exceeded 80 GB where each alone fits. The
result readers park each level they read for the same reason: a 1,236-run DIA-NN result read
back whole took 36 GB.
Parking writes each layer's ``cells`` once to an uncompressed Arrow IPC file and maps it back:
the cells stay an ordinary ``pl.DataFrame``, so every reader of a ``FinalLayerTable`` works
unchanged, but their pages belong to the file, which the system can drop and read again, and
which a process footprint does not count.

The file is unlinked as soon as it is mapped. Its space is freed when the cells are dropped or
the process ends, failed or not, so nothing is left to clean up. The cells are sorted by
``var_pos`` and ``obs_pos`` on the way, so a writer can take a range of feature rows as one
slice.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from apb2.parserV2.parse_quant.data.layer_columns import OBS_POS, VAR_POS
from apb2.parserV2.parse_quant.data.parsed import ParsedLevel


@dataclass(frozen=True, slots=True)
class LayerParking:
    """Moves each layer's cells to a memory-mapped file in ``directory`` (the system default)."""

    directory: Path | None = None

    def park(self, level: ParsedLevel, /) -> None:
        """Replace every non-empty layer's cells by their sorted, memory-mapped copy."""
        for layer in level.layers.values():
            if layer.cells.height:
                layer.cells = self._mapped(layer.cells.sort(VAR_POS, OBS_POS))

    def _mapped(self, cells: pl.DataFrame) -> pl.DataFrame:
        handle, name = tempfile.mkstemp(prefix="apb2-cells-", suffix=".arrow", dir=self.directory)
        os.close(handle)
        path = Path(name)
        try:
            cells.write_ipc(path, compression="uncompressed")
            return pl.read_ipc(path, memory_map=True)
        finally:
            path.unlink()
