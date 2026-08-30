"""Benchmark APB2 result writers, readers, and representative format crossings."""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import median

import polars as pl
from cyclopts import App
from loguru import logger

from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.formats import ResultFormat, reader_for, reformat, writer_for

app = App(name="result-io-benchmark", help=__doc__)


def _plan(layer_names: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "ann_data": {
                "layer_encodings": [
                    {
                        "kind": "plain_numeric",
                        "layer_name": name,
                        "missing_values": [],
                        "number_format": {"decimal_mark": ".", "thousands_marks": []},
                    }
                    for name in layer_names
                ],
                "layer_contract": {
                    "primary_layer_name": layer_names[0],
                    "required_names": [layer_names[0]],
                    "empty_ratio": 0.001,
                    "populated_ratio": 0.5,
                },
            }
        }
    )


def _level(name: str, observations: int, variables: int, layers: int) -> ParsedLevel:
    obs_names = [f"run_{index}" for index in range(observations)]
    var_names = [f"feature_{index}" for index in range(variables)]
    value_columns = {
        f"obs_{obs}": [float(obs + var) for var in range(variables)] for obs in range(observations)
    }
    layer_names = tuple(f"layer_{index}" for index in range(layers))
    layer_values = pl.DataFrame({"feature": var_names, **value_columns})
    return ParsedLevel(
        obs=ObsFinal(
            pl.DataFrame({"run": obs_names, "batch": [f"b{i % 3}" for i in range(observations)]}),
            ("run",),
        ),
        var=VarFinal(
            pl.DataFrame({"feature": var_names, "charge": [(i % 4) + 1 for i in range(variables)]}),
            ("feature",),
        ),
        primary_layer_name=layer_names[0],
        layers={
            layer_name: FinalLayerTable(layer_name, ("feature",), layer_values)
            for layer_name in layer_names
        },
        obsm={"design": pl.DataFrame({"group": [f"g{i % 2}" for i in range(observations)]})},
        varm={},
        obsp={},
        varp={},
        uns={
            "quantification_level": name,
            "software_name": "synthetic",
            "plan_json": _plan(layer_names),
        },
    )


def _timed(run: Callable[[], object], repeats: int) -> float:
    run()
    values: list[float] = []
    for _attempt in range(repeats):
        start = time.perf_counter()
        run()
        values.append(time.perf_counter() - start)
    return median(values)


def _measure(parsed: ParsedLevels, directory: Path, repeats: int) -> None:
    paths = {
        ResultFormat.PARQUET: directory / "result.parquet",
        ResultFormat.DUCKDB: directory / "result.duckdb",
        ResultFormat.H5MU: directory / "result.h5mu",
    }
    for result_format, path in paths.items():
        write_seconds = _timed(partial(writer_for(result_format).write, parsed, path), repeats)
        read_seconds = _timed(partial(reader_for(result_format).read, path), repeats)
        restored = reader_for(result_format).read(path)
        if list(restored.levels) != list(parsed.levels):
            raise AssertionError("a timed adapter changed level order")
        logger.info(
            "format={} write={:.4f}s read={:.4f}s bytes={}",
            result_format,
            write_seconds,
            read_seconds,
            sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            if path.is_dir()
            else path.stat().st_size,
        )
    parquet_to_duckdb = _timed(
        lambda: reformat(paths[ResultFormat.PARQUET], directory / "crossing.duckdb"), repeats
    )
    duckdb_to_h5mu = _timed(
        lambda: reformat(paths[ResultFormat.DUCKDB], directory / "crossing.h5mu"), repeats
    )
    logger.info("parquet->duckdb={:.4f}s duckdb->h5mu={:.4f}s", parquet_to_duckdb, duckdb_to_h5mu)


@app.default
def benchmark(
    observations: int = 12,
    variables: int = 10_000,
    layers: int = 3,
    repeats: int = 3,
) -> None:
    """Measure one- and two-level results in a fresh temporary directory."""
    ion = _level("ion", observations, variables, layers)
    one = ParsedLevels(levels={"ion": ion}, uns={"case": "one-level"})
    protein = _level("protein", observations, variables // 4, layers)
    two = ParsedLevels(
        levels={"ion": ion, "protein": protein},
        uns={"case": "two-level"},
    )
    with tempfile.TemporaryDirectory(prefix="apb2-result-io-") as scratch:
        root = Path(scratch)
        logger.info("one level: obs={} var={} layers={}", observations, variables, layers)
        logger.disable("apb2")
        try:
            _measure(one, root / "one", repeats)
        finally:
            logger.enable("apb2")
        logger.info("two levels: ion_var={} protein_var={}", variables, variables // 4)
        logger.disable("apb2")
        try:
            _measure(two, root / "two", repeats)
        finally:
            logger.enable("apb2")


if __name__ == "__main__":
    app()
