"""Benchmark APB's long-table conversion under pandas, polars, and DuckDB.

Why this and not a published benchmark: the TPC-H suites (including
`pola-rs/polars-benchmark`) measure joins and aggregations over narrow tables. APB never
joins. It reads one wide long-format vendor export, builds string axis keys, deduplicates
them, and scatters the long rows into dense matrices via integer category codes. This script
runs exactly those steps -- the ones in ``parse_quant/table_conversion.py`` -- so the
choice-of-engine question is answered on APB's own shape.

Every variant must produce the same matrices; the script asserts that before reporting
timings, because a wrong factorize can produce correct output at absurd cost (see the
``polars`` note in ``documentation/benchmarks/README.md``).

Run it with a vendor file APB can already parse::

    uv run --group bench python documentation/benchmarks/long_table_conversion.py \
        --table path/to/input_file.tsv
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from cyclopts import App
from loguru import logger
from numpy.typing import NDArray

Variant = Literal["polars", "duckdb", "pandas-pyarrow", "pandas-numpy"]

STEPS = ("read", "filter", "keys", "factorize", "scatter", "axis_frames")

app = App(name="long-table-conversion", help=__doc__)


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Which columns of a long vendor export the benchmarked conversion uses.

    Defaults describe a Spectronaut ion-level export. Point ``--table`` at another vendor and
    override the column names; the conversion itself does not change.
    """

    obs_key: str
    var_keys: tuple[str, ...]
    layers: tuple[str, ...]
    carry: tuple[str, ...]
    filter_column: str
    filter_maximum: float

    @property
    def columns(self) -> list[str]:
        """Every column read from the file, in one list, without duplicates."""
        seen: dict[str, None] = {}
        columns = (self.obs_key, *self.var_keys, *self.layers, *self.carry, self.filter_column)
        for name in columns:
            seen[name] = None
        return list(seen)


@dataclass(frozen=True, slots=True)
class Conversion:
    """One variant's result: the dense layers plus the axis sizes it derived."""

    n_obs: int
    n_var: int
    layers: dict[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class Timings:
    """Per-step seconds for one run of one variant."""

    variant: Variant
    steps: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.steps.values())


class _Clock:
    """Accumulates per-step durations for one run."""

    def __init__(self) -> None:
        self.steps: dict[str, float] = {}

    def step[T](self, name: str, work: Callable[[], T]) -> T:
        gc.collect()
        start = time.perf_counter()
        result = work()
        self.steps[name] = time.perf_counter() - start
        return result


def _scatter(
    obs_codes: NDArray[np.int64],
    var_codes: NDArray[np.int64],
    values: Iterable[tuple[str, NDArray[np.float64]]],
    n_obs: int,
    n_var: int,
) -> dict[str, NDArray[np.float64]]:
    """Place long-format values into dense ``n_obs x n_var`` matrices.

    The shared half of the conversion: both dataframe libraries hand over integer codes and
    float arrays, and the scatter is plain NumPy either way. Keeping it common means the
    timings compare the dataframe work rather than two spellings of the same loop.
    """
    flat = obs_codes * n_var + var_codes
    layers: dict[str, NDArray[np.float64]] = {}
    for name, column in values:
        matrix = np.full(n_obs * n_var, np.nan, dtype=np.float64)
        matrix[flat] = column
        layers[name] = matrix.reshape(n_obs, n_var)
    return layers


def run_pandas(
    table: Path, spec: TableSpec, *, pyarrow_backend: bool
) -> tuple[Conversion, Timings]:
    """Convert with pandas, on either the PyArrow or the conventional NumPy dtype backend."""
    import pandas as pd

    clock = _Clock()

    def read() -> pd.DataFrame:
        if pyarrow_backend:
            return pd.read_csv(
                table,
                sep="\t",
                usecols=spec.columns,
                encoding="utf-8-sig",
                dtype_backend="pyarrow",
            )
        return pd.read_csv(table, sep="\t", usecols=spec.columns, encoding="utf-8-sig")

    frame = clock.step("read", read)
    frame = clock.step(
        "filter",
        lambda: frame[frame[spec.filter_column] <= spec.filter_maximum],
    )

    def build_keys() -> tuple[pd.Series, pd.Series]:
        observations = frame[spec.obs_key].astype("string")
        variables = frame[spec.var_keys[0]].astype("string")
        for key in spec.var_keys[1:]:
            variables = variables + "_" + frame[key].astype("string")
        return observations, variables

    obs_key, var_key = clock.step("keys", build_keys)

    def factorize() -> tuple[pd.Categorical, pd.Categorical]:
        # Explicit categories in first-appearance order, because that is what
        # `table_conversion.LongConversion` does: it passes the deduplicated axis frame's
        # index, and `drop_duplicates` keeps first occurrence. A bare `pd.Categorical`
        # would sort the categories and permute the axes.
        return (
            pd.Categorical(obs_key, categories=pd.unique(obs_key)),
            pd.Categorical(var_key, categories=pd.unique(var_key)),
        )

    obs_cat, var_cat = clock.step("factorize", factorize)
    n_obs, n_var = len(obs_cat.categories), len(var_cat.categories)

    layers = clock.step(
        "scatter",
        lambda: _scatter(
            obs_cat.codes.astype(np.int64),
            var_cat.codes.astype(np.int64),
            (
                (
                    name,
                    pd.to_numeric(frame[name], errors="coerce").to_numpy(
                        dtype=np.float64, na_value=np.nan
                    ),
                )
                for name in spec.layers
            ),
            n_obs,
            n_var,
        ),
    )
    clock.step(
        "axis_frames",
        lambda: (
            frame.assign(_key=obs_key).drop_duplicates(subset="_key"),
            frame.assign(_key=var_key).drop_duplicates(subset="_key"),
        ),
    )

    variant: Variant = "pandas-pyarrow" if pyarrow_backend else "pandas-numpy"
    return Conversion(n_obs, n_var, layers), Timings(variant, clock.steps)


def run_polars(table: Path, spec: TableSpec) -> tuple[Conversion, Timings]:
    """Convert with polars."""
    import polars as pl

    clock = _Clock()

    frame = clock.step(
        "read",
        lambda: pl.read_csv(
            table,
            separator="\t",
            columns=spec.columns,
            infer_schema_length=10_000,
            null_values=["", "NaN"],
        ),
    )
    frame = clock.step(
        "filter",
        lambda: frame.filter(pl.col(spec.filter_column) <= spec.filter_maximum),
    )
    frame = clock.step(
        "keys",
        lambda: frame.with_columns(
            _obs=pl.col(spec.obs_key).cast(pl.String),
            _var=pl.concat_str(
                [pl.col(key).cast(pl.String) for key in spec.var_keys], separator="_"
            ),
        ),
    )

    def factorize() -> pl.DataFrame:
        # A Categorical's physical codes are NOT dense per column: polars shares a string
        # cache across the frame, so a six-value column can carry six-figure codes and sizing
        # a matrix off `max()` allocates hundreds of gigabytes. Joining against the ordered
        # unique values is the factorize that matches pandas' dense `Categorical.codes`.
        out = frame
        for name in ("_obs", "_var"):
            levels = out.select(name).unique(maintain_order=True).with_row_index(f"{name}_code")
            # `maintain_order="left"` keeps the long rows in file order, so a duplicated
            # (obs, var) key resolves to the same row as it does under pandas. APB decides
            # duplicates explicitly (`parse_quant/duplicates.py`); the benchmark only has to
            # avoid introducing a difference of its own.
            out = out.join(levels, on=name, how="left", maintain_order="left")
        return out

    frame = clock.step("factorize", factorize)
    n_obs = frame.select(pl.col("_obs_code").n_unique()).item()
    n_var = frame.select(pl.col("_var_code").n_unique()).item()

    layers = clock.step(
        "scatter",
        lambda: _scatter(
            frame["_obs_code"].cast(pl.Int64).to_numpy(),
            frame["_var_code"].cast(pl.Int64).to_numpy(),
            ((name, frame[name].cast(pl.Float64).to_numpy()) for name in spec.layers),
            n_obs,
            n_var,
        ),
    )
    clock.step(
        "axis_frames",
        lambda: (
            frame.unique(subset="_obs", keep="first"),
            frame.unique(subset="_var", keep="first"),
        ),
    )

    return Conversion(n_obs, n_var, layers), Timings("polars", clock.steps)


def _sql_identifier(name: str) -> str:
    """Quote a vendor column name for SQL: they contain dots, spaces, and parentheses."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _dense_float(column: object) -> NDArray[np.float64]:
    """Fill a possibly-masked DuckDB column with NaN and cast to float64.

    ``fetchnumpy`` returns a masked array for any nullable column -- and, for a dictionary
    type, something that is not an array at all -- so this decodes the one case the scatter
    can use and rejects the rest rather than silently producing an object array.
    """
    if isinstance(column, np.ma.MaskedArray):
        filled: NDArray[np.float64] = np.ma.filled(column.astype(np.float64), np.nan)
        return filled
    if isinstance(column, np.ndarray):
        return column.astype(np.float64)
    raise TypeError(f"expected a numeric DuckDB column, got {type(column).__name__}")


def run_duckdb(table: Path, spec: TableSpec) -> tuple[Conversion, Timings]:
    """Convert with DuckDB, doing read, filter, keys, and factorize in SQL.

    The engine is a query processor rather than a dataframe library, so the steps are the same
    six operations expressed as statements over temporary tables. Each one is materialized with
    ``CREATE OR REPLACE TABLE`` so a step's cost lands in that step rather than being deferred
    into the next by lazy evaluation.
    """
    import duckdb

    clock = _Clock()
    connection = duckdb.connect()

    def read() -> None:
        selected = ", ".join(_sql_identifier(name) for name in spec.columns)
        connection.execute(
            f"CREATE OR REPLACE TABLE source AS SELECT {selected} FROM "
            f"read_csv({_sql_literal(str(table))}, delim='\t', header=true, "
            f"sample_size=10240, nullstr=['', 'NaN'])"
        )

    clock.step("read", read)
    clock.step(
        "filter",
        lambda: connection.execute(
            "CREATE OR REPLACE TABLE filtered AS SELECT * FROM source "
            f"WHERE {_sql_identifier(spec.filter_column)} <= {spec.filter_maximum}"
        ),
    )

    def build_keys() -> None:
        var_parts = ", ".join(f"CAST({_sql_identifier(key)} AS VARCHAR)" for key in spec.var_keys)
        # `row_number() OVER ()` numbers the rows in the order the reader produced them, which
        # is file order: DuckDB preserves insertion order by default. Every later step sorts or
        # groups by it, so duplicate keys resolve to the same long row as under pandas.
        connection.execute(
            "CREATE OR REPLACE TABLE keyed AS SELECT *, "
            "row_number() OVER () - 1 AS _row, "
            f"CAST({_sql_identifier(spec.obs_key)} AS VARCHAR) AS _obs, "
            f"concat_ws('_', {var_parts}) AS _var FROM filtered"
        )

    clock.step("keys", build_keys)

    def factorize() -> None:
        # First appearance, not sorted: `LongConversion` takes its categories from a
        # `drop_duplicates`, so the axis order is the order the keys first occur in the file.
        for axis in ("obs", "var"):
            connection.execute(
                f"CREATE OR REPLACE TABLE {axis}_levels AS "
                f"SELECT _{axis}, row_number() OVER (ORDER BY first_row) - 1 AS _{axis}_code "
                f"FROM (SELECT _{axis}, min(_row) AS first_row FROM keyed GROUP BY _{axis})"
            )

    clock.step("factorize", factorize)
    n_obs = connection.execute("SELECT count(*) FROM obs_levels").fetchone()
    n_var = connection.execute("SELECT count(*) FROM var_levels").fetchone()
    assert n_obs is not None and n_var is not None
    n_obs, n_var = int(n_obs[0]), int(n_var[0])

    def scatter() -> dict[str, NDArray[np.float64]]:
        aliases = {f"layer_{index}": name for index, name in enumerate(spec.layers)}
        projected = ", ".join(
            f"TRY_CAST(keyed.{_sql_identifier(name)} AS DOUBLE) AS {alias}"
            for alias, name in aliases.items()
        )
        columns = connection.execute(
            "SELECT obs_levels._obs_code, var_levels._var_code, "
            f"{projected} FROM keyed "
            "JOIN obs_levels USING (_obs) JOIN var_levels USING (_var) "
            "ORDER BY keyed._row"
        ).fetchnumpy()
        return _scatter(
            _dense_float(columns["_obs_code"]).astype(np.int64),
            _dense_float(columns["_var_code"]).astype(np.int64),
            ((name, _dense_float(columns[alias])) for alias, name in aliases.items()),
            n_obs,
            n_var,
        )

    layers = clock.step("scatter", scatter)

    def axis_frames() -> None:
        for axis in ("obs", "var"):
            connection.execute(
                f"CREATE OR REPLACE TABLE {axis}_frame AS SELECT * FROM keyed "
                f"QUALIFY row_number() OVER (PARTITION BY _{axis} ORDER BY _row) = 1"
            )

    clock.step("axis_frames", axis_frames)
    connection.close()

    return Conversion(n_obs, n_var, layers), Timings("duckdb", clock.steps)


def _require_agreement(results: dict[Variant, Conversion]) -> None:
    """Reject the run unless every variant produced the same matrices."""
    reference_name, reference = next(iter(results.items()))
    for name, conversion in results.items():
        if name == reference_name:
            continue
        shape = (conversion.n_obs, conversion.n_var)
        expected = (reference.n_obs, reference.n_var)
        if shape != expected:
            raise ValueError(f"{name} produced shape {shape}, {reference_name} produced {expected}")
        for layer, matrix in reference.layers.items():
            other = conversion.layers[layer]
            if np.isfinite(other).sum() != np.isfinite(matrix).sum():
                raise ValueError(f"{name} and {reference_name} fill layer {layer!r} differently")
            # Tolerance, not exact equality: the readers agree on float64 values, but a
            # duplicated key can still resolve to a different row, and text-to-float parsing
            # is not guaranteed bit-identical across libraries.
            if not np.allclose(other, matrix, rtol=1e-9, atol=0.0, equal_nan=True):
                raise ValueError(f"{name} and {reference_name} disagree on layer {layer!r}")


def _report(runs: Sequence[Timings], results: dict[Variant, Conversion]) -> None:
    """Log the per-step medians, one column per variant."""
    variants = list(dict.fromkeys(run.variant for run in runs))
    medians = {
        variant: {
            step: statistics.median([run.steps[step] for run in runs if run.variant == variant])
            for step in STEPS
        }
        for variant in variants
    }
    reference = next(iter(results.values()))
    logger.info(f"matrix {reference.n_obs} x {reference.n_var}, every variant in agreement")
    header = f"{'step':14}" + "".join(f"{variant:>18}" for variant in variants)
    logger.info(header)
    for step in STEPS:
        row = f"{step:14}" + "".join(f"{medians[variant][step]:18.3f}" for variant in variants)
        logger.info(row)
    total = f"{'TOTAL':14}" + "".join(
        f"{sum(medians[variant].values()):18.3f}" for variant in variants
    )
    logger.info(total)


@app.default
def benchmark(
    *,
    table: Path,
    obs_key: str = "R.FileName",
    var_keys: tuple[str, ...] = ("EG.ModifiedSequence", "FG.Charge"),
    layers: tuple[str, ...] = ("FG.Quantity", "EG.TotalQuantity (Settings)", "FG.Mass"),
    carry: tuple[str, ...] = ("PG.ProteinAccessions", "PEP.StrippedSequence", "PG.Qvalue"),
    filter_column: str = "EG.Qvalue",
    filter_maximum: float = 0.01,
    repeats: int = 3,
) -> int:
    """Time APB's long-table conversion under each engine.

    Parameters
    ----------
    table
        A long-format vendor export, as APB reads it.
    obs_key
        Column identifying the run, which becomes the observation axis.
    var_keys
        Columns whose combination identifies one ion, which becomes the variable axis.
    layers
        Quantitative columns to scatter into dense matrices.
    carry
        Further columns the axis frames carry, so the read is the size APB's read is.
    filter_column, filter_maximum
        The row filter a rule applies before conversion.
    repeats
        Timed runs per variant. One warm-up run per variant runs first and is discarded.
    """
    spec = TableSpec(
        obs_key=obs_key,
        var_keys=var_keys,
        layers=layers,
        carry=carry,
        filter_column=filter_column,
        filter_maximum=filter_maximum,
    )
    if not table.exists():
        logger.error(f"no such table: {table}")
        return 1

    variants: dict[Variant, Callable[[], tuple[Conversion, Timings]]] = {
        "polars": lambda: run_polars(table, spec),
        "duckdb": lambda: run_duckdb(table, spec),
        "pandas-pyarrow": lambda: run_pandas(table, spec, pyarrow_backend=True),
        "pandas-numpy": lambda: run_pandas(table, spec, pyarrow_backend=False),
    }

    results: dict[Variant, Conversion] = {}
    runs: list[Timings] = []
    for name, convert in variants.items():
        logger.info(f"warming {name}")
        conversion, _ = convert()
        results[name] = conversion
        for index in range(repeats):
            _, timings = convert()
            runs.append(timings)
            logger.debug(f"{name} run {index + 1}: {timings.total:.3f} s")

    _require_agreement(results)
    _report(runs, results)
    return 0


if __name__ == "__main__":
    app()
