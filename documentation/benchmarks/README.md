# Benchmarks

## `long_table_conversion.py` — which engine for the parse core?

Times the steps `parse_quant/table_conversion.py` performs on a long-format vendor export:
read the declared columns, filter rows, build string axis keys, factorize them, scatter the
long rows into dense `obs x var` layers, deduplicate the axis frames. Each variant then
persists what it built, twice — one DuckDB database and one folder of Parquet files — because
converting fast and writing slowly is not a faster engine.

Three engines, four variants: polars, DuckDB, and pandas on each of its two dtype backends.

**Why not a published benchmark.** TPC-H suites, `pola-rs/polars-benchmark` included, measure
joins and aggregations over narrow tables. APB never joins. Running that suite answers a
question APB does not ask; this script answers the one it does. (For reference, polars ran all
22 TPC-H queries at scale factor 5 in 2.7 s while pandas needed ~20 s for query 5 alone — the
same direction, on the wrong workload.)

### Running it

`polars` and `duckdb` are in the `bench` dependency group, not in the runtime dependencies:

```bash
uv sync --group dev --group bench
.venv/bin/python documentation/benchmarks/long_table_conversion.py --table path/to/input_file.tsv
```

Column names default to a Spectronaut ion-level export; override `--obs-key`, `--var-keys`,
`--layers`, `--carry`, and `--filter-column` for another vendor. `--repeats` sets the timed
runs per variant (one warm-up per variant runs first and is discarded).

Every variant must produce the same matrices *and* every serialized copy is read back and
checked against the matrix it came from, or the run fails before reporting. Those checks are not
decoration — the first caught two real defects while this script was being written, both below,
and the layer tables are transposed on the way out, which is exactly where a fast-but-wrong
write would hide.

Output layout, one subtree per variant so the four can be compared instead of overwriting each
other:

```
<output-dir>/<variant>/conversion.duckdb        obs, var, layer_<column> tables
<output-dir>/<variant>/parquet/<table>.parquet  the same tables, one file each
```

`--output-dir` defaults to a fresh temporary directory, reported at the start of the run and
left in place. The axis frames are stored as they are; each layer is stored variable-major —
one row per ion, one column per run, plus the `_var` key — because the variable axis is the one
that grows and a Parquet file with 124 267 columns would be pathological.

### Result, 2026-08-21

`apb/test_data_download/json_dir/Results_quant_ion_DIA_diaPASEF/77d349e2ce7f189e0b80d55a6bbeebde58b539c2/input_file.tsv`
— a Spectronaut diaPASEF ion-level export, 695 MiB, 719 230 rows x 71 columns, 9 columns read,
scattered into a 6 x 124 267 matrix per layer. pandas 3.0.5, polars 1.43.2, DuckDB 1.5.5,
pyarrow 25.0.1, Python 3.13, 14 cores / 48 GB, warm page cache, median of 3 timed runs after a
discarded warm-up.

| step | polars 1.43 | DuckDB 1.5 | pandas 3.0 + pyarrow | pandas 3.0 conventional |
| --- | --- | --- | --- | --- |
| read TSV (9 columns) | **0.109 s** | 0.337 | 2.177 | 2.163 |
| filter rows | 0.002 | 0.063 | 0.001 | 0.000 |
| build axis keys | 0.004 | 0.150 | 0.037 | 0.082 |
| factorize | 0.027 | **0.020** | 0.110 | 0.106 |
| scatter to 3 dense layers | 0.009 | 0.023 | 0.014 | **0.003** |
| deduplicate axis frames | 0.008 | 0.064 | 0.039 | 0.031 |
| build the output tables | 0.008 | 0.034 | 0.014 | 0.014 |
| write 5 Parquet files | **0.056** | 0.124 | 0.158 | 0.157 |
| write 1 DuckDB database | 0.267 | **0.245** | 0.288 | 0.336 |
| **total** | **0.490 s** | **1.061 s** | **2.839 s** | **2.893 s** |

**The entire pandas gap is the TSV reader — 20x against polars, 6x against DuckDB — and nothing
else matters.** Every non-read step costs under 0.35 s in all four. APB's cost is parsing vendor
text, so the reader is the only figure with real leverage.

**Serialization is 0.3–0.5 s and it does not reorder the field.** Writing the same tables costs
polars 0.33 s, DuckDB 0.40 s, pandas 0.46–0.51 s. That is *more than twice* polars' own
conversion time (0.33 s against 0.16 s) and 16% of a pandas run — so on a fast engine the write
is the larger half of the job, and choosing the storage format matters as much as the reader. Parquet is 2–5x cheaper to write than
the DuckDB database, and it is the one step where polars' lead is small and DuckDB's write into
its own format is the fastest of the four.

**Parquet is also smaller, and how much depends on the writer.** Same five tables: polars 18 MB,
DuckDB 26 MB, pandas 29 MB, against 27–28 MB for every DuckDB database. polars' default
compression is the difference, not the data.

**The PyArrow dtype backend buys nothing here** (2.839 s against 2.893 s, inside the noise) and
conventional pandas is *faster* at the scatter, 0.003 s against 0.014 s. Worth knowing because
`polars-benchmark` hardcodes `dtype_backend="pyarrow"` in its pandas queries, so its published
"pandas" numbers already are the PyArrow variant.

**DuckDB lands between them, and wins the factorize.** Deriving the axis codes is a
`GROUP BY` plus a window — the operation a query engine is actually built for — and it is the
one step where DuckDB is fastest. Its overhead is elsewhere: every step here materializes a
table, and the per-row bookkeeping APB needs (file-order row numbers, `QUALIFY` dedup) is
cheap in a dataframe and a window function in SQL. It is also the only variant that hands the
result over the Python boundary as a copy rather than a view.

**The AnnData boundary is free.** `polars.DataFrame.to_pandas()` on a 131 920 x 9 variable
frame is 0.002 s (measured 2026-08-20, on a comparable export). AnnData 0.13 requires pandas for
`.obs`/`.var`, so a non-pandas core converts at the storage edge — at no measurable cost.

Compare columns within one run, not across runs: the same file read by the same pandas variant
was 0.35 s faster before the write steps existed, since the process now holds the output tables
while it reads. Every column in the table above comes from one invocation.

### Three traps this script exists to document

**Polars' categorical codes are not dense per column.** `Series.cast(pl.Categorical)
.to_physical()` returns codes from a string cache shared across the frame: a six-value column
carried a maximum code of 117 092. Sizing a matrix from `max() + 1` therefore asks for a
117 093 x 131 926 allocation (on the export in use when that was measured). The first draft of this benchmark did exactly that and reported
24.8 s for a step that takes 0.007 s — a number that looks like a polars characteristic and is
not. pandas' `Categorical.codes` *are* dense, which is why the mistake survives a port.
Factorizing by joining against the ordered unique values is what the script does now.

**Axis order is a semantic choice, and the libraries differ by default.** A bare
`pd.Categorical` sorts its categories; `unique(maintain_order=True)` keeps first appearance.
`LongConversion` passes `categories=obs_df.index` from a `drop_duplicates`, so **APB's axes are
in first-appearance order** and the polars default matches it while the pandas default does
not. The comparison also pins polars' join to `maintain_order="left"`, because a duplicated
`(obs, var)` key otherwise resolves to a different long row — APB decides duplicates
explicitly in `parse_quant/duplicates.py`, and the benchmark must not introduce a difference of
its own.

**`row_number() OVER ()` fused into a CSV scan serializes the read.** APB needs file order —
duplicate `(obs, var)` keys must resolve to the row pandas would pick — and in SQL that is a
row number over the whole relation. Expressed as one query, with the read inside the same
pipeline, that read takes **1.52 s**; staged into a table first and numbered afterwards it is
**0.42 s**. `WITH ... AS MATERIALIZED` does not help (1.51 s), so this is the window forcing a
single-threaded scan, not a CTE being re-evaluated. The `run_duckdb` variant therefore
materializes each step, which is both the faster shape and the one comparable to the eager
dataframe variants.

### Not yet measured

The fragment-explode and modification-applier paths, which are string-heavy and are where the
remaining risk in a migration sits.
