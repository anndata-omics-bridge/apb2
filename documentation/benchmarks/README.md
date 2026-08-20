# Benchmarks

## `long_table_conversion.py` — pandas or polars for the parse core?

Times the steps `parse_quant/table_conversion.py` performs on a long-format vendor export:
read the declared columns, filter rows, build string axis keys, factorize them, scatter the
long rows into dense `obs x var` layers, deduplicate the axis frames.

**Why not a published benchmark.** TPC-H suites, `pola-rs/polars-benchmark` included, measure
joins and aggregations over narrow tables. APB never joins. Running that suite answers a
question APB does not ask; this script answers the one it does. (For reference, polars ran all
22 TPC-H queries at scale factor 5 in 2.7 s while pandas needed ~20 s for query 5 alone — the
same direction, on the wrong workload.)

### Running it

`polars` is in the `bench` dependency group, not in the runtime dependencies:

```bash
uv sync --group dev --group bench
.venv/bin/python documentation/benchmarks/long_table_conversion.py --table path/to/input_file.tsv
```

Column names default to a Spectronaut ion-level export; override `--obs-key`, `--var-keys`,
`--layers`, `--carry`, and `--filter-column` for another vendor. `--repeats` sets the timed
runs per variant (one warm-up per variant runs first and is discarded).

Every variant must produce the same matrices or the run fails before reporting. That check is
not decoration — it caught two real defects while this script was being written, both below.

### Result, 2026-08-20

Spectronaut diaPASEF export, 706 MB TSV, 752 206 rows x 67 columns, 9 columns read, filtered to
`EG.Qvalue <= 0.01`, producing a 6 x 131 920 matrix with 747 131 finite cells. pandas 3.0.5,
polars 1.43.2, pyarrow 25.0.1, Python 3.13, 14 cores / 48 GB, warm page cache, median of 3.

| step | polars 1.43 | pandas 3.0 + pyarrow | pandas 3.0 conventional |
| --- | --- | --- | --- |
| read TSV (9 columns) | **0.103 s** | 1.928 | 1.916 |
| filter rows | 0.001 | 0.001 | 0.000 |
| build axis keys | 0.004 | 0.036 | 0.083 |
| factorize | 0.026 | 0.105 | 0.101 |
| scatter to 3 dense layers | 0.007 | 0.013 | **0.003** |
| deduplicate axis frames | 0.007 | 0.038 | 0.031 |
| **total** | **0.148 s** | **2.121 s** | **2.134 s** |

**The entire difference is the TSV reader — 19x — and nothing else matters.** Every non-read
step costs under 0.11 s in all three; together they are 9% of pandas' time. APB's cost is
parsing vendor text, so the reader is the only figure with leverage.

**The PyArrow dtype backend buys nothing here** (2.121 s against 2.134 s, inside the noise) and
conventional pandas is *faster* at the scatter, 0.003 s against 0.013 s. Worth knowing because
`polars-benchmark` hardcodes `dtype_backend="pyarrow"` in its pandas queries, so its published
"pandas" numbers already are the PyArrow variant.

**The AnnData boundary is free.** `polars.DataFrame.to_pandas()` on the 131 920 x 9 variable
frame is 0.002 s. AnnData 0.13 requires pandas for `.obs`/`.var`, so a polars core converts at
the storage edge — at no measurable cost.

### Two traps this script exists to document

**Polars' categorical codes are not dense per column.** `Series.cast(pl.Categorical)
.to_physical()` returns codes from a string cache shared across the frame: a six-value column
carried a maximum code of 117 092. Sizing a matrix from `max() + 1` therefore asks for a
117 093 x 131 926 allocation. The first draft of this benchmark did exactly that and reported
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

### Not yet measured

The fragment-explode and modification-applier paths, which are string-heavy and are where the
remaining risk in a migration sits.
