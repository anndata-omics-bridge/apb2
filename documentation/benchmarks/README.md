# Benchmarks

## `result_io.py` — result-format adapters and crossings

This benchmark constructs storage-neutral `ParsedLevels` values, verifies every result after
reading it, and measures the public Parquet, DuckDB, and h5mu writers and readers. It also measures
Parquet-to-DuckDB and DuckDB-to-h5mu through `ParsedLevels`; there is no backend-to-backend shortcut.

```bash
.venv/bin/python documentation/benchmarks/result_io.py \
    --observations 12 --variables 10000 --layers 3 --repeats 3
```

### Result, 2026-08-25

Apple M4 Pro, Python 3.13.9, Polars 1.43.2, DuckDB 1.5.5, MuData 0.4.1, and PyArrow 25.0.1;
median of three measured runs after one discarded warm-up. The one-level value has 12 observations,
10,000 variables, and three dense layers. The two-level value adds a protein level with 2,500
variables.

| result | Parquet write / read | DuckDB write / read | h5mu write / read | Parquet → DuckDB | DuckDB → h5mu |
| --- | ---: | ---: | ---: | ---: | ---: |
| one level | **0.0048 / 0.0053 s** | 0.0335 / 0.0082 s | 0.0724 / 0.0623 s | 0.0391 s | 0.0832 s |
| two levels | **0.0075 / 0.0084 s** | 0.0495 / 0.0113 s | 0.1023 / 0.0827 s | 0.0582 s | 0.1147 s |

The corresponding one-level files are 0.45 MiB Parquet, 2.01 MiB DuckDB, and 5.28 MiB h5mu; the
two-level files are 0.54 MiB, 4.01 MiB, and 6.66 MiB. This synthetic, highly compressible workload
primarily verifies adapter overhead and scaling direction. It does not replace vendor conversion
benchmarks or claim the same ratios for sparse or string-heavy results.

## `diann_cli_conversion.py` — APB versus APB2 through `.h5ad`

This is the user-visible converter benchmark. It invokes `apb convert` and `apb2 convert` as
separate processes from the same APB2 virtual environment and times the entire command: application
startup, parameter and rule selection, vendor-table reading, conversion, AnnData construction, and
the completed `.h5ad` write.

The current case is the cached DIA-NN v1 ion report used by the cross-generation integration tests.
It is 229,148,700 bytes, has 325,788 source rows, and converts to 6 observations by 72,804 variables
with five named layers. Run one discarded warm-up and five measured pairs:

```bash
diann_data=../apb/test_data_download/json_dir/Results_quant_ion_DIA_AIF/dcfb0316d24e51357eaffc5f9e638bd28da609fe/input_file.txt
diann_params=../apb/test_data_download/json_dir/Results_quant_ion_DIA_AIF/dcfb0316d24e51357eaffc5f9e638bd28da609fe/param_0..txt

.venv/bin/python documentation/benchmarks/diann_cli_conversion.py \
    "$diann_data" "$diann_params" \
    --warmups 1 --repeats 5 \
    --output-directory /tmp/apb-diann-v1-ion \
    --result-json documentation/benchmarks/results/diann_v1_ion_2026-08-23.json
```

Measured order alternates between pairs to reduce order and thermal bias. Input pages are warm after
the discarded run. macOS `/usr/bin/time -lp` supplies wall, user, system, and maximum-RSS values.
Deleting the previous output and checking it are outside the timed interval.

Every pair is then read back. The benchmark aligns observations by `Run` and variables by
`ProForma_ion`, checks all obs and var metadata, and compares every cell in every persisted layer
and `X` with `rtol=1e-9`. A performance result is not reported if parity fails. The JSON record keeps
every raw run, exact commands, package versions, machine details, repository commits and statuses,
summary statistics, and the parity coverage.

### Result, 2026-08-23

Apple M4 Pro, 48 GiB RAM, macOS 26.5.2, Python 3.13.9, AnnData 0.13.2, pandas 3.0.5,
Polars 1.43.2, and NumPy 2.5.2. Both commands came from the same virtual environment. The measured
repositories were clean at `apb@7a2cd96` and `apb2@294da9b`.

| converter | measured wall times | median wall | median user | median system | median peak RSS | output |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| APB | 4.86, 4.78, 4.79, 5.18, 5.00 s | 4.86 s | 4.57 s | 0.26 s | 1,598 MiB | 45.5 MiB |
| APB2 | 2.31, 2.29, 2.29, 2.40, 2.44 s | **2.31 s** | 6.27 s | 0.90 s | **858 MiB** | 45.5 MiB |

For this DIA-NN v1 ion conversion, **APB2 is 2.10 times faster by median wall time and uses 46.3%
less median peak resident memory**. APB2 spends more aggregate user and system CPU time while
finishing sooner, consistent with Polars using parallel work to reduce latency. The output files
differ by 1,040 bytes because their provenance differs; byte identity is not the correctness
criterion.

Correctness was checked after every pair. The 6 observation rows, 72,804 variable rows, all 1 obs
and 11 var metadata columns, the five named layers, and `X` agree after alignment by authored keys.
That is 2,620,944 equal matrix cells. This establishes the result for this one representative
DIA-NN v1 ion workload; it is not yet a claim about other DIA-NN versions, levels, or vendors.

The complete machine-readable record is
[`results/diann_v1_ion_2026-08-23.json`](results/diann_v1_ion_2026-08-23.json).

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

Polars and DuckDB are runtime dependencies of the result-I/O boundary:

```bash
uv sync --group dev
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
| read TSV (9 columns) | **0.107 s** | 0.326 | 2.024 | 1.989 |
| filter rows | 0.001 | 0.060 | 0.001 | 0.000 |
| build axis keys | 0.004 | 0.147 | 0.036 | 0.080 |
| factorize | 0.026 | **0.021** | 0.104 | 0.100 |
| scatter to 3 dense layers | 0.009 | 0.022 | 0.013 | **0.003** |
| deduplicate axis frames | 0.007 | 0.061 | 0.037 | 0.030 |
| build the output tables | 0.008 | 0.033 | 0.014 | 0.014 |
| write 5 Parquet files | **0.056** | 0.120 | 0.151 | 0.152 |
| write 1 DuckDB database | 0.267 | **0.240** | 0.287 | 0.326 |
| **total, convert only** | **0.155 s** | **0.637 s** | **2.216 s** | **2.202 s** |
| **total, convert + Parquet** | **0.219 s** | **0.789 s** | **2.381 s** | **2.368 s** |
| **total, convert + DuckDB** | **0.430 s** | **0.909 s** | **2.517 s** | **2.542 s** |

Three totals because a real run persists to one target, not both: the conversion on its own,
then the conversion followed by each way of storing its result. `build the output tables` is
shared prep and counts in both write totals.

**The entire pandas gap is the TSV reader — 19x against polars, 6x against DuckDB — and nothing
else matters.** Every non-read step costs under 0.33 s in all four. APB's cost is parsing vendor
text, so the reader is the only figure with real leverage.

**Serialization does not reorder the field, but it dominates a fast one.** Storing the result
costs 0.06–0.33 s of writing plus 0.01–0.03 s of preparation, near enough the same for every
engine. Against polars' 0.155 s conversion that is +41% for Parquet and +177% for DuckDB — the
database write alone outweighs everything polars did to get there. Against pandas the same two
writes are +7% and +14%, lost in the reader.
**So the storage format is the second decision, and on a fast engine nearly as large as the
first.**

**Parquet is cheaper to write and smaller on disk.** 2–5x faster than the database write in
every variant, and for the same five tables 18 MB (polars) / 26 MB (DuckDB) / 29 MB (pandas)
against 27–28 MB for every DuckDB database. The spread across writers is default compression,
not the data. The DuckDB write is also the one step where polars does not lead: DuckDB writing
its own format is marginally fastest at 0.240 s.

**The PyArrow dtype backend buys nothing here** (2.381 s against 2.368 s to Parquet, inside the
noise) and conventional pandas is *faster* at the scatter, 0.003 s against 0.013 s. Worth
knowing because `polars-benchmark` hardcodes `dtype_backend="pyarrow"` in its pandas queries, so its published
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
was ~0.2 s faster before the write steps existed, since the process now holds the output tables
while it reads. Every number in the table above comes from one invocation.

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

## `parser_v2_stages.py` — where Parser V2 spends its time

The engine benchmark above chose Polars. This one checks the four cost claims the Parser V2
architecture makes, which are properties of the design rather than of a machine, and are
therefore reported rather than thresholded.

```bash
.venv/bin/python documentation/benchmarks/parser_v2_stages.py stages RULES DATA \
    --level ion --output parquet
.venv/bin/python documentation/benchmarks/parser_v2_stages.py scaling
```

`stages` times what can be constructed on its own — binding, source resolution, the projected
read, decomposition — and then the whole `parse()`; what is left over is axis preparation,
duplicate resolution, and final alignment. `scaling` checks the two claims that are about *what*
a stage scales with. Memory is the process resident high-water mark, because Polars allocates in
Rust where `tracemalloc` cannot see it.

### Result, 2026-08-21

`apb/test_data_download/.../Results_quant_ion_DIA_AIF/.../input_file.txt` — a DIA-NN v1 report,
219 MiB, 325 788 rows, 14 of its 60 columns projected, converted to 6 x 72 804 with 5 layers.
Polars 1.43.2, Python 3.13, 14 cores / 48 GB, warm cache, median of 3 after a discarded warm-up.

| Stage | median | peak RSS |
| --- | ---: | ---: |
| bind and observe the dialect | 0.007 s | |
| resolve the level against the header | <0.001 s | |
| read the 14-column projection | 0.049 s | 1.2 GiB |
| decompose to raw axes and wide layers | 0.552 s | 1.2 GiB |
| `parse()` whole | 0.894 s | 1.3 GiB |
| — of which axis prep, duplicates, alignment | ~0.29 s | |
| write the Parquet dataset | 0.048 s | |
| write the `.h5ad` | 0.800 s | |

Reading is 5% of the parse. Decomposition — the long-to-wide pivot over 325 788 rows — is 62%,
and the axis work that follows it runs on 6 and 72 804 rows rather than on 325 788, which is
what putting it after decomposition buys.

### Spectronaut optimization result, 2026-08-24

The unchanged commit and the optimized implementation were run in isolated processes with the
same `parser_v2_stages.py` command, one discarded warm-up, and three measured repetitions. Input:
the cached 695 MiB Spectronaut ion export, 719,230 rows, 51 projected columns, 6 observations,
124,267 variables, and 20 named layers.

| Stage | unchanged median | optimized median | effect |
| --- | ---: | ---: | ---: |
| projected read | 0.216 s | 0.211 s | unchanged |
| decomposition | 5.726 s | 0.351 s | 16.3× faster |
| whole `parse()` | 7.029 s | 1.620 s | 4.34× faster |
| write `.h5ad` | 3.706 s | 0.817 s | 4.54× faster |
| process peak RSS | 7,137 MiB | 6,416 MiB | 10.1% lower |

The decomposition result comes from sharing the long-table join, occurrence count, multi-value
pivot, and sort across all layer sources. The writer result comes from per-layer Polars expression
evaluation and pre-encoding only the repeated axis strings that AnnData itself would categorize.
Category dictionary order follows Polars' stable column-local order; decoded values remain equal.

A post-change cProfile run took 3.64 s end to end. Modification normalization remained the largest
single component at 1.54 s, of which 1.04 s was the complete token algorithm and 0.40 s materialized
its derived columns. It was deliberately left alone: residue localization, terminal handling,
mass/target matching, unknown-token policy, and ProForma rendering do not have a complete generic
replacement in a few string expressions, and a vendor-specific partial fast path would duplicate
the algorithm.

**Allocation, measured rather than asserted.** The Parquet write allocates **no** dense array
for 5 layers. The AnnData write allocates **exactly 5**, all `(72804, 6)`: one per encoded layer
and nothing else.

**Modification normalization scales with distinct variables, not with rows.** At a fixed 200 000
rows, going from 1 000 to 100 000 distinct sequences moves it from 0.023 s to 0.398 s — a 17x
cost for a 100x growth in distinct work, with the row count unchanged. That is why normalization
runs on `VarRaw` and not on the source table.

| distinct sequences (200 000 rows) | median |
| --- | ---: |
| 1 000 | 0.023 s |
| 10 000 | 0.059 s |
| 100 000 | 0.398 s |

**Duplicate resolution stays vectorized.** 10 000 raw keys resolved across 6, 60, and 600
observation columns: 0.001 s, 0.005 s, 0.039 s — linear in the columns, with no per-cell Python.
