# Annotate samples

Sample annotation is a post-conversion operation over storage-neutral `ParsedLevels`. APB2's CLI
accepts an SDRF-Proteomics or a generic prolfquapp-style CSV/TSV table and writes the same result format unless the target
suffix deliberately selects another one. Scientific conventions such as ProteoBench compose the
public annotation extension boundary from their own packages.

## Command-line examples

Keep all observations and attach null metadata where a prolfquapp row is absent:

```bash
apb2 annotate input.h5mu samples.tsv annotated.h5mu
```

Require complete coverage:

```bash
apb2 annotate input.h5mu samples.tsv annotated.h5mu \
  --unmatched error
```

Use annotation membership as an explicit sample allowlist:

```bash
apb2 annotate input.h5mu samples.tsv selected.h5mu \
  --unmatched drop
```

Also require a true Boolean annotation field:

```bash
apb2 annotate input.h5mu samples.tsv selected.h5mu \
  --unmatched drop --include include
```

ProteoBench module semantics are owned by `apb-proteobench`:

```bash
apb-proteobench annotate input.h5mu module_settings.toml annotated.h5mu
```

## Diagnostics and set meanings

For observation keys `Q` and annotation keys `A`, `quant_only` is `Q - A` after accepted fuzzy
corrections and `annotation_only` is `A - Q`.

- prolfquapp warns for `annotation_only`: the annotation declares rows absent from quantification.
- prolfquapp reports `quant_only` at information level: partial annotation may be deliberate.
- An external convention selects its own policy. `apb-proteobench` rejects both unmatched
  observations and unused module samples before constructing a dataset-bound annotation.

Dropping observations subsets `obs`, every layer observation column, every `obsm` row, and both
axes of every `obsp` matrix while remapping coordinates. It never filters only `obs`.

## SDRF tables

A table whose headers include `source name` and `comment[data file]`, compared case-insensitively, is read as SDRF-Proteomics; this takes precedence over prolfquapp recognition.

- Rows are keyed by `comment[data file]`. The extensionless basename is an exact alias, so `run_A.raw` matches a vendor run named `run_A` without a rule-declared normalization.
- Every other column is added to `obs` under its sanitized name. Repeated headers stay separate: two `characteristics[spiked compound]` columns become `characteristics_spiked_compound` and `characteristics_spiked_compound_duplicated_0`.
- `sdrf.provenance.annotation.columns` maps each verbatim header to its `obs` column.
- Only label-free rows are supported; multiplexed `comment[label]` values are rejected because they need run-and-channel observation keys.
- Values are preserved as written. APB2 does not validate SDRF structure or ontology terms; use the official `sdrf-pipelines` validator for that.
- `--unmatched` and `--include` behave as for prolfquapp tables. A data file listed on two rows is an error.

## Matching

Exact matching is the default. Exact aliases named `<key>_alias` or `<key>_aliases` are supported.
A vendor rule may persist `normalize: "mass_spec_basename"` for either exact or fuzzy matching.
This compares only extensionless basenames after removing path components and the case-insensitive
`.mzML`, `.mzML.gz`, `.raw`, `.mgf`, `.d`, or `.wiff` suffix; original observation values and
diagnostics remain unchanged, and two distinct identifiers that normalize to the same value are an
error. The PEAKS rule opts into token-wise fuzzy matching. Exact pairs are reserved first; a fuzzy
pair is accepted only when it reaches the configured cutoff and is the unambiguous best candidate
from both directions. Accepted corrections and bounded near misses are available through
`annotation.matches` and are persisted with the result.

See the [Python API](api.md#annotate-samples) to inspect matching evidence or apply annotation to
an in-memory `ParsedLevels` value.
