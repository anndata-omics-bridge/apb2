# Annotate samples

Sample annotation is a post-conversion operation over storage-neutral `ParsedLevels`. APB2's CLI
accepts a generic prolfquapp-style CSV/TSV table and writes the same result format unless the target
suffix deliberately selects another one. Scientific conventions such as ProteoBench compose the
public annotation extension boundary from their own packages.

## Normal commands

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

## Matching

Exact matching is the default. Exact aliases named `<key>_alias` or `<key>_aliases` are supported.
The PEAKS rule opts into token-wise fuzzy matching. Exact pairs are reserved first; a fuzzy pair is
accepted only when it reaches the configured cutoff and is the unambiguous best candidate from both
directions. Accepted corrections and bounded near misses are available through
`annotation.matches` and are persisted with the result.
