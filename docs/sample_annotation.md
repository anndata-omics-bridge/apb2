# Annotate samples

Sample annotation is a post-conversion operation over storage-neutral `ParsedLevels`. It accepts a
prolfquapp CSV/TSV table or a ProteoBench TOML file and writes the same APB2 result format unless the
target suffix deliberately selects another one.

## Normal commands

Keep all observations and attach null metadata where a prolfquapp row is absent:

```bash
apb2 annotate input.h5mu samples.tsv annotated.h5mu --type prolfquapp
```

Require complete coverage:

```bash
apb2 annotate input.h5mu samples.tsv annotated.h5mu \
  --type prolfquapp --unmatched error
```

Use annotation membership as an explicit sample allowlist:

```bash
apb2 annotate input.h5mu samples.tsv selected.h5mu \
  --type prolfquapp --unmatched drop
```

Also require a true Boolean annotation field:

```bash
apb2 annotate input.h5mu samples.tsv selected.h5mu \
  --type prolfquapp --unmatched drop --include include
```

ProteoBench always requires complete coverage:

```bash
apb2 annotate input.h5mu module_settings.toml annotated.h5mu --type proteobench
```

## Diagnostics and set meanings

For observation keys `Q` and annotation keys `A`, `quant_only` is `Q - A` after accepted fuzzy
corrections and `annotation_only` is `A - Q`.

- prolfquapp warns for `annotation_only`: the annotation declares rows absent from quantification.
- prolfquapp reports `quant_only` at information level: partial annotation may be deliberate.
- ProteoBench rejects non-empty `quant_only` before constructing an annotation.

Dropping observations subsets `obs`, every layer observation column, every `obsm` row, and both
axes of every `obsp` matrix while remapping coordinates. It never filters only `obs`.

## Matching

Exact matching is the default. Exact aliases named `<key>_alias` or `<key>_aliases` are supported.
The PEAKS rule opts into token-wise fuzzy matching. Exact pairs are reserved first; a fuzzy pair is
accepted only when it reaches the configured cutoff and is the unambiguous best candidate from both
directions. Accepted corrections and bounded near misses are available through
`annotation.matches` and are persisted with the result.
