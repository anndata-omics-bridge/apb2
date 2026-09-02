# Convert vendor results

`apb2 convert` reads one supported vendor export, selects or loads its rules, parses one or more
quantification levels, and writes AnnData or MuData.

## Packaged rules

The normal route uses the vendor parameter file and source header to select a packaged rule:

```bash
apb2 convert DATA LEVEL --params PARAMETER_FILE [--software VENDOR] [--output BASENAME]
```

For example:

```bash
apb2 convert report.tsv ion \
    --params search-parameters.txt \
    --software spectronaut \
    --output results/ion
```

APB2 appends `.h5ad` to the output basename. `LEVEL` is one of:

- `ion`
- `peptidoform`
- `peptide`
- `protein`
- `fragment`

The [support matrix](supported_software.md#quantification-levels-by-rule) shows which packaged rule
documents currently produce each level. In particular, `peptide` is part of the public level
vocabulary but is not yet produced by a packaged rule.

`--software` is normally unnecessary. Use it when parameter evidence or source columns leave more
than one packaged rule compatible.

## Convert every compatible level

Omit `LEVEL` to compile and parse every compatible level:

```bash
apb2 convert report.tsv \
    --params search-parameters.txt \
    --output results/all-levels
```

This route appends `.h5mu`. Each level is still parsed independently; MuData is the storage
container that holds the resulting modalities.

## Explicit rule document

Use a schema-0.3 rule document directly when rule selection is owned by the caller:

```bash
apb2 convert report.tsv ion \
    --rule-config rules.json \
    --output results/ion
```

Parameter evidence remains optional on this route:

```bash
apb2 convert report.tsv ion \
    --rule-config rules.json \
    --params search-parameters.txt \
    --params-software spectronaut \
    --output results/ion
```

`--params-software` selects the parameter-file grammar independently of the rule document.

## Layer checks

The standard writer reports layer-occupancy problems. Add `--strict` to promote those findings to
conversion errors:

```bash
apb2 convert report.tsv ion --params parameters.txt --strict
```

Strictness changes validation policy only. It does not select a different parser or modify the
scientific result.

## Output naming

`--output` is a basename. Do not include the suffix that APB2 appends:

```text
explicit LEVEL  -> BASENAME.h5ad
omitted LEVEL   -> BASENAME.h5mu
```

Without `--output`, APB2 replaces the source suffix with `.h5ad` or `.h5mu`.

See the [CLI reference](cli.md) for the complete command surface, the
[Python API](api.md#convert-vendor-results) for programmatic conversion, and
[result I/O](result_io.md) for changing a persisted format afterwards.
