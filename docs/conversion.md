# Convert vendor results

`apb2 convert` reads one supported vendor table or a directory of vendor-result tables, selects or loads its rules, parses one or more quantification levels, and writes the selected APB2 storage format.

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

The [support matrix](supported_software.md#quantification-levels-by-rule) shows which packaged rule documents currently produce each level.

`--software` is normally unnecessary. Use it when parameter evidence or source columns leave more
than one packaged rule compatible.

## Convert every compatible level

Omit `LEVEL` to compile and parse every compatible level:

```bash
apb2 convert report.tsv \
    --params search-parameters.txt \
    --format parquet \
    --output results/all-levels
```

This route appends `.parquet`. Each level is parsed independently into one storage-neutral `ParsedLevels` value, then APB2's public writer persists that value. The default `--format hdf5` appends `.h5mu`; `--format duckdb` appends `.duckdb`.

## Vendor-result directories

Pass a directory when one vendor reports different quantification levels in separate tables:

```bash
apb2 convert maxquant-results --params mqpar.xml --output results/maxquant
apb2 convert maxquant-results peptide --params mqpar.xml --output results/peptides
```

MaxQuant accepts any nonempty subset of its four exports. Evidence is parsed directly into ions keyed by `Raw_File`, retaining `Experiment` and `Fraction` as metadata. The other exports are unpivoted and joined through shared evidence-ID references plus experiment; their table group declares `Experiment` once in `base`. No evidence rows enter that join. Omitting `LEVEL` produces only supplied levels; explicitly requesting an unavailable level fails before writing.

AlphaDIA matrix and precursor metadata are one directory input:

```bash
apb2 convert alphadia-results ion \
    --params parameters.txt --software alphadia --output results/ion
```

AlphaDIA 1.12 joins matrix values with precursor metadata; it never substitutes secondary intensities. Ambiguous roles, missing required inputs and conflicting identities fail before writing. Preparation runs once per selected table group and records sources, duration, row count and estimated frame size in that group's `input_preparation` provenance. Direct evidence has no preparation provenance. DIA-NN is unchanged.

## Explicit rule document

Use a schema-0.8 rule document directly when rule selection is owned by the caller:

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

Explicit rules support the same directory bundles as packaged detection. Each table group independently selects direct input or preparation before source-column validation. See [rule authoring](rule-based.md#one-software-multiple-input-tables) for the `tables` structure.

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
--format hdf5, explicit LEVEL  -> BASENAME.h5ad
--format hdf5, omitted LEVEL   -> BASENAME.h5mu
--format parquet               -> BASENAME.parquet
--format duckdb                -> BASENAME.duckdb
```

Without `--output`, APB2 replaces the source suffix with the selected suffix.

Different observation identities are combined only through an explicit, complete one-to-one metadata mapping. For MaxQuant, higher-level experiment labels are then aligned to raw-file keys, retaining `Experiment` metadata without changing measurement values or order. Otherwise APB2 writes separate results: for example, `maxquant.raw_file.h5mu` and `maxquant.experiment.h5mu` for fractionated or unmapped inputs. It never aggregates ions to experiments or duplicates experiment quantities across fractions. Each split output retains the requested format and receives its own JSON sidecar; no unsuffixed combined result is written. An existing combined target is refused before writing split outputs.

The CLI logs every actual output; the Python conversion summary exposes these paths as `outputs`. Observed mappings and alignment decisions are serialized as JSON in `observation_relationships` parse provenance.

See the [CLI reference](cli.md) for the complete command surface, the
[Python API](api.md#convert-vendor-results) for programmatic conversion, and
[result I/O](result_io.md) for changing a persisted format afterwards.
