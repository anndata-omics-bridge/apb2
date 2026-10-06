# Python API

APB2 exposes programmatic boundaries for in-memory vendor parsing, result I/O, and sample annotation. The compiler/parser APIs expose storage-neutral Polars values for composition in larger applications; file-to-file vendor conversion belongs to the CLI command workflow.

## Discover packaged result rules

`get_rules()` lists packaged quant-result rules by APB2 category, without opening a vendor result or parameter file:

```python
from apb2.api import get_rules

for software in get_rules("DIA", level="ion"):
    print(software.software_name)
    for variant in software.variants:
        print(variant.software_version_pattern, variant.levels)
```

The initial categories are `DDA` and `DIA`; a rule can belong to both. Each variant identifies one packaged rule and retains its own supported quantification levels. `software_version_pattern` is the rule's regular-expression version range, not a list of tested releases. The catalogue reports rule availability, not whether an arbitrary file can be parsed without search parameters. Parameter-parser-only software and ProteoBench-specific UI aliases are not included. Unknown categories raise `ValueError`.

`packaged_rule_declarations()` maps each packaged `(rule, level)`, with `rule` named as `RuleVariant.rule` names it, to every distinct `rule_json` text that conversion can store for that level, one per declaration some search-parameter evidence selects.

## Convert vendor results

Use the public compiler when another package needs canonical parsed values:

```python
from pathlib import Path

from apb2.api import ParseRuleCompiler, write_parsed_levels

compiler = ParseRuleCompiler(
    Path("report.tsv"),
    Path("search-parameters.txt"),
    requested_levels=("ion",),
    software="spectronaut",
    checks="standard",
)
parser = compiler.compile()
parsed_levels = parser.parse()
write_parsed_levels(parsed_levels, Path("results/ion.h5ad"))
```

Construction binds the physical source, chooses the parameter parser, parses typed parameters, detects compatible rules, and resolves the requested levels. `compile()` returns one `ParserCollection`; `parse()` returns canonical `ParsedLevels` and performs no write. `compiler.parameters` and `compiler.detection` retain the typed parameter and detection results. Storage is selected only by the target passed to `write_parsed_levels()`.

For a ProteoBench Custom upload, use `ParseRuleCompiler(Path("custom.txt"), None, software="pb_custom", requested_levels=("ion",))`. The packaged rule needs no parameter file; `compiler.parameters` is unavailable in this case, while `compiler.detection` still describes the selected rule.

For vendor exports uploaded without search parameters, the Python API provides a separate constructor:

```python
compiler = ParseRuleCompiler.from_software(
    Path("combined_ion.tsv"),
    software="fragpipe",
    requested_levels=("ion",),
)
parsed_levels = compiler.compile().parse()
```

`software` is required and names the result producer; for DIA-NN output from a FragPipe workflow, use `software="diann"`. APB2 matches the requested levels against that producer's packaged rules using their source columns and declared version signatures. For DIA-NN 2.x, the declared MS1 columns distinguish DDA from the DIA default without a separate acquisition argument. Ambiguous matches and search settings that columns cannot establish still raise errors. `compiler.detection.version` is `None`, `compiler.parameters` is unavailable, and the normal rule provenance remains in the parsed result. The CLI uses this same constructor for `apb2 convert DATA ion --software DIA-NN` when `--params` is omitted.

A caller-authored schema-0.8 rule file replaces detection:

```python
compiler = ParseRuleCompiler.from_rule(
    Path("report.tsv"),
    Path("my_rules.json"),
    Path("search-parameters.txt"),
)
parsed_levels, timings = compiler.compile().parse_with_timings()
```

`requested_levels` defaults to every level the rule declares; the optional parameter file adds search-parameter evidence and `compiler.detection.version`. `parse_with_timings()` also returns one `LevelParseTimings` per level, its read and parse seconds.

Levels whose observation identities differ cannot share one file: `parsed_levels.observation_groups()` returns the independently writable results, aligning only one-to-one key mappings. The `apb2 convert` command writes each group to its own file.

See [Convert vendor results](conversion.md) for rule selection, supported levels, validation, and
output naming.

## Annotate samples

The annotation API constructs an `Annotation` only after the source has been matched and validated
against one `ParsedLevels`:

```python
from pathlib import Path

from apb2.api import AnnotationCompiler, read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("input.h5mu"))
parser = AnnotationCompiler(unmatched="error").compile(Path("samples.tsv"))
annotation = parser.parse(parsed)

for level, match in annotation.matches.levels.items():
    print(level, match.coverage, match.corrections)

result = annotation.annotate()
write_parsed_levels(result.parsed, Path("annotated.h5mu"))
```

`AnnotationCompiler` loads and validates the SDRF or generic delimited source once. The returned parser is
source-bound and can be parsed against several datasets, producing a separate dataset-bound
annotation each time.
`parse(parsed)` raises before constructing an annotation when the selected policy is invalid—for
example, when complete coverage was requested but cannot be met. `annotate()` uses the stored
matches and does not recompute them, and returns an `AnnotationResult`: the annotated `parsed` and one coverage report per level.

`AnnotationCompiler(unmatched="keep" | "error" | "drop", include=None)` decides what happens to observations without an annotation row: keep them with null metadata, raise, or drop them; `include` names a Boolean annotation column that further selects observations and requires `"drop"`. A prolfquapp table is keyed by its one column matching `^raw`, `^file`, `^run`, `^channel` or `^Relative`, with optional `<key>_aliases` list values; every other column becomes an obs column. All tables and matching evidence are Polars-backed values. Failures raise `AnnotationError`.

`SdrfSource.read(path)` reads an SDRF for tools that need its columns beyond annotation. `SdrfSource.columns(header)` returns every occurrence of a repeated header in file order, and `data_file_basenames()` supplies the run names vendor tables usually report.

## Read and write results

```python
from pathlib import Path

from apb2.api import read_parsed_levels, sidecar_path, write_parsed_levels

parsed = read_parsed_levels(Path("results.parquet"))
write_parsed_levels(parsed, Path("results.duckdb"))
print(sidecar_path(Path("results.duckdb")))  # results.duckdb.apb.json
```

The suffix selects the format: `.h5ad`, `.h5mu`, `.parquet` or `.duckdb`. `write_parsed_levels()` also publishes an adjacent compact `<artifact>.apb.json` representation, which `sidecar_path(artifact)` names. `write_container_representation(container, artifact)` publishes the same sidecar for an AnnData or MuData APB2 did not write, such as an apb-export file. Converting between formats is `write_parsed_levels(read_parsed_levels(source), target)`; the CLI command `apb2 reformat` does the same.

## Result model

```python
from apb2.api import FinalLayerTable, JsonValue, LevelHierarchy, ParsedLevel, ParsedLevels
```

`ParsedLevels` contains an ordered level mapping whose levels share identical observation key columns, values and order, plus shared JSON-compatible provenance. `hierarchy` holds a self-contained `LevelHierarchy(name, identities)`, imported from `apb2.api`, from fine to coarse; identity names resolve against `level.var.roles` or name var columns directly. Each `ParsedLevel` contains:

- `obs`: the observation axis, with `frame` and `key_columns`
- `var`: the variable axis, with `frame`, `key_columns` and `roles`
- `primary_layer_name: str`
- `layers: dict[str, FinalLayerTable]`
- `obsm: dict[str, polars.DataFrame]`
- `varm: dict[str, polars.DataFrame]`
- `obsp: dict[str, polars.DataFrame]`
- `varp: dict[str, polars.DataFrame]`
- `uns: dict[str, JsonValue]`

### Semantic conversion roles

`FinalLayerTable.semantic_roles` holds semantic roles such as `abundance`; `ParsedLevel.abundance_layers()` returns every abundance layer in authored order, and `abundance_layers(names)` validates the named ones, the primary layer included. Every selection requires the abundance role. `level.var.roles` maps semantic roles such as `protein_assignment` to retained var columns; every role column must be `pl.String`. Readers and writers persist both role maps under `uns["apb"]["roles"]` and reject absent or non-String var role columns.

Consumers discover scientific meaning from these typed maps without knowing vendor-specific names; physical adapters serialize and validate them at the result boundary.

### Structural layer roles

Layer tables hold only observation values: row i belongs to var row i and column j to obs row j. They remain Polars frames until an h5ad or h5mu writer performs the matrix projection. `FinalLayerTable.role` defaults to
`MeasurementLayerRole()`. Measurement layers may be primary and participate in h5 matrix-occupancy
comparisons. `AuxiliaryLayerRole()` is for numeric diagnostics such as counts or component IDs: the
writer still validates, encodes, stores, and restores these layers, but excludes them from occupancy
comparisons and does not allow one to be the primary layer.

The structural-role-bearing layer field is:

```python
@dataclass(slots=True)
class FinalLayerTable:
    layer_name: str
    values: polars.DataFrame
    role: FinalLayerRole = field(default_factory=MeasurementLayerRole)
    semantic_roles: tuple[str, ...] = ()
    semantics: FinalLayerSemantics = field(default_factory=QuantitativeLayerSemantics)
```

Pairwise frames have exactly `row`, `column`, and `value` columns. Positions are zero-based local
coordinates into the corresponding final axis.

### Building and querying levels

Every name a consumer needs comes from `apb2.api`. Consumers build levels and attach annotations through methods, never through the axis, role or semantics classes:

```python
from apb2.api import ParsedLevel

level = ParsedLevel.build(
    obs, ("sample",), var, ("peptide",), {"protein_assignment": "protein"},
    primary_layer="Intensity",
    abundance={"Intensity": intensity},
    auxiliary={"Count": counts},
)
level = level.with_layers(abundance={"Summed": summed}, metadata={"history": [...]})
numbers = level.layers["Intensity"].quantitative_values()
```

- `ParsedLevel.build()`: each layer frame has one row per var row and one column per obs row, by position; APB2 names the columns. Abundance layers are measurements with the `abundance` role; auxiliary layers are diagnostics, integer when every column is
- `ParsedLevel.with_layers()`: a new level with layers and `varm` tables added and metadata sections set; existing names raise
- `FinalLayerTable.quantitative_values()`: the numeric value block; a categorical layer raises
- `FinalLayerTable.decoded_values()`: category codes replaced by their labels, numbers unchanged
- `ParsedLevels.with_annotation_table()` and `with_feature_relation()`: a new result with a keyed feature table, or its relation to one level's variable axis

## Errors

```python
from apb2.api import ConversionError, ResultIOError
```

Catch `ConversionError` for expected conversion failures from constructing a `ParseRuleCompiler`, `compile()` or `parse()`: unreadable parameter or rule files, inputs no rule accepts, and values the selected rule rejects. It is a `ValueError`; anything else that escapes is a defect.

Catch `ResultIOError` for expected result-format failures: an unsupported suffix, an invalid in-memory or persisted result, or an encoded layer set that violates an h5 required-name check or the measurement-layer occupancy contract. Its message names the cause; the subclasses that carry it are APB2-internal.

## Parser/result boundary

Parsing never writes: `parse()` returns `ParsedLevels`, and only `write_parsed_levels()` persists a result.

Parsing and result I/O therefore meet at `ParsedLevel`/`ParsedLevels`; neither computation nor the
result model imitates an AnnData container.

For compiler construction, rule-schema details, algorithms, and dependency boundaries, consult the
complete [converter architecture](architecture_converter.md). Nothing from that specification has
been moved into this user reference.

## Level hierarchies

Packaged rules name a hierarchy from `hierarchies.json`: `lfq` (fragment, ion, peptidoform, peptide, protein), `psm` (psm, ion, peptidoform, peptide, protein), or `ptm_site` (peptidoform, multisite, site). Level names are open and validated against the declared hierarchy. The collection persists its hierarchy name and complete ordered identities under `uns["apb"]["hierarchy"]`; custom results remain interpretable without the packaged rule configuration. Parser output aligns explicitly declared sample keys to their ordered union, retaining missing cells; readers and writers reject differing observation axes. Parquet uses format version 6 and DuckDB version 5.
