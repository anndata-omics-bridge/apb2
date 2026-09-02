# APB2

APB2 (anndata proteomics bridge) converts
[quantitative proteomics vendor tables](supported_software.md#packaged-vendor-table-rules)
into AnnData or MuData and can write the results as h5ad, h5mu, Parquet, or DuckDB.

APB2 was discussed and started during the Copenhagen ProteoBench Hackathon, 13–17 April 2026, as
part of the work to improve the backend of the
[ProteoBench platform](https://proteobench.cubimed.rub.de/).

Existing vendor-specific readers for differential expression/abundance analysis provided a second motivation. They are maintained behind
[`prolfquapp::preprocess_software()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_software.R#L137)
and in
[`prolfquappPTMreaders`](https://github.com/prolfqua/prolfquappPTMreaders). We plan to move their
[remaining input variants and PTM/site-level formats](supported_software.md#planned-prolfquapp-migrations)
into APB2, so one rules-driven parser can serve both prolfquapp and ProteoBench.

## Choose an interface

| Interface | Start here | Best suited to |
| --- | --- | --- |
| Command line | [Command-line interface](#command-line-interface) or [complete CLI reference](cli.md) | shell use, scripts, and workflow engines |
| Python | [Python API](#python-api) or [complete API reference](api.md) | libraries, notebooks, and custom pipelines |

Both interfaces expose three separate workflows:

```text
vendor table + parameter evidence
    -> convert
    -> .h5ad or .h5mu

APB2 result
    -> reformat
    -> another result format

APB2 result + sample annotation
    -> annotate
    -> annotated APB2 result
```

## Command-line interface

### Convert

Convert one quantification level to AnnData:

```bash
apb2 convert report.tsv ion --params search-parameters.txt --output results/ion
```

Omit the level to write every compatible level into MuData:

```bash
apb2 convert report.tsv --params search-parameters.txt --output results/all-levels
```

The [vendor-conversion guide](conversion.md) covers rule selection, explicit rule documents,
strictness, and output naming.

### Reformat

Change the storage format without parsing the vendor table again:

```bash
apb2 reformat results/all-levels.h5mu results/all-levels.parquet
```

The [result I/O guide](result_io.md) documents the format contracts and fidelity guarantees.

### Annotate

Attach a generic prolfquapp-style sample table to a converted result:

```bash
apb2 annotate results/all-levels.h5mu samples.tsv results/annotated.h5mu
```

The [sample-annotation guide](sample_annotation.md) covers matching, strictness, filtering, and
diagnostics. ProteoBench module annotation and scoring live in the separate `apb-proteobench`
package.

The [complete CLI reference](cli.md) lists every argument, option, and exit status.

## Python API

File-to-file facades mirror complete CLI operations. Compiler/parser APIs expose the
storage-neutral values between parsing, transformation, and persistence.

### Convert with the facade

```python
from pathlib import Path

from apb2.parserV2.conversion_facade import (
    convert_all_from_packaged_rules,
    convert_from_packaged_rules,
)

convert_from_packaged_rules(
    data=Path("report.tsv"),
    level="ion",
    output=Path("results/ion.h5ad"),
    parameters_path=Path("search-parameters.txt"),
    software=None,
    parameters_software=None,
    checks="standard",
)

convert_all_from_packaged_rules(
    data=Path("report.tsv"),
    output=Path("results/all-levels.h5mu"),
    parameters_path=Path("search-parameters.txt"),
    software=None,
    parameters_software=None,
    checks="standard",
)
```

### Compile and parse

Use the compiler/parser boundary to keep the parsed result in memory:

```python
from pathlib import Path

from apb2.parserV2.compile import AnnDataOutput, ParseRuleCompiler
from apb2.parserV2.detect_document import detect_rule_document, search_parameter_evidence
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_params.registry import parse_params

source = SingleFile(path=Path("report.tsv"))
parameters = parse_params(Path("search-parameters.txt"), software="spectronaut")
document = detect_rule_document(parameters, source).document
parser = ParseRuleCompiler(
    facade=ParseRuleFacade(document, "ion", search_parameter_evidence(parameters)),
    output=AnnDataOutput(checks="standard"),
).compile(source)

parsed = parser.parse()
parser.convert(parsed, Path("results/ion.h5ad"))
```

### Read and write results

Choose the source and target formats explicitly:

```python
from pathlib import Path

from apb2.parserV2.parse_quant.io.formats import ResultFormat, reader_for, writer_for

parsed = reader_for(ResultFormat.PARQUET).read(Path("results.parquet"))
writer_for(ResultFormat.DUCKDB).write(parsed, Path("results.duckdb"))
```

### Annotate a parsed result

The facade performs the complete file-to-file operation:

```python
from pathlib import Path

from apb2.annotation_facade import annotate_result

result = annotate_result(
    Path("results/all-levels.h5mu"),
    Path("samples.tsv"),
    Path("results/annotated.h5mu"),
)
```

The compiler/parser API exposes the storage-neutral transformation:

```python
from pathlib import Path

from apb2.annotation.compiler import AnnotationCompiler
from apb2.result_facade import read_parsed_levels, write_parsed_levels

parsed = read_parsed_levels(Path("results/all-levels.h5mu"))
parser = AnnotationCompiler().compile(Path("samples.tsv"))
annotation = parser.parse(parsed)
annotated = annotation.annotate().parsed
write_parsed_levels(annotated, Path("results/annotated.h5mu"))
```

The [complete Python API reference](api.md) documents result values, helper functions, and errors.

## Supported result formats

| Format | Path | Levels | Result behavior |
| --- | --- | --- | --- |
| AnnData | `.h5ad` | exactly one | configured numeric/factor matrix projection |
| MuData | `.h5mu` | one or more | configured matrix projection per modality |
| APB2 Parquet dataset | `.parquet` directory | one or more | exact Polars values and schemas |
| DuckDB | `.duckdb` file | one or more | exact Polars values and schemas |

Conversion does not perform sample annotation, FASTA annotation, protein inference, or ProteoBench
scoring. Run those operations explicitly after conversion.

The repository [README](https://github.com/anndata-omics-bridge/apb2#readme) gives the compact
project overview. The [converter architecture](architecture_converter.md) and
[sample-annotation architecture](architecture_annotation.md) record the design and dependency
rules.
