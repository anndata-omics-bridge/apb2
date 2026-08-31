# Sample-annotation architecture

## Public object model

```python
parser = AnnotationCompiler(...).compile(annotation_source)
annotation = parser.parse(parsed)
result = annotation.annotate()
```

The compiler owns source loading and convention selection. The source-bound parser owns source
interpretation, matching, and policy validation. A concrete `Annotation` is created only when it is
valid for the supplied dataset, and therefore stores that `ParsedLevels` and its completed
`AnnotationMatches`. Application cannot accidentally combine evidence from one dataset with
another.

Parsing, matching, policy validation, and application remain separate internal functions and
types. They are folded together only at the public construction boundary; matching is not hidden in
a dataclass initializer.

## Values and behavior

`AnnotationTable`, `AnnotationCoverage`, `LevelAnnotationMatch`, `AnnotationMatches`, and
`AnnotationResult` are frozen data values. They select no behavior. Runtime policies implement
keep, complete-coverage, or selection behavior, while concrete prolfquapp and ProteoBench
annotations own their convention-specific diagnostic semantics.

ProteoBench validates complete observation coverage during `parser.parse(parsed)`. prolfquapp
validity depends on the configured application: keep accepts partial coverage, complete rejects it,
and selection validates its Boolean fields before construction. Every policy rejects a level with
zero matches because such an annotation is not an annotation for that dataset.

## Dependency direction

```text
apb2/
├── annotation_facade.py       result I/O + orchestration
└── annotation/
    ├── compiler.py            composition and convention choice
    ├── prolfquapp.py           source-bound parser + bound annotation
    ├── proteobench.py          source-bound parser + bound annotation
    ├── source/                 CSV/TSV/TOML decoding
    ├── application/            retention and selection behavior
    ├── matching/               exact/fuzzy matching
    └── data/                   innermost values and errors
```

Modules directly in `annotation/` compose children. The child packages import only the innermost
data package. Annotation computation depends on the `ParsedLevels` value model but never on result
I/O, Pydantic rule documents, pandas, AnnData, or MuData. `annotation_facade.py` is the outer adapter
that reads and writes physical results.

The optional rule declaration `sample_annotation.matching` is a Pydantic storage schema. Parser V2
projects it into JSON-compatible level provenance. Annotation matching constructs its runtime
matcher from that persisted value and never imports the vendor-rule package.

## Storage behavior

Annotation adds metadata columns to each level's `obs` and records source, convention, coverage,
corrections, and bounded mismatch evidence in JSON-compatible provenance. Existing result writers
then persist the ordinary `ParsedLevels`; no backend-specific annotation implementation exists.
