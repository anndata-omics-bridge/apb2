# Changes

- 2026-08-30: Separated the packaged Unimod storage document from lookup behavior. The canonical
  table is now validated JSON; one runtime `UnimodRegistry` owns accession, normalized name/alias,
  and mass lookup through the shared `UNIMOD_REGISTRY` instance. Removed the cached loader and
  module-level lookup functions, migrated parameter parsers and their ProteoBench oracle, and added
  focused duplicate, ambiguity, unknown-result, and tolerance coverage. The former four-module
  modification-review TODO was superseded by the implemented Parser V2 consolidation in `775569b`
  and `0538af6`.

- 2026-08-30: Added explicit measurement and auxiliary result-layer roles across h5ad/h5mu,
  Parquet, and DuckDB without changing their format versions. Older metadata defaults to
  measurement; h5 occupancy compares measurement layers only, while every layer remains validated,
  encoded, and persisted. Published the storage-neutral observation-label, quantitative projection,
  and numeric-result helpers for downstream APB tools.

- 2026-08-28: Added the missing `LICENSE` file (MIT) and declared `license-files`, without which
  uv_build shipped wheels carrying no licence text.

- 2026-08-26: Moved the persisted-result backends, format registry, metadata, validation, and
  result-I/O errors into `parse_quant/io`, with the single directed sibling dependency `io -> data`.

- 2026-08-26: Added the user-facing MkDocs Material site for vendor conversion, result-format
  reading and writing, the CLI, and the Python API. The repository README links to the rendered
  GitHub Pages site and its source index; the complete converter architecture remains intact as
  the site's architecture reference. A strict documentation build is part of `make check`, and a
  dedicated GitHub Actions workflow publishes `public/` from `main`.

- 2026-08-25: Added one storage-neutral `ParsedLevels` result-I/O boundary and
  `apb2 reformat SOURCE TARGET`. Explicit `reader_for(format).read(path)` and
  `writer_for(format).write(parsed, path)` adapters cover h5ad, h5mu, APB2 Parquet directory
  datasets, and DuckDB; path-inferred convenience functions delegate to the same registry.
  Parquet/DuckDB preserve Polars schemas, nulls, ordering, aligned frames, sparse-coordinate
  frames, and both provenance scopes exactly. h5ad/h5mu use the stored resolved plan for the
  deliberate numeric/factor matrix projection and persist a versioned envelope beside
  `uns["apb"]["parse"]`. Parser remains a one-level strategy with its existing
  `parse()`/`convert(parsed, target)` contract. DuckDB and its dynamically loaded PyArrow
  interchange are runtime dependencies; MuData 0.4 supplies the stable non-pulling container
  semantics used by h5mu.

- 2026-08-25: Removed every pointer from this repo into the workspace's archived planning folder —
  two in `vendor_params/parsers/shared/model.py` (the module docstring and the `scan_window`
  comment), one in `tests/parserV2/vendor_params/proteobench_params.py`, and one in a 2026-08-20
  entry below. The facts they cited are now stated in place: the removed `mode="before"` coercers
  must not come back, and splitting `scan_window` into two fields is an open schema question.
  Documentation only.

- 2026-08-25: Omitting `LEVEL` from `apb2 convert` now compiles and parses every compatible level
  and writes them as one shared-observation `.h5mu`. Parsing remains single-level: `ParsedLevels`
  is an output-boundary collection and `MuDataWriter` loops over the exact configured
  `AnnDataWriter` for each modality. Explicit levels continue to write `.h5ad`; modality variable
  indexes receive the established per-level prefixes while authored keys remain ordinary `.var`
  columns, and shared provenance is written under `uns["apb"]["parse"]` at the MuData root.
- 2026-08-24: Numeric and integer axis columns now use the physical number notation resolved from
  the input contract and source evidence. The resolved level plan retains that notation, and the
  compiler injects it into configured axis coercers exactly as it already does for measurement
  encoders. Locale-formatted Spectronaut metadata such as `PG.Cscore = "23,451117"` therefore
  parses under the existing decimal/thousands candidates instead of failing a direct Polars cast.
- 2026-08-24: Parser V2's measured Spectronaut ion conversion is substantially faster without
  changing parsed values. Long decomposition now performs one join, occurrence count, multi-value
  pivot, and ordering pass for all 20 physical layers instead of repeating them per layer; duplicate
  presence and AnnData numeric encoding evaluate Polars expressions across each layer frame rather
  than collecting one Series operation at a time. The AnnData writer dictionary-encodes repeated
  `obs` and `var` strings before handing them to AnnData, avoiding AnnData's Python natural-sort
  pass while reproducing its cardinality decision, so unique identity columns remain strings and
  nulls remain missing. On the 719,230-row, 695 MiB Spectronaut fixture, median decomposition fell
  from 5.726 s to 0.351 s, whole parse from 7.029 s to 1.620 s, AnnData writing from 3.706 s to
  0.817 s, and peak RSS from 7,137 MiB to 6,416 MiB. Category dictionary order is now Polars'
  stable column-local order; decoded annotations are unchanged. Factor-valued measurement layers
  retain their explicit rule-configured encoding path.
- 2026-08-23: The approved converter architecture is now versioned at
  `docs/architecture_converter.md`. `documentation/benchmarks/diann_cli_conversion.py` measures
  complete `apb convert` versus `apb2 convert` subprocesses through `.h5ad` writing, records every
  run and its environment, and refuses to report performance unless the persisted quantities and
  axis metadata agree. On the first DIA-NN v1 ion case, APB2 was 2.10 times faster by median wall
  time and used 46.3% less median peak resident memory across five measured runs per converter;
  all 2,620,944 compared matrix cells and every axis metadata value agreed.
- 2026-08-22: `apb2 convert --output` refuses only a basename that already ends in `.h5ad`,
  the extension it appends, instead of any name `Path.suffix` finds a dot in. A dotted basename is
  legal — `ion.apb2` beside `ion` is how a caller comparing two converters names the pair — and
  rejecting it made every such conversion fail with an argument error.
- 2026-08-22: The AnnData writer's namespace key is `apb`, matching APB, so a converted object
  carries `uns['apb']['parse']`. The parse provenance also states `produced_by` and the
  `column_roles` the rule declares, which is what lets `apb annotate`, `apb fasta` and
  `apb proteobench` run on an apb2 object: they read the level and the role columns as data
  instead of validating a schema-0.3 rule document with APB's own model. Verified end to end on
  the cached MaxQuant ion fixture — all three steps succeed, and `apb summary` renders the object.
- 2026-08-22: `apb2 convert` now runs Parser V2 exclusively. The CLI is a thin Cyclopts/Loguru
  adapter over `parserV2/conversion_facade.py`; packaged-rule selection inspects only source
  metadata and headers before the selected parser performs the single full read. Vendor-parameter
  parsing retains its established name under `parserV2/vendor_params/`; its shared Pydantic model,
  parsing primitives, and UniMod lookup live below `vendor_params/parsers/shared/`. Complete
  parameter plus rule selection provenance is written under
  `uns["anndata_proteomics"]["parse"]`. The legacy parser,
  rule, parameter, output, serialization, and temporary `parser_v2.py` modules were deleted;
  production `src/apb2` now contains only the CLI, package markers, and `parserV2`. `export-schema`
  is no longer a user command; its developer artifact generator remains in the V2 rule package.

- 2026-08-22: Parser V2's small shared modules now sit with the concepts that own them rather than
  in a generic helper package: the layer-table column convention is in `data/layer_columns.py`,
  resolved-plan JSON is in `parameters/plan_json.py`, and the concrete axis algorithms are named
  explicitly by `axis_columns.py`. Import contracts and architecture tests enforce the new paths.

- 2026-08-22: A compiled parser now records the plan it was built from.
  `parameters/plan_json.py` serializes
  one `ResolvedLevelPlan` as JSON text — every field, sets given a stable order, unknown shapes
  refused rather than stringified — and the composition root stores it in the parse provenance
  under `plan_json`, beside `rule_json`. It reaches `uns['anndata_proteomics']['parse']['plan_json']`
  in a written `.h5ad` and the `uns` block of a Parquet dataset's `manifest.json`, so a result
  states which columns were projected at which dtype, which dialect and notation won, which
  optional columns and layers the source withheld, and what the encoders were told.

- 2026-08-22: Parser V2 now retains every modification token preserved after a failed map lookup,
  reports the distinct tokens in first-observed order through `ParsedLevel.uns["unknown_mod_tokens"]`,
  and persists the same diagnostic through both AnnData and Parquet output. AnnData stores all
  parser-owned metadata under `uns["anndata_proteomics"]["parse"]`; inline-token and parallel
  site-list modification inputs share this behavior.

- 2026-08-21: Every working parse parameter in `parse_quant/parameters/working.py` now documents
  what its fields actually hold, with worked examples taken from packaged rules (MaxQuant,
  PEAKS, FragPipe, DIA-NN, Spectronaut) and a four-line reproducer. The module also states why
  a layer carries both a presence and an AnnData encoding declaration, and that presence runs in
  every parse while only the AnnData writer builds an encoder. Docstrings only; no behaviour changed.

- 2026-08-21: Parser V2's schema-0.3 declarations are split by ownership under the inward-only
  `vendor_parse_rules/schema/` child package; its marker exports nothing, schema modules never
  import the parent document or loader, and consumers import the exact declaring module. The
  former umbrella `schema.py` is gone. The physical-input schema is smaller: each rule authors only real
  extension hints, MaxQuant additionally names `evidence.txt`, and only Spectronaut enables
  delimiter and localized/grouped-number detection. Shared UTF-8, delimiter, quoting, and fixed
  dot-decimal defaults live once in `schema/base_formats.py`. The speculative input `kind`, source
  roles, `SingleTableSource`, and `FileRoles` were removed. Runtime-strategy compatibility checks
  moved to `ParseRuleFacade`, harmless defensive validators were deleted, and composed rules are
  validated before a valid parameter gate can classify them as inapplicable.

- 2026-08-21: Parser V2 is implemented (work packages W12–W13). `src/apb2/parser_v2.py` is the
  outer boundary: it translates the existing `Parameters` model into the two fields schema 0.3
  permits and nothing else. Every comparable packaged level now converts through the generic
  implementation and matches the unchanged one cell for cell — 15 (document, level) pairs on their
  real cached exports — with one accepted difference: MaxQuant aggregates, and a cell with rows but
  no present value is `0.0` there and missing here, which is the architecture's own decision.
  Finding that parity took four fixes on live data: measurements stay text unless the rule sums
  them (PEAKS writes `-`, WOMBAT `NA`), an unreadable token encodes as missing and is reported
  rather than refusing the file, `True`/`False` in a numeric layer reads as 1/0 (Spectronaut), and
  `NaN` counts as absence so one of them cannot poison an aggregated cell. Also new:
  `documentation/benchmarks/parser_v2_stages.py`, which measures the four cost claims the
  architecture makes — on a 219 MiB DIA-NN report, reading is 5% of a 0.89 s parse, decomposition
  62%, the Parquet write allocates no dense array for 5 layers and the AnnData write allocates
  exactly 5, and modification normalization tracks distinct sequences rather than rows (0.023 s to
  0.398 s for 1 000 to 100 000 distinct values at a fixed 200 000 rows).

- 2026-08-21: Parser V2 work packages W8–W11. `parser.py` holds the algorithm — read,
  decompose, prepare each axis, reindex each layer — with identity and validity decided in one
  place: distinct raw keys collapsing into one valid final key raise `CanonicalKeyCollisionError`
  under every duplicate policy, while an incomplete final key removes its axis row and the layer
  cells that pointed at it. `io/parquet_writer.py` persists a parsed level as a directory dataset
  with a manifest, preserving every Polars value and dtype; `io/anndata_writer.py` is the only module
  that encodes, allocates, or touches pandas. `compile.py` consumes every declarative tag once and
  injects tag-free behaviour: `compile_parsers` returns one parser per compatible level in
  canonical order. All 12 packaged documents now compile and 16 of the 19 levels parse their real
  cached exports end to end (the three that do not are the same three the legacy suite skips).
  Two defects found that way: a declared column carrying the name of its own physical source
  shadowed the raw key map, and reading a measurement column as a float is itself an encoding —
  real exports write `-`, `NA`, and `False` in a column a rule calls numeric, so measurements now
  stay text unless the rule sums them.

- 2026-08-21: Parser V2 work packages W3–W7. `parse_rule_facade.py` projects one effective rule
  into storage-neutral working parameters and resolves it against one observed header in a single
  atomic `ResolvedLevelPlan`; the axis key plan comes from one generic dependency walk over the
  authored keys, so all 19 packaged levels compile both axes with no vendor or level branch, and
  the specification's worked AlphaDIA example holds assertion for assertion. `delimited_input.py`
  resolves a dialect by asking which candidate exposes a usable header (ambiguity is reported, not
  guessed) and detects the decimal mark from the file's own values; `parquet_input.py` keeps the
  physical schema. `axis_columns.py`, `modifications.py`, `fragments.py`, `decomposition.py`, and
  `duplicates.py` hold the configured leaf algorithms: the ported modification domain reproduces
  the unchanged implementation exactly on 24 000 real vendor sequences across all 12 documents,
  long and wide input reduce to one raw contract with repeated cells intact, and per-layer presence
  lets keep-first skip AlphaDIA's `0` sentinel without encoding the value it keeps.

- 2026-08-21: Parser V2 work packages W0–W2. `polars` moved from the benchmark group to
  `[project.dependencies]` and `pytest-cov` joined the dev group, so `make test` runs the coverage
  invocation the Makefile always declared. `src/apb2/parserV2/` now holds the parsing-owned
  vocabulary — `parse_quant/data` (source, raw, parsed pipeline states), `parse_quant/parameters`
  (working, source, axis, measurements, resolved), and `parse_quant/contracts.py` (the nine
  Protocols `Parser` consumes plus the runtime axis plans) — and one complete schema-0.3 rule
  generation under `parserV2/vendor_parse_rules/`: `axis` is identity only, `primary_layer`,
  `duplicates`, and `layers` moved under `measurements`, `keep_all_as_raw_table` is gone, search
  conditions have a finite two-field vocabulary, and the physical input policy (extensions,
  delimiters, quoting, encoding, number notation) is declared data instead of reader constants.
  All 12 documents and all 19 effective levels migrated; each is compared declaration by
  declaration against the unchanged 0.2 oracle, and recognition is compared on real cached vendor
  headers. Five `.importlinter` contracts make the folder-nesting import law merge-blocking.

- 2026-08-21: `documentation/benchmarks/long_table_conversion.py` times APB's long-table
  conversion under polars, DuckDB, and pandas on both dtype backends, then serializes the result
  twice per variant — one DuckDB database, one Parquet folder — and reads every copy back before
  it reports. Totals are reported three ways, convert-only and convert-plus-each-target. On a
  695 MiB Spectronaut export, convert + Parquet: 0.22 s (polars) / 0.79 s (DuckDB) / 2.38 s
  (pandas); the conversion gap is entirely the TSV reader (19x), and serializing costs every
  engine about the same 0.07–0.36 s, which is +41% on a polars run and +7% on a pandas one. `polars` and `duckdb` are in the new `bench` dependency
  group, not runtime dependencies.

- 2026-08-20: `vendor_params/` no longer routes typed values through strings. Each parser hands
  `Parameters` exactly the types its fields declare — `MassTolerance`, `Probability`,
  `list[SearchedModification]` — and constructs it by keyword instead of `model_validate({...})`,
  so Pyright checks the boundary. `model.py` 524 -> 209 lines: the nine `mode="before"` coercers,
  `MassTolerance.parse`, `Probability.parse`, `_ENZYME_MAP`, the modification-token resolver, and
  `unparsed_parameters`/`UnparsedParameter` (zero producers) are gone, along with the schema's
  dependency on `unimod_registry`. `parsers/` 2741 -> 2538: `FragPipeParameterData`,
  `FragPipeVariantData`, `MetaMorpheusParameterData`, and `DiannImplicitDefaults` deleted;
  FragPipe's six do-nothing identity types and eight constructor wrappers deleted (596 -> 413);
  DIA-NN's five `_extract_cfg_*` and five `_command_*` staging helpers collapsed (600 -> 508);
  the settings-text reader shared by PEAKS and Spectronaut moved to `_common.py`. Two defects
  fixed at the root: `sage._enzyme` treated a null `restrict` as a restriction (the deleted enzyme
  map had been masking it), and PEAKS's FDR was recovered by a `>= 1 -> /100` magnitude guess after
  the parser discarded the `%`. `DiannParameterData` is kept on purpose — DIA-NN genuinely merges
  four evidence sources by precedence — but now in the schema's own types. Tests: all fifteen
  `apb/tests/test_params_*.py` ported with the `tests/params/` fixtures, their `from_series`
  oracle relocated to the test-only `tests/proteobench_params.py`, plus a snapshot test over all
  87 cached vendor parameter files; 70 -> 261 tests. Verified: across those 87 files the only
  changes are the removed `unparsed_parameters` and 171 modifications gaining the `mod_type` the
  parser already knew.

- 2026-08-17: TODO items 13 + 14 — `vendor_parse_rules/` has one entry point,
  `load_document(path) -> Document`, and `Document.rule(level, parameters)` returns the
  composed, gated, validated rule together with its header recognition. `Rules`,
  `packaged_rules`, `declared_rule`, `get_rule`, `parameter_gate.py`, `runtime.py` and
  `documents/select.py` are all gone; the pydantic shell and the base-level merge are private
  in `rules.py`, `_recognition.py` is private, and `documents/` is only the JSON tree.
  Detection moved out to `detect_document.py`, the consumer's declaration questions to
  `rule_reading.py`, the schema-artifact writer to `export_schema.py`, and the error
  vocabulary to top-level `errors.py` (with `RuleNotApplicable` as the base of the
  skip contract and `IncompatibleSourceError` its file-specific subclass). `modifications/`
  split: the Unimod table is the shared `unimod_registry/`, everything sequence-level lives in
  `parse_quant/modifications/` (`applier.py` — protocol, three appliers, factory —
  `normalize_sequence.py`, `proforma.py`, `modified_sequence.py`).
- 2026-08-17: TODO item 12 — items 6 and 8 applied to `modifications/`, the subpackage copied
  from apb and never reviewed. Names now say what they are: `apply_rules.py` →
  `normalize_sequence.py`, `ModificationRule`/`SiteListRule` →
  `TokenRegexSettings`/`SiteListSettings` (they are compiled normalizer settings, not
  rules.json rules), `apply_rule`/`apply_site_list` → `normalize_token_regex`/
  `normalize_site_list`. `modifications/model.py` → `modified_sequence.py`: `ModifiedSequence`
  and `ModificationOccurrence` are plain classes, since they are computed results nothing
  validates or dumps and one is built per distinct sequence. `SearchedModification` and
  `ModType` moved to `vendor_params/model.py`, which is the schema that parses them.

- 2026-08-17: TODO item 11 — two layers instead of three. The four Protocols and `Parser`
  moved into `parse_quant/parse_strategy.py` (the strategy layer declares its own contracts;
  `Parser.level` is a plain name), and the composition root merged with the selectors into
  one top-level `configure_parse.py` (399 lines, 253 code). Rule-declaration arithmetic —
  `carried_columns`, `var_extras`, `key_closure`, `projected_columns`, `string_typed_sources`
  — moved to `vendor_parse_rules/runtime.py`; `map_entries` to `apply_rules`; the
  conflicting-logical-types check became the rule validator `_check_one_type_per_source`.
  Each implementing module now pins its protocol with `_IMPLEMENTS: type[<Protocol>]`, and
  the two unions that only served as factory return types are gone.
- 2026-08-17: TODO item 9 closed — `parse_quant` imports no schema package. Every strategy
  constructor takes ordinary typed values (`bind_source` takes a header predicate and a label
  instead of the rule); the seven selector factories and the configuration→strategy
  translation they perform moved into the new top-level `selectors.py`, beside
  `parse_strategy.py`. `coercer_for` makes the axis-column `types` lookup a selector as well,
  pinned by a new test. One import-linter contract replaces the old direct-only one and holds
  without `allow_indirect_imports`.
- 2026-08-17: TODO item 10 closed — no inheritance left in the strategies. `modifications.py`
  loses `_NormalizeSequences`: `TokenRegexApplier` and `SiteListApplier` each implement the
  applier surface directly, hold a `SequenceColumns` collaborator (source check + the three
  output columns) and share memoization through one free function.

- 2026-08-15: Four review rounds over the redesign (4 parallel reviewers: DRY / dead code /
  boilerplate / adversarial-vs-baseline, then three adversarial verification rounds; converged).
  Correctness fixes, each with a regression test: the modification applier is never skipped by a
  vendor column named like its outputs (static rule fact, not frame sniffing); key-phase
  optional_select skips survive into the post-pivot materialization (reconstructed from raw
  SOURCE presence) and skipped names never reach the output projection; missing [modifications]
  and required packed-fragment columns fail construction as IncompatibleSourceError (contract);
  absent optional packed columns drop so their layer is skipped like the non-fragment path.
  New model validators: obs computes are value-combinators with declared inputs;
  ProformaSequence pins modifications.output_column; fragments are long-only; the packed label
  column cannot double as a select source. Consolidations: group_names/modification_outputs/
  declared_source_columns each have one home; one merge shape (_merge_blocks); one delimiter
  table; one atomic write (output.write_atomically); appliers split into TokenRegexApplier |
  SiteListApplier; packaged_documents() public and used by the sweeping tests; exploder trim
  deleted (the read plan already projects); defensive frame copies dropped (linear ownership).
  Recorded decision: typed-column validation covers rows that reach the output axes, not rows
  dropped by key dedup. Accepted low: ragged packed-list errors keep pandas' message.
  Final: 19 modules / ~3.1k LOC outside vendor packages (from 41 / 3.6k), 66 passed / 4 skipped
  incl. parity, pyright strict 0, contracts 2/2, CLI smoke green.

- 2026-08-15: Full runtime redesign per [TODO_apb2.md](../TODO/TODO_apb2.md) items 1–8, committed
  on top of the initial apb2 baseline commit. **parse_strategy.py is the core**: `Parser`
  (protocols + run loop) and `make_parse_strategy/-ies` in one file. **Conversion runs first**:
  read → fragments.explode → `columns.prepare_keys` (axis-key closure only; modifications
  memoized on unique source values) → `conversion.parse` (pivot; raw axis frames) →
  `columns.finish` (every remaining declared column materialized on the deduplicated obs/var
  frames — nrObs/nrVars rows instead of nrObs×nrVars). **One module per rules.json block**:
  input.py, duplicates.py, layers.py, columns.py, fragments.py, conversion.py, output.py,
  sources.py. **No behaviour on pydantic models**: `vendor_parse_rules/runtime.py` carries
  recognition + gates; `Document.rule` → `compose_rule`. **Real constructors**: strategy classes
  take `(rule, …)` and derive their own fields; every `make_*` wrapper deleted; only selector
  factories remain. `ConversionPieces` deleted (one `ParsedData`); scipy dropped (layers are
  dense end to end). New `tests/test_selectors.py` pins selector literals to runtime classes.
  Modules 41 → 20, LOC 3629 → 3133 outside the vendor packages. Gates: ruff clean, pyright
  strict 0, deptry clean, contracts 2/2, 59 passed / 4 skipped incl. full parity, CLI smoke.

- 2026-08-14: Rules-side line-count reduction, three passes, 900 → 764 (−15%). Pass 1:
  `documents/load.py` deleted — the document shell is now a pydantic `Document` in `model.py`
  (`extra="forbid"` replaces the hand-rolled key/level checks; `load_document` is 5 lines; the
  merge moved beside it unchanged). Pass 2: `LEVELS`/`PEPTIDE_LEVELS` derived via `get_args`, the
  rule `TypeAdapter` built from the `Rule` alias, `ColumnRoles.declared()` iterates the model,
  long docstrings tightened. Pass 3: per-class model validators merged (one `_core_consistency`
  on the rule core, one `_column_consistency` per shape, one `_consistent_declarations` on
  `ColumnGroup`) — same checks, same messages, fewer frames. Behavior pinned by the full suite:
  37 passed / 4 skipped incl. parity, pyright strict 0, contracts 2/2, schema artifact
  regenerated, CLI smoke green.

- 2026-08-14: Rules side repackaged to mirror `vendor_params/`: `models.py` →
  `vendor_parse_rules/model.py` and `documents/` → `vendor_parse_rules/documents/` (packaged JSON
  tree included; `resources.files` and import-linter contract module names updated). All imports
  rewritten; no code changes. Gates: ruff clean, pyright strict 0, deptry clean, contracts 2/2
  kept, 37 passed / 4 skipped, CLI smoke green.

- 2026-08-14: `vendor_params` copied into apb2 (`registry.py` + 12 vendor parsers + `_common.py`,
  imports rewritten to `apb2.*`; modification/unimod lookups now hit apb2's own
  `modifications/`). The parameters keep **their own model module**, `vendor_params/model.py`
  (rules.json and search parameters are separate schemas in separate files, per review) — copied
  minus the ProteoBench series round-trip (`to_series`/`from_series`, pandas-only, callers only in
  apb's tests). `models.py` is rules.json-only again: the fields-only `Parameters` copy was
  removed and the parameter gates import `Parameters` from `vendor_params/model.py`. The CLI/e2e
  `model_validate(model_dump())` bridge is gone — `parse_params` returns apb2's `Parameters`
  directly. apb2 src now imports **nothing** from `anndata_proteomics`; the dependency moved to
  the dev group (parity oracle + `test_data` lookup only). Deps: +pyyaml, +packaging. Gates: ruff
  clean, pyright strict 0, deptry clean, contracts 2/2 kept, 37 passed / 4 skipped incl. parity,
  CLI smoke on the cached DIA-NN fixture (auto-detected, 5 layers, shape (6, 72804)).

- 2026-08-14: `documents/load.py` stripped to trust the single typed boundary (187 → 168 lines,
  validation logic −70): the pre-validation helpers (`_dict_in`/`_list_in`/`_string`/`_as_dict`/
  `_group_in`) and both unknown-key raises are gone — the merge carries unknown keys and wrong
  types through to `validate_rule`, whose `extra="forbid"` and field errors are the only reporter;
  a fragment too malformed to merge surfaces as one `DocumentError` wrapping the `TypeError`.
  `apb2.serialization` copied in from apb's `core/serialization.py` (28 lines) — `steps.py`,
  `result.py`, `output/namespace.py` retargeted; src no longer imports `anndata_proteomics.core`.

- 2026-08-14: Tabular reading moved into apb2 as one file, `input/tabular.py`: public surface is
  `format_for(path)` + `UnknownFormat`/`DelimitedText`/`Parquet`, readers initialized with the
  path (`format_for(data).columns()`), delimiter resolved in the factory. apb's
  `readers.dispatch`/`readers.tabular` imports removed from src; the three `read_table*` wrappers,
  the full-read functions, and the `FixedDelimiter`/`DetectedDelimiter` classes were not carried
  over. pyarrow added as a direct dependency. Gates: ruff clean, pyright strict 0, deptry clean,
  contracts 2/2 kept, 37 passed / 4 skipped.

- 2026-08-14: Parameter resolution deferred to the composition root, per review. ``load.py``
  now *just parses*: surface = ``load_document(path)`` / ``iter_documents()`` /
  ``Document.rule(level)`` (194 lines). The parameter conditions ride on the composed rule as
  data (``requires_search_parameters``, ``search_parameter_overrides`` — override flattened to
  ``x_layer`` in the format, diann/v2 updated) with ``rule.available_for(parameters)`` and
  ``rule.resolved_for(parameters)`` on the model. ``make_parser``/``make_parsers`` renamed
  **``make_parse_strategy``/``make_parse_strategies``** and gained the ``parameters`` argument:
  the gate is checked and the X-layer override applied where rule, input, and evidence meet.
  Gates: ruff clean, pyright strict 0, contracts 2/2 kept, 37 passed / 4 skipped incl. parity
  (sage's gate now enforced at construction — the parity test hands over cached parameters).

- 2026-08-14: Schema cutover, driven in review. **All pydantic lives in one file, `models.py`**
  (the composed `LongRule | WideRule`, its block unions, and the `Parameters` fields moved in
  from apb — parsers stay in apb). Documents are **raw JSON dicts**: the base×level merge is a
  dict merge in `documents/load.py`; the single typed boundary is `validate_rule()`. The apb2
  document format applies the schema-review renames (`software_version_pattern`, document-level
  `input.shape`, `computed`/`inputs`, `"<sample>"` sentinel and `sample_name_cleanup` dropped) —
  tree regenerated, then the converter deleted. `documents/select.py` is one class:
  `DetectedSoftware(parameters, headers)` — the constructor IS the detection and fails when the
  evidence does not name exactly one packaged rules.json; `get_rule_path()` returns that file.
  Deleted: `rules/` (schema/compose/loader/registry/select/selection/validate), 
  `modifications/schema.py`, `convert_legacy.py`, the sample-namer machinery, the wide obs
  sentinel path, `RuleSelection`/`ParameterResolution`/version-evidence classes. Import-linter
  contracts live and kept: models+documents import nothing from pandas/numpy/anndata/apb; apb2
  never imports apb's parser packages. Gates: ruff clean, pyright strict 0, deptry clean,
  37 passed / 4 skipped incl. parity on every packaged rule with cached data.

- 2026-08-14: Project born. Scaffolded from `python_package_template` (coverage gates deliberately
  absent; pyright strict, Ruff, deptry, pre-commit kept; Python pinned 3.13). The working parser V2
  vertical moved in from apb (never committed there): `construction/parser/result/sources/dialect/
  errors/identity/columns/steps`, `input/`, `output/`, plus apb2-owned copies of the rules tree
  (`rules/`), converters (`convert/`), and modifications (`modifications/`, `sdrf.py` dropped as
  dead, `unimod_registry.toml` restored). All imports rewritten to `apb2.*`; apb remains a
  dependency strictly for `vendor_params`, `readers`, `core.serialization`, and `test_data`.
  New: `rules/selection.py` (parameter-driven selection twins), `output/namespace.py` (AnnData-only
  uns namespace, key kept `anndata_proteomics` while apb is the parity oracle), `cli.py` with
  `apb2 convert <data> LEVEL` (packaged + `--rule-config` routes, `.h5ad` only) and
  `apb2 export-schema`. Suite: 37 passed, 4 skipped — including parity vs legacy on every packaged
  rule with cached data, each side composing its rule from its own tree.
