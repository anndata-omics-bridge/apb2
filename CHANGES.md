# Changes

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
