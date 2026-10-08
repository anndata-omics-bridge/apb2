# How rules-driven conversion works

APB2 keeps vendor-table knowledge in one declarative `rules.json` document per software/version. Each document contains a nonempty `tables` list; every table declares its physical input, observation and variable identity, sourced or computed axis columns, measurement layers, semantic roles, and supported quantification levels. Supporting another layout normally means adding a table declaration rather than another Python reader.

The current rule format is schema `0.8`. The generated [document schema](../src/apb2/parserV2/vendor_parse_rules/documents/_schema/document.schema.json) describes the authored shell and physical inputs. Table-local base/level composition is then validated against the [effective-rule schema](../src/apb2/parserV2/vendor_parse_rules/documents/_schema/rule.schema.json). Older documents must be migrated; there is no compatibility model.

## One software, multiple input tables

Software metadata belongs at the document root. Each `tables` entry contains `input`, `base`, `levels`, and optionally `prepare: {"how": "maxquant_wide"}`, `{"how": "maxquant_evidence"}`, `{"how": "alphadia"}` or `{"how": "metamorpheus"}`. Preparation is table-local: a direct-input group and a prepared group can coexist. Each rule describes one physical or prepared table; there is no separate join schema. Shared declarations merge only within a table, and each level belongs to one group.

MaxQuant's [document](../src/apb2/parserV2/vendor_parse_rules/documents/maxquant_wide/rules.json) has two groups: evidence at run resolution, and joined exports at experiment resolution.

| Input file | Shape | Level |
| --- | --- | --- |
| `evidence.txt` | long | ion |
| `modificationSpecificPeptides.txt` | wide | peptidoform |
| `peptides.txt` | wide | peptide |
| `proteinGroups.txt` | wide | protein |

`join_wide` in [joins/maxquant.py](../src/apb2/parserV2/joins/maxquant.py) accepts any nonempty subset of the three higher-level exports and resolves their foreign keys through shared evidence-ID references. It keeps every table wide: one row per related feature-ID tuple with each table's per-experiment columns side by side, so rows grow with relationships rather than features × experiments, and it reads only the columns its rules declare plus `id` and `Evidence IDs`; its wide layers capture `Experiment` from headers such as `peptide.Intensity <experiment>`. Namespaced source columns preserve each table's measurements, with repeated cells handled by the rule's `keep_first` policy. All three higher-level rules inherit `Experiment` observations from their table's `base`. Evidence never enters that join. Its own preparation, `join_evidence`, keeps every evidence row and adds the `Protein IDs` of the group, among those in `Protein group IDs`, that holds the row's leading razor protein, MaxQuant's own assignment of shared peptides; the ion rule names it `Protein_IDs`, like the protein level's key. The evidence rule inherits `Raw_File` observations and retains `Experiment` and `Fraction` metadata. All 15 nonempty input combinations remain supported.

The function in [joins/alphadia.py](../src/apb2/parserV2/joins/alphadia.py) enriches AlphaDIA 1.12 matrix intensities with precursor metadata and returns long rows. Parent composition prepares once per requested group. Each level projects from that group's shared frame, excluding wholly absent identities before ordinary decomposition. Tool modules import neither schemas nor parser orchestration. After parsing, explicit bijective observation mappings permit alignment; incompatible resolutions are [written separately](conversion.md#output-naming).

The function in [joins/metamorpheus.py](../src/apb2/parserV2/joins/metamorpheus.py) keeps FlashLFQ peaks mapped to exactly one peptidoform, dropping random-RT match-between-runs peaks, which carry their target's own identity, and casts `Peak intensity` to a number, so the rule can take the max of repeated peaks of one ion and run. Decoy peptides stay; the rule marks them `apb_Decoy`.

The function in [joins/wombat.py](../src/apb2/parserV2/joins/wombat.py) drops WOMBAT-P rows whose `modified_peptide` names several peptides (`SEQA|SEQB`, FlashLFQ's ambiguous peak) and logs how many; such a row is no single ion. Prepared tables are read with a comma for a `.csv` file and a tab otherwise.

## Software-only column evidence

When a caller supplies `--software` but no parameter file, the table's optional `input.software_only` declaration can distinguish rule variants beyond their normal required columns. `required_columns` and `forbidden_columns` are exact header signatures used only for this parameter-free route; parameter-backed version selection still uses the parsed software version. For example, DIA-NN 1.8/1.9 requires `PG.Normalised` and `Lib.PG.Q.Value`, while its 2.x rule requires `PG.TopN`.

For a rule with an acquisition-dependent primary layer, `acquisition_method_if_any` lists columns that establish DDA and `acquisition_method_otherwise` declares the fallback. DIA-NN 2.x uses `Ms1.Q.Value` or `Global.Ms1.Q.Value` as DDA evidence and otherwise selects its DIA default. This is authored evidence, not a generic claim that missing columns always mean DIA: only declare a fallback after validating its signature against the supported exports. Unresolved settings such as Sage's charge combination still raise an error rather than being guessed.

## Long format

In a long table, every row holds one observation-variable measurement.

```tsv title="long.tsv"
protein	sample	Intensity
P1	a	10
P1	b	11
P2	a	20
P2	b	21
```

```json title="rules.json"
{
  "hierarchy": "lfq",
  "schema_version": "0.8",
  "file_version": "1",
  "software_name": "MinimalLongExample",
  "software_version_pattern": "^1$",
  "tables": [
    {
      "input": {
        "shape": "long",
        "extensions": [
          ".tsv"
        ]
      },
      "base": {
        "axis": {
          "obs_keys": [
            "sample"
          ],
          "var_keys": [
            "protein"
          ]
        },
        "columns": {
          "obs": [
            {
              "name": "sample",
              "source": "sample"
            }
          ],
          "var": [
            {
              "name": "protein",
              "source": "protein",
              "roles": [
                "protein_assignment",
                "fasta_accessions"
              ]
            },
            {"name": "apb_Decoy", "how": "decoy"},
            {"name": "apb_Contaminant", "how": "contaminant"}
          ]
        },
        "measurements": {
          "primary_layer": "Intensity",
          "layers": [
            {
              "name": "Intensity",
              "source": "Intensity",
              "roles": [
                "abundance"
              ]
            }
          ]
        }
      },
      "levels": {
        "protein": {}
      }
    }
  ]
}
```

`hierarchy` names a level hierarchy from [hierarchies.json](../src/apb2/parserV2/vendor_parse_rules/schema/hierarchies.json), and every level the document declares must belong to it. Quantification tables use `lfq`, which runs from fragment to protein and identifies a `protein` feature by its `protein_assignment` column; `psm` and `ptm_site` cover PSM and modification-site tables.

Convert it with:

```bash
apb2 convert long.tsv protein --rule-config rules.json --output long
```

## Wide format

In a wide table, every row holds one variable and observations are spread across measurement columns. A layer source is therefore a regular expression with a required `sample` capture group.

```tsv title="wide.tsv"
protein	Intensity_a	intensity_b
P1	10	11
P2	20	21
```

```json title="rules.json"
{
  "hierarchy": "lfq",
  "schema_version": "0.8",
  "file_version": "1",
  "software_name": "MinimalWideExample",
  "software_version_pattern": "^1$",
  "tables": [
    {
      "input": {
        "shape": "wide",
        "extensions": [
          ".tsv"
        ]
      },
      "base": {
        "axis": {
          "obs_keys": [
            "sample"
          ],
          "var_keys": [
            "protein"
          ]
        },
        "columns": {
          "var": [
            {
              "name": "protein",
              "source": "protein",
              "roles": [
                "protein_assignment",
                "fasta_accessions"
              ]
            },
            {"name": "apb_Decoy", "how": "decoy"},
            {"name": "apb_Contaminant", "how": "contaminant"}
          ]
        },
        "measurements": {
          "primary_layer": "Intensity",
          "layers": [
            {
              "name": "Intensity",
              "source": "^[Ii]ntensity_(?P<sample>.+)$",
              "roles": [
                "abundance"
              ]
            }
          ]
        }
      },
      "levels": {
        "protein": {}
      }
    }
  ]
}
```

The expression matches both measurement headers and captures `a` and `b` as observation keys. Long and wide inputs produce the same observations-by-variables result contract.

By default the primary layer's captures are the sample names, and every other layer keeps only columns whose capture is one of them. When the primary layer's header shape is shared by non-sample columns, `measurements.sample_layer` names the layer whose captures are the sample names instead. PEAKS sets it to `Sample_Mz`: only runs have `<run> m/z` columns, so group averages such as `A Normalized Area` are not samples. The sample layer is required, and long rules reject the field because their observations come from `columns.obs`.

## Column entries

`columns.obs` and `columns.var` are ordered lists. Every physical column entry carries its own facts:

- `name`: logical output name
- `source`: physical vendor column
- `required`: source must exist; defaults to `true`
- `type`: `string`, `integer`, `number`, or `boolean`; defaults to `string`
- `roles`: semantic meanings owned by this entry

Computed entries replace `source` with `how` and `inputs`. They appear after the entries they consume:

```json
"obs": [
  {"name": "R_Label", "source": "R.Label", "required": false},
  {"name": "R_FileName", "source": "R.FileName", "required": false},
  {"name": "Run", "how": "coalesce", "inputs": ["R_Label", "R_FileName"]}
]
```

Supported computations are `coalesce`, `join_nonempty`, `stripped_sequence`, `proforma_sequence`, `proforma_ion`, `proforma_fragment`, `decoy`, and `contaminant`. Entry names must be unique within each axis group, final axis keys must name required materialized entries, and computed dependencies must be available in declaration order.

## Decoys and contaminants

Every rule declares, on every level's var axis, exactly once each, how the vendor marks decoys and the contaminants the software itself flags or adds. The results carry them as `apb_Decoy` and `apb_Contaminant`: booleans that are never missing. Rows are kept, only marked. Contaminants that the search FASTA defines, such as ProteoBench's `Cont_` entries, are protein_fasta's to mark, not the rule's.

```json
{"name": "apb_Decoy", "inputs": ["Reverse", "Decoy"], "how": "decoy", "equals": "+"}
{"name": "apb_Contaminant", "inputs": ["Proteins"], "how": "contaminant", "prefix": "CON__", "separator": ";"}
{"name": "apb_Decoy", "how": "decoy"}
```

- `equals`: a row is marked when an input's value equals it; a boolean input reads as `true` or `false`
- `prefix`: a row is marked when an input starts with it, or, with `separator`, when any member of the list does
- No inputs: the vendor writes no such rows; nothing is marked
- Several inputs: any match marks the row; inputs a source lacks are left out, so the column never disappears

## Independent sequence computations

Map vendor columns to logical names first. A sequence computation consumes exactly its ordered `inputs`; no normalizer substitutes a hidden input or generates another computation's column. For example, this fragment belongs inside a table's `base` or level, alongside its axis and measurement declarations:

```json
{
  "sequence_syntax": {
    "vendor_sequence": {
      "parser": "token_regex",
      "token_pattern": "^n\\[(?P<nterm>[^\\]]+)\\]|\\[(?P<residue>[^\\]]+)\\]",
      "token_position": "after_residue"
    }
  },
  "modification_maps": {
    "basic_modification_map": [
      {"token": "57.0215", "accession": "UNIMOD:4"}
    ]
  },
  "columns": {
    "var": [
      {"name": "Modified_Sequence", "source": "Modified Sequence"},
      {"name": "Charge", "source": "Charge", "type": "integer"},
      {
        "name": "ProForma_peptide", "how": "stripped_sequence",
        "inputs": ["Modified_Sequence"], "syntax": "vendor_sequence"
      },
      {
        "name": "ProForma_peptidoform", "how": "proforma_sequence",
        "inputs": ["Modified_Sequence"], "syntax": "vendor_sequence",
        "modification_map": "basic_modification_map",
        "case_sensitive": false, "unknown_policy": "preserve"
      },
      {
        "name": "ProForma_ion", "how": "proforma_ion",
        "inputs": ["ProForma_peptidoform", "Charge"]
      }
    ]
  }
}
```

`vendor_sequence` is an arbitrary local grammar name, not an input column. Syntax definitions contain no source or output names. Both registries merge base-to-level by name, replacing each same-named definition wholesale; references are validated after composition. There is no implicit default map.

| Syntax parser | Normalization inputs | Settings |
| --- | --- | --- |
| `token_regex` | sequence | `token_pattern`, `token_position`, `marker_pattern` |
| `site_list` | sequence, modifications, sites | `delimiter`, `site_base` |
| `embedded_site_list` | sequence, modifications | `delimiter`, `entry_pattern`, `site_base` |

A token pattern names the site of each token through its capturing groups: `nterm`, `cterm` or `residue`. Every capturing group carries one of these names, and apb2 places a token where its group says, never by where it sits in the text. A `residue` token modifies the residue it follows or, with `"token_position": "before_residue"`, the one it precedes. So DIA-NN's `…C(UniMod:4)` is a `residue` token on that cysteine, because its pattern has no `cterm` group; AlphaPept's `cC…` is a `residue` token before the cysteine, and its `^(?P<nterm>a)` names the N-terminal acetyl; PEAKS writes N-terminal acetyl after the first residue, or after that residue's carbamidomethyl, so its pattern lists those tokens and places, `(?:(?<=^[A-Z])|(?<=^C\(\+57\.02\)))\((?P<nterm>\+42\.01|\+42\.0106)\)`.

Every character that is not part of a declared token must be a residue letter (ProForma's: the twenty amino acids, `U`, `O`, `B`, `J`, `Z`, `X`); anything else fails the conversion with `UnrecognizedSequenceCharacterError` instead of being dropped. `_`, `-` and `.` are stripped only at the ends. A rule therefore declares every non-residue its vendor writes: FragPipe's terminal `n[…]` and `c[…]` and ProForma's `[…]-` and `-[…]` join the token pattern, anchored to the ends, and `marker_pattern` removes vendor text that is neither residue nor modification, such as AlphaPept's `_decoy` suffix, before tokenizing.

Stripping takes one logical sequence input and either token-regex syntax or `{"parser": "plain_sequence"}` for a bare sequence. It needs no modification map, Unimod lookup, or normalization operation. Normalization alone takes `modification_map`, `case_sensitive` (default `false`), and `unknown_policy` (`preserve`, `drop`, or `error`; default `preserve`). Unknown tokens are diagnostic metadata, not intermediate columns. Each operation memoizes its own distinct inputs.

When migrating from `0.7`, replace the global `modifications` block with named syntax/map definitions, add references to computations, and declare all site-list inputs through logical columns. Remove `source_column`, `sequence_column`, `modification_column`, `site_column`, and `output_column` from configuration definitions; column entries now own those dependencies and outputs. Schema `0.7` conversion rules are rejected, while previously written result artifacts remain readable without rewriting their provenance.

## Semantic roles

Roles state what a column or layer means; they do not choose storage or parser behavior. The packaged [role policy](../src/apb2/parserV2/vendor_parse_rules/schema/role_policy.json) is the single source of truth for the role vocabulary and its permitted owners:

| Owner | Allowed roles |
| --- | --- |
| `obs` | none |
| `var` | `fasta_accessions`, `protein_assignment` |
| `layer` | `abundance` |

A var role identifies one logical column. A layer role may occur on several layers because raw, normalized, MS1, MS2, LFQ, iBAQ, peak-area, and other abundance measurements can coexist.

`measurements.primary_layer` is separate: it names the one layer projected to AnnData `X` and makes that source required. It does not imply that sibling abundance layers are auxiliary or semantically unclassified.

Numeric layers default to logical `"type": "number"`. Declare `"type": "integer"` only when every non-missing measurement is a whole count; conversion rejects fractional and infinite values. Null and NaN remain valid missing cells, so AnnData may still store an integer layer in a `Float64` matrix. Factor layers continue to use `encoding_mode` and have no numeric type.

```json
"measurements": {
  "primary_layer": "Intensity",
  "layers": [
    {"name": "Intensity", "source": "Intensity", "roles": ["abundance"], "missing_values": ["<=0"]},
    {"name": "LFQ_Intensity", "source": "LFQ intensity", "roles": ["abundance"], "missing_values": ["<=0"]}
  ]
}
```

`missing_values` lists what a vendor writes for "not measured": exact numbers such as `0`, and at most one `<=` threshold such as `"<=0"`, which makes every number at or below it missing. Both apply before duplicate resolution and in the final layer. Every packaged `abundance` layer declares `"<=0"`, because a linear abundance is positive by definition and log-scale consumers cannot use zero or negative values; a package test enforces this. Leave it off layers where zero or negative values are meaningful, such as scores, mass errors, or retention-time deltas.

`missing_tokens` lists the text a vendor writes for "not measured" in a numeric column, such as PEAKS' `-` or WOMBAT's `NA`. A cell holding one, after trimming whitespace, is missing like a blank cell: it claims no duplicate cell and is not reported as an unreadable cell. Any other text that is not a number stays an unreadable cell, so an undeclared spelling still shows up in the conversion summary.

Conversion provenance projects var roles as `column_roles`, mapping each role to one logical name. It projects layer roles as `layer_roles`, mapping each role to the ordered retained layer names. Optional layers absent from the bound source are omitted from `layer_roles`.

Semantic rule roles are distinct from the result model's structural `MeasurementLayerRole` and `AuxiliaryLayerRole`. Structural roles control matrix-occupancy validation; semantic roles describe scientific meaning. See [Read and write parsed results](result_io.md#semantic-rule-roles).

## Authoring and verification

Each table places shared declarations under `base` and level-specific declarations under `levels.<level>`. Composition produces one effective rule using that table's input and validates all references at that boundary. Search-parameter overrides may replace `measurements.primary_layer` without changing the authored layer inventory.

Run `make check` in the APB2 repository after changing rules. A rule migration also requires the conversion corpus described in the workspace instructions because schema validation alone cannot prove real vendor files still bind and convert.

To ship a rule as a packaged, detected format, follow [Add a vendor format](adding_a_format.md).

See the [Python API](api.md) for programmatic conversion and [Read and write parsed results](result_io.md) for persisted formats and the compact JSON representation.
