# How rules-driven conversion works

APB2 keeps vendor-table knowledge in one declarative `rules.json` document per software/version. Each document contains a nonempty `tables` list; every table declares its physical input, observation and variable identity, sourced or computed axis columns, measurement layers, semantic roles, and supported quantification levels. Supporting another layout normally means adding a table declaration rather than another Python reader.

The current rule format is schema `0.7`. The generated [document schema](../src/apb2/parserV2/vendor_parse_rules/documents/_schema/document.schema.json) describes the authored shell and physical inputs. Table-local base/level composition is then validated against the [effective-rule schema](../src/apb2/parserV2/vendor_parse_rules/documents/_schema/rule.schema.json). Older documents must be migrated; there is no compatibility model.

## One software, multiple input tables

Software metadata belongs at the document root. Each `tables` entry contains `input`, `base`, `levels`, and optionally `prepare: {"how": "maxquant"}` or `prepare: {"how": "alphadia"}`. Preparation is table-local: a direct-input group and a prepared group can coexist. Each rule describes one physical or prepared table; there is no separate join schema. Shared declarations merge only within a table, and each level belongs to one group.

The [MaxQuant document](../src/apb2/parserV2/vendor_parse_rules/documents/maxquant/rules.json) has two groups: direct evidence at run resolution, and joined wide exports at experiment resolution.

| Input file | Shape | Level |
| --- | --- | --- |
| `evidence.txt` | long | ion |
| `modificationSpecificPeptides.txt` | wide | peptidoform |
| `peptides.txt` | wide | peptide |
| `proteinGroups.txt` | wide | protein |

The function in [joins/maxquant.py](../src/apb2/parserV2/joins/maxquant.py) accepts any nonempty subset of the three higher-level exports, unpivots their quantities and joins shared evidence-ID references plus experiment. Namespaced source columns preserve each table's measurements, with repeated cells handled by the rule's `keep_first` policy. All three higher-level rules inherit `Experiment` observations from their table's `base`. Evidence is never joined: its direct rule inherits `Raw_File` observations and retains `Experiment` and `Fraction` metadata. All 15 nonempty input combinations remain supported.

The function in [joins/alphadia.py](../src/apb2/parserV2/joins/alphadia.py) enriches AlphaDIA 1.12 matrix intensities with precursor metadata and returns long rows. Parent composition prepares once per requested group. Each level projects from that group's shared frame, excluding wholly absent identities before ordinary decomposition. Tool modules import neither schemas nor parser orchestration. After parsing, explicit bijective observation mappings permit alignment; incompatible resolutions are [written separately](conversion.md#output-naming).

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
  "schema_version": "0.7",
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
            }
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
  "schema_version": "0.7",
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
            }
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

Supported computations are `coalesce`, `join_nonempty`, `stripped_sequence`, `proforma_sequence`, `proforma_ion`, and `proforma_fragment`. Entry names must be unique within each axis group, final axis keys must name required materialized entries, and computed dependencies must be available in declaration order.

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
    {"name": "Intensity", "source": "Intensity", "roles": ["abundance"]},
    {"name": "LFQ_Intensity", "source": "LFQ intensity", "roles": ["abundance"]}
  ]
}
```

Conversion provenance projects var roles as `column_roles`, mapping each role to one logical name. It projects layer roles as `layer_roles`, mapping each role to the ordered retained layer names. Optional layers absent from the bound source are omitted from `layer_roles`.

Semantic rule roles are distinct from the result model's structural `MeasurementLayerRole` and `AuxiliaryLayerRole`. Structural roles control matrix-occupancy validation; semantic roles describe scientific meaning. See [Read and write parsed results](result_io.md#semantic-rule-roles).

## Authoring and verification

Each table places shared declarations under `base` and level-specific declarations under `levels.<level>`. Composition produces one effective rule using that table's input and validates all references at that boundary. Search-parameter overrides may replace `measurements.primary_layer` without changing the authored layer inventory.

Run `make check` in the APB2 repository after changing rules. A rule migration also requires the conversion corpus described in the workspace instructions because schema validation alone cannot prove real vendor files still bind and convert.

See the [Python API](api.md) for programmatic conversion and [Read and write parsed results](result_io.md) for persisted formats and the compact JSON representation.
