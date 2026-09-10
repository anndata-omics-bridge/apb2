# How rules-driven conversion works

APB2 keeps vendor-table knowledge in declarative `rules.json` documents. A rule declares the physical table shape, observation and variable identity, sourced or computed axis columns, measurement layers, semantic roles, and the quantification levels the table can produce. Supporting another layout normally means adding a rule rather than another Python reader.

The current rule format is schema `0.4`. The generated [JSON Schema](../src/apb2/parserV2/vendor_parse_rules/documents/_schema/rule.schema.json) is the complete machine-readable contract.

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
  "schema_version": "0.4",
  "file_version": "1",
  "software_name": "MinimalLongExample",
  "software_version_pattern": "^1$",
  "input": {
    "shape": "long",
    "extensions": [".tsv"]
  },
  "base": {
    "axis": {
      "obs_keys": ["sample"],
      "var_keys": ["protein"]
    },
    "columns": {
      "obs": [
        {"name": "sample", "source": "sample"}
      ],
      "var": [
        {
          "name": "protein",
          "source": "protein",
          "roles": ["protein_assignment", "fasta_accessions"]
        }
      ]
    },
    "measurements": {
      "primary_layer": "Intensity",
      "layers": [
        {
          "name": "Intensity",
          "source": "Intensity",
          "roles": ["abundance"]
        }
      ]
    }
  },
  "levels": {
    "protein": {}
  }
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
  "schema_version": "0.4",
  "file_version": "1",
  "software_name": "MinimalWideExample",
  "software_version_pattern": "^1$",
  "input": {
    "shape": "wide",
    "extensions": [".tsv"]
  },
  "base": {
    "axis": {
      "obs_keys": ["sample"],
      "var_keys": ["protein"]
    },
    "columns": {
      "var": [
        {
          "name": "protein",
          "source": "protein",
          "roles": ["protein_assignment", "fasta_accessions"]
        }
      ]
    },
    "measurements": {
      "primary_layer": "Intensity",
      "layers": [
        {
          "name": "Intensity",
          "source": "^[Ii]ntensity_(?P<sample>.+)$",
          "roles": ["abundance"]
        }
      ]
    }
  },
  "levels": {
    "protein": {}
  }
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

Rule documents may place shared declarations under `base` and level-specific declarations under `levels.<level>`. Composition produces one effective rule and validates all references at that boundary. Search-parameter overrides may replace `measurements.primary_layer` without changing the authored layer inventory.

Run `make check` in the APB2 repository after changing rules. A rule migration also requires the conversion corpus described in the workspace instructions because schema validation alone cannot prove real vendor files still bind and convert.

See the [Python API](api.md) for programmatic conversion and [Read and write parsed results](result_io.md) for persisted formats and the compact JSON representation.
