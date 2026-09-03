# How rules-driven conversion works

APB2 separates knowledge about a proteomics table from the code that parses it. A declarative
`rules.json` document says:

- whether the table is long or wide;
- which columns identify observations and variables;
- which columns contain measurements; and
- which quantification level the table represents.

The shared parser reads the table according to that description. Supporting a new layout usually
means writing a new rule rather than a new Python reader.

The two minimal examples below contain the same data: two observations (`a` and `b`) and five
protein variables (`P1`–`P5`). Both produce an AnnData object with shape `2 × 5` and this
`Intensity` layer:

| sample | P1 | P2 | P3 | P4 | P5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | 10 | 20 | 30 | 40 | 50 |
| b | 11 | 21 | 31 | 41 | 51 |

## Long format

In a long table, every row holds one observation-variable measurement. The observation and
variable identifiers are ordinary columns, and the measurement rule names its value column
directly.

```tsv title="long.tsv"
protein	sample	Intensity
P1	a	10
P1	b	11
P2	a	20
P2	b	21
P3	a	30
P3	b	31
P4	a	40
P4	b	41
P5	a	50
P5	b	51
```

Here, `columns.obs` maps the input column `sample` to the AnnData observation key, while
`columns.var` maps `protein` to the variable key. The layer source is the exact column name
`Intensity`.

```json title="rules.json"
{
  "schema_version": "0.3",
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
      "obs": {
        "select": {
          "sample": "sample"
        }
      },
      "var": {
        "select": {
          "protein": "protein"
        }
      }
    },
    "measurements": {
      "primary_layer": "Intensity",
      "layers": [
        {
          "name": "Intensity",
          "source": "Intensity"
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

In a wide table, every row holds one variable and the observations are spread across measurement
columns. The measurement rule therefore uses a regular expression instead of one exact column
name. Its required `sample` capture group extracts the observation identifier from each matched
header.

```tsv title="wide.tsv"
protein	Intensity_a	intensity_b
P1	10	11
P2	20	21
P3	30	31
P4	40	41
P5	50	51
```

The expression `^[Ii]ntensity_(?P<sample>.+)$` matches both `Intensity_a` and `intensity_b`. It
captures `a` and `b` as the two observation keys. The explicit `[Ii]` makes only the first letter
case-insensitive; the remainder of the header is still matched exactly.

```json title="rules.json"
{
  "schema_version": "0.3",
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
      "var": {
        "select": {
          "protein": "protein"
        }
      }
    },
    "measurements": {
      "primary_layer": "Intensity",
      "layers": [
        {
          "name": "Intensity",
          "source": "^[Ii]ntensity_(?P<sample>.+)$"
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
apb2 convert wide.tsv protein --rule-config rules.json --output wide
```

The input layout changes, but the parser's output contract does not: observations form the rows,
protein variables form the columns, and the quantitative values occupy the `Intensity` layer.
