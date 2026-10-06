# Add a vendor format

A single-table format needs no Python. Debug the rule with `--rule-config` first ([How rules-driven conversion works](rule-based.md)); this page packages it.

## Checklist

1. Place `rules.json` under `src/apb2/parserV2/vendor_parse_rules/documents/`
2. Make its header signature unique within the vendor
3. Add it to `catalog.json`
4. Update the test inventory; commit a sample
5. Parameter parser: only for a new grammar
6. Add rows to [Supported software and formats](supported_software.md)
7. Add apb-catalog variants
8. Run `make check` and the routine corpus

Worked example: synthetic ToyQuant, two versions. Files and commands ran as shown; log lines shortened.

## 1. Place the document

```text
src/apb2/parserV2/vendor_parse_rules/documents/
├── sage/rules.json        # one document for every version
└── toyquant/
    ├── v1/rules.json      # one document per version
    └── v2/rules.json
```

- Discovered: `documents/<folder>/rules.json` and `documents/<folder>/v<suffix>/rules.json`; any other path is silently ignored
- Folder name: free; `--software` matches `software_name` lowercased, alphanumerics only (`DIA-NN` → `diann`)
- Required: `hierarchy`, `schema_version`, `file_version`, `software_name`, `software_version_pattern`; per table `input.shape`, `input.extensions`, `base`, `levels`
- Optional: `parameter_file: "none"` when no parameter file exists (`pb_custom`); `selection: "explicit"` to skip detection

```json title="toyquant/v1/rules.json"
{
  "hierarchy": "lfq",
  "schema_version": "0.8",
  "file_version": "1",
  "software_name": "ToyQuant",
  "software_version_pattern": "^1\\.",
  "tables": [
    {
      "input": {
        "shape": "long",
        "extensions": [".tsv"],
        "software_only": {"forbidden_columns": ["Gene Name"]}
      },
      "base": {
        "axis": {"obs_keys": ["Run"], "var_keys": ["Protein"]},
        "columns": {
          "obs": [{"name": "Run", "source": "Run Name"}],
          "var": [
            {"name": "Protein", "source": "Protein Accession", "roles": ["protein_assignment", "fasta_accessions"]}
          ]
        },
        "measurements": {
          "primary_layer": "Abundance",
          "layers": [
            {"name": "Abundance", "source": "Abundance", "roles": ["abundance"], "missing_values": ["<=0"]}
          ]
        }
      },
      "levels": {"protein": {}}
    }
  ]
}
```

ToyQuant 2.x adds a `Gene Name` column:

```diff title="toyquant/v1/rules.json → toyquant/v2/rules.json"
-  "software_version_pattern": "^1\\.",
+  "software_version_pattern": "^2\\.",
 ...
-        "extensions": [".tsv"],
-        "software_only": {"forbidden_columns": ["Gene Name"]}
+        "extensions": [".tsv"]
 ...
-            {"name": "Protein", "source": "Protein Accession", "roles": ["protein_assignment", "fasta_accessions"]}
+            {"name": "Protein", "source": "Protein Accession", "roles": ["protein_assignment", "fasta_accessions"]},
+            {"name": "Gene", "source": "Gene Name"}
```

## 2. Make recognition unambiguous

```tsv title="toy_v1.tsv"
Protein Accession	Run Name	Abundance
P1	a	10
P1	b	11
P2	a	20
P2	b	0
```

```tsv title="toy_v2.tsv"
Protein Accession	Gene Name	Run Name	Abundance
P1	G1	a	10
P1	G1	b	11
P2	G2	a	20
P2	G2	b	0
```

```text
$ apb2 convert toy_v1.tsv protein --software toyquant --output toy_v1
detected level=protein rule=toyquant/v1/rules.json
vendor=toyquant software_version=missing
level=protein shape=(2, 2) layers=['Abundance']
wrote toy_v1.h5ad
$ apb2 convert toy_v2.tsv protein --software toyquant --output toy_v2
detected level=protein rule=toyquant/v2/rules.json
```

A document accepts a file whose header holds every required `source` and required layer source. Each level must match exactly one document:

| Call | Documents tried | Chosen by |
| --- | --- | --- |
| `--params P` | the one vendor whose parameter-bearing documents accept the header | parsed version against `software_version_pattern` (`re.search`), then header |
| `--params P --software S` | vendor `S`, plus the parameter file's `quantification_software` | version, then header |
| `--software S` | vendor `S`, every version | header, then `input.software_only` |
| `--rule-config R` | `R` only | — |

Without v1's `software_only`, a 2.x file also satisfies v1:

```text
$ apb2 convert toy_v2.tsv protein --software toyquant
ERROR level 'protein' matches several packaged documents: ['…/toyquant/v1/rules.json', '…/toyquant/v2/rules.json']
```

- `software_only`: `--software` route only; see [software-only column evidence](rule-based.md#software-only-column-evidence)
- Version patterns: no overlap where headers overlap
- Generic signatures (one `Proteins` column, catch-all wide regex): other vendors' files turn ambiguous under `--params` alone

## 3. Register the category

```json title="src/apb2/parserV2/vendor_parse_rules/catalog.json"
    {"rule": "toyquant/v1/rules.json", "categories": ["DDA"]},
    {"rule": "toyquant/v2/rules.json", "categories": ["DDA"]},
```

`DDA`, `DIA`, or both; `[]` needs a `"reason"`. Missing line:

```text
ValueError: rule catalogue differs from packaged rules: missing=['toyquant/v1/rules.json'], unknown=[]
```

## 4. Update the test inventory

| File | ToyQuant change |
| --- | --- |
| `tests/parserV2/rule_inventory.py` | `EXPECTED_DOCUMENT_COUNT` +2, `EXPECTED_LEVEL_COUNT` +2 |
| `tests/parserV2/test_rule_package.py` | `test_both_rule_shapes_are_represented_by_the_packaged_generation`: `LongRule` count +2, `"error"` mode count +2; a non-primary `abundance` layer also goes into `_NON_PRIMARY_ABUNDANCE` |
| `tests/parserV2/test_schema_input.py` | `EXPECTED_EXTENSIONS`: `"toyquant/v1": [".tsv"]`, `"toyquant/v2": [".tsv"]` |
| `tests/parserV2/data/toyquant/v1/` and `v2/` | `sample.tsv` (the tables above), `header.txt` (one column name per line), `expected.json` |

```json title="tests/parserV2/data/toyquant/v1/expected.json"
{
  "sample": "sample.tsv",
  "params": null,
  "levels": {
    "protein": {"observations": 2, "variables": 2, "layers": ["Abundance"]}
  }
}
```

Real export: add it to the Fixture Manager store, run `.venv/bin/python scripts/make_test_samples.py ../apb_studio/test_data_download`, commit only the new folder.

## 5. Parameter parser: only for a new grammar

Skip when [registry.py](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_params/registry.py) already parses the vendor's parameter file.

```python title="src/apb2/parserV2/vendor_params/parsers/toyquant.py"
"""ToyQuant parameter-file parser: one ``key = value`` pair per line."""

from __future__ import annotations

from pathlib import Path
from typing import IO

from apb2.parserV2.vendor_params.parsers.shared.common import read_text
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters


def extract_params(source: Path | IO[bytes] | IO[str]) -> Parameters:
    """Parse a ToyQuant parameter file."""
    lines = read_text(source).splitlines()
    values = dict(line.split(" = ", 1) for line in lines if " = " in line)
    return Parameters(software_name="ToyQuant", software_version=values["version"])
```

```python title="src/apb2/parserV2/vendor_params/registry.py"
from apb2.parserV2.vendor_params.parsers.toyquant import extract_params as _toyquant_extract

_toyquant_parse = _single_source("ToyQuant", _toyquant_extract)

_REGISTRY: dict[str, ParseFn] = {
    # ...
    "toyquant": _toyquant_parse,
}
```

```text
$ printf 'version = 2.1\n' > toy.params
$ apb2 convert toy_v2.tsv protein --params toy.params
detected level=protein rule=toyquant/v2/rules.json
vendor=toyquant software_version=2.1
```

- `.importlinter`: add `toyquant` to the exhaustive parser layer
- `software_name`: the rule's slug, or version selection is skipped
- Parameter software ≠ result software: FragPipe parameters set `quantification_software="DIA-NN"` and its version, which select a DIA-NN result rule
- Tests: `tests/parserV2/vendor_params/`

## 6. Document support

- [supported_software.md](supported_software.md): one row per document in both tables
- [rule-coverage.md](rule-coverage.md): hand-written; only for a new schema capability

## 7. Add apb-catalog variants

Every set under [apb-catalog's `data/sources/`](https://github.com/anndata-omics-bridge/apb-catalog/tree/main/src/apb_catalog/data/sources) needs the vendor, even with nothing relevant:

```json title="src/apb_catalog/data/sources/<set>/toyquant.json"
{
  "catalogue_id": "toyquant",
  "catalogue_version": "0.1-draft",
  "software_name": "ToyQuant",
  "variants": [
    {"rule": "toyquant/v1/rules.json", "level": "protein", "software_version_pattern": "^1\\."},
    {"rule": "toyquant/v2/rules.json", "level": "protein", "software_version_pattern": "^2\\."}
  ],
  "entries": []
}
```

- Variant: `software_name` + `software_version_pattern` + `level`, each equal to the rule's
- `entries`: only fields the set's consumer reads ([Maintaining](https://anndata-omics-bridge.github.io/apb-catalog/maintaining/))
- `docs/catalogues.md`: a `none relevant` row per set
- apb-catalog installs `../apb2`; a missing set fails its `make check`:

```text
FAILED tests/test_source.py::test_every_packaged_rule_level_is_reviewed[miape]
E   Extra items in the left set:
E   ('ToyQuant', '^2\\.', 'protein')
E   ('ToyQuant', '^1\\.', 'protein')
```

## 8. Verify

```bash
make check                                              # apb2, then apb-catalog
cd ../apb_studio
uv run corpus run routine --workflow convert            # report the datasets it logs
uv run corpus run routine --workflow convert --dry-run  # must schedule zero jobs
```

Routine corpus: existing vendors still convert. Committed sample: the new format converts.

## Boundaries

- Single table: declarative; [What a rule document covers](rule-coverage.md) stays in `rules.json`
- `prepare`: only for several physical files (MaxQuant, AlphaDIA 1.12) or rows that are no hierarchy level; needs Python in `joins/` and a new `prepare.how`, so plan first
- Python for duplicate repair, column normalization, or other single-table quirks: never
