# What a rule document covers

A rule document and the standard reader handle everything listed here. A preparation hook may add only what this list lacks: the work that brings a table's rows to one hierarchy level. Anything listed here stays in `rules.json`, even when the table is prepared. [How rules-driven conversion works](rule-based.md) explains the syntax.

## Reading

- Formats: `.tsv`, `.txt`, `.csv`, `.parquet`, and a named `.xlsx` sheet
- Declared detection of delimiter, decimal and thousands marks, and encoding
- Exact file name inside a vendor directory (`input.file_name`)
- Header signatures for `--software` without a parameter file (`input.software_only`)
- Version selection by `software_version_pattern`
- Levels that apply only under declared search-parameter values (`requires_search_parameters`)

## Table shapes

- Long: one row per observation–variable measurement
- Wide: observations captured from layer headers by a regex `sample` group; `sample_layer` when other columns share the header shape
- Packed fragments: delimiter-packed values with positional or column labels

## Axis columns

- Rename a vendor column; mark it `required` or optional
- Types: `string`, `integer`, `number`, `boolean`
- Computed: `coalesce`, `join_nonempty`
- Vendor markings: `apb_Decoy` and `apb_Contaminant`, declared once on every level, never missing
- Sequences: `stripped_sequence`, `proforma_sequence`, `proforma_ion`, `proforma_fragment`
- Sequence grammars: `token_regex`, `site_list`, `embedded_site_list`, `plain_sequence`
- Sequence characters: residue letters or declared tokens and markers; anything else fails
- Modification maps to UniMod, with unknown tokens kept, dropped or rejected
- Rows with a missing or empty-text final key component are dropped
- Two raw keys mapping to one final key raise an error

## Measurements

- Numeric layers (`number`, `integer`) and factor layers with a category map
- `missing_values`: exact numbers and one `<=` threshold
- `missing_tokens`: text written for a missing value, such as `-` or `NA`
- `value_pattern`: one number extracted from structured cells by regex
- Repeated cells: `error`, `keep_first`, `sum`, `max`, or `keep_best` (one row per cell by a ranking layer, with optional summed layers); another reduction is a new mode here, never hook work
- `sum` and `max` leave a cell null when nothing is present, and reduce each layer separately, so with several layers one feature's cells can come from different repeated rows
- Primary layer, swapped by `search_parameter_overrides`
- Semantic roles: `abundance`, `protein_assignment`, `fasta_accessions`

## Not covered: hook territory

- Joining several files and resolving foreign keys between them
- Dropping rows by a column value, such as FlashLFQ `Full Sequences Mapped` ≠ 1
- Splitting or rejecting multi-valued identities such as `SEQA|SEQB`
- Making rows that are not a hierarchy level, such as FlashLFQ peaks, parseable as one; combining their repeated cells stays a duplicate mode
- Giving rows another table's identity through a vendor foreign key, such as MaxQuant evidence's protein group, so levels share one protein identity

Prepared tables are read as text ([prepare_source.py](../src/apb2/parserV2/prepare_source.py)); when a rule uses `sum` or `max`, their text layer columns are read as numbers, as the direct reader does.
