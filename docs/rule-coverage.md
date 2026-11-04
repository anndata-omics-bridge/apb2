# What a rule document covers

A rule document and the standard reader handle everything listed here. A preparation hook may add only what this list lacks: the work that brings a table's rows to one hierarchy level. Anything listed here stays in `rules.json`, even when the table is prepared. [How rules-driven conversion works](rule-based.md) explains the syntax.

## Reading

- Formats: `.tsv`, `.txt`, `.csv`, their gzipped forms (`.tsv.gz`, `.txt.gz`, `.csv.gz`), `.parquet`, and a named `.xlsx` sheet
- Declared detection of delimiter, decimal and thousands marks, and encoding
- Exact file name inside a vendor directory (`input.file_name`)
- Header signatures for `--software` without a parameter file (`input.without_parameter_file`)
- Version selection by `software_version_pattern`
- Levels that apply only under declared search-parameter values (`requires_search_parameters`)
- Levels a table's header rules out (`forbidden_columns`), such as an ion table at a peptidoform level

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
- A feature's repeated rows in a var column: first nonblank value (`keep_first`, disagreements reported), the smallest or largest number (`min`, `max`), or the row another column ranks best (`keep_best`), selected before the feature table and the layers are built, so both come from that row
- Columns and layers that hold one value or none are not created; the parse record lists them ([conversion](conversion.md#uninformative-columns-and-layers))

## Measurements

- Numeric layers (`number`, `integer`) and factor layers with a category map
- `missing_values`: exact numbers and one `<=` threshold
- `missing_tokens`: text written for a missing value, such as `-` or `NA`
- `value_pattern`: one number extracted from structured cells by regex
- Repeated cells: `error` (any second row stops), `same` (the rows are copies of one value, such as a protein value on each precursor row; an empty row gives way, differing values stop), `sum` (all rows added up), or `keep_best` (one row per cell by a ranking layer, with optional summed layers); none takes one layer from one row and another from a row holding a different value; another reduction is a new mode here, never hook work
- `sum` leaves a cell null when nothing is present; it adds up every layer of the level, so it suits only levels whose layers are all additive
- Primary layer, swapped by `search_parameter_overrides`
- Semantic roles: `abundance`, `protein_assignment`, `fasta_accessions`

## Not covered: hook territory

- Joining several tables, files or sheets of one workbook, and resolving foreign keys between them
- Dropping rows by a column value, such as FlashLFQ `Full Sequences Mapped` ≠ 1
- Splitting or rejecting multi-valued identities such as `SEQA|SEQB`
- Making rows that are not a hierarchy level, such as FlashLFQ peaks, parseable as one; combining their repeated cells stays a duplicate mode
- Giving rows another table's identity through a vendor foreign key, such as MaxQuant evidence's protein group, so levels share one protein identity

Prepared tables are read as text ([prepare_source.py](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/prepare_source.py)); when a rule adds layers up (`sum`, or a `keep_best` `sum` list), those text layer columns are read as numbers, as the direct reader does.
