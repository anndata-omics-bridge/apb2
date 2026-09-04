# Supported software and formats

APB2 selects a packaged conversion rule only when both the software version and the vendor table
match that rule. The tables below describe the rules shipped with APB2. Version labels are concise
readings of the match patterns; follow a rule-document link for its exact version expression,
required source columns, transformations, and output schema.

## Packaged vendor-table rules

This table has one row per packaged rule document.

| Rule document | Supported versions | Vendor input | Shape | Parameter parser input |
| --- | --- | --- | --- | --- |
| [AlphaDIA 1.10](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_10/rules.json) | 1.10.x | `.tsv` | wide | AlphaDIA run log (text) |
| [AlphaDIA 1.12](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_12/rules.json) | 1.12.x | `.tsv` | long | AlphaDIA run log (text) |
| [AlphaDIA 2](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v2/rules.json) | 2.x | `.parquet` | long | AlphaDIA run log (text) |
| [AlphaPept](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphapept/rules.json) | 0.5.x | `.csv` | long | YAML parameter file |
| [DIA-NN 1](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v1/rules.json) | 1.x | `.tsv` | long | DIA-NN log or captured command/cfg text |
| [DIA-NN 1.7](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v1_7/rules.json) | 1.0.x–1.7.x | `.tsv` | long | DIA-NN log or captured command/cfg text |
| [DIA-NN 2](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v2/rules.json) | 2.x | `.parquet` | long | DIA-NN log or captured command/cfg text |
| [FragPipe](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/fragpipe/rules.json) | 22.x or 23.x | `.tsv` | wide | `fragpipe.workflow` |
| [MaxQuant](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant/rules.json) | 1.5.x, 1.6.x, or 2.x | `evidence.txt` | long | `mqpar.xml` |
| [MaxQuant peptides](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_peptides/rules.json) | 1.5.x, 1.6.x, or 2.x | `peptides.txt` | wide | `mqpar.xml` |
| [MaxQuant protein groups](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_proteingroups/rules.json) | 1.5.x, 1.6.x, or 2.x | `proteinGroups.txt` | wide | `mqpar.xml` |
| [MaxQuant modification-specific peptides](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_modificationspecificpeptides/rules.json) | 1.5.x, 1.6.x, or 2.x | `modificationSpecificPeptides.txt` | wide | `mqpar.xml` |
| [PEAKS](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/peaks/rules.json) | 13.x | `.csv` | wide | PEAKS settings text report |
| [Sage](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/sage/rules.json) | 0.x | `.tsv` | wide | Sage JSON parameter file |
| [Spectronaut](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/spectronaut/rules.json) | 19.x or 20.x | `.tsv` | long | Spectronaut settings text report |
| [Spectronaut 15](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/spectronaut/v15/rules.json) | 15.x | `.tsv` | long | Spectronaut settings text report |
| [WOMBAT](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/wombat/rules.json) | 0.9.11 | `.csv` | wide | WOMBAT YAML parameter file |

`--software` and the Python parameter-parser registry use lower-case software names; DIA-NN accepts
both `diann` and `dia-nn`.

## Quantification levels by rule

A check mark means that the linked rule document can convert that level. APB2 recognizes five level names. The `peptide` level comes only from MaxQuant `peptides.txt`, which is a wide table of per-experiment intensities rather than a per-run ion table.

| Rule document | Ion | Peptidoform | Peptide | Protein | Fragment |
| --- | :---: | :---: | :---: | :---: | :---: |
| [AlphaDIA 1.10](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_10/rules.json) | ✓ | — | — | — | — |
| [AlphaDIA 1.12](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v1_12/rules.json) | ✓ | — | — | — | — |
| [AlphaDIA 2](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphadia/v2/rules.json) | ✓ | — | — | — | — |
| [AlphaPept](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/alphapept/rules.json) | ✓ | — | — | — | — |
| [DIA-NN 1](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v1/rules.json) | ✓ | — | — | ✓ | ✓ |
| [DIA-NN 1.7](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v1_7/rules.json) | ✓ | — | — | ✓ | ✓ |
| [DIA-NN 2](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/diann/v2/rules.json) | ✓ | — | — | ✓ | — |
| [FragPipe](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/fragpipe/rules.json) | ✓ | — | — | — | — |
| [MaxQuant](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant/rules.json) | ✓ | — | — | — | — |
| [MaxQuant peptides](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_peptides/rules.json) | — | — | ✓ | — | — |
| [MaxQuant protein groups](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_proteingroups/rules.json) | — | — | — | ✓ | — |
| [MaxQuant modification-specific peptides](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/maxquant_modificationspecificpeptides/rules.json) | — | ✓ | — | — | — |
| [PEAKS](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/peaks/rules.json) | ✓ | — | — | — | — |
| [Sage](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/sage/rules.json) | ✓ | ✓ | — | — | — |
| [Spectronaut](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/spectronaut/rules.json) | ✓ | — | — | ✓ | ✓ |
| [Spectronaut 15](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/spectronaut/v15/rules.json) | ✓ | — | — | ✓ | ✓ |
| [WOMBAT](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_parse_rules/documents/wombat/rules.json) | ✓ | ✓ | — | — | — |

## Parameter parsers without packaged conversion rules

Parameter parsing and vendor-table conversion are independent capabilities. These parsers produce
typed search-parameter evidence, but APB2 does not currently ship a result-table rule for the same
software:

| Software | Parameter parser input | Conversion status |
| --- | --- | --- |
| MetaMorpheus | one TOML settings file plus one version-text file | no packaged vendor-table rule |
| MSAID | parameter CSV | no packaged vendor-table rule |

The complete parser registry is defined in
[`vendor_params/registry.py`](https://github.com/anndata-omics-bridge/apb2/blob/main/src/apb2/parserV2/vendor_params/registry.py).

## Planned prolfquapp migrations

Another motivation for APB2 is to replace hand-written, application-local readers with reusable,
rules-driven conversion. Today, the
[`prolfquapp::preprocess_software()` dispatcher](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_software.R#L137)
and its
[software-function registry](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_software.R#L9)
preprocess DIA-NN, MaxQuant peptide, FragPipe TMT/PSM and DIA/MSstats, and Spectronaut BGS outputs.
The separate
[`prolfquappPTMreaders` registry](https://github.com/prolfqua/prolfquappPTMreaders/blob/main/R/prolfqua_preprocess_functions.R)
adds FragPipe single-site, multi-site, and combined-site STY inputs plus Spectronaut site reports.

These existing pipeline inputs define the next migration targets:

| Source | Software or pipeline | Output to ingest and companion files | Existing reader mode | Upstream preprocessor | Current APB2 coverage |
| --- | --- | --- | --- | --- | --- |
| prolfquapp | DIA-NN | `report.tsv`, `diann-output.tsv`, or `report.parquet`; FASTA | protein or peptide | [`preprocess_DIANN()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_DIANN.R#L210) | DIA-NN 1.x and 2.x tables are packaged; peptide-mode parity remains |
| prolfquapp | FragPipe TMT | `psm.tsv`; FASTA | protein or peptide | [`preprocess_FP_PSM()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_FP_PSM.R#L511) | a different FragPipe TSV rule exists; PSM/TMT parity is not yet claimed |
| prolfquapp | MaxQuant | `peptides.txt`; FASTA | protein or peptide | [`preprocess_MQ_peptide()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_MaxQuant.R#L369) | both `evidence.txt` and `peptides.txt` are packaged |
| prolfquapp | MSstats | `msstats*.csv` or `msstats*.tsv`; FASTA | protein or peptide | [`preprocess_MSstats()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_MSstats.R#L210) | no packaged rule |
| prolfquapp | FragPipe DIA via MSstats | `msstats*.csv` or `msstats*.tsv`; FASTA | protein or peptide | [`preprocess_MSstats_FPDIA()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_MSstats.R#L87) | no packaged rule |
| prolfquapp | Spectronaut BGS | `*BGS Factory Report (Normal).tsv` or `*_Report.tsv`; FASTA | protein or peptide | [`preprocess_BGS()`](https://github.com/prolfqua/prolfquapp/blob/master/R/preprocess_BGS_default.R#L115) | Spectronaut TSV is packaged; BGS and peptide-mode parity remain to be verified |
| prolfquappPTMreaders | FragPipe single-site | `abundance_single-site_None.tsv`; FASTA | single PTM site | [`preprocess_FP_multi_site()`](https://github.com/prolfqua/prolfquappPTMreaders/blob/main/R/preprocess_FP_multisite.R#L147) | no packaged site-level rule |
| prolfquappPTMreaders | FragPipe multi-site | `abundance_multi-site_None.tsv`; FASTA | multiple PTM sites | [`preprocess_FP_multi_site()`](https://github.com/prolfqua/prolfquappPTMreaders/blob/main/R/preprocess_FP_multisite.R#L147) | no packaged site-level rule |
| prolfquappPTMreaders | FragPipe combined STY | `combined_site_STY_*.tsv`; `.fp-manifest`; FASTA | PTM site | [`preprocess_FP_combined_STY()`](https://github.com/prolfqua/prolfquappPTMreaders/blob/main/R/preprocess_FP_combined_STY.R#L152) | no packaged site-level rule |
| prolfquappPTMreaders | Spectronaut PTM | `Report*.tsv`, typically `*Report_WithProteinRollup.tsv`; optional FASTA | single phosphosite | [`preprocess_BGS_site()`](https://github.com/prolfqua/prolfquappPTMreaders/blob/main/R/preprocess_BGS_site.R#L134) | no packaged site-level rule |

We plan to move the remaining input variants and PTM/site-level formats from those R readers into
APB2. The goal is one parser and rule-document API that can serve both prolfquapp and ProteoBench.
This table is a roadmap, not a claim of current support. The packaged-rule tables above remain the
authoritative inventory of what APB2 can convert today. Test-only, simulated, and metabolomics
readers from the prolfquapp registry are outside this proteomics migration list.

## Persisted APB2 results

After conversion, the same storage-neutral `ParsedLevels` value can be written in any supported
result format:

| Format | Path | Levels | Result behavior |
| --- | --- | --- | --- |
| AnnData | `.h5ad` | exactly one | configured numeric/factor matrix projection |
| MuData | `.h5mu` | one or more | configured matrix projection per modality |
| APB2 Parquet dataset | `.parquet` directory | one or more | exact Polars values and schemas |
| DuckDB | `.duckdb` file | one or more | exact Polars values and schemas |
