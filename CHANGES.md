# Changes

## 2026-10-28

- A file the FragPipe parameter parser cannot read, such as a DIA-NN log or an empty file, is a `ParamsError` naming the missing workflow entry, so conversion reports a `ConversionError`; it raised a bare `KeyError` before
- A delimited table in a non-UTF-8 encoding, such as windows-1252, is transcoded to a temporary UTF-8 copy chunk by chunk and streamed like any UTF-8 table; the copy is removed after the read. Polars' eager reader decoded the whole file in memory before: Spectronaut LargeFC (11.1 GB, one windows-1252 dash in `EG.Label` on every row) peaked at a 44 GB footprint for one level. Reading needs temporary disk the size of the decompressed file
- MaxQuant peptides carry their protein group. With proteinGroups.txt, the peptide level's `protein_assignment` is the new `Protein_IDs`, the `Protein IDs` of the group the peptide joins, which holds its leading razor protein, so peptides name the protein level's keys as ions already do; with peptides.txt alone, `Protein_IDs` is `Leading_Razor_Protein`. In the entrapment fixture (MaxQuant 2.8.1.0), all 43,807 peptides join exactly one group, the one holding their razor protein
  - **Breaking:** `Leading_Razor_Protein` keeps its values but loses the role; prolfquapp's export of MaxQuant peptides now names the group in `protein_Id`

## 2026-10-27

- Every result reader parks each level it reads (`LayerParking`, now `data/parking.py`): the layer cells move to an unlinked, memory-mapped Arrow file and stay an ordinary sorted `pl.DataFrame`. Reading back the 1,236-run TPCPA DIA-NN result held 36 GB of process memory before, so two results could not be compared on a 48 GB machine. Reading needs temporary disk space the size of the cells

## 2026-10-26

- MaxQuant's peptidoform level writes `ProForma_peptidoform_unlocalized`, such as `[UNIMOD:35][UNIMOD:35]?PEPTMIDEMK` for `PEPTMIDEMK` with `2 Oxidation (M)`. `modificationSpecificPeptides.txt` names modifications with counts and candidate residues, never positions, so positional isomers share a row and no localized `ProForma_peptidoform` can be written for it
  - **Breaking:** the level is keyed by `ProForma_peptidoform_unlocalized`, not by `ProForma_peptide` plus `Modifications`, as Sage and WOMBAT key theirs by `ProForma_peptidoform`; `Modifications` stays a var column
  - **Unknown tokens:** under `preserve`, every unmatched token is written at its position, repeated by its count, not only the last one there; so `Deamidation (NQ)` and `2 Deamidation (NQ)` stay two strings. A localized tool with two unmatched tokens on one residue now writes both
  - **Rule schema:** `proforma_sequence` may be named `ProForma_peptidoform_unlocalized`, which requires `embedded_site_list` syntax; for it, a site of residues only goes to ProForma's unknown-position group. `entry_pattern` may capture a `count`, and an entry it matches without a `token`, such as `Unmodified`, adds no modification
  - **Runtime:** the three normalizers share one recorder of labels and unknown tokens; modification locations name their label position, so unknown tokens key on it and `render_proforma` takes them by position, not by index
- `apb2.api` exports `FIXED_COLUMNS`, the six column names APB2 fixes for every vendor, and `VAR_ROLES`, the roles a rule may give a var column, so apb-catalog can check its `apb` set against them
- `read_parsed_levels(source, level=None)` reads one level alone (approved `api.py` line): an h5mu reads that modality, Parquet that level's directory, DuckDB that level's tables; annotation tables are kept, feature relations only where every linked level was read; an unknown level raises `ResultIOError`. `level` is a level name, the same `str` alias as `QuantificationLevel`
- Readers no longer load a whole result: h5ad and h5mu are read element by element and each matrix in slabs of feature columns; Parquet and DuckDB layers in batches of rows; each block keeps only its stored cells, sorted by `var_pos`
  - **Breaking:** a number layer's NaN read from h5ad or h5mu is a missing cell, not a stored NaN, so `wide()` of a read-back layer shows null where it showed NaN. h5 writes every missing cell as NaN and cannot tell the two apart; storing every NaN made a read-back number layer dense
- Each level of a file reads its own columns; the shared read of the union of every level's columns is gone. It made the first level parse beside all the others' columns: on a 150-run DIA-NN 1.8 report the fragment level parsed beside about 3 GB of ion and protein columns. It was added for speed (Spectronaut LargeFC 275 → 99 s, before categorical keys and in-scan conversion); the extra reads on the 150-run report cost about 3 s
- The default conversion's levels no longer add up in memory: each parsed level's layer cells are parked in an uncompressed Arrow IPC file, memory-mapped back and unlinked at once (`parse.park` step). Parsed cells are sorted by `var_pos`, then `obs_pos`
- Writers and the JSON sidecar widen one block (about 1 GB) of one layer at a time: h5ad and h5mu fill their dense datasets in feature-column slabs, chunked one block wide; Parquet writes one row group per block; DuckDB appends one block at a time. Stored values, dtypes, encoding attributes and sidecar statistics are unchanged; chunk shapes and Parquet row groups differ
- Packed fragment levels (DIA-NN 1.7, 1.8) take less memory: a packed column of a plain numeric layer is read as numbers inside its lists before the explode (`converted_sources` in the separator's `plan_json`), falling back to text when a token needs it; the label travels as its position or a Categorical; the fragment axis and each scalar's row come from one integer key per (precursor, label) instead of a join. DIA-NN 1.8 TPCPA subset, peak footprint: 150 runs 18.0 → 9.3 GB, 50 runs 6.6 → 4.7 GB
  - **Payload:** a fragment's var payload (`Protein.Group`, `Decoy`) is resolved once per precursor over the rows that hold fragments, and a precursor's disagreement is reported for each of its fragments; it differs from before only where a precursor's runs pack different numbers of fragments and its rows disagree
  - **Repeated cells:** every layer checks them on one UInt64 cell key, a large layer one `var_pos` range at a time (fragment layer at 150 runs: 15.6 → 8.2 GB in that step)
- **Breaking:** the levels `ion` and `fragment` are renamed `peptidoform_ion` and `peptidoform_fragment`. A fragment is an ion too, and so is a metabolite ion; PSI-MS calls the precursor a peptidoform ion (`MS:1003051`). The change covers `hierarchies.json` (`lfq`, `psm`), every packaged rule's `levels` keys, `ParsedLevels` keys, MuData modalities, stored level names and the `apb2 convert` `LEVEL` argument. `ProForma_ion` and `ProForma_fragment` keep their names. There is no alias; results written before this change are reconverted
  - **h5mu var-name prefixes:** `pfi:` and `pff:` replace `ion:` and `frg:`
  - **MSAngel:** the preparation's ion-sheet role is `peptidoform_ion`, so its prepared columns read `peptidoform_ion.<column>`

## 2026-10-23

- **Breaking:** MSAngel converts a protein level, and the ion `protein_assignment` of MSAngel and ProlineStudio is `Same_Set_Proteins`, not `Proteins`. Proline's ion sheet lists a shared peptide once per protein set; each protein column now comes from the set with the highest `protein_set_score` (`keep_best`), not from whichever row came first, and these ions are no longer `var_disagreements` (2,435 in MSAngel 2a5ef619, 2,202 in ProlineStudio 6ed9da37).
  - **Why `Same_Set_Proteins`:** `subsets_accessions` lists only the subset proteins holding that peptide, so `Proteins` split one protein set into several groups (8,112 groups for 7,510 assigned sets in 2a5ef619). `samesets_accessions` identifies a set one-to-one; `Proteins`, the set's accessions holding the peptide, keeps `fasta_accessions`
  - **MSAngel protein level:** the new preparation `msangel` joins the workbook's "Protein sets" sheet onto the ion rows by `protein_set_id`; the level is keyed by `Same_Set_Proteins`, with Proline's protein abundances `Abundance`, `Raw_Abundance`, `PSM_Count` and `Peptides_Count`. A workbook without that sheet converts the ion level only. Sources are namespaced `ion.` and `protein.`, as MaxQuant's are
  - **Hooks read workbook sheets:** a preparation names the sheets it reads, and `prepare_source.py` reads only those, as text; such a table declares no `sheet_name`. The input record lists the workbook once, under the ion sheet's header
  - **ProlineStudio:** `protein_set_score` becomes required; `Quant_Is_Valid` comes from the assigned set's row too (false for 260 ions of 6ed9da37, which Proline quantifies under another set)
- Rule schema 0.9: a var column may declare `"duplicates": {"mode": "keep_best", "by": <column>, "best": "highest"|"lowest"}`. It takes the value of the feature's row whose `by` column ranks best, even an absent value, so the columns ranked by one score describe the same row; ties keep file order, and no disagreement is reported. `by` names a required numeric var column of the same level, neither a key nor the column itself.
  - **Runtime:** `RankedFeatureValue`; `FeatureValuePolicy.value` and `.disagrees` take the frame schema, since a ranking reads a second column
  - **Fix:** `min` and `max` rank a `NaN` as absent; a `NaN` row won `max` before
- Levels that read one file with the same evidence share one read of the union of their columns; each projects its own. Spectronaut LargeFC (fragment, ion, protein): 275 to 99 s, peak memory 33.6 to 25.5 GB; DIA-NN Charité: 69 to 65 s, peak memory 16.2 to 19.0 GB.
- **Breaking:** an annotation's `provenance.annotation.source` is the annotation file's input-file record (`role` `sample_annotation`, `file_name`, `size_bytes`, `sha256`, `compression`, `format`), not its absolute `path`; annotation `schema_version` 4. Producers record base names, so the sidecar no longer rewrites `*_path` fields.
- A preparation finds `.csv` tables in a folder, such as WOMBAT-P's peptide table, where it listed only `.tsv` and `.txt`. A prepared table may not declare delimiter, number or encoding detection: a preparation reads its files by extension and would ignore it.
- **Breaking:** one absence rule for every source: null, a float `NaN`, or text without a non-whitespace character is a missing cell, and a row with no value in any read column is dropped, whichever reader or preparation produced the table.
  - **Parquet:** `""` is missing; DIA-NN 2 ion `Protein_Group` and `Protein_Names` of precursors without a protein group are missing, not `""`
  - **Delimited:** a quoted `""` or whitespace-only cell is missing
  - **Excel:** an empty text cell is missing, not `""`, and an empty row is dropped
  - **Duplicates:** whitespace-only text no longer claims a cell
  - **Routine corpus (13 datasets):** only null counts change: AlphaDIA 2 `Mods` and `Mod_Sites` (95,514 unmodified ions), DIA-NN 2 `Protein_Group` and `Protein_Names` (1,771 ions), PEAKS `Accession` (50 ions)
- **Breaking:** a row without a sequence no longer becomes the ion `"/2"`. Sequence columns keep a missing sequence missing, and `ProForma_ion` raises on it as it does on a missing charge.
- A gzipped table is selected by its unzipped name: `report.tsv.gz` binds the table a rule declares as `report.tsv`, where it was treated as a renamed file. A gzipped Parquet or Excel file is refused at binding with "decompress … first".
- **Breaking:** every result records the files its conversion read, once each, in the root parse record's `provenance.input_files`, so a result names the bytes it came from.
  - **Record:** `role` (`table`, a preparation's binding role, or `parameter_file`), `file_name` (base name), `size_bytes`, `sha256` of the bytes on disk, `compression`, `format`; a table adds `nrows` (data rows before any filter), `ncols` and `columns` (its full header)
  - **Moved:** `search_parameters` keeps only `parameters`; its `file_name` and `sha256` are in the parameter file's `input_files` entry
  - **Removed:** `input_preparation.sources` (absolute paths) and `input_preparation.rows` (the joined frame's height)
  - **Root never empty:** a conversion without a parameter file now has a root `provenance` with `input_files`
  - **Internal:** `LevelSourceTable` moves from `parse_quant.data.source` to `parse_quant.parameters.source`; `PreparedTable`, in `prepare_source`, carries its tables' records
- **Breaking:** Spectronaut 21's library q-values are `var` columns, not layers: ion `EG_LibraryQvalue` and protein `PG_LibraryQvalue` never differ across runs (submission 790d6197: 142,425 ions and 11,588 protein groups in several runs). The fragment level of the 19–20 and 21 rules keeps `F_FrgMz` and `F_TheoreticalMz` only as `var` columns, where each was also a layer: neither differs across runs in a 2021 Spectronaut export (980,998 fragments in several runs), while `F.MeasuredMz` does. Each entry carries a `note`.
- `LayerContractError` is a `ConversionError`: `apb2 convert --strict` on an effectively empty layer, or an empty primary layer, logs the error and exits 1 instead of ending in a traceback.
- **Breaking:** DIA-NN's protein level has no `Protein_Ids` column; `Protein_Group` carries `fasta_accessions` beside `protein_assignment`, as at the fragment level and in the AlphaDIA 2 and MaxQuant protein rules. `Protein.Ids` lists the proteins of each precursor's peptide and differs between the rows of one protein group (71 groups in submission 4cc229a7, 440 in 300beac4), so the protein level kept whichever row came first. apb-fasta now annotates each group by its own members. The ion level keeps `Protein_Ids`.
- **Breaking:** a column or layer that carries no information is not created; each level's `parse` record lists it under `result.uninformative_fields`, by location `obs`, `var` or `layers`, with summary entries `single_valued_fields` and `effectively_empty_fields`. They replace `effectively_empty_layers` and `result.layer_diagnostics.effectively_empty`.
  - **Single-valued:** every filled cell holds one value, judged from two filled cells on; the finding keeps the value. DIA-NN's ion-mobility layers on Astral runs (0) and `Lib_Peptidoform_Q_Value` without peptidoform scoring (0) are such fields
  - **Empty:** no filled cell, finding `{"filled": 0}`; an effectively empty layer (under 0.1% filled while another is at least half filled) is left out too, where it was kept with a warning, its finding adding the occupancy and a six-number summary
  - **Kept:** axis keys, role columns, computed columns, the inputs of `apb_Decoy` and `apb_Contaminant`, the primary and sample layers. `--strict` still makes an effectively empty layer an error
  - **Committed samples:** 44 layers fewer across 11 levels; a consumer asking a catalog for a field left out gets `None`
- **Breaking:** DIA-NN's experiment-wide and library values are `var` columns, not layers: they hold one value per feature, which a long report repeats in every run row. Ion `Lib_Q_Value`, `Global_Q_Value`, and from 1.8 `Lib_PTM_Site_Confidence`, `Lib_Peptidoform_Q_Value`; protein `Global_PG_Q_Value`, and from 1.8 `Lib_PG_Q_Value`, both with duplicates mode `min`, since DIA-NN writes 1.0 into the rows of precursors without a global q-value (Q8N1G4 in submission 4cc229a7: 0.000651, was 1.0 in some runs). Each entry carries a `note`. `Predicted_IM` and `Predicted_iIM` stay layers: they differ across runs on timsTOF data; so does AlphaDIA 1.12's `Mz_Observed`.
- Rule schema 0.9: an optional `note` on every column and layer entry, which parsing ignores; a `duplicates` block on var columns; the duplicate mode `min`. Packaged rules and tests move to `0.9`; schema 0.8 documents are refused.
  - **Var duplicates:** `keep_first` (default) keeps a feature's first nonblank value, where it kept the first row's value, blank or not; `min` and `max` keep the smallest or largest number as the vendor wrote it, for a tool that writes placeholders into some of a feature's rows. Obs payload keeps its first row
  - **Record:** each level's `parse` record lists `result.var_disagreements` (per vendor column, the features whose rows disagree, with up to five examples) and a summary entry `var_disagreements`, `attention` above zero
  - **Layers:** `min` mirrors `max`, read and reduced as numbers
  - **Runtime:** the resolver builds one `FeatureValuePolicy` per var payload column; `LongSourceDecomposer` and `WideSourceDecomposer` take them as `var_values`
- **Breaking:** the WOMBAT-P rule and parameter parser write `software_name` `WOMBAT-P`, as the tool's publication names it (doi 10.1021/acs.jproteome.3c00636), not `WOMBAT` and `Wombat`. The slug follows: `--software wombatp`; the parameter-parser registry takes `wombatp` and `wombat-p`. Names APB reads from ProteoBench keep ProteoBench's spelling; folder, file and `prepare.how` names stay `wombat`. WOMBAT results written before bind no apb-catalog entry; regenerate them.

## 2026-10-22

- **Breaking:** a `FeatureRelation` links features of any levels and annotation tables, not one annotation table to one level. Step 2b of the workspace plan "Exports as ParsedLevels": apb-export's linked targets map peptides, peptidoforms and ions to proteins in one matrix, as mulink reads it.
  - **Model:** `FeatureRelation(coordinates, metadata)`; coordinates `source`, `row`, `target`, `column`, `value`, where `source` and `target` name a level or an annotation table; `annotation_table` and `target_level` are gone
  - **API:** `ParsedLevels.with_feature_relation(name, coordinates, metadata=None)`; it refuses a name that is neither a level nor an annotation table
  - **h5mu:** one directed matrix per relation in the root `varp`, under the relation's own name, without the `relation_` prefix; rows are sources, columns targets
  - **Parquet and DuckDB:** the coordinates as they are; the manifest no longer names a table and a level
  - **Sidecar:** each relation lists its `blocks`, `{source, target, links}`
  - **observation_groups:** a group keeps the links between its own levels and the annotation tables
  - Storage versions: HDF5 descriptor `8`, Parquet manifest `9`, DuckDB manifest `8`, representation document `7`
- **Breaking:** `write_container_representation` leaves `apb2.api`: apb-export, its only caller, now writes every export through `write_parsed_levels`.
- `ParsedLevel.export_uns` and `ParsedLevels.export_uns`: keys a consumer reads from a file's top-level `uns`, beside APB's own `uns["apb"]`. Exporters set them; every parsed result leaves them empty. Step 2 of the workspace plan "Exports as ParsedLevels", so apb-export's targets can be written by `write_parsed_levels`.
  - **API:** a new field on both classes and a new last parameter `export_uns` on `ParsedLevel.build`; `api.py` exposes no new names
  - **Written:** natively, never as JSON text; a level's into its modality or the h5ad's `uns`, the result's into the MuData's or the h5ad's `uns`
  - **Validated:** the key `apb` and any value AnnData cannot store as it is (lists of records, lists with nulls, mixed lists, keys HDF5 cannot link) are refused before the target is touched; an h5ad also refuses the level's name and a key on both level and result
  - **Read:** exactly the keys the storage descriptor lists; keys another tool adds to `uns` are not read
  - **Parquet and DuckDB:** an `export_uns` object beside `apb` in the manifest, root and levels
  - **Sidecar:** `root.export_uns` and `levels[].export_uns`; containers APB2 did not write show an empty object
- `write_parsed_levels(parsed, target, var_name_prefix=True)`: with `False`, an `.h5mu` keeps each modality's var names as written, not prefixed with `ion:` or `prt:`, and the index keeps the name of a single String key. AlphaPeptTools reads modalities this way. Other formats prefix nothing. `H5muWriter` and `MuDataWriter` are frozen dataclasses with that field.
- **Breaking:** storage versions: HDF5 descriptor `7`, Parquet manifest `8`, DuckDB manifest `7`, representation document `6`. Results written before cannot be read; regenerate them.
- h5ad and h5mu results compress their arrays with gzip level 4; they were written uncompressed while Parquet and DuckDB compress by default. gzip rather than lzf, which only h5py reads. Variable-length strings stay uncompressed, as HDF5 filters cannot reach them. No version change: readers see the same layout.

## 2026-10-21

- **Breaking:** `ParseRuleCompiler(data, parameters_path, …)` becomes `ParseRuleCompiler.detect(data, parameters_path, …)` with the same parameters. `ParseRuleCompiler` is now a frozen dataclass of its resolved state, `detection` and `parameter_file`, which `detect`, `from_software` and `from_rule` build; the alternate constructors no longer bypass `__init__`. Callers in apb-proteobench and apb-export are updated in the same change.

- A conversion that reads a parameter file keeps it in the result: the root parse record holds `provenance.search_parameters`, and the `.apb.json` sidecar shows it at `root.apb.parse.provenance.search_parameters`. Before, the typed record existed only during conversion, so a report's search-engine version, a per-run normalisation answer and a per-corpus settings view had nothing to read.
  - **Record:** `file_name`, the parameter file's base name, never a path; `sha256` of its bytes; `parameters`, `Parameters.model_dump(mode="json")` with every field, `null` where the parser recovered nothing
  - **Reading it:** `apb2.api.Parameters.model_validate(record["parameters"])`; `api.py` is unchanged
  - **Once per result:** root only, never per level; every file of a split conversion carries it
  - **Without a parameter file** (`ParseRuleCompiler.from_software`, `--software` alone, explicit rule compilation): no `provenance` key, and the root parse record stays as before
  - **Reverses** the documented rule that typed vendor parameters are never embedded in parsed provenance
  - **Schema:** the published representation schema embeds the `Parameters` model's own JSON Schema for the record
  - **Read once:** the compiler reads the parameter file's bytes once, then parses and hashes those same bytes
  - **Storage:** unchanged. h5ad and h5mu keep the record as a native group, with nulls and modification lists as recorded JSON text; no storage, manifest or representation version changes

- **Breaking:** `Parameters.ident_fdr_psm`, `ident_fdr_peptide` and `ident_fdr_protein` are bare numbers in [0, 1], `0.01` rather than `{"value": 0.01}`, in the model, in `parse_search_parameters()` output and in the stored record. Code that reads `.value` or `["value"]` from them must read the number itself; apb-proteobench's `_probability_parameter` already passes numbers through.

- The `Parameters` model moved from `parserV2/vendor_params/parsers/shared/model.py` to `parserV2/search_parameters/model.py`, a leaf package below both vendor parameter parsing and result I/O, so the representation schema uses the model itself instead of a copy. An architecture test keeps the package free of other apb2 imports. `ParamsError` moved to `vendor_params/parsers/shared/errors.py`. `apb2.api` exposes the same names.

- Vendor parameter parsers read a byte stream with the same newline translation as a file path, so `parse_search_parameters()` gives the same record for a path and for its bytes. Across 329 committed and corpus parameter files under every single-file parser, the two already agreed (578 parses); the change makes that hold by construction.

## 2026-10-20

- The result representation, the `<artifact>.apb.json` sidecar beside every result, has a published JSON Schema: [representation.schema.json](src/apb2/parserV2/parse_quant/io/_schema/representation.schema.json), generated from the models in `parse_quant/io/representation_schema.py`. The document itself is unchanged and stays format version 5.
  - **Strict structure:** levels, tables, columns, observations, both layer kinds and their statistics allow no unknown keys
  - **Open metadata:** `apb` parts and annotation `metadata`, except that producer records follow the summary contract, with `status` ok, attention or not_checked and no value when not checked
  - **Checked at write:** every sidecar, from `write_parsed_levels` and from `write_container_representation` (apb-export), is validated before it is written; a document outside the schema fails the producing call and leaves no sidecar
  - **Dependencies:** `pydantic>=2.12`, the first version that checks a TypedDict's `extra_items`; `typing-extensions` is now declared directly; `jsonschema` joins the dev group, which tests written sidecars against the published file

- The DIA-NN rules (v1.7, v1.8–1.9, v2) convert every report column they find, not only the columns current consumers read: a rule is a passive dump of the software output, and meanings a consumer needs belong in apb-catalog. All added entries are optional, so older exports without a column still convert.
  - **Ion `var`:** columns constant per precursor, such as `Precursor.Mz`, `iRT` and `Protein.Sites`. `Site.Occupancy.Probabilities` keeps its first value, so it is lossy for the few precursors whose value differs between runs (39 and 80 in two ProteoBench plasma reports).
  - **Ion layers:** per-run values, such as `RT.Start`, `FWHM`, `Evidence`, the q-value family, `PTM.*` and `Protein.Q.Value`
  - **Protein layers:** per-group values (`PG.TopN`, `PG.MaxLFQ.Quality`, the `Genes.*` quantities); v1.7 also gains `PG_MaxLFQ` as an abundance layer, as in v1.8
  - **obs:** `Run.Index`, `File.Name` (v1.8), the vendor `Run` column as `Run_Name` (v1.7)
  - **Fragment level:** v1.7 also reads `Fragment.Quant.Corrected`. v1.8 does not, because a packed fragment value column cannot be optional, and 1.9 reports lack it.
  - **Roles:** new layers carry no role and no missing-value policy, so they are stored as reported, and consumers that select abundance layers by role see the same layers as before
  - **Warnings:** columns that are entirely empty, such as `Genes.Quantity` in exports without gene annotations, are reported as effectively empty
  - **Size:** HDF5 output grows, for example 9.0 → 21.0 MB for a DIA-NN 2.3 Astral submission
  - **Guard:** a test checks that each rule converts every column of its committed header

- Ion levels keep the vendor's peptide-uniqueness flags where the export carries them, all optional: DIA-NN `Proteotypic` and Spectronaut `PEP.IsProteotypic` as booleans, Spectronaut 15 also `PEP.IsProteinGroupSpecific`, and AlphaPept's integer `n_possible_proteins`. They keep the vendor's meaning and names. On the committed samples DIA-NN 1.8 marks 271 of 281 precursors proteotypic, and Spectronaut 15 marks 19 of 43 protein-group specific.

- **Breaking:** protein-list columns, those with the `fasta_accessions` or `protein_assignment` role, are written `;`-separated: a `,` or `;` with surrounding spaces becomes `;`, so FragPipe's `A, B` is `A;B`. This happens when the column is selected, so a protein-list var key such as DIA-NN's `Protein_Group` is checked for uniqueness in that spelling. A protein-list column may declare the vendor's own further `separator`, and any other column declaring one is refused; PEAKS `Accession` declares `:`. `:` cannot be general: MaxQuant and WOMBAT contaminant ids contain it (`CON__ENSEMBL:…`). apb-fasta and apb-proteobench drop their protein-group separator options.

## 2026-10-09

- i2MassChroQ X!Tandem parameters take the fragment tolerance unit from `spectrum, fragment monoisotopic mass error units`; it used to reuse the precursor unit, so 6 of 13 ProteoBench i2MassChroQ submissions read `0.02 ppm` where the file states `0.02 Daltons`. The tolerance-unit lookup accepts X!Tandem's spelling `Daltons`.

## 2026-10-08

- DIA-NN rules map `UniMod:21` (phospho) and `UniMod:121` (GlyGly). A ProteoBench DIA Astral submission searched with both reported them as unknown tokens: 4,386 and 36 of 698,009 precursor rows.

- Numeric layers accept `missing_tokens`, the text a vendor writes for a missing value. Such a cell is missing, not an unreadable cell, and claims no duplicate cell. PEAKS declares `-` on `Sample_Mz` and `Sample_RT_Mean`, and WOMBAT declares `NA` on `Abundance`; these were the only unreadable cells in the ProteoBench corpus (22 datasets).

- A row whose final key has an empty-text component is dropped, like one with a null component. A delimited reader already reads an empty cell as null, but Parquet keeps `""`, so DIA-NN 2 precursors without a protein group (1,771 of 7,357 in a ProteoBench Astral submission) formed a protein feature `""` with no abundances. That feature is gone; the ion level keeps those precursors, with an empty `Protein_Group`. The other 14 routine datasets convert unchanged.

- **Breaking:** a `token_pattern` names the site of every token: each capturing group is `nterm`, `cterm` or `residue`, and a pattern with any other or unnamed group is refused. apb2 places a token where its group says instead of guessing from its place in the text, so the start-is-N-terminal and end-is-C-terminal guesses and the C-terminal fallback are gone. AlphaPept's `cC…` and `oxM…` at the sequence start now resolve on their residue, PEAKS' N-terminal acetyl written after the first residue (`A(+42.01)…`) resolves, all 15 token-pattern syntaxes name their groups, Sage maps `+42` and Spectronaut `Deamidation (NQ)`, and the Unimod registry gains UNIMOD:7 Deamidated.
- **Breaking:** every format keeps a root APB part and one part per level, never merged. An h5ad stores its level part at `uns[<level>]["apb"]` with its own storage descriptor and its sidecar exposes `root.apb`; the h5ad merge and `metadata_ownership` are gone. HDF5 descriptor 6, Parquet manifest 7, DuckDB manifest 6, representation 5.
- **Breaking:** `ParsedLevel.uns` is the conversion record as stored: `provenance` (`rule_json`, `plan_json`, input preparation, sample matching), `result` (unknown tokens, layer diagnostics) and `summary` (unknown modification tokens, unreadable cells, effectively empty layers); observation grouping writes into `result`. Annotation records carry `schema_version` 3 and `provenance.annotation` at the root and `result.annotation` and a `summary` on each level.
- Writers and readers check every record's `summary` entries and `details` references without importing producers.
- `apb2.api` exposes `UnsJsonCodec`, so apb-export stores APB parts in its containers exactly as APB2 results do. A standalone AnnData's sidecar names its level by the `uns` key holding its level part.

- MaxQuant ions no longer add up PEP, score and retention time over repeated evidence rows of one ion and run: the ion rule keeps the lowest-PEP row for those layers (`keep_best`) and still sums `Intensity` and `MS_MS_Count`, through the new `duplicates.sum` list of layers `keep_best` adds up. Summing had produced PEP above 1 (1.41 and 1.11 in two single-cell submissions), which apb-aggregate rejects. PEP is now a required MaxQuant ion layer.

- Every rule now marks decoys and the contaminants the software itself flags or adds: new computed columns `apb_Decoy` (`how: "decoy"`) and `apb_Contaminant` (`how: "contaminant"`), declared exactly once on every level and never missing. They read the vendor's flag (`equals`, such as MaxQuant `Reverse` `+`, AlphaPept `decoy`, Spectronaut `EG.IsDecoy`), a protein-name prefix (`prefix`, with an optional list `separator`, such as WOMBAT's `CON__`), or declare that the vendor writes none. Rows are kept, only marked. MaxQuant evidence, DIA-NN and Spectronaut fragment and protein levels read their flag columns where present. The MetaMorpheus preparation keeps decoy peptides instead of dropping them.
- A sequence character that is neither a residue letter nor part of a declared token now fails the conversion with `UnrecognizedSequenceCharacterError`; it used to be dropped, which glued WOMBAT's `SEQA|SEQB` into one peptide and kept FragPipe's terminal `n` as a residue. FragPipe's token pattern places `n[…]` and `c[…]` on the termini, so `n[42.0106]` resolves to `[UNIMOD:1]-`; Sage, i2MassChroQ and WOMBAT accept ProForma's `[…]-` and `-[…]`; the new `marker_pattern` removes AlphaPept's `_decoy` suffix, which used to become the unknown token `decoy`.
- WOMBAT tables have a preparation, `wombat`, that drops rows naming several peptides (1,183 and 1,507 rows in two ProteoBench Compomics submissions) and logs the count; prepared tables read a `.csv` file comma-separated.

- New duplicate mode `keep_best` keeps, in each cell, the repeated row whose `by` layer is best (`best`: `highest` or `lowest`) and takes every layer's value from that row, so one ion's intensity, score and q-value describe the same PSM. `max` and `sum` reduce each layer on its own and could mix rows; `keep_first` chose by file order. The ranking layer must be a required, plain numeric layer; a row without a ranking value loses to one with it, and ties keep file order. AlphaPept, whose `results_peptides.csv` holds one row per PSM, now keeps the highest-`score` PSM instead of the first, and requires `score`. AlphaPept writes each ion's PSMs of a run best-scored first, so its values do not change; the rule no longer depends on that order.

## 2026-10-07

- A modification token with no residue to modify and at neither terminus, such as the second token in `(ac)(ox)PEPTIDE`, now fails the conversion with `UnplaceableModificationError`; it used to be dropped from the sequence without a trace. No corpus export produced one. A token after the last residue still modifies that residue when the map places it there (`PEPTIDEM(ox)` → `PEPTIDEM[UNIMOD:35]`), now pinned for MaxQuant and DIA-NN.
- The MSAID parameter parser is removed: no MSAID conversion rule exists and the ProteoBench catalogue holds no MSAID submission, so `parse_search_parameters(..., "msaid")` now fails like any unregistered software.
- DIA-NN 1.8/1.9 fragments take their labels from `Fragment.Info` when the report carries it, as FragPipe's DIA-NN reports do (21 ProteoBench submissions, FragPipe 22.0–24.0): `ProForma_fragment` becomes `…/2/y6-unknown^1` instead of `…/2/frag_0`. DIA-NN's own reports carry no `Fragment.Info` and keep positional labels. The column-labelled fragment declaration gains `label_required` (default true); `false` lets a source without the label column fall back to positions. The column-labelled separator existed but no packaged rule used it.
- The explicit long-join MaxQuant document and its join (`join`, which unpivoted the higher-level exports before joining them) are removed; `maxquant_wide` produced the same levels, identities and values for all 15 input subsets and is the only MaxQuant document. The document field `selection`, which only kept that document out of detection, is removed with it, as is the `maxquant` preparation name.
- AlphaDIA 1.12 rejects `precursors.tsv` without `precursor.matrix.tsv`. Its `intensity` column repeats the protein-group intensity on every precursor row, and the lone file was read as ion abundance. Precursor intensities come only from the matrix, written when `search_output.precursor_level_lfq` is true (default false); a folder with both files converts as before. The join now recognizes the precursor table only with a `run` column and the matrix only without sequence columns, so ProteoBench's merged 1.10 matrix still selects the 1.10 rule. The committed v1_12 sample gains the matching rows of its real matrix (`input_file_secondary.tsv.gz`), cut by `make_test_samples.py`, which now samples `<export stem>_*` companions.
- MaxQuant ions carry their protein group. In `maxquant_wide` the evidence table has its own preparation, `maxquant_evidence`, which keeps every evidence row and adds the `Protein IDs` of the referenced group (`Protein group IDs`) holding the row's leading razor protein, MaxQuant's own assignment of shared peptides. The ion rule names it `Protein_IDs`, the protein level's key, and gives it `protein_assignment`; `Proteins` keeps `fasta_accessions`. Ion and protein levels now share one identity: in the entrapment record evidence's `Proteins` (`DLSPVLK_target`) and the group (`DLSPVLK_target;REV_DPSPVPK_target`) differed for 14 groups, in BoxCar for 51,156 of 81,449 ions, so apb-aggregate could not roll ions into the vendor protein level. With evidence alone `Protein_IDs` is evidence's `Proteins`. A row whose razor protein lies in no or several referenced groups fails the conversion; a file named `evidence.txt` without the evidence columns is reported as such. All other levels and every value are unchanged; the 1,622-run yeast set peaks at 22.9 GB instead of 16.7 GB because evidence is now read through the preparation path.
- Prepared and Parquet tables read their text layer columns as numbers when the rule takes the `sum` or `max` of repeated cells, as the direct reader does; such rules were rejected on prepared tables before.
- AlphaDIA 2.x ion levels assign proteins through `Protein_Group` (`pg.name`), the column that keys the protein level, instead of `Proteins` (`pg.proteins`), which keeps `fasta_accessions`. The two levels now share one protein identity, so apb-aggregate rolls ions into the existing protein level; it previously stopped because the ion groups' column was absent from the protein axis.

## 2026-10-06

- `api.py` adds `write_container_representation(container, artifact)`, the version-4 `<artifact>.apb.json` sidecar for an AnnData or MuData APB2 did not write, such as an apb-export file, so APB Studio's viewer can show exports. Each AnnData, or MuData modality, is one level keyed by its obs and var names; X and every layer are summarised as numbers, unstored sparse cells as missing, and `uns["apb"]` becomes the level's metadata. Both sidecar writers share one atomic publication.
- Code that only tests reached is removed, after a coverage run of every APB Studio corpus workflow found nothing else calling it: the single-level writer path (`ParsedLevelWriter`, `Parser.writer` and `Parser.convert`, `ParsedLevelFormatWriter`, `ParquetWriter`, `AnnDataWriter.write` and `.to_anndata`), since every write goes through `write_parsed_levels` and the `ParsedLevels` writers; the `DelimitedFile` input source and its stated-dialect evidence, which nothing constructed, so an ambiguous dialect error no longer suggests binding one; `detect_rule_document`, superseded by `detect_rule_documents`; and the registry's `available_software`. Conversion results and `api.py` are unchanged.

## 2026-10-05

- Added the MetaMorpheus 1.x rule for FlashLFQ `AllQuantifiedPeaks.tsv` at ion level (category DDA), with the new `metamorpheus` preparation. The preparation keeps peaks with `Full Sequences Mapped` = 1, because an ambiguous `SEQA|SEQB` peak would otherwise parse into one concatenated peptide, drops decoy peaks (`Decoy Peptide`) and FlashLFQ's random-RT match-between-runs decoys (`Random RT`), and casts `Peak intensity` to a number, because prepared tables are read as text. Repeated peaks of one ion and run collapse with the `max` duplicate mode, as FlashLFQ computes its own peptide intensity as the max over a run's peaks; real match-between-runs peaks are not filtered by q-value, since the threshold is a MetaMorpheus setting the table does not carry. Modification tokens such as `Common Fixed:Carbamidomethyl on C` map to UniMod. The committed fixture records five runs and 178 ions. ProteoBench's legacy parser sums repeated peaks, so ion values differ from it where a run holds more than one peak of an ion.
- New duplicate mode `max` keeps the largest present value of repeated cells, beside `sum`. Like `sum`, it requires plain numeric layers read as numbers and leaves a cell null when nothing is present; each layer reduces separately. FlashLFQ computes its own peptide intensity as the max over a run's peaks, so MetaMorpheus `AllQuantifiedPeaks.tsv` ([PR #2](https://github.com/anndata-omics-bridge/apb2/pull/2)) is its first user. `NUMERIC_DUPLICATE_MODES` in `operations.py` now drives the three numeric checks.
- The duplicate mode `aggregate` is renamed `sum`, so the mode names its reduction. The packaged MaxQuant and `maxquant_wide` ion rules use `sum`, conversion results are unchanged, and a rule document still declaring `aggregate` now fails validation. The policy class keeps its name, `AggregateNumericDuplicates`.
- New [rule coverage](docs/rule-coverage.md) page lists what a rule document and the standard reader handle, what only a preparation hook may do, and the known gap that prepared tables are read as text. `AGENTS.md` now allows a hook for a single file whose rows are not a hierarchy level, such as FlashLFQ peaks, and limits every hook to bringing rows to a level.
- MaxQuant's higher-level exports are joined wide: the new packaged `maxquant_wide` document and `join_wide` resolve the same foreign keys as before but keep each table's per-experiment columns side by side, so rows grow with feature relationships rather than features × experiments, and they read only the columns the rules declare plus `id` and `Evidence IDs`. Detection uses `maxquant_wide`; the long-join `maxquant` document stays packaged with the new document field `selection: "explicit"`, which keeps it out of detection for use through `--rule-config`. Both produce identical levels, identities, values and missing cells for all 15 input subsets and for the four-table entrapment record, where peak memory fell from 1.2 GB to 0.7 GB. Prepared tables now filter absent rows only on key columns they read, since a wide level's observation keys come from header captures.
- MaxQuant preparation no longer copies per-sample columns such as `Experiment <sample>` and `Identification type <sample>` into every sample's long rows; they describe one sample, not the feature, and no rule reads them. Feature annotations are shared by all samples' rows. The 1,622-run yeast interactome `peptides.txt` (6,521 columns) now converts in about 3 minutes at a 27 GB peak instead of being killed by the operating system above a 100 GB footprint.
- Conversion logs every step's start and its `done` or `failed` end with seconds and peak resident memory: rule detection (plus the rule chosen per level), each MaxQuant/AlphaDIA preparation table read and the join with its result shape, per-level reading and the five parse steps, and writing. Previously the first line came after detection and preparation, so a conversion killed for memory there, such as the 1,622-run MaxQuant yeast interactome reshaping `peptides.txt`, left no trace.
- Parameter-free Spectronaut selection partitions exports by two columns instead of the optional `EG.Comment`: 15 has neither `E.Errors` nor `EG.LibraryQvalue`, 19/20 has `E.Errors` without `EG.LibraryQvalue`, and 21 has `EG.LibraryQvalue`. Spectronaut 15 exports without `EG.Comment` (the directLFQ SN15 re-run, iq long-format and quicktests reports) now convert with `--software Spectronaut`, and every level of one export resolves to a single rule; the selection tests now cover all levels, not only ion.
- DIA-NN rules retain the library and experiment-wide protein-group q-values as protein layers, declared like `PG_Q_Value`: 1.8/1.9 and 2.x keep `Lib_PG_Q_Value` (`Lib.PG.Q.Value`) and `Global_PG_Q_Value` (`Global.PG.Q.Value`); 1.7, which reports no `Lib.PG.Q.Value`, keeps `Global_PG_Q_Value`. prolfquapp filters DIA-NN protein groups by `Lib.PG.Q.Value` and `PG.Q.Value`. The DIA-NN protein fingerprints change.
- `uv.lock` is no longer committed: `make sync` and CI resolve the environment from `pyproject.toml`, `make check` drops `uv lock --check`, CI caches by `pyproject.toml`, and the dev group pins `ruff==0.16.10` and `pyright==1.1.414` so lint and type results stay stable.
- README shows the Zenodo DOI badge, linking the concept DOI that resolves to the latest archived release, and the PyPI version badge.
- 0.1.1: `CITATION.cff` records the author's ORCID, so the Zenodo archive of each GitHub release carries complete citation metadata and a DOI.

## 2026-10-04

- PyPI release setup: `.github/workflows/publish.yml` builds and checks the distributions, then publishes to PyPI through trusted publishing for a published GitHub release tagged `v<version>`; a manual run only builds and checks. README links point at the documentation site so they resolve on PyPI, README gains an installation section, and `pyproject.toml` adds keywords and classifiers.
- The `apb2` command imports APB2 only through `api.py`: `apb2/cli/` replaces `cli.py` and `command/` (console script unchanged). `api.py` adds `ConversionError`, which every expected conversion failure now is (unreadable parameter or rule files raise it instead of `ParamsError`), `AnnotationResult` and `LevelParseTimings`. `ParseRuleCompiler.from_rule(data, rule, ...)` compiles a caller's rule file, and `ParsedLevels.observation_groups()` replaces `group_observations()`; the internal `formats.reformat` is gone.
- Public signatures drop the `/` and `*` markers (`ParseRuleCompiler`, `from_software`, `get_rules`, `read_parsed_levels`, `write_parsed_levels`, `sidecar_path`, `AnnotationCompiler.compile`, `SdrfSource`); every call that worked before still works. The API guide, index and result-I/O pages import only from `apb2.api`: the explicit-format readers and writers, `reformat` and the result-model types are internal, and the guide's `AnnDataLayerContractError`, which no longer existed, is gone.

## 2026-10-03

- `scripts/make_test_samples.py` replaces APB Studio's `make_apb2_test_samples.py`, so APB Studio no longer imports apb2. It takes the fixture store as `STORE` and reads its current `downloads.csv`; the old script still looked for `raw_file_db_downloaded.csv` and `json_dir/` and found no exports. `scripts/` is linted and type-checked with `src` and `tests`.
- `apb2.api` adds `LevelHierarchy` and `packaged_rule_declarations()`, which lists every distinct `rule_json` each packaged rule level can store, so catalogue and test code no longer read the rule loader.
- `apb2.annotation_extension` is gone; `apb2.api` adds `AnnotationCompiler`, `AnnotationError` and `SdrfSource`. `AnnotationCompiler(unmatched="keep" | "error" | "drop", include=None)` replaces `ProlfquappAnnotationParameters` and `SdrfAnnotationParameters`; `SdrfSource.read(path)` reads an SDRF. The `apb2 annotate` workflow moved from `annotation_facade.py` to `command/annotation.py`.
- Typed var/layer semantic roles, strict shared observation axes, values-only matrices, and persisted configurable level hierarchies; Parquet 6 and DuckDB 5.

- 2026-10-03: The JSON result representation decodes embedded JSON text only in `rule_json` and `plan_json`. A level metadata entry named `aggregate` is no longer special: apb-aggregate now stores its history there as a native JSON list, which projects unchanged.

- 2026-10-03: `apb2.api` is the one public module. `result_facade.py` and `modification_facade.py` are gone. `api.py` adds `ParsedLevel`, `FinalLayerTable`, `JsonValue`, `ResultIOError`, `canonical_modification_names` and `sidecar_path`. Consumers build levels with `ParsedLevel.build()` and `with_layers()`, select abundance layers with `ParsedLevel.abundance_layers()`, read values with `FinalLayerTable.quantitative_values()` and `decoded_values()`, and attach annotations with `ParsedLevels.with_annotation_table()` and `with_feature_relation()`. The layer-selection classes and constants, `observation_labels`, `quantitative_layer_values` and `get_provenance` are no longer public.

- 2026-10-02: Numeric `missing_values` accept, besides exact numbers, at most one `"<=<number>"` threshold. Values at or below it are missing before duplicate resolution and in the final layer, and aggregate-duplicate rules may declare it. Every packaged `abundance` layer declares `"<=0"`, enforced by a package test, so vendor zero and negative abundances now convert to missing values. Before, only some AlphaDIA, FragPipe, MaxQuant, PEAKS and Sage layers mapped exact `0`; DIA-NN, Spectronaut, AlphaPept, WOMBAT and the others kept zeros as observations. Rule fingerprints changed for every packaged rule.

- 2026-10-02: Fix: the i2MassChroQ rule derives `ProForma_peptide` with `stripped_sequence` from `Sequence` (inline `[MOD:nnnnn]`, `[42]` tokens, `after_residue`), instead of copying the vendor `sequence` column, which carries PSI-MOD annotations and ProForma N-terminal prefixes in most exports. `ProForma_peptidoform`, `ProForma_ion` and `Sequence` are unchanged. Across the 13 ProteoBench i2MassChroQ corpus submissions, FASTA-unmatched features drop from 16–24% to 0 in the 9 exports with modified `sequence` values, and ProteoBench `nr_feature` moves from 16–24% below upstream to within 0.9%.

- 2026-10-02: Fix: an inline modification token written after a peptide's last residue (`after_residue` syntax) is now that residue's modification whenever the modification map places it on that residue: DIA-NN `…STAC(UniMod:4)` becomes `…STAC[UNIMOD:4]` instead of the unresolved C-terminal `…STAC-[UniMod:4]`. It stays C-terminal only when no entry fits the residue. Affects the canonical ProForma columns and feature keys of every `after_residue` rule (DIA-NN, Spectronaut, FragPipe, MaxQuant, PEAKS, quantms, Sage, WOMBAT); such tokens no longer appear in `unknown_mod_tokens`.

- 2026-10-02: DIA-NN 1.7, 1.8/1.9 and 2.x rules retain the library and experiment-wide precursor q-values as ion layers `Lib_Q_Value` (`Lib.Q.Value`) and `Global_Q_Value` (`Global.Q.Value`), declared like `Q_Value`. ProteoBench's entrapment module ranks DIA-NN precursors by one of these.

- 2026-10-02: Wide rules may declare `measurements.sample_layer`, naming the layer whose header captures are the sample names; omitted, the primary layer supplies them as before, and long rules reject it. The PEAKS ion rule sets it to `Sample_Mz` and drops the `LFQ_` prefix from its four layer patterns, so runs of any name are samples, while group-average columns (`A Normalized Area`, `Group 1 Normalized Area`) still are not. The ProteoBench DIA plasma PEAKS submission now converts with 12 samples. The PEAKS ion fingerprint changes; all previously converting PEAKS corpus files keep identical output except diaPASEF `5691d485`, whose `m/z` and `RT mean` headers carry a garbled run name, so that run is no longer a sample.

- 2026-10-02: `apb2 annotate` and `AnnotationCompiler` recognise SDRF-Proteomics tables by their `source name` and `comment[data file]` headers. Rows are keyed by data file, with the extensionless basename as an exact alias. Other columns are added to `obs` under sanitized names, and repeated headers stay separate columns. Provenance under `sdrf.provenance.annotation.columns` maps each verbatim header to its `obs` column. Multiplexed labels are rejected, and SDRF structure and ontology terms are not validated. `LoadedAnnotationSource` gains a required `headers` field with the verbatim header text, and `apb2.annotation_extension` now exports `SdrfSource` and `load_annotation_file`. Existing prolfquapp tables and conversion output are unchanged.

- 2026-10-01: The AlphaDIA 1.12 rule no longer retains `pg_qval` as ion var column `Protein_Group_QValue`: it is a global protein-group value, carried since the parser v2 port without a recorded reason, and the precursor TSV has no protein quantity to give it a protein level. The AlphaDIA 1.12 ion fingerprint changes. AlphaDIA 2.x moves `pg.qval` from ion var to a new protein level keyed by `pg.name` (roles `protein_assignment`, `fasta_accessions`), quantified by `pg.intensity` as layer `PG_Intensity`, with `pg.master_protein`; `pg.proteins` and `pg.genes` stay on ion because they differ between precursors of one group. The AlphaDIA 2.x ion fingerprint changes and a protein level appears.

- 2026-10-01: DIA-NN protein levels retain the run-specific protein-group q-value as layer `PG_Q_Value` (1.7, 1.8/1.9 and 2.x) and its posterior error probability as layer `PG_PEP` (2.x; the 1.7 and 1.8 fixtures carry no `PG.PEP`). Effective-rule fingerprints of all three DIA-NN protein levels change. `Global.PG.Q.Value` remains unretained.

- 2026-10-01: DIA-NN 1.8/1.9 and 2.x rules retain the run-specific precursor posterior error probability as ion layer `PEP`, declared like `Q_Value`; DIA-NN 1.7 reports carry no `PEP` column, so that rule is unchanged. The PEAKS rule retains the per-feature identification score `-10LgP` as the numeric var column `Minus_10LgP`. Effective-rule fingerprints of the DIA-NN 1.8 ion, DIA-NN 2.x ion and PEAKS ion levels change. `Global.Q.Value`, `PG.Q.Value` and `PG.PEP` remain unretained.

- 2026-09-23: Added the Python-only `ParseRuleCompiler.from_software()` constructor for vendor results uploaded without parameter files. It reuses packaged column/rule matching, requires an explicit result producer, and rejects ambiguous versions or missing scientific evidence without older-rule fallback. No CLI behavior changed.

- 2026-09-23: Added the ProteoBench Custom wide ion-upload fixture and packaged `pb_custom` rule. `apb2 convert ... ion --software pb_custom` now selects that rule without a parameter file; other packaged rules still require their vendor parameters. The fixture records six runs and 8,493 ions.

- 2026-09-21: **Breaking:** conversion has one optional `--software` hint, selecting parameter-file grammar and restricting packaged result probes to that vendor and its declared quantification software. Removed `--params-software` and compiler `parameters_software`; FragPipe parameters can still select DIA-NN results. Unhinted recognition scans all vendors but resolves only plausible rules afterward; mismatches never trigger a global fallback and ambiguous vendors are explicit errors. Explicit rules retain their own selection. No persisted-format or scientific-output change.

- 2026-09-21: `apb2 convert --timings-output PATH` now optionally publishes a versioned, atomic JSON file of internal phase and per-level timings. The file is separate from the APB result and scientific representation; existing INFO logs remain unchanged.

- 2026-09-21: `apb2 convert` now logs separate wall times for rule compilation, bound input reading, parsing (including observation alignment), and result writing, with read/parse timings for each selected level. Multi-file source preparation occurs during compilation and is included in that phase; the measured read phase covers each compiled level's bound reader. No result format or conversion semantics changed.

- 2026-09-18: Added the normative [APB metadata specification](docs/metadata_specification.md), separating APB2-owned storage/ownership invariants from tool-owned payload schemas. Linked the I/O and architecture guides to it, corrected stale version and annotation paths, and documented existing checks plus remaining read/write-envelope validation gaps. Documentation only; no runtime or format change.

- 2026-09-18: **Breaking:** tool namespaces now live directly under their owning object's `uns["apb"]`, without shared/level wrappers. Generic H5AD composition rejects conflicting leaves and preserves scope ownership with paths, including empty mappings, in the storage descriptor. Columnar manifests rename root `shared` to `apb`. HDF storage 3, Parquet 4, DuckDB 3 and representation 4 reject older layouts. Annotation provenance is tool-owned and stores each source descriptor once. Representation reuses persistence projection, with one combined namespace for H5AD and separate root/modality trees for collections; matrices, axes, aligned tables and calculations are unchanged.

- 2026-09-17: **Breaking:** result storage now has one owner for every metadata field. H5 uses `uns["apb"]["shared" | "level" | "storage"]`, keeps the primary matrix only in `X`, and rejects the previous envelope version. Parquet and DuckDB use the same shared/level scope split and new manifest versions. Resolved plans no longer embed provenance, semantic roles live under the level `roles` section, and the adjacent representation sidecar moves to version 3 without becoming part of `uns`.

- 2026-09-17: WOMBAT declares `keep_first` duplicate handling for both packaged levels, matching its repeated precursor rows without adding vendor-specific single-file preprocessing. Multi-file preparation remains directory-only at the public conversion boundary, with no per-file auxiliary CLI or facade parameters.

- 2026-09-17: MaxQuant now declares two table groups: direct evidence produces ions keyed by `Raw_File`, retaining experiment/fraction metadata; only the three higher exports are prepared together, with shared `Experiment` observations in their group base. Preparation is table-local and all 15 input subsets remain supported. Explicit complete one-to-one metadata mappings align observations without changing quantitative cells; fractionated or unmapped resolutions produce separate key-qualified results. Conversion summaries and CLI logs report actual outputs. Mapping decisions remain ordinary parse provenance; no storage schema changes.

- 2026-09-17: MaxQuant preparation accepts all 15 nonempty subsets of evidence, modification-specific peptide, peptide and protein-group exports. Higher-level combinations no longer require the evidence export: shared evidence-ID references and wide-column sample names provide their join keys. Only supplied levels are produced; missing requested levels still fail before writing. Tests cover every subset through HDF5, Parquet and DuckDB, plus renamed-input recognition and sample-sensitive joins.

- 2026-09-17: Rule schema 0.7 adds rule-selected input preparation. Tool-specific joins live in `parserV2/joins/alphadia.py` and `maxquant.py`; parent composition reads directory inputs and shares one prepared frame across detection and level parsing. AlphaDIA 1.12 enriches authoritative matrix quantities with precursor metadata. MaxQuant replaces independent-table rules with one long-table rule: wide measurements are unpivoted, evidence foreign keys are joined with experiment identity, and join fan-out does not multiply ion measurements. Preparation provenance records sources, elapsed time, row count and estimated frame size. Generated schemas, tests and packaged documents use 0.7 without a compatibility path. DIA-NN is unchanged.

- 2026-09-16: Sample-annotation matching can now persist `normalize: "mass_spec_basename"` for exact or fuzzy policies. It compares run identifiers after removing directory components and recognized mass-spectrometry file suffixes while retaining original observation values and diagnostics, and rejects normalization collisions. quantms, i2MassChroQ, ProlineStudio, and DIA-NN opt into exact normalized matching; PEAKS retains fuzzy matching with a calibrated ambiguity margin; MSAngel canonicalizes its DDA-prefixed measurement names.

- 2026-09-16: Breaking rule schema 0.6 adds logical numeric measurement types. Numeric layers default to `number`; seven packaged count layers declare `integer`, whose encoding accepts whole values and missing cells but rejects fractional and infinite values. Resolved plans and compact result sidecars retain the logical type independently of the physical matrix dtype, so nullable AnnData counts remain `Float64` while the viewer labels them as integer measurements. Schema-0.5 documents require an explicit version update; no migration model is provided.

- 2026-09-15: Breaking rule schema 0.5 replaces root `input`/`base`/`levels` with `tables: [{input, base, levels}]` for every vendor/version. MaxQuant now has one rule document with four table-local declarations; the three separate peptide, protein-group, and modification-specific documents are removed without aliases. Declared filenames select direct-input groups, renamed files require unambiguous table recognition, and folder compatibility is checked per table. Packaged and explicit rule-config conversions share selection and parse-before-write behavior. All 35 declared levels and 101 effective parameter configurations retain their parsing semantics, excluding revision labels. Generated schemas distinguish authored documents from composed effective rules.

- 2026-09-15: PEAKS parameter parsing now accepts the newer Astral/diaPASEF report layout: `Peptide Length`, `Missed Cleavage`, compact LFQ charge bounds, modification names without masses, `Q Method`, `Normalization`, and FDR values whose percentage unit is in the label. Modification lists stop at their own section boundary. Missing match-between-runs and digestion settings remain unknown rather than becoming inferred booleans. Identification FDR labels remain distinct from plural protein and LFQ labels; all extracted settings are metadata, with no additional measurement filtering or conversion-rule changes.

- 2026-09-14: Added packaged ion conversion rules and typed parameter parsers for i2MassChroQ 1.x, quantms 1.x, ProlineStudio 2.x, and MSAngel 2.x. Physical input now supports an explicitly named Excel sheet even when the downloaded filename retains `.txt`; Proline/MSAngel modification lists normalize embedded residue and terminal sites such as `Oxidation (M5)` and `Acetyl (Protein N-term)`. Rules keep vendor recognition narrow: Proline requires its validation column, while MSAngel requires its raw-abundance layer.

- 2026-09-13: Added a dedicated Spectronaut 21 conversion rule for ion, protein, and fragment data, including optional `EG.LibraryQvalue` and `PG.LibraryQvalue` measurements. Spectronaut parameter parsing now accepts both scalar peptide charges and enabled `Min Charge`/`Max Charge` ranges. The DIA-NN 1.8/1.9 rule now explicitly accepts TSV, TXT, and Parquet reports and retains optional `Ms1.Normalised` abundance; the DIA-NN 2.x version gate remains unchanged.

- 2026-09-13: `apb2 convert` now accepts a vendor-result directory. Packaged detection resolves at most one independently bound rule document per quantification level; an explicit level converts its one named table, while an omitted level converts every available table in canonical order and writes once. MaxQuant folders recognize `evidence.txt`, `modificationSpecificPeptides.txt`, `peptides.txt`, and `proteinGroups.txt`; missing tables are skipped, requested missing tables and duplicate matches are focused errors, and present incompatible tables abort the conversion before output. Explicit `--rule-config` still selects one document. Empty MaxQuant `<enzymes>` now maps to a missing enzyme.

- 2026-09-10: `apb2 convert --format hdf5|parquet|duckdb` now parses into storage-neutral `ParsedLevels` and selects the public APB2 writer directly. An omitted level writes every compatible level in the selected format; HDF5 retains the existing `.h5mu` default, while explicit single-level HDF5 retains `.h5ad`.

- 2026-09-10: Schema 0.4 measurement layers can now declare semantic `roles`, and every packaged rule marks all declared abundance layers rather than only the primary matrix. The role vocabulary and its permitted `obs`, `var`, and `layer` owners live in packaged `role_policy.json`; the generated schema derives the same vocabulary, validation enforces ownership without another Pydantic validator, and conversion projects layer roles into provenance as role-to-layer-name lists alongside `column_roles`.

- 2026-09-09: Every public APB2 result write now publishes an adjacent, atomic `<artifact>.apb.json` scientific representation. Version 2 is projected from storage-neutral `ParsedLevels`; records schemas, dimensions, provenance, structural slots, and layer roles; restores known JSON-text provenance fields as structured objects or arrays; distinguishes quantitative summaries from fixed-size categorical counts; stores bounded observation identities once per level; and labels bounded deterministic quartile samples explicitly. Only finite cells consume the quartile-sample budget, and factor semantics survive APB-authored result reads, rewrites, and reformats without re-encoding stored category codes. A failed replacement invalidates the old sidecar while retaining the new scientific artifact. Conversion and reformatting use the same projection across h5ad, h5mu, Parquet, and DuckDB.

- 2026-09-04 (evening, second): Added a packaged MaxQuant `modificationSpecificPeptides.txt` rule
  producing the `peptidoform` level — the first wide-table peptidoform. The table carries
  modification names and counts but no positions ("2 Oxidation (M)"), so a positioned
  `ProForma_peptidoform` cannot be derived faithfully; identity is instead the honest composite
  `["ProForma_peptide", "Modifications"]`, where `ProForma_peptide` is the stripped `Sequence`.
  That composite also aggregates cleanly to the peptide level. A future computed function could
  express these as ProForma 2.0 unlocalized modifications (`[Oxidation (M)]^2?…`).

- 2026-09-04 (evening): Added a packaged MaxQuant `proteinGroups.txt` rule producing the `protein`
  level: a wide table keyed on `Protein IDs` with one `Intensity <experiment>` column per
  experiment and optional `LFQ intensity`/`iBAQ` layers. The rule admits both generations in hand —
  a 2.x DDA export (Intensity + LFQ intensity, `Reverse`) and a 2.8 DIA export (Intensity only,
  `Decoy` instead of `Reverse`) — by keeping everything beyond identity and the primary layer
  optional. Committed sample artifacts come from the boxcar `proteinGroups.txt`.

- 2026-09-04 (later still): Committed per-rule test artifacts under `tests/parserV2/data/<rule_key>/`
  — a header snapshot, a stratified ~500-row real-export sample (gzip; parquet for parquet rules),
  the fixture's parameter file where one exists, and `expected.json` with per-level conversion
  dimensions. Header- and source-driven tests now fall back to these when no downloaded corpus is
  present, and `test_committed_samples.py` converts every sample end to end against its recorded
  expectations, so a fresh checkout or CI runs the full suite at 1005 passed / 9 skipped instead of
  skipping ~200 corpus-backed tests. The artifacts are append-only and created once by
  `apb_studio/scripts/make_apb2_test_samples.py` and its `extend_directflq_benchmark` twin; the
  downloaded corpus remains the ground truth wherever it exists.

- 2026-09-04 (later): The corpus default moved with the store: `APB2_TEST_DATA` now defaults to
  `../apb_studio/test_data_download`, the Fixture-Manager-owned location. The `vendor_params`
  cached-parameter snapshot test derives its cache index from the same `corpus_root()` instead of a
  hardcoded path that no longer existed, un-skipping its 87 cases wherever the store is present.

- 2026-09-04: `make test` and `make check` now default `APB2_TEST_DATA` to the workspace corpus at
  `../legacy/test_data_download`, so corpus-backed tests run instead of skipping wherever that
  download exists; elsewhere they keep skipping. The test fixture now picks the cached export a
  document's own header predicate admits rather than the first export of the software name: the
  corpus stores every export as `input_file.txt`, so name and version alone cannot separate MaxQuant
  `evidence.txt` from `peptides.txt`, and the peptides document was previously handed an evidence
  export it rightly rejected.

- 2026-09-03: A rule's `input` block can now declare text-encoding detection, mirroring the
  existing `delimiter` and `numbers` blocks: `{"mode": "detect", "candidates": ["utf8",
  "windows-1252"]}`. Candidates are a preference order — the first that decodes a bounded probe of
  the file wins — so a UTF-8 file resolves `utf8` and streams exactly as before, and the fallback is
  consulted only for bytes UTF-8 rejects. A non-UTF-8 resolution reads through polars' eager
  transcoding path, costing that file one extra pass. Rules that declare no encoding block keep the
  fixed `utf8` contract, including its loud failure on invalid bytes. The Spectronaut 15 document
  declares the detection: Spectronaut writes locale-dependent exports, and a real 15.x report puts a
  Windows-1252 em dash in `EG.Label` on every row.

- 2026-09-03: Added a packaged MaxQuant `peptides.txt` rule, the first to produce the `peptide`
  level. `peptides.txt` is a wide table with one `Intensity <experiment>` column per experiment, and
  optionally one `LFQ intensity <experiment>` column, so it is a different quantification level
  rather than a different shape of the same one. Peptide identity is the stripped `Sequence`, which
  is already a valid ProForma string for an unmodified peptide, so the rule needs no modifications
  block. Sample identity is the experiment, not the raw file: `evidence.txt` for the same search can
  therefore carry more observations than `peptides.txt`.

- 2026-09-03: Added a packaged Spectronaut 15 conversion rule. A 15.x report keys runs on `R.Label`
  and carries none of the `E.*`, `PG.GroupLabel`, `PEP.GroupingKey`, `FG.Mass` or `FG.XICDBID`
  columns the 19/20 document requires, so that document rejects it at every level. The new one
  declares ion, fragment and protein: fragment identity comes from `FG.Id` plus `F.FrgIon`,
  `F.Charge` and `F.FrgLossType`, since 15.x has no `F.FrgType`, `F.FrgNum` or `F.TheoreticalMz`.
  Its ion level takes `keep_first` duplicates because a 15.x fragment-level export repeats each
  precursor once per fragment while `FG.Quantity` stays constant across those rows.

  Spectronaut's export template is user-configurable, so the rule requires only the columns that
  carry identity or fill a column role and declares the rest through `optional_select`. The run key
  is a `coalesce` of `R.Label` and `R.FileName`, which are the two run identifiers 15.x templates
  use. Together these let one document read four differently configured 15.x exports.

- 2026-09-03: Added a packaged DIA-NN 1.7 conversion rule for reports that key runs on `File.Name`
  rather than `Run`, which DIA-NN added in 1.8. It declares ion, fragment and protein, takes
  `PG.Normalised` as the protein primary layer since `PG.MaxLFQ` is also 1.8, and otherwise matches
  the DIA-NN 1 document. Its version pattern admits 1.0 through 1.7. A 1.8 report satisfies both
  documents' headers because 1.7's columns are a subset of 1.8's, so a report carrying no version
  banner is ambiguous between them and needs `--rule-config`.

- 2026-09-03: Added storage-neutral annotation tables and feature relations to the APB result. A
  `ParsedLevels` now carries keyed feature tables that have no quantitative matrix, plus directed
  coordinates from such a table to a level's variable axis, and validates both before any writer
  touches its target. h5mu stores an annotation table as an obs-aligned modality and a relation as a
  root `varp` block within the declared modality windows; Parquet and DuckDB round-trip both
  exactly. h5ad rejects them, since a single AnnData has nowhere to put them. `AnnotationTable` and
  `FeatureRelation` are public on the result facade.

- 2026-09-02: Published the complete packaged vendor-conversion and parameter-parser support
  matrix. Documented APB2's origin in the 13–17 April 2026 Copenhagen ProteoBench Hackathon and
  its role in the wider effort to improve the ProteoBench platform backend. Recorded the
  Published a roadmap table for the prolfquapp and prolfquappPTMreaders outputs that motivate the
  planned reader migration.
  Separated the CLI and Python paths through the home page, README, navigation, and API reference.

- 2026-09-01: Aligned local and GitHub Pages documentation builds with the workspace Zensical
  toolchain.

- 2026-09-01: Made APB2 observation annotation convention-neutral and moved ProteoBench module
  interpretation to the separately released `apb-proteobench` package. Added public annotation and
  result facades, a canonical modification-name facade, and independent storage-neutral extension
  metadata. h5ad, h5mu, Parquet, and DuckDB now round-trip downstream `uns["apb"]` sections beside
  parse/result metadata; annotation provenance no longer lives inside parse provenance.

- 2026-08-31: Added storage-neutral, Polars-backed sample annotation with source recognition and
  source-bound prolfquapp/ProteoBench parsers. `parser.parse(parsed)` now validates matching policy
  before constructing a dataset-bound annotation; `annotation.annotate()` reuses inspectable match
  evidence. Added exact aliases, PEAKS rules-driven fuzzy matching, explicit keep/error/drop and
  Boolean selection behavior, consistent layer/obsm/obsp subsetting, provenance across every result
  backend, and the Cyclopts `apb2 annotate` command. Published the user and architecture docs.

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
