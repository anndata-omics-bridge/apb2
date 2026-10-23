"""Coherent related vendor tables: MaxQuant's, with fan-out and unmatched measurements, and
Proline workbooks, whose ion sheet lists a shared peptide once per protein set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import polars as pl

type Cells = Mapping[str, Sequence[str | float | bool | None]]
"""One sheet, column by column, in header order; ``None`` leaves a cell empty."""

PROLINE_TEMPLATE = Path(__file__).parent / "data" / "prolinestudio" / "sample.txt"
"""A Proline workbook: four parameter sheets, then "Quantified peptide ions"."""


def maxquant_tables() -> dict[str, pl.DataFrame]:
    evidence = pl.DataFrame(
        {
            "id": ["0", "1", "2"],
            "Raw file": ["raw1", "raw1", "raw2"],
            "Experiment": ["A", "A", "A"],
            "Modified sequence": ["_PEPTIDE_"] * 3,
            "Sequence": ["PEPTIDE"] * 3,
            "Modifications": ["Unmodified"] * 3,
            "Charge": ["2"] * 3,
            "Mass": ["800"] * 3,
            "Proteins": ["P;Q"] * 3,
            "Leading razor protein": ["P"] * 3,
            "Intensity": ["7", "13", "5"],
            "PEP": ["0.01", "0.002", "0.05"],
            "Protein group IDs": ["0;1"] * 3,
            "Peptide ID": ["0"] * 3,
            "Mod. peptide ID": ["0"] * 3,
        }
    )
    return {
        "evidence": evidence,
        "peptidoform": pl.DataFrame(
            {
                "id": ["0"],
                "Sequence": ["PEPTIDE"],
                "Modifications": ["Unmodified"],
                "Proteins": ["P;Q"],
                "Evidence IDs": ["0;1;2"],
                "Intensity A": ["25"],
            }
        ),
        "peptide": pl.DataFrame(
            {
                "id": ["0"],
                "Sequence": ["PEPTIDE"],
                "Proteins": ["P;Q"],
                "Leading razor protein": ["P"],
                "Mod. peptide IDs": ["0"],
                "Evidence IDs": ["0;1;2"],
                "Intensity A": ["25"],
            }
        ),
        "protein": pl.DataFrame(
            {
                "id": ["0", "1", "2"],
                "Protein IDs": ["P", "Q", "UNMATCHED"],
                "Peptide IDs": ["0", "0", None],
                "Evidence IDs": ["0;1;2", "0;1;2", None],
                "Intensity A": ["100", "200", "300"],
                "Intensity B": ["101", None, "301"],
                "LFQ intensity A": ["110", "210", "310"],
                "LFQ intensity B": ["111", None, "311"],
            }
        ),
    }


def proline_workbook(path: Path, ions: Cells, protein_sets: Cells | None = None) -> Path:
    """Write a Proline workbook at ``path`` from the template, with these sheets' cells.

    ``ions`` becomes "Quantified peptide ions". ``protein_sets``, when given, takes the place of
    the template's fourth sheet as "Protein sets", the sheet MSAngel exports beside the ions.
    The template is read before ``path`` is written, so ``path`` may be the template itself.
    """
    with ZipFile(PROLINE_TEMPLATE) as template:
        entries = [(entry, template.read(entry)) for entry in template.infolist()]
    replaced = {"xl/worksheets/sheet5.xml": _sheet(ions)}
    if protein_sets is not None:
        replaced["xl/worksheets/sheet4.xml"] = _sheet(protein_sets)
        workbook_xml = next(
            content for entry, content in entries if entry.filename == "xl/workbook.xml"
        )
        replaced["xl/workbook.xml"] = workbook_xml.decode().replace(
            'name="Dataset statistics and infos"', 'name="Protein sets"'
        )
    with ZipFile(path, "w") as workbook:
        for entry, content in entries:
            workbook.writestr(entry, replaced.get(entry.filename, content))
    return path


def msangel_sheets() -> tuple[Cells, Cells]:
    """MSAngel's ion and protein-set sheets: PEPTIDE is shared, its weaker set listed first."""
    ions: Cells = {
        "sequence": ["ACDMK", "PEPTIDE", "PEPTIDE"],
        "modifications": ["Carbamidomethyl (C2); Oxidation (M4)", None, None],
        "master_quant_peptide_ion_charge": [2, 3, 3],
        "protein_set_id": [1, 4, 2],
        "accession": ["sp|P1|ONE", "sp|P4|FOUR", "sp|P2|TWO"],
        "samesets_accessions": ["sp|P1|ONE", "sp|P4|FOUR", "sp|P2|TWO"],
        "subsets_accessions": [None, None, "sp|P3|THREE"],
        "protein_set_score": [120.5, 30.0, 80.25],
        **_ion_values(),
        "raw_abundance_sample_A": [90, 190, 190],
        "raw_abundance_sample_B": [95, 195, 195],
    }
    protein_sets: Cells = {
        "protein_set_id": [1, 2, 4],
        "accession": ["sp|P1|ONE", "sp|P2|TWO", "sp|P4|FOUR"],
        "gene_name": ["ONE", "TWO", "FOUR"],
        "samesets_accessions": ["sp|P1|ONE", "sp|P2|TWO", "sp|P4|FOUR"],
        "subsets_accessions": [None, "sp|P3|THREE; sp|P5|FIVE", None],
        "description": ["protein one", "protein two", "protein four"],
        "protein_set_score": [120.5, 80.25, 30.0],
        "is_validated": ["true", "true", "true"],
        "peptides_count_sample_A": [1, 1, 1],
        "peptides_count_sample_B": [1, 2, 1],
        "psm_count_sample_A": [2, 1, 1],
        "psm_count_sample_B": [3, 2, 1],
        "raw_abundance_sample_A": [90, 290, 190],
        "raw_abundance_sample_B": [95, 295, 195],
        "abundance_sample_A": [100, 300, 200],
        "abundance_sample_B": [110, 310, 210],
    }
    return ions, protein_sets


def prolinestudio_ions() -> Cells:
    """ProlineStudio's ion sheet: PEPTIDE is shared and quantified under its weaker set."""
    return {
        "sequence": ["ACDMK", "PEPTIDE", "PEPTIDE"],
        "modifications": ["Carbamidomethyl (C2); Oxidation (M4)", None, None],
        "master_quant_peptide_ion_charge": [2, 3, 3],
        "samesets_accessions": ["sp|P1|ONE", "sp|P4|FOUR", "sp|P2|TWO"],
        "subsets_accessions": [None, None, "sp|P3|THREE"],
        "protein_set_score": [120.5, 30.0, 80.25],
        **_ion_values(),
        "quant_is_valid_for_protein_set": [True, True, False],
    }


def _ion_values() -> Cells:
    """Per-sample values of ACDMK and the two rows of the shared PEPTIDE, which repeat them."""
    return {
        "best_score_sample_A": [50, 40, 40],
        "best_score_sample_B": [51, 41, 41],
        "psm_count_sample_A": [2, 1, 1],
        "psm_count_sample_B": [3, 1, 1],
        "elution_time_sample_A": [10, 20, 20],
        "elution_time_sample_B": [11, 21, 21],
        "corrected_time_sample_A": [10.5, 20.5, 20.5],
        "corrected_time_sample_B": [11.5, 21.5, 21.5],
        "abundance_sample_A": [100, 200, 200],
        "abundance_sample_B": [110, 210, 210],
    }


def _sheet(cells: Cells) -> str:
    names = list(cells)
    height = len(cells[names[0]])
    rows = [names, *([cells[name][index] for name in names] for index in range(height))]
    body = "".join(
        f'<row r="{number}">'
        + "".join(_cell(f"{_letters(column)}{number}", value) for column, value in enumerate(row))
        + "</row>"
        for number, row in enumerate(rows, start=1)
    )
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    return f'<worksheet xmlns="{namespace}"><sheetData>{body}</sheetData></worksheet>'


def _cell(reference: str, value: str | float | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, int | float):
        return f'<c r="{reference}" t="n"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'


def _letters(index: int) -> str:
    """The spreadsheet column name of a zero-based index: A, ..., Z, AA, ..."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters
