"""Physical inputs join once, then ordinary rules decompose the shared result."""

from __future__ import annotations

from io import StringIO
from itertools import combinations
from pathlib import Path

import mudata
import polars as pl
import pytest
from loguru import logger
from polars.testing import assert_frame_equal

from apb2.cli import app
from apb2.parserV2 import prepare_source as preparation_module
from apb2.parserV2.compile import ParquetOutput, compile_parsers
from apb2.parserV2.conversion_facade import (
    ConversionError,
    convert_all_from_rule_config,
    convert_from_rule_config,
)
from apb2.parserV2.detect_document import (
    UNKNOWN_SEARCH_PARAMETERS,
    AmbiguousRuleError,
    detect_rule_documents,
    select_document_levels,
)
from apb2.parserV2.joins import alphadia, maxquant
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels
from apb2.parserV2.parse_quant.parameters.source import (
    Folder,
    InputFiles,
    PreparedTable,
    SingleFile,
)
from apb2.parserV2.prepare_source import InputPreparationError, prepare_source
from apb2.parserV2.vendor_params.parsers.shared.model import Parameters
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema.base import QuantificationLevel
from parserV2.join_fixtures import maxquant_tables

RULES = Path("src/apb2/parserV2/vendor_parse_rules/documents")
MAXQUANT_FILES = {
    "evidence": "evidence.txt",
    "peptidoform": "modificationSpecificPeptides.txt",
    "peptide": "peptides.txt",
    "protein": "proteinGroups.txt",
}
MAXQUANT_LEVELS: dict[str, QuantificationLevel] = {
    "evidence": "ion",
    "peptidoform": "peptidoform",
    "peptide": "peptide",
    "protein": "protein",
}
MAXQUANT_SUBSETS = [subset for size in range(1, 5) for subset in combinations(MAXQUANT_FILES, size)]


def _alphadia() -> dict[str, pl.DataFrame]:
    metadata = pl.DataFrame(
        {
            "mod_seq_charge_hash": ["18446744073709551615", "18446744073709551615", "2"],
            "sequence": ["PEPTIDE", "PEPTIDE", "OTHER"],
            "charge": ["2", "2", "3"],
            "mods": ["", "", ""],
            "mod_sites": ["", "", ""],
            "genes": ["G", "G", "H"],
            "proteins": ["P", "P", "Q"],
            "pg": ["P", "P", "Q"],
            "pg_master": ["P", "P", "Q"],
            "decoy": ["0", "0", "0"],
            "intensity": ["999", "888", "777"],
            "run": ["A", "B", "A"],
        }
    )
    matrix = pl.DataFrame(
        {"mod_seq_charge_hash": ["18446744073709551615"], "A": ["12"], "B": [None]}
    )
    return {"matrix": matrix, "precursors": metadata}


def test_alphadia_uses_matrix_not_secondary_measurements() -> None:
    result = alphadia.join(_alphadia())
    assert result["intensity"].to_list() == ["12", None]
    assert result["mod_seq_charge_hash"].to_list() == ["18446744073709551615"] * 2
    assert result["sequence"].to_list() == ["PEPTIDE"] * 2


def test_alphadia_rejects_conflicting_or_missing_metadata() -> None:
    tables = _alphadia()
    tables["precursors"] = tables["precursors"].with_columns(
        pl.Series("sequence", ["PEPTIDE", "CONFLICT", "OTHER"])
    )
    with pytest.raises(ValueError, match="conflicting"):
        alphadia.join(tables)
    tables = _alphadia()
    tables["matrix"] = tables["matrix"].with_columns(pl.lit("absent").alias("mod_seq_charge_hash"))
    with pytest.raises(ValueError, match="without precursor metadata"):
        alphadia.join(tables)
    with pytest.raises(ValueError, match="companion"):
        alphadia.join({"matrix": tables["matrix"]})


def test_maxquant_higher_join_fanout_preserves_original_cells(tmp_path: Path) -> None:
    tables = maxquant_tables()
    del tables["evidence"]
    original = {name: frame.clone() for name, frame in tables.items()}
    joined = maxquant.join(tables)
    for name in tables:
        assert_frame_equal(tables[name], original[name])
    assert not any(column.startswith("evidence.") for column in joined.columns)
    source = PreparedTable(tmp_path, joined, "maxquant", (), 0.0)
    document = load_rule_document(RULES / "maxquant/rules.json")
    parsed = {
        parser.level: parser.parse()
        for parser in compile_parsers(
            document=document,
            levels=("peptidoform", "peptide", "protein"),
            parameter_evidence=UNKNOWN_SEARCH_PARAMETERS,
            source=source,
            output=ParquetOutput(),
        )
    }
    assert set(parsed) == {"peptidoform", "peptide", "protein"}
    protein = parsed["protein"]
    assert protein.obs.frame["Experiment"].to_list() == ["A", "B"]
    assert protein.layers["Intensity"].values.rows() == [
        ("P", "100", "101"),
        ("Q", "200", None),
        ("UNMATCHED", "300", "301"),
    ]
    assert parsed["peptide"].layers["Intensity"].values.row(0)[1:] == ("25",)


def test_maxquant_join_does_not_accept_evidence() -> None:
    tables = maxquant_tables()
    with pytest.raises(ValueError, match="evidence is parsed directly"):
        maxquant.join(tables)


@pytest.mark.parametrize("suffix", [".parquet", ".h5mu", ".duckdb"])
def test_folder_conversion_joins_all_levels_and_round_trips(tmp_path: Path, suffix: str) -> None:
    names = {
        "evidence": "evidence.txt",
        "peptide": "peptides.txt",
        "peptidoform": "modificationSpecificPeptides.txt",
        "protein": "proteinGroups.txt",
    }
    for role, frame in maxquant_tables().items():
        frame.write_csv(tmp_path / names[role], separator="\t")
    result = convert_all_from_rule_config(
        data=tmp_path,
        output=tmp_path / f"output{suffix}",
        rule_config=RULES / "maxquant/rules.json",
        parameters_path=None,
        parameters_software=None,
        checks="standard",
    )
    assert len(result.levels) == 4
    assert result.outputs == (
        tmp_path / f"output.raw_file{suffix}",
        tmp_path / f"output.experiment{suffix}",
    )
    assert not (tmp_path / f"output{suffix}").exists()
    ion = read_parsed_levels(result.outputs[0]).levels["ion"]
    assert "input_preparation" not in ion.uns
    loaded = read_parsed_levels(result.outputs[1])
    sources = [level.uns["input_preparation"] for level in loaded.levels.values()]
    assert all(item == sources[0] for item in sources)
    provenance = sources[0]
    assert isinstance(provenance, dict)
    paths = provenance["sources"]
    assert isinstance(paths, list)
    assert len(paths) == 3
    assert all("evidence.txt" not in str(path) for path in paths)


def test_renamed_alphadia_companions_and_shared_preparation(tmp_path: Path) -> None:
    paths = {}
    for index, (role, frame) in enumerate(_alphadia().items()):
        paths[role] = tmp_path / f"renamed_{index}.tsv"
        frame.write_csv(paths[role], separator="\t")
    document = load_rule_document(RULES / "alphadia/v1_12/rules.json")
    selected = select_document_levels(
        document, InputFiles(tmp_path, paths), document.levels, UNKNOWN_SEARCH_PARAMETERS
    )
    assert len(selected) == 1
    prepared = selected[0].source
    assert isinstance(prepared, PreparedTable)
    assert prepare_source(prepared, "alphadia") is prepared
    assert prepared.frame["intensity"].to_list() == ["12", None]
    with pytest.raises(InputPreparationError, match="rule requests"):
        prepare_source(prepared, "maxquant")


def test_conflicting_renamed_inputs_are_rejected(tmp_path: Path) -> None:
    for index in range(2):
        maxquant_tables()["evidence"].write_csv(tmp_path / f"renamed_{index}.txt", separator="\t")
    document = load_rule_document(RULES / "maxquant/rules.json")
    with pytest.raises(AmbiguousRuleError, match="multiple inputs"):
        select_document_levels(
            document, Folder(tmp_path), document.levels, UNKNOWN_SEARCH_PARAMETERS
        )


@pytest.mark.parametrize("roles", MAXQUANT_SUBSETS, ids="+".join)
@pytest.mark.parametrize("suffix", [".parquet", ".h5mu", ".duckdb"])
def test_maxquant_every_nonempty_subset_round_trips_available_levels(
    tmp_path: Path, roles: tuple[str, ...], suffix: str
) -> None:
    tables = maxquant_tables()
    for role in roles:
        tables[role].write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")
    target = tmp_path / f"subset{suffix}"
    expected_levels = [MAXQUANT_LEVELS[role] for role in roles]
    detected = detect_rule_documents(
        Parameters(software_name="MaxQuant", software_version="2.6"),
        Folder(tmp_path),
        tuple(MAXQUANT_LEVELS.values()),
    )
    assert [item.level for item in detected.levels] == expected_levels
    prepared = [item.source for item in detected.levels if item.level != "ion"]
    assert all(source is prepared[0] for source in prepared)
    if "evidence" in roles:
        assert isinstance(detected.levels[0].source, SingleFile)

    summary = convert_all_from_rule_config(
        data=tmp_path,
        output=target,
        rule_config=RULES / "maxquant/rules.json",
        parameters_path=None,
        parameters_software=None,
        checks="standard",
    )
    assert [item.level for item in summary.levels] == expected_levels
    parsed = {
        name: level
        for path in summary.outputs
        for name, level in read_parsed_levels(path).levels.items()
    }
    assert list(parsed) == expected_levels
    assert len(summary.outputs) == (2 if "evidence" in roles and len(roles) > 1 else 1)
    expected_values = {
        "ion": [[20.0, 5.0]],
        "peptidoform": [[25.0]],
        "peptide": [[25.0]],
        "protein": [[100.0, 101.0], [200.0, None], [300.0, 301.0]],
    }
    for name, level in parsed.items():
        # HDF5 represents missing numeric cells as NaN; table backends retain nulls.
        values = (
            level.layers["Intensity"]
            .values.drop(level.var.key_columns)
            .cast(pl.Float64)
            .fill_nan(None)
        )
        assert values.rows() == [tuple(row) for row in expected_values[name]]
        obs_key = "Raw_File" if name == "ion" else "Experiment"
        expected_samples = {"ion": ["raw1", "raw2"], "protein": ["A", "B"]}.get(name, ["A"])
        assert level.obs.frame[obs_key].to_list() == expected_samples
        if name == "ion":
            assert "input_preparation" not in level.uns
            continue
        preparation = level.uns["input_preparation"]
        assert isinstance(preparation, dict)
        paths = preparation["sources"]
        assert isinstance(paths, list)
        assert set(paths) == {
            str(tmp_path / MAXQUANT_FILES[role]) for role in roles if role != "evidence"
        }
    if "protein" in parsed:
        protein = parsed["protein"]
        lfq = (
            protein.layers["LFQ_Intensity"]
            .values.drop(protein.var.key_columns)
            .cast(pl.Float64)
            .fill_nan(None)
        )
        assert lfq.rows() == [(110.0, 111.0), (210.0, None), (310.0, 311.0)]


@pytest.mark.parametrize("roles", MAXQUANT_SUBSETS, ids="+".join)
def test_maxquant_renamed_subsets_select_only_present_levels(
    tmp_path: Path, roles: tuple[str, ...]
) -> None:
    paths = {}
    tables = maxquant_tables()
    for index, role in enumerate(roles):
        path = tmp_path / f"input_{index}.txt"
        tables[role].write_csv(path, separator="\t")
        paths[path.name] = path
    document = load_rule_document(RULES / "maxquant/rules.json")
    selected = select_document_levels(
        document, InputFiles(tmp_path, paths), document.levels, UNKNOWN_SEARCH_PARAMETERS
    )
    assert [item.level for item in selected] == [MAXQUANT_LEVELS[role] for role in roles]
    prepared = [item.source for item in selected if item.level != "ion"]
    assert all(source is prepared[0] for source in prepared)
    if "evidence" in roles:
        assert isinstance(selected[0].source, SingleFile)


def test_maxquant_higher_tables_join_by_both_evidence_reference_and_sample() -> None:
    tables = maxquant_tables()
    peptide = tables["peptide"].rename({"Intensity A": "Intensity C"})
    joined = maxquant.join({"peptide": peptide, "protein": tables["protein"]})
    assert joined.filter(
        pl.col("peptide.id").is_not_null() & pl.col("protein.id").is_not_null()
    ).is_empty()
    assert set(joined["peptide.sample"].drop_nulls()) == {"C"}
    assert set(joined["protein.sample"].drop_nulls()) == {"A", "B"}
    assert "evidence.Intensity" not in joined.columns

    joined = maxquant.join({"peptide": tables["peptide"], "protein": tables["protein"]})
    paired = joined.filter(pl.col("peptide.id").is_not_null() & pl.col("protein.id").is_not_null())
    assert paired.height == 2
    assert paired["protein.sample"].unique().to_list() == ["A"]
    assert set(paired["protein.id"]) == {"0", "1"}


def test_maxquant_repeated_evidence_references_do_not_expand_measurement_rows() -> None:
    tables = maxquant_tables()
    del tables["evidence"]
    for role, frame in tables.items():
        tables[role] = frame.with_columns(
            pl.lit(";".join(map(str, range(1000)))).alias("Evidence IDs")
        )
    joined = maxquant.join(tables)
    assert joined.height == 6  # Three protein links times two experiments, not 6,000 rows.
    assert joined["protein.Intensity"].to_list() == ["100", "101", "200", None, "300", "301"]


def test_maxquant_absent_requested_level_fails_without_writing(tmp_path: Path) -> None:
    for role, table in maxquant_tables().items():
        if role != "evidence":
            table.write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")
    target = tmp_path / "missing.parquet"
    with pytest.raises(ConversionError, match="ion"):
        convert_from_rule_config(
            data=tmp_path,
            level="ion",
            output=target,
            rule_config=RULES / "maxquant/rules.json",
            parameters_path=None,
            parameters_software=None,
            checks="standard",
        )
    assert not target.exists()


def test_maxquant_empty_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        maxquant.join({})


@pytest.mark.parametrize("suffix", [".parquet", ".h5mu", ".duckdb"])
def test_maxquant_one_to_one_experiments_align_to_raw_files(tmp_path: Path, suffix: str) -> None:
    tables = maxquant_tables()
    tables["evidence"] = tables["evidence"].with_columns(pl.Series("Experiment", ["B", "B", "A"]))
    for role, frame in tables.items():
        frame.write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")
    target = tmp_path / f"aligned{suffix}"
    summary = convert_all_from_rule_config(
        data=tmp_path,
        output=target,
        rule_config=RULES / "maxquant/rules.json",
        parameters_path=None,
        parameters_software=None,
        checks="standard",
    )
    assert summary.outputs == (target,)
    result = read_parsed_levels(target)
    assert all(level.obs.key_columns == ("Raw_File",) for level in result.levels.values())
    protein = result.levels["protein"]
    assert protein.obs.frame["Raw_File"].to_list() == ["raw2", "raw1"]
    assert protein.obs.frame["Experiment"].to_list() == ["A", "B"]
    values = (
        protein.layers["Intensity"]
        .values.drop(protein.var.key_columns)
        .cast(pl.Float64)
        .fill_nan(None)
    )
    assert values.rows() == [(100.0, 101.0), (200.0, None), (300.0, 301.0)]
    ion = result.levels["ion"]
    assert ion.layers["Intensity"].values.row(0)[1:] == (20.0, 5.0)
    if suffix == ".h5mu":
        stored = mudata.read_h5mu(target)
        assert stored.n_obs == 2
        assert set(stored.obs_names) == {"raw1", "raw2"}


def test_ion_only_never_reads_or_joins_higher_table_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for role, frame in maxquant_tables().items():
        frame.write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")

    def unexpected_read(*args: object, **kwargs: object) -> pl.DataFrame:
        pytest.fail("ion selection must not read rows for preparation")

    monkeypatch.setattr(preparation_module.pl, "read_csv", unexpected_read)
    document = load_rule_document(RULES / "maxquant/rules.json")
    selected = select_document_levels(
        document, Folder(tmp_path), ("ion",), UNKNOWN_SEARCH_PARAMETERS
    )
    assert len(selected) == 1
    assert isinstance(selected[0].source, SingleFile)


def test_higher_group_reads_each_input_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for role, frame in maxquant_tables().items():
        frame.write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")
    original_read = preparation_module.pl.read_csv
    calls: list[Path] = []

    def counted_read(path: Path, **kwargs: object) -> pl.DataFrame:
        calls.append(path)
        return original_read(path, separator="\t", infer_schema=False, null_values=[""])

    monkeypatch.setattr(preparation_module.pl, "read_csv", counted_read)
    document = load_rule_document(RULES / "maxquant/rules.json")
    select_document_levels(document, Folder(tmp_path), document.levels, UNKNOWN_SEARCH_PARAMETERS)
    assert sorted(calls) == sorted(
        tmp_path / filename for role, filename in MAXQUANT_FILES.items() if role != "evidence"
    )


def test_unknown_explicit_companion_is_not_silently_ignored(tmp_path: Path) -> None:
    evidence = tmp_path / "input.txt"
    maxquant_tables()["evidence"].write_csv(evidence, separator="\t")
    unknown = tmp_path / "unknown.txt"
    unknown.write_text("unrecognized\n1\n")
    document = load_rule_document(RULES / "maxquant/rules.json")
    with pytest.raises(InputPreparationError, match="unrecognized explicit companion"):
        select_document_levels(
            document,
            InputFiles(tmp_path, {path.name: path for path in (evidence, unknown)}),
            document.levels,
            UNKNOWN_SEARCH_PARAMETERS,
        )


def test_cli_reports_both_resolution_outputs(tmp_path: Path) -> None:
    for role, frame in maxquant_tables().items():
        frame.write_csv(tmp_path / MAXQUANT_FILES[role], separator="\t")
    captured = StringIO()
    sink = logger.add(captured, format="{message}")
    with pytest.raises(SystemExit) as result:
        app(
            [
                "convert",
                str(tmp_path),
                "--rule-config",
                str(RULES / "maxquant/rules.json"),
                "--format",
                "parquet",
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert result.value.code == 0
    logger.remove(sink)
    assert "output.raw_file.parquet" in captured.getvalue()
    assert "output.experiment.parquet" in captured.getvalue()
    assert not (tmp_path / "output.parquet").exists()


def test_cli_directory_joins_before_ion_conversion(tmp_path: Path) -> None:
    for index, frame in enumerate(_alphadia().values()):
        frame.write_csv(tmp_path / f"input_{index}.tsv", separator="\t")
    with pytest.raises(SystemExit) as result:
        app(
            [
                "convert",
                str(tmp_path),
                "ion",
                "--rule-config",
                str(RULES / "alphadia/v1_12/rules.json"),
                "--format",
                "parquet",
                "--output",
                str(tmp_path / "output"),
            ]
        )
    assert result.value.code == 0
    parsed = read_parsed_levels(tmp_path / "output.parquet").levels["ion"]
    assert parsed.layers["Intensity"].values.row(0)[1:] == ("12", None)
