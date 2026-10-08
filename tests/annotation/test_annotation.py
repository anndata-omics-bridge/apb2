"""Dataset-bound sample-annotation API, policies, matching, and CLI workflow."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal

import polars as pl
import pytest
from loguru import logger
from polars.testing import assert_frame_equal

from apb2.annotation.compiler import AnnotationCompiler
from apb2.annotation.data.model import (
    IN_MEMORY_ANNOTATION,
    AnnotationError,
    LoadedAnnotationSource,
)
from apb2.annotation.matching.core import (
    ExactAnnotationMatching,
    FuzzyAnnotationMatching,
    annotation_matching_for,
    make_annotation_table,
    match_annotation,
    normalize_mass_spec_basename,
)
from apb2.annotation.sdrf import SdrfSource
from apb2.annotation.source.load import load_annotation_file
from apb2.cli.app import annotate as annotate_command
from apb2.parserV2.parse_quant.data.parsed import (
    AuxiliaryLayerRole,
    CategoricalLayerSemantics,
    FinalLayerSemantics,
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    QuantitativeLayerSemantics,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels


def _plan() -> str:
    return json.dumps(
        {
            "canonicalization": {
                "layer_values": [
                    {
                        "kind": "plain_numeric",
                        "layer_name": "Intensity",
                        "missing_values": [],
                        "number_format": {"decimal_mark": ".", "thousands_marks": []},
                    }
                ],
                "layer_contract": {
                    "primary_layer_name": "Intensity",
                    "required_names": ["Intensity"],
                    "empty_ratio": 0.001,
                    "populated_ratio": 0.5,
                },
            }
        }
    )


def _parsed(
    runs: tuple[str, ...] = ("run_A", "run_B", "run_C"),
    *,
    matching: dict[str, JsonValue] | None = None,
) -> ParsedLevels:
    pair_rows = [index for index in range(len(runs) - 1) for _ in range(2)]
    pair_columns = [neighbor for index in range(len(runs) - 1) for neighbor in (index + 1, index)]
    pair_rows[1::2] = range(1, len(runs))
    layer = FinalLayerTable(
        layer_name="Intensity",
        values=(
            pl.DataFrame(
                {
                    "feature": ["p1", "p2"],
                    **{
                        f"obs_{index}": [float(index + 1), float(index + 11)]
                        for index in range(len(runs))
                    },
                }
            )
        ).drop(("feature",), strict=False),
        semantic_roles=("abundance",),
    )
    provenance: dict[str, JsonValue] = {"plan_json": _plan()}
    if matching is not None:
        provenance["sample_annotation_matching"] = matching
    uns: dict[str, JsonValue] = {"provenance": provenance}
    level = ParsedLevel(
        obs=ObsFinal(frame=pl.DataFrame({"run": runs}), key_columns=("run",)),
        var=VarFinal(frame=pl.DataFrame({"feature": ["p1", "p2"]}), key_columns=("feature",)),
        primary_layer_name="Intensity",
        uns=uns,
        layers={"Intensity": layer},
        obsm={"quality": pl.DataFrame({"score": list(range(len(runs)))})},
        varm={},
        obsp={
            "links": pl.DataFrame(
                {
                    "row": pair_rows,
                    "column": pair_columns,
                    "value": [0.5] * len(pair_rows),
                }
            )
        },
        varp={},
    )
    return ParsedLevels(levels={"ion": level}, uns={"produced_by": "apb2"})


def test_parser_constructs_a_dataset_bound_annotation_with_inspectable_matches() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "run_B", "unused"],
            "condition": ["A", "B", "X"],
        }
    )
    parsed = _parsed()

    annotation = AnnotationCompiler().compile(source).parse(parsed)

    coverage = annotation.matches.levels["ion"].coverage
    assert coverage.matched_observation_count == 2
    assert coverage.quant_only_examples == ("run_C",)
    assert coverage.annotation_only_examples == ("unused",)
    result = annotation.annotate()
    assert result.parsed.levels["ion"].obs.frame.to_dict(as_series=False) == {
        "run": ["run_A", "run_B", "run_C"],
        "condition": ["A", "B", None],
    }
    assert parsed.levels["ion"].obs.frame.columns == ["run"]
    root = result.parsed.metadata["prolfquapp"]
    assert isinstance(root, dict) and root["schema_version"] == "3"
    local = result.parsed.levels["ion"].metadata["prolfquapp"]
    assert isinstance(local, dict)
    summary = local["summary"]
    assert isinstance(summary, list)
    assert [
        (entry["name"], entry["value"], entry["status"])
        for entry in summary
        if isinstance(entry, dict)
    ] == [
        ("annotation_only_rows", 1, "attention"),
        ("quantification_only_samples", 1, "attention"),
        ("fuzzy_corrections", 0, "ok"),
    ]


def test_generic_compiler_rejects_source_specific_toml(tmp_path: Path) -> None:
    source = tmp_path / "module_settings.toml"
    source.write_text(
        """
[[samples]]
raw_file = "run_A"
sample_name = "A"
condition = "A"
""",
        encoding="utf-8",
    )
    with pytest.raises(AnnotationError, match=r"expected \.csv or \.tsv"):
        AnnotationCompiler().compile(source)


def test_prolfquapp_parser_does_not_construct_an_annotation_with_zero_matches() -> None:
    source = pl.DataFrame({"raw_file": ["elsewhere"], "condition": ["A"]})

    with pytest.raises(AnnotationError, match="matched no observations"):
        AnnotationCompiler().compile(source).parse(_parsed(("run_A",)))


@pytest.mark.parametrize("suffix", [".csv", ".tsv"])
def test_compiler_loads_delimited_prolfquapp_sources(
    suffix: str,
    tmp_path: Path,
) -> None:
    delimiter = "," if suffix == ".csv" else "\t"
    source = tmp_path / f"samples{suffix}"
    source.write_text(
        f"raw_file{delimiter}condition\nrun_A{delimiter}A\n",
        encoding="utf-8",
    )

    parser = AnnotationCompiler().compile(source)

    annotation = parser.parse(_parsed(("run_A",)))
    assert annotation.matches.levels["ion"].coverage.matched_observation_count == 1


def test_memory_annotation_requires_one_supported_observation_key() -> None:
    with pytest.raises(AnnotationError, match="no supported observation key"):
        AnnotationCompiler().compile(pl.DataFrame({"other": ["value"]}))


def test_compilation_reads_a_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "samples.tsv"
    source.write_text("raw_file\tcondition\nrun_A\tA\n", encoding="utf-8")
    from apb2.annotation import compiler as compiler_module

    original = compiler_module.load_annotation_file
    calls = 0

    def counted(path: Path) -> LoadedAnnotationSource:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(compiler_module, "load_annotation_file", counted)

    parser = AnnotationCompiler().compile(source)
    parser.parse(_parsed(("run_A",)))
    parser.parse(_parsed(("run_A",)))

    assert calls == 1


def test_drop_and_boolean_selection_subsets_every_observation_aligned_value() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "run_B", "run_C"],
            "include": [True, False, True],
        }
    )
    annotation = AnnotationCompiler("drop", "include").compile(source).parse(_parsed())

    result = annotation.annotate().parsed.levels["ion"]

    assert result.obs.frame.to_dict(as_series=False) == {
        "run": ["run_A", "run_C"],
        "include": [True, True],
    }
    assert result.layers["Intensity"].values.columns == ["obs_0", "obs_1"]
    assert result.layers["Intensity"].values.get_column("obs_1").to_list() == [3.0, 13.0]
    assert result.obsm["quality"].get_column("score").to_list() == [0, 2]
    assert result.obsp["links"].is_empty()


def test_boolean_selection_rejects_null_for_a_matched_annotation() -> None:
    source = pl.DataFrame({"raw_file": ["run_A"], "include": [None]})

    with pytest.raises(AnnotationError, match="must be Boolean"):
        AnnotationCompiler("drop", "include").compile(source).parse(_parsed(("run_A",)))


@pytest.mark.parametrize(
    ("unmatched", "kept"),
    [("keep", (0, 1, 2)), ("drop", (0, 2))],
    ids=["retain-all", "select-subset"],
)
@pytest.mark.parametrize(
    "semantics",
    [
        CategoricalLayerSemantics(categories=(("MS/MS", 1), ("MBR", 2)), missing_code=-1),
        QuantitativeLayerSemantics(logical_type="integer"),
    ],
    ids=["categorical", "integer"],
)
def test_annotation_preserves_layer_semantics_and_round_trips(
    unmatched: Literal["keep", "drop"],
    kept: tuple[int, ...],
    semantics: FinalLayerSemantics,
    tmp_path: Path,
) -> None:
    parsed = _parsed()
    layer = FinalLayerTable(
        layer_name="Evidence",
        values=(
            pl.DataFrame(
                {"feature": ["p1", "p2"], "obs_0": [1, -1], "obs_1": [2, 1], "obs_2": [-1, 2]}
            )
        ).drop(("feature",), strict=False),
        role=AuxiliaryLayerRole(),
        semantics=semantics,
    )
    parsed.levels["ion"].layers["Evidence"] = layer
    source = pl.DataFrame({"raw_file": ["run_C", "run_A"], "condition": ["C", "A"]})
    result = AnnotationCompiler(unmatched).compile(source).parse(parsed).annotate().parsed
    annotated = result.levels["ion"].layers["Evidence"]
    assert annotated.semantics is semantics
    assert annotated.role is layer.role
    target = tmp_path / "annotated.h5ad"
    write_parsed_levels(result, target)
    restored = read_parsed_levels(target)
    expected = layer.values.select(
        *(pl.col(f"obs_{old}").alias(f"obs_{new}") for new, old in enumerate(kept))
    )
    for level in (result.levels["ion"], restored.levels["ion"]):
        copied = level.layers["Evidence"]
        assert copied.semantics == semantics
        assert copied.layer_name == layer.layer_name
        assert copied.semantic_roles == layer.semantic_roles
        assert isinstance(copied.role, AuxiliaryLayerRole)
        assert_frame_equal(copied.values, expected)
        assert level.obs.frame["run"].to_list() == [["run_A", "run_B", "run_C"][i] for i in kept]
    assert layer.values.columns == ["obs_0", "obs_1", "obs_2"]


def test_exact_matching_supports_composite_keys() -> None:
    parsed = _parsed(("unused",))
    level = parsed.levels["ion"]
    level.obs = ObsFinal(
        frame=pl.DataFrame({"run": ["A", "B"], "channel": [1, 2]}),
        key_columns=("run", "channel"),
    )
    level.layers["Intensity"].values = pl.DataFrame(
        {"feature": ["p1", "p2"], "obs_0": [1.0, 2.0], "obs_1": [3.0, 4.0]}
    )
    level.obsm = {}
    level.obsp = {}
    table = make_annotation_table(
        pl.DataFrame({"raw": ["A", "B"], "channel": [1, 2], "condition": ["x", "y"]}),
        ("raw", "channel"),
        (),
        IN_MEMORY_ANNOTATION,
    )

    matches = match_annotation(table, parsed, {"ion": ExactAnnotationMatching()})

    assert matches.levels["ion"].matched_rows.to_list() == [True, True]
    assert matches.levels["ion"].aligned.get_column("condition").to_list() == ["x", "y"]


def test_fuzzy_matching_reserves_exact_pairs_and_records_corrections() -> None:
    parsed = _parsed(("sample-alpha.raw", "sample-beta"))
    table = make_annotation_table(
        pl.DataFrame(
            {
                "raw_file": ["sample-alpha", "sample-beta"],
                "condition": ["A", "B"],
            }
        ),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    matches = match_annotation(
        table,
        parsed,
        {"ion": FuzzyAnnotationMatching(cutoff=0.6, margin=0.1, near_miss_limit=3)},
    )

    match = matches.levels["ion"]
    assert match.matched_rows.to_list() == [True, True]
    assert [(item.observed, item.expected) for item in match.corrections] == [
        ("sample-alpha.raw", "sample-alpha")
    ]


def test_exact_aliases_match_without_fuzzy_correction() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["canonical"],
            "raw_file_aliases": [["alias_A", "alias_B"]],
            "condition": ["A"],
        }
    )

    annotation = AnnotationCompiler().compile(source).parse(_parsed(("alias_B",)))

    match = annotation.matches.levels["ion"]
    assert match.matched_rows.to_list() == [True]
    assert match.corrections == ()


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ("/data/run_A.mzML", "run_A"),
        (r"C:\data\run_A.MZML.GZ", "run_A"),
        ("/data/run_A.RAW", "run_A"),
        ("/data/run_A.mgf", "run_A"),
        ("/data/run_A.d", "run_A"),
        ("/data/run_A.WIFF", "run_A"),
        ("/data/run_A", "run_A"),
    ],
)
def test_mass_spec_basename_normalization_matches_without_changing_evidence(
    observed: str,
    expected: str,
) -> None:
    table = make_annotation_table(
        pl.DataFrame({"raw_file": [expected], "condition": ["A"]}),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    matches = match_annotation(
        table,
        _parsed((observed,)),
        {
            "ion": ExactAnnotationMatching(
                normalization=normalize_mass_spec_basename,
            )
        },
    )

    match = matches.levels["ion"]
    assert match.matched_rows.to_list() == [True]
    assert match.corrections == ()
    assert match.coverage.quant_only_examples == ()


def test_mass_spec_basename_normalization_rejects_observation_collisions() -> None:
    table = make_annotation_table(
        pl.DataFrame({"raw_file": ["run_A"], "condition": ["A"]}),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    with pytest.raises(AnnotationError, match="observation normalization collision"):
        match_annotation(
            table,
            _parsed(("/first/run_A.raw", "/second/run_A.mzML")),
            {
                "ion": ExactAnnotationMatching(
                    normalization=normalize_mass_spec_basename,
                )
            },
        )


def test_mass_spec_basename_normalization_rejects_annotation_collisions() -> None:
    table = make_annotation_table(
        pl.DataFrame(
            {
                "raw_file": ["/first/run_A.raw", "/second/run_A.mzML"],
                "condition": ["A", "B"],
            }
        ),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    with pytest.raises(AnnotationError, match="annotation normalization collision"):
        match_annotation(
            table,
            _parsed(("run_A",)),
            {
                "ion": ExactAnnotationMatching(
                    normalization=normalize_mass_spec_basename,
                )
            },
        )


def test_fuzzy_mass_spec_matching_preserves_original_correction_labels() -> None:
    observed = "LFQ_timstofSCP_diaPASEF_Condition_A_Sample_Alpha_01.mzML"
    expected = "LFQ_ttSCP_diaPASEF_Condition_A_Sample_Alpha_01"
    table = make_annotation_table(
        pl.DataFrame(
            {
                "raw_file": [expected, expected.replace("_01", "_02")],
                "condition": ["A", "B"],
            }
        ),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    matches = match_annotation(
        table,
        _parsed((observed,)),
        {
            "ion": FuzzyAnnotationMatching(
                cutoff=0.6,
                margin=0.01,
                near_miss_limit=3,
                normalization=normalize_mass_spec_basename,
            )
        },
    )

    match = matches.levels["ion"]
    assert match.matched_rows.to_list() == [True]
    assert [(item.observed, item.expected) for item in match.corrections] == [(observed, expected)]


def test_persisted_exact_matching_constructs_mass_spec_normalization() -> None:
    parsed = _parsed(
        ("/data/run_A.mzML",),
        matching={"mode": "exact", "normalize": "mass_spec_basename"},
    )
    table = make_annotation_table(
        pl.DataFrame({"raw_file": ["run_A"], "condition": ["A"]}),
        ("raw_file",),
        (),
        IN_MEMORY_ANNOTATION,
    )

    matches = match_annotation(
        table,
        parsed,
        {"ion": annotation_matching_for(parsed.levels["ion"])},
    )

    assert matches.levels["ion"].matched_rows.to_list() == [True]


def test_prolfquapp_logs_annotation_only_as_warning_and_quant_only_as_info() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "unused"],
            "condition": ["A", "X"],
        }
    )
    annotation = AnnotationCompiler().compile(source).parse(_parsed(("run_A", "run_B")))

    messages: list[str] = []
    sink = logger.add(messages.append, format="{level}:{message}")
    try:
        annotation.annotate()
    finally:
        logger.remove(sink)

    assert any(
        message.startswith("WARNING:") and "annotation rows absent from quantification" in message
        for message in messages
    )
    assert any(
        message.startswith("INFO:") and "quantification rows without annotation" in message
        for message in messages
    )


@pytest.mark.parametrize("suffix", [".h5ad", ".h5mu", ".parquet", ".duckdb"])
def test_cli_annotation_round_trips_through_every_result_format(
    suffix: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / f"input{suffix}"
    target = tmp_path / f"annotated{suffix}"
    annotation = tmp_path / "samples.tsv"
    write_parsed_levels(_parsed(("run_A", "run_B")), source)
    annotation.write_text(
        "raw_file\tcondition\nrun_A\tA\nrun_B\tB\n",
        encoding="utf-8",
    )

    exit_code = annotate_command(
        source,
        annotation,
        target,
    )

    assert exit_code == 0
    restored = read_parsed_levels(target)
    assert restored.levels["ion"].obs.frame.get_column("condition").to_list() == ["A", "B"]
    root = restored.metadata["prolfquapp"]
    local = restored.levels["ion"].metadata["prolfquapp"]
    assert isinstance(root, dict) and "provenance" in root
    assert isinstance(local, dict) and set(local) == {"result", "summary"}
    assert "annotation" not in restored.metadata


def test_annotating_again_leaves_the_first_result_unchanged() -> None:
    once = (
        AnnotationCompiler()
        .compile(pl.DataFrame({"raw_file": ["run_A", "run_B"], "condition": ["A", "B"]}))
        .parse(_parsed(("run_A", "run_B")))
        .annotate()
        .parsed
    )
    before = copy.deepcopy((once.metadata, once.levels["ion"].metadata))

    AnnotationCompiler().compile(
        pl.DataFrame({"raw_file": ["run_A", "unused"], "batch": ["1", "2"]})
    ).parse(once).annotate()

    assert (once.metadata, once.levels["ion"].metadata) == before


def test_annotation_does_not_recompute_matching_during_application() -> None:
    source = pl.DataFrame({"raw_file": ["run_A"], "condition": ["A"]})
    annotation = AnnotationCompiler().compile(source).parse(_parsed(("run_A",)))
    before = annotation.matches

    first = annotation.annotate()
    second = annotation.annotate()

    assert annotation.matches is before
    assert_frame_equal(
        first.parsed.levels["ion"].obs.frame,
        second.parsed.levels["ion"].obs.frame,
    )


_SDRF_HEADER = (
    "source name",
    "characteristics[organism]",
    "characteristics[spiked compound]",
    "characteristics[spiked compound]",
    "assay name",
    "comment[label]",
    "comment[data file]",
    "factor value[spiked compound]",
)
_LABEL_FREE = "AC=MS:1002038;NT=label free sample"


def _sdrf_rows(*runs: str, label: str = _LABEL_FREE) -> list[tuple[str, ...]]:
    return [
        (
            f"mixture_{run[-1]}",
            "homo sapiens",
            "CT=mixture;SP=Saccharomyces cerevisiae;QY=30%",
            "CT=mixture;SP=Escherichia coli;QY=5%",
            f"assay_{run}",
            label,
            f"/data/{run}.raw",
            run[-1],
        )
        for run in runs
    ]


def _write_sdrf(path: Path, rows: list[tuple[str, ...]]) -> Path:
    path.write_text(
        "\n".join("\t".join(row) for row in (_SDRF_HEADER, *rows)) + "\n",
        encoding="utf-8",
    )
    return path


def test_loader_keeps_verbatim_repeated_sdrf_headers(tmp_path: Path) -> None:
    loaded = load_annotation_file(_write_sdrf(tmp_path / "design.sdrf.tsv", _sdrf_rows("run_A")))

    sdrf = SdrfSource(loaded)

    assert loaded.headers == _SDRF_HEADER
    spiked = sdrf.columns("Characteristics[Spiked Compound]")
    assert [loaded.frame.get_column(name).to_list() for name in spiked] == [
        ["CT=mixture;SP=Saccharomyces cerevisiae;QY=30%"],
        ["CT=mixture;SP=Escherichia coli;QY=5%"],
    ]
    with pytest.raises(AnnotationError, match="must occur exactly once; found 2"):
        sdrf.column("characteristics[spiked compound]")


def test_compiler_recognises_sdrf_and_matches_extensionless_runs(tmp_path: Path) -> None:
    source = _write_sdrf(tmp_path / "design.sdrf.tsv", _sdrf_rows("run_A", "run_B"))

    annotation = AnnotationCompiler().compile(source).parse(_parsed(("run_A", "run_B")))
    result = annotation.annotate()

    obs = result.parsed.levels["ion"].obs.frame
    assert obs.columns == [
        "run",
        "source_name",
        "characteristics_organism",
        "characteristics_spiked_compound",
        "characteristics_spiked_compound_duplicated_0",
        "assay_name",
        "comment_label",
        "factor_value_spiked_compound",
    ]
    assert obs.get_column("assay_name").to_list() == ["assay_run_A", "assay_run_B"]
    root = result.parsed.metadata["sdrf"]
    assert isinstance(root, dict)
    provenance = root["provenance"]
    assert isinstance(provenance, dict)
    record = provenance["annotation"]
    assert isinstance(record, dict)
    assert record["source"] == {"path": str(source.resolve())}
    columns = record["columns"]
    assert isinstance(columns, list)
    assert {
        "header": "characteristics[spiked compound]",
        "column": "characteristics_spiked_compound",
    } in columns
    assert {
        "header": "characteristics[spiked compound]",
        "column": "characteristics_spiked_compound_duplicated_0",
    } in columns
    assert all(
        isinstance(entry, dict) and entry["header"] != "comment[data file]" for entry in columns
    )


def test_in_memory_sdrf_headers_ignore_case() -> None:
    source = pl.DataFrame(
        {
            "Source Name": ["mixture_A"],
            "Comment[Data File]": ["run_A.d"],
            "Factor Value[spiked compound]": ["A"],
        }
    )

    annotation = AnnotationCompiler().compile(source).parse(_parsed(("run_A",)))

    match = annotation.matches.levels["ion"]
    assert match.matched_rows.to_list() == [True]
    assert match.corrections == ()


def test_sdrf_rejects_multiplexed_rows(tmp_path: Path) -> None:
    source = _write_sdrf(
        tmp_path / "design.sdrf.tsv",
        _sdrf_rows("run_A", label="AC=MS:1002624;NT=TMT126"),
    )

    with pytest.raises(AnnotationError, match="label-free rows only"):
        AnnotationCompiler().compile(source)


def test_sdrf_rejects_a_data_file_on_two_rows(tmp_path: Path) -> None:
    rows = _sdrf_rows("run_A", "run_B")
    rows[1] = (*rows[1][:6], rows[0][6], rows[1][7])
    parser = AnnotationCompiler().compile(_write_sdrf(tmp_path / "design.sdrf.tsv", rows))

    with pytest.raises(AnnotationError, match="duplicate annotation identifier"):
        parser.parse(_parsed(("run_A",)))


def test_sdrf_applies_the_configured_application(tmp_path: Path) -> None:
    compiler = AnnotationCompiler("error")
    parser = compiler.compile(_write_sdrf(tmp_path / "design.sdrf.tsv", _sdrf_rows("run_A")))

    with pytest.raises(AnnotationError, match="complete sample annotation required"):
        parser.parse(_parsed(("run_A", "run_B")))


def test_cli_annotates_with_sdrf(tmp_path: Path) -> None:
    source = tmp_path / "input.parquet"
    target = tmp_path / "annotated.parquet"
    write_parsed_levels(_parsed(("run_A", "run_B")), source)
    annotation = _write_sdrf(tmp_path / "design.sdrf.tsv", _sdrf_rows("run_A", "run_B"))

    exit_code = annotate_command(source, annotation, target)

    assert exit_code == 0
    restored = read_parsed_levels(target)
    obs = restored.levels["ion"].obs.frame
    assert obs.get_column("factor_value_spiked_compound").to_list() == ["A", "B"]
    local = restored.levels["ion"].metadata["sdrf"]
    assert isinstance(local, dict) and set(local) == {"result", "summary"}


def test_include_requires_dropping_unmatched_observations() -> None:
    with pytest.raises(AnnotationError, match="include requires unmatched='drop'"):
        AnnotationCompiler("keep", "include")
