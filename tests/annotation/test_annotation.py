"""Dataset-bound sample-annotation API, policies, matching, and CLI workflow."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from loguru import logger
from polars.testing import assert_frame_equal

from apb2.annotation.application.policies import (
    AllAnnotationSelections,
    BooleanAnnotationSelection,
    MatchedAnnotationSelection,
    SelectAnnotatedObservations,
)
from apb2.annotation.compiler import AnnotationCompiler, RequireAnnotation
from apb2.annotation.data.model import (
    IN_MEMORY_ANNOTATION,
    AnnotationError,
    AnnotationKind,
    LoadedAnnotationSource,
)
from apb2.annotation.matching.core import (
    ExactAnnotationMatching,
    FuzzyAnnotationMatching,
    make_annotation_table,
    match_annotation,
)
from apb2.annotation.prolfquapp import ProlfquappAnnotationParameters
from apb2.annotation.proteobench import ProteobenchAnnotationParser
from apb2.cli import annotate as annotate_command
from apb2.parserV2.parse_quant.data.parsed import (
    FinalLayerTable,
    JsonValue,
    ObsFinal,
    ParsedLevel,
    ParsedLevels,
    VarFinal,
)
from apb2.parserV2.parse_quant.io.formats import read_parsed_levels, write_parsed_levels


def _plan() -> str:
    return json.dumps(
        {
            "ann_data": {
                "layer_encodings": [
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
        var_key_columns=("feature",),
        values=pl.DataFrame(
            {
                "feature": ["p1", "p2"],
                **{
                    f"obs_{index}": [float(index + 1), float(index + 11)]
                    for index in range(len(runs))
                },
            }
        ),
    )
    uns: dict[str, JsonValue] = {
        "quantification_level": "ion",
        "software_name": "Synthetic",
        "plan_json": _plan(),
    }
    if matching is not None:
        uns["sample_annotation_matching"] = matching
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


def _prolfquapp_compiler(
    parameters: ProlfquappAnnotationParameters | None = None,
) -> AnnotationCompiler:
    return AnnotationCompiler(
        recognition=RequireAnnotation(AnnotationKind.PROLFQUAPP),
        prolfquapp=parameters or ProlfquappAnnotationParameters(),
    )


def test_parser_constructs_a_dataset_bound_annotation_with_inspectable_matches() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "run_B", "unused"],
            "condition": ["A", "B", "X"],
        }
    )
    parsed = _parsed()

    annotation = _prolfquapp_compiler().compile(source).parse(parsed)

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


def test_proteobench_parser_raises_before_constructing_on_incomplete_coverage(
    tmp_path: Path,
) -> None:
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
    parser = AnnotationCompiler().compile(source)
    assert isinstance(parser, ProteobenchAnnotationParser)

    with pytest.raises(AnnotationError, match="complete sample annotation required"):
        parser.parse(_parsed())


def test_prolfquapp_parser_does_not_construct_an_annotation_with_zero_matches() -> None:
    source = pl.DataFrame({"raw_file": ["elsewhere"], "condition": ["A"]})

    with pytest.raises(AnnotationError, match="matched no observations"):
        _prolfquapp_compiler().compile(source).parse(_parsed(("run_A",)))


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


def test_explicit_annotation_type_is_verified_without_fallback(tmp_path: Path) -> None:
    source = tmp_path / "samples.tsv"
    source.write_text("raw_file\tcondition\nrun_A\tA\n", encoding="utf-8")

    with pytest.raises(AnnotationError, match="not valid proteobench"):
        AnnotationCompiler(recognition=RequireAnnotation(AnnotationKind.PROTEOBENCH)).compile(
            source
        )


def test_automatic_memory_recognition_reports_unsupported_and_ambiguous_sources() -> None:
    with pytest.raises(AnnotationError, match="unsupported"):
        AnnotationCompiler().compile(pl.DataFrame({"other": ["value"]}))

    ambiguous = pl.DataFrame(
        {
            "raw_file": ["run_A"],
            "sample_name": ["A"],
            "condition": ["A"],
        }
    )
    with pytest.raises(AnnotationError, match="ambiguous"):
        AnnotationCompiler().compile(ambiguous)


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
    application = SelectAnnotatedObservations(
        AllAnnotationSelections(
            (
                MatchedAnnotationSelection(),
                BooleanAnnotationSelection("include"),
            )
        )
    )
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "run_B", "run_C"],
            "include": [True, False, True],
        }
    )
    annotation = (
        _prolfquapp_compiler(ProlfquappAnnotationParameters(application=application))
        .compile(source)
        .parse(_parsed())
    )

    result = annotation.annotate().parsed.levels["ion"]

    assert result.obs.frame.to_dict(as_series=False) == {
        "run": ["run_A", "run_C"],
        "include": [True, True],
    }
    assert result.layers["Intensity"].values.columns == ["feature", "obs_0", "obs_1"]
    assert result.layers["Intensity"].values.get_column("obs_1").to_list() == [3.0, 13.0]
    assert result.obsm["quality"].get_column("score").to_list() == [0, 2]
    assert result.obsp["links"].is_empty()


def test_boolean_selection_rejects_null_for_a_matched_annotation() -> None:
    application = SelectAnnotatedObservations(BooleanAnnotationSelection("include"))
    source = pl.DataFrame({"raw_file": ["run_A"], "include": [None]})

    with pytest.raises(AnnotationError, match="must be Boolean"):
        _prolfquapp_compiler(ProlfquappAnnotationParameters(application=application)).compile(
            source
        ).parse(_parsed(("run_A",)))


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

    annotation = _prolfquapp_compiler().compile(source).parse(_parsed(("alias_B",)))

    match = annotation.matches.levels["ion"]
    assert match.matched_rows.to_list() == [True]
    assert match.corrections == ()


def test_prolfquapp_logs_annotation_only_as_warning_and_quant_only_as_info() -> None:
    source = pl.DataFrame(
        {
            "raw_file": ["run_A", "unused"],
            "condition": ["A", "X"],
        }
    )
    annotation = _prolfquapp_compiler().compile(source).parse(_parsed(("run_A", "run_B")))

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
        annotation_type=AnnotationKind.PROLFQUAPP,
    )

    assert exit_code == 0
    restored = read_parsed_levels(target)
    assert restored.levels["ion"].obs.frame.get_column("condition").to_list() == ["A", "B"]
    assert restored.uns["sample_annotation"]


def test_annotation_does_not_recompute_matching_during_application() -> None:
    source = pl.DataFrame({"raw_file": ["run_A"], "condition": ["A"]})
    annotation = _prolfquapp_compiler().compile(source).parse(_parsed(("run_A",)))
    before = annotation.matches

    first = annotation.annotate()
    second = annotation.annotate()

    assert annotation.matches is before
    assert_frame_equal(
        first.parsed.levels["ion"].obs.frame,
        second.parsed.levels["ion"].obs.frame,
    )
