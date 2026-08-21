"""Where Parser V2 spends its time, and what it allocates while spending it.

Not a threshold gate. The architecture makes four claims about cost that are properties of the
design rather than of a machine, and this measures each one so a change that breaks one is
visible:

1. modification normalization scales with *distinct* variable rows, not with measurement rows —
   which is the whole reason axis preparation happens after decomposition;
2. the remaining axis computation scales with the small axis frames;
3. duplicate resolution stays vectorized over wide Polars frames;
4. Parquet allocates no layer matrix, and only ``AnnDataWriter`` allocates one dense
   ``n_obs x n_var`` array per encoded layer.

Stage timings come from calling the stages, not from instrumenting the parser: binding, source
resolution, reading, and decomposition are each constructible on their own, and what is left of
``parse()`` is axis preparation, duplicate resolution, and final alignment.
"""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

import polars as pl
from cyclopts import App
from loguru import logger

from apb2.parserV2 import compile as composition
from apb2.parserV2.parse_quant.data.raw import RawLayerTable
from apb2.parserV2.parse_quant.duplicates import KeepFirstDuplicate, NullOnlyRawValuePresence
from apb2.parserV2.parse_quant.modifications import TokenRegexNormalizer, TokenRegexRules
from apb2.parserV2.parse_quant.parameters.axis import ModificationMapEntry
from apb2.parserV2.parse_quant.parameters.source import SingleFile
from apb2.parserV2.parse_rule_facade import ParseRuleFacade
from apb2.parserV2.vendor_parse_rules.document import SearchParameterEvidence
from apb2.parserV2.vendor_parse_rules.loader import load_rule_document
from apb2.parserV2.vendor_parse_rules.schema import QuantificationLevel

app = App(name="parser-v2-stages", help=__doc__)

NO_EVIDENCE = SearchParameterEvidence(acquisition_method="unknown", combine_charge_states=None)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One timed stage: its median wall time and the process high-water mark after it."""

    label: str
    seconds: float
    peak_rss_mib: float


def peak_rss_mib() -> float:
    """The process resident high-water mark.

    Polars allocates in Rust, outside anything ``tracemalloc`` can see, so the only honest
    memory number here is the one the operating system keeps. It is a high-water mark, so it
    never falls: read the column as "how much the process had needed by this point".
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return peak / divisor


def measured(label: str, run: Callable[[], object], repeats: int) -> Measurement:
    """Run one stage after a discarded warm-up, reporting the median of the timed runs."""
    run()
    timings: list[float] = []
    for _attempt in range(repeats):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    return Measurement(label=label, seconds=median(timings), peak_rss_mib=peak_rss_mib())


def report(title: str, measurements: Iterable[Measurement]) -> None:
    logger.info(title)
    logger.info(f"{'stage':<34}{'median s':>12}{'peak RSS MiB':>14}")
    for entry in measurements:
        logger.info(f"{entry.label:<34}{entry.seconds:>12.3f}{entry.peak_rss_mib:>14.0f}")


@app.command
def stages(
    rules: Path,
    data: Path,
    level: QuantificationLevel = "ion",
    output: Literal["parquet", "anndata"] = "parquet",
    repeats: int = 3,
) -> None:
    """Time binding, resolution, reading, decomposition, the parse remainder, and the writer."""
    document = load_rule_document(rules)
    facade = ParseRuleFacade(document, level, NO_EVIDENCE)
    working = facade.working_parameters
    accepts = composition.header_predicate(working)
    source = SingleFile(path=data)

    bound = composition.bind_source(source, working.input)
    evidence = composition.source_evidence(source, bound, accepts)
    resolved = facade.resolve_source(evidence)
    reader = composition.make_reader(bound, evidence, resolved.read)
    decomposer = composition.make_source_decomposer(
        resolved.decomposition, resolved.obs.source, resolved.var.source
    )
    table = reader.read()
    decomposer.decompose(table)
    declaration = (
        composition.ParquetOutput() if output == "parquet" else composition.AnnDataOutput()
    )
    parser = composition.ParseRuleCompiler(facade=facade, output=declaration).compile(source)
    parsed = parser.parse()

    logger.info(
        f"{data.name}: {table.frame.height} source rows x {table.frame.width} projected "
        f"columns -> {parsed.obs.frame.height} obs x {parsed.var.frame.height} var, "
        f"{len(parsed.layers)} layer(s); polars {pl.__version__}"
    )
    report(
        "stage timings (median of N, one discarded warm-up)",
        [
            measured(
                "bind + observe evidence",
                lambda: composition.source_evidence(source, bound, accepts),
                repeats,
            ),
            measured("resolve source", lambda: facade.resolve_source(evidence), repeats),
            measured("read projection", reader.read, repeats),
            measured("decompose", lambda: decomposer.decompose(table), repeats),
            measured("parse (whole)", parser.parse, repeats),
        ],
    )
    logger.info(
        "the parse remainder -- axis preparation, duplicate resolution, final alignment -- is "
        "the whole minus read and decompose"
    )
    with tempfile.TemporaryDirectory() as folder, _counted_arrays() as counted:
        suffix = ".h5ad" if output == "anndata" else ""
        write = measured(
            f"write {output}",
            lambda: parser.convert(parsed, Path(folder) / f"level{suffix}"),
            repeats,
        )
    report("serialization", [write])
    logger.info(
        f"{output} allocated {len(counted) // max(repeats + 1, 1)} dense array(s) per write "
        f"for {len(parsed.layers)} layer(s); shapes {sorted(set(counted))}"
    )


@app.command
def scaling(repeats: int = 3) -> None:
    """Check the two claims that are about *what* a stage scales with, not how fast it is."""
    entries = (
        ModificationMapEntry(
            token="ox",
            name="Oxidation",
            accession="UNIMOD:35",
            target=("M",),
            position="Anywhere",
            mass_delta=15.994915,
        ),
    )
    normalizer = TokenRegexNormalizer(
        rules=TokenRegexRules(
            token_pattern=r"\(([^()]*)\)",
            token_position="after_residue",
            case_sensitive=False,
            unknown_policy="preserve",
            entries=entries,
        ),
        sources=("Modified.Sequence",),
        proforma_output="proforma_sequence",
        stripped_output="stripped_sequence",
    )

    logger.info("modification normalization against distinct values at a fixed row count")
    rows = 200_000
    for distinct in (1_000, 10_000, 100_000):
        values = [f"PEPM(ox)IDE{index % distinct}" for index in range(rows)]
        series = pl.Series("Modified.Sequence", values)
        entry = measured(
            f"{rows} rows, {distinct} distinct",
            lambda series=series: normalizer.normalize((series,)),
            repeats,
        )
        logger.info(f"  {entry.label:<32}{entry.seconds:>10.3f} s{entry.peak_rss_mib:>10.0f} MiB")

    logger.info("duplicate resolution over a wide layer, growing the observation axis")
    policy = KeepFirstDuplicate()
    presence = NullOnlyRawValuePresence()
    for observations in (6, 60, 600):
        layer = RawLayerTable(
            layer_name="Intensity",
            raw_var_key_columns=("Feature",),
            values=pl.DataFrame(
                {
                    "Feature": [f"F{index // 2}" for index in range(20_000)],
                    **{
                        f"obs_{column}": pl.Series(
                            [float(row) for row in range(20_000)], dtype=pl.Float64
                        )
                        for column in range(observations)
                    },
                }
            ),
        )
        entry = measured(
            f"10000 keys x {observations} observations",
            lambda layer=layer: policy.resolve(layer, presence),
            repeats,
        )
        logger.info(f"  {entry.label:<32}{entry.seconds:>10.3f} s{entry.peak_rss_mib:>10.0f} MiB")


class _counted_arrays:
    """Count the dense arrays a writer allocates, by watching the one call that makes them."""

    def __init__(self) -> None:
        self._shapes: list[tuple[int, ...]] = []
        self._original = pl.DataFrame.to_numpy

    def __enter__(self) -> list[tuple[int, ...]]:
        original = self._original
        shapes = self._shapes

        def counting(frame: pl.DataFrame) -> object:
            result = original(frame)
            shapes.append(result.shape)
            return result

        pl.DataFrame.to_numpy = counting  # type: ignore[method-assign]
        return shapes

    def __exit__(self, *_exception: object) -> None:
        pl.DataFrame.to_numpy = self._original  # type: ignore[method-assign]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
