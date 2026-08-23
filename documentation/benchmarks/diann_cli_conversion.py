"""Benchmark APB against APB2 on DIA-NN conversion through ``.h5ad`` writing.

The benchmark invokes both installed console scripts as separate processes from the same
virtual environment. Timed work includes application startup, parameter and rule selection,
the complete vendor-table read, conversion, AnnData construction, and ``.h5ad`` writing.
Output deletion, result loading, and parity checks are outside the timed interval.

Run from the APB2 repository root::

    .venv/bin/python documentation/benchmarks/diann_cli_conversion.py \
        DATA PARAMETER_FILE --result-json RESULT.json
"""

from __future__ import annotations

import json
import platform
import re
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import anndata
import numpy as np
import pandas as pd
from cyclopts import App, Parameter
from loguru import logger

REPOSITORY = Path(__file__).resolve().parents[2]
WORKSPACE = REPOSITORY.parent
TIME = Path("/usr/bin/time")
DEFAULT_APB = REPOSITORY / ".venv/bin/apb"
DEFAULT_APB2 = REPOSITORY / ".venv/bin/apb2"
OBS_KEY = "Run"
VAR_KEY = "ProForma_ion"

_SECONDS = {
    "wall": re.compile(r"^real\s+([0-9.]+)$", re.MULTILINE),
    "user": re.compile(r"^user\s+([0-9.]+)$", re.MULTILINE),
    "system": re.compile(r"^sys\s+([0-9.]+)$", re.MULTILINE),
}
_MAXIMUM_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size$", re.MULTILINE)

app = App(name="diann-cli-conversion", help=__doc__)


@dataclass(frozen=True, slots=True)
class Tool:
    """One converter executable and its isolated output directory."""

    name: str
    executable: Path
    output_directory: Path

    @property
    def output_basename(self) -> Path:
        """Extensionless basename passed to either converter."""
        return self.output_directory / "ion"

    @property
    def output_file(self) -> Path:
        """The ``.h5ad`` both converters append to the basename."""
        return self.output_basename.with_suffix(".h5ad")


@dataclass(frozen=True, slots=True)
class Measurement:
    """One measured end-to-end subprocess run."""

    tool: str
    repeat: int
    order: int
    wall_seconds: float
    user_seconds: float
    system_seconds: float
    maximum_rss_bytes: int
    output_bytes: int


@dataclass(frozen=True, slots=True)
class ToolSummary:
    """Robust summary over all measured runs for one converter."""

    tool: str
    repeats: int
    median_wall_seconds: float
    minimum_wall_seconds: float
    maximum_wall_seconds: float
    median_user_seconds: float
    median_system_seconds: float
    median_maximum_rss_bytes: float
    median_output_bytes: float


@dataclass(frozen=True, slots=True)
class ParitySummary:
    """What was compared between the two written AnnData objects."""

    observation_key: str
    variable_key: str
    observations: int
    variables: int
    named_layers: tuple[str, ...]
    compared_matrices: int
    compared_matrix_cells: int
    obs_metadata_columns: int
    var_metadata_columns: int


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Execution controls that are not part of the DIA-NN input identity."""

    warmups: int = 1
    repeats: int = 5
    output_directory: Path | None = None
    result_json: Path | None = None
    apb: Path = DEFAULT_APB
    apb2: Path = DEFAULT_APB2


DEFAULT_OPTIONS = BenchmarkOptions()


def _required_match(pattern: re.Pattern[str], output: str, label: str) -> str:
    found = pattern.search(output)
    if found is None:
        raise RuntimeError(f"/usr/bin/time did not report {label!r}:\n{output}")
    return found.group(1)


def _command(tool: Tool, data: Path, parameters: Path) -> tuple[str, ...]:
    return (
        str(TIME),
        "-lp",
        str(tool.executable),
        "convert",
        str(data),
        "ion",
        "--params",
        str(parameters),
        "--software",
        "diann",
        "--output",
        str(tool.output_basename),
    )


def _run(
    tool: Tool,
    data: Path,
    parameters: Path,
    *,
    repeat: int,
    order: int,
) -> Measurement:
    tool.output_directory.mkdir(parents=True, exist_ok=True)
    tool.output_file.unlink(missing_ok=True)
    command = _command(tool, data, parameters)
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{tool.name} exited {completed.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not tool.output_file.is_file():
        raise RuntimeError(f"{tool.name} succeeded without writing {tool.output_file}")
    return Measurement(
        tool=tool.name,
        repeat=repeat,
        order=order,
        wall_seconds=float(_required_match(_SECONDS["wall"], completed.stderr, "real time")),
        user_seconds=float(_required_match(_SECONDS["user"], completed.stderr, "user time")),
        system_seconds=float(_required_match(_SECONDS["system"], completed.stderr, "system time")),
        maximum_rss_bytes=int(
            _required_match(_MAXIMUM_RSS, completed.stderr, "maximum resident set size")
        ),
        output_bytes=tool.output_file.stat().st_size,
    )


def _axis_positions(expected: pd.DataFrame, actual: pd.DataFrame, key: str) -> np.ndarray:
    if key not in expected or key not in actual:
        raise AssertionError(f"authored key {key!r} is absent from one converted axis")
    expected_keys = expected[key].astype("string")
    actual_keys = actual[key].astype("string")
    if not expected_keys.is_unique or not actual_keys.is_unique:
        raise AssertionError(f"authored key {key!r} is not unique in one converted axis")
    positions = pd.Series(range(len(actual_keys)), index=actual_keys).reindex(expected_keys)
    if positions.isna().any():
        missing = expected_keys[positions.isna()].head(5).tolist()
        raise AssertionError(f"{key!r} differs between converters; missing examples={missing}")
    if set(expected_keys) != set(actual_keys):
        extra = sorted(set(actual_keys) - set(expected_keys))[:5]
        raise AssertionError(f"{key!r} differs between converters; extra examples={extra}")
    return positions.to_numpy(dtype=np.intp)


def _dense(matrix: object) -> np.ndarray:
    toarray = getattr(matrix, "toarray", None)
    materialized = toarray() if callable(toarray) else matrix
    return np.asarray(materialized, dtype=np.float64)


def _assert_axis_metadata(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    positions: np.ndarray,
    label: str,
) -> None:
    if list(expected.columns) != list(actual.columns):
        raise AssertionError(
            f"{label} metadata columns differ: {list(expected.columns)} != {list(actual.columns)}"
        )
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        actual.iloc[positions].reset_index(drop=True),
        check_dtype=False,
        check_categorical=False,
        check_names=True,
        obj=f"{label} metadata",
    )


def assert_output_parity(apb_output: Path, apb2_output: Path) -> ParitySummary:
    """Compare both persisted results by authored axis identity, not incidental index order."""
    expected = anndata.read_h5ad(apb_output)
    actual = anndata.read_h5ad(apb2_output)
    if expected.shape != actual.shape:
        raise AssertionError(f"AnnData shapes differ: {expected.shape} != {actual.shape}")

    obs_positions = _axis_positions(expected.obs, actual.obs, OBS_KEY)
    var_positions = _axis_positions(expected.var, actual.var, VAR_KEY)
    _assert_axis_metadata(expected.obs, actual.obs, obs_positions, "obs")
    _assert_axis_metadata(expected.var, actual.var, var_positions, "var")

    # AnnData 0.13 exposes ``layers[None]`` as an in-memory alias of ``X``. It is not a
    # persisted HDF5 layer, so compare only named layers here and compare ``X`` once below.
    expected_layers = tuple(name for name in expected.layers if isinstance(name, str))
    actual_layers = tuple(name for name in actual.layers if isinstance(name, str))
    if set(expected_layers) != set(actual_layers):
        raise AssertionError(f"AnnData layers differ: {expected_layers} != {actual_layers}")

    matrix_names: tuple[str | None, ...] = (*expected_layers, "X")
    compared_cells = 0
    for name in matrix_names:
        left = _dense(expected.X if name == "X" else expected.layers[name])
        right_source = _dense(actual.X if name == "X" else actual.layers[name])
        right = right_source[np.ix_(obs_positions, var_positions)]
        equal = np.isclose(left, right, rtol=1e-9, atol=0.0, equal_nan=True)
        if not bool(equal.all()):
            differing = int((~equal).sum())
            raise AssertionError(f"matrix {name!r} differs in {differing}/{left.size} cells")
        compared_cells += left.size

    return ParitySummary(
        observation_key=OBS_KEY,
        variable_key=VAR_KEY,
        observations=expected.n_obs,
        variables=expected.n_vars,
        named_layers=expected_layers,
        compared_matrices=len(matrix_names),
        compared_matrix_cells=compared_cells,
        obs_metadata_columns=expected.obs.shape[1],
        var_metadata_columns=expected.var.shape[1],
    )


def _summarize(tool: str, measurements: list[Measurement]) -> ToolSummary:
    selected = [measurement for measurement in measurements if measurement.tool == tool]
    wall = [measurement.wall_seconds for measurement in selected]
    return ToolSummary(
        tool=tool,
        repeats=len(selected),
        median_wall_seconds=statistics.median(wall),
        minimum_wall_seconds=min(wall),
        maximum_wall_seconds=max(wall),
        median_user_seconds=statistics.median(measurement.user_seconds for measurement in selected),
        median_system_seconds=statistics.median(
            measurement.system_seconds for measurement in selected
        ),
        median_maximum_rss_bytes=statistics.median(
            measurement.maximum_rss_bytes for measurement in selected
        ),
        median_output_bytes=statistics.median(measurement.output_bytes for measurement in selected),
    )


def _command_output(*command: str) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _sysctl(name: str) -> str:
    return _command_output("sysctl", "-n", name)


def _metadata(data: Path, parameters: Path, tools: tuple[Tool, ...]) -> dict[str, object]:
    apb_repository = WORKSPACE / "apb"
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("apb2", "anndata-proteomics", "anndata", "pandas", "polars", "numpy")
        },
        "git": {
            "apb": {
                "commit": _command_output("git", "-C", str(apb_repository), "rev-parse", "HEAD"),
                "status": _command_output("git", "-C", str(apb_repository), "status", "--short"),
            },
            "apb2": {
                "commit": _command_output("git", "-C", str(REPOSITORY), "rev-parse", "HEAD"),
                "status": _command_output("git", "-C", str(REPOSITORY), "status", "--short"),
            },
        },
        "data": str(data),
        "data_bytes": data.stat().st_size,
        "parameters": str(parameters),
        "level": "ion",
        "software": "diann",
        "commands": {tool.name: list(_command(tool, data, parameters))[2:] for tool in tools},
    }


def _write_json(
    path: Path,
    metadata: dict[str, object],
    measurements: list[Measurement],
    summaries: tuple[ToolSummary, ...],
    parity: ParitySummary,
    warmups: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": {
            "warmups_per_tool": warmups,
            "measured_repeats_per_tool": summaries[0].repeats,
            "run_order": "alternating by repeat",
            "filesystem_cache": (
                "warm after discarded warm-up" if warmups else "uncontrolled; no warm-up"
            ),
            "timed_scope": "CLI startup through completed .h5ad write",
            "excluded_scope": "output deletion, .h5ad loading, and parity checks",
        },
        "environment": metadata,
        "measurements": [asdict(measurement) for measurement in measurements],
        "summary": [asdict(summary) for summary in summaries],
        "comparison": {
            "median_wall_speedup_apb2_over_apb": (
                summaries[0].median_wall_seconds / summaries[1].median_wall_seconds
            ),
            "median_rss_reduction_fraction": 1
            - summaries[1].median_maximum_rss_bytes / summaries[0].median_maximum_rss_bytes,
        },
        "parity": asdict(parity),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _report(summaries: tuple[ToolSummary, ...], parity: ParitySummary) -> None:
    apb_summary, apb2_summary = summaries
    logger.info("converter | median wall | range | median peak RSS | output")
    for summary in summaries:
        logger.info(
            "{} | {:.2f} s | {:.2f}-{:.2f} s | {:.0f} MiB | {:.1f} MiB",
            summary.tool,
            summary.median_wall_seconds,
            summary.minimum_wall_seconds,
            summary.maximum_wall_seconds,
            summary.median_maximum_rss_bytes / 1024**2,
            summary.median_output_bytes / 1024**2,
        )
    logger.info(
        "APB2 median wall speedup: {:.2f}x; median peak RSS reduction: {:.1%}",
        apb_summary.median_wall_seconds / apb2_summary.median_wall_seconds,
        1 - apb2_summary.median_maximum_rss_bytes / apb_summary.median_maximum_rss_bytes,
    )
    logger.info(
        "parity: {} obs x {} var, {} matrices / {} cells, {} obs and {} var metadata columns",
        parity.observations,
        parity.variables,
        parity.compared_matrices,
        parity.compared_matrix_cells,
        parity.obs_metadata_columns,
        parity.var_metadata_columns,
    )


@app.default
def benchmark(  # noqa: C901 - linear orchestration; splitting would add forwarding stages
    data: Path,
    parameters: Path,
    options: Annotated[BenchmarkOptions, Parameter(name="*")] = DEFAULT_OPTIONS,
) -> None:
    """Run the DIA-NN v1 ion conversion benchmark against APB and APB2."""
    if platform.system() != "Darwin" or not TIME.is_file():
        raise RuntimeError(
            "this resource-measured benchmark currently requires macOS /usr/bin/time"
        )
    if options.warmups < 0 or options.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    for required in (data, parameters, options.apb, options.apb2):
        if not required.is_file():
            raise FileNotFoundError(required)

    root = options.output_directory or Path(tempfile.mkdtemp(prefix="apb-diann-benchmark."))
    tools = (
        Tool("apb", options.apb.resolve(), root / "apb"),
        Tool("apb2", options.apb2.resolve(), root / "apb2"),
    )
    logger.info("outputs: {}", root)
    logger.info("discarding {} warm-up(s) per converter", options.warmups)
    for warmup in range(options.warmups):
        for order, tool in enumerate(tools):
            _run(tool, data, parameters, repeat=-(warmup + 1), order=order)
        assert_output_parity(tools[0].output_file, tools[1].output_file)

    measurements: list[Measurement] = []
    parity: ParitySummary | None = None
    for repeat in range(options.repeats):
        ordered = tools if repeat % 2 == 0 else tuple(reversed(tools))
        logger.info("measured repeat {}/{}: {}", repeat + 1, options.repeats, ordered[0].name)
        for order, tool in enumerate(ordered):
            measurement = _run(tool, data, parameters, repeat=repeat + 1, order=order + 1)
            measurements.append(measurement)
            logger.info(
                "  {} {:.2f} s, {:.0f} MiB peak RSS",
                tool.name,
                measurement.wall_seconds,
                measurement.maximum_rss_bytes / 1024**2,
            )
        parity = assert_output_parity(tools[0].output_file, tools[1].output_file)

    if parity is None:
        raise AssertionError("the benchmark completed without a parity check")
    summaries = tuple(_summarize(tool.name, measurements) for tool in tools)
    _report(summaries, parity)
    if options.result_json is not None:
        _write_json(
            options.result_json,
            _metadata(data, parameters, tools),
            measurements,
            summaries,
            parity,
            options.warmups,
        )
        logger.info("wrote {}", options.result_json)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
