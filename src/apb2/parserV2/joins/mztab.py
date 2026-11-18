"""Read an mzTab file into Parquet: each section as it is, then long tables per sample axis.

An mzTab file interleaves metadata lines (``MTD``) with tab-separated sections: a header line
(``PRH``, ``PEH``, ``PSH``, ``SMH``, ``SFH``, ``SEH``) and its rows (``PRT``, ``PEP``, ``PSM``,
``SML``, ``SMF``, ``SME``). Per-sample values sit in indexed columns such as
``protein_abundance_assay[3]`` or ``opt_ms_run[2]_x``. Each sample axis (``assay``, ``ms_run``,
``study_variable``) becomes one long table, a row per feature and sample, with the metadata of
every entity linked to the sample joined in. PSM and SME rows name their own run in
``spectra_ref``. PSM and PEP rows that mzTab repeats once per protein are folded into one row.
OpenMS writes a protein once per ``opt_global_result_type``; each type's columns move onto the
accession's one row, prefixed by the type, so no table is obs x var x type. Score slots take
the accession their metadata line gives them. Every value stays text; a line with a prefix no
section reads is refused, and comment lines are kept.
"""

from __future__ import annotations

import gzip
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import polars as pl

type Axis = Literal["assay", "ms_run", "study_variable"]

AXES: tuple[Axis, ...] = ("assay", "ms_run", "study_variable")
_AXIS: Mapping[str, Axis] = {axis: axis for axis in AXES}

SECTIONS: Mapping[str, str] = {
    "PRH": "PRT",
    "PEH": "PEP",
    "PSH": "PSM",
    "SMH": "SML",
    "SFH": "SMF",
    "SEH": "SME",
}
"""Each section's header prefix and the prefix of its rows."""

_SCORE_DEFINITIONS: Mapping[str, str] = {
    "PRT": "protein_search_engine_score",
    "PEP": "peptide_search_engine_score",
    "PSM": "psm_search_engine_score",
    "SML": "smallmolecule_search_engine_score",
    "SME": "id_confidence_measure",
}
"""The metadata key that names each section's score slots."""

_SCORE_COLUMNS = (
    (re.compile(r"^best_search_engine_score\[(\d+)\]$"), "best_score"),
    (re.compile(r"^search_engine_score\[(\d+)\]$"), "score"),
    (re.compile(r"^id_confidence_measure\[(\d+)\]$"), "score"),
)
_OPT_SAMPLE = re.compile(r"^opt_(assay|ms_run|study_variable)\[(\d+)\]_(.+)$")
_SAMPLE_SUFFIX = re.compile(r"^(.+)_(assay|ms_run|study_variable)\[(\d+)\]$")
_ENTITY = re.compile(r"^([A-Za-z_]+)\[(\d+)\](?:-(.+))?$")
_REFERENCE = re.compile(r"\b[A-Za-z_]+\[\d+\]")
_INDEX = re.compile(r"\[(\d+)\]")
_RUN_OF_SPECTRUM = r"^\s*(ms_run\[\d+\])"

_RUN_ROWS = frozenset({"PSM", "SME"})
"""Sections whose rows name their own run in ``spectra_ref``."""

_PROTEIN_COLUMNS = frozenset({"accession", "unique", "database", "database_version"})
"""PEP columns that differ between the rows mzTab writes for one peptide, one per protein."""

_FACTORS: Mapping[str, tuple[str, str]] = {"PRT": ("accession", "opt_global_result_type")}
"""Per section, a feature key and the column that splits its rows into kinds of one feature.

OpenMS writes a protein as a ``single_protein`` or ``indistinguishable_protein_group`` row and
again as a ``protein_details`` row.
"""

_LIST_SEPARATOR = "|"
_MISSING = "null"


class MzTabFormatError(ValueError):
    """An mzTab file that cannot be read without guessing which column a value belongs to."""


@dataclass(frozen=True, slots=True)
class Metadata:
    """The ``MTD`` lines of one file, in file order."""

    lines: tuple[tuple[int, str, str], ...]
    """Line number, key and value of every metadata line."""

    def value(self, key: str) -> str | None:
        """The first value given for ``key``, ignoring case: Proteome Discoverer writes
        ``mzTab-Version``."""
        wanted = key.casefold()
        return next((value for _, name, value in self.lines if name.casefold() == wanted), None)

    def frame(self) -> pl.DataFrame:
        """Every metadata line as a row."""
        return pl.DataFrame(
            {
                "line": [number for number, _, _ in self.lines],
                "key": [key for _, key, _ in self.lines],
                "value": [value for _, _, value in self.lines],
            },
            schema={"line": pl.Int64, "key": pl.String, "value": pl.String},
        )

    def entities(self) -> dict[str, dict[str, dict[str, str]]]:
        """Every ``kind[n]`` entity and its fields, by kind; a bare line's value is its ``name``.

        A field given on several lines keeps every value, separated by ``|``.
        """
        found: dict[str, dict[str, dict[str, str]]] = {}
        for _, key, value in self.lines:
            match = _ENTITY.match(key)
            if match is None:
                continue
            kind = match.group(1)
            fields = found.setdefault(kind, {}).setdefault(f"{kind}[{match.group(2)}]", {})
            name = match.group(3) or "name"
            fields[name] = f"{fields[name]}|{value}" if name in fields else value
        return found

    def score_names(self, section: str) -> dict[str, str]:
        """Each score slot of ``section``, named by its accession, or by its label without one."""
        definition = _SCORE_DEFINITIONS.get(section)
        if definition is None:
            return {}
        pattern = re.compile(rf"^{definition}\[(\d+)\]$")
        names = {
            match.group(1): _term(value)
            for _, key, value in self.lines
            if (match := pattern.match(key)) is not None
        }
        repeated = Counter(names.values())
        return {
            slot: name if repeated[name] == 1 else f"{name},{slot}" for slot, name in names.items()
        }

    def samples(self, axis: Axis) -> pl.DataFrame:
        """One row per sample of ``axis``: its own fields and those of every entity linked to it.

        An entity's columns are ``<kind>:<field>`` and ``<kind>:key``, the keys reached; several
        entities of one kind join their values with ``|`` in the order they were reached. The
        sample keeps its own reference fields; a linked entity's are resolved, not repeated.
        """
        entities = self.entities()
        links = _Links.of(entities)
        own = entities.get(axis, {})
        rows: list[dict[str, str]] = []
        for key in sorted(own, key=_index):
            row = {axis: key, **{f"{axis}:{name}": value for name, value in own[key].items()}}
            for kind, keys in _Walk(links, key).found.items():
                row.update(_described(kind, keys, entities.get(kind, {})))
            rows.append(row)
        columns = dict.fromkeys([axis, *(name for row in rows for name in row)])
        return pl.from_dicts(rows, schema=dict.fromkeys(columns, pl.String))


@dataclass(frozen=True, slots=True)
class _Links:
    """The references between metadata entities, read from their ``*_ref`` and ``*_refs`` fields.

    ``down`` holds each entity's targets, ``up`` each entity's holders, both in file order.
    """

    down: Mapping[str, tuple[str, ...]]
    up: Mapping[str, tuple[str, ...]]

    @classmethod
    def of(cls, entities: Mapping[str, Mapping[str, Mapping[str, str]]]) -> _Links:
        down: dict[str, tuple[str, ...]] = {}
        up: dict[str, list[str]] = {}
        for members in entities.values():
            for key, fields in members.items():
                targets = tuple(
                    dict.fromkeys(
                        target
                        for name, value in fields.items()
                        if name.endswith(("_ref", "_refs"))
                        for target in _REFERENCE.findall(value)
                    )
                )
                down[key] = targets
                for target in targets:
                    up.setdefault(target, []).append(key)
        return cls(down, {key: tuple(holders) for key, holders in up.items()})


class _Walk:
    """Every entity linked to one sample, reached once each, grouped by kind.

    The walk follows references down from the sample, and up to the entities holding it. From
    an entity reached upwards it goes down again only into kinds not yet on its way up, so a
    run reaches its assays' samples but not the other assays of their study variable. The
    sample's own kind is never entered.
    """

    def __init__(self, links: _Links, start: str) -> None:
        self._links = links
        self._seen = {start}
        self.found: dict[str, list[str]] = {}
        barred = frozenset({_kind(start)})
        self._down(start, barred)
        self._up(start, barred)

    def _reach(self, key: str) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        self.found.setdefault(_kind(key), []).append(key)
        return True

    def _down(self, key: str, barred: frozenset[str]) -> None:
        for target in self._links.down.get(key, ()):
            if _kind(target) not in barred and self._reach(target):
                self._down(target, barred)

    def _up(self, key: str, barred: frozenset[str]) -> None:
        for holder in self._links.up.get(key, ()):
            if _kind(holder) in barred or not self._reach(holder):
                continue
            self._down(holder, barred | {_kind(key), _kind(holder)})
            self._up(holder, barred | {_kind(key)})


@dataclass(frozen=True, slots=True)
class Section:
    """One section's header and rows, copied to a tab-separated file of their own."""

    source: Path
    name: str
    header: tuple[str, ...]
    line: int
    """The header's line in the source."""
    path: Path
    rows: int

    def write_parquet(self, target: Path) -> WideSection:
        """Store the section as it is, every cell as text; ``null`` and empty cells are missing.

        Small row groups keep a section with thousands of columns from being held whole.
        """
        pl.scan_csv(
            self.path,
            separator="\t",
            has_header=True,
            infer_schema=False,
            quote_char=None,
            null_values=["null"],
        ).sink_parquet(target, row_group_size=_WIDE_ROW_GROUP)
        return WideSection(self.source, self.name, self.header, self.line, target, self.rows)


_WIDE_ROW_GROUP = 5_000
"""Rows per Parquet row group of a wide section; MTBLS14508's 1,403-column SMF peaks at 4.6 GB."""


@dataclass(frozen=True, slots=True)
class WideSection:
    """One section stored as Parquet under the file's own column names."""

    source: Path
    name: str
    header: tuple[str, ...]
    line: int
    """The header's line in the source."""
    path: Path
    rows: int

    def scan(self) -> pl.LazyFrame:
        """The section, read only as far as a query needs."""
        return pl.scan_parquet(self.path)

    def table(self) -> WrittenTable:
        """The stored section's size."""
        return WrittenTable(self.path, self.rows, len(self.header))


@dataclass(frozen=True, slots=True)
class MzTabContent:
    """One file's metadata, comments and sections."""

    source: Path
    metadata: Metadata
    comments: tuple[tuple[int, str], ...]
    """Line number and text of every ``COM`` line."""
    sections: tuple[Section, ...]


class _SectionWriter:
    """Copies one section's rows to its own file, each fitted to the header's width."""

    def __init__(
        self, source: Path, line: int, name: str, header: list[str], workdir: Path
    ) -> None:
        names = [column.strip() for column in header]
        while names and not names[-1]:
            names.pop()
        if "" in names:
            raise MzTabFormatError(f"{source}:{line}: {name} header has an unnamed column")
        repeated = sorted(column for column, count in Counter(names).items() if count > 1)
        if repeated:
            raise MzTabFormatError(f"{source}:{line}: {name} header repeats {', '.join(repeated)}")
        self._source = source
        self._name = name
        self._header = tuple(names)
        self._line = line
        self._path = workdir / f"{name}.tsv"
        self._rows = 0
        self._handle = self._path.open("w", encoding="utf-8", newline="\n")
        self._handle.write("\t".join(names) + "\n")

    def write(self, line: int, fields: list[str]) -> None:
        """Copy one row; empty fields past the header are dropped, missing ones left empty."""
        width = len(self._header)
        if len(fields) > width:
            if any(field.strip() for field in fields[width:]):
                raise MzTabFormatError(
                    f"{self._source}:{line}: {self._name} row has values beyond its "
                    f"{width}-column header ({len(fields)} fields)"
                )
            fields = fields[:width]
        elif len(fields) < width:
            fields = [*fields, *([""] * (width - len(fields)))]
        self._handle.write("\t".join(fields) + "\n")
        self._rows += 1

    def close(self) -> Section:
        """Finish the copy."""
        self._handle.close()
        return Section(self._source, self._name, self._header, self._line, self._path, self._rows)

    def abandon(self) -> None:
        """Close the copy after a failure."""
        self._handle.close()


_HEADER_OF_ROWS: Mapping[str, str] = {row: header for header, row in SECTIONS.items()}


class _Split:
    """What one pass over a file has collected so far."""

    def __init__(self, source: Path, workdir: Path) -> None:
        self._source = source
        self._workdir = workdir
        self._metadata: list[tuple[int, str, str]] = []
        self._comments: list[tuple[int, str]] = []
        self._writers: dict[str, _SectionWriter] = {}

    def take(self, number: int, fields: list[str]) -> None:
        """Sort one line by its prefix: metadata, comment, a section header or a row."""
        prefix = fields[0].strip()
        if prefix == "MTD":
            key = fields[1].strip() if len(fields) > 1 else ""
            self._metadata.append((number, key, "\t".join(fields[2:]).strip()))
        elif prefix == "COM":
            self._comments.append((number, "\t".join(fields[1:]).strip()))
        elif prefix in SECTIONS:
            name = SECTIONS[prefix]
            if name in self._writers:
                raise MzTabFormatError(f"{self._source}:{number}: second {prefix} header")
            self._writers[name] = _SectionWriter(
                self._source, number, name, fields[1:], self._workdir
            )
        elif prefix in _HEADER_OF_ROWS:
            writer = self._writers.get(prefix)
            if writer is None:
                raise MzTabFormatError(
                    f"{self._source}:{number}: {prefix} row before its "
                    f"{_HEADER_OF_ROWS[prefix]} header"
                )
            writer.write(number, fields[1:])
        else:
            raise MzTabFormatError(f"{self._source}:{number}: unknown line prefix {prefix!r}")

    def close(self) -> MzTabContent:
        """Finish every section's copy."""
        sections = tuple(writer.close() for writer in self._writers.values())
        metadata = Metadata(tuple(self._metadata))
        return MzTabContent(self._source, metadata, tuple(self._comments), sections)

    def abandon(self) -> None:
        """Close every copy after a failure."""
        for writer in self._writers.values():
            writer.abandon()


class MzTabSplitter:
    """Split an mzTab file into its metadata, comments and one file per section."""

    def split(self, source: Path, workdir: Path) -> MzTabContent:
        """Read ``source`` once; refuse rows whose values do not fit their header."""
        split = _Split(source, workdir)
        try:
            with _open_text(source) as handle:
                for number, raw in enumerate(handle, start=1):
                    line = raw.rstrip("\r\n")
                    if line.strip():
                        split.take(number, line.split("\t"))
        except UnicodeDecodeError as error:
            split.abandon()
            raise MzTabFormatError(f"{source}: not UTF-8 text ({error})") from error
        except MzTabFormatError:
            split.abandon()
            raise
        return split.close()


@dataclass(frozen=True, slots=True)
class SectionColumns:
    """One section's header, split into feature columns and per-sample values by axis."""

    features: tuple[tuple[str, str], ...]
    """Each feature column's source and output name."""
    samples: Mapping[Axis, Mapping[str, Mapping[str, str]]]
    """Per axis, each sample key's values: output name to source column."""

    @classmethod
    def of(cls, section: WideSection, scores: Mapping[str, str]) -> SectionColumns:
        """Sort each column by the axis its name gives; score slots take their accession."""
        features: list[tuple[str, str]] = []
        samples: dict[Axis, dict[str, dict[str, str]]] = {}
        for column in section.header:
            match = _OPT_SAMPLE.match(column)
            if match is not None:
                axis, index, name = _AXIS[match.group(1)], match.group(2), f"opt_{match.group(3)}"
            elif (match := _SAMPLE_SUFFIX.match(column)) is not None:
                axis, index = _AXIS[match.group(2)], match.group(3)
                name = _score_name(match.group(1), scores)
            else:
                features.append((column, _score_name(column, scores)))
                continue
            values = samples.setdefault(axis, {}).setdefault(f"{axis}[{index}]", {})
            if name in values:
                raise MzTabFormatError(
                    f"{section.source}:{section.line}: {section.name} columns {values[name]} "
                    f"and {column} both give {name} of {axis}[{index}]"
                )
            values[name] = column
        return cls(tuple(features), samples)

    def value_names(self, axis: Axis) -> list[str]:
        """The value names of ``axis``, in header order."""
        by_key = self.samples.get(axis, {})
        return list(dict.fromkeys(name for values in by_key.values() for name in values))

    def spread(self, key: str, factor: str, values: tuple[str, ...]) -> SectionColumns:
        """The columns once ``FactorColumns`` spreads them: the key, then per factor value
        every other column, source and output name prefixed by the value."""
        kept = [(source, name) for source, name in self.features if source == key]
        moved = [
            (f"{value}.{source}", f"{value}.{name}")
            for value in values
            for source, name in self.features
            if source not in (key, factor)
        ]
        samples: dict[Axis, dict[str, dict[str, str]]] = {
            axis: {
                sample: {
                    f"{value}.{name}": f"{value}.{source}"
                    for value in values
                    for name, source in by_name.items()
                }
                for sample, by_name in by_sample.items()
            }
            for axis, by_sample in self.samples.items()
        }
        return SectionColumns((*kept, *moved), samples)


@dataclass(frozen=True, slots=True)
class SectionFold:
    """The rows mzTab writes once per protein, folded into one row per feature.

    Exactly repeated rows go first. Within a group, a column equal on every row keeps that
    value; otherwise it lists every row's value in row order, ``|``-joined with missing values
    written ``null``, so the listed columns stay aligned position by position.
    """

    keys: tuple[str, ...]

    @classmethod
    def of(cls, section: WideSection) -> SectionFold | _AsWritten:
        """A PSM is one ``PSM_ID`` in one ``spectra_ref``; a PEP row folds only the rows that
        differ in protein columns, so LC-MS features of one peptide ion stay apart."""
        header = set(section.header)
        if section.name == "PSM" and {"PSM_ID", "spectra_ref"} <= header:
            return cls(("PSM_ID", "spectra_ref"))
        if section.name == "PEP" and _PROTEIN_COLUMNS & header:
            return cls(tuple(column for column in section.header if column not in _PROTEIN_COLUMNS))
        return _AsWritten()

    def apply(self, frame: pl.LazyFrame, header: tuple[str, ...]) -> pl.LazyFrame:
        """The section with each group of rows folded into one."""
        listed = [column for column in header if column not in self.keys]
        return (
            frame.unique(maintain_order=True)
            .group_by(list(self.keys), maintain_order=True)
            .agg(_folded(column) for column in listed)
            .select(header)
        )


class _AsWritten:
    """A section whose rows are already one per feature."""

    def apply(self, frame: pl.LazyFrame, header: tuple[str, ...]) -> pl.LazyFrame:
        """The section unchanged."""
        return frame


@dataclass(frozen=True, slots=True)
class FactorColumns:
    """Rows a factor column splits into kinds of one feature, spread into columns per kind.

    Keyed by its feature alone, such a section is obs x var x factor. Each factor value's
    columns become ``<value>.<column>`` on the feature's one row, values in file order. A
    feature written twice under one value, or a row without a value, is refused: spreading it
    would pair rows that do not belong together.
    """

    key: str
    factor: str

    @classmethod
    def of(cls, section: WideSection) -> FactorColumns | _NoFactor:
        """The section's factor where ``_FACTORS`` names one and the header carries both columns."""
        named = _FACTORS.get(section.name)
        if named is not None and set(named) <= set(section.header):
            return cls(*named)
        return _NoFactor()

    def spread(
        self, section: WideSection, frame: pl.LazyFrame, columns: SectionColumns
    ) -> tuple[pl.LazyFrame, SectionColumns]:
        """One row per feature, each factor value's columns prefixed by the value."""
        values = self._values(section, frame)
        moved = [column for column in section.header if column not in (self.key, self.factor)]
        spread = frame.select(self.key).unique(maintain_order=True)
        for value in values:
            spread = spread.join(
                frame.filter(pl.col(self.factor) == value).select(
                    self.key, *(pl.col(column).alias(f"{value}.{column}") for column in moved)
                ),
                on=self.key,
                how="left",
                nulls_equal=True,
                maintain_order="left",
            )
        return spread, columns.spread(self.key, self.factor, values)

    def _values(self, section: WideSection, frame: pl.LazyFrame) -> tuple[str, ...]:
        """The factor's values in file order, once every row is placeable."""
        counts = (
            frame.group_by(self.key, self.factor, maintain_order=True)
            .agg(pl.len().alias("rows"))
            .collect()
        )
        where = f"{section.source}:{section.line}: {section.name}"
        unnamed = counts.filter(pl.col(self.factor).is_null()).get_column("rows").sum()
        if unnamed:
            raise MzTabFormatError(f"{where} rows without {self.factor}: {unnamed}")
        repeated = counts.filter(pl.col("rows") > 1)
        if repeated.height:
            key, value, rows = repeated.row(0)
            raise MzTabFormatError(
                f"{where} repeats {self.key} {key!r} under {self.factor} {value!r} "
                f"({rows} rows); {repeated.height} such {self.key} values"
            )
        return tuple(counts.get_column(self.factor).unique(maintain_order=True).to_list())


class _NoFactor:
    """A section whose rows no column splits into kinds."""

    def spread(
        self, section: WideSection, frame: pl.LazyFrame, columns: SectionColumns
    ) -> tuple[pl.LazyFrame, SectionColumns]:
        """The section and its columns unchanged."""
        return frame, columns


@dataclass(frozen=True, slots=True)
class WrittenTable:
    """One table written, with its size."""

    path: Path
    rows: int
    columns: int


@dataclass(frozen=True, slots=True)
class LongTable:
    """One section's long table for one sample axis."""

    section: str
    axis: Axis
    table: WrittenTable
    sample_columns: tuple[str, ...]
    """The axis key and the sample's metadata columns; every other column is the section's."""

    def scan(self) -> pl.LazyFrame:
        """The table with the section's columns named after it, as ``PRT.accession``."""
        frame = pl.scan_parquet(self.table.path)
        return frame.rename(
            {
                name: f"{self.section}.{name}"
                for name in frame.collect_schema().names()
                if name not in self.sample_columns
            }
        )


@dataclass(frozen=True, slots=True)
class LongTables:
    """What one file became."""

    version: str | None
    comments: int
    tables: tuple[WrittenTable, ...]
    """Every table written: stored sections, metadata, comments, samples and long tables."""
    long: tuple[LongTable, ...]

    def stacked(self) -> pl.LazyFrame:
        """Every long table in one frame, its section's columns prefixed by the section's name.

        A row of one section and axis holds no column of another section and no other axis
        key, so a level reading ``PRT.accession`` and ``assay`` finds only its own rows.
        """
        return pl.concat([table.scan() for table in self.long], how="diagonal")


class LongTableWriter:
    """Write each stored section as long tables, one per sample axis, streamed from Parquet."""

    def write(
        self,
        metadata: Metadata,
        samples: Mapping[Axis, pl.DataFrame],
        sections: tuple[WideSection, ...],
        out_dir: Path,
    ) -> tuple[LongTable, ...]:
        """Write every section's long tables; a section with no per-sample value gets none."""
        written: list[LongTable] = []
        for section in sections:
            columns = SectionColumns.of(section, metadata.score_names(section.name))
            frame = SectionFold.of(section).apply(section.scan(), section.header)
            frame, columns = FactorColumns.of(section).spread(section, frame, columns)
            run_rows = section.name in _RUN_ROWS and "spectra_ref" in section.header
            for axis in columns.samples:
                if axis == "ms_run" and run_rows:
                    continue
                melted = self._melted(frame, columns, axis)
                written.append(self._long_table(section, axis, melted, samples[axis], out_dir))
            if run_rows:
                by_run = self._by_spectrum(section, frame, columns)
                written.append(
                    self._long_table(section, "ms_run", by_run, samples["ms_run"], out_dir)
                )
        return tuple(written)

    def _long_table(
        self,
        section: WideSection,
        axis: Axis,
        long: pl.LazyFrame,
        samples: pl.DataFrame,
        out_dir: Path,
    ) -> LongTable:
        joined = long.join(samples.lazy(), on=axis, how="left")
        table = _sink(joined, out_dir / f"{section.name}.{axis}.parquet")
        return LongTable(section.name, axis, table, tuple(samples.columns))

    def _melted(self, frame: pl.LazyFrame, columns: SectionColumns, axis: Axis) -> pl.LazyFrame:
        """A row per sample of ``axis`` and feature, its values side by side.

        One selection per sample, stacked, streams; a single melt holds the long table whole.
        """
        by_key = columns.samples[axis]
        names = columns.value_names(axis)
        features = [pl.col(source).alias(name) for source, name in columns.features]
        return pl.concat(
            [
                frame.select(
                    *features,
                    pl.lit(key).alias(axis),
                    *(
                        (pl.col(values[name]) if name in values else pl.lit(None, pl.String)).alias(
                            name
                        )
                        for name in names
                    ),
                )
                for key, values in sorted(by_key.items(), key=lambda item: _index(item[0]))
            ],
            how="vertical",
        )

    def _by_spectrum(
        self, section: WideSection, frame: pl.LazyFrame, columns: SectionColumns
    ) -> pl.LazyFrame:
        """Each row once per run its ``spectra_ref`` names, with that run's own values.

        A row holding a value for a run its ``spectra_ref`` does not name is refused: the value
        would have no row to go to.
        """
        by_key = columns.samples.get("ms_run", {})
        runs = (
            pl.col("spectra_ref")
            .str.split("|")
            .list.eval(pl.element().str.extract(_RUN_OF_SPECTRUM, 1))
            .list.unique(maintain_order=True)
        )
        if any(source == "ms_run" for source, _ in columns.features):
            raise MzTabFormatError(
                f"{section.source}:{section.line}: {section.name} with spectra_ref already has "
                "an ms_run column"
            )
        self._refuse_unnamed_runs(section, frame, by_key, runs)
        values = [
            pl.coalesce(
                [
                    pl.when(pl.col("ms_run") == key).then(pl.col(by_key[key][name]))
                    for key in sorted(by_key, key=_index)
                    if name in by_key[key]
                ]
            ).alias(name)
            for name in columns.value_names("ms_run")
        ]
        return (
            frame.with_columns(runs.alias("ms_run"))
            .explode("ms_run", empty_as_null=True)
            .select(
                *(pl.col(source).alias(name) for source, name in columns.features),
                pl.col("ms_run"),
                *values,
            )
        )

    def _refuse_unnamed_runs(
        self,
        section: WideSection,
        frame: pl.LazyFrame,
        by_key: Mapping[str, Mapping[str, str]],
        runs: pl.Expr,
    ) -> None:
        if not by_key:
            return
        counts = (
            frame.select(
                (
                    pl.any_horizontal(pl.col(source).is_not_null() for source in values.values())
                    & ~runs.list.contains(pl.lit(key)).fill_null(value=False)
                )
                .sum()
                .alias(key)
                for key, values in by_key.items()
            )
            .collect()
            .row(0, named=True)
        )
        unnamed = {key: count for key, count in counts.items() if count}
        if unnamed:
            listed = ", ".join(f"{key} ({count} rows)" for key, count in unnamed.items())
            raise MzTabFormatError(
                f"{section.source}:{section.line}: {section.name} rows hold values for runs "
                f"their spectra_ref does not name: {listed}"
            )


def write_long_tables(source: Path, out_dir: Path) -> LongTables:
    """Store each section of ``source`` as Parquet in ``out_dir/sections``, then its long tables.

    The stored sections keep the file's own column names; the long tables are built from them.
    ``MTD.tsv``, ``COM.tsv`` and ``samples.<axis>.tsv`` keep the metadata, comments and sample
    tables. A file with no section holding per-sample values is refused.
    """
    stored = out_dir / "sections"
    stored.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mztab-", dir=out_dir) as workdir:
        content = MzTabSplitter().split(source, Path(workdir))
        sections = tuple(
            section.write_parquet(stored / f"{section.name}.parquet")
            for section in content.sections
        )
    metadata = content.metadata
    written = [_write_tsv(metadata.frame(), out_dir / "MTD.tsv")]
    if content.comments:
        written.append(_write_tsv(_comment_frame(content.comments), out_dir / "COM.tsv"))
    samples: dict[Axis, pl.DataFrame] = {axis: metadata.samples(axis) for axis in AXES}
    written.extend(
        _write_tsv(table, out_dir / f"samples.{axis}.tsv")
        for axis, table in samples.items()
        if table.height
    )
    long = LongTableWriter().write(metadata, samples, sections, out_dir)
    if not long:
        raise MzTabFormatError(f"{source}: no section has per-sample values")
    return LongTables(
        metadata.value("mzTab-version"),
        len(content.comments),
        (*(section.table() for section in sections), *written, *(table.table for table in long)),
        long,
    )


def first_section_header(handle: TextIO) -> tuple[str, ...]:
    """The first section header line's fields, prefix first, of a text that starts with ``MTD``.

    Comment lines may come anywhere, the first line too. Reads no further than that header; a
    text whose first other line is not ``MTD`` gives none.
    """
    started = False
    for raw in handle:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        prefix = fields[0].strip()
        if prefix == "COM":
            continue
        if not started:
            if prefix != "MTD":
                return ()
            started = True
        elif prefix in SECTIONS:
            return tuple(field.strip() for field in fields)
    return ()


def identify(headers: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """Bind the mzTab file: metadata first, then a section header.

    A pre-1.0 file, whose sections carry ``unit_id``, is refused.
    """
    for name, header in headers.items():
        if not header or header[0] not in SECTIONS:
            continue
        if "unit_id" in header[1:]:
            raise MzTabFormatError(f"{name}: pre-1.0 mzTab (a unit_id column) is not read")
        return {"mztab": name}
    return {}


def join(frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """The mzTab file's stacked long tables, as its reader built them."""
    return frames["mztab"]


def _open_text(source: Path) -> TextIO:
    """Plain or gzipped text; a byte-order mark is dropped, line ends are kept."""
    if source.suffix.lower() == ".gz":
        return gzip.open(source, "rt", encoding="utf-8-sig", newline="")
    return source.open(encoding="utf-8-sig", newline="")


def _index(key: str) -> int:
    match = _INDEX.search(key)
    return int(match.group(1)) if match else 0


def _kind(key: str) -> str:
    """``ms_run`` of ``ms_run[3]``."""
    return key.split("[", 1)[0]


def _described(
    kind: str, keys: list[str], members: Mapping[str, Mapping[str, str]]
) -> dict[str, str]:
    """The columns of the entities of one kind: their keys, and each field aligned across them.

    An entity without a field leaves its position empty. Their own references are left out: the
    walk has resolved them, and a holder's list of members would repeat its other members.
    """
    names = dict.fromkeys(
        name
        for key in keys
        for name in members.get(key, {})
        if not name.endswith(("_ref", "_refs"))
    )
    described = {f"{kind}:key": _LIST_SEPARATOR.join(keys)}
    for name in names:
        described[f"{kind}:{name}"] = _LIST_SEPARATOR.join(
            members.get(key, {}).get(name, "") for key in keys
        )
    return described


def _folded(column: str) -> pl.Expr:
    """One value where a group's rows agree, else every row's value in order."""
    values = pl.col(column)
    return (
        pl.when(values.n_unique() == 1)
        .then(values.first())
        .otherwise(values.fill_null(_MISSING).str.join(_LIST_SEPARATOR))
        .alias(column)
    )


def _term(value: str) -> str:
    """A CV parameter's accession, or its label where the accession is empty.

    ``[MS, MS:1001171, Mascot:score, ]`` gives ``MS:1001171``; a value that is no CV parameter
    stays as it is.
    """
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return text
    parts = text[1:-1].split(",", 2)
    if len(parts) < 3:
        return text
    accession = parts[1].strip()
    return accession or parts[2].rsplit(",", 1)[0].strip() or text


def _score_name(column: str, scores: Mapping[str, str]) -> str:
    """A score slot named by its metadata term, ``best_score[MS:1001171]``; others unchanged."""
    for pattern, stem in _SCORE_COLUMNS:
        match = pattern.match(column)
        if match is not None and match.group(1) in scores:
            return f"{stem}[{scores[match.group(1)]}]"
    return column


def _comment_frame(comments: tuple[tuple[int, str], ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {"line": [line for line, _ in comments], "text": [text for _, text in comments]},
        schema={"line": pl.Int64, "text": pl.String},
    )


def _write_tsv(table: pl.DataFrame, path: Path) -> WrittenTable:
    table.write_csv(path, separator="\t", quote_style="never")
    return WrittenTable(path, table.height, table.width)


def _sink(long: pl.LazyFrame, path: Path) -> WrittenTable:
    try:
        names = long.collect_schema().names()
    except pl.exceptions.DuplicateError as error:
        raise MzTabFormatError(f"{path.name}: {error}") from error
    long.sink_parquet(path)
    rows = pl.scan_parquet(path).select(pl.len()).collect().item()
    return WrittenTable(path, int(rows), len(names))
