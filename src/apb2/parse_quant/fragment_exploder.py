"""``FragmentExploder``: the ``[fragments]`` block — packed lists to one row per fragment.

DIA-NN-style reports pack per-fragment values as delimiter-joined lists inside each
precursor row (``Fragment.Info`` plus parallel ``Fragment.Quant.*`` lists, aligned by
index, often terminated by a trailing delimiter). ``exploder_for(rule, header)`` reads
``label_strategy`` once and resolves the packed columns against the inspected header: a
missing column backing a required layer is an ``IncompatibleSourceError`` at
construction, a missing optional one drops out so the conversion skips its layer. The
read plan projects the frame before it reaches the exploder, so no trim happens here.
"""

from __future__ import annotations

import math

import pandas as pd

from apb2.parse_quant.errors import IncompatibleSourceError
from apb2.vendor_parse_rules.model import (
    ColumnLabeledFragments,
    Fragments,
    LongRule,
    PositionalFragments,
    WideRule,
)
from apb2.vendor_parse_rules.runtime import layer_required


def _split_packed(value: object, delimiter: str) -> list[str]:
    """Split one packed cell into tokens, dropping a trailing empty terminator."""
    if (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, float) and math.isnan(value))
    ):
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.rstrip(delimiter)  # drop DIA-NN's trailing-delimiter terminator
    if not text:
        return []
    return [token.strip() for token in text.split(delimiter)]


def _positions(tokens: list[str]) -> list[int]:
    """Positions for one token list, exploded alongside the values."""
    return list(range(len(tokens)))


class PackedLists:
    """The packed-cell mechanics both exploders hold: split into tokens, then coerce.

    A collaborator, never a base class: each exploder owns one of these and calls it.
    """

    def __init__(
        self, columns: tuple[str, ...], value_columns: tuple[str, ...], delimiter: str
    ) -> None:
        self.columns = columns
        self.value_columns = value_columns
        self.delimiter = delimiter

    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split each packed column into token lists.

        No trim: the read plan already projected the frame to exactly the columns the
        rule reads, so everything present survives the explode.
        """
        missing = [column for column in self.columns if column not in df.columns]
        if missing:
            raise KeyError(
                f"[fragments] references column(s) missing from the input: {missing}; "
                f"available: {list(df.columns)[:10]}…"
            )
        work = df.copy()
        for column in self.columns:
            work[column] = work[column].map(lambda value: _split_packed(value, self.delimiter))
        return work

    def to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce exploded values to numeric so they ride the frame as float64."""
        for column in self.value_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        return df


class PositionalExplode:
    """Fan packed fragment values out, labelling them ``frag_0``, ``frag_1``, … by index."""

    def __init__(self, fragments: PositionalFragments, value_columns: tuple[str, ...]) -> None:
        self.value_columns = value_columns
        self.label_output = fragments.label_output
        self.packed = PackedLists(value_columns, value_columns, fragments.delimiter)

    def packed_columns(self) -> tuple[str, ...]:
        return self.packed.columns

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self.packed.split(df)
        # A parallel index list per precursor, exploded alongside the values.
        first = self.value_columns[0]
        work["_frag_pos"] = work[first].map(_positions)
        work = work.explode([*self.value_columns, "_frag_pos"], ignore_index=True)
        work = work.dropna(subset=[first]).reset_index(drop=True)
        work[self.label_output] = [f"frag_{int(pos)}" for pos in work["_frag_pos"]]
        work = work.drop(columns=["_frag_pos"])
        return self.packed.to_numeric(work)


class ColumnLabeledExplode:
    """Fan packed fragment values out, taking each label from a packed label column."""

    def __init__(self, fragments: ColumnLabeledFragments, value_columns: tuple[str, ...]) -> None:
        self.label_column = fragments.label_column
        self.label_output = fragments.label_output
        self.packed = PackedLists(
            (fragments.label_column, *value_columns), value_columns, fragments.delimiter
        )

    def packed_columns(self) -> tuple[str, ...]:
        return self.packed.columns

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        work = self.packed.split(df)
        # Multi-column explode keeps the lists aligned and raises if their lengths differ.
        work = work.explode(list(self.packed.columns), ignore_index=True)
        # Precursors with no fragments explode to a NaN row; drop them.
        work = work.dropna(subset=[self.label_column]).reset_index(drop=True)
        work[self.label_output] = work[self.label_column].astype(str).str.split("/").str[0]
        # Drop the packed label column so the (now ~12x longer) frame does not carry a
        # redundant long string column.
        work = work.drop(columns=[self.label_column])
        return self.packed.to_numeric(work)


class NoFragments:
    """The rule declares no fragment expansion, so the frame passes through untouched."""

    def packed_columns(self) -> tuple[str, ...]:
        return ()

    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


type FragmentExploder = PositionalExplode | ColumnLabeledExplode | NoFragments


def exploder_for(rule: LongRule | WideRule, header: list[str]) -> FragmentExploder:
    """Read the rule's ``label_strategy`` once, and return the exploder it names.

    Packed columns resolve against the header here: a missing column backing a required
    layer fails construction; a missing optional one is dropped so the conversion skips
    its layer, exactly as it does on the non-fragment path.
    """
    fragments: Fragments | None = rule.fragments
    if fragments is None:
        return NoFragments()
    header_set = set(header)
    required_sources = {layer.source for layer in rule.layers if layer_required(rule, layer)}
    missing_required = [
        column
        for column in fragments.value_columns
        if column not in header_set and column in required_sources
    ]
    if missing_required:
        raise IncompatibleSourceError(
            f"input lacks the packed fragment column(s) {missing_required} required by "
            f"{rule.software_name!r} level {rule.quantification_level!r}"
        )
    value_columns = tuple(column for column in fragments.value_columns if column in header_set)
    if not value_columns:
        raise IncompatibleSourceError(
            f"input carries none of the packed fragment columns {list(fragments.value_columns)} "
            f"declared by {rule.software_name!r} level {rule.quantification_level!r}"
        )
    if isinstance(fragments, ColumnLabeledFragments):
        return ColumnLabeledExplode(fragments, value_columns)
    return PositionalExplode(fragments, value_columns)
