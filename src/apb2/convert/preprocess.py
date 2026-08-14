"""The two normalisation steps that run before a table is converted, as types.

``convert_table`` used to open by asking three questions about which kind of rule it had
received — is there a ``[modifications]`` block, does any compute consume it, is there a
``[fragments]`` block — and every helper downstream re-derived the answers. Three optional
blocks give one rule 2^3 possible shapes, and each consumer had to work out which
one it got. That is what left ``_columns_needed_for_long`` with a branch on
``rule.fragments is not None`` that could never be false, because its only call site was
already inside that check.

Here the absent case is a **member** of each union with the identity behaviour, never a
``| None`` wrapped around it. ``NoModifications.apply`` returns the frame; so does
``NoFragments.explode``. Each of those bodies is literally the line the deleted branch
used to skip to.

The modification-consumed check is not lost, it moves. A rule that only inherits a
``[modifications]`` block from its vendor base without a consuming compute skips the
per-row tokenization; deciding that once, when the applier is built, is strictly better
than deciding it per table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from apb2.convert._fragments import (
    coerce_fragment_values,
    fragment_positions,
    split_packed_columns,
)
from apb2.modifications.pipeline import apply_modifications
from apb2.vendor_parse_rules.model import (
    ColumnLabeledFragments,
    LongRule,
    Modifications,
    ProformaSequence,
    StrippedSequence,
    WideRule,
)


@dataclass(frozen=True, slots=True)
class ApplyModifications:
    """Normalize a vendor modified-sequence column before the computes read it."""

    modifications: Modifications

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # apply_modifications adds its output columns in place, so copy first: callers
        # converting several levels from one table must not see another level's columns.
        return apply_modifications(df.copy(), self.modifications)

    def source_columns(self) -> frozenset[str]:
        return frozenset(
            {
                *self.modifications.source_columns,
                self.modifications.output_column,
                "stripped_sequence",
            }
        )


@dataclass(frozen=True, slots=True)
class NoModifications:
    """No modification normalization runs, either because none is declared or none is read."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def source_columns(self) -> frozenset[str]:
        return frozenset()


type ModificationApplier = ApplyModifications | NoModifications


@dataclass(frozen=True, slots=True)
class PositionalExplode:
    """Fan packed fragment values out, labelling them ``frag_0``, ``frag_1``, … by index."""

    value_columns: tuple[str, ...]
    delimiter: str
    label_output: str

    def packed_columns(self) -> tuple[str, ...]:
        return self.value_columns

    def explode(self, df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
        work = split_packed_columns(df[keep], self.packed_columns(), self.delimiter)
        # A parallel index list per precursor, exploded alongside the values.
        first = self.value_columns[0]
        work["_frag_pos"] = work[first].map(fragment_positions)
        work = work.explode([*self.value_columns, "_frag_pos"], ignore_index=True)
        work = work.dropna(subset=[first]).reset_index(drop=True)
        work[self.label_output] = [f"frag_{int(pos)}" for pos in work["_frag_pos"]]
        work = work.drop(columns=["_frag_pos"])
        return coerce_fragment_values(work, self.value_columns)


@dataclass(frozen=True, slots=True)
class ColumnLabeledExplode:
    """Fan packed fragment values out, taking each label from a packed label column."""

    value_columns: tuple[str, ...]
    label_column: str
    delimiter: str
    label_output: str

    def packed_columns(self) -> tuple[str, ...]:
        return (self.label_column, *self.value_columns)

    def explode(self, df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
        packed = self.packed_columns()
        work = split_packed_columns(df[keep], packed, self.delimiter)
        # Multi-column explode keeps the lists aligned and raises if their lengths differ.
        work = work.explode(list(packed), ignore_index=True)
        # Precursors with no fragments explode to a NaN row; drop them.
        work = work.dropna(subset=[self.label_column]).reset_index(drop=True)
        work[self.label_output] = work[self.label_column].astype(str).str.split("/").str[0]
        # Drop the packed label column so the (now ~12x longer) frame does not carry a
        # redundant long string column.
        work = work.drop(columns=[self.label_column])
        return coerce_fragment_values(work, self.value_columns)


@dataclass(frozen=True, slots=True)
class NoFragments:
    """The rule declares no fragment expansion, so the frame passes through untouched."""

    def packed_columns(self) -> tuple[str, ...]:
        return ()

    def explode(self, df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
        """Return the frame untouched, trimming nothing.

        The ``keep`` trim exists only to stop the explode multiplying unused columns ~12x,
        so a rule with no fragment expansion must not have its frame narrowed here.
        """
        return df


type FragmentExploder = PositionalExplode | ColumnLabeledExplode | NoFragments


def make_modification_applier(rule: LongRule | WideRule) -> ModificationApplier:
    """Decide once whether modification normalization runs for this rule.

    Both conditions are absence questions rather than kind questions — is a block declared,
    does anything read its output — so they are ordinary control flow. What changes is that
    they are asked once, when the applier is built, instead of on every table.
    """
    if rule.modifications is None:
        return NoModifications()
    consumed = any(
        isinstance(column, ProformaSequence | StrippedSequence)
        for column in rule.columns.var.computed
    )
    if not consumed:
        return NoModifications()
    return ApplyModifications(rule.modifications)


def make_fragment_exploder(rule: LongRule | WideRule) -> FragmentExploder:
    """Read a rule's ``[fragments]`` block once, and return the exploder it names.

    Takes the rule rather than ``Fragments | None`` deliberately: an optional parameter here
    would mean the absence had merely moved from the consumers into this signature, and a
    guarded computation module may not carry one.
    """
    fragments = rule.fragments
    if fragments is None:
        return NoFragments()
    values = tuple(fragments.value_columns)
    if isinstance(fragments, ColumnLabeledFragments):
        return ColumnLabeledExplode(
            values,
            fragments.label_column,
            fragments.delimiter,
            fragments.label_output,
        )
    return PositionalExplode(values, fragments.delimiter, fragments.label_output)
