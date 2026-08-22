"""Working parse parameters: the rule with its search-parameter evidence already consumed.

This is the pre-source stage. Levels are chosen, gates and overrides applied, and every
Pydantic model gone — but physical column matches, dialects, read dtypes, and optional-source
presence are all still open. ``ParseRuleFacade`` constructs these values and then resolves
them against observed evidence; nothing here reads a file.

``JsonScalar`` and ``JsonValue`` are declared locally for the same reason as in
``data/parsed.py``: provenance enters as data, and a shared parent alias module would force
this child to import upward.

Every example in this module is a value a packaged rule really produces, named by the vendor
and level it came from. Any of them can be reproduced in four lines:

    from apb2.parser_v2 import unknown_search_parameters
    from apb2.parserV2.parse_rule_facade import ParseRuleFacade
    from apb2.parserV2.vendor_parse_rules.loader import PACKAGED, load_rule_document

    document = load_rule_document(next(p for p in PACKAGED if p.parent.name == "peaks"))
    working = ParseRuleFacade(document, "ion", unknown_search_parameters()).working_parameters

The five vendors the examples draw on, and why each was chosen:

    MaxQuant     long layout, ``aggregate`` duplicates, optional obs columns
    PEAKS        wide layout, a declared numeric sentinel, one regex-valued layer
    FragPipe     wide layout, one factor layer
    DIA-NN       packed positional fragments
    Spectronaut  several permitted delimiters and both decimal notations
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from apb2.parserV2.parse_quant.parameters.axis import (
    AxisColumnDeclaration,
    ModificationConfig,
)
from apb2.parserV2.parse_quant.parameters.measurements import DuplicateMode
from apb2.parserV2.parse_quant.parameters.source import InputContract

# Ruff RUF036 wants ``None`` last; the specification's ordering is otherwise identical.
type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type QuantificationLevel = Literal["ion", "peptidoform", "peptide", "protein", "fragment"]
"""The parsing-owned level vocabulary, structurally equal to the rule package's own."""

LEVELS: tuple[QuantificationLevel, ...] = (
    "ion",
    "peptidoform",
    "peptide",
    "protein",
    "fragment",
)
"""Canonical level order, which ``compile_parsers`` preserves."""


# ---------------------------------------------------------------------------- source layout


@dataclass(frozen=True, slots=True)
class LongSourceLayout:
    """One physical row per (observation, feature).

    ``kind`` is the whole record: under a long layout both axes read ordinary columns of the
    same row, and each layer names one physical column, so there is nothing further to state.

    Examples:
        MaxQuant ``evidence.txt``, level ``ion`` — one row carries ``Raw file`` (the
        observation), ``Modified sequence`` and ``Charge`` (the feature), and ``Intensity``
        (the measurement):

            LongSourceLayout(kind="long")
    """

    kind: Literal["long"]


@dataclass(frozen=True, slots=True)
class WideSourceLayout:
    """One physical row per feature; observations are header captures.

    Also a tag only. Where the observation names live is stated by each layer's ``source``
    regex, not here, so this record has nothing to carry.

    Examples:
        PEAKS ion export — one row per precursor, and the observation names come out of
        column headers such as ``LFQ_Orbitrap_DDA_Condition_A_Sample_Alpha_01 Normalized
        Area``. Its obs axis therefore declares no selection at all: its only column is the
        synthesized ``sample``:

            WideSourceLayout(kind="wide")
    """

    kind: Literal["wide"]


@dataclass(frozen=True, slots=True)
class PositionalFragmentLayout:
    """Long rows whose packed fragment lists carry no labels; labels are positional.

    Examples:
        DIA-NN v1, level ``fragment``:

            PositionalFragmentLayout(
                kind="positional_fragment",
                delimiter=";",
                label_output="fragment_label",
                packed_value_sources=("Fragment.Quant.Raw", "Fragment.Correlations"),
            )

        A precursor row whose ``Fragment.Quant.Raw`` cell holds ``1204.5;0;88.1;`` becomes
        three rows, labelled ``frag_0``, ``frag_1`` and ``frag_2`` in the synthesized
        ``fragment_label`` column. The trailing delimiter is a terminator, not a fourth
        fragment. ``Fragment.Correlations`` is split in parallel and must hold the same
        number of scalars in that row, or ``PackedLengthError`` names the offending rows.
        ``label_output`` is synthesized, and it is the column ``ProForma_fragment`` is later
        computed from, so no physical source column may carry that name — a rule that lets the
        two collide is rejected during projection.
    """

    kind: Literal["positional_fragment"]
    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ColumnLabeledFragmentLayout:
    """Long rows whose fragment labels are packed in parallel in ``label_source``.

    Examples:
        No packaged rule uses this layout yet; the shape is exercised by
        ``tests/parserV2/test_rule_package.py``, and a rule declaring ``"label_strategy":
        "column"`` with ``"value_columns": ["Quantity"]`` and ``"label_column": "Info"``
        projects to:

            ColumnLabeledFragmentLayout(
                kind="column_labeled_fragment",
                label_source="Info",
                delimiter=";",
                label_output="fragment_label",
                packed_value_sources=("Quantity",),
            )

        ``Info`` is split alongside ``Quantity``, and each scalar's label is that token up to
        its first ``/`` — a packed ``y7/1204.6;b3/301.2`` labels the two rows ``y7`` and
        ``b3``. ``label_source`` is dropped afterwards: it has done its work, and carrying it
        would leave a redundant long string on every scalar row.
    """

    kind: Literal["column_labeled_fragment"]
    label_source: str
    delimiter: str
    label_output: str
    packed_value_sources: tuple[str, ...]


type SourceLayoutDeclaration = (
    LongSourceLayout | WideSourceLayout | PositionalFragmentLayout | ColumnLabeledFragmentLayout
)


# ------------------------------------------------------------------------ layer declarations
#
# Each measurement layer carries two declarations about the same source column, because two
# different questions are asked of one raw cell. *Presence* — this group — asks whether the
# cell claims to hold a measurement, and duplicate resolution asks it in every parse, whatever
# the output backend. *Encoding* — the group after it — is the lossy conversion to a number,
# and only the AnnData writer ever builds one. A Parquet-only parse holds presence strategies
# and no encoders at all.


@dataclass(frozen=True, slots=True)
class NullOnlyRawValuePresenceDeclaration:
    """No sentinel and no structure: only null claims nothing.

    What a layer projects to when its rule declares neither ``missing_values`` nor a
    ``value_pattern`` — and what *every* factor layer projects to regardless of what it
    declares, because a category label is a label and claims its cell.

    Examples:
        MaxQuant ``Intensity``, and FragPipe ``Match_Type``:

            NullOnlyRawValuePresenceDeclaration(kind="null_only")

        A null claims nothing, and so does ``NaN`` — that is what a float column writes
        where a null cannot be stored. The text ``0``, an empty label, and any other token
        all claim their cell.
    """

    kind: Literal["null_only"]


@dataclass(frozen=True, slots=True)
class PlainNumericRawValuePresenceDeclaration:
    """Declared numeric sentinels claim nothing, alongside null and blank text.

    Examples:
        PEAKS ``Normalized_Area`` and FragPipe ``Intensity``, both of which declare
        ``"missing_values": [0]``:

            PlainNumericRawValuePresenceDeclaration(
                kind="plain_numeric", missing_values=(0.0,)
            )

        ``missing_values`` holds floats even where the rule wrote the integer ``0``, and the
        comparison happens after the file's own notation is applied — so ``0``, ``0.0000``
        and, in a comma-decimal file, ``0,0000`` are all the same sentinel and claim nothing.
        A null or blank cell claims nothing. ``7576388.5000`` claims its cell. A non-blank
        token this notation cannot read — ``-``, ``NA`` — still claims its cell, on purpose:
        keep-first must not be able to hide a value that will fail to encode later.
    """

    kind: Literal["plain_numeric"]
    missing_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegexNumericRawValuePresenceDeclaration:
    r"""The comparable number is one capture of a structured token.

    For a layer whose cells are not bare numbers but a composite the vendor writes, of which
    one capture group is the comparable quantity. ``pattern``'s group 1 is extracted first,
    and only that capture is compared against ``missing_values``.

    Examples:
        PEAKS ``AScore``, whose rule declares
        ``"value_pattern": {"mode": "regex", "pattern": ":(-?\\d+(?:\\.\\d+)?)(?:;|$)"}``:

            RegexNumericRawValuePresenceDeclaration(
                kind="regex_numeric",
                missing_values=(),
                pattern=r":(-?\d+(?:\.\d+)?)(?:;|$)",
            )

        In the cached PEAKS ion export, one sample's AScore column reads
        ``M7:Oxidation (M):1000.00`` where a site was scored and is blank in 74 178 of its
        76 129 cells. Group 1 captures ``1000.00``, so the cell is present. ``missing_values``
        is empty here, so only a blank cell claims nothing — a token whose structure does not
        match keeps claiming its cell, and it is the encoder, not presence, that later turns
        it into a missing value.

        Note what this type is *not*: it is the presence half, used by duplicate resolution in
        every parse. Its AnnData counterpart, carrying the same two fields for the conversion,
        is ``RegexNumericEncodingDeclaration`` below.
    """

    kind: Literal["regex_numeric"]
    missing_values: tuple[float, ...]
    pattern: str


type RawValuePresenceDeclaration = (
    NullOnlyRawValuePresenceDeclaration
    | PlainNumericRawValuePresenceDeclaration
    | RegexNumericRawValuePresenceDeclaration
)


@dataclass(frozen=True, slots=True)
class PlainNumericEncodingDeclaration:
    """A layer whose cells are directly parseable numbers.

    Examples:
        MaxQuant ``Intensity``, whose rule declares no sentinel:

            PlainNumericEncodingDeclaration(kind="plain_numeric", missing_values=())

        PEAKS ``Normalized_Area``, which declares ``0``:

            PlainNumericEncodingDeclaration(kind="plain_numeric", missing_values=(0.0,))

        The text ``7576388.5000`` encodes to ``7576388.5``; ``0`` encodes to missing under
        the second declaration and to ``0.0`` under the first. A non-blank token the notation
        cannot read encodes to missing and is reported by name — the vendors these rules were
        written for do write ``-``, ``NA`` and ``False`` in a column their own rule calls
        numeric, and refusing the file would convert nothing at all. Whether enough survived
        is the encoded-layer contract's question, not this record's.
    """

    kind: Literal["plain_numeric"]
    missing_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RegexNumericEncodingDeclaration:
    r"""A layer whose numeric value is one capture group of a structured cell.

    Examples:
        PEAKS ``AScore`` — the same pattern its presence declaration carries:

            RegexNumericEncodingDeclaration(
                kind="regex_numeric",
                missing_values=(),
                pattern=r":(-?\d+(?:\.\d+)?)(?:;|$)",
            )

        ``M7:Oxidation (M):1000.00`` encodes to ``1000.0``. A cell whose structure the
        pattern does not match encodes to missing; so does a captured number listed in
        ``missing_values``. The duplicated fields are deliberate: presence never converts,
        and this is where the conversion — the lossy step — is declared.
    """

    kind: Literal["regex_numeric"]
    missing_values: tuple[float, ...]
    pattern: str


@dataclass(frozen=True, slots=True)
class FactorEncodingDeclaration:
    """A layer whose cells are category labels with declared codes.

    Examples:
        FragPipe ``Match_Type``, declared as ``{"unmatched": 0, "MS/MS": 1, "MBR": 2}``:

            FactorEncodingDeclaration(
                kind="factor",
                categories=(("unmatched", 0), ("MS/MS", 1), ("MBR", 2)),
            )

        Ordered pairs rather than a mapping, so the authored order survives into storage.
        ``MBR`` encodes to ``2``; a null or a label the rule never declared encodes to
        ``-1``. A factor layer's presence declaration is always null-only, so a label the
        rule does not know still occupies its cell and is visible as ``-1`` rather than
        disappearing.
    """

    kind: Literal["factor"]
    categories: tuple[tuple[str, int], ...]


type AnnDataLayerEncodingDeclaration = (
    PlainNumericEncodingDeclaration | RegexNumericEncodingDeclaration | FactorEncodingDeclaration
)


# ------------------------------------------------------------------- the working configuration


@dataclass(frozen=True, slots=True)
class WorkingAxisConfiguration:
    """One axis's authored identity and the columns it declares.

    ``final_key_columns`` is the authored identity only — the identity that reaches the
    result. The raw columns that distinguish physical rows and the direct inputs of the
    authored key are decided against real evidence later, in ``AxisKeyPlan``.

    Examples:
        MaxQuant ``evidence.txt`` level ``ion``, obs — one required selection and two
        optional ones, and the source names differ from the output names:

            WorkingAxisConfiguration(
                final_key_columns=("Raw_File",),
                columns=AxisColumnDeclaration(
                    required_selections=(
                        AxisColumnSelection("Raw_File", "Raw file", "string"),
                    ),
                    optional_selections=(
                        AxisColumnSelection("Experiment", "Experiment", "string"),
                        AxisColumnSelection("Fraction", "Fraction", "string"),
                    ),
                    computed=(),
                    declared_order=("Raw_File", "Experiment", "Fraction"),
                ),
            )

        The same level's var axis keys on a column no file contains — the last of its four
        computed columns:

            final_key_columns=("ProForma_ion",)
            declared_order=("Sequence", "Modified_Sequence", ..., "ProForma_ion")

        PEAKS ion, obs — a wide layout synthesizes the observation name from the header
        captures, so the axis declares nothing to select:

            WorkingAxisConfiguration(
                final_key_columns=("sample",),
                columns=AxisColumnDeclaration((), (), (), ("sample",)),
            )

        Spectronaut level ``fragment``, var — six authored key columns, one of them the
        fragment identity:

            final_key_columns=(
                "EG_PrecursorId", "F_FrgIon", "F_Charge",
                "F_FrgLossType", "F_FrgType", "F_FrgNum",
            )
    """

    final_key_columns: tuple[str, ...]
    columns: AxisColumnDeclaration


@dataclass(frozen=True, slots=True)
class WorkingMeasurementLayer:
    r"""One named measurement: where its values come from and what they mean.

    ``source`` is a physical column name under a long layout and a regex with a ``sample``
    capture group under a wide one — the layout already decided which, so nothing downstream
    asks. Both declarations describe the same column and are always both present, including
    in a parse whose output is Parquet and which therefore never builds an encoder.

    Examples:
        MaxQuant ``Intensity`` — a named column, no sentinel:

            WorkingMeasurementLayer(
                name="Intensity",
                source="Intensity",
                raw_presence=NullOnlyRawValuePresenceDeclaration(kind="null_only"),
                ann_data_encoding=PlainNumericEncodingDeclaration(
                    kind="plain_numeric", missing_values=()
                ),
            )

        PEAKS ``Normalized_Area`` — one regex matching many headers, each match contributing
        one observation, and ``0`` declared as "not measured":

            WorkingMeasurementLayer(
                name="Normalized_Area",
                source=r"^(?P<sample>LFQ_.+?)(?:\.raw|_raw)? Normalized Area$",
                raw_presence=PlainNumericRawValuePresenceDeclaration(
                    kind="plain_numeric", missing_values=(0.0,)
                ),
                ann_data_encoding=PlainNumericEncodingDeclaration(
                    kind="plain_numeric", missing_values=(0.0,)
                ),
            )

        FragPipe ``Match_Type`` — the one shape where the two declarations disagree, because
        a label is present whatever it says and only the encoding knows its code:

            WorkingMeasurementLayer(
                name="Match_Type",
                source=r"^(?P<sample>.+?)(?:_[12])? Match Type$",
                raw_presence=NullOnlyRawValuePresenceDeclaration(kind="null_only"),
                ann_data_encoding=FactorEncodingDeclaration(
                    kind="factor",
                    categories=(("unmatched", 0), ("MS/MS", 1), ("MBR", 2)),
                ),
            )
    """

    name: str
    source: str
    raw_presence: RawValuePresenceDeclaration
    ann_data_encoding: AnnDataLayerEncodingDeclaration


@dataclass(frozen=True, slots=True)
class WorkingMeasurements:
    """Every declared measurement, with the primary layer already promoted to required.

    Required and optional are separate collections because a missing source means different
    things for each: incompatible for one, omitted for the other. Splitting them loses the
    authored interleaving, which the parsed result must still preserve, so ``authored_order``
    states it — the order the document declared, not a flag on a record.

    Examples:
        MaxQuant level ``ion``, whose ``evidence.txt`` holds several rows per ion and so
        declares ``"duplicates": {"mode": "aggregate"}``:

            WorkingMeasurements(
                primary_layer_name="Intensity",
                duplicate_mode="aggregate",
                required_layers=(<Intensity>,),
                optional_layers=(<MS_MS_Count>, <Retention_Time>, <Score>, <PEP>),
                authored_order=(
                    "Intensity", "MS_MS_Count", "Retention_Time", "Score", "PEP",
                ),
            )

        Only ``Intensity`` is required, because it is the primary layer and no other layer
        set ``"required": true``. An export missing ``Score`` still converts, without it;
        an export missing ``Intensity`` raises ``IncompatibleSourceError``. Either way
        ``authored_order`` still names all five, so the result reassembles the declared
        order rather than "required first, then optional".

        PEAKS, FragPipe, DIA-NN and Spectronaut all declare ``duplicate_mode="error"``:
        one row per feature is their contract, and two rows claiming one cell is a defect in
        the export rather than something to average.
    """

    primary_layer_name: str
    duplicate_mode: DuplicateMode
    required_layers: tuple[WorkingMeasurementLayer, ...]
    optional_layers: tuple[WorkingMeasurementLayer, ...]
    authored_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkingParseConfiguration:
    """One level's complete rule, storage-model free and not yet source-resolved.

    Examples:
        MaxQuant ``evidence.txt``, level ``ion``, one field per line and abbreviated where a
        field has its own example above:

            level          "ion"
            input          InputContract(
                               file_name="evidence.txt",
                               formats=(DelimitedFormatContract(
                                   extensions=(".txt",), encoding="utf8", quote_char='"',
                                   delimiter_candidates=("\\t",),
                                   number_format_candidates=(
                                       NumericTextFormat(".", ()),
                                   ),
                               ),),
                           )
            source_layout  LongSourceLayout(kind="long")
            obs            key ("Raw_File",); 1 required and 2 optional selections
            var            key ("ProForma_ion",); 13 declared columns, 4 of them computed
            measurements   primary "Intensity", mode "aggregate", 5 layers
            modifications  (TokenRegexModificationConfig(...),) with 5 resolved entries
            provenance     {"rule_json": "<2803 characters>", "schema_version": "0.3",
                            "software_name": "MaxQuant", "shape": "long",
                            "quantification_level": "ion"}

        ``modifications`` is empty unless a computed column actually reads it, and its
        entries are already resolved: each vendor token carries the Unimod name, accession,
        targets, position and mass delta it looked up during projection, so no strategy
        consults a registry. ``provenance`` values are JSON-safe by construction, which is
        what lets a writer persist them without knowing the rule package exists.

        Spectronaut level ``fragment`` shows the other end of the input contract — a rule
        that permits several dialects rather than fixing one, which binding then narrows to
        exactly one against the observed header:

            formats=(DelimitedFormatContract(
                extensions=(".tsv",), encoding="utf8", quote_char='"',
                delimiter_candidates=("\\t", ";", ","),
                number_format_candidates=(
                    NumericTextFormat(decimal_mark=".", thousands_marks=(",", " ")),
                    NumericTextFormat(decimal_mark=",", thousands_marks=(".", " ")),
                ),
            ),)
    """

    level: QuantificationLevel
    input: InputContract
    source_layout: SourceLayoutDeclaration
    obs: WorkingAxisConfiguration
    var: WorkingAxisConfiguration
    measurements: WorkingMeasurements
    modifications: tuple[ModificationConfig, ...]
    provenance: Mapping[str, JsonValue]
