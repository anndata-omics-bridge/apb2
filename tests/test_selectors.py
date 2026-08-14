"""Every schema selector literal has a runtime class behind it, and vice versa.

rules.json keys divide into configuration and *selectors*: each discriminator literal
names a strategy class a factory instantiates. These tests pin that connection — a
selector value with no class behind it must fail here, not partway through a conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from apb2.columns import (
    CoalesceColumn,
    DerivedSequenceColumn,
    JoinNonEmptyColumn,
    ProformaFragmentColumn,
    ProformaIonColumn,
    computer_for,
)
from apb2.conversion import LongConversion, WideConversion, conversion_for
from apb2.duplicates import POLICY_BY_MODE, policy_for
from apb2.fragments import exploder_for
from apb2.layers import (
    FactorCoercion,
    PlainNumericCoercion,
    RegexNumericCoercion,
    coercion_for,
)
from apb2.modifications.pipeline import applier_for
from apb2.vendor_parse_rules.documents.select import packaged_documents
from apb2.vendor_parse_rules.model import (
    Coalesce,
    DuplicateMode,
    Duplicates,
    FactorLayer,
    JoinNonempty,
    NumericLayer,
    ProformaFragment,
    ProformaIon,
    ProformaSequence,
    QuantificationLevel,
    RegexValuePattern,
    StrippedSequence,
    compose_rule,
    load_document,
)
from apb2.vendor_parse_rules.runtime import recognition_for

_RULES = [
    pytest.param(document.path, level, id=f"{document.path.parent.name}/{level}")
    for document in packaged_documents()
    for level in document.levels
]


def test_every_duplicate_mode_has_a_policy_or_is_the_documented_exception() -> None:
    assert set(get_args(DuplicateMode)) == set(POLICY_BY_MODE) | {"keep_all_as_raw_table"}
    with pytest.raises(NotImplementedError, match="keep_all_as_raw_table"):
        policy_for(Duplicates(mode="keep_all_as_raw_table"))


def test_every_computed_how_selects_its_computer_class() -> None:
    cases = [
        (Coalesce(how="coalesce", name="c", inputs=["a", "b"]), CoalesceColumn),
        (
            JoinNonempty(how="join_nonempty", name="j", inputs=["a", "b"], separator=";"),
            JoinNonEmptyColumn,
        ),
        (StrippedSequence(how="stripped_sequence", inputs=["a"]), DerivedSequenceColumn),
        (ProformaSequence(how="proforma_sequence", inputs=["a"]), DerivedSequenceColumn),
        (ProformaIon(how="proforma_ion", inputs=["a", "b"]), ProformaIonColumn),
        (ProformaFragment(how="proforma_fragment", inputs=["a", "b"]), ProformaFragmentColumn),
    ]
    for column, expected in cases:
        assert type(computer_for(column)) is expected


def test_every_layer_encoding_selects_its_coercion_class() -> None:
    numeric = NumericLayer(name="n", source="s")
    structured = NumericLayer(
        name="n",
        source="s",
        value_pattern=RegexValuePattern(mode="regex", pattern=r"(\d+)"),
    )
    factor = FactorLayer(encoding_mode="factor", name="f", source="s", categories={"a": 1})
    assert type(coercion_for(numeric)) is PlainNumericCoercion
    assert type(coercion_for(structured)) is RegexNumericCoercion
    assert type(coercion_for(factor)) is FactorCoercion


@pytest.mark.parametrize(("document", "level"), _RULES)
def test_every_packaged_selector_value_constructs_its_strategy(
    document: Path, level: QuantificationLevel
) -> None:
    """Sweep the packaged tree: every selector in every rule builds a runtime object."""
    rule = compose_rule(load_document(document), level)
    recognition = recognition_for(rule)
    applier_for(rule)
    exploder_for(rule)
    conversion = conversion_for(rule, strict=False)
    assert isinstance(conversion, LongConversion | WideConversion)
    for layer in rule.layers:
        coercion_for(layer)
    for _axis, group in recognition.column_groups():
        for column in group.computed:
            computer_for(column)
