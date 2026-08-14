"""Turn one composed rule into the conversion it describes.

The single composition-root dispatch over the rule shape lives here. Past
``make_conversion`` the shape does not exist: a ``LongConversion`` has obs columns to
select, a ``WideConversion`` has none — its observation axis comes from the layer regex
captures — and neither carries a flag the other could misread.
"""

from __future__ import annotations

from apb2.convert._axis import non_sample_columns
from apb2.convert.duplicates import policy_for
from apb2.convert.layers import LayerPlan, make_layer_plan
from apb2.convert.long import LongConversion
from apb2.convert.wide import WideConversion
from apb2.vendor_parse_rules.model import Layer, LongRule, WideRule


def _layer_plans(layers: list[Layer], x_layer: str) -> tuple[LayerPlan, ...]:
    return tuple(make_layer_plan(layer, x_layer) for layer in layers)


type Conversion = LongConversion | WideConversion


def make_conversion(rule: LongRule | WideRule) -> Conversion:
    """Read a rule's shape once, and return the conversion it names."""
    x_layer = rule.axis.x_layer
    layers = _layer_plans(rule.layers, x_layer)
    duplicates = policy_for(rule.axis.duplicates)
    var_keys = tuple(rule.axis.var_keys)
    var_columns = tuple(rule.columns.var.names)

    if isinstance(rule, LongRule):
        return LongConversion(
            obs_keys=tuple(rule.axis.obs_keys),
            var_keys=var_keys,
            obs_columns=tuple(rule.columns.obs.names),
            var_columns=var_columns,
            layers=layers,
            x_layer=x_layer,
            duplicates=duplicates,
        )
    return WideConversion(
        var_keys=var_keys,
        var_columns=var_columns,
        layers=layers,
        x_layer=x_layer,
        duplicates=duplicates,
        obs_outputs=tuple(rule.axis.obs_keys),
        declared_columns=non_sample_columns(rule.columns.var, rule.modifications),
        software_name=rule.software_name,
    )
