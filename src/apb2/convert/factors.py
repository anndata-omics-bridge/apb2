"""Factor-encode string-valued layer data to integer codes per the rule's categories map."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

UNKNOWN_CODE = -1


def encode_factor(
    series: pd.Series, categories: Mapping[str, int], default: int = UNKNOWN_CODE
) -> pd.Series:
    """Map string values in `series` to integer codes via `categories`.

    Values not present in `categories` (including NaN) map to `default` (-1 by default).
    Returns an int64 Series.
    """
    out = series.map(categories)
    out = out.fillna(default).astype("int64")
    return out
