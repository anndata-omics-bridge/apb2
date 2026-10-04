"""The one error type every expected conversion failure belongs to."""

from __future__ import annotations


class ConversionError(ValueError):
    """Vendor files cannot be converted as requested: the input, its rules or its values."""
