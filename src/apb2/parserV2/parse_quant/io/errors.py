"""Errors owned by the APB2 parsed-result I/O boundary."""

from __future__ import annotations


class ResultIOError(ValueError):
    """A path or persisted result cannot satisfy the APB2 result-I/O contract."""


class UnsupportedResultFormatError(ResultIOError):
    """A path suffix does not name one of APB2's result formats."""


class InvalidResultError(ResultIOError):
    """A result value or persisted result violates its declared format contract."""


class AnnDataLayerContractError(ResultIOError):
    """An encoded measurement is too sparse to be a usable quantitative layer."""
