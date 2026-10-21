"""The input error every vendor parameter parser raises."""

from __future__ import annotations


class ParamsError(Exception):
    """A parameter file could not be parsed for the requested software.

    Raised when the file is clearly not the expected format (e.g. a FragPipe workflow handed to the
    DIA-NN parser) or lacks the markers a parser needs. It is a clean input-error signal that
    conversion propagates and interactive callers may translate into a focused diagnostic.
    """
