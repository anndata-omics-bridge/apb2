"""Log where a conversion is: each step's start, then its end with duration and peak memory.

A start line is what locates a step that never returns: a process the operating system kills
for memory writes no error, so its last ``start`` line names the step it died in.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from contextlib import contextmanager
from time import perf_counter

from loguru import logger

if sys.platform != "win32":
    import resource


def _peak_rss() -> str:
    """Return the process's peak resident memory so far in GB, ``n/a`` where unmeasured."""
    if sys.platform == "win32":
        return "n/a"
    # ru_maxrss counts bytes on macOS and kibibytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 1e9:.1f}"


@contextmanager
def logged_step(step: str, **fields: object) -> Generator[None]:
    """Log one step's start and its ``done`` or ``failed`` end, even when it raises."""
    details = "".join(f" {name}={value}" for name, value in fields.items())
    logger.info("step={} start{}", step, details)
    started = perf_counter()
    outcome = "failed"
    try:
        yield
        outcome = "done"
    finally:
        logger.info(
            "step={} {}{} seconds={:.3f} peak_rss_gb={}",
            step,
            outcome,
            details,
            perf_counter() - started,
            _peak_rss(),
        )
