"""Console output of the scripts: Russian and emoji wherever it goes."""

from __future__ import annotations

import sys


def utf8_output() -> None:
    """Make stdout and stderr UTF-8, whatever the console or the file behind them expects.

    On Windows a script whose output goes to a file or a pipe writes in the ANSI code page, and the first
    emoji of a card stopped it with UnicodeEncodeError (11.11.2026: a replay of an hour by hand). The night
    sets PYTHONIOENCODING for its own steps; a run by hand had to remember it. A character the stream still
    cannot take is replaced instead of raising: a report never dies of its own text."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
