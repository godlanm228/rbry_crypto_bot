"""Keep the test suite away from production-relative ``data/`` paths."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path


try:
    import pandas as _pd

    # pandas 3's copy-on-write on the pandas 2 that the laptop and the VM still run: an array taken from a frame
    # is read-only there, and a write into one passed every test here and failed on GitHub's fresh install
    # (12.11.2026, watch.closes_on). The weekly updater would have met it on the VM.
    if int(_pd.__version__.split(".")[0]) < 3:
        _pd.options.mode.copy_on_write = True
except Exception:                                          # no pandas (a bare bot install): nothing to set
    pass

_ORIGINAL_CWD = Path.cwd()
_TEST_WORKDIR = tempfile.TemporaryDirectory(
    prefix="rbry_pytest_",
    ignore_cleanup_errors=True,
)


def pytest_sessionstart(session):
    # Several legacy components intentionally use relative data paths. Running
    # the entire session from an isolated directory protects the live SQLite
    # databases even when a test imports a production singleton.
    os.chdir(_TEST_WORKDIR.name)


def pytest_sessionfinish(session, exitstatus):
    os.chdir(_ORIGINAL_CWD)
    gc.collect()
    _TEST_WORKDIR.cleanup()
