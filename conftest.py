"""Keep the test suite away from production-relative ``data/`` paths."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path


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
