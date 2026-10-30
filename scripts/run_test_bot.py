"""Run the owner-only test bot.

Loads ``.env.test`` (its own token, ALLOWED_USER_IDS, ALERTS_ENABLED) and
keeps all state under ``runtime/test`` so the production ``data/`` and
``logs/`` are never touched. On first run the current ``data/`` is copied in.

    .venv/Scripts/python.exe scripts/run_test_bot.py
"""

import os
import runpy
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime" / "test"
ENV_FILE = REPO / ".env.test"

if not ENV_FILE.exists():
    sys.exit(f"{ENV_FILE} not found")

RUNTIME.mkdir(parents=True, exist_ok=True)
if not (RUNTIME / "data").exists() and (REPO / "data").exists():
    shutil.copytree(
        REPO / "data", RUNTIME / "data",
        ignore=shutil.ignore_patterns("*-wal", "*-shm", "*.lock"),
    )

os.environ["RBRY_ENV_FILE"] = str(ENV_FILE)
os.environ["RBRY_PAPER_VIEW"] = "1"          # the owner's portfolio button shows the paper account (modules/paper_view.py)
os.chdir(RUNTIME)
sys.path.insert(0, str(REPO))
runpy.run_path(str(REPO / "main.py"), run_name="__main__")
