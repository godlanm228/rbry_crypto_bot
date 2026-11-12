"""Every menu button, typed reply and command answers, for a free user and for the owner."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SILENT = {"say 'привет как дела'", "say 'XYZQ'"}      # chatter and unknown words get no answer, on purpose


def test_every_menu_route_answers(tmp_path):
    out = tmp_path / "routes.json"
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "menu_routes_harness.py"), str(ROOT), str(out)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    # an exception a handler swallowed still shows in the log (an import that broke in a refactor did)
    assert "Traceback" not in done.stderr, done.stderr[-2000:]
    cases = json.loads(out.read_text(encoding="utf-8"))["cases"]
    assert len(cases) > 100
    failed = [(c["user"], c["case"], c["error"]) for c in cases if c["error"]]
    assert not failed, failed
    unanswered = [(c["user"], c["case"]) for c in cases if not c["calls"] and c["case"] not in SILENT]
    assert not unanswered, unanswered
    # the owner's panel is the owner's alone
    for c in cases:
        if c["case"].startswith("press admin_") and c["user"] != 1:
            assert [call[0] for call in c["calls"]] == ["query.answer"], c
