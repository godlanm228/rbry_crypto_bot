"""A logged exception says what it was: str of a timeout is empty (, "Binance klines error for
ADAUSDT:" and nothing after it), repr always names the type."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALL = re.compile(r"logger\.(?:error|warning|info)\(")
BARE_F = re.compile(r"\{(?:e|exc|err|ex)\}")                       # f"...: {e}"
BARE_PCT = re.compile(r"%s[^%\"']*[\"'](?:,.*)?,\s*(?:e|exc|err|ex)\)\s*$")   # "...: %s", exc


def test_no_log_line_prints_an_exception_by_str_alone():
    offenders = []
    for path in [ROOT / "main.py", *ROOT.glob("core/**/*.py"), *ROOT.glob("modules/**/*.py"), *ROOT.glob("scripts/*.py")]:
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CALL.search(line) and "exc_info=True" not in line and (BARE_F.search(line) or BARE_PCT.search(line)):
                offenders.append(f"{path.relative_to(ROOT)}:{no}: {line.strip()}")
    assert not offenders, "log the exception with !r / %r:\n" + "\n".join(offenders)


def test_repr_names_an_empty_timeout():
    assert repr(TimeoutError()) == "TimeoutError()"
    assert str(TimeoutError()) == ""
