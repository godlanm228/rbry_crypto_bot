"""A file to the owner: the full night report goes after its summary, to the owner's chat only."""

import urllib.request

from core import notify


class _Answer:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_report_file_goes_to_the_owner_with_its_caption(tmp_path, monkeypatch):
    env = tmp_path / ".env.test"
    env.write_text("TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_OWNER_ID=42\n", encoding="utf-8")
    report = tmp_path / "night-20261105.md"
    report.write_text("# Ночь 05.11\nвсё по делу\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: sent.append(req) or _Answer())
    assert notify.notify_owner_file(report, "Полный отчёт ночи", env, silent=True)
    req = sent[0]
    body = req.data
    assert req.full_url.endswith("/sendDocument") and "multipart/form-data; boundary=" in req.headers["Content-type"]
    assert b'name="chat_id"\r\n\r\n42\r\n' in body and "Полный отчёт ночи".encode() in body
    assert b'filename="night-20261105.md"' in body and "всё по делу".encode() in body
    assert b'name="disable_notification"\r\n\r\ntrue' in body
    assert not notify.notify_owner_file(tmp_path / "missing.md", "", env)


def test_a_long_text_goes_in_parts_at_line_ends(tmp_path, monkeypatch):
    env = tmp_path / ".env.test"
    env.write_text("TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_OWNER_ID=42\n", encoding="utf-8")
    text = "\n".join(f"строка {i} " + "x" * 90 for i in range(100))         # ~10 000 characters
    pieces = notify.parts(text)
    assert len(pieces) == 3 and all(len(p) <= notify.PART for p in pieces) and "\n".join(pieces) == text
    assert notify.parts("y" * 8000) == ["y" * 3800, "y" * 3800, "y" * 400]
    sent = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, data, timeout: sent.append(data) or _Answer())
    assert notify.notify_owner(text, env) and len(sent) == 3
