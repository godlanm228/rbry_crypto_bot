"""The strategy journal: every idea is written down before it is tested, every result after.

A loss is knowledge (that does not work, and where), a small plus is knowledge
(maybe useful next to something else), and a big win is a suspect until it
holds across years and without its best trades. The journal keeps all of them,
so every report can say how many ideas were tried: the more tries, the more
one of them looks good by luck, and the higher the bar.

runtime/learning/brain/journal.jsonl, one JSON object per line:
  {"event": "idea", "id", "title", "hypothesis", "rule", "at"}
  {"event": "result", "id", "period", "numbers", "at"}
  {"event": "note", "id", "text", "at"}       e.g. "built on a misreading of the request"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "runtime" / "learning" / "brain" / "journal.jsonl"


class Journal:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append(self, entry: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**entry, "at": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False) + "\n")

    def register(self, idea_id: str, title: str, hypothesis: str, rule: str) -> bool:
        """Write the idea down; False if it already is (an idea cannot be reworded after the fact)."""
        if any(e["event"] == "idea" and e["id"] == idea_id for e in self.entries()):
            return False
        self._append({"event": "idea", "id": idea_id, "title": title, "hypothesis": hypothesis, "rule": rule})
        return True

    def result(self, idea_id: str, period: str, numbers: dict) -> None:
        if not any(e["event"] == "idea" and e["id"] == idea_id for e in self.entries()):
            raise KeyError(f"{idea_id}: register the idea before its result")
        clean = {k: (None if isinstance(v, float) and v != v else v) for k, v in numbers.items()}
        self._append({"event": "result", "id": idea_id, "period": period, "numbers": clean})

    def note(self, idea_id: str, text: str) -> None:
        """A remark on an idea after the fact; the idea and its results stay as they were."""
        self._append({"event": "note", "id": idea_id, "text": text})

    def ideas(self) -> list[dict]:
        return [e for e in self.entries() if e["event"] == "idea"]

    def tried(self) -> int:
        """Ideas with at least one result: the count every report must show."""
        return len({e["id"] for e in self.entries() if e["event"] == "result"})
