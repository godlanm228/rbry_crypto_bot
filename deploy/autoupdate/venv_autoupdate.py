#!/usr/bin/env python3
"""Weekly library updates for the bots' virtualenvs, without breaking them.

For every target in the config:

1. Build a fresh venv next to the live one with the same packages at the
   newest versions the rules allow: the project's requirements files, and never
   a new major version (2.x -> 3.0, or 0.4 -> 0.5 for 0.x packages). Packages
   versioned by date (certifi 2026.10.23) and pip itself move freely.
2. Check the candidate on a clean copy of the deployed commit: pip check, the
   test suite (a test that already failed on the old versions does not count)
   or a smoke import, and the OSV vulnerability database (only package names
   and versions are sent).
3. Point the venv symlink at the candidate, restart the service and watch it
   for five minutes: it must stay up, not restart, and log no code-level error
   it did not log before (ImportError, AttributeError, ...). Network errors
   are normal and do not count.
4. Any failure rolls back to the previous venv. The owner gets one Telegram
   message per run when something happened: what changed, or why nothing did,
   and which new major versions wait for a manual review (each one once).

Runs as root from venv-autoupdate.timer; builds and tests run as the venv
owner. Stdlib only, so it does not depend on the venvs it replaces.

    venv-autoupdate --config /etc/venv-autoupdate.json [--dry-run] [--target SLUG]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

WATCH_SECONDS = 300
FREE_TO_MOVE = {"pip"}      # tooling, versioned by year
CALVER_FROM = 2000          # certifi 2026.10.23, pytz 2026.4: no major to guard
CODE_ERRORS = re.compile(
    r"\b(ImportError|ModuleNotFoundError|AttributeError|TypeError|NameError|SyntaxError)\b")
OSV_URL = "https://api.osv.dev/v1/querybatch"
LOCK_PATH = "/run/venv-autoupdate.lock"
NO_RESULTS = "(тесты не запустились)"


class StepFailed(Exception):
    """A check said no: the reason is already in plain words for the report."""


@dataclass
class Target:
    name: str
    slug: str
    repo: Path
    venv: Path
    service: str
    user: str = "ubuntu"
    requirements: list[str] = field(default_factory=list)
    tests: Optional[str] = None
    smoke: Optional[str] = None
    log_file: Optional[Path] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Target":
        return cls(name=d["name"], slug=d["slug"], repo=Path(d["repo"]), venv=Path(d["venv"]),
                   service=d["service"], user=d.get("user", "ubuntu"),
                   requirements=list(d.get("requirements", [])), tests=d.get("tests"),
                   smoke=d.get("smoke"), log_file=Path(d["log_file"]) if d.get("log_file") else None)


@dataclass
class Outcome:
    target: str
    status: str             # same | updated | dry_run | blocked | rolled_back | skipped | failed
    changes: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    reason: str = ""
    recovered: bool = True
    majors: list[tuple[str, str, str]] = field(default_factory=list)


# -- version rules -------------------------------------------------------------

def canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def version_head(version: str) -> Optional[tuple[int, int]]:
    m = re.match(r"(\d+)(?:\.(\d+))?", version)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def upper_bound(name: str, version: str) -> Optional[str]:
    """The first version that counts as a new major, or None when there is nothing to guard."""
    head = version_head(version)
    if canon(name) in FREE_TO_MOVE or head is None or head[0] >= CALVER_FROM:
        return None
    major, minor = head
    return f"0.{minor + 1}" if major == 0 else str(major + 1)


def is_major_jump(name: str, current: str, latest: str) -> bool:
    bound, head = upper_bound(name, current), version_head(latest)
    return bound is not None and head is not None and head >= version_head(bound)


def constraints(packages: dict[str, tuple[str, str]]) -> list[str]:
    lines = []
    for key, (name, version) in sorted(packages.items()):
        if key in FREE_TO_MOVE:
            continue
        bound = upper_bound(name, version)
        lines.append(f"{name}>={version}" + (f",<{bound}" if bound else ""))
    return lines


def diff(old: dict[str, tuple[str, str]], new: dict[str, tuple[str, str]]) -> list[str]:
    """Changes in plain words, pip itself left out."""
    lines = [f"{new[k][0]} {old[k][1]} → {new[k][1]}" for k in sorted(old.keys() & new.keys())
             if old[k][1] != new[k][1] and k not in FREE_TO_MOVE]
    lines += [f"+ {new[k][0]} {new[k][1]} (новая зависимость)" for k in sorted(new.keys() - old.keys())
              if k not in FREE_TO_MOVE]
    lines += [f"− {old[k][0]} {old[k][1]} (больше не нужна)" for k in sorted(old.keys() - new.keys())
              if k not in FREE_TO_MOVE]
    return lines


# -- checks ------------------------------------------------------------------

def junit_result(path: Path) -> tuple[int, int, set[str]]:
    """(total, skipped, failed test ids) from a pytest --junitxml file."""
    if not path.exists():
        return 0, 0, {NO_RESULTS}
    total, skipped, failed = 0, 0, set()
    for case in ET.parse(path).getroot().iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(f"{case.get('classname', '')}::{case.get('name', '')}")
        elif case.find("skipped") is not None:
            skipped += 1
    return total, skipped, failed if total else {NO_RESULTS}


def _norm(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip())


def new_code_errors(before: list[str], after: list[str]) -> list[str]:
    """Code-level error lines after the restart that the old process never logged."""
    known = {_norm(line) for line in before if CODE_ERRORS.search(line)}
    return [line.strip() for line in after if CODE_ERRORS.search(line) and _norm(line) not in known]


def vulnerabilities(packages: dict[str, tuple[str, str]]) -> Optional[dict[str, str]]:
    """{vulnerability id: package} from OSV, or None when the database did not answer."""
    items = sorted(packages.values())
    body = json.dumps({"queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v}
                                   for n, v in items]}).encode()
    request = urllib.request.Request(OSV_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            results = json.load(response)["results"]
    except (OSError, ValueError, KeyError):
        return None
    return {v["id"]: name for (name, _), res in zip(items, results) for v in res.get("vulns") or []}


# -- shell ---------------------------------------------------------------------

def _tail(text: str, limit: int = 300) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "без вывода")[:limit]


def run(cmd, user: Optional[str] = None, cwd=None, timeout: int = 600) -> subprocess.CompletedProcess:
    import pwd
    cmd = [str(c) for c in cmd]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INPUT": "1"}
    me = pwd.getpwuid(os.geteuid())
    env["HOME"] = pwd.getpwnam(user).pw_dir if user else me.pw_dir
    if user and user != me.pw_name:
        cmd = ["runuser", "-u", user, "--", *cmd]
    try:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return subprocess.CompletedProcess(cmd, 124, out, f"не уложилось в {timeout} с")


def packages(python: Path, user: str) -> dict[str, tuple[str, str]]:
    r = run([python, "-m", "pip", "list", "--format=json", "--exclude-editable"], user, timeout=120)
    if r.returncode:
        raise StepFailed(f"не читается список пакетов: {_tail(r.stderr)}")
    return {canon(p["name"]): (p["name"], p["version"]) for p in json.loads(r.stdout)}


def base_python(venv: Path) -> str:
    cfg = venv / "pyvenv.cfg"
    for line in cfg.read_text().splitlines() if cfg.exists() else []:
        key, _, value = line.partition("=")
        if key.strip() == "executable" and value.strip():
            return value.strip()
    return "/usr/bin/python3"


def service_state(service: str) -> dict[str, str]:
    out = run(["systemctl", "show", service, "-p", "ActiveState", "-p", "NRestarts", "-p", "MainPID"]).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def journal(service: str, since: float, until: Optional[float] = None) -> list[str]:
    cmd = ["journalctl", "-u", service, "--no-pager", "-o", "cat", "--since", f"@{int(since)}"]
    if until:
        cmd += ["--until", f"@{int(until)}"]
    return run(cmd, timeout=120).stdout.splitlines()


def file_lines(path: Optional[Path], start: int, end: int) -> list[str]:
    if not path or not path.exists() or end <= start:
        return []
    with open(path, "rb") as f:
        f.seek(max(start, 0))
        return f.read(end - max(start, 0)).decode("utf-8", "replace").splitlines()


# -- venv generations ------------------------------------------------------------

def point(link: Path, target: Path, owner: Optional[tuple[int, int]] = None) -> None:
    """Atomically make ``link`` a symlink to ``target``."""
    tmp = link.with_name(link.name + ".swap")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    if owner:
        os.lchown(tmp, *owner)
    os.replace(tmp, link)


def swap(venv: Path, new: Path, store: Path, stamp: str,
         owner: Optional[tuple[int, int]] = None) -> tuple[Path, bool]:
    """Point the venv at ``new``; returns (previous venv, whether it was a plain directory)."""
    if venv.is_symlink():
        prev, legacy = Path(os.readlink(venv)), False
    else:
        prev, legacy = store / f"legacy-{stamp}", True
        os.rename(venv, prev)
    point(venv, new, owner)
    return prev, legacy


def restore(venv: Path, prev: Path, legacy: bool, owner: Optional[tuple[int, int]] = None) -> None:
    if legacy:
        venv.unlink()
        os.rename(prev, venv)
    else:
        point(venv, prev, owner)


def prune(store: Path, keep: set[Path]) -> None:
    keep = {k.resolve() for k in keep}
    for d in store.iterdir():
        if d.is_dir() and not d.is_symlink() and d.resolve() not in keep:
            shutil.rmtree(d, ignore_errors=True)


# -- one target --------------------------------------------------------------------

def _run_tests(python: Path, src: Path, xml: Path, tests: str, user: str) -> tuple[int, int, set[str], str]:
    r = run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", tests],
            user, cwd=src, timeout=2400)
    total, skipped, failed = junit_result(xml)
    return total, skipped, failed, _tail(r.stdout + "\n" + r.stderr)


def check_candidate(t: Target, src: Path, work: Path, old_py: Path, new_py: Path,
                    old: dict, new: dict) -> list[str]:
    """Passed checks for the report; raises StepFailed on the first one that blocks the swap."""
    passed = []
    pip_new = run([new_py, "-m", "pip", "check"], t.user)
    if pip_new.returncode == 0:
        passed.append("совместимость ✅")
    elif run([old_py, "-m", "pip", "check"], t.user).returncode == 0:
        raise StepFailed(f"новые версии конфликтуют между собой: {_tail(pip_new.stdout)}")
    else:
        passed.append("совместимость: те же замечания, что и раньше")

    if t.tests:
        total, skipped, failed, last = _run_tests(new_py, src, work / "new.xml", t.tests, t.user)
        if NO_RESULTS in failed:
            raise StepFailed(f"тесты не запустились: {last}")
        fresh = failed
        if failed:
            _, _, before, _ = _run_tests(old_py, src, work / "old.xml", t.tests, t.user)
            fresh = failed - before
        if fresh:
            names = ", ".join(sorted(fresh)[:3]) + (" …" if len(fresh) > 3 else "")
            raise StepFailed(f"с новыми версиями падают тесты ({len(fresh)}): {names}")
        line = f"тесты: {total - skipped - len(failed)} прошли ✅"
        passed.append(line + (f" ({len(failed)} падали и раньше)" if failed else ""))
    if t.smoke:
        r = run([new_py, "-c", t.smoke], t.user, cwd=src, timeout=300)
        if r.returncode:
            if run([old_py, "-c", t.smoke], t.user, cwd=src, timeout=300).returncode:
                raise StepFailed("проверка запуска не проходит и на старых версиях — надо поправить настройку")
            raise StepFailed(f"с новыми версиями код не запускается: {_tail(r.stderr)}")
        passed.append("запуск ✅")

    old_v, new_v = vulnerabilities(old), vulnerabilities(new)
    if old_v is None or new_v is None:
        passed.append("уязвимости: база OSV не ответила, проверю в следующий раз")
    else:
        introduced = sorted({new_v[i] for i in new_v.keys() - old_v.keys()})
        if introduced:
            raise StepFailed(f"в новых версиях известные уязвимости: {', '.join(introduced)}")
        fixed = {old_v[i] for i in old_v.keys() - new_v.keys()}
        left = sorted(set(new_v.values()))
        if fixed:
            passed.append(f"закрыл уязвимости: {', '.join(sorted(fixed))} ✅")
        passed.append(f"остались уязвимости без исправления в рамках правил: {', '.join(left)}" if left
                      else "известных уязвимостей нет ✅")
    return passed


def restart_and_watch(t: Target, seconds: int = WATCH_SECONDS) -> Optional[str]:
    """None when the service stayed healthy, else what went wrong in plain words."""
    offset = t.log_file.stat().st_size if t.log_file and t.log_file.exists() else 0
    started = time.time()
    before = journal(t.service, started - 86400, started) + file_lines(t.log_file, offset - 2_000_000, offset)
    r = run(["systemctl", "restart", t.service], timeout=120)
    if r.returncode:
        return f"служба не перезапустилась: {_tail(r.stderr)}"
    time.sleep(3)
    first = service_state(t.service)
    while True:
        state = service_state(t.service)
        if state.get("ActiveState") != "active":
            return "служба остановилась после перезапуска"
        if (state.get("NRestarts"), state.get("MainPID")) != (first.get("NRestarts"), first.get("MainPID")):
            return "служба падала и перезапускалась"
        if time.time() - started >= seconds:
            break
        time.sleep(10)
    size = t.log_file.stat().st_size if t.log_file and t.log_file.exists() else 0
    after = journal(t.service, started) + file_lines(t.log_file, offset if size >= offset else 0, size)
    errors = new_code_errors(before, after)
    return f"в логе новая ошибка: {errors[0][:200]}" if errors else None


def update_target(t: Target, store_root: Path, dry_run: bool) -> Outcome:
    import pwd
    out = Outcome(t.name, "same")
    if service_state(t.service).get("ActiveState") != "active":
        out.status, out.reason = "skipped", "служба не запущена — не трогаю"
        return out
    pw = pwd.getpwnam(t.user)
    owner = (pw.pw_uid, pw.pw_gid)
    store = store_root / t.slug
    run(["mkdir", "-p", store], t.user)
    work = Path(tempfile.mkdtemp(prefix=f"autoupdate-{t.slug}-", dir="/var/tmp"))
    os.chown(work, *owner)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    new = store / stamp
    try:
        src = work / "src"
        r = run(["sh", "-c", f"mkdir {shlex.quote(str(src))} && git -C {shlex.quote(str(t.repo))} archive HEAD"
                             f" | tar -x -C {shlex.quote(str(src))}"], t.user, timeout=300)
        if r.returncode:
            raise StepFailed(f"не удалось взять код из репозитория: {_tail(r.stderr)}")
        old_py, new_py = t.venv / "bin" / "python", new / "bin" / "python"
        old = packages(old_py, t.user)

        r = run([base_python(t.venv), "-m", "venv", new], t.user, timeout=300)
        if r.returncode:
            raise StepFailed(f"не создаётся новое окружение: {_tail(r.stderr)}")
        run([new_py, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "pip"], t.user)
        cfile = work / "constraints.txt"
        cfile.write_text("\n".join(constraints(old)) + "\n")
        reqs = [arg for name in t.requirements if (src / name).exists() for arg in ("-r", src / name)]
        names = [name for key, (name, _) in sorted(old.items()) if key not in FREE_TO_MOVE]
        r = run([new_py, "-m", "pip", "install", "-q", "--no-cache-dir", "-c", cfile, *reqs, *names],
                t.user, timeout=1800)
        if r.returncode:
            raise StepFailed(f"новые версии не ставятся вместе: {_tail(r.stderr)}")
        fresh = packages(new_py, t.user)
        out.changes = diff(old, fresh)
        if not out.changes:
            shutil.rmtree(new, ignore_errors=True)
            return out
        out.checks = check_candidate(t, src, work, old_py, new_py, old, fresh)
        if dry_run:
            shutil.rmtree(new, ignore_errors=True)
            out.status = "dry_run"
            return out

        prev, legacy = swap(t.venv, new, store, stamp, owner)
        try:
            problem = restart_and_watch(t)
        except Exception as e:  # never leave the bot on an unwatched venv
            problem = f"сбой проверки после перезапуска: {type(e).__name__}: {e}"
        if problem:
            restore(t.venv, prev, legacy, owner)
            run(["systemctl", "restart", t.service], timeout=120)
            time.sleep(30)
            out.status, out.reason = "rolled_back", problem
            out.recovered = service_state(t.service).get("ActiveState") == "active"
            shutil.rmtree(new, ignore_errors=True)
        else:
            out.status = "updated"
            prune(store, {new, prev})
        return out
    except StepFailed as e:
        shutil.rmtree(new, ignore_errors=True)
        out.status, out.reason = "blocked", str(e)
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def new_majors(t: Target, state: dict, remember: bool) -> list[tuple[str, str, str]]:
    """(package, ours, latest) for major versions not announced before."""
    r = run([t.venv / "bin" / "python", "-m", "pip", "list", "--outdated", "--format=json", "--exclude-editable"],
            t.user, timeout=900)
    if r.returncode:
        return []
    seen = state.setdefault(t.slug, {}).setdefault("majors", {})
    found = []
    for p in json.loads(r.stdout or "[]"):
        name, ours, latest = p["name"], p["version"], p["latest_version"]
        if is_major_jump(name, ours, latest) and seen.get(canon(name)) != latest:
            found.append((name, ours, latest))
            if remember:
                seen[canon(name)] = latest
    return found


# -- report ------------------------------------------------------------------------

def report(label: str, outcomes: list[Outcome]) -> Optional[str]:
    blocks = []
    for o in outcomes:
        changes = o.changes[:10] + ([f"…и ещё {len(o.changes) - 10}"] if len(o.changes) > 10 else [])
        listed = "\n".join(f"• {c}" for c in changes)
        if o.status in ("updated", "dry_run"):
            head = (f"🔄 {o.target}: обновил {len(o.changes)}" if o.status == "updated"
                    else f"🧪 {o.target}: можно обновить {len(o.changes)} (пробный прогон, ничего не менял)")
            tail = "\nПерезапустил — 5 минут работает без ошибок." if o.status == "updated" else ""
            blocks.append(f"{head}\n{listed}\nПроверки: {' · '.join(o.checks)}{tail}")
        elif o.status == "blocked":
            blocks.append(f"⏸ {o.target}: не обновлял — {o.reason}\nБот не трогал, работает как раньше.")
        elif o.status == "rolled_back":
            after = ("Вернул прежние версии, бот работает ✅" if o.recovered
                     else "❗ После отката служба не поднялась — нужна проверка.")
            blocks.append(f"⏪ {o.target}: откатил обновление — {o.reason}\n{after}")
        elif o.status == "failed":
            blocks.append(f"⚠️ {o.target}: автообновление сломалось — {o.reason}")
    majors = [(o.target, *m) for o in outcomes for m in o.majors]
    if majors:
        lines = "\n".join(f"• {name} {latest} (сейчас {ours}) — {target}" for target, name, ours, latest in majors)
        blocks.append(f"📦 Вышли крупные версии — сами не ставятся, проверю вручную по списку изменений:\n{lines}")
    if not blocks:
        return None
    return (f"🧰 Библиотеки · сервер {label}\n\n" + "\n\n".join(blocks))[:4000]


def read_kv(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def send(notify_conf: Path, text: str) -> bool:
    """Telegram message to the owner, with the same bot and chat as the SSH login notice."""
    try:
        conf = read_kv(notify_conf)
        env = read_kv(Path(conf["ENV_FILE"]))
        token = env.get(conf["TOKEN_VAR"], "")
        chat = env.get(conf["CHAT_VAR"], "").split(",")[0].strip()
    except (OSError, KeyError):
        return False
    if not token or not chat:
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text, "disable_web_page_preview": "true"}).encode()
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=20) as r:
            return r.status == 200
    except OSError:
        return False


def main(argv: Optional[list[str]] = None) -> int:
    import fcntl
    parser = argparse.ArgumentParser(description="Safe weekly library updates for the bot venvs")
    parser.add_argument("--config", default="/etc/venv-autoupdate.json")
    parser.add_argument("--dry-run", action="store_true", help="build and check, change nothing, print the report")
    parser.add_argument("--target", default="", help="only this slug")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another run is in progress")
        return 0
    notify = Path(config.get("notify", "/etc/ssh-login-notify.conf"))
    label = read_kv(notify).get("LABEL", "") if notify.exists() else ""
    state_path = Path(config.get("state", "/var/lib/venv-autoupdate/state.json"))
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    store_root = Path(config.get("store", "/home/ubuntu/.venvs"))

    outcomes = []
    for raw in config["targets"]:
        t = Target.from_dict(raw)
        if args.target and args.target != t.slug:
            continue
        try:
            o = update_target(t, store_root, args.dry_run)
        except Exception as e:
            o = Outcome(t.name, "failed", reason=f"{type(e).__name__}: {e}")
        if o.status != "skipped":
            o.majors = new_majors(t, state, remember=not args.dry_run)
        print(f"[{t.slug}] {o.status}: {o.reason or '; '.join(o.changes) or 'no changes'}", flush=True)
        outcomes.append(o)

    text = report(label, outcomes)
    if args.dry_run:
        print(text or "nothing to report")
        return 0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    if text and not send(notify, text):
        print("telegram report was not delivered", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
