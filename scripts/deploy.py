"""One roll-out from the laptop: the tests, GitHub, the test VM, the services that changed, a look at them.

    python scripts/deploy.py [--dry-run] [--skip-tests] [--no-github] [--no-vm] [--restart auto|all|none]

1. main, nothing uncommitted, the whole test suite;
2. git push origin main, so GitHub never lags the VM again (09-11.11.2026: four commits lived on the VM
   only);
3. the VM has no way to GitHub (no key of ours on it), so the commits go to it over the existing ssh access:
   push to the branch deploy-incoming, fast-forward main to it there, drop the branch;
4. the long-running services whose code changed are restarted (the timers' jobs start on the new code by
   themselves); a changed requirements file is only named: the weekly updater owns the venv;
5. after --wait seconds every restarted service must be active, not restarting, and without an error in its
   journal; GitHub, the laptop and the VM must stand on one commit.
Host and key come from .env.test (RBRY_TEST_VM, RBRY_TEST_VM_KEY), as for the night. Exit code 1 when a
step fails; nothing is rolled back by itself (the previous commit is named, so a revert is one more deploy).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_DIR = "rbry_crypto_bot"
INCOMING = "deploy-incoming"
# a service and the paths its process imports: a change there needs a restart
SERVICES = {
    "rbry-test-bot": ("main.py", "config.py", "modules/", "core/", "services/", "locales/",
                      "scripts/run_test_bot.py"),
    "rbry-urgent": ("config.py", "core/sandbox/", "core/notify.py", "core/llm.py", "scripts/urgent_watch.py"),
    "rbry-web": ("web/", "core/notify.py", "scripts/web.py"),
}
ERROR_WORDS = ("Traceback", " ERROR ", " CRITICAL ")


def services_for(changed: list[str], mode: str = "auto") -> list[str]:
    """The services to restart for these changed files."""
    if mode == "none":
        return []
    if mode == "all":
        return list(SERVICES)
    return [name for name, paths in SERVICES.items()
            if any(f == p or (p.endswith("/") and f.startswith(p)) for f in changed for p in paths)]


def needs_install(changed: list[str]) -> list[str]:
    return [f for f in changed if f.startswith("requirements") and f.endswith(".txt")]


def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
    if check and done.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd[:6])}\n{(done.stdout + done.stderr)[-2000:]}")
    return done


def vm_settings() -> tuple[str, str]:
    from dotenv import dotenv_values

    env = {**dotenv_values(ROOT / ".env.test"), **os.environ}
    host, key = env.get("RBRY_TEST_VM", ""), env.get("RBRY_TEST_VM_KEY", "")
    if not host or not key or not Path(key).exists():
        raise SystemExit("RBRY_TEST_VM / RBRY_TEST_VM_KEY are not set in .env.test (or the key file is missing)")
    return host, key


def ssh(host: str, key: str, command: str, check: bool = True) -> str:
    done = run(["ssh", "-i", key, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], check=check)
    return done.stdout.strip()


def say(text: str) -> None:
    print(text, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="say what would be done, change nothing")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-github", action="store_true")
    parser.add_argument("--no-vm", action="store_true")
    parser.add_argument("--restart", choices=("auto", "all", "none"), default="auto")
    parser.add_argument("--wait", type=int, default=45, help="seconds before the restarted services are checked")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if branch != "main":
        raise SystemExit(f"on branch {branch}: the roll-out goes from main")
    dirty = run(["git", "status", "--porcelain"]).stdout.strip()
    if dirty:
        raise SystemExit("uncommitted changes, commit them first:\n" + dirty[:1500])
    say(f"main at {head[:7]}: {run(['git', 'log', '-1', '--format=%s']).stdout.strip()[:100]}")

    if not args.skip_tests:
        say("tests ...")
        done = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], check=False)
        last = (done.stdout.strip().splitlines() or ["no output"])[-1]
        if done.returncode != 0:
            raise SystemExit(f"tests failed, nothing was pushed: {last}\n{done.stdout[-3000:]}")
        say(f"tests: {last}")

    if not args.no_github:
        behind = run(["git", "ls-remote", "origin", "refs/heads/main"]).stdout.split()
        was = behind[0] if behind else ""
        if was == head:
            say("GitHub: already at this commit")
        elif args.dry_run:
            say(f"GitHub: would push {was[:7] or 'nothing'} -> {head[:7]}")
        else:
            run(["git", "push", "origin", "main"])
            say(f"GitHub: {was[:7] or 'nothing'} -> {head[:7]}")

    if args.no_vm:
        return 0
    host, key = vm_settings()
    old = ssh(host, key, f"cd {REMOTE_DIR} && git status --porcelain | head -5; git rev-parse HEAD").splitlines()
    if len(old) != 1:
        raise SystemExit("the VM's working tree has changes of its own, not touched:\n" + "\n".join(old[:-1]))
    vm_was = old[0]
    if vm_was == head:
        say("VM: already at this commit")
        return 0
    if run(["git", "merge-base", "--is-ancestor", vm_was, head], check=False).returncode != 0:
        raise SystemExit(f"the VM stands on {vm_was[:7]}, which is not an ancestor of {head[:7]}: not a fast-forward")
    changed = run(["git", "diff", "--name-only", vm_was, head]).stdout.split()
    restart = services_for(changed, args.restart)
    say(f"VM: {vm_was[:7]} -> {head[:7]}, {len(changed)} files; restart: {', '.join(restart) or 'nothing'}")
    for f in needs_install(changed):
        say(f"  note: {f} changed; a new library is installed by hand (.venv/bin/python -m pip install -r {f})")
    if args.dry_run:
        return 0

    git_ssh = {**os.environ, "GIT_SSH_COMMAND": f'ssh -i "{key}" -o BatchMode=yes'}
    run(["git", "push", f"{host}:{REMOTE_DIR}", f"main:refs/heads/{INCOMING}"], env=git_ssh)
    merged = ssh(host, key, f"cd {REMOTE_DIR} && git merge --ff-only {INCOMING} >/dev/null && "
                            f"git branch -D {INCOMING} >/dev/null && git rev-parse HEAD")
    if merged != head:
        raise SystemExit(f"the VM is at {merged[:7]} after the merge, expected {head[:7]}")
    started = int(time.time())
    if restart:
        ssh(host, key, "sudo systemctl restart " + " ".join(restart))
        say(f"restarted, looking at them in {args.wait} s ...")
        time.sleep(args.wait)
    bad = []
    for name in restart:
        state = ssh(host, key, f"systemctl is-active {name}; systemctl show {name} -p NRestarts --value", check=False)
        log = ssh(host, key, f"journalctl -u {name} --since @{started} --no-pager -o cat | tail -n 400", check=False)
        errors = [line for line in log.splitlines() if any(w in line for w in ERROR_WORDS)]
        active = state.splitlines()[0] if state else "?"
        say(f"  {name}: {active}, {len(errors)} error lines since the restart")
        if active != "active" or errors:
            bad.append(name)
            for line in errors[:5]:
                say(f"    {line[:300]}")
    failed = ssh(host, key, "systemctl --failed --no-legend | head -5", check=False)
    if failed:
        say("failed units on the VM:\n" + failed)
    github = "" if args.no_github else run(["git", "ls-remote", "origin", "refs/heads/main"]).stdout.split()[0]
    same = merged == head and (args.no_github or github == head)
    say(f"laptop {head[:7]} · VM {merged[:7]}" + ("" if args.no_github else f" · GitHub {github[:7]}")
        + (" — one commit everywhere" if same else " — NOT the same"))
    if bad or failed or not same:
        say(f"something is wrong; the commit before was {vm_was[:7]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
