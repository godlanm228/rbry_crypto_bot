"""The roll-out script: which services a set of changed files restarts."""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy.py"
_spec = importlib.util.spec_from_file_location("deploy", _PATH)
deploy = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = deploy
_spec.loader.exec_module(deploy)


def test_only_the_services_whose_code_changed_restart():
    assert deploy.services_for(["web/pages.py", "README.md"]) == ["rbry-web"]
    assert deploy.services_for(["core/sandbox/urgent.py"]) == ["rbry-test-bot", "rbry-urgent"]
    assert deploy.services_for(["modules/handlers.py", "locales/strings/ru.json"]) == ["rbry-test-bot"]
    # the timers' jobs (the watcher, the paper journal, the data) start on the new code by themselves
    assert deploy.services_for(["scripts/panic_watch.py", "scripts/night.py", "deploy/rbry-panic.timer"]) == []
    assert deploy.services_for(["README.md", "tests/test_x.py", ".github/workflows/tests.yml"]) == []


def test_a_file_is_not_mistaken_for_a_folder_of_the_same_start():
    assert deploy.services_for(["webhooks.py", "config.py.bak", "core.md"]) == []
    assert deploy.services_for(["config.py"]) == ["rbry-test-bot", "rbry-urgent"]


def test_restart_modes_and_requirement_notes():
    assert deploy.services_for(["README.md"], "all") == list(deploy.SERVICES)
    assert deploy.services_for(["web/app.py"], "none") == []
    assert deploy.needs_install(["requirements.txt", "requirements-dev.txt", "docs/requirements.md"]) == [
        "requirements.txt", "requirements-dev.txt"]


def test_every_restartable_service_has_a_unit_in_the_repo():
    units = Path(__file__).resolve().parents[1] / "deploy"
    for name in deploy.SERVICES:
        assert (units / f"{name}.service").exists(), name
