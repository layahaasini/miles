"""Tests for the per-task sandbox recipe: verifier-asset hygiene, ownership
labels, and the orphan-TTL keepalive lifecycle.

Not collected by the repo-level pytest run (testpaths = ./tests); run manually
when touching the recipe:

    pytest examples/experimental/openenv/tests/ -q
"""

import inspect
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tb2_task_sandbox as recipe  # noqa: E402


def _make_tasks_repo(root: Path, task_name: str = "some-task") -> Path:
    """A minimal tasks checkout: git repo with a GitHub origin and one task."""
    repo = root / "tb2repo"
    task = repo / task_name
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image = "debian:12"\n')
    for cmd in (
        ["git", "init", "-q"],
        ["git", "remote", "add", "origin", "https://github.com/acme/tb2tasks.git"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@e.st", "-c", "user.name=t", "commit", "-qm", "x"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return task


def test_task_layer_excludes_solution(tmp_path: Path):
    task = _make_tasks_repo(tmp_path)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=task.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    cmd = recipe._task_layer_command(task)

    # The exclusion is anchored to this one task's solution/ (dir + contents),
    # not a loose pattern that could drop task files elsewhere.
    assert f"--exclude='tb2tasks-{sha}/some-task/solution'" in cmd
    assert f"--exclude='tb2tasks-{sha}/some-task/solution/*'" in cmd
    assert cmd.rstrip().endswith(f"'tb2tasks-{sha}/some-task'")


def test_server_cmd_sets_withhold_gate():
    assert "TB2_WITHHOLD_TESTS=1" in recipe.server_cmd()


def test_server_cmd_defaults_to_the_staged_task():
    # A per-task sandbox stages one task; a reset() with no task_id must land
    # on it, not the env's built-in headless-terminal default.
    cmd = recipe.server_cmd(default_task_id="fix-git")
    assert "TB2_DEFAULT_TASK_ID=fix-git " in cmd


def test_sandbox_labels_default_to_unix_user(monkeypatch):
    monkeypatch.delenv("OPENENV_LAUNCHER", raising=False)
    monkeypatch.delenv("OPENENV_RUN_ID", raising=False)
    labels = recipe.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-tbench2-task"] == "regex-chess"
    assert labels["openenv-launcher"]  # some non-empty identity, never absent
    assert "openenv-run-id" not in labels  # omitted when unset, not ""


def test_sandbox_labels_explicit_launcher_and_run_id(monkeypatch):
    monkeypatch.setenv("OPENENV_LAUNCHER", "tao-lin")
    monkeypatch.setenv("OPENENV_RUN_ID", "tb2-grpo-0717")
    labels = recipe.sandbox_labels(Path("/opt/tb2-tasks/regex-chess"))
    assert labels["openenv-launcher"] == "tao-lin"
    assert labels["openenv-run-id"] == "tb2-grpo-0717"


def test_create_arms_ttl_by_default():
    # The dead-man's-switch contract: creates must arm auto-stop/auto-delete,
    # or a hard-killed caller's orphans run (and bill) forever.
    sig = inspect.signature(recipe.create_task_sandbox)
    assert sig.parameters["auto_stop_minutes"].default > 0
    assert sig.parameters["auto_delete_minutes"].default > 0


def test_keepalive_beats_then_exits_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(recipe, "_KEEPALIVE_INTERVAL_S", 0.02)

    class Stub:
        def __init__(self):
            self.beats = 0
            self.dead = False

        def refresh_activity(self):
            if self.dead:
                raise RuntimeError("sandbox deleted")
            self.beats += 1

    stub = Stub()
    recipe._start_keepalive(stub, "regex-chess")
    deadline = time.time() + 2.0
    while stub.beats < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert stub.beats >= 3  # beats while the sandbox is alive

    stub.dead = True  # episode over, sandbox deleted -> thread must exit
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if not any("keepalive" in t.name for t in threading.enumerate()):
            break
        time.sleep(0.01)
    assert not any("keepalive" in t.name for t in threading.enumerate())
