"""Run the real diagnostic shell entry in an isolated Git repo with a fake uv."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench/reproduce_windows_clean_clone_v2.sh"


def _bash():
    if os.name == "nt":
        # System32/bash.exe would enter WSL, which this CPU regression must not do.
        path = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if not path.is_file():
            pytest.fail("Git Bash is required for this regression; WSL is not a substitute")
        return str(path)
    path = shutil.which("bash")
    assert path, "Bash is required"
    return path


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          encoding="utf-8", errors="replace")


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / "repo"
    candidate = repo / "policies/candidates/keep.yaml"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(b"committed: true\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", "policies/candidates/keep.yaml")
    _git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "fixture")
    tools = tmp_path / "bin"
    tools.mkdir()
    uv = tools / "uv"
    uv.write_text("""#!/usr/bin/env bash
printf '%s\n' "$*" >> ../calls.txt
if [[ "$*" == *'evals/prove_policies.py'* && "${GENERATE:-0}" == 1 ]]; then
  printf 'generated: true\n' > policies/candidates/keep.yaml
fi
if [[ "$*" == *'--collect-only'* && "${FAIL_COLLECT:-0}" == 1 ]]; then exit 7; fi
exit 0
""", encoding="utf-8", newline="\n")
    uv.chmod(0o755)
    return tmp_path, repo, candidate


def _run(fixture, tag="fresh", **extra):
    root, _, _ = fixture
    env = {**os.environ, **extra}
    # Keep log paths local to the fixture, regardless of the caller's settings.
    env.pop("REPRO_LOGS", None)
    bootstrap = r'''
if command -v cygpath >/dev/null; then
  root=$(cygpath -u "$1"); script=$(cygpath -u "$2")
else
  root=$1; script=$2
fi
export PATH="$root/bin:$PATH"
cd "$root/repo" || exit 98
exec bash "$script" "$3"
'''
    return subprocess.run([_bash(), "-c", bootstrap, "--", str(root), str(SCRIPT), tag],
                          check=False, env=env, capture_output=True, encoding="utf-8", errors="replace",
                          timeout=30)


@pytest.mark.parametrize("dirty", ["unstaged", "staged", "untracked"])
def test_policy_edits_are_rejected_before_commands_or_logs(fixture, dirty):
    root, repo, candidate = fixture
    if dirty == "untracked":
        candidate = candidate.with_name("new.yaml")
    candidate.write_bytes(b"user_edit: keep\r\n")
    if dirty == "staged":
        _git(repo, "add", "policies/candidates/keep.yaml")
    index_before = _git(repo, "diff", "--cached", "--binary").stdout
    result = _run(fixture)
    assert result.returncode == 2, result.stderr
    assert candidate.read_bytes() == b"user_edit: keep\r\n"
    assert _git(repo, "diff", "--cached", "--binary").stdout == index_before
    assert not (root / "calls.txt").exists()
    assert not (root / "repro-logs").exists()


def test_clean_run_keeps_generated_candidates_and_reports_all_steps(fixture):
    root, _, candidate = fixture
    result = _run(fixture, GENERATE="1")
    assert result.returncode == 0, result.stderr
    assert candidate.read_bytes() == b"generated: true\n"
    timing = (root / "repro-logs/fresh/timings.txt").read_text()
    assert len(timing.splitlines()) == 9
    assert "prove_tests exit=0" in timing
    assert "M policies/candidates/keep.yaml" in (root / "repro-logs/fresh/candidate_status.log").read_text()


def test_failed_command_is_not_masked_and_later_checks_still_run(fixture):
    root, _, _ = fixture
    result = _run(fixture, FAIL_COLLECT="1")
    assert result.returncode == 1, result.stderr
    timing = (root / "repro-logs/fresh/timings.txt").read_text()
    assert "collect exit=7" in timing
    assert "prove_tests exit=0" in timing


@pytest.mark.parametrize("tag", ["../escape", "-option", "has space", ""])
def test_invalid_tags_do_not_launch_commands(fixture, tag):
    root, _, _ = fixture
    result = _run(fixture, tag)
    assert result.returncode == 2, result.stderr
    assert not (root / "calls.txt").exists()
    assert not (root / "repro-logs").exists()


def test_existing_log_directory_is_not_overwritten(fixture):
    root, _, _ = fixture
    log = root / "repro-logs/fresh"
    log.mkdir(parents=True)
    marker = log / "timings.txt"
    marker.write_bytes(b"prior run\n")
    result = _run(fixture)
    assert result.returncode == 2, result.stderr
    assert marker.read_bytes() == b"prior run\n"
    assert not (root / "calls.txt").exists()
