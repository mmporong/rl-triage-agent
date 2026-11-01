# pyright: reportMissingImports=false
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]
import run_five_policy_recovery_with_lease_v1 as H


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def copy(repo: Path, relative: str) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / relative, target)


def commit(repo: Path, message: str) -> str:
    git(repo, "add", "--", *[str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file()])
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def test_build_and_validate_outer_host_uses_distinct_stage_and_registration_anchors(
    tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "fixture")
    for relative in set(H.SOURCE_FILES) | {"bench/run_five_policy_recovery_loop_v1.py"}:
        copy(repo, relative)
    stage = repo / "bench/protocols/first.json"
    stage.parent.mkdir(parents=True)
    stage.write_text('{"schema":"fixture-first"}\n', encoding="utf-8")
    stage_anchor = commit(repo, "stage")
    protocol = repo / "bench/protocols/outer.json"
    H.build_host_protocol({"first": {"path": stage, "source_anchor": stage_anchor}},
                          protocol, repo=repo)
    host_anchor = commit(repo, "outer")
    observed = []
    verifiers = {"first": lambda p, a, lab: observed.append((p, a, lab)),
                 "second": lambda *_: None}
    value = H.validate_host_protocol(protocol, host_anchor, phase="first", lab=repo,
                                     repo=repo, phase_verifiers=verifiers)
    assert host_anchor != stage_anchor
    assert value["stage_protocols"]["first"]["source_anchor"] == stage_anchor
    assert observed == [(stage.resolve(), stage_anchor, repo.resolve())]


def host_argv(command: str, tmp_path: Path) -> list[str]:
    common = ["--host-protocol", str(tmp_path / "host.json"),
              "--host-anchor", "b" * 40, "--host-lab", str(tmp_path)]
    if command == "run-first":
        return [*common, command, "--plan", "plan.json", "--lab", str(tmp_path),
                "--isaac-python", "python.bat", "--out", str(tmp_path / "out"),
                "--approver", H.RUN.APPROVER]
    if command == "run-second":
        return [*common, command, "--plan", "plan.json", "--transport-root", "transport",
                "--lab", str(tmp_path), "--isaac-python", "python.bat",
                "--out", str(tmp_path / "out"), "--approver", H.RUN.APPROVER]
    raise AssertionError(command)


def test_actual_run_main_parser_routes_lease_reads_through_retry_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = H.RUN.LEASE
    retry_value = {"owner": H.RUN.LEASE.OWNER, "leaseId": "retry-lease"}
    monkeypatch.setattr(H, "validate_host_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(H.RETRY, "lease_context", lambda *_a, **_k: retry_value)
    monkeypatch.setattr(original, "lease_context", lambda: pytest.fail("raw lease_context called"))
    seen = []

    def operational(plan: Path, *, lab: Path, isaac_python: Path, out: Path,
                    approver: str) -> dict[str, Any]:
        seen.append((plan, H.RUN.LEASE.lease_context(), approver))
        return {}

    monkeypatch.setattr(H.RUN, "run_first_operational", operational)
    assert H.main(host_argv("run-first", tmp_path)) == 0
    assert seen == [(Path("plan.json"), retry_value, H.RUN.APPROVER)]
    assert H.RUN.LEASE is original


def test_exception_restores_lease_module(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    original = H.RUN.LEASE
    monkeypatch.setattr(H, "validate_host_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(H.RETRY, "lease_context", lambda *_a, **_k: {})

    def failed(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        assert isinstance(H.RUN.LEASE, H.LeaseProxy)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(H.RUN, "run_first_operational", failed)
    with pytest.raises(RuntimeError, match="fixture failure"):
        H.main(host_argv("run-first", tmp_path))
    assert H.RUN.LEASE is original


@pytest.mark.parametrize("failure", [
    ValueError("공용 GPU lease owner가 다르다"),
    ValueError("GPU lease 기간·heartbeat·ID가 잘못됐다"),
])
def test_owner_and_period_fail_closed_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: ValueError
) -> None:
    original = H.RUN.LEASE
    monkeypatch.setattr(H, "validate_host_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(H.RETRY, "lease_context", lambda *_a, **_k: (_ for _ in ()).throw(failure))

    def operational(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        H.RUN.LEASE.lease_context()
        return {}

    monkeypatch.setattr(H.RUN, "run_first_operational", operational)
    with pytest.raises(ValueError) as caught:
        H.main(host_argv("run-first", tmp_path))
    assert caught.value is failure and H.RUN.LEASE is original


def test_prepare_cli_is_delegated_without_lease_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = H.RUN.LEASE
    monkeypatch.setattr(H, "validate_host_protocol", lambda *_a, **_k: {})
    seen = []
    monkeypatch.setattr(H.RUN, "prepare_second", lambda *a, **k: seen.append((a, k)) or {})
    argv = ["--host-protocol", str(tmp_path / "host.json"), "--host-anchor", "b" * 40,
            "--host-lab", str(tmp_path), "prepare-second", "--first-result", "first.json",
            "--loop-protocol", "loop.json", "--anchor", "a" * 40,
            "--lab", str(tmp_path), "--out", str(tmp_path / "out")]
    assert H.main(argv) == 0 and len(seen) == 1
    assert H.RUN.LEASE is original


def test_underlying_disk_files_remain_unchanged_during_delegation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = [ROOT / "bench/run_five_policy_recovery_loop_v1.py",
              *[ROOT / name for name in H.RETRY.SOURCE_FILES]]
    before = {path: H.sha(path) for path in frozen}
    monkeypatch.setattr(H, "validate_host_protocol", lambda *_a, **_k: {})
    monkeypatch.setattr(H.RETRY, "lease_context", lambda *_a, **_k: {
        "owner": H.RUN.LEASE.OWNER, "leaseId": "fixture"})
    monkeypatch.setattr(H.RUN, "run_first_operational", lambda *_a, **_k: {})
    H.main(host_argv("run-first", tmp_path))
    assert {path: H.sha(path) for path in frozen} == before
