from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_brev_mailbox_service_v6 as S  # pyright: ignore[reportMissingImports]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> None:
    result = subprocess.run(["git", *args], cwd=repo, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def fixture(tmp_path: Path) -> tuple[Any, dict[str, Any], Path]:
    repo = tmp_path / "repo"
    lifecycle = repo / "bench/protocols/lifecycle.json"
    lifecycle.parent.mkdir(parents=True)
    source = repo / "source.txt"
    source.write_text("source\n", encoding="utf-8", newline="\n")
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    phase = {
        "schema": S.SCHEMA + "_lifecycle_protocol",
        "registered_at_utc": (now - timedelta(seconds=10)).isoformat(),
        "deadline_utc": (now + timedelta(hours=1)).isoformat(),
        "resource_id": "resource-1", "resource_name": "vm-1", "instance_type": "gpu-test",
        "quoted_usd_per_h": "1", "compute_cap_usd": "100", "stop_reserve_s": 30,
        "max_lifecycle_s": 7200, "budget_origin": "immutable_registered_at_utc",
        "source_commit": "fixture", "sources_lf": {"source.txt": sha(source)},
    }
    lifecycle.write_text(json.dumps(phase, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    git(repo, "init")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "fixture")
    git(repo, "add", "source.txt", "bench/protocols/lifecycle.json")
    git(repo, "commit", "-m", "fixture")
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                            text=True, check=True).stdout.strip()
    runtime = SimpleNamespace(repo=repo, anchor=anchor,
                              protocol={"lifecycle_protocol_sha256": sha(lifecycle)})
    contract = {
        "lifecycle_protocol": {"path": "bench/protocols/lifecycle.json", "sha256": sha(lifecycle)},
        "lifecycle_stop_intent_path": str(tmp_path / "stop-intent.json"),
        "deadline_utc": phase["deadline_utc"],
        "target": {"resource_id": "resource-1", "resource_name": "vm-1", "instance_type": "gpu-test"},
        "cli_argv_prefix": ["wsl", "-d", "Ubuntu", "--exec", "/home/test/.local/bin/brev"],
    }
    return runtime, contract, lifecycle


def service(runtime: Any, contract: dict[str, Any]) -> S.BrevMailboxService:
    value = object.__new__(S.BrevMailboxService)
    value.runtime = runtime
    value.contract = contract
    return value


def test_budget_resolves_committed_relative_ref_and_restores_contract(tmp_path: Path) -> None:
    runtime, contract, _ = fixture(tmp_path)
    value = service(runtime, contract)
    value._budget(1)
    assert value.contract is contract


@pytest.mark.parametrize("bad", ["C:/absolute.json", "/absolute.json", "../escape.json", "a\\b.json"])
def test_budget_rejects_non_public_relative_paths(tmp_path: Path, bad: str) -> None:
    runtime, contract, _ = fixture(tmp_path)
    contract["lifecycle_protocol"] = {"path": bad, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="repo-relative"):
        service(runtime, contract)._budget()
    assert service(runtime, contract).contract is contract


def test_budget_rejects_tamper_and_uncommitted_blob(tmp_path: Path) -> None:
    runtime, contract, lifecycle = fixture(tmp_path)
    contract["lifecycle_protocol"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw SHA"):
        service(runtime, contract)._budget()
    contract["lifecycle_protocol"]["sha256"] = sha(lifecycle)
    lifecycle.write_text(lifecycle.read_text(encoding="utf-8") + " ", encoding="utf-8", newline="\n")
    contract["lifecycle_protocol"]["sha256"] = sha(lifecycle)
    runtime.protocol["lifecycle_protocol_sha256"] = sha(lifecycle)
    with pytest.raises(ValueError, match="blob"):
        service(runtime, contract)._budget()


def test_budget_rejects_symlinked_lifecycle(tmp_path: Path) -> None:
    runtime, contract, lifecycle = fixture(tmp_path)
    target = lifecycle.with_name("target.json")
    lifecycle.replace(target)
    try:
        os.symlink(target, lifecycle)
    except OSError:
        if os.name != "nt":
            pytest.skip("symlink creation unavailable")
        target_dir = tmp_path / "junction-target"
        target_dir.mkdir()
        junction_file = target_dir / "lifecycle.json"
        junction_file.write_bytes(target.read_bytes())
        junction = runtime.repo / "linked"
        created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target_dir)],
                                 capture_output=True, text=True, check=False)
        if created.returncode != 0:
            pytest.skip("junction creation unavailable")
        contract["lifecycle_protocol"] = {"path": "linked/lifecycle.json", "sha256": sha(junction_file)}
    else:
        contract["lifecycle_protocol"]["sha256"] = sha(target)
    with pytest.raises(ValueError, match="symlink/reparse"):
        service(runtime, contract)._budget()


def test_actual_serve_rpc_budget_reaches_stubbed_first_brev_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, contract, _ = fixture(tmp_path)
    mailbox = tmp_path / "mailbox"
    operation = "a" * 64
    request = mailbox / operation / "reference/request.json"
    request.parent.mkdir(parents=True)
    registration = tmp_path / "registration.json"
    registration.write_text("{}\n", encoding="utf-8")
    contract.update({"mailbox": str(mailbox), "response_wait_s": 1,
                     "host_mount_projection": {"host": str(tmp_path), "wsl": "/mnt/test"}})
    request.write_text(json.dumps({"schema": S.V3.T.SCHEMA + "_request", "operation_id": operation,
        "phase": "reference", "contract_sha256": "c" * 64,
        "registration": {"path": str(registration), "sha256": sha(registration)}}), encoding="utf-8")

    class FixtureService(S.BrevMailboxService):
        def _registered(self, _request: dict[str, Any]):
            return ({"parent_request": {"sha256": "d" * 64}},
                    {"remote_repo": "/repo", "remote_python": "/python", "remote_request": "/request"}, {})

        def _validate_host_intent(self, _request: dict[str, Any]) -> None:
            return None

    value = object.__new__(FixtureService)
    value.runtime, value.contract = runtime, contract
    value.root, value.bindings = mailbox, {operation: json.loads(request.read_text())["registration"]}
    value.contract_ref = {"path": str(tmp_path / "contract.json"), "sha256": "c" * 64}
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        calls.append(command)
        if len(calls) == 1:
            raw = json.dumps({"workspaces": [{"id": "resource-1", "name": "vm-1",
                                               "instance_type": "gpu-test", "status": "RUNNING"}]}).encode()
            return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")
        raise RuntimeError("external transport stub boundary")

    monkeypatch.setattr(S.V3.V2.V1.subprocess, "run", fake_run)
    response = value.serve_once(request)
    assert response["status"] == "unknown" and response["reason"] == "external transport stub boundary"
    assert calls[0][-2:] == ["ls", "--json"] and "exec" in calls[1]
    assert value.contract is contract
