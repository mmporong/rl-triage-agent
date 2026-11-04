from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

CANDIDATE_SRC = Path(__file__).parents[1] / "src"
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

import test_investigation_session as fixtures

from rl_triage.investigate_v2 import run
from rl_triage.investigation_session import InvestigationSession

v1 = import_module("rl_triage.investigate")


def _session(tmp_path: Path, method: str = "iterative_agent") -> InvestigationSession:
    registry = fixtures.registry()
    registry["mechanisms"].append("m3")
    for probe in registry["probes"].values():
        probe["expect"]["m3"] = "normal"
    return InvestigationSession.prepare(
        fixtures.source(tmp_path),
        tmp_path / "session",
        registry=registry,
        method=method,
        max_checks=2,
        gpu_budget_per_check_s=1,
        source_verifier=fixtures.verifier,
        model_binding=fixtures.binding() if method in {"one_shot_llm", "iterative_agent"} else None,
    )


def _choice(action: str = "hold", probe: str | None = None, facts: list[str] | None = None) -> dict:
    return {
        "content": json.dumps({
            "action": action,
            "probe": probe,
            "probes": [],
            "fact_ids": ["visible_metric"] if facts is None else facts,
        }),
    }


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def _status(session: InvestigationSession) -> dict:
    return run(
        "status",
        session.folder,
        {},
        backend_name=v1.REGISTERED_BACKEND,
        fixture=False,
    )


def test_open_status_reports_same_tool_ledger_status_without_writes(tmp_path: Path) -> None:
    session = _session(tmp_path)
    before = _snapshot(tmp_path)
    result = _status(session)
    assert result["status"] == result["tool_ledger_status"] == "open"
    assert _snapshot(tmp_path) == before


def test_confirmed_status_reports_same_tool_ledger_status(tmp_path: Path) -> None:
    session = _session(tmp_path, method="static_rule")
    session.next(source_verifier=fixtures.verifier)
    fixtures.execute_outcome(session, "abnormal")
    result = _status(session)
    assert result["status"] == result["tool_ledger_status"] == "confirmed"
    assert result["confirmed_mechanism"] == "m1"


def test_verified_hold_distinguishes_model_from_open_tool_ledger(tmp_path: Path) -> None:
    session = _session(tmp_path)
    calls: list[int] = []

    def complete(*_args: object) -> dict:
        calls.append(1)
        return _choice(facts=[])

    assert session.next(complete=complete, source_verifier=fixtures.verifier)["status"] == "model_hold"
    assert session.status()["status"] == "open"
    before = _snapshot(tmp_path)
    result = _status(session)
    assert result["status"] == "model_hold"
    assert result["tool_ledger_status"] == "open"
    assert result["new_model_call"] is False
    assert len(calls) == 1
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("target", ["terminal", "response", "source"])
def test_tampered_terminal_source_or_response_is_rejected(tmp_path: Path, target: str) -> None:
    session = _session(tmp_path)
    session.next(complete=lambda *_: _choice(), source_verifier=fixtures.verifier)
    if target == "terminal":
        path = session.folder / "model/0000/terminal.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["engine_result"]["model_status"] = "contract_invalid"
    elif target == "response":
        path = session.folder / "api/responses/0001.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["result"]["content"] = "{}"
    else:
        path = tmp_path / "source.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["case_id"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    before = _snapshot(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        _status(session)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "command",
    [
        "prepare", "next", "approve", "dispatch", "settle", "export",
        *v1.RECOVERY_COMMANDS,
    ],
)
def test_non_status_commands_delegate_exactly(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    session_path = Path("opaque-session")
    payload = {"opaque": object()}
    sentinel = {"sentinel": object()}
    seen: list[tuple[object, ...]] = []

    def delegated(*args: object, **kwargs: object) -> dict:
        seen.append((*args, kwargs))
        return sentinel

    monkeypatch.setattr(v1, "run", delegated)
    result = run(command, session_path, payload, backend_name="opaque-backend", fixture=True)
    assert result is sentinel
    assert seen == [(command, session_path, payload, {"backend_name": "opaque-backend", "fixture": True})]


def test_real_subprocess_cli_returns_json_and_rejection_exit_contract(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.next(complete=lambda *_: _choice(), source_verifier=fixtures.verifier)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(CANDIDATE_SRC), str(REPO / "src"), str(REPO)))
    command = [sys.executable, "-m", "rl_triage.investigate_v2", "status", "--session", str(session.folder)]
    accepted = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    assert accepted.returncode == 0, accepted.stderr
    result = json.loads(accepted.stdout)
    assert result["status"] == "model_hold" and result["tool_ledger_status"] == "open"

    terminal = session.folder / "model/0000/terminal.json"
    value = json.loads(terminal.read_text(encoding="utf-8"))
    value["engine_result"]["model_status"] = "contract_invalid"
    terminal.write_text(json.dumps(value), encoding="utf-8")
    rejected = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, check=False)
    assert rejected.returncode == 2, rejected.stderr
    error = json.loads(rejected.stdout)
    assert error["status"] == "rejected" and error["completed"] is False
