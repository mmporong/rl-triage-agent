from __future__ import annotations

import json
from pathlib import Path

from test_investigation_session import binding, registry

from rl_triage import investigate
from rl_triage.investigation_session import write_new


def fixture_source(tmp_path: Path) -> Path:
    path = tmp_path / "fixture-source.json"
    write_new(path, {"schema": "fixture_source_v1", "case_id": "fixture-case",
                     "checkpoint_sha256": "a" * 64,
                     "facts": {"visible_metric": 1},
                     "binding": {"reference_sha256": "b" * 64,
                                 "fixture_outcomes": {"p1": "normal", "p2": "abnormal"}}})
    return path


def test_cli_protocol_runs_single_fixture_entry_flow(tmp_path: Path, capsys) -> None:
    source = fixture_source(tmp_path)
    session = tmp_path / "session"
    prepare = tmp_path / "prepare.json"
    prepare.write_text(json.dumps({"source_path": str(source), "registry": registry(),
        "model_binding": binding(), "method": "one_shot_llm", "max_checks": 2,
        "gpu_budget_per_check_s": 1}), encoding="utf-8")
    assert investigate.main(["prepare", "--session", str(session), "--backend",
                             investigate.FIXTURE_BACKEND, "--fixture", "--input", str(prepare)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "open"

    assert investigate.main(["next", "--session", str(session), "--backend",
                             investigate.FIXTURE_BACKEND, "--fixture"]) == 0
    pending = json.loads(capsys.readouterr().out)["pending"]
    approve = tmp_path / "approve.json"
    approve.write_text(json.dumps({"request_id": pending["request_id"], "actor": "fixture_actor",
                                   "authorization": {"fixture": True}}), encoding="utf-8")
    assert investigate.main(["approve", "--session", str(session), "--backend",
                             investigate.FIXTURE_BACKEND, "--fixture", "--input", str(approve)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready_to_dispatch"

    dispatch = tmp_path / "dispatch.json"
    dispatch.write_text(json.dumps({"request_id": pending["request_id"]}), encoding="utf-8")
    assert investigate.main(["dispatch", "--session", str(session), "--backend",
                             investigate.FIXTURE_BACKEND, "--fixture", "--input", str(dispatch)]) == 0
    assert json.loads(capsys.readouterr().out)["observed"] == {"p1": "normal"}

    assert investigate.main(["status", "--session", str(session)]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "one_shot_llm"
    export = tmp_path / "export-input.json"
    export.write_text(json.dumps({"path": str(tmp_path / "export.json")}), encoding="utf-8")
    assert investigate.main(["export", "--session", str(session), "--input", str(export)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"].endswith("_export")


def test_cli_blocks_unregistered_backend_and_unflagged_fixture(tmp_path: Path) -> None:
    assert investigate.run("prepare", tmp_path / "s", {}, backend_name="some.module", fixture=False) == {
        "status": "blocked_backend", "reason": "backend_name_not_registered", "completed": False,
    }
    result = investigate.run("prepare", tmp_path / "s", {},
                             backend_name=investigate.FIXTURE_BACKEND, fixture=False)
    assert result["status"] == "blocked_backend"
    assert result["reason"] == "fixture_requires_explicit_flag"


def test_cli_does_not_claim_completion_when_registered_backend_missing(tmp_path: Path, monkeypatch) -> None:
    import sys

    import rl_triage

    monkeypatch.delattr(rl_triage, "registered_diagnostic_backend", raising=False)
    monkeypatch.setitem(sys.modules, "rl_triage.registered_diagnostic_backend", None)
    result = investigate.run("prepare", tmp_path / "s", {},
                             backend_name=investigate.REGISTERED_BACKEND, fixture=False)
    assert result["status"] == "blocked_backend"
    assert result["completed"] is False
