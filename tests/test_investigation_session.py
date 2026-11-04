from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rl_triage.investigation_session import (
    InvestigationSession,
    digest,
    file_sha,
    reference,
    write_new,
)


def registry() -> dict:
    return {
        "mechanisms": ["m1", "m2"],
        "outcomes": ["abnormal", "normal", "unknown"],
        "probes": {
            "p1": {"measures": "first", "cost_gpu_s": 0, "args": {},
                   "expect": {"m1": "abnormal", "m2": "normal"}},
            "p2": {"measures": "second", "cost_gpu_s": 0, "args": {},
                   "expect": {"m1": "normal", "m2": "abnormal"}},
        },
        "fixed_order": ["p1", "p2"],
    }


def source(tmp_path: Path, *, fixture: bool = False) -> Path:
    path = tmp_path / "source.json"
    facts = {"visible_metric": 1}
    if fixture:
        facts["fixture_outcomes"] = {"p1": "normal", "p2": "abnormal"}
    write_new(path, {"schema": "test_source_v1", "case_id": "case-1",
                     "checkpoint_sha256": "a" * 64, "facts": facts,
                     "binding": {"reference_sha256": "b" * 64}})
    return path


def verifier(path: Path, execution_class: str = "actual") -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"status": "verified", "verifier_id": "test-host", "source_sha256": file_sha(path),
            "checkpoint_sha256": value["checkpoint_sha256"], "execution_class": execution_class,
            "case_id": value["case_id"]}


def binding() -> dict:
    return {"source_anchor": "1" * 40, "protocol_sha256": "2" * 64,
            "params": {"temperature": 0}, "api_cap": 4}


def authorize(request: dict, actor: str, authorization: dict) -> dict:
    assert authorization == {"token": "host-owned"}
    return {"status": "authorized", "authority_id": "test-host", "actor": actor,
            "request_id": request["request_id"], "effect_key": request["effect_key"]}


def raw_receipt(intent: dict, outcome: str) -> dict:
    request, prereg = intent["request"], intent["preregistration"]
    raw = {"outcome": outcome, "measurement": {"value": 1}, "gpu_s": 0.0,
           "execution_wall_s": 0.1, "exit_code": 0, "source_sha256": prereg["source_sha256"],
           "checkpoint_sha256": prereg["checkpoint_sha256"], "case_id": prereg["case_id"],
           "probe": request["probe"], "request_id": request["request_id"],
           "effect_key": request["effect_key"], "registry_sha256": prereg["registry_sha256"],
           "execution_class": prereg["execution_class"]}
    write_new(Path(intent["host_receipt_path"]), raw)
    return raw


def verify_receipt(raw: dict, intent: dict) -> dict:
    return {**raw, "receipt_ref": reference(Path(intent["host_receipt_path"]))}


def approve_current(session: InvestigationSession) -> str:
    request_id = session.status()["pending"]["request_id"]
    session.approve(request_id, actor="registered-host", authorization={"token": "host-owned"},
                    authorize=authorize, source_verifier=verifier)
    return request_id


def execute_outcome(session: InvestigationSession, outcome: str) -> None:
    request_id = approve_current(session)
    result = session.dispatch(request_id, execute=lambda intent: raw_receipt(intent, outcome),
                              verify_receipt=verify_receipt, source_verifier=verifier)
    assert result["pending"] is None


def test_one_shot_plan_binding_stays_fixed_and_calls_model_once(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    calls = []

    def complete(messages: list, _params: dict) -> dict:
        calls.append(messages)
        return {"content": json.dumps({"action": "plan", "probe": None,
                                        "probes": ["p1", "p2"],
                                        "fact_ids": ["visible_metric"]})}

    assert session.next(complete=complete, source_verifier=verifier)["pending"]["probe"] == "p1"
    execute_outcome(session, "normal")
    second = session.next(complete=complete, source_verifier=verifier)
    assert second["pending"]["probe"] == "p2"
    assert len(calls) == 1


def test_iterative_agent_second_turn_receives_verified_observation(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="iterative_agent", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    seen = []

    def complete(messages: list, _params: dict) -> dict:
        body = json.loads(messages[-1]["content"])
        seen.append(body)
        probe = "p1" if len(seen) == 1 else "p2"
        fact = "visible_metric" if len(seen) == 1 else "observation:0:p1:outcome"
        return {"content": json.dumps({"action": "run_probe", "probe": probe,
                                        "probes": [], "fact_ids": [fact]})}

    session.next(complete=complete, source_verifier=verifier)
    execute_outcome(session, "normal")
    session.next(complete=complete, source_verifier=verifier)
    assert seen[0]["observed"] == {}
    assert seen[1]["observed"] == {"p1": "normal"}
    assert seen[1]["facts"]["observation:0:p1:outcome"] == "normal"


def test_model_failure_is_terminal_hold_without_retry(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    calls = 0

    def fail(_messages: list, _params: dict) -> dict:
        nonlocal calls
        calls += 1
        raise TimeoutError("network")

    first = session.next(complete=fail, source_verifier=verifier)
    second = session.next(complete=fail, source_verifier=verifier)
    assert first["status"] == second["status"] == "model_hold"
    assert first["model"]["model_status"] == "infrastructure_error"
    assert calls == 1


def test_successful_transport_with_invalid_model_contract_is_separate_from_network_failure(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    calls = []
    def invalid(_messages: list, _params: dict) -> dict:
        calls.append(1)
        return {"status": "ok", "content": "not JSON"}
    first = session.next(complete=invalid, source_verifier=verifier)
    second = session.next(complete=invalid, source_verifier=verifier)
    assert first["model"]["model_status"] == second["model"]["model_status"] == "contract_invalid"
    assert len(calls) == 1


def test_pending_model_reservation_never_sends_again(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    assert session.next(source_verifier=verifier)["status"] == "blocked_backend"
    manifest_path = session.folder / "model" / "0000" / "manifest.json"
    decision_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bound = decision_manifest["binding"]
    write_new(session.folder / "api" / "reservations" / "0001.json", {
        "schema": "model_decision_reservation_v1", "index": 1,
        "manifest_sha256": file_sha(manifest_path),
        "binding_sha256": digest(bound), "run_id": bound["run_id"], "ordinal": 1,
        "messages_sha256": bound["initial_messages_sha256"],
        "params_sha256": decision_manifest["params_sha256"],
        "reserved_at": datetime.now(UTC).isoformat(),
    })
    calls = 0

    def complete(*_args) -> dict:
        nonlocal calls
        calls += 1
        return {}

    result = session.next(complete=complete, source_verifier=verifier)
    assert result["status"] == "hold_blocked_pending"
    assert calls == 0


def test_shared_study_api_root_enforces_cap_across_sessions(tmp_path: Path) -> None:
    shared = tmp_path / "study-api"
    model = {**binding(), "api_cap": 1, "api_root": str(shared.resolve())}
    first = InvestigationSession.prepare(source(tmp_path), tmp_path / "session-1", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=model)
    second_source = tmp_path / "second-source.json"
    write_new(second_source, {"schema": "test_source_v1", "case_id": "case-2",
        "checkpoint_sha256": "c" * 64, "facts": {"visible_metric": 2},
        "binding": {"reference_sha256": "d" * 64}})
    second = InvestigationSession.prepare(second_source, tmp_path / "session-2", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=model)
    calls = 0

    def complete(_messages: list, _params: dict) -> dict:
        nonlocal calls
        calls += 1
        return {"content": json.dumps({"action": "plan", "probe": None, "probes": ["p1"],
                                        "fact_ids": ["visible_metric"]})}

    assert first.next(complete=complete, source_verifier=verifier)["status"] == "approval_pending"
    result = second.next(complete=complete, source_verifier=verifier)
    assert result["status"] == "model_hold"
    assert result["model"]["api"]["remaining"] == 0
    assert calls == 1


def test_shared_api_root_rejects_nonlocal_and_excluded_scope(tmp_path: Path) -> None:
    source_path = source(tmp_path)
    for api_root in ("https://example.invalid/api", "C:/ri_codex2_tokenfactory_live_160/api"):
        with pytest.raises(ValueError, match="API root"):
            InvestigationSession.prepare(source_path, tmp_path / ("s-" + str(len(api_root))),
                registry=registry(), method="one_shot_llm", max_checks=1,
                gpu_budget_per_check_s=0, source_verifier=verifier,
                model_binding={**binding(), "api_root": api_root})


def test_approval_requires_registered_proof_and_consumption_is_once(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="fixed_sequence", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier)
    request_id = session.next(source_verifier=verifier)["pending"]["request_id"]
    with pytest.raises(ValueError, match="authorize"):
        session.approve(request_id, actor="caller", authorization={},
                        authorize=lambda *_: {"status": "authorized"}, source_verifier=verifier)
    session.approve(request_id, actor="registered-host", authorization={"token": "host-owned"},
                    authorize=authorize, source_verifier=verifier)
    calls = 0

    def broken(_intent: dict) -> dict:
        nonlocal calls
        calls += 1
        raise RuntimeError("host died after launch")

    assert session.dispatch(request_id, execute=broken, verify_receipt=verify_receipt,
                            source_verifier=verifier)["status"] == "pending_result"
    with pytest.raises(ValueError, match="approved unused"):
        session.dispatch(request_id, execute=broken, verify_receipt=verify_receipt,
                         source_verifier=verifier)
    assert calls == 1
    settled = session.settle(request_id, host_receipt=None, verify_receipt=verify_receipt,
                             source_verifier=verifier)
    assert settled["status"] == "hold_unknown"


def test_settle_accepts_verified_host_receipt_after_execute_exception(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="fixed_sequence", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier)
    session.next(source_verifier=verifier)
    request_id = approve_current(session)
    captured = {}

    def executed_then_crashed(intent: dict) -> dict:
        captured.update(raw_receipt(intent, "normal"))
        raise RuntimeError("process exited before returning receipt")

    result = session.dispatch(request_id, execute=executed_then_crashed,
                              verify_receipt=verify_receipt, source_verifier=verifier)
    assert result["status"] == "pending_result"
    settled = session.settle(request_id, host_receipt=captured, verify_receipt=verify_receipt,
                             source_verifier=verifier)
    assert settled["observed"] == {"p1": "normal"}


def test_receipt_and_source_tampering_are_rejected(tmp_path: Path) -> None:
    source_path = source(tmp_path)
    session = InvestigationSession.prepare(source_path, tmp_path / "session", registry=registry(),
        method="fixed_sequence", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier)
    request_id = session.next(source_verifier=verifier)["pending"]["request_id"]
    session.approve(request_id, actor="registered-host", authorization={"token": "host-owned"},
                    authorize=authorize, source_verifier=verifier)

    def mismatched(intent: dict) -> dict:
        raw = raw_receipt(intent, "normal")
        raw["checkpoint_sha256"] = "f" * 64
        return raw

    result = session.dispatch(request_id, execute=mismatched, verify_receipt=verify_receipt,
                              source_verifier=verifier)
    assert result["status"] == "pending_result"
    assert result["dispatch_error"]["reexecute_allowed"] is False

    source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA changed"):
        session.status()


def test_host_no_intervention_skips_model_and_status_is_read_only(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="one_shot_llm", max_checks=2, gpu_budget_per_check_s=1,
        source_verifier=verifier, model_binding=binding())
    calls = 0

    def complete(*_args) -> dict:
        nonlocal calls
        calls += 1
        return {}

    result = session.next(complete=complete, source_verifier=verifier,
                          observation_gate=lambda _context: {
                              "decision": "host_no_intervention", "gate_id": "u2", "reason": "healthy",
                          })
    assert result["status"] == "host_no_intervention"
    assert calls == 0
    before = sorted(str(path.relative_to(session.folder)) for path in session.folder.rglob("*"))
    assert session.status()["status"] == "host_no_intervention"
    after = sorted(str(path.relative_to(session.folder)) for path in session.folder.rglob("*"))
    assert before == after


def test_export_is_fresh_only(tmp_path: Path) -> None:
    session = InvestigationSession.prepare(source(tmp_path), tmp_path / "session", registry=registry(),
        method="fixed_sequence", max_checks=1, gpu_budget_per_check_s=1,
        source_verifier=verifier)
    output = tmp_path / "export.json"
    assert session.export(output)["schema"].endswith("_export")
    with pytest.raises(FileExistsError):
        session.export(output)


def test_hidden_label_cannot_enter_source_facts(tmp_path: Path) -> None:
    path = tmp_path / "hidden.json"
    write_new(path, {"schema": "test_source_v1", "case_id": "case-hidden",
                     "checkpoint_sha256": "a" * 64, "facts": {"fault_label": "m1"},
                     "binding": {"reference_sha256": "b" * 64}})
    with pytest.raises(ValueError, match="hidden truth"):
        InvestigationSession.prepare(path, tmp_path / "session", registry=registry(),
            method="fixed_sequence", max_checks=1, gpu_budget_per_check_s=0,
            source_verifier=verifier)
