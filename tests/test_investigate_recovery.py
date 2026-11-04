from pathlib import Path

import pytest

from rl_triage import investigate, investigation_recovery


def test_recovery_dispatch_routes_to_registered_consuming_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def dispatch(folder, request_id):
        calls.append((folder, request_id))
        return {"status": "running", "request_id": request_id}

    monkeypatch.setattr(investigation_recovery, "dispatch", dispatch)
    result = investigate.run("recovery-dispatch", tmp_path, {"request_id": "r-1"},
                             backend_name=investigate.REGISTERED_BACKEND, fixture=False)
    assert calls == [(tmp_path, "r-1")]
    assert result == {"status": "running", "request_id": "r-1"}


@pytest.mark.parametrize(("backend", "fixture"), [
    (investigate.FIXTURE_BACKEND, True), (investigate.REGISTERED_BACKEND, True),
    ("caller.selected.module", False),
])
def test_recovery_fixture_or_caller_backend_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, fixture: bool,
) -> None:
    monkeypatch.setattr(investigation_recovery, "dispatch",
                        lambda *args: pytest.fail("host boundary bypassed"))
    result = investigate.run("recovery-dispatch", tmp_path, {"request_id": "r-1"},
                             backend_name=backend, fixture=fixture)
    assert result["status"] == "blocked_backend"
    assert result["completed"] is False


def test_recovery_approval_preserves_separate_scope_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof = {"path": str(tmp_path / "scope.json"), "sha256": "a" * 64}
    calls = []

    def approve(folder, request_id, actor, approval_ref):
        calls.append((folder, request_id, actor, approval_ref))
        return {"status": "approved"}

    monkeypatch.setattr(investigation_recovery, "approve", approve)
    investigate.run("recovery-approve", tmp_path,
                    {"request_id": "r-1", "actor": "registered-host", "approval_ref": proof},
                    backend_name=investigate.REGISTERED_BACKEND, fixture=False)
    assert calls == [(tmp_path, "r-1", "registered-host", proof)]
    with pytest.raises(ValueError, match="input fields"):
        investigate.run("recovery-approve", tmp_path,
                        {"request_id": "r-1", "actor": "registered-host", "approval_ref": proof,
                         "approved": True},
                        backend_name=investigate.REGISTERED_BACKEND, fixture=False)


def test_recovery_settle_passes_unknown_without_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(investigation_recovery, "dispatch",
                        lambda *args: pytest.fail("settle repeated a side effect"))

    def settle(folder, request_id, receipt):
        calls.append((folder, request_id, receipt))
        return {"status": "unknown"}

    monkeypatch.setattr(investigation_recovery, "settle", settle)
    result = investigate.run("recovery-settle", tmp_path, {"request_id": "r-1", "receipt": None},
                             backend_name=investigate.REGISTERED_BACKEND, fixture=False)
    assert calls == [(tmp_path, "r-1", None)]
    assert result["status"] == "unknown"


def test_recovery_review_uses_source_bound_host_policy_without_executing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    policy = {"schema": "intervention_host_gate_source_v1",
              "receipt": {"path": str(tmp_path / "policy.json"), "sha256": "a" * 64}}
    monkeypatch.setattr(investigation_recovery, "dispatch", lambda *_args: pytest.fail("review dispatched"))

    def review(folder, request_id, host_gate_sources):
        calls.append((folder, request_id, host_gate_sources))
        return {"decision": {"effective_action": "hold", "authority_created": False}}

    monkeypatch.setattr(investigation_recovery, "review", review)
    result = investigate.run("recovery-review", tmp_path,
                             {"request_id": "r-1", "host_gate_sources": policy},
                             backend_name=investigate.REGISTERED_BACKEND, fixture=False)
    assert calls == [(tmp_path, "r-1", policy)]
    assert result["decision"]["authority_created"] is False
    with pytest.raises(ValueError, match="input fields differ"):
        investigate.run("recovery-review", tmp_path,
                        {"request_id": "r-1", "host_gate_sources": policy, "approved": True},
                        backend_name=investigate.REGISTERED_BACKEND, fixture=False)
