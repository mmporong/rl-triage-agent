from __future__ import annotations

import pytest

from evals import local_pair_input_runtime_guard_v1 as G
from evals import visual_localization_experiment_v2 as V2


def quality(outcome: str) -> dict:
    available = outcome == "available"
    return {"schema": "agentic_frame_quality_guard_v1_evidence_receipt",
            "approval_granted": False, "visual_outcome": outcome,
            "frame_refs": [{"frame_id": "f03"}] if available else [],
            "image_count": 1 if available else 0,
            "quality_receipts": [{"outcome": "available" if available
                                  else "quality_below_registered_floor"}]}


def transport(status: str) -> dict:
    outcomes = {"completed": "semantic_valid", "unknown": "semantic_invalid",
                "infra_error": "infrastructure_unknown"}
    return {"schema": "local_pair_nim_verified_call_v1", "status": status,
            "outcome": outcomes[status], "live": False,
            "call_ref": {"path": "fixture", "sha256": "a" * 64},
            "content": '{"action":"hold"}' if status == "completed" else None}


def geometry(offset: float = 0.0) -> dict:
    frame = V2.V1.FRAME_IDS[0]
    packet = {"samples": [{"expected_root_pixel": [100.0, 100.0]} for _ in V2.V1.FRAME_IDS]}
    localization = {"frame_id": frame, "visibility": "visible",
                    "bbox_1000": [(100 + offset) / 640 * 1000, 90 / 480 * 1000,
                                  (100 + offset) / 640 * 1000, 110 / 480 * 1000]}
    return V2.derive_host_decision(packet, localization)


def test_expected_faults_hold_but_programming_errors_escape(tmp_path) -> None:
    registered = tmp_path / "registered"
    registered.mkdir()
    missing_path = registered / "missing.input"
    missing = G.invoke_guarded("input", missing_path.read_bytes,
                               registered_input_roots=[registered])
    assert missing["decision"] == "HOLD" and missing["reason"] == "registered_input_unavailable"
    pending = G.invoke_guarded("transport", lambda: (_ for _ in ()).throw(RuntimeError(G.PENDING_NO_RESEND)))
    assert pending["decision"] == "HOLD" and pending["automatic_retry"] is False
    assert G.classify_exception("transport", ValueError("bug"))["disposition"] == "fatal_invalid"
    with pytest.raises(ValueError, match="bug"):
        G.invoke_guarded("transport", lambda: (_ for _ in ()).throw(ValueError("bug")))


def test_unregistered_or_filename_less_missing_is_fatal(tmp_path) -> None:
    registered = tmp_path / "registered"
    outside = tmp_path / "outside" / "missing.input"
    registered.mkdir()
    with pytest.raises(FileNotFoundError):
        G.invoke_guarded("input", outside.read_bytes, registered_input_roots=[registered])
    with pytest.raises(FileNotFoundError):
        G.invoke_guarded("input", lambda: (_ for _ in ()).throw(FileNotFoundError("no filename")),
                         registered_input_roots=[registered])
    assert G.classify_exception("input", FileNotFoundError("no filename"),
                                registered_input_roots=[registered])["disposition"] == "fatal_invalid"


def test_quality_and_geometry_unknown_or_conflict_hold() -> None:
    assert G.gate_quality(quality("unknown"))["decision"] == "HOLD"
    assert G.gate_quality(quality("available"))["decision"] == "continue"
    near = geometry()
    assert near["status"] == "consistent"
    conflict = G.gate_geometry(near, "inconsistent")
    assert conflict["decision"] == "HOLD" and conflict["reason"] == "verified_visual_numeric_conflict"
    assert G.gate_geometry(near, "consistent")["decision"] == "continue"


def test_dual_inconsistent_still_holds_quarantined_visual_evidence() -> None:
    far = geometry(100.0)
    assert far["status"] == "inconsistent" and far["gate"] == "quarantine_visual_evidence"
    result = G.gate_geometry(far, "inconsistent")
    assert result["decision"] == "HOLD"
    assert result["reason"] == "visual_evidence_quarantined"


def test_transport_and_busy_stop_before_dispatch_without_authority() -> None:
    assert G.gate_transport(transport("infra_error"))["decision"] == "HOLD"
    assert G.gate_transport(transport("unknown"))["decision"] == "HOLD"
    assert G.gate_transport(transport("completed"))["decision"] == "continue"
    calls = {"busy": 0, "dispatch": 0}

    def busy(_operation_id: str, _intent: dict) -> dict:
        calls["busy"] += 1
        return {"status": "busy", "observation_id": "fixture-busy-1"}

    result = G.pre_dispatch_gate(
        quality_receipt=quality("available"), geometry_decision=geometry(),
        numeric_status="consistent", transport_result=transport("completed"),
        busy_reader=busy, operation_id="op-1", intent={"action": "run_probe"})
    assert result["decision"] == "HOLD" and result["stage"] == "busy"
    assert result["approval_granted"] is result["dispatch_granted"] is False
    assert calls == {"busy": 1, "dispatch": 0}


def test_malformed_contracts_are_fatal_invalid_not_quiet_hold() -> None:
    with pytest.raises(G.GuardFatalInvalid, match="quality receipt"):
        G.gate_quality({"schema": "wrong"})
    with pytest.raises(G.GuardFatalInvalid, match="busy observation"):
        G.gate_busy({"status": "busy", "observation_id": "x", "extra": True})
    with pytest.raises(G.GuardFatalInvalid, match="transport contract"):
        G.gate_transport({**transport("completed"), "status": "new-status"})
