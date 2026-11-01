# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]
import lineage_five_policy_driver_v1 as M
import test_local_pair_four_arm_v1 as F


def source(
    tmp: Path, *, kind: str = "F1", label: str = "eligible", frames: bool = True
) -> dict:
    original = F.source(tmp)
    packet, backend = deepcopy(original["packet"]), deepcopy(original["backend_packet"])
    packet["budget_left"] = backend["budget_left"] = 3
    providers = {}
    for name, provider in original["providers"].items():
        snapshot = M.C.V.build_snapshot(
            packet,
            "pair_preparation",
            "model_adaptive",
            M.C.V.file_digest(M.C.V.PROTOCOL),
            tmp / "pair-empty.jsonl",
        )
        context = M.A.prepare(
            snapshot,
            goal=M.BASE.P.PAIR_GOAL,
            extra_facts=provider["context"]["extra_facts"],
            decisions_left=3,
        )
        messages = M.A.messages(context)
        if name == "logs_plus_frames":
            messages[1]["content"] = [
                {"type": "text", "text": messages[1]["content"]},
                *deepcopy(provider["messages"][1]["content"][1:]),
            ]
        outgoing = M.CALL._messages(context, messages)
        providers[name] = {
            "context": context,
            "messages": outgoing,
            "params": M.CALL.params_for_messages(outgoing),
        }
    return {
        "schema": M.SOURCE_SCHEMA,
        "case_id": "case-03",
        "packet": packet,
        "backend_packet": backend,
        "providers": providers if frames else None,
        "source_binding": {"fixture_only": True},
        "index": original["prepared"],
        "qualification": {
            "arm": kind,
            "label": label,
            "diagnosis_eligible": kind != "NONE"
            and label in {"eligible", "out_of_support"},
        },
        "visual_evidence": {
            "quality_receipt": {
                "schema": "agentic_frame_quality_guard_v1_evidence_receipt",
                "approval_granted": False,
                "visual_outcome": "available",
                "frame_refs": [{}] * 6,
                "image_count": 6,
                "quality_receipts": [{"outcome": "available"}] * 6,
            }
        },
        "approval_granted": False,
        "api_authorized": False,
    }


def run(tmp: Path, data: dict, arm: str, **kwargs) -> dict:
    protocol = tmp / "fixture.json"
    M.BASE.write(protocol, {"fixture": True})
    return M.run_arm(
        data,
        arm=arm,
        run_id="test_" + arm,
        output=tmp / "result",
        protocol_path=protocol,
        execution_anchor="a" * 40,
        complete=kwargs.pop("complete", F.valid_model),
        execute_probe=kwargs.pop("execute_probe", F.fake_probe),
        verify_probe=F.verify,
        allow_fixture=True,
        **kwargs,
    )


def run_actual(
    tmp: Path,
    data: dict,
    arm: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    registered_input_roots: tuple[Path, ...],
    **kwargs,
) -> dict:
    protocol = tmp / "registered-protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    registered = {"source_digests": {data["case_id"]: M.BASE.digest(data)}}
    monkeypatch.setattr(M, "verify_protocol", lambda *_: registered)
    return M.run_arm(
        data,
        arm=arm,
        run_id="actual_" + arm,
        output=tmp / "result",
        protocol_path=protocol,
        execution_anchor="a" * 40,
        complete=kwargs.pop("complete", None),
        execute_probe=kwargs.pop("execute_probe", F.fake_probe),
        verify_probe=kwargs.pop("verify_probe", F.verify),
        busy_reader=kwargs.pop(
            "busy_reader",
            lambda *_: {"status": "available", "observation_id": "fixture-available"},
        ),
        registered_input_roots=registered_input_roots,
        **kwargs,
    )


@pytest.mark.parametrize("arm", M.ARMS)
def test_five_policies_use_real_ledger_and_keep_selector_carrier_separate(
    tmp_path, arm
):
    result = run(tmp_path, source(tmp_path / "source"), arm)
    assert result["scope"] == "cpu_fixture"
    assert len(result["steps"]) <= 3
    assert result["selector_policy"] == arm
    if arm == M.OPT.POLICY:
        assert result["host_storage_policy"] == "registry_next"
        assert result["host_storage_alias"] is True
        assert [row["probe"] for row in result["steps"]] == [
            "P_optimizer_updates",
            "P_reward",
        ]
    if arm in M.MODEL_ARMS:
        assert result["scoring"]["correct_any_family"] is True
        assert result["model_calls"] == 1
    assert result["repair"] is False and result["recovery"] is False


@pytest.mark.parametrize("arm", M.ARMS)
def test_normal_shadow_never_generates_spec_or_approval(tmp_path, monkeypatch, arm):
    data = source(tmp_path / "source", kind="NONE")

    def forbidden(*args, **kwargs):
        raise AssertionError("normal source must not request or dispatch a probe")

    monkeypatch.setattr(M.C, "request_spec", forbidden)
    result = run(tmp_path, data, arm, execute_probe=forbidden)
    assert result["steps"] == [] and result["end_state"] == "HOLD"
    assert result["model_calls"] == (1 if arm in M.MODEL_ARMS else 0)
    assert result["native_normal_intervention_proposal"] is (
        True if arm in M.MODEL_ARMS else None
    )
    assert result["confirmed_subtype"] is None


def test_unknown_transport_does_not_fall_back_or_retry(tmp_path):
    calls = []

    def pending(request):
        calls.append(request)
        raise RuntimeError(M.GUARD.PENDING_NO_RESEND)

    result = run(tmp_path, source(tmp_path / "source"), "logs_only", complete=pending)
    assert len(calls) == 1 and result["steps"] == []
    assert result["hold_reason"] == "transport_pending_no_resend"
    assert result["infra_unknown"] == 1


def test_busy_guard_stops_before_approval_and_programming_errors_escape(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("no dispatch while resource is busy")

    result = run(
        tmp_path,
        source(tmp_path / "source"),
        "fixed_eligible",
        execute_probe=forbidden,
        busy_reader=lambda *_: {
            "status": "busy",
            "observation_id": "controlled-fixture",
        },
    )
    assert result["steps"] == [] and result["hold_reason"] == "gpu_resource_busy"


def test_missing_frames_hold_only_image_arm_and_never_use_another_case(tmp_path):
    data = source(tmp_path / "source", frames=False)
    result = run(tmp_path, data, "logs_plus_frames")
    assert result["model_calls"] == 0 and result["steps"] == []
    assert result["hold_reason"] == "same_play_frames_unavailable"


def test_compound_singleton_is_partial_not_two_or_full_cases(tmp_path):
    result = run(tmp_path, source(tmp_path / "source", kind="COMPOUND"), "logs_only")
    assert result["confirmed_subtype"] == "reward"
    assert result["scoring"]["compound_partial"] is True
    assert result["scoring"]["full_cause_confirmation"] is False


def test_supported_physics_diagnosis_still_holds_unsupported_repair(tmp_path):
    data = source(tmp_path / "source", kind="F5", label="out_of_support")

    def model(request):
        text = request["messages"][1]["content"]
        view = json.loads(text)
        return {
            "status": "completed",
            "content": json.dumps(
                {
                    "action": "run_probe",
                    "target": "P_physics",
                    "fact_ids": [next(iter(view["evidence_facts"]))],
                }
            ),
        }

    def probe(snapshot, choice, *, out):
        spec = M.C.request_spec(snapshot, choice)
        ledger = M.C.S.M.RegistryLedger(
            Path(snapshot["inputs"]["ledger_path"]), M.C.RC.registry()
        )
        request = ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
        ledger.approve(request["request_id"], "fixture")
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
        ledger.receipt(request["request_id"], "abnormal", {}, 1, 0)
        M.BASE.write(
            out / "result.json", {"probe": spec["probe"], "outcome": "abnormal"}
        )
        return {"result_ref": M.BASE.ref(out / "result.json")}

    result = run(tmp_path, data, "logs_only", complete=model, execute_probe=probe)
    assert result["scoring"]["correct_any_family"] is True
    assert result["confirmed_subtype"] == "physics"
    assert (
        result["end_state"] == "HOLD" and result["hold_reason"] == "repair_unsupported"
    )
    assert result["recovery"] is False


@pytest.mark.parametrize(
    "label", ["baseline_invalid", "source_failure_absent", "not_applicable"]
)
def test_ineligible_case_preserves_exclusion_without_call_or_replacement(
    tmp_path, label
):
    result = run(tmp_path, source(tmp_path / "source", label=label), "logs_only")
    assert result["model_calls"] == 0 and result["steps"] == []
    assert result["hold_reason"] == "source_" + label


def test_new_verified_transport_schema_has_same_guard_and_rejects_unknown(tmp_path):
    value = {
        "schema": M.TRANSPORT.VERIFIED_SCHEMA,
        "status": "completed",
        "outcome": "semantic_valid",
        "live": False,
        "call_ref": {},
        "content": "{}",
    }
    assert M.GUARD.gate_transport(value)["decision"] == "continue"
    with pytest.raises(M.GUARD.GuardFatalInvalid):
        M.GUARD.gate_transport({**value, "schema": "made_up_verified"})


def test_verified_nonlive_infrastructure_is_hold_evidence_not_a_model_success(tmp_path):
    call = tmp_path / "call"
    M.BASE.write(call / "receipt.json", {"fixture": True})
    request = {"request_id": "bound"}
    response = {
        "status": "infra_error",
        "content": None,
        "usage": None,
        "wall_s": 1.0,
        "call_ref": {"path": str(call), "sha256": M.BASE.sha(call / "receipt.json")},
    }
    verified = {
        **response,
        "request": request,
        "live": False,
        "outcome": "infrastructure_unknown",
    }
    assert (
        M.verify_transport(response, request, lambda _: verified)
        == response["call_ref"]
    )
    forged = {**response, "status": "completed", "content": "{}"}
    with pytest.raises(ValueError, match="non-live"):
        M.verify_transport(
            forged,
            request,
            lambda _: {**verified, **forged, "outcome": "semantic_valid"},
        )


def test_registered_source_input_loss_holds_after_protocol_digest_before_any_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source(tmp_path / "source")
    missing = Path(data["index"]["path"])
    missing.unlink()
    calls = {"dispatch": 0, "busy": 0}

    def forbidden(*_args, **_kwargs):
        calls["dispatch"] += 1
        raise AssertionError("input HOLD must not dispatch")

    monkeypatch.setattr(
        M,
        "source_registration_key",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("registration hook must wait for checked source")
        ),
    )

    result = run_actual(
        tmp_path,
        data,
        "fixed_eligible",
        monkeypatch,
        registered_input_roots=(missing,),
        execute_probe=forbidden,
        busy_reader=lambda *_: calls.__setitem__("busy", calls["busy"] + 1),
    )
    assert result["hold_reason"] == "registered_input_unavailable"
    assert result["input_guard"]["approval_granted"] is False
    assert result["input_guard"]["dispatch_granted"] is False
    assert result["steps"] == [] and result["model_calls"] == 0
    assert result["qualification"] is result["scoring"] is None
    assert calls == {"dispatch": 0, "busy": 0}
    assert not (tmp_path / "result" / "probes.ledger.jsonl").exists()


def test_unregistered_raw_source_digest_is_fatal_before_missing_input_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source(tmp_path / "source")
    missing = Path(data["index"]["path"])
    missing.unlink()
    protocol = tmp_path / "registered-protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        M,
        "verify_protocol",
        lambda *_: {"source_digests": {data["case_id"]: "0" * 64}},
    )
    with pytest.raises(ValueError, match="source not preregistered"):
        M.run_arm(
            data,
            arm="fixed_eligible",
            run_id="digest_mismatch",
            output=tmp_path / "result",
            protocol_path=protocol,
            execution_anchor="a" * 40,
            complete=None,
            execute_probe=F.fake_probe,
            verify_probe=F.verify,
            busy_reader=lambda *_: {
                "status": "available",
                "observation_id": "fixture-available",
            },
            registered_input_roots=(missing,),
        )
    assert not (tmp_path / "result").exists()


def test_only_registered_filename_input_errors_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source(tmp_path / "source")
    registered = tmp_path / "registered-material.json"

    def permission(*_args, **_kwargs):
        raise PermissionError(13, "denied", str(registered))

    monkeypatch.setattr(M, "checked_source", permission)
    result = run_actual(
        tmp_path,
        data,
        "fixed_eligible",
        monkeypatch,
        registered_input_roots=(registered,),
    )
    assert result["hold_reason"] == "registered_input_unavailable"

    for error in (
        FileNotFoundError("filename absent"),
        ValueError("malformed registered input"),
        TypeError("programming error"),
        AssertionError("programming assertion"),
    ):
        branch = tmp_path / type(error).__name__
        branch.mkdir()
        branch_data = source(branch / "source")

        def fatal(*_args, _error=error, **_kwargs):
            raise _error

        monkeypatch.setattr(M, "checked_source", fatal)
        with pytest.raises(type(error)):
            run_actual(
                branch,
                branch_data,
                "fixed_eligible",
                monkeypatch,
                registered_input_roots=(registered,),
            )


def test_registered_host_prepare_loss_holds_before_probe_receipt_or_ledger_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source(tmp_path / "source")
    registered = tmp_path / "case-probe-input.json"
    monkeypatch.setattr(M, "checked_source", lambda value, **_: deepcopy(value))
    calls = {"host": 0}

    def missing_host(*_args, **_kwargs):
        calls["host"] += 1
        raise FileNotFoundError(2, "missing", str(registered))

    result = run_actual(
        tmp_path,
        data,
        "fixed_eligible",
        monkeypatch,
        registered_input_roots=(registered,),
        execute_probe=missing_host,
    )
    assert calls == {"host": 1}
    assert result["hold_reason"] == "registered_input_unavailable"
    assert result["input_unavailable"] == 1 and result["steps"] == []
    assert not list((tmp_path / "result").rglob("probe_receipt.json"))


def test_broad_input_roots_are_rejected_even_without_an_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = source(tmp_path / "source")
    with pytest.raises(ValueError, match="broad registered input root"):
        run_actual(
            tmp_path,
            data,
            "fixed_eligible",
            monkeypatch,
            registered_input_roots=(M.ROOT,),
        )
