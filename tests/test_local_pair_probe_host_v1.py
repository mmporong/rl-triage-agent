from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]
import local_pair_probe_host_v1 as H  # pyright: ignore[reportMissingImports]


def packet(runtime: bool = False) -> dict:
    facts = {f"telemetry:{role}:{metric}": float(index)
             for role in ("case", "reference")
             for index, metric in enumerate(H.C.V.TELEMETRY_FIELDS)}
    if runtime:
        for role in ("case", "reference"):
            facts.update({
                f"runtime:{role}:available": True,
                f"runtime:{role}:updates_observed": 3,
                f"runtime:{role}:expected_optimizer_steps": 20,
                f"runtime:{role}:optimizer_steps_min": 20,
                f"runtime:{role}:optimizer_steps_max": 20,
                f"runtime:{role}:expected_minibatches": 20,
                f"runtime:{role}:minibatches_min": 20,
                f"runtime:{role}:minibatches_max": 20,
            })
    bindings = {"case": {"checkpoint_sha256": "1" * 64, "telemetry_sha256": "2" * 64},
                "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64}}
    return H.C.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                              facts=facts, input_bindings=bindings, execution_budget_s=1200)


def snapshot(tmp: Path) -> dict:
    return H.C.V.build_snapshot(packet(), "public_pair", "fixed_eligible",
                                H.C.V.file_digest(H.C.V.PROTOCOL), tmp / "public.ledger.jsonl")


def measurement(probe: str, outcome: str) -> dict:
    if probe == "P_episode":
        return {"timeout_ratio": 0.1 if outcome == "abnormal" else 1.0}
    if probe == "P_noise":
        return {"noise_ratio": 4.0 if outcome == "abnormal" else 1.0}
    if probe == "P_reward":
        value = 0.1 if outcome == "abnormal" else 0.0
        return {"track_lin_vel_xy_exp_rel_error": value,
                "track_ang_vel_z_exp_rel_error": value}
    if probe == "P_torque":
        return {"low_speed_saturation": 0.1 if outcome == "abnormal" else 0.01}
    if probe == "P_physics":
        return {"mismatch_count": 1 if outcome == "abnormal" else 0}
    if probe == "P_value":
        return {"critic_change": 0.01 if outcome == "abnormal" else 1.0}
    return {"status": outcome}


def install_backend(monkeypatch: pytest.MonkeyPatch, *, fail_role: str | None = None,
                    case_outcome: str = "abnormal") -> None:
    @contextmanager
    def scope(path: Path, *_args, **_kwargs):
        previous = H.C.V.PROTOCOL, H.C.V.verify_protocol, H.C.V.BASE_FACT_IDS
        path = Path(path)
        expected = H.C.V.file_digest(path)
        H.C.V.PROTOCOL = path
        H.C.V.BASE_FACT_IDS = H.RV.BASE_FACT_IDS
        H.C.V.verify_protocol = lambda value: {} if value == expected else (_ for _ in ()).throw(
            ValueError("fixture protocol SHA"))
        try:
            yield {}
        finally:
            H.C.V.PROTOCOL, H.C.V.verify_protocol, H.C.V.BASE_FACT_IDS = previous

    def prepare(_input: Path, probe: str, *, stage: Path, role: str, **_kwargs):
        stage.mkdir(parents=True)
        body = {"schema": "fixture_obs_plan", "role": role, "probe": probe,
                "stage": str(stage), "checkpoint_sha256": "1" * 64}
        H.write_new(stage / "plan.json", body)
        return body

    def execute(plan_path: Path, authorization: Path, **_kwargs):
        plan = H.read(plan_path)
        assert H.read(authorization)["request_spec"]["probe"] == plan["probe"]
        if plan["role"] == fail_role:
            raise RuntimeError("fixture backend unknown")
        value = {"schema": "fixture_obs_receipt", "status": "complete", "role": plan["role"],
                 "probe": plan["probe"], "measurement": measurement(
                     plan["probe"], case_outcome if plan["role"] == "case" else "normal"),
                 "execution": {"wall_s": 1.0}}
        H.write_new(Path(plan["stage"]) / "probe_receipt.json", value)
        return value

    def verify(plan_path: Path, receipt_path: Path):
        plan, receipt = H.read(plan_path), H.read(receipt_path)
        if receipt.get("role") != plan.get("role") or receipt.get("probe") != plan.get("probe"):
            raise ValueError("fixture receipt binding")
        return receipt

    monkeypatch.setattr(H.RV, "view_scope", scope)
    monkeypatch.setattr(H.OBS, "prepare_probe", prepare)
    monkeypatch.setattr(H.OBS, "execute_probe", execute)
    monkeypatch.setattr(H.OBS, "verify_receipt", verify)


def approval_log() -> tuple[list[dict], object]:
    rows: list[dict] = []

    def callback(value: dict) -> dict:
        rows.append(value)
        return {"approved": True}

    return rows, callback


def protocols(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    probe, view = root / "probe.json", root / "view.json"
    probe.write_text("{}", encoding="utf-8")
    view.write_text("{}", encoding="utf-8")
    return probe, view


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_role: str | None = None):
    install_backend(monkeypatch, fail_role=fail_role)
    frozen = snapshot(tmp_path)
    choice = H.C.choose(frozen, "fixed_eligible")
    rows, callback = approval_log()
    probe, view = protocols(tmp_path)
    result = H.execute_pair(
        frozen, choice, backend_packet=packet(True), case_input=tmp_path / "case.json",
        reference_input=tmp_path / "reference.json", probe_protocol=probe,
        view_protocol=view, anchor="a" * 40, lab=tmp_path,
        out=tmp_path / "out", verify_context=callback)
    return frozen, choice, rows, result


def test_executes_and_verifies_single_pair_with_two_done_ledgers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen, choice, approvals, result = run(tmp_path, monkeypatch)
    assert [row["phase"] for row in approvals] == ["prepare", "prepare", "execute", "execute"]
    assert result["status"] == "complete" and result["retry_count"] == 0
    assert result["result_ref"]["path"].endswith("result.json")
    assert result["evidence_kind"] == "actual_runtime_observation"
    assert result["owned_child_wall_s"] == 2.0
    assert result["public_selected_by"] == choice["selected_by"]
    assert result["backend_choice_role"] == "registered_rule_adapter_not_model"
    verified = H.verify_pair(tmp_path / "out/result.json", frozen, choice)
    assert verified["result_ref"] == result["result_ref"]
    assert verified["request_id"] == result["request_id"]


def test_unknown_is_settled_once_and_holds(tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    frozen, choice, _approvals, result = run(tmp_path, monkeypatch, fail_role="case")
    assert result["status"] == "unknown_hold" and result["outcome"] == "unknown"
    assert result["outer_unknown_receipts"]["case"]["error_type"] == "RuntimeError"
    assert H.verify_pair(tmp_path / "out/result.json", frozen, choice)["status"] == "unknown_hold"


def test_rejects_wrong_backend_cp_or_stale_snapshot(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    install_backend(monkeypatch)
    frozen = snapshot(tmp_path)
    choice = H.C.choose(frozen, "fixed_eligible")
    probe, view = protocols(tmp_path)
    wrong = packet(True); wrong["input_bindings"]["case"]["checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="packet scope"):
        H.execute_pair(frozen, choice, backend_packet=wrong, case_input=tmp_path / "case",
                       reference_input=tmp_path / "ref", probe_protocol=probe,
                       view_protocol=view, anchor="a" * 40, lab=tmp_path,
                       out=tmp_path / "bad", verify_context=lambda _: {"approved": True})
    # Snapshot을 봉인한 뒤 원장에 pending request를 추가하면 stale이다.
    stale = snapshot(tmp_path / "second")
    stale_choice = H.C.choose(stale, "fixed_eligible")
    ledger = H.C.S.M.RegistryLedger(Path(stale["inputs"]["ledger_path"]), H.C.RC.registry())
    spec = H.C.request_spec(stale, stale_choice)
    ledger.propose(spec["prereg"], spec["probe"], spec["budget_gpu_s"])
    with pytest.raises(ValueError):
        H.execute_pair(stale, stale_choice, backend_packet=packet(True),
                       case_input=tmp_path / "case", reference_input=tmp_path / "ref",
                       probe_protocol=tmp_path / "p", view_protocol=tmp_path / "v",
                       anchor="a" * 40, lab=tmp_path, out=tmp_path / "stale",
                       verify_context=lambda _: {"approved": True})


def test_rejects_invalid_choice_or_denied_context(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    install_backend(monkeypatch)
    frozen = snapshot(tmp_path)
    choice = H.C.choose(frozen, "fixed_eligible")
    probe, view = protocols(tmp_path)
    changed = deepcopy(choice); changed["view_digest"] = "0" * 64
    with pytest.raises(ValueError):
        H.execute_pair(frozen, changed, backend_packet=packet(True), case_input=tmp_path / "case",
                       reference_input=tmp_path / "ref", probe_protocol=probe,
                       view_protocol=view, anchor="a" * 40, lab=tmp_path,
                       out=tmp_path / "bad", verify_context=lambda _: {"approved": True})
    with pytest.raises(ValueError, match="approval denied"):
        H.execute_pair(frozen, choice, backend_packet=packet(True), case_input=tmp_path / "case",
                       reference_input=tmp_path / "ref", probe_protocol=probe,
                       view_protocol=view, anchor="a" * 40, lab=tmp_path,
                       out=tmp_path / "denied", verify_context=lambda _: {"approved": False})


def test_verify_rejects_receipt_tamper_and_duplicate_execution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen, choice, _approvals, _result = run(tmp_path, monkeypatch)
    probe, view = protocols(tmp_path)
    receipt = tmp_path / "out/probe/case/probe_receipt.json"
    body = H.read(receipt); body["role"] = "reference"
    receipt.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA"):
        H.verify_pair(tmp_path / "out/result.json", frozen, choice)
    with pytest.raises(ValueError):
        H.execute_pair(frozen, choice, backend_packet=packet(True), case_input=tmp_path / "case",
                       reference_input=tmp_path / "ref", probe_protocol=probe,
                       view_protocol=view, anchor="a" * 40, lab=tmp_path,
                       out=tmp_path / "out", verify_context=lambda _: {"approved": True})


def test_two_sequential_pairs_share_public_ledger_and_old_prefix_stays_verifiable(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe, view = protocols(tmp_path)
    install_backend(monkeypatch, case_outcome="normal")
    first_snapshot = snapshot(tmp_path)
    first_choice = H.C.choose(first_snapshot, "fixed_eligible")
    first = H.execute_pair(
        first_snapshot, first_choice, backend_packet=packet(True),
        case_input=tmp_path / "case", reference_input=tmp_path / "reference",
        probe_protocol=probe, view_protocol=view, anchor="a" * 40, lab=tmp_path,
        out=tmp_path / "step-1", verify_context=lambda _: {"approved": True})
    assert first["probe"] == "P_episode" and first["outcome"] == "normal"

    second_snapshot = snapshot(tmp_path)
    second_choice = H.C.choose(second_snapshot, "fixed_eligible")
    assert "P_reward" in second_snapshot["view"]["eligible_probes"]
    second_choice = deepcopy(second_choice)
    second_choice["probe"] = "P_reward"
    install_backend(monkeypatch, case_outcome="abnormal")
    second = H.execute_pair(
        second_snapshot, second_choice, backend_packet=packet(True),
        case_input=tmp_path / "case", reference_input=tmp_path / "reference",
        probe_protocol=probe, view_protocol=view, anchor="a" * 40, lab=tmp_path,
        out=tmp_path / "step-2", verify_context=lambda _: {"approved": True})
    assert second["probe"] == "P_reward" and second["outcome"] == "abnormal"
    assert second["public_ledger"]["observation_index"] == 1

    # 두 번째 append 뒤에도 첫 결과가 byte-prefix proof와 observation index로 검증된다.
    assert H.verify_pair(tmp_path / "step-1/result.json", first_snapshot, first_choice)[
        "outcome"] == "normal"
    assert H.verify_pair(tmp_path / "step-2/result.json", second_snapshot, second_choice)[
        "outcome"] == "abnormal"
