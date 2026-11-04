import json
from pathlib import Path

import pytest

from evals import registered_recovery_resources_v1 as resources
from evals import registered_recovery_timing_v1 as recovery_timing
from evals import summarize_registered_recovery_comparison_v1 as summary
from rl_triage.investigation_session import file_sha


def _write(path: Path, value: dict | bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


def _resource(tmp_path: Path, name: str, *, status: str, training: float | None,
              fixed: float | None, checkpoint: float | None, telemetry: float | None = None) -> dict:
    root = tmp_path / "resource" / name
    admission = _write(root / "admission.json", {"request_id": name, "effect_key": "effect-" + name})
    sampler = _write(root / "sampler.json", {
        "schema": "periodic_device_sampling_stop_v1", "scope": "device_wide", "lease_matches": True,
        "energy": {"devices": {"GPU-1": {"energy_j": 12.5}}},
        "process_gpu_seconds": None, "provider_gpu_seconds": None})
    execution_refs = {}
    for field, wall in (("training", training), ("fixed", fixed), ("checkpoint", checkpoint),
                        ("telemetry", telemetry)):
        execution_refs[field] = (_write(root / f"{field}.json", {"wall_s": wall})
                                 if wall is not None else None)
    simulator = int(training is not None) + int(fixed is not None)
    gpu_wall = (training or 0.0) + (fixed or 0.0) if simulator else None
    raw = _write(root / "raw-resource.json", {
        "schema": resources.SCHEMA, "request_id": name, "effect_key": "effect-" + name,
        "status": status, "phase_wall_s": 25.0, "lease_await_s": None, "locked_wall_s": 25.0,
        "owned_gpu_s": gpu_wall, "owned_gpu_s_complete": True,
        "simulator_sdk_launches": simulator, "cpu_checkpoint_launches": int(checkpoint is not None),
        "cpu_telemetry_launches": int(telemetry is not None),
        "total_owned_children": simulator + int(checkpoint is not None) + int(telemetry is not None),
        "admission_ref": admission, "sampler_ref": sampler, "sampler_error_type": None,
        "execution_refs": execution_refs, "raw_receipt_ref": None,
        "error_type": "RuntimeError" if status == "failed" else None})
    projected = root / "projected.json"
    resources.project(raw, projected, case_id="case-05", method="fixed_sequence",
                      plan_sha256="a" * 64, request_id=name, effect_key="effect-" + name)
    body = json.loads(projected.read_text(encoding="utf-8"))
    return {**body, "receipt_ref": {"path": str(projected.resolve()), "sha256": file_sha(projected)}}


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with recovery_timing.transition(prepared, case_id="case-05", method="fixed_sequence"):
        pass
    with recovery_timing.transition(prepared, case_id="case-05", method="static_rule"):
        pass
    manifest_arms = []
    result_arms = []
    for case in summary.CASES:
        for method in summary.METHODS:
            diag = _write(tmp_path / "diagnostic" / case / method / "manifest.json",
                          {"case_id": case, "method": method})
            recovery = tmp_path / "recoveries" / case / method
            recovery.mkdir(parents=True)
            manifest_arms.append({"case_id": case, "method": method,
                                  "diagnostic_manifest": diag,
                                  "session_path": str((tmp_path / "diagnostic" / case / method).resolve()),
                                  "recovery_folder": str(recovery)})
            result_arms.append({"case_id": case, "method": method, "outcome": "hold",
                                "diagnostic_status": "unidentifiable", "recovery_started": False,
                                "resource_cost": None})
    result_arms[0] = {"case_id": "case-05", "method": "fixed_sequence", "outcome": "recovered",
                      "diagnostic_status": "confirmed", "recovery_started": True,
                      "resource_cost": _resource(tmp_path, "r1", status="complete",
                                                 training=10.0, fixed=5.0, checkpoint=1.5, telemetry=2.5)}
    result_arms[1] = {"case_id": "case-05", "method": "static_rule", "outcome": "unknown",
                      "diagnostic_status": "confirmed", "recovery_started": True,
                      "resource_cost": _resource(tmp_path, "r2", status="failed",
                                                 training=7.0, fixed=None, checkpoint=None)}
    status = {"total_arms": 16, "completed_arms": 15, "halted": False, "halt": None,
              "arms": result_arms}
    policy_path = summary.ROOT / summary.controller.REVIEW_POLICY_RELATIVE
    plan_ref = _write(tmp_path / "registered-plan.json", {
        "inputs": [{"role": "recovery_review_policy",
                    "path": summary.controller.REVIEW_POLICY_RELATIVE,
                    "sha256": summary.controller._lf_sha(policy_path)}]})
    manifest = {"arms": manifest_arms, "plan": plan_ref}
    monkeypatch.setattr(summary.controller, "status", lambda _prepared: status)
    monkeypatch.setattr(summary.controller, "_load", lambda _prepared: manifest)
    monkeypatch.setattr(summary.controller, "_verified_diagnostic_status", lambda session: {
        "status": ("confirmed" if Path(session).parent.name == "case-05"
                   and Path(session).name in {"fixed_sequence", "static_rule"} else "unidentifiable")})

    def recovery_status(folder):
        method, case = Path(folder).name, Path(folder).parent.name
        state = "done" if case == "case-05" and method in {"fixed_sequence", "static_rule"} else "pending"
        return {"requests": {"r1": {"request_id": "r1", "state": state,
                                     "receipt": {"result_sha": (None if method == "static_rule"
                                                                 else "f" * 64)}}}}

    monkeypatch.setattr(summary.investigation_recovery, "status", recovery_status)
    fixed = _write(tmp_path / "post/fixed.json", {"metrics": {"fall_rate": 0.0}})
    checkpoint = _write(tmp_path / "post/model.pt", b"checkpoint")
    monkeypatch.setattr(summary.investigation_recovery, "verified_post_observation",
                        lambda *_args: {"behavior_outcome": "normal", "fixed_report": fixed,
                                       "new_checkpoint": checkpoint})
    return prepared, status


def test_summary_preserves_complete_partial_failed_and_unknown_without_path_leak(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, _status = _fixture(tmp_path, monkeypatch)
    result = summary.summarize(prepared, tmp_path / "summary")
    fixed = next(row for row in result["arms"] if row["method"] == "fixed_sequence"
                 and row["case_id"] == "case-05")
    static = next(row for row in result["arms"] if row["method"] == "static_rule"
                  and row["case_id"] == "case-05")
    assert fixed["recovery_outcome"] == "recovered"
    assert fixed["costs"]["simulator_owned_wall_s"] == 15.0
    assert fixed["costs"]["checkpoint_cpu_owned_wall_s"] == 1.5
    assert fixed["costs"]["telemetry_cpu_owned_wall_s"] == 2.5
    assert fixed["costs"]["lease_wait_s"] is None
    assert static["costs"]["status"] == "failed" and static["costs"]["simulator_owned_wall_s"] == 7.0
    assert result["common_source_and_diagnostic_costs"] is None
    assert result["methods"][0]["outcomes"]["recovered"] == 1
    assert result["methods"][0]["outcomes"]["hold"] == 3
    assert result["timing"]["transition_count"] == 2
    assert result["timing"]["unknown_transition_count"] == 0
    assert result["timing"]["per_arm"][0]["active_wall_s"] is not None
    public_text = (tmp_path / "summary/public/summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text and '"path"' not in public_text
    assert (tmp_path / "summary/private/source-refs.json").is_file()


def test_diagnostic_only_completion_never_counts_as_recovery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, status = _fixture(tmp_path, monkeypatch)
    status["arms"][0] = {"case_id": "case-05", "method": "fixed_sequence", "outcome": "unsupported",
                         "diagnostic_status": "confirmed", "recovery_started": False,
                         "resource_cost": None}
    status["arms"][1] = {"case_id": "case-05", "method": "static_rule", "outcome": "unsupported",
                         "diagnostic_status": "confirmed", "recovery_started": False,
                         "resource_cost": None}
    monkeypatch.setattr(summary.investigation_recovery, "status", lambda _folder: {
        "requests": {"r1": {"request_id": "r1", "state": "pending"}}})
    result = summary.summarize(prepared, tmp_path / "summary")
    fixed = result["methods"][0]
    assert fixed["diagnostic_confirmed"] == 1
    assert fixed["outcomes"]["recovered"] == 0 and fixed["recovery_attempted"] == 0
    assert result["arms"][0]["endpoint"] is None
    assert result["arms"][0]["costs"]["simulator_owned_wall_s"] is None
    assert result["arms"][0]["costs"]["api_calls"] == 0


def test_summary_rejects_tampered_raw_resource_after_controller_status(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, status = _fixture(tmp_path, monkeypatch)
    raw_ref = status["arms"][0]["resource_cost"]["recovery_resource_ref"]
    Path(raw_ref["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="changed"):
        summary.summarize(prepared, tmp_path / "summary")


def test_failed_resource_and_halt_are_public_infrastructure_outcome(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, status = _fixture(tmp_path, monkeypatch)
    status["halted"] = True
    status["halt"] = {"case_id": "case-05", "method": "static_rule",
                      "reason": "infrastructure_or_integrity_failure"}
    status["arms"][1]["outcome"] = "not_started"
    status["arms"][1].pop("recovery_started")
    result = summary.summarize(prepared, tmp_path / "summary")
    arm = next(row for row in result["arms"] if row["case_id"] == "case-05"
               and row["method"] == "static_rule")
    assert arm["recovery_outcome"] == "infrastructure_failure"
    assert arm["recovery_attempted"] is True
    assert arm["costs"]["status"] == "failed"
    assert arm["costs"]["phase_elapsed_s"] == 25.0
    static = next(row for row in result["methods"] if row["method"] == "static_rule")
    assert static["recovery_attempted"] == 1
    assert static["costs"]["phase_elapsed_s"]["known_sum"] == 25.0


def test_halt_before_dispatch_has_no_attempt_or_zero_fabricated_cost(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, status = _fixture(tmp_path, monkeypatch)
    target = next(index for index, row in enumerate(status["arms"])
                  if row["case_id"] == "case-06" and row["method"] == "iterative_agent")
    status["halted"] = True
    status["halt"] = {"case_id": "case-06", "method": "iterative_agent",
                      "reason": "infrastructure_or_integrity_failure"}
    status["arms"][target] = {"case_id": "case-06", "method": "iterative_agent",
                              "outcome": "not_started", "resource_cost": None}
    result = summary.summarize(prepared, tmp_path / "summary")
    arm = result["arms"][target]
    assert arm["recovery_outcome"] == "infrastructure_failure"
    assert arm["recovery_attempted"] is False
    assert arm["costs"]["phase_elapsed_s"] is None
    assert arm["costs"]["simulator_owned_wall_s"] is None


def test_unknown_with_actual_receipt_is_reverified_not_treated_as_orphan(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, _status = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(summary.investigation_recovery, "status", lambda _folder: {
        "requests": {"r1": {"request_id": "r1", "state": "done",
                            "receipt": {"result_sha": "f" * 64}}}})
    fixed = _write(tmp_path / "unknown/fixed.json", {"metrics": {"fall_rate": None}})
    checkpoint = _write(tmp_path / "unknown/model.pt", b"checkpoint")
    calls = []

    def post(folder, _request_id):
        method = Path(folder).name
        calls.append(method)
        return {"behavior_outcome": "unknown" if method == "static_rule" else "normal",
                "fixed_report": fixed, "new_checkpoint": checkpoint}

    monkeypatch.setattr(summary.investigation_recovery, "verified_post_observation", post)
    result = summary.summarize(prepared, tmp_path / "summary")
    arm = next(row for row in result["arms"] if row["case_id"] == "case-05"
               and row["method"] == "static_rule")
    assert arm["recovery_outcome"] == "unknown" and arm["endpoint"] is None
    assert calls == ["fixed_sequence", "static_rule"]


@pytest.mark.parametrize("leak", [r"C:\\private\\artifact.json", "/home/user/artifact.json",
                                  "metadata/bench/private/secret.json"])
def test_public_guard_rejects_cross_platform_and_private_paths(leak: str) -> None:
    with pytest.raises(summary.SummaryError, match="filesystem path"):
        summary._assert_public({"sampler_metadata": leak})


def test_summary_preserves_begin_only_transition_as_unknown(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, _status = _fixture(tmp_path, monkeypatch)
    binding = {"sequence": 3, "case_id": "case-05", "method": "iterative_agent",
               "started_utc": "2026-11-04T00:00:00+00:00", "started_perf_ns": 1}
    begin = {"schema": recovery_timing.BEGIN_SCHEMA, **binding,
             "transition_id": summary.io.digest(binding)}
    path = prepared / "timing/transitions/000003-begin.json"
    path.write_text(json.dumps(begin, sort_keys=True), encoding="utf-8")
    result = summary.summarize(prepared, tmp_path / "summary")
    assert result["timing"]["unknown_transition_count"] == 1
    assert result["timing"]["active_command_wall_s"] is None
    assert result["timing"]["known_active_command_wall_s"] is not None
    assert result["timing"]["window_span_s"] is None


def test_recorded_post_review_is_reverified_but_original_cpu_wall_is_preserved(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared, status = _fixture(tmp_path, monkeypatch)
    decision = {"status": "reference_review_required", "reason_code": "post_behavior_below_reference",
                "effective_action": "reference_review"}
    stable = {"schema": "registered_recovery_post_review_v1", "decision": decision,
              "behavior_outcome": "normal", "confirmation_scope": "partial",
              "confirmation_catalog_complete": False, "unobserved_components": ["actuator_mapping"],
              "api_calls": 0, "gpu_launches": 0}
    status["arms"][0]["post_review"] = {**stable, "cpu_review_wall_s": 0.75}
    monkeypatch.setattr(summary.investigation_recovery, "review",
                        lambda *_args, **_kwargs: {**stable, "cpu_review_wall_s": 9.0})
    result = summary.summarize(prepared, tmp_path / "summary")
    review = result["arms"][0]["post_review"]
    assert review["status"] == "reference_review_required"
    assert review["confirmation_catalog_complete"] is False
    assert review["unobserved_components"] == ["actuator_mapping"]
    assert review["cpu_review_wall_s"] == 0.75
