import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_active_replay", ROOT / "evals/build_active_replay.py")
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
from evals import run_active_diagnosis as DRIVER


CASES = [f"case_{number}" for number in range(8)]
RANKING = ["reward", "actuator", "exploration", "optimizer", "physics", "termination"]
PROBE_ARGS = {"steps": 200, "num_envs": 256}
THRESHOLDS = {"reward_relative_error": 0.2}


def write_json(root: Path, relative: str, value) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sha_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def decision(leader: str, probe: str | None, action: str, reason: str = "측정값을 확인한다.") -> dict:
    ranking = [leader, *[item for item in RANKING if item != leader]]
    return {"ranking": ranking, "probe": probe, "action": action, "reason": reason,
            "claims": [{"id": "telemetry:return", "value": -1.0}]}


def model_stage(root: Path, case_dir: str, stage: str, value: dict) -> dict[str, str]:
    packet = write_json(root, f"{case_dir}/model/{stage}.packet.json", {"stage": stage})
    parsed = write_json(root, f"{case_dir}/model/{stage}.decision.json", value)
    call = write_json(root, f"{case_dir}/model/{stage}.attempt1.json", {
        "schema": "active_model_call_v1",
        "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
        "content": json.dumps(value, ensure_ascii=False), "infra_error": None, "exit_code": 0,
    })
    return {"packet_sha256_lf": sha_lf(packet), "model_call_sha256_lf": sha_lf(call),
            "decision_sha256_lf": sha_lf(parsed)}


def make_case(root: Path, tag: str, case: str, protocol_sha: str, *, xss=False) -> Path:
    case_dir = f"evals/results/{tag}/cases/{case}"
    before = decision("actuator", "P_reward", "continue",
                      "</script><img src=x onerror=alert(1)>" if xss else "보상 검사를 고른다.")
    after = decision("reward", None, "confirm", "receipt 값을 인용해 순위를 바꾼다.")
    first_hashes = model_stage(root, case_dir, "initial", before)
    model_stage(root, case_dir, "after_1", after)
    model_stage(root, case_dir, "without_receipt_1", before)
    prereg = {"case": case, "checkpoint_sha256": "c" * 64}
    write_json(root, f"{case_dir}/prereg.json", prereg)
    envelope = {"case": case, "checkpoint_sha256": "c" * 64,
                "protocol_sha256_lf": protocol_sha, "prior_receipt_digests": [],
                "probe": "P_reward", "args": PROBE_ARGS, "thresholds": THRESHOLDS,
                "budget_probes": 2, "reference_source": {
                    "path": "bench/public/input.json",
                    "sha256_lf": sha_lf(root / "bench/public/input.json")}, "binding": {
        "controller": "model", "protocol_sha256_lf": protocol_sha, **first_hashes,
    }}
    write_json(root, f"{case_dir}/approvals/r1.json", envelope)
    report = write_json(root, f"evals/results/{tag}/probes/{case}__P_reward.json", {
        "name": case, "checkpoint": {"sha256": "c" * 64},
        "probes": {"P_reward": {"measurement": {"ratio": 1.5}, "exit": 0}},
    })
    report_sha = sha_lf(report)
    write_json(root, f"evals/results/{tag}/execution/{case}__P_reward.json", {
        "schema": "probe_execution_v1", "case": case, "probe": "P_reward",
        "checkpoint_sha256": "c" * 64, "report_sha256_lf": report_sha,
        "gpu_wall_s": 2.5, "exit_code": 0, "parent_cpu_s": 0.5,
        "cpu_s": 3.0, "simulator_steps": 100,
    })
    write_json(root, f"evals/results/{tag}/resources/{case}__P_reward.json", {
        "schema": "probe_resources_v1", "case": case, "probe": "P_reward",
        "checkpoint_sha256": "c" * 64, "report_sha256_lf": report_sha,
        "cpu_s": 2.5, "simulator_steps": 100,
    })
    ledger_receipt = {"at": "2026-01-01T00:00:03Z", "event": "receipt", "request_id": "r1",
                      "outcome": "abnormal", "measurement": {"ratio": 1.5}, "gpu_s": 2.5,
                      "exit_code": 0, "over_budget": False}
    prereg_digest = B.canonical_digest(envelope)
    args_digest = B.canonical_digest(PROBE_ARGS)
    events = [
        {"at": "2026-01-01T00:00:00Z", "event": "proposed", "request_id": "r1",
         "prereg_digest": prereg_digest, "probe": "P_reward", "args_digest": args_digest,
         "effect_key": B.canonical_digest({"prereg": prereg_digest, "probe": "P_reward",
                                           "args": args_digest})[:16], "budget_gpu_s": 180},
        {"at": "2026-01-01T00:00:01Z", "event": "approved", "request_id": "r1", "approver": "operator"},
        {"at": "2026-01-01T00:00:02Z", "event": "consumed", "request_id": "r1"},
        ledger_receipt,
    ]
    ledger = root / case_dir / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    observed = [{**ledger_receipt, "receipt_digest": B.canonical_digest(ledger_receipt),
                 "probe": "P_reward", "checkpoint_sha256": "c" * 64,
                 "report_sha256_lf": report_sha}]
    result = {
        "schema": B.CASE_SCHEMA, "case": case, "tag": tag, "scope": "integration",
        "protocol_sha256_lf": protocol_sha, "checkpoint_sha256": "c" * 64,
        "approval_actor": "operator", "approval_identity_attested": False,
        "human_review_s": None, "gpu_device_busy_s": None, "controller": "model",
        "status": "completed", "decisions": [before, after], "observed": observed,
        "contrasts": [{"leader_changed": True, "latest_receipt_cited": True,
                       "changed_leader_rejected_by_actual_observation": True,
                       "after_leader_survives_actual_observation": True,
                       "ablation_preserves_before_leader": True}],
        "without_receipt": [before], "evidence_state": {"hypotheses": ["reward"], "dropped": []},
        "costs": {"model_calls": 3, "input_tokens": 10, "output_tokens": 4,
                  "model_wall_s": 1.25, "model_cpu_s": 0.2, "isaac_wall_s": 2.5,
                  "isaac_cpu_s": 3.0, "isaac_simulator_steps": 100,
                  "human_review_s": None, "gpu_device_busy_s": None},
    }
    return write_json(root, f"{case_dir}/result.json", result)


def make_baseline(root: Path, tag: str, case: str, controller: str, protocol_sha: str) -> None:
    case_dir = f"evals/results/{tag}/cases/{case}"
    initial = write_json(root, f"{case_dir}/initial_packet.json", {"case": case, "observed": []})
    write_json(root, f"{case_dir}/prereg.json", {"case": case, "checkpoint_sha256": "c" * 64})
    envelope = {"case": case, "checkpoint_sha256": "c" * 64,
                "protocol_sha256_lf": protocol_sha, "prior_receipt_digests": [],
                "probe": "P_reward", "args": PROBE_ARGS, "thresholds": THRESHOLDS,
                "budget_probes": 2, "reference_source": {
                    "path": "bench/public/input.json",
                    "sha256_lf": sha_lf(root / "bench/public/input.json")},
                "binding": {"controller": controller, "protocol_sha256_lf": protocol_sha,
                            "initial_packet_sha256_lf": sha_lf(initial)}}
    write_json(root, f"{case_dir}/approvals/b1.json", envelope)
    report = write_json(root, f"evals/results/{tag}/probes/{case}__P_reward.json", {
        "name": case, "checkpoint": {"sha256": "c" * 64},
        "probes": {"P_reward": {"measurement": {"ratio": 1.0}, "exit": 0}},
    })
    report_sha = sha_lf(report)
    write_json(root, f"evals/results/{tag}/execution/{case}__P_reward.json", {
        "case": case, "probe": "P_reward", "checkpoint_sha256": "c" * 64,
        "report_sha256_lf": report_sha, "gpu_wall_s": 2.0, "exit_code": 0,
        "parent_cpu_s": 0.25, "cpu_s": 1.5, "simulator_steps": 100,
    })
    write_json(root, f"evals/results/{tag}/resources/{case}__P_reward.json", {
        "case": case, "probe": "P_reward", "checkpoint_sha256": "c" * 64,
        "report_sha256_lf": report_sha, "cpu_s": 1.25, "simulator_steps": 100,
    })
    canonical = {"at": "2026-01-01T00:00:03Z", "event": "receipt", "request_id": "b1",
                 "outcome": "normal", "measurement": {"ratio": 1.0}, "gpu_s": 2.0,
                 "exit_code": 0, "over_budget": False}
    prereg_digest = B.canonical_digest(envelope)
    args_digest = B.canonical_digest(PROBE_ARGS)
    events = [
        {"at": "2026-01-01T00:00:00Z", "event": "proposed", "request_id": "b1",
         "prereg_digest": prereg_digest, "probe": "P_reward", "args_digest": args_digest,
         "effect_key": B.canonical_digest({"prereg": prereg_digest, "probe": "P_reward",
                                           "args": args_digest})[:16], "budget_gpu_s": 180},
        {"at": "2026-01-01T00:00:01Z", "event": "approved", "request_id": "b1", "approver": "operator"},
        {"at": "2026-01-01T00:00:02Z", "event": "consumed", "request_id": "b1"}, canonical,
    ]
    ledger = root / case_dir / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
    observed = [{**canonical, "receipt_digest": B.canonical_digest(canonical), "probe": "P_reward",
                 "checkpoint_sha256": "c" * 64, "report_sha256_lf": report_sha}]
    write_json(root, f"{case_dir}/result.json", {
        "schema": B.CASE_SCHEMA, "case": case, "tag": tag, "controller": controller,
        "protocol_sha256_lf": protocol_sha, "checkpoint_sha256": "c" * 64,
        "approval_actor": "operator", "approval_identity_attested": False,
        "human_review_s": None, "gpu_device_busy_s": None,
        "status": "completed", "decisions": [], "observed": observed,
        "costs": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                  "model_wall_s": 0, "model_cpu_s": 0, "isaac_wall_s": 2.0,
                  "isaac_cpu_s": 1.5, "isaac_simulator_steps": 100,
                  "human_review_s": None, "gpu_device_busy_s": None},
    })


@pytest.fixture()
def public_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    source = write_json(tmp_path, "bench/public/input.json", {"public": True})
    frozen_code = {}
    for relative in DRIVER.CODE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {relative}\n", encoding="utf-8")
        frozen_code[relative] = sha_lf(path)
    registered_source = {"path": "bench/public/input.json", "sha256_lf": sha_lf(source)}
    protocol_path = write_json(tmp_path, B.PROTOCOL.as_posix(), {
        "schema": "active_diagnosis_protocol_v1", "scope": "integration",
        "approver": "operator", "probe_arguments": {"P_reward": PROBE_ARGS},
        "thresholds": THRESHOLDS, "max_probes": 2, "probe_execution_budget_s": 180,
        "code_sha256_lf": frozen_code,
        "cases": [{"case": case, "sources": {"telemetry": registered_source,
                                                "probe_reference": registered_source}} for case in CASES],
    })
    protocol_sha = sha_lf(protocol_path)
    results = {}
    for index, case in enumerate(CASES):
        result = make_case(tmp_path, "run1", case, protocol_sha, xss=index == 0)
        results[case] = sha_lf(result)
        make_baseline(tmp_path, "run1_fixed", case, "fixed_order", protocol_sha)
        make_baseline(tmp_path, "run1_cheap", case, "cheapest_discrimination", protocol_sha)
    audit = write_json(tmp_path, "evals/results/run1/audit.json", {
        "schema": "active_diagnosis_audit_v1", "tag": "run1", "protocol_sha256_lf": protocol_sha,
        "cases": [{"case": case, "valid": index != 2, "proof_changed": index != 2,
                   "errors": [] if index != 2 else ["receipt mismatch"],
                   "verified_stage_indices": [1] if index != 2 else [],
                   "result_sha256_lf": results[case]} for index, case in enumerate(CASES)],
    })
    return tmp_path, audit


def test_builds_eight_case_offline_snapshot_without_fabricating_nulls(public_snapshot, tmp_path):
    root, audit = public_snapshot
    summary = B.build_summary("run1", audit)
    assert len(summary["cases"]) == 8
    assert summary["spotlight_case"] == CASES[0]
    assert summary["cases"][2]["state"] == "hold"
    assert summary["cases"][0]["result"]["costs"]["gpu_device_busy_s"] is None
    assert summary["cases"][0]["baselines"]["fixed_order"]["state"] == "verified"
    assert summary["cases"][0]["baselines"]["cheapest_discrimination"]["state"] == "verified"
    html = B.render_html(summary)
    assert "</script><img" not in html
    assert "\\u003c/script\\u003e" in html
    assert "innerHTML" not in html and "eval(" not in html
    assert "textContent" in html
    out = tmp_path / "artifact" / "active.html"
    html_path, json_path = B.write_outputs(summary, out)
    assert html_path.is_file() and json_path.is_file()
    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored["cases"][0]["result"]["human_review_s"] is None
    assert all("path" in source and "sha256_lf" in source for source in stored["sources"])


def test_missing_independent_audit_keeps_claimed_success_on_hold(public_snapshot):
    summary = B.build_summary("run1")
    assert not summary["spotlight_verified"]
    assert all(row["state"] == "hold" for row in summary["cases"])
    assert any("auditor" in reason for reason in summary["cases"][0]["reasons"])


def test_missing_actual_report_overrides_claimed_audit_pass(public_snapshot):
    root, audit = public_snapshot
    (root / f"evals/results/run1/probes/{CASES[0]}__P_reward.json").unlink()
    summary = B.build_summary("run1", audit)
    row = summary["cases"][0]
    assert row["state"] == "hold" and not row["proof"]
    assert any("공개 산출물이 없다" in reason for reason in row["reasons"])


def test_baseline_extra_request_and_fabricated_cost_fail_closed(public_snapshot):
    root, audit = public_snapshot
    case = CASES[0]
    ledger = root / f"evals/results/run1_fixed/cases/{case}/ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "proposed", "request_id": "orphan"}) + "\n")
    result_path = root / f"evals/results/run1_cheap/cases/{case}/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["costs"]["isaac_cpu_s"] = 1.25
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary = B.build_summary("run1", audit)
    baselines = summary["cases"][0]["baselines"]
    assert baselines["fixed_order"]["state"] == "hold"
    assert any("전체 요청 집합" in reason for reason in baselines["fixed_order"]["reasons"])
    assert baselines["cheapest_discrimination"]["state"] == "hold"
    assert any("isaac_cpu_s" in reason for reason in baselines["cheapest_discrimination"]["reasons"])


def test_baseline_protocol_budget_and_approver_are_canonical(public_snapshot):
    root, audit = public_snapshot
    case = CASES[0]
    tag = "run1_fixed"
    ledger_path = root / f"evals/results/{tag}/cases/{case}/ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["budget_gpu_s"] = 999
    rows[1]["approver"] = "different-operator"
    result_path = root / f"evals/results/{tag}/cases/{case}/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["approval_actor"] = "different-operator"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    baseline = B.build_summary("run1", audit)["cases"][0]["baselines"]["fixed_order"]
    assert baseline["state"] == "hold"
    assert any("실행 예산" in reason for reason in baseline["reasons"])
    assert any("승인 주체" in reason for reason in baseline["reasons"])


def test_unmeasured_baseline_cost_stays_null_and_is_not_verified(public_snapshot):
    root, audit = public_snapshot
    case = CASES[0]
    tag = "run1_fixed"
    execution_path = root / f"evals/results/{tag}/execution/{case}__P_reward.json"
    resources_path = root / f"evals/results/{tag}/resources/{case}__P_reward.json"
    result_path = root / f"evals/results/{tag}/cases/{case}/result.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    resources = json.loads(resources_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    execution["parent_cpu_s"] = execution["cpu_s"] = resources["cpu_s"] = None
    result["costs"]["isaac_cpu_s"] = None
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    resources_path.write_text(json.dumps(resources), encoding="utf-8")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    summary = B.build_summary("run1", audit)
    baseline = summary["cases"][0]["baselines"]["fixed_order"]
    assert baseline["state"] == "hold"
    assert baseline["result"]["costs"]["isaac_cpu_s"] is None
    assert any("CPU 측정값이 없다" in reason for reason in baseline["reasons"])


def test_protocol_source_cannot_escape_repository(public_snapshot):
    root, audit = public_snapshot
    protocol_path = root / B.PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["cases"][0]["sources"]["telemetry"]["path"] = "../private.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(B.BuildError, match="경계"):
        B.build_summary("run1", audit)


def test_frozen_evidence_code_change_blocks_snapshot(public_snapshot):
    root, audit = public_snapshot
    frozen = root / DRIVER.CODE[0]
    frozen.write_text(frozen.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
    with pytest.raises(B.BuildError, match="사전등록 이후 코드 변경"):
        B.build_summary("run1", audit)


def test_existing_output_is_never_overwritten(public_snapshot, tmp_path):
    _, audit = public_snapshot
    summary = B.build_summary("run1", audit)
    out = tmp_path / "replay.html"
    out.write_text("keep", encoding="utf-8")
    with pytest.raises(B.BuildError, match="덮어쓰지"):
        B.write_outputs(summary, out)
    assert out.read_text(encoding="utf-8") == "keep"
