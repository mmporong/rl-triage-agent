from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
M = importlib.import_module("lineage_comparison_batch_v1")


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def cases() -> list[dict]:
    arms = ("NONE", "NONE", "F1", "F1", "F2", "F2", "COMPOUND", "COMPOUND", "F5", "F3")
    result = []
    for index, arm in enumerate(arms, 1):
        seed = 2300 + index
        reference = f"none-s{seed}"
        result.append({"case_id": f"case-{index:02d}", "seed": seed, "arm": arm,
                       "source_job": reference if arm == "NONE" else f"{arm.lower()}-s{seed}",
                       "reference_job": reference})
    return result


def source_protocol() -> dict:
    rows = cases()
    jobs = []
    for case in rows:
        arms = ("NONE",) if case["arm"] == "NONE" else ("NONE", case["arm"])
        for arm in arms:
            jobs.append({"job_id": f"{arm.lower()}-s{case['seed']}", "case_id": case["case_id"],
                         "arm": arm, "seed": case["seed"]})
    return {"schema": "lineage_source_collection_protocol_v1", "cases": rows,
            "training_jobs": jobs, "runtime_source_sha256_lf": {"runtime.py": "a" * 64}}


def test_latin_rotation_and_transport_budget_are_fixed() -> None:
    plan = M.latin_plan(cases())
    assert len(plan) == 30
    assert [row["policy"] for row in plan[:9]] == [
        "registry_next", "fixed_eligible", "model_adaptive",
        "fixed_eligible", "model_adaptive", "registry_next",
        "model_adaptive", "registry_next", "fixed_eligible",
    ]
    transport = M.transport_plan(cases())
    assert len(transport["logical_slots"]) == 30
    assert transport["global_request_cap"] == 90
    assert transport["per_slot_max"] == 3
    assert len({row["slot_id"] for row in transport["logical_slots"]}) == 30


def test_expected_subtypes_use_registry_canonical_names_and_compound_r1_is_not_false(
        monkeypatch) -> None:
    assert M.validate_arm_expected()["COMPOUND"] == frozenset({"reward", "optimizer_update_count"})
    case = next(row for row in cases() if row["arm"] == "COMPOUND")
    result = {"confirmed_subtype": "optimizer_update_count", "decision_count": 1}
    classified = M._classification(case, result)
    assert classified["correct_confirmation"] is True
    assert classified["false_confirmation"] is False
    assert classified["compound_partial"] is True
    assert classified["full_cause_confirmed"] is False
    monkeypatch.setitem(M.ARM_EXPECTED, "COMPOUND", frozenset({"reward", "optimizer_count"}))
    with pytest.raises(ValueError, match="registry 범위 밖"):
        M.validate_arm_expected()


@pytest.mark.parametrize("seconds", [1, 480, 600])
def test_paired_watchdog_budget_accepts_only_positive_values_within_two_role_cap(seconds) -> None:
    assert M.validate_paired_watchdogs({"watchdogs_s": {
        "legacy": seconds, "optimizer": seconds}}) == {
            "legacy": seconds, "optimizer": seconds}


@pytest.mark.parametrize("watchdogs", [
    {"legacy": 601, "optimizer": 480},
    {"legacy": 480, "optimizer": 1200},
    {"legacy": 0, "optimizer": 480},
    {"legacy": 480.0, "optimizer": 480},
])
def test_paired_watchdog_budget_rejects_overflow_and_non_positive_or_non_integer(watchdogs) -> None:
    with pytest.raises(ValueError, match="watchdog"):
        M.validate_paired_watchdogs({"watchdogs_s": watchdogs})


def test_public_protocol_has_one_way_component_dag_and_no_private_paths(tmp_path, monkeypatch) -> None:
    source = save(tmp_path / "source.json", source_protocol())
    transport = save(tmp_path / "transport.json", {
        "schema": M.T.SCHEMA, "plan": M.transport_plan(cases())})
    probe = save(tmp_path / "probe.json", {"schema": M.P.PROTOCOL_SCHEMA,
                                           "watchdogs_s": {"legacy": 480, "optimizer": 480}})
    monkeypatch.setattr(M.SOURCE, "validate_protocol", lambda value: value)
    monkeypatch.setattr(M, "_component_source_bindings", lambda _values: {})
    relative = {source.resolve(): "bench/protocols/source.json",
                transport.resolve(): "bench/protocols/transport.json",
                probe.resolve(): "bench/protocols/probe.json",
                M.E.C.V.PROTOCOL.resolve(): "bench/protocols/view.json",
                M.E.C.PROTOCOL.resolve(): "bench/protocols/controller.json"}
    monkeypatch.setattr(M, "_relative", lambda path: relative[Path(path).resolve()])
    value = M.prepare_protocol(source, transport, probe)
    assert value["cases"] == cases()
    assert value["budget"]["model_logical_slots_total"] == 30
    assert value["budget"]["model_post_cap_total"] == 90
    assert value["budget"]["execution_request_budget_s"] == 1200
    assert value["budget"]["paired_probe_watchdogs_s"] == {"legacy": 480, "optimizer": 480}
    assert value["budget"]["paired_roles_per_logical_request"] == 2
    assert value["budget"]["replacement_seeds"] == 0
    assert value["error_rules"] == {
        "model_invalid_unknown_hold_rule_fallback": False,
        "pending_auto_resend": False, "resume_supported": False,
        "output": "fresh_write_once_fail_closed"}
    assert "source_commit" not in value
    assert all(not Path(row["path"]).is_absolute()
               for row in value["component_protocols"].values())
    assert value["transport_dag"].startswith("batch -> transport protocol")
    assert set(M.SOURCE_FILES) == set(value["source_sha256_lf"])


def test_selection_component_protocols_cover_registry_ledger_dependencies() -> None:
    view = M._read(M.E.C.V.PROTOCOL)["source_sha256_lf"]
    controller = M._read(M.E.C.PROTOCOL)["source_sha256_lf"]
    required = {"src/rl_triage/mechanism_registry.py", "src/rl_triage/probe_loop.py",
                "evals/selection_comparison_view_v1.py",
                "evals/selection_comparison_controller_v1.py"}
    assert required <= set(view) | set(controller)


def test_collected_index_binds_all_18_job_seed_checkpoint_and_telemetry(tmp_path, monkeypatch) -> None:
    protocol = source_protocol()
    records = {}
    entries = {}
    values = {}
    for job in protocol["training_jobs"]:
        checkpoint = (job["job_id"].encode().hex() + "0" * 64)[:64]
        telemetry = (job["job_id"].encode().hex() + "1" * 64)[:64]
        path = save(tmp_path / f"{job['job_id']}.json", {"job": job["job_id"]})
        records[job["job_id"]] = {"checkpoint": {"sha256": checkpoint}}
        entries[job["job_id"]] = {"path": str(path), "sha256": M._sha(path),
                                    "job_id": job["job_id"], "arm": job["arm"],
                                    "seed": job["seed"], "checkpoint_sha256": checkpoint,
                                    "telemetry_sha256": telemetry}
        values[path.resolve()] = ({"private_source_map": {"job_id": job["job_id"],
                                                           "arm": job["arm"], "seed": job["seed"]},
                                   "source_binding": {"checkpoint_sha256": checkpoint,
                                                      "telemetry_sha256": telemetry}}, {})
    monkeypatch.setattr(M.INPUT, "_checked_collected", lambda path: values[Path(path).resolve()])
    checked = M._collected({"schema": "lineage_collected_index_v1", "jobs": entries},
                           protocol, records)
    assert len(checked) == 18
    first = next(iter(entries))
    entries[first]["seed"] += 1
    with pytest.raises(ValueError, match="index"):
        M._collected({"schema": "lineage_collected_index_v1", "jobs": entries},
                     protocol, records)


def fixture_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    protocol = {"cases": cases(), "latin_execution_plan": M.latin_plan(cases())}
    output = tmp_path / "batch"
    output.mkdir()
    manifest = {"output_root": str(output), "execution_plan": protocol["latin_execution_plan"],
                "lab": str(tmp_path / "lab"), "probe_protocol": str(tmp_path / "probe.json"),
                "execution_anchor": "b" * 40, "gate_path": str(tmp_path / "gate.json"),
                "gate_sha256": "c" * 64, "transport_root": str(tmp_path / "transport")}
    collected = {}
    for case in cases():
        for job in (case["source_job"], case["reference_job"]):
            collected[job] = {"path": str(tmp_path / f"{job}.json")}
    labels = {row["case_id"]: {"label": "eligible"} for row in cases()}
    labels["case-03"]["label"] = "source_failure_absent"
    labels["case-04"]["label"] = "measurement_unknown"
    labels["case-05"]["label"] = "baseline_invalid"
    labels["case-09"]["label"] = "out_of_support"
    monkeypatch.setattr(M, "_validate_manifest",
                        lambda _path: (manifest, protocol, collected, labels))
    monkeypatch.setattr(M, "_initial_view", lambda case, *_args: f"view-{case['case_id']}")
    return output, manifest, protocol, labels


def fake_result(case_id: str, policy: str) -> dict:
    index = int(case_id[-2:])
    arms = {row["case_id"]: row["arm"] for row in cases()}
    arm = arms[case_id]
    excluded = index in {3, 4, 5}
    subtype = None
    if not excluded and arm != "NONE":
        subtype = {"F1": "reward", "F2": "actuator", "COMPOUND": "reward",
                   "F5": "reward", "F3": None}[arm]
    decisions = 0 if excluded or arm == "NONE" else 1
    model_invalid = policy == "model_adaptive" and case_id == "case-06"
    if model_invalid:
        subtype, decisions = None, 0
    steps = ([{"paired_receipts": {
        role: {"status": "complete", "claim": {"role": role},
               "execution": {"wall_s": 0.25}}
        for role in ("reference", "case")}}] if decisions else [])
    return {
        "schema": M.E.SCHEMA, "end_state": "confirmed" if subtype else "HOLD",
        "confirmed_subtype": subtype, "decision_count": decisions,
        "requests": 1 if policy == "model_adaptive" else 0,
        "model_validity": ["invalid"] if model_invalid else (["valid"] if policy == "model_adaptive" else []),
        "tokens": {"total_tokens": 15 if policy == "model_adaptive" else 0},
        "wall": {"model_wall_s": 0.1 if policy == "model_adaptive" else 0.0,
                 "known_model_wall_subtotal_s": 0.1 if policy == "model_adaptive" else 0.0,
                 "probe_child_known_s": decisions * 0.5,
                 "probe_child_unknown_roles": [],
                 "probe_parent_known_s": decisions * 0.6},
        "steps": steps,
        "correction_executed": False, "training_executed": False, "epoch2_registered": False,
    }


def runner(out: Path, _case: Path, _reference: Path, policy: str, **kwargs) -> dict:
    result = fake_result(kwargs["case_id"], policy)
    save(out / "result.json", result)
    return result


def test_execute_preserves_30_pairs_denominator_exclusions_and_cost_scopes(tmp_path, monkeypatch) -> None:
    output, _manifest, protocol, _labels = fixture_context(tmp_path, monkeypatch)
    result = M.execute(output / "manifest.json", epoch_runner=runner)
    assert result["status"] == "complete"
    assert result["evidence_scope"] == "synthetic_cpu_fixture"
    assert len(result["rows"]) == 30
    assert result["summary"]["denominator_case_policy_pairs"] == 30
    for policy in M.POLICIES:
        summary = result["summary"]["by_policy"][policy]
        assert summary["denominator_cases"] == 10
        assert summary["excluded"] == 3
        assert summary["compound_partial"] == 2
        assert summary["false_confirmations"] == 1
        assert summary["none_interventions"] == 0
        expected = 8 if policy == "model_adaptive" else 10
        assert summary["paired_backend_attempts"] == expected
        assert summary["paired_backend_successes"] == expected
        assert summary["paired_backend_unknown"] == 0
    assert result["summary"]["costs"]["api_requests"] == 10
    assert result["summary"]["costs"]["tokens"]["total"] == 150
    for case in protocol["cases"]:
        paired = json.loads((output / "execution/case-pairs" / f"{case['case_id']}.json").read_text())
        assert set(paired["policies"]) == set(M.POLICIES)
        assert paired["fresh_view_digest"] == f"view-{case['case_id']}"
    with pytest.raises(RuntimeError, match="resume_not_supported"):
        M.execute(output / "manifest.json", epoch_runner=runner)


def test_paired_receipts_and_unknown_owned_wall_follow_actual_role_evidence() -> None:
    result = {"steps": [{"paired_receipts": {
        "reference": {"status": "complete", "claim": {"role": "reference"},
                      "execution": {"wall_s": 0.2}},
        "case": {"status": "unknown", "claim": {"role": "case"}, "execution": {}}}}],
        "wall": {"probe_child_known_s": 0.2, "probe_child_unknown_roles": ["case"]}}
    stats = M._paired_stats(result)
    assert stats == {"attempted": 2, "succeeded": 1, "unknown": 1, "unverified": 0}
    wall = M._owned_wall([result])
    assert wall == {"total": None, "known_subtotal": 0.2, "unknown_rows": 0,
                    "unknown_roles": 1}
    hold = {"decision_count": 0, "steps": [],
            "wall": {"probe_child_known_s": 0.0, "probe_child_unknown_roles": []}}
    assert M._paired_stats(hold) == {"attempted": 0, "succeeded": 0,
                                     "unknown": 0, "unverified": 0}


def test_interruption_preserves_rows_and_refuses_resume(tmp_path, monkeypatch) -> None:
    output, _manifest, _protocol, _labels = fixture_context(tmp_path, monkeypatch)
    count = 0

    def interrupted(out, case, reference, policy, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("fixture interruption")
        return runner(out, case, reference, policy, **kwargs)

    with pytest.raises(RuntimeError, match="fixture interruption"):
        M.execute(output / "manifest.json", epoch_runner=interrupted)
    failure = json.loads((output / "execution/failed.json").read_text())
    assert failure["completed_rows"] == 1
    assert failure["resume_supported"] is False
    assert failure["automatic_resend"] is False
    assert len(list((output / "execution/rows").glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="resume_not_supported"):
        M.execute(output / "manifest.json", epoch_runner=interrupted)
