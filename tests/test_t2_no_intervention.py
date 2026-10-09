"""동결 healthy 게이트·입력 오류·예산 소비 없는 종료와 결과 보존을 확인한다."""
import importlib.util
import json

import pytest

from test_behavior_oracle import GATES, ROOT, report


@pytest.fixture
def gate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("t2_gate_test", ROOT / "evals/t2_no_intervention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def inputs(gate, candidate=None, reference=None):
    reference = reference or report()
    candidate = candidate or report("candidate", "b" * 64)
    records = {}
    for key, obj in (("candidate", candidate), ("reference", reference)):
        path = gate.ROOT / f"{key}.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        records[key] = {"path": path.name, "sha256_lf": gate.sha(path)}
    contract = gate.O.register_contract(report(), [], GATES)
    spec = {"case": "synthetic", "inputs": records, "checkpoint_sha256_as_recorded": {
            "candidate": candidate["checkpoint"]["sha256"], "reference": reference["checkpoint"]["sha256"]},
            "reference_identity": contract["evaluation"], "relative_bands": contract["bands"]}
    return spec, {"absolute_gates": GATES}


def test_healthy_gate_stops_without_changing_existing_approval_ledger(gate):
    spec, base = inputs(gate)
    folder = gate.ROOT / "evals/loops/existing"
    folder.mkdir(parents=True)
    ledger = folder / "ledger.jsonl"
    ledger.write_bytes(b'{"event":"approved","request_id":"reserved"}\n')
    before = gate.loop_snapshot()
    row = gate.gate_case(spec, base)
    assert row["decision"] == "do_not_open_loop" and row["behavior_label"] == "healthy"
    assert not row["loop_opened"] and all(row[key] == 0 for key in gate.COUNTERS)
    assert before == gate.loop_snapshot()


def test_unhealthy_case_stops_outside_t2_instead_of_starting_diagnosis(gate):
    spec, base = inputs(gate, report("bad", lin=0.8))
    row = gate.gate_case(spec, base)
    assert row["behavior_label"] == "unhealthy" and row["decision"] == "outside_T2_healthy_scope_stop"
    assert not row["loop_opened"] and row["probe_executions"] == row["interventions"] == 0


@pytest.mark.parametrize("mutation", ["hash", "checkpoint", "task", "bands", "reference"])
def test_invalid_inputs_stop_undetermined_without_consuming_approval(gate, mutation):
    spec, base = inputs(gate)
    if mutation == "hash":
        (gate.ROOT / "candidate.json").write_text('{}', encoding="utf-8")
    elif mutation == "checkpoint":
        spec["checkpoint_sha256_as_recorded"]["candidate"] = "c" * 64
    elif mutation == "bands":
        spec["relative_bands"]["err_xy_ratio_max"] = 9
    else:
        key = "reference" if mutation == "reference" else "candidate"
        path = gate.ROOT / f"{key}.json"
        obj = json.loads(path.read_text())
        if mutation == "task":
            obj["task"] = "another-task"
        else:
            obj = report("standing_reference", lin=0.8)
        path.write_text(json.dumps(obj), encoding="utf-8")
        spec["inputs"][key]["sha256_lf"] = gate.sha(path)
    row = gate.gate_case(spec, base)
    assert row["behavior_label"] == "undetermined" and row["decision"] == "stop_undetermined"
    assert row["error"] and row["consumed"] == row["receipts"] == 0


def test_existing_tag_is_preserved_before_any_case_execution(gate, monkeypatch):
    out = gate.ROOT / "evals/results/existing"
    out.mkdir(parents=True)
    summary = out / "summary.json"
    summary.write_bytes(b'original-result')
    monkeypatch.setattr(gate, "verified_protocol", lambda *_: pytest.fail("must stop before preflight"))
    with pytest.raises(FileExistsError):
        gate.run(gate.ROOT / "unused.json", "existing")
    assert summary.read_bytes() == b'original-result'


def test_output_tag_cannot_escape_results_folder(gate):
    with pytest.raises(ValueError):
        gate.run(gate.ROOT / "unused.json", "../outside")


@pytest.mark.parametrize("planned", ["other_tag", "../bad", None])
def test_wrong_planned_tag_stops_before_decisions_snapshots_or_output(gate, monkeypatch, planned):
    monkeypatch.setattr(gate, "verified_protocol", lambda *_: ({"planned_tag": planned}, {}))
    monkeypatch.setattr(gate, "gate_case", lambda *_: pytest.fail("no case evaluation"))
    monkeypatch.setattr(gate, "loop_snapshot", lambda: pytest.fail("no loop snapshot"))
    with pytest.raises(ValueError, match="planned_tag"):
        gate.run(gate.ROOT / "unused.json", "synthetic_gate")
    assert not (gate.ROOT / "evals/results").exists()


def test_gate_run_records_cpu_wall_and_preserves_ledger(gate, monkeypatch):
    spec, base = inputs(gate)
    base["cases"] = [{**spec, "case": name} for name in ("X2", "X3", "NONE")]
    base["claim_limits"] = ["synthetic inputs", "not executed yet"]
    prereg = gate.ROOT / "prereg.json"
    prereg.write_text('{}', encoding="utf-8")
    monkeypatch.setattr(gate, "verified_protocol", lambda *_: ({"base_protocol": {}, "planned_tag": "synthetic_gate"}, base))
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *_a, **_kw: "a" * 40)
    summary = gate.run(prereg, "synthetic_gate")
    assert summary["healthy"] == 3 and summary["accepted"]
    assert summary["costs"]["cpu_s"] >= 0 and summary["costs"]["execution_wall_s"] > 0
    assert summary["loop_snapshots"]["before"] == summary["loop_snapshots"]["after"]
    assert all(value == 0 for value in summary["counts"].values())
    assert (gate.ROOT / "evals/results/synthetic_gate/summary.json").is_file()
