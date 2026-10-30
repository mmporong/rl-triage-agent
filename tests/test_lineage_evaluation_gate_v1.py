from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
G = importlib.import_module("lineage_evaluation_gate_v1")

PROTOCOL = ROOT / "bench/protocols/lineage_source_collection_20261030_v2.json"
BANDS = ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json"
REAL_NONE = ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_none.json"
REAL_F1 = ROOT / "evals/results/brev_recovery_v4_s2196_20261024/source_fixed_f1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def gate(tmp_path: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    none = json.loads(REAL_NONE.read_text(encoding="utf-8"))
    fault = json.loads(REAL_F1.read_text(encoding="utf-8"))
    records, index, reports = [], {}, {}
    duplicate = None
    for job in protocol["training_jobs"]:
        checkpoint = hashlib.sha256((job["job_id"] + "-checkpoint").encode()).hexdigest()
        if job["job_id"] == "f1-s2303":
            duplicate = checkpoint
        elif job["job_id"] == "f1-s2304":
            checkpoint = duplicate
        records.append({"job_id": job["job_id"], "purpose": "source", "arm": job["arm"],
                        "seed": job["seed"], "checkpoint": {"sha256": checkpoint}})
        report = copy.deepcopy(none if job["arm"] == "NONE" else fault)
        report["name"] = job["job_id"]
        report["checkpoint"] = {"file": "model_299.pt", "sha256": checkpoint}
        path = tmp_path / f"{job['job_id']}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        reports[job["job_id"]] = path
        index[job["job_id"]] = {"report_path": str(path), "report_sha256": _sha(path),
                                "checkpoint_sha256": checkpoint}
    source = {"status": "complete", "source_jobs_completed": 18, "records": records}
    return {"protocol": protocol, "source": source, "index": index, "reports": reports,
            "tmp": tmp_path}


def _rewrite(gate: dict, job_id: str, report: dict) -> None:
    path = gate["reports"][job_id]
    path.write_text(json.dumps(report), encoding="utf-8")
    gate["index"][job_id]["report_sha256"] = _sha(path)


def test_consumes_18_reports_into_10_same_seed_cases_and_reports_dedup(gate: dict) -> None:
    result = G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                            repo=ROOT, bands_path=BANDS)
    assert result["status"] == "complete" and result["planned_cases"] == 10
    assert result["valid_pairs"] == 10 and result["excluded_total"] == 0
    assert result["label_counts"] == {"baseline_invalid": 0, "source_failure_absent": 0,
                                      "measurement_unknown": 0, "eligible": 8,
                                      "out_of_support": 2}
    by_case = {row["case_id"]: row for row in result["cases"]}
    assert by_case["case-01"]["expected_correction_interventions"] == 0
    assert by_case["case-09"]["label"] == "out_of_support"
    assert by_case["case-09"]["expected_action"] == "HOLD"
    assert by_case["case-09"]["diagnosis_eligible"] is True
    assert by_case["case-09"]["expected_correction_interventions"] == 0
    assert all(row["diagnosis_claimed"] is False for row in result["cases"])
    assert result["duplicate_checkpoint_groups"] == [{
        "checkpoint_sha256": gate["index"]["f1-s2303"]["checkpoint_sha256"],
        "case_ids": ["case-03", "case-04"], "dedup_required": True,
    }]
    assert result["independent_case_count"] == 9
    assert result["independent_valid_pairs"] == 9
    assert result["development_coupling"]["heldout_claim"] is False


def test_baseline_invalid_and_failure_absent_are_distinct(gate: dict) -> None:
    baseline = json.loads(gate["reports"]["none-s2303"].read_text(encoding="utf-8"))
    baseline["metrics"]["lin_vel_rmse_mps"] = baseline["standstill"]["lin_vel_rmse_mps"]
    _rewrite(gate, "none-s2303", baseline)
    source = json.loads(gate["reports"]["f1-s2304"].read_text(encoding="utf-8"))
    healthy = json.loads(gate["reports"]["none-s2304"].read_text(encoding="utf-8"))
    source["metrics"] = healthy["metrics"]
    source["by_condition"] = healthy["by_condition"]
    _rewrite(gate, "f1-s2304", source)
    result = G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                            repo=ROOT, bands_path=BANDS)
    rows = {row["case_id"]: row for row in result["cases"]}
    assert rows["case-03"]["label"] == "baseline_invalid"
    assert rows["case-04"]["label"] == "source_failure_absent"
    assert result["excluded_total"] == 2
    assert result["independent_case_count"] == 9
    assert result["independent_valid_pairs"] == 8


def test_undetermined_measurement_is_held_and_not_reported_as_failure_absent(
        gate: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(row for row in gate["protocol"]["cases"] if row["case_id"] == "case-03")
    source = json.loads(gate["reports"][case["source_job"]].read_text(encoding="utf-8"))
    reference = json.loads(gate["reports"][case["reference_job"]].read_text(encoding="utf-8"))
    bands = json.loads(BANDS.read_text(encoding="utf-8"))["bands"]
    monkeypatch.setattr(G.RESIDUAL, "recovery", lambda *_args: {"verdict": {"label": "undetermined"}})
    measured = G._classify(case, source, reference, bands)
    assert measured["label"] == "measurement_unknown"
    assert measured["expected_action"] == "HOLD"
    assert measured["diagnosis_eligible"] is False
    assert measured["expected_correction_interventions"] == 0


@pytest.mark.parametrize("tamper", ["task", "eval_seed", "checkpoint", "missing_condition", "different_row"])
def test_rejects_wrong_report_binding_and_26_condition_tamper(gate: dict, tamper: str) -> None:
    job_id = "f1-s2303"
    report = json.loads(gate["reports"][job_id].read_text(encoding="utf-8"))
    if tamper == "task":
        report["task"] = "wrong-task"
    elif tamper == "eval_seed":
        report["eval_seed"] = 2303
    elif tamper == "checkpoint":
        report["checkpoint"]["sha256"] = "f" * 64
    elif tamper == "missing_condition":
        report["by_condition"].pop()
    else:
        report["by_condition"][-1] = copy.deepcopy(report["by_condition"][0])
    _rewrite(gate, job_id, report)
    with pytest.raises(ValueError):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=BANDS)


def test_rejects_report_from_another_job_even_if_index_sha_is_updated(gate: dict) -> None:
    first, second = "f1-s2303", "f1-s2304"
    wrong = json.loads(gate["reports"][second].read_text(encoding="utf-8"))
    _rewrite(gate, first, wrong)
    with pytest.raises(ValueError, match="binding"):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=BANDS)


@pytest.mark.parametrize("missing", ["source", "index"])
def test_requires_all_18_source_receipts_and_explicit_reports(gate: dict, missing: str) -> None:
    if missing == "source":
        gate["source"]["records"].pop()
    else:
        gate["index"].pop("f3-s2310")
    with pytest.raises(ValueError, match="18"):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=BANDS)


@pytest.mark.parametrize("field", ["seed", "checkpoint"])
def test_rejects_source_receipt_seed_or_checkpoint_mismatch(gate: dict, field: str) -> None:
    row = next(item for item in gate["source"]["records"] if item["job_id"] == "f1-s2303")
    if field == "seed":
        row["seed"] += 1
    else:
        row["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt|checkpoint"):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=BANDS)


def test_rejects_changed_original_bands_and_writes_output_exclusively(gate: dict) -> None:
    changed = gate["tmp"] / "bands.json"
    value = json.loads(BANDS.read_text(encoding="utf-8"))
    value["bands"]["fall_frac_max"] = 0.06
    changed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="bands SHA"):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=changed)
    output = gate["tmp"] / "gate.json"
    result = G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                            repo=ROOT, bands_path=BANDS, out=output)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(FileExistsError):
        G.consume_gate(gate["source"], gate["protocol"], gate["index"],
                       repo=ROOT, bands_path=BANDS, out=output)


def test_existing_real_fixed_reports_keep_schema_and_recovery_result() -> None:
    none = json.loads(REAL_NONE.read_text(encoding="utf-8"))
    fault = json.loads(REAL_F1.read_text(encoding="utf-8"))
    none_job = {"job_id": none["name"], "run_name": none["name"]}
    fault_job = {"job_id": fault["name"], "run_name": fault["name"]}
    assert G._validate_report(none, none_job, none["checkpoint"]["sha256"]) == none
    assert G._validate_report(fault, fault_job, fault["checkpoint"]["sha256"]) == fault
    bands = json.loads(BANDS.read_text(encoding="utf-8"))["bands"]
    assert G.RESIDUAL.D1.V.R1.U.walks(none) is True
    measured = G.RESIDUAL.recovery(fault, none, bands)
    assert measured["verdict"]["label"] == "unhealthy"
    assert measured["outcome"] == "not_recovered"


def test_cpu_cli_writes_gate(gate: dict) -> None:
    source = gate["tmp"] / "source.json"
    protocol = gate["tmp"] / "protocol.json"
    index = gate["tmp"] / "index.json"
    output = gate["tmp"] / "cli-gate.json"
    source.write_text(json.dumps(gate["source"]), encoding="utf-8")
    protocol.write_text(json.dumps(gate["protocol"]), encoding="utf-8")
    index.write_text(json.dumps(gate["index"]), encoding="utf-8")
    assert G.main(["--protocol", str(protocol), "--source-result", str(source),
                   "--report-index", str(index), "--bands", str(BANDS),
                   "--out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
