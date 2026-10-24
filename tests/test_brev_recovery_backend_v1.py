from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("brev_recovery_backend_v1_test", ROOT / "evals/brev_recovery_backend_v1.py")
assert SPEC is not None and SPEC.loader is not None
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def protocol() -> dict:
    protocol_path = ROOT / "bench/protocols/fixed_eval_v1.json"
    return {
        "environment": {"python": "3.10.12", "isaaclab_sha": "9" * 40},
        "source_sha256_lf": {"bench/residual_training_v1.py": "a" * 64},
        "runtime_source_sha256_lf": {
            "scripts/reinforcement_learning/rsl_rl/train.py": "b" * 64,
            "source/isaaclab/isaaclab/envs/mdp/rewards.py": "c" * 64,
        },
        "python_package_source_sha256_lf": {
            "rsl_rl/algorithms/ppo.py": "d" * 64,
            "rsl_rl/storage/rollout_storage.py": "e" * 64,
            "rsl_rl/runners/on_policy_runner.py": "f" * 64,
        },
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256_lf": B.sha256(protocol_path, lf=True),
    }


def backend(tmp_path: Path, gpu=None):
    obj = object.__new__(B.Backend)
    obj.lab = tmp_path
    obj.python = Path(sys.executable)
    obj.protocol = protocol()
    obj.anchor = "1" * 40
    obj.read_gpu = gpu or (list)
    obj.popen = B.subprocess.Popen
    obj._source_snapshot = lambda: {"bound": True}
    return obj


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / f"child_{len(list(tmp_path.glob('child_*.py')))}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _job(stage: Path) -> dict:
    return {"schema": "fixture", "mode": "train", "stage": str(stage)}


@pytest.mark.skipif(os.name != "posix", reason="실제 POSIX process-group 회귀는 Linux에서 실행")
@pytest.mark.parametrize("case", ["success", "missing", "nonzero", "timeout"])
def test_real_owned_process_requires_receipt_exit_cleanup_and_deadline(tmp_path, monkeypatch, case):
    stage = tmp_path / case
    if case == "success":
        body = "import json,sys;from pathlib import Path;Path(sys.argv[-1]).write_text(json.dumps({'completed':True}))"
    elif case == "missing":
        body = "raise SystemExit(0)"
    elif case == "nonzero":
        body = "raise SystemExit(7)"
    else:
        body = "import time;time.sleep(30)"
    monkeypatch.setattr(B, "WORKER", _script(tmp_path, body))
    monkeypatch.setitem(B.TIMEOUTS, "train", 0.1 if case == "timeout" else 10)
    obj = backend(tmp_path)
    if case == "success":
        worker, receipt = obj._execute(stage, "train", _job(stage))
        assert worker["completed"] is True and receipt["completed"] is True
    else:
        with pytest.raises(B.BackendError) as error:
            obj._execute(stage, "train", _job(stage))
        receipt = error.value.receipt
        assert receipt["completed"] is False and receipt["cleanup_verified"] is True
        if case == "missing":
            assert receipt["exit_code"] == 0
        elif case == "nonzero":
            assert receipt["exit_code"] == 7
        else:
            assert receipt["timed_out"] is True


def test_foreign_gpu_busy_refuses_launch_and_writes_receipt(tmp_path):
    stage = tmp_path / "busy"
    obj = backend(tmp_path, gpu=lambda: [9191])
    with pytest.raises(B.BackendError, match="ForeignGPUBusy") as error:
        obj._execute(stage, "train", _job(stage))
    assert error.value.receipt["launch"]["session_id"] is None
    assert json.loads((stage / "execution.json").read_text(encoding="utf-8"))["gpu_before"] == [9191]


def test_source_preflight_failure_writes_nonlaunch_receipt(tmp_path):
    stage = tmp_path / "source-mismatch"
    obj = backend(tmp_path)
    obj._source_snapshot = lambda: (_ for _ in ()).throw(ValueError("mismatch"))
    with pytest.raises(B.BackendError) as error:
        obj._execute(stage, "train", _job(stage))
    assert error.value.receipt["launch"]["session_id"] is None
    assert error.value.receipt["cleanup_verified"] is True
    assert (stage / "execution.json").is_file()


def test_protocol_mutation_after_prepare_refuses_launch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_path = repo / "protocol.json"
    original = '{"schema":"prepared"}\n'
    protocol_path.write_text(original, encoding="utf-8", newline="\n")
    expected = B.sha256(protocol_path, lf=True)
    obj = backend(tmp_path)
    obj.protocol["protocol_path"] = str(protocol_path.resolve())
    obj.protocol["protocol_sha256_lf"] = expected
    launches = []
    obj.popen = lambda *_args, **_kwargs: launches.append(True)  # pragma: no cover - 호출되면 실패
    monkeypatch.setattr(B, "ROOT", repo)

    def fake_git(_root, *args, **_kwargs):
        assert args == ("show", f"{obj.anchor}:protocol.json")
        return type("R", (), {"stdout": original, "returncode": 0})()

    monkeypatch.setattr(B, "_git", fake_git)
    obj._source_snapshot = lambda: {
        "protocol": B.verify_protocol_binding(protocol_path, expected, obj.anchor),
    }
    protocol_path.write_text('{"schema":"changed"}\n', encoding="utf-8", newline="\n")

    with pytest.raises(B.BackendError, match="ValueError") as error:
        obj._execute(tmp_path / "mutated", "train", _job(tmp_path / "mutated"))
    assert error.value.receipt["launch"]["session_id"] is None
    assert launches == []


def test_protocol_rejects_missing_distribution_source_and_bad_relative_path():
    value = protocol()
    del value["python_package_source_sha256_lf"]["rsl_rl/algorithms/ppo.py"]
    with pytest.raises(ValueError, match="distribution"):
        B.validate_protocol(value)
    value = protocol()
    value["source_sha256_lf"] = {"../escape.py": "a" * 64}
    with pytest.raises(ValueError, match="형식"):
        B.validate_protocol(value)


def test_source_snapshot_fails_closed_on_anchor_mismatch(tmp_path, monkeypatch):
    obj = backend(tmp_path)
    monkeypatch.setattr(B, "_git", lambda *_args, **_kwargs: type("R", (), {"stdout": "2" * 40 + "\n"})())
    with pytest.raises(ValueError, match="HEAD"):
        B.Backend._source_snapshot(obj)


def test_source_snapshot_fails_closed_on_runtime_hash_mismatch(tmp_path, monkeypatch):
    obj = backend(tmp_path)
    for relative in obj.protocol["runtime_source_sha256_lf"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("different", encoding="utf-8")

    def fake_git(_root, *args, **_kwargs):
        if args[:2] == ("rev-parse", "HEAD"):
            value = obj.protocol["environment"]["isaaclab_sha"] + "\n"
        elif args[:1] == ("show",):
            value = Path(obj.protocol["protocol_path"]).read_text(encoding="utf-8")
        else:
            value = ""
        return type("R", (), {"stdout": value, "returncode": 0})()

    monkeypatch.setattr(B, "_git", fake_git)
    with pytest.raises(ValueError, match="runtime source hash"):
        B.Backend._source_snapshot(obj)


def _legacy_probe_result(probe="P_reward"):
    measurement = {"ratio": 1.0}
    artifact = {"seed": 2190, "arm": "F1", "checkpoint": {"sha256": "a" * 64}}
    report = {
        "name": "f1-s2190", "task": B.TASK, "seed": 2190,
        "checkpoint": {"file": "model_299.pt", "sha256": "a" * 64},
        "probes": {probe: {"exit": 0, "measurement": measurement}},
    }
    return artifact, {"report": report, "measurement": measurement}


def test_validate_probe_result_accepts_bound_legacy_raw_measurement():
    artifact, result = _legacy_probe_result()
    assert B.validate_probe_result(artifact, "P_reward", "f1-s2190", result) == {"ratio": 1.0}


@pytest.mark.parametrize("mutation", ["returned", "checkpoint", "seed", "extra", "missing"])
def test_validate_probe_result_rejects_unbound_legacy_report(mutation):
    artifact, result = _legacy_probe_result()
    if mutation == "returned":
        result["measurement"] = {"ratio": 2.0}
    elif mutation == "checkpoint":
        result["report"]["checkpoint"]["sha256"] = "b" * 64
    elif mutation == "seed":
        result["report"]["seed"] = 2191
    elif mutation == "extra":
        result["report"]["probes"]["P_value"] = {"exit": 0, "measurement": {}}
    else:
        result["report"]["probes"] = {}
    with pytest.raises(ValueError):
        B.validate_probe_result(artifact, "P_reward", "f1-s2190", result)


def _optimizer_probe_result():
    sources = {name: "d" * 64 for name in B.RESIDUAL.MINIMUM_BOUND_SOURCES}
    common = {
        "source_commit_sha": "b" * 40, "protocol_path": "bench/protocols/brev_recovery_v1.json",
        "protocol_sha256_lf": "c" * 64, "source_sha256_lf": sources,
        "runtime_source_sha256_lf": {"scripts/train.py": "e" * 64},
    }
    training_manifest = {**common, "schema": B.RESIDUAL.SCHEMA, "mode": "train", "arm": "F1",
        "seed": 2190, "run_name": "f1_s2190", "task": B.TASK, "num_envs": 4096,
        "max_iterations": 300, "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "runtime_updates_to_observe": 3, "records_path": str((ROOT / "fixture/train-records.jsonl").resolve()),
        "readback_path": str((ROOT / "fixture/train-readback.json").resolve()), "source_checkpoint_sha256": None,
        "original_faults": ["F1"], "applied_corrections": []}
    artifact = {"seed": 2190, "arm": "F1", "checkpoint": {"sha256": "a" * 64},
                "manifest": training_manifest}
    inspection_manifest = {**training_manifest, "mode": "inspection", "max_iterations": 3,
                           "records_path": str((ROOT / "fixture/inspect-records.jsonl").resolve()),
                           "readback_path": str((ROOT / "fixture/inspect-readback.json").resolve()),
                           "source_checkpoint_sha256": "a" * 64}
    manifest = {
        "schema": B.OPTIMIZER.MANIFEST_SCHEMA,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "expected_optimizer_steps_per_update": 20, "expected_minibatches_per_update": 20,
        "runtime_updates_to_observe": 3, "arm": "F1", "seed": 2190, "task": B.TASK,
        "num_envs": 4096, "max_iterations": 3, **{key: common[key] for key in (
            "source_commit_sha", "protocol_path", "protocol_sha256_lf", "source_sha256_lf")},
        "inspection_primitives": list(B.OPTIMIZER.PRIMITIVES),
    }
    primitives = {name: {"status": "normal"} for name in B.OPTIMIZER.PRIMITIVES}
    for name in ("pre_step_gradient_norm", "learning_rate_trajectory"):
        primitives[name]["values"] = [1.0]
        primitives[name]["status"] = "observed"
    for name in ("approx_kl", "clip_fraction"):
        primitives[name]["status"] = "unavailable"
    report = {"schema": "optimizer_observability_inspection_v1", "manifest": manifest,
              "records": 3, "status": "normal", "primitives": primitives}
    encoded = (json.dumps(inspection_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    descriptor = {"path": str((ROOT / "fixture/inspection.manifest.json").resolve()),
                  "sha256": B.hashlib.sha256(encoded).hexdigest(), "body": inspection_manifest}
    return artifact, {"report": report, "measurement": copy.deepcopy(report),
                      "inspection_manifest": descriptor}


def test_validate_probe_result_accepts_optimizer_first_three_update_report():
    artifact, result = _optimizer_probe_result()
    assert B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result) == result["measurement"]


def test_validate_probe_result_maps_corrected_artifact_to_first_inspection_arm():
    artifact, result = _optimizer_probe_result()
    artifact["arm"] = "CORRECTED"
    artifact["manifest"]["arm"] = "FIRST"
    artifact["manifest"]["run_name"] = "corrected_s2190"
    artifact["manifest"]["original_faults"] = ["F1"]
    artifact["manifest"]["applied_corrections"] = ["restore_body_frame_reward"]
    artifact["manifest"]["source_checkpoint_sha256"] = "a" * 64
    result["report"]["manifest"]["arm"] = "FIRST"
    result["measurement"] = copy.deepcopy(result["report"])
    inspection = result["inspection_manifest"]["body"]
    inspection["arm"] = "FIRST"
    inspection["run_name"] = "corrected_s2190"
    inspection["applied_corrections"] = ["restore_body_frame_reward"]
    result["report"]["manifest"] = B.RESIDUAL.observer_manifest(inspection)
    result["measurement"] = copy.deepcopy(result["report"])
    encoded = (json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    result["inspection_manifest"]["sha256"] = B.hashlib.sha256(encoded).hexdigest()
    assert B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "corrected_s2190", result)["records"] == 3


@pytest.mark.parametrize("mutation", ["returned", "seed", "records", "status", "primitives"])
def test_validate_probe_result_rejects_unbound_optimizer_report(mutation):
    artifact, result = _optimizer_probe_result()
    if mutation == "returned":
        result["measurement"]["records"] = 2
    elif mutation == "seed":
        result["report"]["manifest"]["seed"] = 2191
        result["measurement"] = copy.deepcopy(result["report"])
    elif mutation == "records":
        result["report"]["records"] = 2
        result["measurement"] = copy.deepcopy(result["report"])
    elif mutation == "status":
        result["report"]["status"] = "abnormal"
        result["measurement"] = copy.deepcopy(result["report"])
    else:
        result["report"]["primitives"].pop("clip_fraction")
        result["measurement"] = copy.deepcopy(result["report"])
    with pytest.raises(ValueError):
        B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result)


def test_validate_probe_result_rejects_wrong_optimizer_source_checkpoint():
    artifact, result = _optimizer_probe_result()
    result["inspection_manifest"]["body"]["source_checkpoint_sha256"] = "f" * 64
    encoded = (json.dumps(result["inspection_manifest"]["body"], ensure_ascii=False,
                          indent=2, allow_nan=False) + "\n").encode()
    result["inspection_manifest"]["sha256"] = B.hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="학습 artifact"):
        B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result)


def test_validate_probe_result_rejects_optimizer_manifest_body_tamper():
    artifact, result = _optimizer_probe_result()
    result["inspection_manifest"]["body"]["seed"] = 2191
    with pytest.raises(ValueError, match="본문 SHA"):
        B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result)


def test_validate_probe_result_rejects_optimizer_observer_source_mismatch():
    artifact, result = _optimizer_probe_result()
    result["report"]["manifest"]["protocol_sha256_lf"] = "f" * 64
    result["measurement"] = copy.deepcopy(result["report"])
    with pytest.raises(ValueError, match="첫 3 update"):
        B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result)


@pytest.mark.parametrize("field", ["source_sha256_lf", "num_envs"])
def test_optimizer_observer_manifest_must_equal_inspection_projection(field):
    artifact, result = _optimizer_probe_result()
    result["report"] = copy.deepcopy(result["report"])
    raw = result["report"]["manifest"]
    if field == "num_envs":
        raw[field] = 2048
    else:
        first = next(iter(raw[field]))
        raw[field][first] = "f" * 64
    result["measurement"] = copy.deepcopy(result["report"])
    with pytest.raises(ValueError):
        B.validate_probe_result(artifact, B.OPTIMIZER_PROBE, "f1_s2190", result)


def _fixed_fixture() -> dict:
    return json.loads((ROOT / "evals/results/e2e_reward_s2030_20261013/runs/"
                       "baseline_p0c_s2030.json").read_text(encoding="utf-8"))


def test_evaluate_returns_bound_full_fixed_reports_for_controller(tmp_path):
    obj = backend(tmp_path)
    fixed = ROOT / "bench/protocols/fixed_eval_v1.json"
    obj.protocol["fixed_eval_protocol"] = str(fixed)
    obj._base_job = lambda _stage, _mode: {"schema": "base"}
    report_path = tmp_path / "none.json"
    full = _fixed_fixture()
    report_path.write_text(json.dumps(full), encoding="utf-8")
    receipt = {"completed": True}
    obj._execute = lambda _stage, _mode, _job: (
        {"reports": {full["name"]: {"path": str(report_path), "sha256": B.sha256(report_path),
                                "metrics": full["metrics"]}}}, receipt)
    artifact = {"checkpoint": {"path": str(tmp_path / full["checkpoint"]["file"]),
                               "sha256": full["checkpoint"]["sha256"]}}
    result = obj.evaluate(tmp_path / "stage", {full["name"]: artifact}, "eval")
    assert result == {"reports": {full["name"]: full}, "execution": receipt}


@pytest.mark.parametrize("mutation", ["protocol", "condition", "checkpoint", "nan"])
def test_evaluate_rejects_malformed_fixed_report(tmp_path, mutation):
    obj = backend(tmp_path)
    fixed = ROOT / "bench/protocols/fixed_eval_v1.json"
    obj.protocol["fixed_eval_protocol"] = str(fixed)
    obj._base_job = lambda _stage, _mode: {"schema": "base"}
    full = copy.deepcopy(_fixed_fixture())
    if mutation == "protocol":
        full["protocol"] = {}
    elif mutation == "condition":
        full["by_condition"] = full["by_condition"][:1]
    elif mutation == "checkpoint":
        full["checkpoint"]["sha256"] = "0" * 64
    else:
        full["metrics"]["fall_rate"] = float("nan")
    report_path = tmp_path / f"{mutation}.json"
    report_path.write_text(json.dumps(full), encoding="utf-8")
    obj._execute = lambda _stage, _mode, _job: (
        {"reports": {full["name"]: {"path": str(report_path), "sha256": B.sha256(report_path),
                                       "metrics": full["metrics"]}}}, {"completed": True})
    artifact = {"checkpoint": {"path": str(tmp_path / "model_299.pt"),
                               "sha256": _fixed_fixture()["checkpoint"]["sha256"]}}
    with pytest.raises(ValueError):
        obj.evaluate(tmp_path / "stage", {full["name"]: artifact}, "eval")
