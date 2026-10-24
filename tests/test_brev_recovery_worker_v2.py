from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("brev_recovery_worker_v2_test", ROOT / "bench/brev_recovery_worker_v2.py")
assert SPEC is not None and SPEC.loader is not None
W = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(W)


def _hashes() -> dict[str, str]:
    return {"x.py": "a" * 64}


def _train_job(tmp_path: Path, *, arm: str = "F1") -> dict:
    repo, lab, stage = tmp_path / "repo", tmp_path / "lab", tmp_path / "stage"
    repo.mkdir(); lab.mkdir(); stage.mkdir()
    protocol = repo / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    return {
        "schema": "brev_recovery_worker_job_v1", "mode": "train",
        "stage": str(stage), "lab": str(lab), "repo": str(repo),
        "protocol_path": str(protocol), "protocol_sha256_lf": "b" * 64,
        "source_commit_sha": "c" * 40, "source_sha256_lf": _hashes(),
        "runtime_source_sha256_lf": _hashes(), "python_package_source_sha256_lf": _hashes(),
        "arm": arm, "seed": 2190, "run_name": "recovery-s2190", "iterations": 300,
        "manifest_path": str(stage / "manifest.json"), "records_path": str(stage / "records.jsonl"),
        "readback_path": str(stage / "readback.json"), "telemetry_path": str(stage / "telemetry.json"),
        "source_checkpoint_sha256": "d" * 64 if arm == "CORRECTED" else None,
    }


def test_validate_job_rejects_stage_escape(tmp_path):
    job = _train_job(tmp_path)
    job["telemetry_path"] = str(tmp_path / "escaped.json")
    with pytest.raises(ValueError, match="벗어났다"):
        W.validate_job(job)


def _install_fake_runner(monkeypatch, tmp_path: Path, *, fail: bool = False):
    events = []

    class Runner:
        current_learning_iteration = 0

        def __init__(self):
            self.log_dir = str(tmp_path / "run")

        def learn(self, num_learning_iterations, **_kwargs):
            events.append("learn")
            if fail:
                raise RuntimeError("hidden path /home/person")
            self.current_learning_iteration = num_learning_iterations - 1
            run = Path(self.log_dir)
            run.mkdir()
            (run / "model_299.pt").write_bytes(b"checkpoint")
            return "ok"

    runners = ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = Runner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rsl_rl", ModuleType("rsl_rl"))
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners)
    return Runner, events


def test_train_writes_completion_after_learn_before_false_zero_close(tmp_path, monkeypatch):
    job = _train_job(tmp_path)
    output = Path(job["stage"]) / "worker.json"
    runner, events = _install_fake_runner(monkeypatch, tmp_path)
    readback = Path(job["readback_path"])

    def fake_manifest(_job, **_kwargs):
        return {"schema": "fixture"}

    def fake_execute(_manifest):
        readback.write_text(json.dumps({"remaining_faults": ["F1"]}), encoding="utf-8")
        Path(job["records_path"]).write_text("{}\n", encoding="utf-8")
        runner().learn(num_learning_iterations=300)
        assert output.exists()
        raise SystemExit(0)

    def fake_telemetry(_run, path):
        body = {"summary": {"loss": {"n": 300}}, "series": {"loss": [1.0]}}
        path.write_text(json.dumps(body), encoding="utf-8")
        return body

    monkeypatch.setattr(W, "_residual_manifest", fake_manifest)
    monkeypatch.setattr(W, "_telemetry", fake_telemetry)
    with pytest.raises(SystemExit) as error:
        W.run_train(job, output, execute=fake_execute)
    assert error.value.code == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["completed"] is True
    assert receipt["iterations"] == 300 and events == ["learn"]
    assert receipt["checkpoint"]["sha256"] == W.sha256(tmp_path / "run/model_299.pt")


def test_train_failure_cannot_create_completed_receipt(tmp_path, monkeypatch):
    job = _train_job(tmp_path)
    output = Path(job["stage"]) / "worker.json"
    runner, _ = _install_fake_runner(monkeypatch, tmp_path, fail=True)

    monkeypatch.setattr(W, "_residual_manifest", lambda *_args, **_kwargs: {"schema": "fixture"})

    def fake_execute(_manifest):
        runner().learn(num_learning_iterations=300)

    with pytest.raises(RuntimeError):
        W.run_train(job, output, execute=fake_execute)
    assert not output.exists()


def _dry_job(tmp_path: Path) -> dict:
    job = _train_job(tmp_path, arm="NONE")
    job["iterations"] = 20
    return job


def _fake_compute(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "fake_compute.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_dry_train_parent_finishes_artifacts_after_child_hard_exit(tmp_path, monkeypatch):
    job = _dry_job(tmp_path)
    output = Path(job["stage"]) / "worker.json"
    body = (
        "import argparse,json,os\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for n in ('mode','output','isaaclab','seed','num_envs','iterations','run_name','timeout_s'): "
        "p.add_argument('--'+n.replace('_','-'))\n"
        "a=p.parse_args()\n"
        "assert a.mode=='train' and a.num_envs=='4096' and a.iterations=='20' and a.timeout_s=='1800'\n"
        "cp=Path(a.output).parent/'logs'/'model_19.pt'\n"
        "cp.parent.mkdir();cp.write_bytes(b'dry-checkpoint')\n"
        "Path(a.output).write_text(json.dumps({'completed':True,'observed_iterations':20,"
        "'checkpoint_path':str(cp),'learn_wall_s':14.0,'wall_s':15.0}))\n"
        "os._exit(0)\n"
    )
    monkeypatch.setattr(W, "COMPUTE", _fake_compute(tmp_path, body))

    def fake_telemetry(_run, path):
        value = {"summary": {"loss": {"n": 20}}, "series": {"loss": [1.0]}}
        path.write_text(json.dumps(value), encoding="utf-8")
        return value

    monkeypatch.setattr(W, "_telemetry", fake_telemetry)
    W.run_train(job, output)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["completed"] is True and receipt["iterations"] == 20
    assert receipt["checkpoint"]["sha256"] == W.sha256(Path(receipt["checkpoint"]["path"]))
    assert Path(receipt["manifest_path"]).is_file()
    assert Path(job["readback_path"]).is_file()
    assert Path(job["telemetry_path"]).is_file()


@pytest.mark.parametrize("case", ["nonzero", "missing_receipt"])
def test_dry_train_rejects_failed_or_receiptless_child(tmp_path, monkeypatch, case):
    job = _dry_job(tmp_path)
    output = Path(job["stage"]) / "worker.json"
    code = 7 if case == "nonzero" else 0
    monkeypatch.setattr(W, "COMPUTE", _fake_compute(tmp_path, f"import os;os._exit({code})\n"))
    expected = W.subprocess.CalledProcessError if case == "nonzero" else FileNotFoundError
    with pytest.raises(expected):
        W.run_train(job, output)
    assert not output.exists()
    assert not Path(job["manifest_path"]).exists()


def test_corrected_arm_requires_empty_remaining_faults(tmp_path, monkeypatch):
    job = _train_job(tmp_path, arm="CORRECTED")
    output = Path(job["stage"]) / "worker.json"
    runner, _ = _install_fake_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(W, "_residual_manifest", lambda *_args, **_kwargs: {"schema": "fixture"})
    monkeypatch.setattr(W, "_telemetry", lambda _run, path: (
        path.write_text(json.dumps({"summary": {}, "series": {}}), encoding="utf-8")
        or {"summary": {}, "series": {}}
    ))

    def bad_execute(_manifest):
        Path(job["readback_path"]).write_text(json.dumps({"remaining_faults": ["F1"]}), encoding="utf-8")
        Path(job["records_path"]).write_text("{}\n", encoding="utf-8")
        runner().learn(num_learning_iterations=300)

    with pytest.raises(ValueError, match="readback"):
        W.run_train(job, output, execute=bad_execute)
    assert not output.exists()


def test_main_accepts_false_zero_close_only_after_completed_receipt(tmp_path, monkeypatch):
    job = _train_job(tmp_path)
    job_path, output = tmp_path / "job.json", tmp_path / "worker.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")

    def complete_then_close(_job, path):
        W.write_new(path, {"completed": True})
        raise SystemExit(0)

    monkeypatch.setattr(W, "run_train", complete_then_close)
    assert W.main(["--job", str(job_path), "--output", str(output)]) == 0

    output.unlink()

    def close_without_receipt(_job, _path):
        raise SystemExit(0)

    monkeypatch.setattr(W, "run_train", close_without_receipt)
    assert W.main(["--job", str(job_path), "--output", str(output)]) == 1


def test_optimizer_probe_worker_returns_the_used_inspection_manifest_descriptor(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    inspection_path = stage / "inspection.manifest.json"
    records_path = stage / "records.jsonl"
    report_path = stage / "report.json"
    output = stage / "worker.json"
    inspection = {"schema": "fixture", "source_checkpoint_sha256": "a" * 64}
    measurement = {"schema": "optimizer_observability_inspection_v1", "status": "normal"}
    job = {"probe": W.OPTIMIZER_PROBE, "inspection_manifest_path": str(inspection_path),
           "records_path": str(records_path), "readback_path": str(stage / "readback.json"),
           "report_path": str(report_path)}
    import residual_training_v1 as training

    monkeypatch.setattr(W, "_residual_manifest", lambda *_args, **_kwargs: inspection)
    monkeypatch.setattr(W, "_run_checked", lambda _argv: records_path.write_text("{}\n", encoding="utf-8"))
    monkeypatch.setattr(training, "observer_manifest", lambda _inspection: {"schema": "observer"})
    monkeypatch.setattr(training.OBSERVE, "assemble_inspection", lambda _rows, _manifest: measurement)

    W.run_probe(job, output)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["inspection_manifest"] == {
        "path": str(inspection_path), "sha256": W.sha256(inspection_path),
    }
    assert json.loads(report_path.read_text(encoding="utf-8")) == measurement


def _fixed_fixture() -> dict:
    return json.loads((ROOT / "evals/results/e2e_reward_s2030_20261013/runs/"
                       "baseline_p0c_s2030.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("mutation", [None, "protocol", "condition", "checkpoint", "nan"])
def test_evaluate_validates_complete_fixed_report(tmp_path, monkeypatch, mutation):
    repo, lab, stage = tmp_path / "repo", tmp_path / "lab", tmp_path / "stage"
    repo.mkdir(); lab.mkdir(); stage.mkdir()
    fixed_protocol = repo / "fixed_eval_v1.json"
    fixed_protocol.write_bytes((ROOT / "bench/protocols/fixed_eval_v1.json").read_bytes())
    protocol = repo / "brev.json"
    protocol.write_text("{}", encoding="utf-8")
    checkpoint = stage / "model_299.pt"
    checkpoint.write_bytes(b"fixture")
    fixture = copy.deepcopy(_fixed_fixture())
    if mutation == "protocol":
        fixture["protocol"] = {}
    elif mutation == "condition":
        fixture["by_condition"] = fixture["by_condition"][:1]
    elif mutation == "checkpoint":
        fixture["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "nan":
        fixture["metrics"]["fall_rate"] = float("nan")
    artifact_sha = _fixed_fixture()["checkpoint"]["sha256"]
    job = {
        "schema": "brev_recovery_worker_job_v1", "mode": "evaluate",
        "stage": str(stage), "lab": str(lab), "repo": str(repo),
        "protocol_path": str(protocol), "protocol_sha256_lf": "b" * 64,
        "source_commit_sha": "c" * 40, "source_sha256_lf": _hashes(),
        "runtime_source_sha256_lf": _hashes(), "python_package_source_sha256_lf": _hashes(),
        "artifacts": {fixture["name"]: {"checkpoint": {"path": str(checkpoint), "sha256": artifact_sha}}},
        "fixed_eval_protocol": str(fixed_protocol), "reports_dir": str(stage / "reports"),
        "jobs_path": str(stage / "jobs.json"),
    }
    output = stage / "worker.json"
    real_sha = W.sha256

    def fake_sha(path, *, lf=False):
        return artifact_sha if Path(path) == checkpoint else real_sha(Path(path), lf=lf)

    def fake_run(_argv):
        jobs = json.loads(Path(job["jobs_path"]).read_text(encoding="utf-8"))
        Path(jobs["jobs"][0]["output"]).write_text(json.dumps(fixture), encoding="utf-8")

    monkeypatch.setattr(W, "sha256", fake_sha)
    monkeypatch.setattr(W, "_run_checked", fake_run)
    if mutation is None:
        W.run_evaluate(job, output)
        assert json.loads(output.read_text(encoding="utf-8"))["completed"] is True
    else:
        with pytest.raises(ValueError):
            W.run_evaluate(job, output)
        assert not output.exists()
