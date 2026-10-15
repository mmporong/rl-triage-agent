from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import run_optimizer_observability_batch as B  # noqa: E402


def gpu(*, busy=False):
    return {
        "busy": busy,
        "busy_reasons": ["test_busy"] if busy else [],
        "admission_policy": B.GPU.EXPECTED_GPU_ADMISSION["policy"],
        "processes": [],
        "observed_at_utc": "2026-10-15T00:00:00+00:00",
    }


def fake_lab(tmp_path):
    lab = tmp_path / "IsaacLab"
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").write_text("test", encoding="utf-8")
    upstream = lab / "scripts/reinforcement_learning/rsl_rl/train.py"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("# test", encoding="utf-8")
    return lab


def fake_manifest(_epochs, _batches, _updates, **values):
    return {
        "arm": values["arm"], "seed": values["seed"], "task": values["task"],
        "num_envs": values["num_envs"], "max_iterations": values["max_iterations"],
        "source_commit_sha": "a" * 40,
    }


def install_lightweight_contract(monkeypatch):
    monkeypatch.setattr(B.INSPECT, "verify_commitment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(B, "inspect_records", lambda _manifest, _records: {"status": "normal"})
    monkeypatch.setattr(B, "validate_run_artifacts", lambda *_args: None)
    monkeypatch.setattr(B, "validate_plan", lambda plan: plan)


def test_plan_registers_exact_23_runs_and_fixed_development_scope():
    plan = B.build_plan("a" * 40)
    assert len(plan["runs"]) == 23
    assert len({row["run_id"] for row in plan["runs"]}) == 23
    assert [(row["arm"], row["seed"]) for row in plan["runs"][:2]] == [
        ("NONE", 2042), ("R1", 2042),
    ]
    assert {row["arm"] for row in plan["runs"][2:]} == {"NONE", "h01", "h02", "h03", "h04", "h05", "h06"}
    assert {row["seed"] for row in plan["runs"][2:]} == {2026, 2027, 2028}
    assert plan["measurement_kind"] == "development"
    assert plan["stop_on_first_failure"] is True and plan["tag_write_once"] is True


def test_busy_gpu_rejects_before_tag_is_consumed(tmp_path):
    lab = fake_lab(tmp_path)
    with pytest.raises(RuntimeError, match="GPU가 사용 중"):
        B.run_batch("busy", lab, tmp_path / "out", gpu_probe=lambda _profile: gpu(busy=True),
                    source_verifier=lambda: "a" * 40)
    assert not (tmp_path / "out/busy").exists()


def test_malformed_gpu_snapshot_rejects_before_launch(tmp_path):
    lab = fake_lab(tmp_path)
    with pytest.raises(ValueError, match="GPU 입장 검사 필드"):
        B.run_batch("badgpu", lab, tmp_path / "out", gpu_probe=lambda _profile: {"busy": False},
                    source_verifier=lambda: "a" * 40)
    assert not (tmp_path / "out/badgpu").exists()


def test_first_execution_failure_stops_and_consumes_tag(tmp_path, monkeypatch):
    lab = fake_lab(tmp_path)
    install_lightweight_contract(monkeypatch)
    launched = []

    def fail_first(folder, _args, _cwd, _watchdog):
        launched.append(folder.name)
        return {"job_assigned_before_release": True, "natural_exit": False, "timed_out": False,
                "error_type": None, "exit_code": 1, "cleanup_verified": True,
                "owned_process_identities": {}}

    with pytest.raises(RuntimeError, match="owned 실행"):
        B.run_batch("failed", lab, tmp_path / "out", gpu_probe=lambda _profile: gpu(),
                    owned_runner=fail_first, source_verifier=lambda: "a" * 40,
                    manifest_builder=fake_manifest,
                    record_inspector=lambda *_args: {"status": "normal"})
    assert launched == ["none_s2042"]
    assert (tmp_path / "out/failed/runs/none_s2042/failure.json").is_file()
    with pytest.raises(FileExistsError, match="소비된 tag"):
        B.run_batch("failed", lab, tmp_path / "out", gpu_probe=lambda _profile: gpu(),
                    source_verifier=lambda: "a" * 40)


def test_owned_gpu_process_remaining_stops_before_inspection(tmp_path, monkeypatch):
    lab = fake_lab(tmp_path)
    install_lightweight_contract(monkeypatch)
    snapshots = iter((gpu(), gpu(), {**gpu(), "processes": [{"pid": 100}]}))

    def succeed(_folder, _args, _cwd, _watchdog):
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                "error_type": None, "exit_code": 0, "cleanup_verified": True,
                "owned_process_identities": {"100": 200}}

    with pytest.raises(RuntimeError, match="owned GPU process"):
        B.run_batch("residual", lab, tmp_path / "out", gpu_probe=lambda _profile: next(snapshots),
                    owned_runner=succeed, source_verifier=lambda: "a" * 40,
                    manifest_builder=fake_manifest,
                    record_inspector=lambda *_args: pytest.fail("잔류 프로세스 뒤 inspect 실행"))
    failure = json.loads((tmp_path / "out/residual/runs/none_s2042/failure.json").read_text(encoding="utf-8"))
    assert failure["type"] == "RuntimeError"


def test_success_runs_sequentially_and_writes_verified_summary(tmp_path, monkeypatch):
    lab = fake_lab(tmp_path)
    install_lightweight_contract(monkeypatch)
    launched = []

    def succeed(folder, args, cwd, _watchdog):
        assert cwd == lab
        launched.append(folder.name)
        B.write_new(folder / "launch.json", {"args": args, "cwd": str(cwd)})
        records = Path(args[args.index("--records") + 1])
        records.write_text("{}\n", encoding="utf-8")
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                "error_type": None, "exit_code": 0, "cleanup_verified": True,
                "owned_process_identities": {"100": 200}}

    def inspect(manifest, _records):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return {"status": "abnormal" if data["arm"] == "R1" else "normal"}

    monkeypatch.setattr(B, "inspect_records", inspect)
    summary = B.run_batch("complete", lab, tmp_path / "out", gpu_probe=lambda _profile: gpu(),
                          owned_runner=succeed, source_verifier=lambda: "a" * 40,
                          manifest_builder=fake_manifest, record_inspector=inspect)
    assert launched == [run.run_id for run in B.registered_runs()]
    assert summary["run_count"] == 23 and summary["status"] == "complete"
    assert summary["r1_pair"] == {"none_s2042": "normal", "r1_s2042": "abnormal"}
    assert summary["specificity_status_counts"] == {"normal": 21, "abnormal": 0, "unknown": 0}
    assert (tmp_path / "out/complete/summary.json").is_file()
    launch_path = tmp_path / "out/complete/runs/none_s2042/launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    launch["args"].append("--unregistered")
    launch_path.write_text(json.dumps(launch), encoding="utf-8")
    with pytest.raises(ValueError, match="SDK launch argv/cwd"):
        B.aggregate(tmp_path / "out/complete", json.loads(
            (tmp_path / "out/complete/batch_plan.json").read_text(encoding="utf-8")))


def test_unknown_report_stops_but_abnormal_is_a_valid_observation(tmp_path):
    manifest = tmp_path / "manifest.json"
    records = tmp_path / "records.jsonl"
    manifest.write_text(json.dumps({"runtime_updates_to_observe": 3}), encoding="utf-8")
    records.write_text("".join(json.dumps({"update_index": index, "error": None}) + "\n" for index in range(3)),
                       encoding="utf-8")
    B.validate_run_artifacts(manifest, records, {"status": "abnormal", "records": 3,
                                                 "primitives": {"approx_kl": {"status": "unavailable"}}})
    with pytest.raises(ValueError, match="완결되지 않았다"):
        B.validate_run_artifacts(manifest, records, {"status": "unknown", "records": 3})


def test_incomplete_or_error_record_stops_batch_artifact_acceptance(tmp_path):
    manifest = tmp_path / "manifest.json"
    records = tmp_path / "records.jsonl"
    manifest.write_text(json.dumps({"runtime_updates_to_observe": 3}), encoding="utf-8")
    records.write_text(json.dumps({"update_index": 0, "error": None}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="고정 3개"):
        B.validate_run_artifacts(manifest, records, {"status": "normal", "records": 3})
    records.write_text("".join(json.dumps({"update_index": index, "error": {"type": "Boom"} if index == 1 else None}) + "\n"
                               for index in range(3)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="학습 예외"):
        B.validate_run_artifacts(manifest, records, {"status": "unknown", "records": 3})


def test_plan_validation_rejects_missing_registered_run(monkeypatch):
    plan = B.build_plan("a" * 40)
    monkeypatch.setattr(B, "git_head", lambda: "a" * 40)
    plan["runs"].pop()
    with pytest.raises(ValueError, match="고정 23개"):
        B.validate_plan(plan)


def test_aggregate_rejects_missing_run(tmp_path, monkeypatch):
    install_lightweight_contract(monkeypatch)
    plan = B.build_plan("a" * 40)
    tag = tmp_path / "tag"
    (tag / "runs").mkdir(parents=True)
    B.write_new(tag / "batch_plan.json", plan)
    B.write_new(tag / "gpu_admission_initial.json", gpu())
    with pytest.raises(ValueError, match="필수 산출물 누락"):
        B.aggregate(tag, plan)


def test_run_rejects_uncommitted_batch_source_before_gpu(monkeypatch, tmp_path):
    lab = fake_lab(tmp_path)
    monkeypatch.setattr(B, "verify_committed_sources", lambda: (_ for _ in ()).throw(ValueError("커밋 필요")))
    called = False

    def no_gpu(_profile):
        nonlocal called
        called = True
        return gpu()

    with pytest.raises(ValueError, match="커밋 필요"):
        B.run_batch("uncommitted", lab, tmp_path / "out", gpu_probe=no_gpu)
    assert called is False
