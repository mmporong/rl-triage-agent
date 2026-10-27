from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vision_transferred_probe_v1_test", ROOT / "evals/vision_transferred_probe_v1.py")
assert SPEC is not None and SPEC.loader is not None
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _expected() -> dict:
    return {
        "source_anchor": "1" * 40,
        "protocol": {"path": "runtime.json", "sha256_lf": "2" * 64},
        "environment": {
            "python": "3.10.12", "isaaclab_sha": "3" * 40,
            "isaacsim": "4.5.0.0", "isaaclab": "0.41.3", "isaaclab_rl": "0.1.8",
            "torch": "2.7.0+cu128", "rsl-rl-lib": "2.3.3", "gymnasium": "1.2.0",
            "cuda_available": True, "cuda_device_count": 1,
        },
        "source_sha256_lf": {"bench/source.py": "4" * 64},
        "runtime_source_sha256_lf": {"lab/source.py": "5" * 64},
        "python_package_source_sha256_lf": {"rsl/source.py": "6" * 64},
        "task": "Isaac-Velocity-Flat-Unitree-Go2-v0",
    }


def _six(root: Path, stage: Path, *, arm: str, seed: int, checkpoint: str,
         measurement: dict) -> Path:
    expected = _expected()
    stage.mkdir(parents=True, exist_ok=False)
    side = "reference" if arm == "NONE" else "case"
    run_name = f"probe_{seed}_{arm.lower()}"
    report_path = stage / "report.json"
    checkpoint_path = stage / "model.pt"
    checkpoint_path.write_bytes(arm.encode())
    source = {
        "lab_head": expected["environment"]["isaaclab_sha"], "lab_tracked_clean": True,
        "source_anchor": expected["source_anchor"], "protocol": expected["protocol"],
        "environment": expected["environment"],
        "selected_interpreter": {"python": expected["environment"]["python"],
            "packages": {key: value for key, value in expected["environment"].items()
                         if key not in {"python", "isaaclab_sha", "cuda_available", "cuda_device_count"}}},
        "source_sha256_lf": expected["source_sha256_lf"],
        "runtime_source_sha256_lf": expected["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": expected["python_package_source_sha256_lf"],
    }
    job = {
        "schema": "brev_recovery_worker_job_v1", "mode": "probe", "stage": str(stage),
        "lab": "/runtime/IsaacLab", "repo": "/runtime/repo",
        "protocol_path": "/runtime/repo/runtime.json", "protocol_sha256_lf": "2" * 64,
        "source_commit_sha": "1" * 40, "source_sha256_lf": expected["source_sha256_lf"],
        "runtime_source_sha256_lf": expected["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": expected["python_package_source_sha256_lf"],
        "arm": arm, "seed": seed, "run_name": run_name, "probe": P.PROBE,
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint,
        "report_path": str(report_path), "training_manifest_path": str(stage / "training.json"),
        "inspection_manifest_path": str(stage / "inspection.json"),
        "records_path": str(stage / "records.jsonl"), "readback_path": str(stage / "readback.json"),
    }
    report = {"name": run_name, "checkpoint": {"file": "model.pt", "sha256": checkpoint},
              "task": expected["task"], "seed": seed, "num_envs": 64,
              "probes": {P.PROBE: {"measurement": measurement, "exit": 0, "elapsed_s": 1.0}}}
    jobs = {"fault": arm, "seed": seed, "task": expected["task"],
            "jobs": [{"name": run_name, "checkpoint": str(checkpoint_path),
                      "probes": [P.PROBE], "output": str(report_path)}]}
    execution = {"schema": "brev_recovery_posix_execution_v1", "mode": "probe",
                 "completed": True, "started_at": "2026-10-27T00:00:00+00:00",
                 "finished_at": "2026-10-27T00:00:02+00:00", "exit_code": 0,
                 "timed_out": False, "cleanup_verified": True, "launch": {"session_id": 7},
                 "gpu_before": [], "gpu_after": [], "wall_s": 2.0,
                 "source_before": source, "source_after": source}
    paths = {name: stage / filename for name, filename in {
        "job": "job.json", "worker": "worker.json", "execution": "execution.json",
        "backend_result": "backend_result.json", "report_jobs": "report.jobs.json",
        "report": "report.json"}.items()}
    _write(paths["job"], job); _write(paths["report"], report); _write(paths["report_jobs"], jobs)
    worker = {"schema": "brev_recovery_probe_worker_v1", "completed": True,
              "probe": P.PROBE, "measurement": measurement,
              "report": {"path": str(report_path), "sha256": P.file_sha(paths["report"])},
              "inspection_manifest": None, "wall_s": 1.0}
    _write(paths["worker"], worker); _write(paths["execution"], execution)
    _write(paths["backend_result"], {"execution": execution, "measurement": measurement,
                                     "report": report})
    descriptor = {"schema": "vision_selection_brev_probe_evidence_v1", "side": side,
                  "probe": P.PROBE, "checkpoint_sha256": checkpoint,
                  "sources": {name: P._ref(root, path) for name, path in paths.items()}}
    descriptor_path = stage / "descriptor.json"
    _write(descriptor_path, descriptor)
    return descriptor_path


def test_protocol_is_preregistered_draft_with_two_exact_pairs() -> None:
    protocol = P.validate_protocol(ROOT / "bench/protocols/vision_transferred_probe_v1.json")
    assert [row["seed"] for row in protocol["groups"]] == [2220, 2221]
    assert protocol["execution_root"] == ".triage-runtime/vision-transferred-probe-v1"
    assert protocol["limits"] == {"probe_pairs": 2, "gpu_probe_runs": 4, "total_gpu_s": 600,
                                  "pair_gpu_s": 300, "api_requests": 0, "training_runs": 0}
    with pytest.raises(ValueError, match="committed"):
        P.validate_protocol(ROOT / "bench/protocols/vision_transferred_probe_v1.json",
                            require_bound=True)


def test_actual_six_source_consumer_and_pair_receipt_replay(tmp_path: Path, monkeypatch) -> None:
    group = {"group": "source-01", "seed": 2220,
             "checkpoints": {"NONE": {"sha256": "a" * 64}, "F1": {"sha256": "b" * 64}},
             "execution": {"NONE": {"stage": "none", "run_name": "probe_2220_none"},
                           "F1": {"stage": "f1", "run_name": "probe_2220_f1"}}}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, {"identity": "fixture"})
    protocol = {"_path": str(protocol_path), "groups": [group],
                "probe_runtime": {"source_anchor": "1" * 40,
                                  "source_protocol": {"path": "runtime.json", "sha256_lf": "2" * 64}}}
    source_inputs = {"runtime_protocol": {
        "environment": {key: value for key, value in _expected()["environment"].items()
                        if key not in {"cuda_available", "cuda_device_count"}},
        "source_sha256_lf": _expected()["source_sha256_lf"],
        "runtime_source_sha256_lf": _expected()["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": _expected()["python_package_source_sha256_lf"]},
        "groups": {"source-01": {"arms": {"NONE": {"body": {"task": _expected()["task"]}}}}}}
    output = tmp_path / "pair"
    output.mkdir()
    auth = P.authorize_pair(output, group, P.file_sha(protocol_path), "delegated_operator")
    normal = {"track_lin_vel_xy_exp_rel_error": 0.0,
              "track_ang_vel_z_exp_rel_error": 0.0}
    abnormal = {"track_lin_vel_xy_exp_rel_error": 0.6,
                "track_ang_vel_z_exp_rel_error": 0.7}
    none = _six(tmp_path, output / "none", arm="NONE", seed=2220,
                checkpoint="a" * 64, measurement=normal)
    fault = _six(tmp_path, output / "f1", arm="F1", seed=2220,
                 checkpoint="b" * 64, measurement=abnormal)
    expected = _expected()
    assert P.CONSUMER._brev_actual_probe(
        tmp_path, P.read(none), side="reference", probe=P.PROBE,
        checkpoint_sha256="a" * 64, expected_source=expected)["measurement"] == normal
    assert P.CONSUMER._brev_actual_probe(
        tmp_path, P.read(fault), side="case", probe=P.PROBE,
        checkpoint_sha256="b" * 64, expected_source=expected)["measurement"] == abnormal
    outcomes = {"NONE": "normal", "F1": "abnormal"}
    descriptors = {"NONE": P._ref(tmp_path, none), "F1": P._ref(tmp_path, fault)}
    P._append(auth["ledger"], {"schema": P.LEDGER_SCHEMA, "event": "receipt",
              "request_id": auth["request_id"], "request_sha256": P.digest(auth["request"]),
              "at": "2026-10-27T00:00:03+00:00", "outcomes": outcomes,
              "gpu_s": 4.0, "descriptors": descriptors})
    monkeypatch.setattr(P, "expected_probe_source", lambda *_: expected)
    result = P.validate_pair(tmp_path, protocol, source_inputs, output)
    assert result["usable"] is True
    assert result["outcomes"] == outcomes
    assert [row["event"] for row in P._ledger(auth["ledger"])] == [
        "proposed", "approved", "consumed", "receipt"]


def test_pair_replay_rejects_descriptor_tamper(tmp_path: Path) -> None:
    stage = tmp_path / "case"
    descriptor = _six(tmp_path, stage, arm="F1", seed=2220,
                      checkpoint="b" * 64,
                      measurement={"track_lin_vel_xy_exp_rel_error": 0.6,
                                   "track_ang_vel_z_exp_rel_error": 0.7})
    execution = P.read(stage / "execution.json")
    execution["cleanup_verified"] = False
    _write(stage / "execution.json", execution)
    body = P.read(descriptor)
    body["sources"]["execution"] = P._ref(tmp_path, stage / "execution.json")
    _write(descriptor, body)
    with pytest.raises(ValueError, match="Brev actual probe"):
        P.CONSUMER._brev_actual_probe(
            tmp_path, P.read(descriptor), side="case", probe=P.PROBE,
            checkpoint_sha256="b" * 64, expected_source=_expected())


def test_pair_replay_rejects_approval_prefix_tamper(tmp_path: Path, monkeypatch) -> None:
    group = {"group": "source-01", "seed": 2220,
             "checkpoints": {"NONE": {"sha256": "a" * 64}, "F1": {"sha256": "b" * 64}},
             "execution": {"NONE": {"stage": "none", "run_name": "probe_2220_none"},
                           "F1": {"stage": "f1", "run_name": "probe_2220_f1"}}}
    protocol_path = tmp_path / "protocol.json"
    _write(protocol_path, {"identity": "fixture"})
    protocol = {"_path": str(protocol_path), "groups": [group],
                "probe_runtime": {"source_anchor": "1" * 40,
                                  "source_protocol": {"path": "runtime.json", "sha256_lf": "2" * 64}}}
    source_inputs = {"runtime_protocol": {
        "environment": {key: value for key, value in _expected()["environment"].items()
                        if key not in {"cuda_available", "cuda_device_count"}},
        "source_sha256_lf": _expected()["source_sha256_lf"],
        "runtime_source_sha256_lf": _expected()["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": _expected()["python_package_source_sha256_lf"]},
        "groups": {"source-01": {"arms": {"NONE": {"body": {"task": _expected()["task"]}}}}}}
    output = tmp_path / "pair"
    output.mkdir()
    auth = P.authorize_pair(output, group, P.file_sha(protocol_path), "delegated_operator")
    rows = P._ledger(auth["ledger"])
    rows.append({"schema": P.LEDGER_SCHEMA, "event": "receipt",
                 "request_id": auth["request_id"],
                 "request_sha256": P.digest(auth["request"]),
                 "at": "2026-10-27T00:00:03+00:00"})
    rows[1]["approver"] = ""
    auth["ledger"].write_text("".join(P.canonical(row).decode("utf-8") + "\n" for row in rows),
                                     encoding="utf-8")
    monkeypatch.setattr(P, "expected_probe_source", lambda *_: _expected())
    with pytest.raises(ValueError, match="request가 protocol group"):
        P.validate_pair(tmp_path, protocol, source_inputs, output)
