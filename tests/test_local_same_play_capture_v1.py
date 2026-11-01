from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import local_same_play_capture_v1 as C


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def source_batch(tmp_path: Path) -> tuple[Path, Path, str]:
    checkpoint = tmp_path / "model_299.pt"
    checkpoint.write_bytes(b"checkpoint")
    sha256 = C._sha(checkpoint)
    path = tmp_path / "source-result.json"
    write(path, {"schema": "prospective_diagnosis_batch_result_v1", "status": "complete",
                 "stages": [{"stage": "s2320-none", "checkpoint": {
                     "path": str(checkpoint), "sha256": sha256, "iteration": 299,
                     "size_bytes": checkpoint.stat().st_size}}]})
    return path, checkpoint, sha256


def protocol(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    batch, checkpoint, checkpoint_sha = source_batch(tmp_path)
    lab = tmp_path / "IsaacLab"
    pins = {}
    for relative in C.RUNTIME_PINS:
        target = lab / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
        pins[relative] = C._sha(target)
    training_relative = "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py"
    training_target = lab / training_relative
    training_target.parent.mkdir(parents=True, exist_ok=True)
    training_target.write_text("runner", encoding="utf-8")
    training_pins = {training_relative: C._sha(training_target, lf=True)}
    value = {
        "schema": C.SCHEMA, "status": "registered_before_local_gpu",
        "source_anchor": "a" * 40,
        "source_files_lf": {name: C._sha(C.ROOT / name, lf=True) for name in C.source_files()},
        "runtime_pins": pins, "training_runtime_pins": training_pins,
        "source_batch": {"logical_id": "prospective_source_batch_20261101_v1",
                         "sha256": C._sha(batch)},
        "checkpoint": {"sha256": checkpoint_sha, "logical_id": "s2320-none", "iteration": 299},
        "platform": {"os": "windows", "isaacsim": "4.5.0", "gpu": "RTX 3060"},
        "gpu": {"lease_helper": "Use-GpuLease.ps1", "owner": C.OWNER,
                "cloud_lease": False, "exclusive": True},
        "play": {"task": C.TASK, "num_envs": 1, "fps": 50, "full_frames": 625,
                 "smoke_frames": 1, "camera_profile": "same_play_fixed_v1",
                 "kit_args": C.KIT_ARGS, "new_playback_injection": False},
        "shutdown": {"smoke": "record_agentic_state_sidecar_v2",
                     "full": "vision_shutdown_policy_v1", "terminal_count": 1},
    }
    path = tmp_path / "protocol.json"
    write(path, value)
    inspection = tmp_path / "inspection.json"
    write(inspection, {"schema": C.INSPECTION_SCHEMA, "status": "cpu_input_shape_verified",
                       "source_batch": C._ref(batch), "checkpoint": C._ref(checkpoint),
                       "logical_id": "s2320-none", "runtime_pins": pins,
                       "training_runtime_pins": training_pins})
    return path, lab, inspection, value


def test_protocol_validates_source_cp_and_runtime_pins(tmp_path: Path) -> None:
    path, lab, inspection, value = protocol(tmp_path)
    assert C.validate_protocol(path, lab, inspection, verify_anchor=False) == value
    Path(C._read(inspection)["checkpoint"]["path"]).write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checkpoint/source binding"):
        C.validate_protocol(path, lab, inspection, verify_anchor=False)


def test_prepare_protocol_keeps_unregistered_candidate_non_executable(tmp_path: Path) -> None:
    batch, checkpoint, checkpoint_sha = source_batch(tmp_path)
    inspection = tmp_path / "inspection.json"
    write(inspection, {"schema": C.INSPECTION_SCHEMA, "status": "cpu_input_shape_verified",
                       "sdk_executed": False, "runtime_pins": {name: "a" * 64 for name in C.RUNTIME_PINS},
                       "training_runtime_pins": {"rsl.py": "b" * 64},
                       "source_batch": C._ref(batch), "checkpoint": C._ref(checkpoint),
                       "logical_id": "s2320-none"})
    candidate_path = tmp_path / "candidate.json"
    candidate = C.prepare_protocol(inspection, candidate_path)
    assert candidate["status"] == "candidate_not_registered" and candidate["source_anchor"] is None
    assert "bench/private" not in candidate_path.read_text(encoding="utf-8")
    assert "C:\\" not in candidate_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="schema/status"):
        C.validate_protocol(candidate_path, tmp_path / "lab", inspection, verify_anchor=False)
    assert candidate["checkpoint"]["sha256"] == checkpoint_sha
    assert {"bench/vision_pull_record_worker_v1.py", "bench/vision_pull_record_worker_v2.py",
            "bench/record_vision_evidence_v1.py", "evals/lineage_lease_context_v2.py"} <= set(
                candidate["source_files_lf"])


def test_protocol_blob_requires_ancestor_and_exact_registered_bytes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = tmp_path / "bench" / "protocols" / "local.json"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_bytes(b'{"registered":true}\r\n')
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if "merge-base" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(command, 0, b'{"registered":true}\n', b"")

    monkeypatch.setattr(C.subprocess, "run", run)
    C._verify_protocol_blob(protocol_path, tmp_path, "a" * 40, "b" * 40)
    assert calls[-1][-1] == "b" * 40 + ":bench/protocols/local.json"
    protocol_path.write_text('{"registered":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="protocol blob"):
        C._verify_protocol_blob(protocol_path, tmp_path, "a" * 40, "b" * 40)


def test_compose_full_worker_uses_worker_render_and_sidecar_chain(tmp_path: Path) -> None:
    calls: list[str] = []

    def base_maker(job, app):
        calls.append("capture")
        return (job, app)

    def render_maker(maker):
        calls.append("render")
        return maker

    def run_worker(validated, app, **kwargs):
        calls.append("worker")
        return {"ok": True, "factory": kwargs["capture_factory"]}

    def with_sidecar(*args, **kwargs):
        calls.append("sidecar")

        def selected(validated, app, **worker_kwargs):
            return kwargs["run_worker"](
                validated, app, capture_factory=kwargs["capture_factory_maker"], **worker_kwargs
            )

        return selected

    validated = {"sidecar_directory": str(tmp_path / "sidecar"), "source_id": "source-none",
                 "checkpoint_sha256": "a" * 64, "frames": 625}
    result = C.compose_full_worker(
        validated, object(), worker=SimpleNamespace(run_policy_record=run_worker),
        capture=SimpleNamespace(make_capture_factory=base_maker),
        render=SimpleNamespace(render_sync_capture_factory=render_maker),
        sidecar=SimpleNamespace(with_sidecar_worker=with_sidecar, ABORT="abort"),
        seal_report=lambda value: None, reader_factory=lambda base, index: None,
    )
    assert result["ok"] is True
    assert calls == ["render", "sidecar", "worker"]


def test_preclose_uses_worker_report_descriptors_and_registered_verifiers(tmp_path: Path) -> None:
    output, sidecar = tmp_path / "capture", tmp_path / "sidecar"
    output.mkdir()
    sidecar.mkdir()
    artifact_keys = {"video", "raw_trace", "encoder_receipt", "timing_receipt",
                     "measurement_1", "spec_1", "measurement_2", "spec_2"}
    descriptors = {}
    for name in artifact_keys:
        path = output / "evidence" / f"{name}.json"
        write(path, {"name": name})
        descriptors[name] = {"file": path.name, "sha256": C._sha(path), "bytes": path.stat().st_size}
    native = output / "native_capture_manifest.json"
    write(native, {"frame_count": 625})
    native_descriptor = {"file": native.name, "sha256": C._sha(native), "bytes": native.stat().st_size}
    worker = SimpleNamespace(
        SCHEMA="vision_pull_record_worker_v3", REPORT_NAME="native-policy-record.report.json",
        PREVIOUS=SimpleNamespace(RECORD_VERIFY=SimpleNamespace(
            _record_artifacts=lambda validated, artifacts: set(artifacts) == artifact_keys)))
    report = {"schema": worker.SCHEMA, "completed": True, "technical_recording_complete": True,
              "job_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
              "artifacts": descriptors, "native_capture_manifest": native_descriptor}
    write(output / worker.REPORT_NAME, report)
    write(sidecar / "manifest.json", {})
    write(sidecar / "records.jsonl", {})
    write(tmp_path / "render-sync.json", {})
    sidecar_module = SimpleNamespace(
        MANIFEST_NAME="manifest.json", RECORDS_NAME="records.jsonl",
        load_sidecar=lambda folder: ({}, []),
        join_with_trace=lambda folder, trace, manifest: {
            "schema": "joined", "joined": True, "frame_count": 625,
            "checkpoint_sha256": "b" * 64})
    render = SimpleNamespace(RENDER_SYNC_NAME="render-sync.json",
                             verify_render_sync=lambda stage, frames: {"calls": frames})
    validated = {"job_path": tmp_path / "job.json", "output_directory": output,
                 "sidecar_directory": sidecar, "job_sha256": "a" * 64,
                 "checkpoint_sha256": "b" * 64, "frames": 625}
    proof = C._verify_preclose_proof(validated, {
        "worker": worker, "sidecar": sidecar_module, "render": render})
    assert proof["status"] == "raw_proofs_verified"
    assert set(proof["proofs"]) == {"worker_report", "native_manifest", "raw_trace",
                                     "sidecar_manifest", "sidecar_records", "render_sync", "same_pts"}


def test_parent_finalizes_exit_zero_without_app_close_return_marker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"cp")
    output = tmp_path / "capture"
    output.mkdir()
    write(output / "result.json", {"completed": True})
    job = {"schema": C.JOB_SCHEMA, "mode": "full", "frames": 625, "fps": 50,
           "checkpoint_path": str(checkpoint), "checkpoint_sha256": C._sha(checkpoint),
           "checkpoint_logical_id": "s2320-none", "source_anchor": "a" * 40,
           "output_directory": str(output), "sidecar_directory": str(tmp_path / "sidecar"),
           "source": {"commit": "a" * 40, "files_lf": {}}, "job_path": str(tmp_path / "job.json")}
    write(tmp_path / "job.json", job)
    job_sha = C._sha(tmp_path / "job.json")
    write(tmp_path / "local-app-close-entered.json", {
        "schema": "local_same_play_app_close_v1", "status": "entered",
        "job_sha256": job_sha, "checkpoint_sha256": C._sha(checkpoint)})
    write(tmp_path / "shutdown-policy.json", {"completed": True})
    shutdown = SimpleNamespace(SIDECAR="shutdown-policy.json",
                               verify=lambda job_path, validated, terminal: {"completed": True})
    monkeypatch.setattr(C, "_verify_preclose_proof", lambda validated, modules: {"proofs": {}})
    monkeypatch.setattr(C.importlib, "import_module", lambda name: shutdown)
    execution = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                 "exit_code": 0, "cleanup_verified": True, "error_type": None}
    result = C.verify_result(tmp_path, {**job, "job_sha256": job_sha}, execution)
    assert result["app_close_returned"] is False and result["status"] == "technical_complete"


class RawApp:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = 0

    def update(self):
        return None

    def close(self):
        self.closed += 1
        if self.fail:
            raise RuntimeError("close failed")
        return "closed"


def terminal_validated(tmp_path: Path) -> dict:
    return {"output_directory": tmp_path / "capture", "job_path": tmp_path / "job.json",
            "job_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}


def test_full_terminal_closes_after_result_and_records_return(tmp_path: Path) -> None:
    validated = terminal_validated(tmp_path)
    write(validated["job_path"], {})
    raw = RawApp()
    app = C.FullAppProxy(raw, tmp_path, validated)

    def worker():
        Path(validated["output_directory"]).mkdir()
        return {"technical_recording_complete": True}

    states = []
    exit_code, _ = C.run_full_terminal(
        validated, app, worker,
        lambda report: states.append((report, (Path(validated["output_directory"]) / "result.json").exists())))
    assert exit_code == 0 and raw.closed == 1
    assert (Path(validated["output_directory"]) / "result.json").is_file()
    assert not (Path(validated["output_directory"]) / "failure.json").exists()
    assert C._read(tmp_path / "local-app-close-returned.json")["status"] == "returned"
    assert states == [({"technical_recording_complete": True}, False)]


def test_full_terminal_preserves_worker_and_close_failures(tmp_path: Path) -> None:
    validated = terminal_validated(tmp_path)
    write(validated["job_path"], {})
    raw = RawApp(fail=True)
    app = C.FullAppProxy(raw, tmp_path, validated)

    def worker():
        raise RuntimeError("worker cleanup failed")

    exit_code, report = C.run_full_terminal(validated, app, worker, lambda value: None)
    assert exit_code == 1 and report is None and raw.closed == 1
    failure = C._read(Path(validated["output_directory"]) / "failure.json")
    assert failure["error_message"] == "worker cleanup failed"
    assert C._read(tmp_path / "local-app-close-error.json")["error_message"] == "close failed"


def test_full_app_proxy_refuses_early_terminal_and_still_calls_close(tmp_path: Path) -> None:
    class PolicyProxy:
        def __init__(self) -> None:
            self.closed = 0

        def update(self):
            return None

        def close(self):
            self.closed += 1
            raise ValueError("terminal evidence가 정확히 하나여야 한다")

    policy = PolicyProxy()
    validated = terminal_validated(tmp_path)
    app = C.FullAppProxy(policy, tmp_path, validated)
    with pytest.raises(ValueError, match="terminal evidence"):
        app.close()
    assert policy.closed == 1
    assert C._read(tmp_path / "local-app-close-error.json")["status"] == "error"


def test_module_does_not_import_sdk_packages() -> None:
    assert "isaaclab" not in C.__dict__
    assert "torch" not in C.__dict__
    assert sys.version_info >= (3, 10)


def test_sdk_child_argv_reaches_child_with_kit_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[Path] = []
    monkeypatch.setattr(C, "sdk_child", lambda path: received.append(path) or 0)
    assert C.main(["sdk-child", "not-opened-job.json", *C.KIT_ARGS]) == 0
    assert received == [Path("not-opened-job.json")]


def test_documented_execute_argv_parses_inspection_for_smoke_and_full(
        monkeypatch: pytest.MonkeyPatch) -> None:
    modes: list[str] = []

    def fake_execute(protocol, anchor, lab, inspection, stage, isaac_python, *, mode):
        modes.append(mode)
        return {"status": "technical_complete"}

    monkeypatch.setattr(C, "execute", fake_execute)
    base = ["execute", "--protocol", "bench/protocols/local.json", "--anchor", "a" * 40,
            "--lab", "C:/IsaacLab", "--inspection", "private/inspection.json",
            "--isaac-python", "C:/IsaacLab/python.bat"]
    assert C.main([*base, "--stage", "private/smoke", "--mode", "smoke"]) == 0
    assert C.main([*base, "--stage", "private/full", "--mode", "full"]) == 0
    assert modes == ["smoke", "full"]


def test_execute_binds_same_lease_owned_child_and_full_proof(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_path = tmp_path / "protocol.json"
    write(protocol_path, {"registered": True})
    isaac_python = tmp_path / "python.bat"
    isaac_python.write_text("@echo off\n", encoding="utf-8")
    stage = tmp_path / "stage"
    lease_calls: list[Path | None] = []
    runner_calls: list[tuple[Path, list[str], Path, float]] = []

    inspection = tmp_path / "inspection.json"
    write(inspection, {})

    def fake_job(protocol, lab, actual_input, destination, *, mode, execution_anchor):
        destination.mkdir()
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"cp")
        job = {"mode": mode, "frames": 625, "fps": 50, "protocol": C._ref(protocol),
               "source_anchor": "a" * 40, "execution_anchor": execution_anchor,
               "checkpoint_path": str(checkpoint), "checkpoint_sha256": C._sha(checkpoint),
               "checkpoint_logical_id": "s2320-none", "output_directory": str(destination / "capture"),
               "sidecar_directory": str(destination / "sidecar"),
               "job_path": str(destination / "job.json"), "source_id": "local-none-full"}
        write(destination / "job.json", job)
        return {**job, "job_sha256": C._sha(destination / "job.json")}

    def lease_reader(evidence):
        lease_calls.append(evidence)
        return {"owner": C.OWNER, "state": "running", "leaseId": "lease-1"}

    def runner(folder, command, cwd, watchdog):
        runner_calls.append((folder, command, cwd, watchdog))
        return {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
                "exit_code": 0, "cleanup_verified": True, "error_type": None}

    monkeypatch.setattr(C, "build_job", fake_job)
    monkeypatch.setattr(C, "verify_result", lambda target, job, receipt: {"status": "technical_complete"})
    result = C.execute(protocol_path, "b" * 40, tmp_path, inspection, stage, isaac_python,
                       mode="full", runner=runner, lease_reader=lease_reader)
    assert result["status"] == "technical_complete" and result["lease"]["lease_id"] == "lease-1"
    assert len(lease_calls) == 2 and runner_calls[0][0] == stage / "owned"
    assert runner_calls[0][1][-2:] == C.KIT_ARGS and runner_calls[0][3] == 2400.0
    assert runner_calls[0][1][2] == "sdk-child"
    assert C._read(stage / "execution-result.json")["capture_approved"] is False
