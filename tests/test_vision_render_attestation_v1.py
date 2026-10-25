from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "tests"):
    sys.path.insert(0, str(folder))
A = importlib.import_module("vision_render_attestation_v1")
C = importlib.import_module("vision_render_context_v1")
CF = importlib.import_module("test_vision_render_context_v1")


def _sha(path: Path, *, lf: bool = False) -> str:
    data = path.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _case(tmp_path: Path):
    job_path, snapshot, python, job = CF._fixture(tmp_path)
    repo = Path(job["source"]["repo_root"])
    wrapper = repo / A.WRAPPER_RELATIVE
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    assert A.__file__ is not None
    wrapper.write_bytes(Path(A.__file__).read_bytes())
    wrapper_sha = _sha(wrapper, lf=True)
    snapshot["source_sha256_lf"][A.WRAPPER_RELATIVE] = wrapper_sha
    job["source"]["files_lf"][A.WRAPPER_RELATIVE] = wrapper_sha
    record_protocol = repo / "bench/protocols/record.json"
    _write(record_protocol, {
        "source_commit": snapshot["source_anchor"],
        "source_files_lf": job["source"]["files_lf"],
    })
    job["protocol"] = {"path": str(record_protocol.resolve()), "sha256": _sha(record_protocol)}
    job["output_directory"] = str((tmp_path / "record-stage/output").resolve())
    _write(job_path, job)
    return job_path, snapshot, python, job


def _outputs(job: dict, job_sha: str, snapshot: dict, *, exit_code: int = 0):
    output = Path(job["output_directory"])
    output.mkdir(parents=True)
    result_path = output / "result.json"
    _write(result_path, {
        "schema": "vision_play_record_result_v1", "completed": True,
        "job_sha256": job_sha,
    })
    result_sha = _sha(result_path)
    execution_path = output.parent / "execution.json"
    _write(execution_path, {
        "schema": "vision_brev_record_execution_v1", "completed": True,
        "job_sha256": job_sha, "exit_code": exit_code, "timed_out": False,
        "cleanup_verified": True, "gpu_before": [], "gpu_after": [],
        "gpu_before_observation": {}, "gpu_after_observation": {},
        "source_before": snapshot, "source_after": snapshot,
        "result_sha256": result_sha, "launch": {"pid": 1}, "wall_s": 1.0,
    })
    return {
        "execution": {"path": str(execution_path), "sha256": _sha(execution_path)},
        "result": {"path": str(result_path), "sha256": result_sha},
    }


class Clock:
    def __init__(self):
        self.values = iter([
            datetime(2027, 1, 1, 1, 3, tzinfo=timezone.utc),  # noqa: UP017
            datetime(2027, 1, 1, 1, 4, tzinfo=timezone.utc),  # noqa: UP017
            datetime(2027, 1, 1, 1, 5, tzinfo=timezone.utc),  # noqa: UP017
        ])

    def __call__(self) -> datetime:
        return next(self.values)


def _capture(tmp_path: Path, *, snapshot_after: dict | None = None, exit_code: int = 0):
    job_path, snapshot, python, job = _case(tmp_path)
    events: list[str] = []
    snapshots = iter([snapshot, snapshot if snapshot_after is None else snapshot_after])

    def source_provider():
        events.append("source")
        return next(snapshots)

    output = (tmp_path / "attestation").resolve()
    def execute():
        events.append("execute")
        return _outputs(job, _sha(job_path), snapshot, exit_code=exit_code)

    def context_capture(job_arg, snapshot_arg, python_arg, output_arg):
        events.append("context")
        return C.capture_context(
            job_arg, snapshot_arg, python_arg, output_arg, runner=CF.Runner(),
        )

    attestation = A.attest_capture(
        job_path, source_provider, python, output, execute,
        clock=Clock(), context_capture=context_capture,
    )
    refs = {
        "execution": {"path": str(Path(job["output_directory"]).parent / "execution.json")},
        "result": {"path": str(Path(job["output_directory"]) / "result.json")},
    }
    return attestation, refs, job_path, snapshot, python, job, events


def test_attest_capture_hard_codes_source_context_execute_source_seal_order(tmp_path: Path) -> None:
    attestation, refs, job_path, snapshot, _, job, events = _capture(tmp_path)
    assert events == ["source", "context", "execute", "source"]
    body = A.verify_attestation(
        attestation, job, _sha(job_path), snapshot,
        attestation.parent / "context/context.receipt.json",
        Path(refs["execution"]["path"]), Path(refs["result"]["path"]),
        protocol_path=Path(job["protocol"]["path"]),
    )
    assert body["phase_order"] == A.PHASE_ORDER and body["completed"] is True
    context_body = json.loads((attestation.parent / "context/context.receipt.json").read_text())
    assert C._utc_stamp(context_body["finished_at_utc"], "context") <= C._utc_stamp(
        body["execute_started_at_utc"], "execute",
    )


def test_execute_failure_is_called_once_and_same_output_cannot_retry(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _case(tmp_path)
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        raise RuntimeError("failure")

    output = (tmp_path / "attestation").resolve()
    capture = lambda *args: C.capture_context(*args, runner=CF.Runner())
    with pytest.raises(RuntimeError, match="failure"):
        A.attest_capture(job_path, lambda: snapshot, python, output, execute,
                         clock=Clock(), context_capture=capture)
    failure = json.loads((output / A.FAILURE_FILE).read_text())
    assert calls == 1 and failure["stage"] == "execute" and failure["job_sha256"] == _sha(job_path)
    with pytest.raises(FileExistsError):
        A.attest_capture(job_path, lambda: snapshot, python, output, execute,
                         clock=Clock(), context_capture=capture)
    assert calls == 1


def test_source_drift_after_execute_fails_without_attestation(tmp_path: Path) -> None:
    _, snapshot, _, _ = _case(tmp_path)
    changed = copy.deepcopy(snapshot)
    changed["lab_head"] = "9" * 40
    with pytest.raises(ValueError, match="전후 source"):
        _capture(tmp_path / "run", snapshot_after=changed)
    output = tmp_path / "run/attestation"
    assert (output / A.FAILURE_FILE).is_file() and not (output / A.ATTESTATION_FILE).exists()


def test_context_timestamp_after_execute_start_is_rejected_before_callback(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _case(tmp_path)
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        return {}

    def late_context(*args):
        path = C.capture_context(*args, runner=CF.Runner())
        body = json.loads(path.read_text())
        body["finished_at_utc"] = "2028-01-01T00:00:00+00:00"
        path.unlink()
        _write(path, body)
        return path

    with pytest.raises(ValueError, match="먼저 시작"):
        A.attest_capture(
            job_path, lambda: snapshot, python, (tmp_path / "attestation").resolve(), execute,
            clock=Clock(), context_capture=late_context,
        )
    assert calls == 0


def test_verify_rejects_execution_and_result_byte_tamper(tmp_path: Path) -> None:
    attestation, refs, job_path, snapshot, _, job, _ = _capture(tmp_path)
    execution = Path(refs["execution"]["path"])
    execution.write_bytes(execution.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA"):
        A.verify_attestation(
            attestation, job, _sha(job_path), snapshot,
            attestation.parent / "context/context.receipt.json", execution,
            Path(refs["result"]["path"]),
            protocol_path=Path(job["protocol"]["path"]),
        )


def test_callback_path_symlink_escape_is_rejected_when_supported(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _case(tmp_path)
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(tmp_path / "absent")
        probe.unlink()
    except OSError:
        pytest.skip("이 환경은 symlink 생성을 허용하지 않는다")

    def execute():
        refs = _outputs(job, _sha(job_path), snapshot)
        fixed_result = Path(refs["result"]["path"])
        external = tmp_path / "external-result.json"
        fixed_result.replace(external)
        fixed_result.symlink_to(external)
        return refs

    capture = lambda *args: C.capture_context(*args, runner=CF.Runner())
    with pytest.raises(ValueError, match="symlink"):
        A.attest_capture(
            job_path, lambda: snapshot, python, (tmp_path / "attestation").resolve(),
            execute, clock=Clock(), context_capture=capture,
        )


def test_cached_record_targets_are_rejected_before_callback(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _case(tmp_path)
    _outputs(job, _sha(job_path), snapshot)
    calls = 0

    def execute():
        nonlocal calls
        calls += 1
        return {}

    capture = lambda *args: C.capture_context(*args, runner=CF.Runner())
    with pytest.raises(FileExistsError, match="execute 전에 이미 존재"):
        A.attest_capture(
            job_path, lambda: snapshot, python, (tmp_path / "attestation").resolve(), execute,
            clock=Clock(), context_capture=capture,
        )
    assert calls == 0


def test_verify_treats_archived_remote_reference_paths_as_opaque(tmp_path: Path) -> None:
    attestation, refs, job_path, snapshot, _, job, _ = _capture(tmp_path)
    local_protocol = Path(job["protocol"]["path"])
    archived_job = copy.deepcopy(job)
    archived_job["protocol"]["path"] = "/remote/task/record-protocol.json"
    archived_job["output_directory"] = "/remote/archive/output"
    body = json.loads(attestation.read_text())
    body["execution"]["path"] = "/remote/archive/execution.json"
    body["result"]["path"] = "/remote/archive/output/result.json"
    attestation.unlink()
    _write(attestation, body)
    verified = A.verify_attestation(
        attestation, archived_job, _sha(job_path), snapshot,
        attestation.parent / "context/context.receipt.json",
        Path(refs["execution"]["path"]), Path(refs["result"]["path"]),
        protocol_path=local_protocol,
    )
    assert verified["execution"]["path"].startswith("/remote/")


def test_failed_execution_exit_code_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="완료"):
        _capture(tmp_path, exit_code=1)
    assert (tmp_path / "attestation/failure.json").is_file()


def test_wrapper_source_is_required_in_job_snapshot_and_record_protocol(tmp_path: Path) -> None:
    job_path, snapshot, python, job = _case(tmp_path)
    job["source"]["files_lf"].pop(A.WRAPPER_RELATIVE)
    _write(job_path, job)
    capture = lambda *args: C.capture_context(*args, runner=CF.Runner())
    with pytest.raises(ValueError, match="wrapper"):
        A.attest_capture(
            job_path, lambda: snapshot, python, (tmp_path / "attestation").resolve(),
            dict, clock=Clock(), context_capture=capture,
        )


def test_attestation_output_root_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    job_path, snapshot, python, _ = _case(tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("이 환경은 symlink 생성을 허용하지 않는다")
    with pytest.raises(ValueError, match="symlink|정규화"):
        A.attest_capture(
            job_path, lambda: snapshot, python, (linked / "attestation").absolute(),
            dict, clock=Clock(), context_capture=lambda *args: Path(),
        )
