from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import run_vision_semantic_diagnostic_v1 as W
import vision_semantic_diagnostic_v1 as D
import vision_semantic_sdk_v1 as SDK


def _validated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text('{"fixture":true}\n', encoding="utf-8")
    value = {
        "job_path": job,
        "job_sha256": W._sha(job),
        "output_directory": tmp_path / "output",
        "job": {
            "mode": W.JOB.MODE,
            "python": sys.executable,
            "isaaclab": str(tmp_path),
        },
        "resource": {"resource_id": "owned-vm", "gpu_uuid": "GPU-owned"},
    }
    monkeypatch.setattr(W.JOB, "validate_job", lambda *_args, **_kwargs: value)
    return value


def _envelope() -> dict[str, Any]:
    diagnostic = D.collect_diagnostic(
        D.DiagnosticObserver(),
        run_event=lambda _index: None,
        physics_snapshot=lambda: {"step": 0},
        event_count=2,
    )
    window = {
        "starts": "before_writer_attach",
        "ends": "after_writer_detach_return_before_render_product_cleanup",
        "detach_callbacks_included": True,
        "complete_drain_proven": False,
    }
    diagnostic["collection_scope"] = {
        "starts": window["starts"],
        "ends": window["ends"],
        "callbacks_before_detach_preserved": True,
        "callbacks_during_detach_included": True,
        "complete_drain_proven": False,
        "all_future_callbacks_collected": False,
    }
    setup = {
        "phase_order": [
            "extension_enable",
            "extension_app_update",
            "replicator_startup",
            "annotator_registrations",
            "marker_create",
            "capture_readiness",
            "marker_bind",
            "writer_attach",
            "event_0",
            "event_1",
            "writer_detach",
        ],
        "setup_physics_before_after": {
            "before": {"step": 0},
            "after": {"step": 0},
            "equal": True,
        },
        "extension": {
            "name": "isaacsim.core.nodes",
            "enable_result": True,
            "app_update_completed": True,
        },
        "startup": {"before": {}, "after": {}},
        "registrations": dict.fromkeys(
            ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes"), True
        ),
        "callback_count_before_detach": 0,
        "callback_count_after_detach": 0,
        "observation_window": window,
    }
    return {
        "schema": SDK.SDK_REPORT_SCHEMA,
        "diagnostic": diagnostic,
        "lifecycle": {**dict.fromkeys(W.LIFECYCLE_FLAGS, True), "errors": []},
        "renderer": {"fixture": True},
        "no_capture_approved": True,
        "setup_evidence": setup,
    }


def _run_sdk(_validated: Any, _app: Any, *, seal_report: Any) -> dict[str, Any]:
    final = _envelope()
    sealed = copy.deepcopy(final)
    for key in ("report_sealed", "render_product_destroyed", "environment_closed"):
        sealed["lifecycle"][key] = False
    seal_report(sealed)
    return final


class Settings:
    value = False

    def get_as_bool(self, _key: str) -> bool:
        return self.value

    def set_bool(self, _key: str, value: bool) -> None:
        self.value = value


class App:
    def __init__(
        self, validated: Any, *, close_error: BaseException | None = None
    ) -> None:
        self.validated, self.close_error, self.closed = validated, close_error, False

    def close(self, *, wait_for_replicator: bool) -> None:
        assert wait_for_replicator is False
        output = self.validated["output_directory"]
        present = [
            name
            for name in ("result.json", "failure.json")
            if (output / name).is_file()
        ]
        assert len(present) == 1, "terminal evidence가 close 전에 있어야 한다"
        assert (self.validated["job_path"].parent / W.SHUTDOWN.SIDECAR).is_file()
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _app_factory(validated: Any, *, close_error: BaseException | None = None) -> Any:
    return W.SHUTDOWN.AppProxy(
        App(validated, close_error=close_error), validated, Settings()
    )


class Reader:
    """실제 nvidia-smi 대신 원문 receipt verifier를 통과하는 CPU fixture."""

    def __init__(self, tmp_path: Path, *, pids: list[int] | None = None) -> None:
        self.tmp_path, self.pids, self.count = tmp_path, pids or [], 0
        self.last_receipt: dict[str, str] | None = None

    def __call__(self) -> list[int]:
        self.count += 1
        prefix = f"{self.count:06d}"
        gpu = self.tmp_path / f"{prefix}.gpu.csv"
        process = self.tmp_path / f"{prefix}.process.csv"
        gpu.write_text("GPU-owned\n", encoding="utf-8")
        process.write_text(
            "".join(f"GPU-owned, {pid}\n" for pid in self.pids), encoding="utf-8"
        )
        receipt = self.tmp_path / f"{prefix}.receipt.json"
        W.JOB.write_new(
            receipt,
            {
                "schema": "vision_remote_gpu_observation_v1",
                "ordinal": self.count,
                "resource_id": "owned-vm",
                "gpu_uuid": "GPU-owned",
                "gpu_command": W.REMOTE.GPU_COMMAND,
                "gpu_exit_code": 0,
                "gpu_stdout_file": gpu.name,
                "gpu_stdout_sha256": W._sha(gpu),
                "gpu_stderr": "",
                "process_command": W.REMOTE.PROCESS_COMMAND,
                "process_exit_code": 0,
                "process_stdout_file": process.name,
                "process_stdout_sha256": W._sha(process),
                "process_stderr": "",
            },
        )
        self.last_receipt = {"path": str(receipt), "sha256": W._sha(receipt)}
        return self.pids


class GateInput(io.BytesIO):
    def close(self) -> None:
        self.accepted = self.getvalue()
        super().close()


class Process:
    pid = 12345
    returncode: int | None = None

    def __init__(self, action: Any, *, timeout: bool = False) -> None:
        self.stdin, self.action, self.timeout = GateInput(), action, timeout

    def wait(self, *, timeout: float) -> int:
        assert 0 < timeout <= 600
        self.wait_timeout = timeout
        assert self.stdin.accepted == b"GO\n"
        if self.timeout:
            raise subprocess.TimeoutExpired("fixture", timeout)
        self.returncode = self.action()
        return self.returncode


def _run_parent(
    value: Any,
    *,
    action: Any,
    reader: Reader,
    timeout: bool = False,
    identify: Any = None,
    cleanup: Any = None,
) -> tuple[dict[str, Any], Process]:
    process = Process(action, timeout=timeout)

    def popen(command: list[str], **kwargs: Any) -> Process:
        assert "--sdk-child" in command and command[-2] == "--job"
        assert kwargs["start_new_session"] is True and kwargs["shell"] is False
        assert kwargs["cwd"] == value["job"]["isaaclab"]
        return process

    receipt = W.run_job(
        value["job_path"],
        popen=popen,
        gpu_reader=reader,
        identify=identify or (lambda pid: {"pid": pid, "start_ticks": 1}),
        cleanup=cleanup or (lambda _identity, _process: True),
    )
    return receipt, process


def test_child_and_parent_accept_only_bound_diagnostic_result(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    receipt, process = _run_parent(
        value,
        reader=Reader(tmp_path),
        action=lambda: W.sdk_child(
            value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk
        ),
    )
    assert receipt["completed"] is True and process.returncode == 0
    assert receipt["cleanup_verified"] is True and receipt["gpu_after"] == []
    assert (
        receipt["training_actions"]
        == receipt["model_api_requests"]
        == receipt["full_recordings"]
        == 0
    )
    assert receipt["no_capture_approved"] is True
    result = W.verify_result(value)
    assert result["diagnostic_collection_completed"] is True
    assert result["sdk"]["diagnostic"]["callbacks_observed"] == 0


@pytest.mark.parametrize(
    "corruption",
    ["report", "job_sha", "capture", "cleanup", "failure", "policy", "drain"],
)
def test_parent_rejects_tampered_terminal_or_artifact(
    tmp_path: Path, monkeypatch: Any, corruption: str
) -> None:
    value = _validated(tmp_path, monkeypatch)
    assert (
        W.sdk_child(value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk) == 0
    )
    output = value["output_directory"]
    result_path = output / "result.json"
    result = W._read(result_path)
    if corruption == "report":
        (output / W.REPORT_NAME).write_text("{}", encoding="utf-8")
    elif corruption == "job_sha":
        result["job_sha256"] = "f" * 64
    elif corruption == "capture":
        result["sdk"]["no_capture_approved"] = False
    elif corruption == "cleanup":
        result["sdk"]["lifecycle"]["writer_detached"] = False
    elif corruption == "failure":
        W.JOB.write_new(output / "failure.json", {})
    elif corruption == "policy":
        (tmp_path / W.SHUTDOWN.SIDECAR).unlink()
    elif corruption == "drain":
        result["sdk"]["diagnostic"]["step_return_is_complete_drain"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        W.verify_result(value)


def test_exit_zero_without_terminal_is_not_success(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    receipt, _process = _run_parent(value, reader=Reader(tmp_path), action=lambda: 0)
    assert receipt["exit_code"] == 0 and receipt["completed"] is False
    assert receipt["cleanup_verified"] is True and receipt["gpu_after"] == []
    assert (tmp_path / "execution.json").is_file()


def test_foreign_gpu_blocks_before_process_launch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    calls: list[str] = []
    receipt = W.run_job(
        value["job_path"],
        gpu_reader=Reader(tmp_path, pids=[999]),
        popen=lambda *_args, **_kwargs: calls.append("launch"),
    )
    assert calls == [] and receipt["completed"] is False
    assert receipt["gpu_before"] == [999]


def test_timeout_cleans_only_attested_group_and_preserves_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    cleaned: list[Any] = []
    receipt, _process = _run_parent(
        value,
        action=lambda: 0,
        reader=Reader(tmp_path),
        timeout=True,
        cleanup=lambda identity, process: cleaned.append(identity) or True,
    )
    assert cleaned == [{"pid": 12345, "start_ticks": 1}]
    assert receipt["timed_out"] is True and receipt["completed"] is False
    assert receipt["cleanup_verified"] is True


def test_identity_failure_never_sends_go_and_reaps_waiting_gate(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    cleaned: list[Any] = []
    monkeypatch.setattr(
        W.GATE, "_cleanup_waiting_gate", lambda process: cleaned.append(process) or True
    )

    def identify(_pid: int) -> Any:
        raise ValueError("not owned")

    receipt, process = _run_parent(
        value, reader=Reader(tmp_path), action=lambda: 0, identify=identify
    )
    assert cleaned == [process] and process.stdin.getvalue() == b""
    assert receipt["completed"] is False and receipt["cleanup_verified"] is True


def test_job_mutation_after_child_is_failure(tmp_path: Path, monkeypatch: Any) -> None:
    value = _validated(tmp_path, monkeypatch)

    def action() -> int:
        value["job_path"].write_text('{"changed":true}', encoding="utf-8")
        return 0

    receipt, _process = _run_parent(value, action=action, reader=Reader(tmp_path))
    assert receipt["completed"] is False and receipt["error_type"] == "ValueError"


def test_missing_gpu_raw_receipt_is_failure_before_launch(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    calls: list[str] = []
    receipt = W.run_job(
        value["job_path"],
        gpu_reader=list,
        popen=lambda *_args, **_kwargs: calls.append("launch"),
    )
    assert receipt["completed"] is False and calls == []


def test_sdk_error_and_close_error_keep_source_failure_before_shutdown(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)

    def fail(_value: Any, _app: Any, *, seal_report: Any) -> Any:
        envelope = _run_sdk(_value, _app, seal_report=seal_report)
        raise SDK.SDKDiagnosticFailure(envelope, RuntimeError("source error"))

    result = W.sdk_child(
        value["job_path"],
        run_sdk=fail,
        app_factory=lambda v: _app_factory(v, close_error=ValueError("close error")),
    )
    assert result == 1
    failure = W._read(value["output_directory"] / "failure.json")
    shutdown = W._read(value["output_directory"] / "shutdown_failure.json")
    assert failure["error_type"] == "SDKDiagnosticFailure"
    assert failure["sdk"]["schema"] == SDK.SDK_REPORT_SCHEMA
    assert shutdown["error_type"] == "ValueError"
    assert not (value["output_directory"] / "result.json").exists()


def test_keyboard_interrupt_is_not_diagnostic_completion(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)

    def cancel(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        W.sdk_child(value["job_path"], app_factory=_app_factory, run_sdk=cancel)
    failure = W._read(value["output_directory"] / "failure.json")
    assert (
        failure["completed"] is False and failure["error_type"] == "KeyboardInterrupt"
    )
    assert not (value["output_directory"] / "result.json").exists()


def test_child_failure_descriptor_error_does_not_skip_terminal_or_close(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)

    def fail(_value: Any, _app: Any, *, seal_report: Any) -> Any:
        _run_sdk(_value, _app, seal_report=seal_report)
        raise RuntimeError("first failure")

    monkeypatch.setattr(
        W, "_descriptor", lambda _path: (_ for _ in ()).throw(ValueError("descriptor"))
    )
    assert W.sdk_child(value["job_path"], app_factory=_app_factory, run_sdk=fail) == 1
    failure = W._read(value["output_directory"] / "failure.json")
    assert (
        failure["error_type"] == "RuntimeError"
        and failure["error_message"] == "first failure"
    )
    assert failure["partial_report"] is None


def test_no_rerun_over_existing_execution(tmp_path: Path, monkeypatch: Any) -> None:
    value = _validated(tmp_path, monkeypatch)
    W.JOB.write_new(tmp_path / "execution.json", {})
    with pytest.raises(FileExistsError):
        W.run_job(value["job_path"], gpu_reader=list, popen=lambda: None)


def test_duplicate_result_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"completed":false,"completed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        W._read(path)


def test_result_verification_time_is_included_in_600_second_bound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(W.time, "monotonic", lambda: clock[0])
    verify = W.verify_result

    def delayed_verify(request: Any) -> Any:
        result = verify(request)
        clock[0] = 600.1
        return result

    monkeypatch.setattr(W, "verify_result", delayed_verify)
    receipt, _process = _run_parent(
        value,
        reader=Reader(tmp_path),
        action=lambda: W.sdk_child(
            value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk
        ),
    )
    assert receipt["completed"] is False and receipt["error_type"] == "TimeoutError"
    assert receipt["wall_s"] == 600.1


@pytest.mark.parametrize(
    "corruption",
    ["registrations", "sequence", "physics", "scope", "summary", "callbacks"],
)
def test_diagnostic_summary_and_setup_must_match_observed_records(
    corruption: str,
) -> None:
    value = _envelope()
    if corruption == "registrations":
        value["setup_evidence"]["registrations"]["IsaacReadTimes"] = False
    elif corruption == "sequence":
        value["setup_evidence"]["phase_order"].reverse()
    elif corruption == "physics":
        value["diagnostic"]["events"][0]["physics_after"] = {"step": 1}
    elif corruption == "scope":
        value["diagnostic"]["collection_scope"]["complete_drain_proven"] = True
    elif corruption == "summary":
        value["diagnostic"]["observations"]["valid_observed"] = True
    elif corruption == "callbacks":
        value["diagnostic"]["callbacks"] = [{"ordinal": 0}]
        value["diagnostic"]["callbacks_observed"] = value["diagnostic"][
            "callbacks_preserved"
        ] = 1
    assert W._successful_sdk_envelope(value) is False


@pytest.mark.skipif(os.name != "posix", reason="실제 Linux 소유 process group 회귀")
def test_actual_owned_gate_reaps_child_that_rejects_job_before_sdk(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    receipt = W.run_job(value["job_path"], gpu_reader=Reader(tmp_path))
    assert receipt["completed"] is False
    assert receipt["exit_code"] != 0 and receipt["cleanup_verified"] is True
    assert receipt["launch"]["pid"] > 0
    assert receipt["gpu_before"] == receipt["gpu_after"] == []
    assert not (value["output_directory"] / "result.json").exists()
    assert "job 필드가 계약과 다르다" in (tmp_path / "stderr.log").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("child_s", [399.0, 400.0])
def test_child_leaves_full_postflight_reserve(
    tmp_path: Path, monkeypatch: Any, child_s: float
) -> None:
    value = _validated(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(W.time, "monotonic", lambda: clock[0])

    class TimedReader(Reader):
        def __call__(self) -> list[int]:
            if self.count:
                clock[0] += 60.0
            return super().__call__()

    original_verify = W.verify_result

    def verify(request: Any) -> Any:
        clock[0] += 99.0
        return original_verify(request)

    monkeypatch.setattr(W, "verify_result", verify)

    def action() -> int:
        clock[0] += child_s
        return W.sdk_child(
            value["job_path"], app_factory=_app_factory, run_sdk=_run_sdk
        )

    def cleanup(_identity: Any, _process: Any) -> bool:
        clock[0] += 40.0
        return True

    receipt, process = _run_parent(
        value, reader=TimedReader(tmp_path), action=action, cleanup=cleanup
    )
    assert receipt["completed"] is True and process.wait_timeout == 400.0
    assert receipt["wall_s"] == child_s + 199.0
    assert receipt["timing"]["cleanup_wall_s"] == 40.0
    assert receipt["timing"]["postflight_wall_s"] == 199.0
    assert receipt["timing"]["deadline_overrun_s"] == 0.0


def test_no_go_if_admission_consumes_postflight_reserve(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(W.time, "monotonic", lambda: clock[0])

    class SlowReader(Reader):
        def __call__(self) -> list[int]:
            clock[0] = 401.0
            return super().__call__()

    calls: list[str] = []
    receipt = W.run_job(
        value["job_path"],
        gpu_reader=SlowReader(tmp_path),
        popen=lambda *_args, **_kwargs: calls.append("launch"),
    )
    assert receipt["completed"] is False and receipt["error_type"] == "TimeoutError"
    assert calls == [] and not (tmp_path / "launch.json").exists()


def test_gpu_query_uses_remaining_deadline_and_rejects_overrun(
    monkeypatch: Any,
) -> None:
    clock = [590.0]
    monkeypatch.setattr(W.time, "monotonic", lambda: clock[0])
    seen: list[float] = []

    def run(*_args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["timeout"])
        return "fixture"

    query = W._gpu_query_run(600.0, run)
    assert query(["unused"], timeout=30) == "fixture"
    assert seen == [10.0]
    clock[0] = 600.0
    with pytest.raises(TimeoutError):
        query(["unused"], timeout=30)
    assert seen == [10.0]


def test_cleanup_overrun_is_reported_without_retrying_or_skipping_safety(
    tmp_path: Path, monkeypatch: Any
) -> None:
    value = _validated(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(W.time, "monotonic", lambda: clock[0])
    calls: list[str] = []

    def cleanup(_identity: Any, _process: Any) -> bool:
        calls.append("cleanup")
        clock[0] += 601.0
        return True

    receipt, _process = _run_parent(
        value, reader=Reader(tmp_path), action=lambda: 0, cleanup=cleanup
    )
    assert calls == ["cleanup"] and receipt["cleanup_verified"] is True
    assert (
        receipt["completed"] is False and receipt["timing"]["deadline_overrun_s"] == 1.0
    )
