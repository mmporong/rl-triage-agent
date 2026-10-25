from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]
import vision_play_worker_v3 as W  # pyright: ignore[reportMissingImports]
import vision_shutdown_policy_v1 as SHUTDOWN  # pyright: ignore[reportMissingImports]


def _validated(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    job = tmp_path / "job.json"; job.write_text('{"job":1}\n', encoding="utf-8")
    output = tmp_path / "output"
    value = {
        "job": {"mode": R.RENDER_SYNC_SMOKE_MODE}, "job_path": job,
        "job_sha256": R.sha256_file(job), "output_directory": output,
        "protocol_sha256": "1" * 64, "checkpoint_sha256": "2" * 64,
        "request_binding": {"request": "bound"}, "resource": {"id": "resource"},
        "resource_parent_sha256": "3" * 64, "live_lease_proof_sha256": "4" * 64,
        "admission_sha256": "5" * 64, "admission_issued_at_utc": "2026-10-25T00:00:00+00:00",
        "admission_expires_at_utc": "2026-10-25T01:00:00+00:00",
    }
    return job, value


def _write_success(validated: dict[str, Any]) -> None:
    output = validated["output_directory"]; output.mkdir()
    artifact = output / "render_sync_smoke.json"; artifact.write_text('{"ok":true}\n', encoding="utf-8")
    R.write_new(output / "result.json", R.result_body(validated, {"render_sync_smoke": artifact}, artifact))
    class Settings:
        value = False
        def get_as_bool(self, _name: str) -> bool: return self.value
        def set_bool(self, _name: str, value: bool) -> None: self.value = value
    SHUTDOWN.apply(validated, Settings(), startup_fast_shutdown=False)


class Process:
    def __init__(self, action: Any, code: int = 0) -> None:
        self.action, self.code, self.waited = action, code, False

    def wait(self) -> int:
        self.waited = True
        self.action()
        return self.code


def test_supervisor_waits_for_child_and_accepts_only_verified_result(tmp_path: Path, monkeypatch) -> None:
    job, validated = _validated(tmp_path)
    monkeypatch.setattr(W.R, "validate_job", lambda _job: validated)
    captured: dict[str, Any] = {}

    def popen(argv: list[str], **kwargs: Any) -> Process:
        captured.update({"argv": argv, "kwargs": kwargs})
        process = Process(lambda: _write_success(validated))
        captured["process"] = process
        return process

    def seal(job_path: Path, result_path: Path, *, sdk_child_exit_code: int) -> Path:
        assert captured["process"].waited is True
        assert job_path == job and result_path == validated["output_directory"] / "result.json"
        assert sdk_child_exit_code == 0
        captured["sealed"] = True
        return job.parent / "job.execution-evidence.json"

    monkeypatch.setattr(W.EVIDENCE, "write_manifest", seal)
    assert W.supervise(job, popen=popen) == 0
    assert captured["sealed"] is True
    assert captured["process"].waited is True
    assert W.__file__ is not None
    assert captured["argv"] == [sys.executable, "-B", str(Path(W.__file__).resolve()),
                                "--sdk-child", "--job", str(job.resolve())]
    assert captured["kwargs"] == {"shell": False, "start_new_session": False}
    assert not (validated["output_directory"] / "supervisor_failure.json").exists()


@pytest.mark.parametrize("case", ["missing", "failure", "tamper", "policy_missing", "aux_missing", "nonzero", "signal"])
def test_supervisor_rejects_child_exit_or_semantic_failure(tmp_path: Path, monkeypatch, case: str) -> None:
    job, validated = _validated(tmp_path)
    monkeypatch.setattr(W.R, "validate_job", lambda _job: validated)

    def action() -> None:
        if case in {"failure", "tamper", "policy_missing", "aux_missing"}:
            _write_success(validated)
        if case == "failure":
            (validated["output_directory"] / "failure.json").write_text("{}", encoding="utf-8")
        if case == "tamper":
            artifact = validated["output_directory"] / "render_sync_smoke.json"
            artifact.write_text("changed", encoding="utf-8")
        if case == "policy_missing":
            (job.parent / SHUTDOWN.SIDECAR).unlink()

    code = 7 if case == "nonzero" else -9 if case == "signal" else 0
    assert W.supervise(job, popen=lambda *_args, **_kwargs: Process(action, code)) == 1
    failure = json.loads((validated["output_directory"] / "supervisor_failure.json").read_text(encoding="utf-8"))
    assert failure["schema"] == W.SUPERVISOR_FAILURE_SCHEMA and failure["child_exit_code"] == code


def test_supervisor_rejects_job_mutation_after_child(tmp_path: Path, monkeypatch) -> None:
    job, validated = _validated(tmp_path)
    monkeypatch.setattr(W.R, "validate_job", lambda _job: validated)
    assert W.supervise(job, popen=lambda *_args, **_kwargs: Process(
        lambda: job.write_text('{"changed":true}\n', encoding="utf-8"))) == 1
    assert json.loads((validated["output_directory"] / "supervisor_failure.json").read_text(
        encoding="utf-8"))["error_type"] == "ValueError"


def test_sdk_child_wires_readiness_runner_into_v2(tmp_path: Path, monkeypatch) -> None:
    import vision_capture_readiness_v1 as capture  # pyright: ignore[reportMissingImports]
    import vision_play_worker_v2 as worker_v2  # pyright: ignore[reportMissingImports]

    job, validated = _validated(tmp_path)
    monkeypatch.setattr(W.R, "validate_job", lambda _path: validated)
    monkeypatch.setattr(worker_v2, "execute", lambda path, **kwargs: 0 if (
        path == job and callable(kwargs["app_factory"])
        and kwargs["runner"] is capture.run_capture) else 9)
    assert W.sdk_child(job) == 0


def test_sdk_app_factory_uses_fast_shutdown_and_close_proxy(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, Any] = {}
    app = object()
    class Launcher:
        def __init__(self, **kwargs: Any) -> None:
            observed.update(kwargs)
            self.app = app
    original_import = builtins.__import__
    def checked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "carb.settings":
            assert observed == R.fixed_profile()["launcher"], "Kit 설정은 AppLauncher 생성 뒤 읽어야 한다"
        return original_import(name, *args, **kwargs)
    package = types.ModuleType("isaaclab")
    module = types.ModuleType("isaaclab.app"); module.AppLauncher = Launcher  # type: ignore[attr-defined]
    class Settings:
        def get_as_bool(self, _name: str) -> bool: return False
    carb = types.ModuleType("carb")
    settings_module = types.ModuleType("carb.settings")
    settings_module.get_settings = lambda: Settings()  # type: ignore[attr-defined]
    carb.settings = settings_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", settings_module)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    _, validated = _validated(tmp_path)
    proxy = W._sdk_app_factory(validated)
    assert observed == R.fixed_profile()["launcher"]
    assert proxy._app is app


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group 계약")
def test_actual_subprocess_inherits_parent_process_group() -> None:
    code = "import os,sys;sys.exit(0 if os.getpgrp()==int(sys.argv[1]) else 9)"
    process = subprocess.Popen([sys.executable, "-c", code, str(os.getpgrp())],  # pyright: ignore[reportAttributeAccessIssue]
                               shell=False, start_new_session=False)
    assert process.wait() == 0
