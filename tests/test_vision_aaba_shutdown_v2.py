from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", Path(__file__).parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import test_vision_annotator_pull_checkpoint_v2 as FIXTURE  # pyright: ignore[reportMissingImports]
import vision_aaba_shutdown_v2 as S  # pyright: ignore[reportMissingImports]
import vision_pull_record_checkpoint_v3 as CHECKPOINT  # pyright: ignore[reportMissingImports]
import vision_shutdown_policy_v1 as OLD  # pyright: ignore[reportMissingImports]


class Settings:
    def __init__(self) -> None:
        self.value = False
        self.events: list[str] = []

    def get_as_bool(self, _name: str) -> bool:
        self.events.append(f"read:{self.value}")
        return self.value

    def set_bool(self, _name: str, value: bool) -> None:
        self.events.append(f"set:{value}")
        self.value = value


class RawApp:
    def __init__(self, events: list[str], close_error: BaseException | None = None) -> None:
        self.events = events
        self.close_error = close_error

    def update(self) -> None:
        self.events.append("update")

    def close(self, **kwargs: Any) -> str:
        assert kwargs == {"wait_for_replicator": False}
        self.events.append("raw_close")
        if self.close_error is not None:
            raise self.close_error
        return "closed"


def valid_terminal(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "aaba-output"
    validated = {
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": output,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "job": {
            "mode": CHECKPOINT.MODE,
            "callback_variant": CHECKPOINT.VARIANT,
            "profile": FIXTURE.V1_FIXTURE.RECORD.fixed_profile(),
        },
    }
    worker = FIXTURE._worker()
    worker["policy_probe"]["checkpoint_sha256"] = validated["checkpoint_sha256"]
    worker["lifecycle"].update(
        {"render_product_destroyed": True, "environment_closed": True}
    )
    report_path = output / S.REPORT_NAME
    report = CHECKPOINT.run_annotator_pull_diagnostic(
        validated,
        object(),
        run_worker=lambda *_args, **_kwargs: worker,
        seal_report=lambda value: report_path.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        ),
    )
    return validated, report


def test_old_proxy_report_only_path_cannot_apply_fast_shutdown(tmp_path: Path) -> None:
    validated, _ = valid_terminal(tmp_path)
    events: list[str] = []
    settings = Settings()
    proxy = OLD.AppProxy(RawApp(events), validated, settings)

    with pytest.raises(ValueError, match="terminal evidence"):
        proxy.close()

    assert events == ["raw_close"]
    assert settings.value is False
    assert not (tmp_path / OLD.SIDECAR).exists()


def test_new_proxy_binds_report_then_fast_shutdown_then_raw_close(tmp_path: Path) -> None:
    validated, report = valid_terminal(tmp_path)
    events: list[str] = []
    settings = Settings()
    proxy = S.AppProxy(RawApp(events), validated, settings)

    assert proxy._app is not None
    assert proxy.close() == "closed"
    assert events == ["raw_close"]
    assert settings.events == ["read:False", "read:False", "set:True", "read:True"]
    assert settings.value is True
    assert S.verify(validated)["terminal_evidence"]["sha256"] == hashlib.sha256(
        (tmp_path / "aaba-output" / S.REPORT_NAME).read_bytes()
    ).hexdigest()
    assert report["worker"]["lifecycle"]["environment_closed"] is True
    assert not (tmp_path / OLD.SIDECAR).exists()


def test_descriptor_tamper_blocks_policy_but_still_closes_and_records_phases(
    tmp_path: Path,
) -> None:
    validated, report = valid_terminal(tmp_path)
    artifact = report["raw_artifacts"][0]["npz"]["file"]
    (tmp_path / "aaba-output" / artifact).write_bytes(b"tampered")
    events: list[str] = []
    proxy = S.AppProxy(RawApp(events), validated, Settings())

    with pytest.raises(ValueError, match="SHA/bytes"):
        proxy.close()

    assert events == ["raw_close"]
    assert (tmp_path / S.POLICY_FAILURE_NAME).is_file()
    assert (tmp_path / S.CLOSE_ENTERED_NAME).is_file()
    assert (tmp_path / S.CLOSE_RETURNED_NAME).is_file()
    assert not (tmp_path / S.POLICY_NAME).exists()


def test_failure_terminal_is_sealed_before_close_policy(tmp_path: Path) -> None:
    validated, _ = valid_terminal(tmp_path)
    output = tmp_path / "aaba-output"
    partial = S._descriptor(output / S.REPORT_NAME)
    first = output / "native-aaba-preflight.primary-error.json"
    first.write_text(
        json.dumps(
            {
                "schema": "vision_pull_record_primary_error_v2",
                "error_type": "builtins.ValueError",
                "error_message": "first",
                "traceback": "trace",
            }
        ),
        encoding="utf-8",
    )
    descriptor = S._descriptor(first)
    (output / S.FAILURE_NAME).write_text(
        json.dumps(
            {
                "schema": "vision_pull_record_aaba_failure_v2",
                "completed": False,
                "job_sha256": validated["job_sha256"],
                "checkpoint_sha256": validated["checkpoint_sha256"],
                "first_cause": descriptor,
                "partial_report": partial,
            }
        ),
        encoding="utf-8",
    )
    proxy = S.AppProxy(RawApp([]), validated, Settings())

    proxy.close()

    assert S.verify(validated)["terminal_evidence"]["kind"] == "failure"


def test_fast_exit_is_approved_by_parent_process_receipt_without_close_return(
    tmp_path: Path,
) -> None:
    validated, _ = valid_terminal(tmp_path)
    code = r'''
import hashlib, os, pathlib, sys
import record_vision_evidence_v1 as RECORD
import vision_aaba_shutdown_v2 as S

root = pathlib.Path(sys.argv[1])
job = root / "job.json"
checkpoint = root / "model.pt"
validated = {
    "job_path": job,
    "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
    "output_directory": root / "aaba-output",
    "checkpoint_path": checkpoint,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "job": {"mode": "checkpoint_annotator_pull_diagnostic_only", "callback_variant": "annotator_pull", "profile": RECORD.fixed_profile()},
}
class Settings:
    value = False
    def get_as_bool(self, _name): return self.value
    def set_bool(self, _name, value): self.value = value
class FastExitApp:
    def close(self, **kwargs):
        assert kwargs == {"wait_for_replicator": False}
        os._exit(0)
S.AppProxy(FastExitApp(), validated, Settings()).close()
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=False,
        env=env,
        timeout=10,
    )

    assert completed.returncode == 0
    assert (tmp_path / S.POLICY_NAME).is_file()
    assert (tmp_path / S.CLOSE_ENTERED_NAME).is_file()
    assert not (tmp_path / S.CLOSE_RETURNED_NAME).exists()
    assert S.verify(validated)["completed"] is True
    observation = {"path": "gpu-after.json", "sha256": "c" * 64}
    value = S.write_process_exit(
        validated,
        exit_code=completed.returncode,
        cleanup_verified=True,
        gpu_observation=observation,
    )
    assert value["close_returned_observed"] is False
    assert S.verify_process_exit(validated, observation) == value


def test_process_exit_rejects_nonzero_or_missing_preclose_evidence(tmp_path: Path) -> None:
    validated, _ = valid_terminal(tmp_path)
    proxy = S.AppProxy(RawApp([]), validated, Settings())
    proxy.close()
    observation = {"path": "gpu-after.json", "sha256": "c" * 64}

    with pytest.raises(ValueError, match="exit 0"):
        S.process_exit_body(
            validated,
            exit_code=3,
            cleanup_verified=True,
            gpu_observation=observation,
        )
    (tmp_path / S.POLICY_NAME).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        S.process_exit_body(
            validated,
            exit_code=0,
            cleanup_verified=True,
            gpu_observation=observation,
        )


def test_product_source_does_not_force_process_exit() -> None:
    assert S.__file__ is not None
    source = Path(S.__file__).read_text(encoding="utf-8")
    assert "os._exit" not in source


def test_cold_factory_bootstraps_app_before_importing_carb() -> None:
    code = r'''
import pathlib, sys, types

events = []
sys.modules.pop("carb", None)
sys.modules.pop("carb.settings", None)

class Settings:
    def get_as_bool(self, _name): return False
class RawApp:
    def close(self, **_kwargs): events.append("raw_close")
    def update(self): return None
class AppLauncher:
    def __init__(self, **_kwargs):
        events.append("app_launcher")
        settings = types.ModuleType("carb.settings")
        settings.get_settings = lambda: Settings()
        carb = types.ModuleType("carb")
        carb.__path__ = []
        carb.settings = settings
        sys.modules["carb"] = carb
        sys.modules["carb.settings"] = settings
        self.app = RawApp()

isaaclab = types.ModuleType("isaaclab")
isaaclab.__path__ = []
isaaclab_app = types.ModuleType("isaaclab.app")
isaaclab_app.AppLauncher = AppLauncher
isaaclab.app = isaaclab_app
sys.modules["isaaclab"] = isaaclab
sys.modules["isaaclab.app"] = isaaclab_app

import vision_aaba_shutdown_v2 as S
proxy = S.sdk_app_factory({"job_path": pathlib.Path("job.json")})
assert isinstance(proxy, S.AppProxy)
assert events == ["app_launcher"]
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
