from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
V1 = importlib.import_module("vision_play_worker_v1")
V2 = importlib.import_module("vision_play_worker_v2")
SOURCES = importlib.import_module("vision_selection_sources_v1")


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: list[str], *, mode: str = "record",
) -> tuple[Path, dict[str, Any]]:
    output = tmp_path / "out"
    validated = {
        "output_directory": output,
        "job_sha256": "a" * 64,
        "job": {"mode": mode},
    }
    monkeypatch.setattr(V1.R, "validate_job", lambda path: validated)
    monkeypatch.setattr(
        V1.R,
        "result_body",
        lambda validated, artifacts, primary: {
            "schema": "fixture_result",
            "completed": True,
            "artifacts": sorted(artifacts),
            "primary": primary.name,
        },
    )
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    validated["job_path"] = job
    return job, validated


class App:
    def __init__(self, output: Path, events: list[str]) -> None:
        self.output = output
        self.events = events

    def update(self) -> None:
        self.events.append("update")

    def close(self) -> None:
        self.events.append("close")
        assert (self.output / V2.RECEIPT_FILE).is_file()
        assert (self.output / "result.json").is_file() or (self.output / "failure.json").is_file()


def test_enable_update_registration_capture_and_close_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events, mode=V1.R.RENDER_SYNC_SMOKE_MODE)

    def app_factory() -> App:
        events.append("app")
        return App(validated["output_directory"], events)

    def enable(name: str) -> bool:
        events.append(f"enable:{name}")
        return True

    def probe() -> dict[str, bool]:
        events.append("probe")
        return {name: True for name in V2.ANNOTATORS}

    def startup() -> dict[str, dict[str, bool]]:
        events.append("startup")
        assert not validated["output_directory"].exists()
        return {
            "before": {"orchestrator_enabled": False, "capture_on_play": True},
            "after": {"orchestrator_enabled": True, "capture_on_play": False},
        }

    def capture(current: dict[str, Any], app: App) -> tuple[dict[str, Path], Path]:
        del app
        events.append("capture")
        pre_capture = json.loads(job.with_suffix(".renderer_registration.json").read_text(encoding="utf-8"))
        assert pre_capture["status"] == "ready"
        assert pre_capture["replicator_startup"]["after"]["orchestrator_enabled"] is True
        current["output_directory"].mkdir()
        artifacts: dict[str, Path] = {}
        for name in SOURCES._SMOKE_ARTIFACTS:
            artifact = current["output_directory"] / f"{name}.fixture"
            artifact.write_text("{}\n", encoding="utf-8")
            artifacts[name] = artifact
        return artifacts, artifacts["render_sync_smoke"]

    assert V2.execute(
        job, app_factory=app_factory, runner=capture,
        enable_extension=enable, registration_probe=probe, startup_configure=startup,
    ) == 0
    assert events == [
        "app", f"enable:{V2.EXTENSION}", "update", "startup", "probe", "capture", "close",
    ]
    evidence = json.loads((validated["output_directory"] / V2.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert evidence["status"] == "ready" and all(evidence["annotators"].values())
    assert evidence["replicator_startup"]["after"]["orchestrator_enabled"] is True
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["replicator_startup"] == evidence["replicator_startup"]
    result = json.loads((validated["output_directory"] / "result.json").read_text(encoding="utf-8"))
    assert set(result["artifacts"]) == SOURCES._SMOKE_ARTIFACTS
    assert "renderer_registration" not in result["artifacts"]


@pytest.mark.parametrize("enable_behavior", ["false", "raise"])
def test_enable_failure_skips_update_probe_and_capture_but_closes_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enable_behavior: str,
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events)

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError((args, kwargs))

    def enable(name: str) -> bool:
        if enable_behavior == "raise":
            raise OSError(f"extension unavailable: {name}")
        return False

    assert V2.execute(
        job, app_factory=lambda: App(validated["output_directory"], events), runner=forbidden,
        enable_extension=enable, registration_probe=forbidden, startup_configure=forbidden,
    ) == 1
    assert events == ["close"]
    failure = json.loads((validated["output_directory"] / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "record_play"
    assert failure["error_type"] == ("OSError" if enable_behavior == "raise" else "RuntimeError")
    evidence = json.loads((validated["output_directory"] / V2.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert evidence["status"] == "failed" and evidence["enable_result"] is False


def test_missing_annotator_does_not_enter_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events)

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError((args, kwargs))

    registered = {name: True for name in V2.ANNOTATORS}
    registered["ReferenceTime"] = False
    assert V2.execute(
        job, app_factory=lambda: App(validated["output_directory"], events), runner=forbidden,
        enable_extension=lambda name: True, registration_probe=lambda: registered,
        startup_configure=lambda: {"before": {}, "after": {}},
    ) == 1
    evidence = json.loads((validated["output_directory"] / V2.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert evidence["app_update_completed"] is True
    assert evidence["annotators"]["ReferenceTime"] is False
    assert events == ["update", "close"]


def test_capture_failure_keeps_registration_and_v1_failure_before_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events)

    def capture(current: dict[str, Any], app: App) -> tuple[dict[str, Path], Path]:
        del current, app
        events.append("capture")
        raise RuntimeError("existing capture failure")

    assert V2.execute(
        job, app_factory=lambda: App(validated["output_directory"], events), runner=capture,
        enable_extension=lambda name: True,
        registration_probe=lambda: {name: True for name in V2.ANNOTATORS},
        startup_configure=lambda: {"before": {}, "after": {}},
    ) == 1
    assert events == ["update", "capture", "close"]
    evidence = json.loads((validated["output_directory"] / V2.RECEIPT_FILE).read_text(encoding="utf-8"))
    failure = json.loads((validated["output_directory"] / "failure.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "ready"
    assert failure["stage"] == "record_play" and failure["error_message"] == "existing capture failure"


@pytest.mark.parametrize(
    ("mode", "selected"),
    [("record", "real"), (V1.R.RENDER_SYNC_SMOKE_MODE, "smoke")],
)
def test_default_runner_selects_original_v1_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, selected: str,
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events, mode=mode)

    def capture(current: dict[str, Any], app: App) -> tuple[dict[str, Path], Path]:
        del app
        events.append(selected)
        current["output_directory"].mkdir()
        primary = current["output_directory"] / "primary.json"
        primary.write_text("{}\n", encoding="utf-8")
        return {"primary": primary}, primary

    monkeypatch.setattr(V1, "run_real" if selected == "real" else "run_render_sync_smoke", capture)
    assert V2.execute(
        job,
        app_factory=lambda: App(validated["output_directory"], events),
        enable_extension=lambda name: True,
        registration_probe=lambda: {name: True for name in V2.ANNOTATORS},
        startup_configure=lambda: {"before": {}, "after": {}},
    ) == 0
    assert events == ["update", selected, "close"]


def test_startup_readback_failure_skips_capture_and_v1_closes_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    job, validated = _fixture(tmp_path, monkeypatch, events)

    def startup() -> dict[str, dict[str, bool]]:
        events.append("startup")
        assert not validated["output_directory"].exists()
        raise RuntimeError("startup readback failed")

    def forbidden(*args: object, **kwargs: object) -> Any:
        raise AssertionError((args, kwargs))

    assert V2.execute(
        job,
        app_factory=lambda: App(validated["output_directory"], events),
        runner=forbidden,
        enable_extension=lambda name: True,
        registration_probe=forbidden,
        startup_configure=startup,
    ) == 1
    assert events == ["update", "startup", "close"]
    failure = json.loads((validated["output_directory"] / "failure.json").read_text(encoding="utf-8"))
    evidence = json.loads((validated["output_directory"] / V2.RECEIPT_FILE).read_text(encoding="utf-8"))
    assert failure["stage"] == "record_play" and failure["error_message"] == "startup readback failed"
    assert evidence["replicator_startup"] is None and evidence["status"] == "failed"
