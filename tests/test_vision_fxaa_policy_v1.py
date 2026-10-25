from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "tests"):
    sys.path.insert(0, str(folder))
F = importlib.import_module("vision_fxaa_policy_v1")
C = importlib.import_module("vision_capture_readiness_v1")
E = importlib.import_module("vision_execution_evidence_v1")
CF = importlib.import_module("test_vision_capture_readiness_v1")
EF = importlib.import_module("test_vision_execution_evidence_v1")


class Settings:
    def __init__(self) -> None:
        self.values = dict(zip(F.SETTING_KEYS, ("RaytracedLighting", 3, True), strict=True))
        self.calls: list[str] = []

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set_render_rtx_realtime(self, *, antialiasing: str) -> None:
        self.calls.append(antialiasing)
        self.values.update(dict(zip(F.SETTING_KEYS, ("RayTracedLighting", 2, False), strict=True)))


def test_official_settings_call_and_readback_only() -> None:
    settings = Settings()
    receipt = F.apply(SimpleNamespace(settings=settings), settings)
    F.verify_receipt(receipt)
    assert settings.calls == ["FXAA"]
    assert receipt["before"]["/rtx/post/aa/op"] == 3
    assert receipt["after"]["/rtx/post/aa/op"] == 2


@pytest.mark.parametrize("key,value", [
    ("/rtx/post/aa/op", 3), ("/rtx/post/aa/op", 2.0),
    ("/rtx/rendermode", "PathTracing"), ("/rtx-transient/post/aa/limitedOps", True),
])
def test_readback_mismatch_rejected(key: str, value: Any) -> None:
    settings = Settings()
    settings.set_render_rtx_realtime(antialiasing="FXAA")
    settings.values[key] = value
    with pytest.raises(ValueError, match="readback"):
        F.verify_readback(F.readback(settings))


def test_registered_capture_applies_once_checks_each_frame_and_preserves_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated, context, base, app, events = CF._install_capture(monkeypatch, tmp_path)
    validated["job"]["source"] = {"files_lf": {F.SOURCE_FILE: "a" * 64}}
    settings = Settings()
    rep = SimpleNamespace(settings=settings)
    monkeypatch.setattr(C, "_carb_settings", lambda: settings)

    def runner(current: Any, selected_app: Any, *, camera_capture: Any) -> tuple[dict[str, Path], Path]:
        del current, selected_app
        for ordinal in range(2):
            camera_capture(base, rep, [], CF.PROFILE, [1.0] * 3, [0.0] * 3, ordinal)
        return {}, tmp_path / "primary"

    monkeypatch.setattr(C.V1, "run_real", runner)
    C.run_capture(validated, app)
    assert settings.calls == ["FXAA"]
    receipt = json.loads((tmp_path / "job.capture-readiness.json").read_text())
    E._verify_readiness(tmp_path / "job.capture-readiness.json", require_fxaa=True)
    assert receipt["physics_before"] == receipt["physics_after"]
    assert events.count("camera") == events.count("reset") == 2
    for ordinal in range(2):
        E._verify_reset(tmp_path / f"job.capture-{ordinal:06d}.renderer-reset.json", ordinal, require_fxaa=True)
    assert context.index == 4


def test_registered_manifest_cannot_omit_policy_or_use_dlss_readback(tmp_path: Path) -> None:
    stage, job, result = EF._stage(tmp_path)
    body = json.loads(job.read_text())
    body["source"] = {"files_lf": {F.SOURCE_FILE: "a" * 64}}
    job.write_text(json.dumps(body))
    readiness = stage / "job.capture-readiness.json"
    with pytest.raises(ValueError, match="FXAA"):
        E._verify_readiness(readiness, require_fxaa=F.required(body))
    value = json.loads(readiness.read_text())
    settings = Settings()
    value["renderer"]["render_policy"] = F.apply(SimpleNamespace(settings=settings), settings)
    value["renderer"]["carb_settings"].update({key: settings.get(key) for key in C.CARB_SETTING_KEYS})
    readiness.write_text(json.dumps(value))
    E._verify_readiness(readiness, require_fxaa=True)
    value["renderer"]["carb_settings"]["/rtx/post/aa/op"] = 3
    readiness.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="FXAA"):
        E._verify_readiness(readiness, require_fxaa=True)
    del result


def test_policy_drift_after_capture_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    validated, _context, base, app, _events = CF._install_capture(monkeypatch, tmp_path)
    validated["job"]["source"] = {"files_lf": {F.SOURCE_FILE: "a" * 64}}
    settings = Settings()
    rep = SimpleNamespace(settings=settings)
    monkeypatch.setattr(C, "_carb_settings", lambda: settings)
    original_factory = C.V2._journaled_camera_capture

    def factory(*args: Any, **kwargs: Any) -> Any:
        capture = original_factory(*args, **kwargs)
        def drift(*capture_args: Any, **capture_kwargs: Any) -> Any:
            value = capture(*capture_args, **capture_kwargs)
            settings.values["/rtx/post/aa/op"] = 3
            return value
        return drift

    monkeypatch.setattr(C.V2, "_journaled_camera_capture", factory)
    def runner(current: Any, selected_app: Any, *, camera_capture: Any) -> Any:
        del current, selected_app
        return camera_capture(base, rep, [], CF.PROFILE, [1.0] * 3, [0.0] * 3, 0)
    monkeypatch.setattr(C.V1, "run_real", runner)
    with pytest.raises(ValueError, match="readback"):
        C.run_capture(validated, app)
