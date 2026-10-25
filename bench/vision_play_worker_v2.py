"""Isaac 노드 확장과 capture annotator 등록을 확인한 V04 Play worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_play_worker_v1 as V1
import vision_replicator_startup_v1 as STARTUP
import vision_timing_adapter_v1 as TIMING

SCHEMA = "vision_play_worker_registration_v2"
EXTENSION = "isaacsim.core.nodes"
ANNOTATORS = ("rgb", "IsaacReadTimes", "ReferenceTime")
RECEIPT_FILE = "renderer_registration.json"

CaptureRunner = Callable[[Mapping[str, Any], Any], tuple[dict[str, Path], Path]]
EnableExtension = Callable[[str], bool]
RegistrationProbe = Callable[[], Mapping[str, bool]]
StartupConfigure = Callable[[], Mapping[str, Mapping[str, Any]]]


class _Isaac45AppProxy:
    def __init__(self, app: Any) -> None:
        self._app = app

    def update(self) -> Any:
        return self._app.update()

    def close(self) -> Any:
        return self._app.close(wait_for_replicator=False)


def _default_app_factory() -> _Isaac45AppProxy:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = AppLauncher(headless=True, enable_cameras=True, fast_shutdown=False).app
    return _Isaac45AppProxy(app)


def _enable_extension(name: str) -> bool:
    from isaacsim.core.utils.extensions import (  # pyright: ignore[reportMissingImports]
        enable_extension,
    )

    return enable_extension(name)


def _registered_annotators() -> dict[str, bool]:
    import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]

    return {
        "rgb": rep.AnnotatorRegistry.get_annotator("rgb") is not None,
        "IsaacReadTimes": rep.AnnotatorRegistry.get_annotator("IsaacReadTimes") is not None,
        "ReferenceTime": rep.annotators.get("ReferenceTime") is not None,
    }


def _write_registration(output: Path, evidence: Mapping[str, Any]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / RECEIPT_FILE
    R.write_new(path, dict(evidence))
    return path


def _guarded_runner(
    validated: Mapping[str, Any],
    app: Any,
    *,
    capture_runner: CaptureRunner,
    enable_extension: EnableExtension,
    registration_probe: RegistrationProbe,
    startup_configure: StartupConfigure,
) -> tuple[dict[str, Path], Path]:
    output = validated["output_directory"]
    evidence: dict[str, Any] = {
        "schema": SCHEMA,
        "extension": EXTENSION,
        "enable_result": False,
        "app_update_completed": False,
        "replicator_startup": None,
        "annotators": {name: False for name in ANNOTATORS},
        "status": "failed",
    }
    try:
        enabled = enable_extension(EXTENSION)
        evidence["enable_result"] = enabled
        if enabled is not True:
            raise RuntimeError(f"{EXTENSION} extension enable이 true를 반환하지 않았다")
        app.update()
        evidence["app_update_completed"] = True
        configured = startup_configure()
        before = configured.get("before")
        after = configured.get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise TypeError("Replicator startup readback은 before/after mapping이어야 한다")
        evidence["replicator_startup"] = {"before": dict(before), "after": dict(after)}
        print(json.dumps({
            "schema": SCHEMA,
            "replicator_startup": evidence["replicator_startup"],
        }, ensure_ascii=False, sort_keys=True), flush=True)
        registered = dict(registration_probe())
        evidence["annotators"] = {name: registered.get(name) is True for name in ANNOTATORS}
        missing = [name for name in ANNOTATORS if evidence["annotators"][name] is not True]
        if missing:
            raise RuntimeError(f"capture annotator 등록 누락: {','.join(missing)}")
        evidence["status"] = "ready"
    except BaseException:  # v1이 failure 영수증과 app.close를 이어서 소유한다.
        _write_registration(output, evidence)
        raise

    # capture 중 강제 종료되어도 설정·등록 증거를 보존한다. v1이 output 생성은 소유한다.
    job_path = validated["job_path"]
    R.write_new(job_path.with_suffix(".renderer_registration.json"), evidence)
    try:
        artifacts, primary = capture_runner(validated, app)
    except BaseException:  # 등록 증거를 봉인한 뒤 v1 경계로 전달한다.
        _write_registration(output, evidence)
        raise
    _write_registration(output, evidence)
    return artifacts, primary


def _journaled_camera_capture(
    validated: Mapping[str, Any], *, preserve_rgb: bool,
) -> V1.CameraCapture:
    def capture(
        base: Any, rep: Any, annotators: list[Any], camera_profile: Mapping[str, Any],
        eye: list[float], lookat: list[float], ordinal: int,
    ) -> dict[str, Any]:
        raw = V1._set_camera_and_capture(
            base, rep, annotators, camera_profile, eye, lookat
        )
        rgb_sha256 = hashlib.sha256(raw["rgb"]).hexdigest()
        TIMING.write_capture_journal(
            validated["job_path"], ordinal, raw["read_times"], raw["reference_time"],
            rgb_sha256, rgb_bytes=raw["rgb"] if preserve_rgb else None,
        )
        normalized = TIMING.normalize_render_timing(
            raw["read_times"], raw["reference_time"], float(base.sim.current_time)
        )
        return {"rgb": raw["rgb"], "read_times": normalized["read_times"],
                "reference_time": normalized["reference_time"]}

    return capture


def _default_capture_runner(
    validated: Mapping[str, Any], app: Any,
) -> tuple[dict[str, Path], Path]:
    smoke = validated["job"]["mode"] == R.RENDER_SYNC_SMOKE_MODE
    selected = V1.run_render_sync_smoke if smoke else V1.run_real
    return selected(
        validated, app,
        camera_capture=_journaled_camera_capture(validated, preserve_rgb=smoke),
    )


def execute(
    job_path: Path,
    *,
    app_factory: Callable[[], Any] | None = None,
    runner: CaptureRunner | None = None,
    enable_extension: EnableExtension | None = None,
    registration_probe: RegistrationProbe | None = None,
    startup_configure: StartupConfigure | None = None,
) -> int:
    selected_enable = enable_extension or _enable_extension
    selected_probe = registration_probe or _registered_annotators
    selected_startup = startup_configure or STARTUP.configure_replicator_startup

    def injected(validated: Mapping[str, Any], app: Any) -> tuple[dict[str, Path], Path]:
        selected_runner = runner or _default_capture_runner
        return _guarded_runner(
            validated,
            app,
            capture_runner=selected_runner,
            enable_extension=selected_enable,
            registration_probe=selected_probe,
            startup_configure=selected_startup,
        )

    return V1.execute(
        job_path, app_factory=app_factory or _default_app_factory, runner=injected
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args(argv)
    return execute(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
