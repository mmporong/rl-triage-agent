from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
M = importlib.import_module("vision_prepared_record_sdk_v1")
R = importlib.import_module("record_vision_evidence_v1")


def _times(frame: int, simulation_time: float) -> dict[str, Any]:
    return {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": frame * 1_000_000,
        "frameNumber": frame,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": frame,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": simulation_time,
        "simulationTimeMonotonic": simulation_time,
        "swhFrameNumber": frame,
        "systemTime": 1000.0 + frame,
    }


def _payload(
    frame: int, width: int, height: int, simulation_time: float
) -> dict[str, Any]:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = frame % 251
    rgba[:, :, 3] = 255
    return {
        "swhFrameNumber": frame,
        "reference_time": (frame, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {"callback_ref": f"callback-{frame}"},
        "annotators": {
            "camera": {"RenderProduct_0": "/OmniverseKit_Persp"},
            "resolution": {
                "RenderProduct_0": np.array([width, height], dtype=np.int32)
            },
            "rgb": {"RenderProduct_0": {"data": rgba}},
            "semantic_segmentation": {
                "RenderProduct_0": {
                    "data": np.zeros((height, width), dtype=np.uint32),
                    "idToLabels": {"0": {"class": "robot/body"}},
                }
            },
            "CameraParams": {
                "RenderProduct_0": {
                    "cameraViewTransform": np.arange(16, dtype=np.float64),
                    "cameraProjection": np.eye(4).reshape(16),
                    "renderProductResolution": np.array(
                        [width, height], dtype=np.int32
                    ),
                }
            },
            "IsaacReadTimes": {
                "RenderProduct_0": _times(frame, simulation_time)
            },
        },
    }


class _Tensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)

    def __getitem__(self, item: Any) -> _Tensor:
        return _Tensor(self.values[item])

    def detach(self) -> _Tensor:
        return self

    def cpu(self) -> _Tensor:
        return self

    def flatten(self) -> _Tensor:
        return _Tensor(self.values.flatten())

    def tolist(self) -> list[Any]:
        return self.values.tolist()


class _Sim:
    def __init__(self, log: list[str]) -> None:
        self.current_time = 0.0
        self.log = log
        self.camera_calls: list[dict[str, Any]] = []

    def set_camera_view(self, **kwargs: Any) -> None:
        self.camera_calls.append(kwargs)


class _Base:
    def __init__(self, log: list[str]) -> None:
        self.sim = _Sim(log)
        self.common_step_counter = 0
        data = SimpleNamespace(
            root_pos_w=_Tensor([[0.0, 0.0, 0.4]]),
            root_lin_vel_b=_Tensor([[0.0, 0.0, 0.0]]),
            root_ang_vel_b=_Tensor([[0.0, 0.0, 0.0]]),
            projected_gravity_b=_Tensor([[0.0, 0.0, -1.0]]),
        )
        self.scene = {"robot": SimpleNamespace(data=data)}


class _Context:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def reset_renderer_accumulation(self) -> None:
        self.log.append("renderer_reset")


class _RenderProduct:
    path = "/Render/RenderProduct_0"


class _App:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.close_calls = 0

    def update(self) -> None:
        self.log.append("app.update")

    def close(self) -> None:
        self.close_calls += 1


class _Rep:
    class Writer:
        def write(self, data: dict[str, Any]) -> None:
            del data
            raise NotImplementedError

        def attach(self, render_product: Any, *, trigger: Any) -> None:
            del render_product, trigger
            _Rep.writer = self

        def detach(self) -> None:
            _Rep.log.append("detach")
            if _Rep.late_callback:
                self.write(_Rep.payload())

    class AnnotatorRegistry:
        @staticmethod
        def get_annotator(name: str, **kwargs: Any) -> object:
            del name, kwargs
            return object()

    class _Trigger:
        @staticmethod
        def on_custom_event(name: str) -> str:
            return name

    class _Utils:
        @staticmethod
        def send_og_event(name: str) -> None:
            del name
            _Rep.pending = True

    trigger = _Trigger()
    utils = _Utils()
    orchestrator = object()
    writer: Any = None
    pending = False
    frame = 10
    width = 4
    height = 3
    late_callback = False
    simulation_time = 0.0
    log: ClassVar[list[str]] = []

    @classmethod
    def payload(cls) -> dict[str, Any]:
        value = _payload(cls.frame, cls.width, cls.height, cls.simulation_time)
        cls.frame += 1
        return value


def _validated(tmp_path: Path) -> dict[str, Any]:
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"mode": M.MODE}), encoding="utf-8")
    return {
        "job": {"mode": M.MODE},
        "job_path": job,
        "job_sha256": hashlib.sha256(job.read_bytes()).hexdigest(),
        "output_directory": tmp_path / "output",
    }


def _dependencies(
    tmp_path: Path,
    log: list[str],
    *,
    registration_ok: bool = True,
    late_callback: bool = False,
    fail_prepare_render: bool = False,
) -> dict[str, Any]:
    context = _Context(log)
    _Rep.log = log
    _Rep.frame = 10
    _Rep.pending = False
    _Rep.writer = None
    _Rep.late_callback = late_callback
    _Rep.simulation_time = 0.0

    def render(_orchestrator: Any, profile: dict[str, Any]) -> None:
        assert profile["delta_time_s"] == 0.0
        assert profile["rt_subframes"] == 1
        assert profile["pause_timeline"] is False
        log.append("delta0_render")
        if fail_prepare_render and log.count("delta0_render") == 1:
            raise RuntimeError("prepare render failed")
        if _Rep.pending:
            _Rep.pending = False
            _Rep.writer.write(_Rep.payload())

    def write_reset(
        validated: dict[str, Any], ordinal: int, **kwargs: Any
    ) -> Path:
        del kwargs
        path = validated["job_path"].with_name(f"reset-{ordinal}.json")
        path.write_text("{}", encoding="utf-8")
        return path

    def write_atomic(
        job_path: Path,
        ordinal: int,
        mode: str,
        decoded: dict[str, Any],
        bundle: dict[str, Any],
    ) -> tuple[Path, Path]:
        del mode, decoded, bundle
        json_path = job_path.with_name(f"atomic-{ordinal}.json")
        npz_path = job_path.with_name(f"atomic-{ordinal}.npz")
        json_path.write_text("{}", encoding="utf-8")
        npz_path.write_bytes(b"npz")
        return json_path, npz_path

    def write_timing(
        job_path: Path, ordinal: int, *args: Any, **kwargs: Any
    ) -> Path:
        del args, kwargs
        path = job_path.with_name(f"timing-{ordinal}.json")
        path.write_text("{}", encoding="utf-8")
        return path

    def run_real(
        validated: dict[str, Any],
        app: Any,
        *,
        camera_capture_factory: Any,
    ) -> tuple[dict[str, Path], Path]:
        del app
        log.append("run_real")
        assert validated["job"]["mode"] == R.RECORD_MODE
        output = validated["output_directory"]
        output.mkdir()
        base = _Base(log)
        profile = R.fixed_profile()["camera"]
        capture, cleanup = camera_capture_factory(
            base, _Rep, _RenderProduct(), profile
        )
        try:
            for ordinal in range(R.FRAME_COUNT):
                base.common_step_counter = ordinal
                base.sim.current_time = ordinal / R.FPS
                _Rep.simulation_time = base.sim.current_time
                result = capture(
                    base,
                    _Rep,
                    [],
                    profile,
                    [3.0, 3.0, 2.4],
                    [0.0, 0.0, 0.75],
                    ordinal,
                )
                assert len(result["rgb"]) == _Rep.width * _Rep.height * 3
        finally:
            cleanup()
        evidence = output / "evidence"
        evidence.mkdir()
        trace = evidence / "raw_trace.json"
        trace.write_text(
            json.dumps(
                {
                    "encoder_receipt": {"frames_written": R.FRAME_COUNT},
                    "timing_receipt": {"frame_count": R.FRAME_COUNT},
                }
            ),
            encoding="utf-8",
        )
        paths = {"trace": trace}
        for name in ("measurement_1", "spec_1", "measurement_2", "spec_2"):
            path = evidence / f"{name}.json"
            path.write_text("{}", encoding="utf-8")
            paths[name] = path
        video = output / "same_play.mp4"
        video.write_bytes(b"video")
        return paths, video

    return {
        "rep": _Rep,
        "settings": lambda: object(),
        "usd_context": lambda: (context, object()),
        "enable_extension": lambda name: log.append(f"enable:{name}") or True,
        "startup": lambda rep, settings: (
            log.append("startup")
            or {"before": {"status": "stopped"}, "after": {"status": "started"}}
        ),
        "registration_probe": lambda: {
            name: registration_ok for name in M.ANNOTATORS
        },
        "run_real": run_real,
        "writer": M._create_writer,
        "physics_snapshot": lambda base: {
            "step": base.common_step_counter,
            "time": base.sim.current_time,
        },
        "readiness": lambda *args: log.append("readiness"),
        "renderer_readback": lambda *args: {
            "attached_hydra_engine_names": ["rtx"],
            "carb_settings": {},
            "expected_capture_profile": {},
            "renderer_accumulation_reset": "reset",
        },
        "fxaa_required": lambda job: False,
        "fxaa_apply": lambda rep, settings: None,
        "fxaa_readback": lambda settings: {},
        "fxaa_verify": lambda value: None,
        "capture_render_step": render,
        "write_reset": write_reset,
        "write_atomic_journal": write_atomic,
        "write_timing_journal": write_timing,
    }


def test_record625_roundtrip_preserves_startup_prepare_and_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    app = _App(log)
    sealed: list[dict[str, Any]] = []
    monkeypatch.setattr(M.R, "WIDTH", _Rep.width)
    monkeypatch.setattr(M.R, "HEIGHT", _Rep.height)
    terminal = M.run_prepared_record(
        _validated(tmp_path),
        app,
        seal_report=lambda report: sealed.append(dict(report)),
        dependencies=_dependencies(tmp_path, log),
    )
    assert terminal["approved"] is True
    assert terminal["lifecycle"]["arms_completed"] == 625
    assert terminal["prepared_record"]["committed_count"] == 625
    assert terminal["prepared_record"]["raw_callback_count"] == 626
    assert terminal["record_result"]["frame_count"] == 625
    assert set(terminal["artifacts"]) == {
        "raw_trace",
        "encoder_receipt",
        "timing_receipt",
        "measurement_1",
        "spec_1",
        "measurement_2",
        "spec_2",
        "video",
    }
    assert log.index(f"enable:{M.EXTENSION}") < log.index("app.update")
    assert log.index("app.update") < log.index("startup") < log.index("run_real")
    assert log.count("delta0_render") == 626
    assert sealed[0]["lifecycle"]["envelope_sealed"] is False
    assert app.close_calls == 0


def test_startup_phase_failure_seals_partial_evidence_without_run_real(
    tmp_path: Path,
) -> None:
    log: list[str] = []
    app = _App(log)
    sealed: list[dict[str, Any]] = []
    with pytest.raises(M.PreparedRecordSDKFailure) as caught:
        M.run_prepared_record(
            _validated(tmp_path),
            app,
            seal_report=lambda report: sealed.append(dict(report)),
            dependencies=_dependencies(tmp_path, log, registration_ok=False),
        )
    assert caught.value.report["approved"] is False
    assert "run_real" not in log
    assert sealed[0]["lifecycle"]["errors"][0]["phase"] == "annotator_registrations"
    assert app.close_calls == 0


def test_callback_during_detach_holds_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log: list[str] = []
    app = _App(log)
    monkeypatch.setattr(M.R, "WIDTH", _Rep.width)
    monkeypatch.setattr(M.R, "HEIGHT", _Rep.height)
    with pytest.raises(M.PreparedRecordSDKFailure) as caught:
        M.run_prepared_record(
            _validated(tmp_path),
            app,
            seal_report=lambda report: None,
            dependencies=_dependencies(tmp_path, log, late_callback=True),
        )
    report = caught.value.report
    assert report["approved"] is False
    assert report["prepared_record"]["failure"]["code"] == "callback_during_detach"
    assert report["record_result"] is None


def test_factory_prepare_failure_detaches_and_preserves_primary(tmp_path: Path) -> None:
    log: list[str] = []
    app = _App(log)
    with pytest.raises(M.PreparedRecordSDKFailure) as caught:
        M.run_prepared_record(
            _validated(tmp_path),
            app,
            seal_report=lambda report: None,
            dependencies=_dependencies(tmp_path, log, fail_prepare_render=True),
        )
    assert str(caught.value.primary) == "prepare render failed"
    assert caught.value.report["lifecycle"]["writer_detach_attempted"] is True
    assert caught.value.report["lifecycle"]["writer_detached"] is True
    assert caught.value.report["prepared_record"]["approved"] is False
    assert log.count("detach") == 1
