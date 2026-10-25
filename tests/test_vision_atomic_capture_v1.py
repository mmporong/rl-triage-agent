from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
C = importlib.import_module("vision_atomic_capture_v1")
P = importlib.import_module("vision_writer_payload_v1")


@pytest.fixture(autouse=True)
def _fixed_test_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C.R, "WIDTH", 10)
    monkeypatch.setattr(C.R, "HEIGHT", 10)


class Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, key: Any) -> Tensor:
        del key
        return self

    def detach(self) -> Tensor:
        return self

    def cpu(self) -> Tensor:
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class Sim:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.current_time = 2.0

    def set_camera_view(self, **kwargs: Any) -> None:
        del kwargs
        self.events.append("camera")


class Base:
    def __init__(self, events: list[str]) -> None:
        self.sim = Sim(events)
        self.scene = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_pos_w=Tensor([0.0, 0.0, 0.5]))
            )
        }


def _payload(ordinal: int, *, smoke: bool) -> dict[str, Any]:
    rp = "RenderProduct_0"
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint32)
    labels: dict[int, dict[str, str]] = {0: {"class": "robot/body"}}
    if smoke:
        for index, label in enumerate(sorted(C.FRAME.marker_labels(ordinal)), 1):
            mask[index, 1:9] = index
            labels[index] = {"class": label}
            role = "sentinel" if label == C.FRAME.SENTINEL else label.rsplit("/", 1)[-1]
            rgba[index, 1:9, :3] = C.FRAME.PALETTE[role]
    view = np.arange(16, dtype=float) + (100 if ordinal == 2 else 0)
    times = {
        "durationDenominator": 60,
        "durationNumerator": 1,
        "externalTimeOfSimNs": 2_000_000_000,
        "frameNumber": 55 + ordinal,
        "rationalTimeOfSimDenominator": 30,
        "rationalTimeOfSimNumerator": 55 + ordinal,
        "sampleTimeOffsetInSimFrames": 0,
        "simulationTime": 2.0,
        "simulationTimeMonotonic": 2.0,
        "swhFrameNumber": 0,
        "systemTime": 1000.0 + ordinal,
    }
    return {
        "swhFrameNumber": 0,
        "reference_time": (55 + ordinal, 30),
        "distribution_outputs": {},
        "trigger_outputs": {},
        "named_outputs": {},
        "annotators": {
            "camera": {rp: C.R.fixed_profile()["camera"]["camera_prim_path"]},
            "resolution": {rp: [10, 10]},
            "rgb": {rp: rgba},
            "semantic_segmentation": {
                rp: {"data": mask, "info": {"idToLabels": labels}}
            },
            "CameraParams": {
                rp: {
                    "cameraViewTransform": view,
                    "cameraProjection": np.eye(4).reshape(16),
                    "renderProductResolution": [10, 10],
                }
            },
            "IsaacReadTimes": {rp: times},
        },
    }


def test_marker_validation_uses_usd_prim_range_for_root_and_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision_api = object()
    rigid_api = object()

    class Prim:
        def __init__(self, blocked: bool = False) -> None:
            self.blocked = blocked

        def IsValid(self) -> bool:
            return True

        def HasAPI(self, api: object) -> bool:
            return self.blocked and api is collision_api

    class Attribute:
        def __init__(self, path: str) -> None:
            self.path = path

        def get(self) -> list[str]:
            return [self.path]

    class Node:
        def __init__(self, path: str) -> None:
            self.path = path

        def get_attribute(self, _name: str) -> Attribute:
            return Attribute(self.path)

    root = Prim()
    child = Prim()
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: root)
    pxr = SimpleNamespace(
        Usd=SimpleNamespace(PrimRange=lambda prim: [prim, child]),
        UsdGeom=SimpleNamespace(Imageable=lambda prim: prim),
        UsdPhysics=SimpleNamespace(CollisionAPI=collision_api, RigidBodyAPI=rigid_api),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    rig = object.__new__(C.MarkerRig)
    rig._items = {
        f"marker-{index}": SimpleNamespace(node=Node(f"/World/marker-{index}"))
        for index in range(9)
    }
    rig._paths = {}
    rig.bind_and_validate(stage)
    assert len(rig._paths) == 9

    child.blocked = True
    with pytest.raises(ValueError, match="physics API"):
        rig.bind_and_validate(stage)


class WriterBase:
    rep: Rep
    trigger: str

    def attach(self, render_product: Any, *, trigger: Any) -> None:
        self.rep.events.append(f"attach:{trigger}")
        self.trigger = trigger
        self.rep.writers.append(self)

    def detach(self) -> None:
        self.rep.events.append(f"detach:{self.trigger}")
        self.rep.writers.remove(self)


class Registry:
    @staticmethod
    def get_annotator(name: str, **kwargs: Any) -> str:
        return f"{name}:{kwargs}"


class Rep:
    Writer = WriterBase
    AnnotatorRegistry = Registry

    def __init__(self, events: list[str], *, smoke: bool) -> None:
        self.events = events
        self.smoke = smoke
        self.writers: list[Any] = []
        self.pending: str | None = None
        self.job = Path("job.json")
        self.trigger = SimpleNamespace(on_custom_event=lambda name: name)
        self.utils = SimpleNamespace(send_og_event=self._send)
        self.orchestrator = SimpleNamespace(step=self._step)

    def _send(self, event: str) -> None:
        self.events.append(f"event:{event}")
        self.pending = event

    def _step(self, **kwargs: Any) -> None:
        assert kwargs["delta_time"] == 0.0
        self.events.append("step")
        if self.pending is None:
            raise RuntimeError("test event가 없다")
        ordinal = (
            int(self.pending.rsplit("-", 1)[-1])
            if self.smoke
            else len(list(Path(self.job.parent).glob(f"{self.job.stem}.atomic-*.json")))
        )
        for writer in list(self.writers):
            if writer.trigger == self.pending:
                writer.write(_payload(ordinal, smoke=self.smoke))
        self.pending = None


class Marker:
    def __init__(self, rep: Rep, root: list[float]) -> None:
        rep.events.append(f"marker:{root}")
        self.rep = rep

    def bind_and_validate(self, stage: Any) -> None:
        del stage
        self.rep.events.append("marker-bind")

    def set_code(self, stage: Any, code: int) -> None:
        del stage
        self.rep.events.append(f"marker-code:{code}")


class Context:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def get_stage(self) -> object:
        return object()

    def reset_renderer_accumulation(self) -> None:
        self.events.append("reset")


PROFILE = {
    "camera_prim_path": C.R.fixed_profile()["camera"]["camera_prim_path"],
    "capture_primitive": "replicator_orchestrator_step_v1",
    "rt_subframes": 1,
    "pause_timeline": False,
    "delta_time_s": 0.0,
}


@pytest.mark.parametrize("mutation", ["camera_path", "resolution"])
def test_decoded_camera_must_match_fixed_profile(mutation: str) -> None:
    decoded = P.decode_writer_payload(_payload(0, smoke=False))
    if mutation == "camera_path":
        decoded["camera_path"] = "/World/WrongCamera"
    else:
        decoded["render_product_resolution"] = [9, 10]
    with pytest.raises(ValueError, match="fixed profile"):
        C._verify_decoded_camera(decoded)


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str
) -> tuple[dict[str, Any], Base, Rep, list[str]]:
    events: list[str] = []
    job = tmp_path / "job.json"
    job.write_text("{}\n", encoding="utf-8")
    validated = {
        "job": {"mode": mode, "source": {"files_lf": {}}},
        "job_path": job,
        "output_directory": tmp_path / "output",
    }
    base = Base(events)
    rep = Rep(events, smoke=mode == C.R.RENDER_SYNC_SMOKE_MODE)
    rep.job = job
    context = Context(events)
    monkeypatch.setattr(C.READY, "_physics_snapshot", lambda current: {"state": 1})
    monkeypatch.setattr(
        C.READY, "_renderer_readback", lambda *args: {"renderer": "fixed"}
    )

    def ready(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        events.append("readiness")
        return {"status": "ready"}

    monkeypatch.setattr(C.READY, "_wait_capture_ready", ready)
    dependencies = {
        "usd_context": lambda: (context, object()),
        "settings": lambda: SimpleNamespace(get=lambda key: None),
        "writer": lambda selected_rep, sink: _writer(selected_rep, sink),
        "marker": Marker,
    }
    factory = C.make_capture_factory(validated, object(), dependencies=dependencies)
    return {"validated": validated, "factory": factory}, base, rep, events


def _writer(rep: Rep, sink: Any) -> Any:
    writer = P.create_payload_writer(rep, sink)
    writer.rep = rep
    return writer


def test_smoke_waits_before_four_fresh_writers_and_seals_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup, base, rep, events = _setup(
        tmp_path, monkeypatch, mode=C.R.RENDER_SYNC_SMOKE_MODE
    )
    capture, cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", PROFILE)
    for ordinal in range(4):
        result = capture(
            base,
            rep,
            [],
            PROFILE,
            [3.0, 3.0, 2.0] if ordinal != 2 else [-3.0, 3.0, 2.0],
            [0.0, 0.0, 0.35],
            ordinal,
        )
        assert set(result) == {
            "rgb",
            "read_times",
            "reference_time",
            "atomic_bundle",
            "atomic_journal",
            "atomic_npz",
        }
        P.load_capture_journal(setup["validated"]["job_path"], ordinal)
    assert events.index("readiness") < next(
        i for i, item in enumerate(events) if item.startswith("attach:")
    )
    cleanup()
    assert sum(item.startswith("attach:") for item in events) == 4
    assert sum(item.startswith("detach:") for item in events) == 4
    assert [item for item in events if item.startswith("marker-code:")] == [
        "marker-code:0",
        "marker-code:1",
        "marker-code:2",
        "marker-code:3",
    ]
    assert (
        events.index("marker-code:0") < events.index("camera") < events.index("reset")
    )


def test_record_uses_one_persistent_writer_and_rejects_stray_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C.R, "FRAME_COUNT", 2)
    setup, base, rep, events = _setup(tmp_path, monkeypatch, mode=C.R.RECORD_MODE)
    capture, cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", PROFILE)
    capture(base, rep, [], PROFILE, [3.0] * 3, [0.0] * 3, 0)
    rep.writers[0].write(_payload(1, smoke=False))
    with pytest.raises(ValueError, match="unsolicited"):
        capture(base, rep, [], PROFILE, [3.0] * 3, [0.0] * 3, 1)
    with pytest.raises(ValueError, match="끝나지 않았다|unsolicited"):
        cleanup()
    assert sum(item.startswith("attach:") for item in events) == 1
    assert sum(item.startswith("detach:") for item in events) == 1


def test_record_two_capture_success_detaches_on_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(C.R, "FRAME_COUNT", 2)
    setup, base, rep, events = _setup(tmp_path, monkeypatch, mode=C.R.RECORD_MODE)
    capture, cleanup = setup["factory"](base, rep, "/Render/RenderProduct_0", PROFILE)
    for ordinal in range(2):
        capture(base, rep, [], PROFILE, [3.0] * 3, [0.0] * 3, ordinal)
    cleanup()
    assert sum(item.startswith("attach:") for item in events) == 1
    assert sum(item.startswith("detach:") for item in events) == 1


def test_marker_specs_create_exact_nine_fixed_semantic_variants() -> None:
    rows = C._marker_specs([1.0, 2.0, 3.0])
    assert len(rows) == 9
    assert {row["label"] for row in rows} == {
        C.FRAME.SENTINEL,
        *{
            f"{C.FRAME.MARKER_PREFIX}bit{bit}/rail{rail}/{role}"
            for bit in range(2)
            for rail in range(2)
            for role in ("active", "inactive")
        },
    }
    assert sum(row["visible"] is True for row in rows) == 1


def test_marker_rig_uses_replicator_cube_material_and_fixed_semantics() -> None:
    created: list[tuple[str, dict[str, Any]]] = []

    class Create:
        @staticmethod
        def material_omnipbr(**kwargs: Any) -> str:
            created.append(("material", kwargs))
            return f"material-{len(created)}"

        @staticmethod
        def cube(**kwargs: Any) -> object:
            created.append(("cube", kwargs))
            return object()

    C.MarkerRig(SimpleNamespace(create=Create), [0.0, 0.0, 0.0])
    cubes = [row for kind, row in created if kind == "cube"]
    assert len(cubes) == 9
    assert all(
        row["as_mesh"] is False and row["scale"] == (0.09, 0.09, 0.09) for row in cubes
    )
    assert {row["semantics"][0][1] for row in cubes} == {
        row["label"] for row in C._marker_specs([0.0, 0.0, 0.0])
    }
