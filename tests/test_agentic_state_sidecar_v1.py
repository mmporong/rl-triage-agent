from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import record_agentic_state_sidecar_v1 as S  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as R  # pyright: ignore[reportMissingImports]

CK = "a" * 64
OTHER_CK = "b" * 64
JOINTS = ["FL_hip_joint", "FR_hip_joint"]
BODIES = ["base", "FL_foot"]


def _source(**changes: Any) -> Any:
    values: dict[str, Any] = {"source_id": "go2-f2-seed2033", "checkpoint_sha256": CK, "task": R.TASK}
    values.update(changes)
    return S.SourceIdentity(**values)


def _description(**changes: Any) -> dict[str, Any]:
    available: dict[str, Any] = {name: {"unit": unit} for name, _, unit in S.ARTICULATION_FIELDS}
    available.update({name: {"unit": unit} for name, _, unit, _ in S.CONTACT_FIELDS})
    value: dict[str, Any] = {
        "reader": "synthetic_v1",
        "joint_names": list(JOINTS),
        "contact_body_names": list(BODIES),
        "available": available,
        "unavailable": {},
        "not_recorded": dict(S.NOT_RECORDED),
        "contact_force_threshold_n": 1.0,
        "actuators": None,
    }
    value.update(changes)
    return value


def _sample(names: list[str], base: float = 0.5) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in names:
        count, components = S._field_shape(name, _description())
        if components == 1:
            fields[name] = [base + index for index in range(count)]
        else:
            fields[name] = [[base, 0.0, float(index)] for index in range(count)]
    return {"fields": fields, "frame_unavailable": {}}


class SyntheticReader:
    """테스트가 시각과 상태를 정하는 reader. Isaac 없이 recorder 계약만 시험한다."""

    def __init__(self, description: dict[str, Any] | None = None) -> None:
        self.description = description or _description()
        self.now: Any = (0.01, 0)
        self.sample: Any = _sample(sorted(self.description["available"]))

    def describe(self) -> dict[str, Any]:
        return copy.deepcopy(self.description)

    def clock(self) -> Any:
        return self.now

    def read(self) -> Any:
        return copy.deepcopy(self.sample)


def _started(tmp_path: Path, reader: SyntheticReader | None = None, **kwargs: Any) -> Any:
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source(), **kwargs)
    recorder.start(reader or SyntheticReader())
    return recorder


def _drive(recorder: Any, reader: SyntheticReader, frames: int) -> None:
    for index in range(frames):
        reader.now = (0.01 + 0.02 * index, index)
        recorder.record(index, capture_ordinal=index)


def _records(directory: Path) -> list[dict[str, Any]]:
    lines = (directory / S.RECORDS_NAME).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


# --- import 경계 ---------------------------------------------------------------


def test_cpu_import_does_not_load_isaac_kit_or_torch() -> None:
    script = (
        "import sys; sys.dont_write_bytecode = True; "
        f"sys.path.insert(0, {str(ROOT / 'bench')!r}); "
        "import record_agentic_state_sidecar_v1; "
        "heavy = ('isaaclab', 'isaaclab_tasks', 'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'gymnasium', 'rsl_rl'); "
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))"
    )
    # -I를 쓰지 않는다: 사용자 site의 torch가 import 가능한 상태에서도 모듈이 불러오지 않아야 한다.
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == "[]"


# --- recorder 계약 -------------------------------------------------------------


def test_recorder_seals_manifest_with_identity_time_and_no_labels(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader, expected_frames=3)
    _drive(recorder, reader, 3)
    manifest = recorder.finalize()
    directory = tmp_path / "sidecar"
    rows = _records(directory)
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert [row["simulation_step"] for row in rows] == [0, 1, 2]
    assert rows[1]["simulation_time_s"] == 0.01 + 0.02
    assert rows[0]["fields"]["joint_pos"] == [0.5, 1.5]
    assert manifest["source"]["checkpoint_sha256"] == CK
    assert manifest["source"]["task"] == R.TASK
    assert manifest["source"]["env_index"] == 0
    assert manifest["frame_count"] == 3
    assert manifest["time_basis"] == S.TIME_BASIS
    assert manifest["records"]["sha256"] == R.sha256_file(directory / S.RECORDS_NAME)
    assert manifest["sidecar_source"] == {"path": S.SOURCE_PATH, "sha256_lf": S.self_sha256_lf()}
    assert manifest["behavior_label"] is None
    assert manifest["thresholds_created"] is False
    assert manifest["fixed_evaluation_used_as_behavior_ground_truth"] is False
    assert manifest["used_for_decision"] is False
    assert "joint_acc" in manifest["not_recorded_fields"]
    assert R.read_object(directory / S.MANIFEST_NAME) == manifest
    loaded, records = S.load_sidecar(directory)
    assert loaded == manifest and records == rows


def test_missing_field_is_declared_and_never_zero_filled(tmp_path: Path) -> None:
    description = _description()
    del description["available"]["computed_torque"]
    description["unavailable"] = {"computed_torque": "articulation_attribute_missing"}
    reader = SyntheticReader(description)
    recorder = _started(tmp_path, reader)
    _drive(recorder, reader, 2)
    manifest = recorder.finalize()
    assert manifest["unavailable_fields"] == {"computed_torque": "articulation_attribute_missing"}
    for row in _records(tmp_path / "sidecar"):
        assert "computed_torque" not in row["fields"]


def test_field_disappearing_mid_run_is_rejected(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    _drive(recorder, reader, 1)
    del reader.sample["fields"]["applied_torque"]
    reader.now = (0.03, 1)
    with pytest.raises(S.SidecarError, match="사라졌"):
        recorder.record(1, capture_ordinal=1)


def test_undeclared_field_and_unregistered_frame_skip_are_rejected(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.sample["fields"]["joint_acc"] = [0.0, 0.0]
    with pytest.raises(S.SidecarError):
        recorder.record(0, capture_ordinal=0)
    reader.sample = _sample(sorted(reader.description["available"]))
    reader.sample["fields"].pop("joint_pos")
    reader.sample["frame_unavailable"] = {"joint_pos": "contact_sensor_outdated_read_skipped"}
    with pytest.raises(S.SidecarError, match="등록 밖"):
        recorder.record(0, capture_ordinal=0)


def test_description_must_cover_every_registered_field_once() -> None:
    description = _description()
    del description["available"]["joint_vel"]
    with pytest.raises(S.SidecarError, match="정확히 한 곳"):
        S._validate_description(description)
    description = _description(unavailable={"joint_pos": "x"})
    with pytest.raises(S.SidecarError, match="정확히 한 곳"):
        S._validate_description(description)
    with pytest.raises(S.SidecarError, match="중복"):
        S._validate_description(_description(joint_names=["a", "a"]))


@pytest.mark.parametrize(
    ("now", "message"),
    [((0.005, 1), "시각"), ((0.01, 2), "시각"), ((0.05, 0), "step")],
    ids=["time-reversal", "time-stall", "step-stall"],
)
def test_time_reversal_or_stall_is_rejected(tmp_path: Path, now: tuple[float, int], message: str) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.now = (0.01, 0)
    recorder.record(0, capture_ordinal=0)
    reader.now = now
    with pytest.raises(S.SidecarError, match=message):
        recorder.record(1, capture_ordinal=1)


@pytest.mark.parametrize(
    ("prior", "index", "ordinal"),
    [(0, 1, 1), (0, 0, 1), (1, 0, 0), (1, 2, 2), (1, 1, 0)],
    ids=["first-skip", "ordinal", "duplicate", "skip", "ordinal-after"],
)
def test_frame_skip_duplicate_or_ordinal_mismatch_is_rejected(
    tmp_path: Path, prior: int, index: int, ordinal: int
) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    _drive(recorder, reader, prior)
    reader.now = (0.5, 9)
    with pytest.raises(S.SidecarError, match="순서"):
        recorder.record(index, capture_ordinal=ordinal)


def test_render_changing_simulation_clock_is_rejected(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.now = (0.03, 1)
    with pytest.raises(S.SidecarError, match="render 전후"):
        recorder.record(0, capture_ordinal=0, clock_before=(0.01, 0))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "1.0"])
def test_non_finite_or_non_numeric_state_is_rejected(tmp_path: Path, bad: Any) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.sample["fields"]["joint_vel"] = [0.0, bad]
    with pytest.raises(S.SidecarError):
        recorder.record(0, capture_ordinal=0)


@pytest.mark.parametrize("clock", [(math.nan, 0), (0.01, True), (0.01, -1), (0.01,), [0.01, 0]])
def test_bad_clock_is_rejected(tmp_path: Path, clock: Any) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.now = clock
    with pytest.raises(S.SidecarError):
        recorder.record(0, capture_ordinal=0)


def test_vector_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    reader.sample["fields"]["contact_net_force_w"] = [[0.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(S.SidecarError, match="벡터"):
        recorder.record(0, capture_ordinal=0)
    reader.sample = _sample(sorted(reader.description["available"]))
    reader.sample["fields"]["joint_pos"] = [0.0]
    with pytest.raises(S.SidecarError, match="길이"):
        recorder.record(0, capture_ordinal=0)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "sidecar"
    target.mkdir()
    (target / "keep.txt").write_text("old", encoding="utf-8")
    recorder = S.SidecarRecorder(target, _source())
    with pytest.raises(FileExistsError):
        recorder.start(SyntheticReader())
    recorder.fail(RuntimeError("x"), stage="start")
    assert sorted(path.name for path in target.iterdir()) == ["keep.txt"]
    with pytest.raises(FileExistsError):
        S.check_destination(target)


def test_finalize_rejects_frame_count_other_than_registered(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader, expected_frames=3)
    _drive(recorder, reader, 2)
    assert recorder.seal() is None
    assert recorder.status == "failed"
    failure = R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)
    assert failure["stage"] == "finalize" and failure["records_written"] == 2
    assert failure["partial_records"]["sha256"] == R.sha256_file(tmp_path / "sidecar" / S.RECORDS_NAME)
    assert not (tmp_path / "sidecar" / S.MANIFEST_NAME).exists()


def test_failure_is_sticky_and_written_once(tmp_path: Path) -> None:
    recorder = _started(tmp_path)
    recorder.fail(RuntimeError("first"), stage="a")
    recorder.fail(RuntimeError("second"), stage="b")
    failure = R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)
    assert failure["error_message"] == "first" and failure["stage"] == "a"
    with pytest.raises(S.SidecarError):
        recorder.record(0, capture_ordinal=0)


def test_source_identity_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        _source(checkpoint_sha256="A" * 64)
    with pytest.raises(ValueError):
        _source(env_index=-1)
    with pytest.raises(ValueError):
        _source(source_commit="abc")
    with pytest.raises(ValueError):
        _source(source_id="../x")


# --- Isaac reader (fake tensor) --------------------------------------------------


class FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __getitem__(self, index: int) -> FakeTensor:
        return FakeTensor(self.value[index])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> Any:
        return copy.deepcopy(self.value)

    def any(self) -> FakeTensor:
        return FakeTensor(any(bool(item) for item in self.value))

    def item(self) -> Any:
        return self.value


class FakeArticulationData:
    def __init__(self) -> None:
        self.joint_names = list(JOINTS)
        self.joint_pos = FakeTensor([[0.1, 0.2], [9.0, 9.0]])
        self.joint_vel = FakeTensor([[1.0, -1.0], [9.0, 9.0]])
        self.joint_pos_target = FakeTensor([[0.0, 0.0], [9.0, 9.0]])
        self.computed_torque = FakeTensor([[30.0, -30.0], [9.0, 9.0]])
        self.applied_torque = FakeTensor([[23.5, -23.5], [9.0, 9.0]])
        self.joint_acc_reads = 0

    @property
    def joint_acc(self) -> Any:
        self.joint_acc_reads += 1
        raise AssertionError("joint_acc는 부수효과가 있어 읽으면 안 된다")


class FakeContact:
    def __init__(self, *, outdated: list[bool] | None = None, track_air_time: bool = True) -> None:
        self.cfg = SimpleNamespace(force_threshold=1.0, track_air_time=track_air_time)
        self.body_names = list(BODIES)
        self._is_outdated = FakeTensor(outdated or [False, False])
        self.data_reads = 0
        self._data = SimpleNamespace(
            net_forces_w=FakeTensor([[[0.0, 0.0, 10.0], [0.0, 0.0, 40.0]], [[9.0] * 3] * 2]),
            current_contact_time=FakeTensor([[0.0, 0.3], [9.0, 9.0]]),
            current_air_time=FakeTensor([[0.0, 0.0], [9.0, 9.0]]),
        )

    @property
    def data(self) -> Any:
        self.data_reads += 1
        return self._data


class FakeDCMotor:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
                                   stiffness=25.0, damping=0.5)
        self.joint_names = list(JOINTS)

    def _clip_effort(self, effort: Any) -> Any:
        return effort


def _base(contact: FakeContact | None = None) -> Any:
    robot = SimpleNamespace(data=FakeArticulationData(), joint_names=list(JOINTS),
                            actuators={"base_legs": FakeDCMotor()})
    scene: dict[str, Any] = {"robot": robot}
    if contact is not None:
        scene["contact_forces"] = contact
    return SimpleNamespace(scene=scene, sim=SimpleNamespace(current_time=0.01), common_step_counter=0)


def test_isaac_reader_reads_only_selected_env_and_never_joint_acc() -> None:
    contact = FakeContact()
    base = _base(contact)
    reader = S.IsaacStateReader(base, env_index=0)
    description = S._validate_description(reader.describe())
    assert description["unavailable"] == {}
    assert description["contact_force_threshold_n"] == 1.0
    sample = reader.read()
    assert sample["fields"]["applied_torque"] == [23.5, -23.5]
    assert sample["fields"]["computed_torque"] == [30.0, -30.0]
    assert sample["fields"]["contact_net_force_w"] == [[0.0, 0.0, 10.0], [0.0, 0.0, 40.0]]
    assert sample["frame_unavailable"] == {}
    assert contact.data_reads == 1
    assert base.scene["robot"].data.joint_acc_reads == 0
    assert reader.clock() == (0.01, 0)


def test_isaac_reader_skips_contact_when_any_env_outdated() -> None:
    contact = FakeContact(outdated=[False, True])
    reader = S.IsaacStateReader(_base(contact), env_index=0)
    sample = reader.read()
    names = [name for name, _, _, _ in S.CONTACT_FIELDS]
    assert sample["frame_unavailable"] == {name: "contact_sensor_outdated_read_skipped" for name in names}
    assert not set(names) & set(sample["fields"])
    assert contact.data_reads == 0


def test_isaac_reader_declares_missing_sensor_and_disabled_air_time() -> None:
    reader = S.IsaacStateReader(_base(None))
    described = reader.describe()
    for name, _, _, _ in S.CONTACT_FIELDS:
        assert described["unavailable"][name] == "scene_contact_sensor_missing"
    assert described["contact_body_names"] is None
    reader = S.IsaacStateReader(_base(FakeContact(track_air_time=False)))
    described = reader.describe()
    assert described["unavailable"] == {
        name: "contact_sensor_track_air_time_disabled" for name in S.AIR_TIME_FIELDS
    }
    assert "contact_net_force_w" in described["available"]


def test_isaac_reader_records_active_clip_function_without_judging(monkeypatch: pytest.MonkeyPatch) -> None:
    stock = S.IsaacStateReader(_base(FakeContact())).describe()["actuators"]["base_legs"]
    assert stock["clip_effort_function"].endswith("FakeDCMotor._clip_effort")
    assert stock["cfg"]["velocity_limit"] == 30.0 and stock["joint_names"] == JOINTS

    def patched(self: Any, effort: Any) -> Any:  # hidden_faults F2처럼 클래스 메서드를 교체한 경우
        return effort

    monkeypatch.setattr(FakeDCMotor, "_clip_effort", patched)
    replaced = S.IsaacStateReader(_base(FakeContact())).describe()["actuators"]["base_legs"]
    assert replaced["clip_effort_function"].endswith("patched")
    assert "behavior_label" not in replaced


def test_isaac_reader_declares_missing_articulation_attribute() -> None:
    base = _base(FakeContact())
    del base.scene["robot"].data.computed_torque
    reader = S.IsaacStateReader(base)
    assert reader.describe()["unavailable"] == {"computed_torque": "articulation_attribute_missing"}
    assert "computed_torque" not in reader.read()["fields"]


# --- 실제 capture 콜백 결합 -------------------------------------------------------


def _native_factory(log: list[Any]) -> Callable[..., Any]:
    def factory(base: Any, rep: Any, render_product: Any, profile: Any) -> Any:
        log.append(("factory", base, rep, render_product))

        def capture(capture_base: Any, capture_rep: Any, annotators: Any, camera_profile: Any,
                    eye: Any, lookat: Any, ordinal: int) -> dict[str, Any]:
            log.append(("capture", ordinal))
            return {"rgb": b"x", "ordinal": ordinal}

        def cleanup() -> None:
            log.append(("cleanup",))

        return capture, cleanup

    return factory


def _play_like_run_real(factory: Callable[..., Any], base: Any, frames: int) -> list[dict[str, Any]]:
    """vision_play_worker_v1.run_real의 순서: step → snapshot → capture → snapshot 재확인."""
    capture, cleanup = factory(base, "rep", "render_product", {"camera": "fixed"})
    rows: list[dict[str, Any]] = []
    try:
        for index in range(frames):
            if index:
                base.common_step_counter += 1
                base.sim.current_time += 0.02
                data = base.scene["robot"].data
                data.joint_pos = FakeTensor([[0.1 * index, 0.2], [9.0, 9.0]])
            before = (base.sim.current_time, base.common_step_counter)
            result = capture(base, "rep", [], {"camera": "fixed"}, [0.0] * 3, [0.0] * 3, index)
            after = (base.sim.current_time, base.common_step_counter)
            assert before == after and result["ordinal"] == index
            rows.append({
                "frame_index": index,
                "capture_ordinal": index,
                "simulation_step": after[1],
                "state": {"simulation_time_s": after[0]},
            })
    finally:
        cleanup()
    return rows


def _write_trace(path: Path, rows: list[dict[str, Any]], checkpoint: str = CK) -> Path:
    R.write_new(path, {
        "schema": R.TRACE_SCHEMA,
        "checkpoint_sha256": checkpoint,
        "video_sha256": "c" * 64,
        "frame_count": len(rows),
        "records": rows,
    })
    return path


def test_wrapped_factory_records_inside_native_callback_and_joins_trace(tmp_path: Path) -> None:
    log: list[Any] = []
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source(), expected_frames=4)
    base = _base(FakeContact())
    wrapped = S.wrap_capture_factory(_native_factory(log), recorder)
    rows = _play_like_run_real(wrapped, base, 4)
    assert [item for item in log if item[0] == "capture"] == [("capture", index) for index in range(4)]
    assert log[-1] == ("cleanup",)
    assert recorder.seal() is not None
    sidecar_rows = _records(tmp_path / "sidecar")
    assert [row["fields"]["joint_pos"][0] for row in sidecar_rows] == [0.1, 0.1, 0.2, 0.1 * 3]
    trace = _write_trace(tmp_path / "raw_trace.json", rows)
    native_manifest = tmp_path / "native_capture_manifest.json"
    R.write_new(native_manifest, {"checkpoint_sha256": CK, "frame_count": 4})
    joined = S.join_with_trace(tmp_path / "sidecar", trace, native_manifest)
    assert joined["joined"] is True and joined["frame_count"] == 4
    assert joined["behavior_label"] is None and joined["used_for_decision"] is False


def test_join_rejects_other_checkpoint_step_time_or_count(tmp_path: Path) -> None:
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source())
    rows = _play_like_run_real(S.wrap_capture_factory(_native_factory([]), recorder), _base(FakeContact()), 3)
    recorder.seal()
    with pytest.raises(S.SidecarError, match="checkpoint"):
        S.join_with_trace(tmp_path / "sidecar", _write_trace(tmp_path / "t1.json", rows, OTHER_CK))
    shifted = copy.deepcopy(rows)
    shifted[2]["simulation_step"] += 1
    with pytest.raises(S.SidecarError, match="frame 2"):
        S.join_with_trace(tmp_path / "sidecar", _write_trace(tmp_path / "t2.json", shifted))
    moved = copy.deepcopy(rows)
    moved[1]["state"]["simulation_time_s"] += 1e-9
    with pytest.raises(S.SidecarError, match="frame 1"):
        S.join_with_trace(tmp_path / "sidecar", _write_trace(tmp_path / "t3.json", moved))
    with pytest.raises(S.SidecarError, match="frame 수"):
        S.join_with_trace(tmp_path / "sidecar", _write_trace(tmp_path / "t4.json", rows[:2]))
    manifest = tmp_path / "m.json"
    R.write_new(manifest, {"checkpoint_sha256": OTHER_CK, "frame_count": 3})
    with pytest.raises(S.SidecarError, match="native capture manifest"):
        S.join_with_trace(tmp_path / "sidecar", _write_trace(tmp_path / "t5.json", rows), manifest)


def test_tampered_records_fail_sidecar_load(tmp_path: Path) -> None:
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    _drive(recorder, reader, 2)
    recorder.finalize()
    records = tmp_path / "sidecar" / S.RECORDS_NAME
    records.write_text(records.read_text(encoding="utf-8").replace("0.5", "0.6"), encoding="utf-8")
    with pytest.raises(S.SidecarError, match="해시"):
        S.load_sidecar(tmp_path / "sidecar")


def test_record_line_parser_rejects_nan_and_duplicate_keys() -> None:
    with pytest.raises(S.SidecarError):
        S._parse_line('{"a": NaN}')
    with pytest.raises(S.SidecarError):
        S._parse_line('{"a": 1, "a": 2}')


def test_isolate_policy_keeps_native_capture_running_and_seals_failure(tmp_path: Path) -> None:
    log: list[Any] = []
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source())
    contact = FakeContact()
    base = _base(contact)
    wrapped = S.wrap_capture_factory(_native_factory(log), recorder)
    capture, cleanup = wrapped(base, "rep", "rp", {})
    capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    base.scene["robot"].data.joint_vel = FakeTensor([[math.nan, 0.0]])
    base.common_step_counter, base.sim.current_time = 1, 0.03
    result = capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 1)
    base.common_step_counter, base.sim.current_time = 2, 0.05
    capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 2)
    cleanup()
    assert result == {"rgb": b"x", "ordinal": 1}
    assert [item for item in log if item[0] == "capture"] == [("capture", 0), ("capture", 1), ("capture", 2)]
    assert recorder.status == "failed"
    failure = R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)
    assert failure["stage"] == "record_frame" and failure["records_written"] == 1
    assert recorder.seal() is None


def test_abort_policy_stops_native_capture(tmp_path: Path) -> None:
    log: list[Any] = []
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source(), error_policy=S.ABORT)
    base = _base(FakeContact())
    capture, _cleanup = S.wrap_capture_factory(_native_factory(log), recorder)(base, "rep", "rp", {})
    base.scene["robot"].data.applied_torque = FakeTensor([[math.inf, 0.0]])
    with pytest.raises(S.SidecarError):
        capture(base, "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    assert (tmp_path / "sidecar" / S.FAILURE_NAME).exists()


def test_capture_on_other_env_object_is_rejected(tmp_path: Path) -> None:
    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source())
    base = _base(FakeContact())
    capture, _cleanup = S.wrap_capture_factory(_native_factory([]), recorder)(base, "rep", "rp", {})
    capture(_base(FakeContact()), "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    assert recorder.status == "failed" and recorder.failure_stage == "bind_env"


def test_reader_start_failure_is_isolated_and_sealed(tmp_path: Path) -> None:
    def broken(_base: Any, _env: int) -> Any:
        raise KeyError("robot")

    recorder = S.SidecarRecorder(tmp_path / "sidecar", _source())
    log: list[Any] = []
    capture, _cleanup = S.wrap_capture_factory(_native_factory(log), recorder, reader_factory=broken)(
        _base(None), "rep", "rp", {}
    )
    capture(None, "rep", [], {}, [0.0] * 3, [0.0] * 3, 0)
    assert ("capture", 0) in log
    failure = R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)
    assert failure["stage"] == "start" and failure["partial_records"] is None


# --- sdk_child worker 어댑터 --------------------------------------------------------


def _validated(tmp_path: Path, *, files_lf: dict[str, str] | None = None) -> dict[str, Any]:
    stage = tmp_path / "stage"
    return {
        "checkpoint_sha256": CK,
        "output_directory": stage / "output",
        "job_path": stage / "job.json",
        "job_sha256": "d" * 64,
        "job": {"source": {
            "commit": "e" * 40,
            "files_lf": {S.SOURCE_PATH: S.self_sha256_lf()} if files_lf is None else files_lf,
        }},
    }


def _fake_run_worker(base: Any, frames: int, events: list[Any], *, fail_at: int | None = None) -> Any:
    def run(validated: Any, app: Any, *, seal_report: Any, capture_factory: Any) -> dict[str, Any]:
        factory = capture_factory(validated, app)
        if fail_at is not None:
            raise RuntimeError("native failed")
        rows = _play_like_run_real(factory, base, frames)
        seal_report({"frames": len(rows)})
        events.append("native_done")
        return {"technical_recording_complete": True}

    return run


def _worker(tmp_path: Path, base: Any, events: list[Any], **kwargs: Any) -> Any:
    options: dict[str, Any] = {
        "source_id": "go2-f2-seed2033",
        "expected_frames": 3,
        "run_worker": _fake_run_worker(base, 3, events, fail_at=kwargs.pop("fail_at", None)),
        "capture_factory_maker": lambda job, app: _native_factory([]),
    }
    options.update(kwargs)
    return S.with_sidecar_worker(tmp_path / "sidecar", **options)


def test_worker_binds_job_identity_and_seals_before_return(tmp_path: Path) -> None:
    events: list[Any] = []
    worker = _worker(tmp_path, _base(FakeContact()), events, expected_checkpoint_sha256=CK)
    report = worker(_validated(tmp_path), object(), seal_report=lambda value: events.append(("report", value)))
    assert report == {"technical_recording_complete": True}
    manifest = R.read_object(tmp_path / "sidecar" / S.MANIFEST_NAME)
    assert manifest["source"]["job_sha256"] == "d" * 64
    assert manifest["source"]["source_commit"] == "e" * 40
    assert manifest["frame_count"] == 3
    assert ("report", {"frames": 3}) in events


def test_worker_rejects_other_checkpoint_before_native_run(tmp_path: Path) -> None:
    events: list[Any] = []
    worker = _worker(tmp_path, _base(FakeContact()), events, expected_checkpoint_sha256=OTHER_CK)
    with pytest.raises(S.SidecarError, match="checkpoint"):
        worker(_validated(tmp_path), object(), seal_report=lambda value: None)
    assert events == [] and not (tmp_path / "sidecar").exists()


def test_worker_requires_protocol_source_binding(tmp_path: Path) -> None:
    worker = _worker(tmp_path, _base(FakeContact()), [])
    with pytest.raises(S.SidecarError, match="source closure"):
        worker(_validated(tmp_path, files_lf={}), object(), seal_report=lambda value: None)
    with pytest.raises(S.SidecarError, match="source closure"):
        worker(_validated(tmp_path, files_lf={S.SOURCE_PATH: "0" * 64}), object(), seal_report=lambda value: None)


@pytest.mark.parametrize("inside", ["stage/output/sidecar", "stage/sidecar", "stage"])
def test_worker_rejects_sidecar_inside_native_stage(tmp_path: Path, inside: str) -> None:
    worker = S.with_sidecar_worker(
        tmp_path / inside, source_id="x", run_worker=_fake_run_worker(_base(None), 1, []),
        capture_factory_maker=lambda job, app: _native_factory([]),
    )
    with pytest.raises(ValueError, match="겹친다"):
        worker(_validated(tmp_path), object(), seal_report=lambda value: None)


def test_worker_seals_failure_when_native_record_raises(tmp_path: Path) -> None:
    worker = _worker(tmp_path, _base(FakeContact()), [], fail_at=0)
    with pytest.raises(RuntimeError, match="native failed"):
        worker(_validated(tmp_path), object(), seal_report=lambda value: None)
    failure = R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)
    assert failure["stage"] == "native_record" and failure["records_written"] == 0


def test_worker_records_not_started_sidecar(tmp_path: Path) -> None:
    def run(validated: Any, app: Any, *, seal_report: Any, capture_factory: Any) -> dict[str, Any]:
        return {"technical_recording_complete": True}

    worker = S.with_sidecar_worker(
        tmp_path / "sidecar", source_id="x", run_worker=run,
        capture_factory_maker=lambda job, app: _native_factory([]),
    )
    assert worker(_validated(tmp_path), object(), seal_report=lambda value: None)["technical_recording_complete"]
    assert R.read_object(tmp_path / "sidecar" / S.FAILURE_NAME)["stage"] == "not_started"


def test_worker_refuses_external_capture_factory(tmp_path: Path) -> None:
    worker = _worker(tmp_path, _base(FakeContact()), [])
    with pytest.raises(TypeError):
        worker(_validated(tmp_path), object(), seal_report=lambda value: None, capture_factory=object())


# --- child CLI: app.close 전 봉인 ----------------------------------------------------


def _environ(tmp_path: Path, **changes: str) -> dict[str, str]:
    values = {
        S.ENV_SIDECAR_DIR: str(tmp_path / "sidecar"),
        S.ENV_SOURCE_ID: "go2-f2-seed2033",
        S.ENV_EXPECTED_CHECKPOINT: CK,
        S.ENV_ERROR_POLICY: S.ISOLATE,
    }
    values.update(changes)
    return values


def _transferred(tmp_path: Path, events: list[Any], *, native_error: bool = False) -> Any:
    def sdk_child(job_path: Path, *, run_worker: Any) -> int:
        events.append(("sdk_child", job_path))
        try:
            run_worker(_validated(tmp_path), object(), seal_report=lambda value: None)
            return 0
        except RuntimeError:
            return 1
        finally:
            sidecar = tmp_path / "sidecar"
            events.append(("app.close", (sidecar / S.MANIFEST_NAME).exists(), (sidecar / S.FAILURE_NAME).exists()))

    def aaba_child(job_path: Path) -> int:
        events.append(("aaba", job_path))
        return 7

    return SimpleNamespace(sdk_child=sdk_child, aaba_child=aaba_child)


def test_child_main_seals_sidecar_before_app_close(tmp_path: Path) -> None:
    events: list[Any] = []
    code = S.child_main(
        "--sdk-child", tmp_path / "job.json", environ=_environ(tmp_path),
        transferred=_transferred(tmp_path, events),
        run_worker=_fake_run_worker(_base(FakeContact()), 3, []),
        capture_factory_maker=lambda job, app: _native_factory([]), expected_frames=3,
    )
    assert code == 0
    assert events[-1] == ("app.close", True, False)


def test_child_main_seals_failure_before_app_close(tmp_path: Path) -> None:
    events: list[Any] = []
    code = S.child_main(
        "--sdk-child", tmp_path / "job.json", environ=_environ(tmp_path),
        transferred=_transferred(tmp_path, events),
        run_worker=_fake_run_worker(_base(FakeContact()), 3, [], fail_at=0),
        capture_factory_maker=lambda job, app: _native_factory([]), expected_frames=3,
    )
    assert code == 1
    assert events[-1] == ("app.close", False, True)


def test_child_main_delegates_aaba_and_validates_env(tmp_path: Path) -> None:
    events: list[Any] = []
    assert S.child_main("--aaba-child", tmp_path / "job.json", transferred=_transferred(tmp_path, events)) == 7
    with pytest.raises(ValueError):
        S.child_main("--other", tmp_path / "job.json", transferred=_transferred(tmp_path, events))
    for broken in (
        {S.ENV_SIDECAR_DIR: "relative/path"},
        {S.ENV_SOURCE_ID: "../bad"},
        {S.ENV_EXPECTED_CHECKPOINT: "x"},
        {S.ENV_ERROR_POLICY: "ignore"},
    ):
        with pytest.raises(ValueError):
            S.sidecar_config_from_env(_environ(tmp_path, **broken))


# --- 부모 실행기 결합 ------------------------------------------------------------------


def test_child_launcher_rewrites_entry_and_injects_env() -> None:
    calls: list[Any] = []

    def popen(command: list[str], **kwargs: Any) -> str:
        calls.append((command, kwargs))
        return "process"

    launch = S.sidecar_child_launcher(popen, old_entry="/x/run5.py", sidecar_env={S.ENV_SOURCE_ID: "s"})
    assert launch(["py", "-B", "/x/run5.py", "--sdk-child"], env={"A": "1"}, cwd="/lab") == "process"
    command, kwargs = calls[0]
    assert command == ["py", "-B", str(Path(S.__file__).resolve()), "--sdk-child"]
    assert kwargs["env"] == {"A": "1", S.ENV_SOURCE_ID: "s"} and kwargs["cwd"] == "/lab"
    with pytest.raises(ValueError):
        launch(["py", "other.py"], env={})


def _job_file(tmp_path: Path, files_lf: dict[str, str] | None = None) -> Path:
    stage = tmp_path / "stage"
    job = stage / "job.json"
    R.write_new(job, {
        "source": {"files_lf": {S.SOURCE_PATH: S.self_sha256_lf()} if files_lf is None else files_lf},
        "output_directory": str(stage / "output"),
    })
    return job


def _fake_transferred_parent(calls: list[Any]) -> Any:
    run5 = "/repo/bench/run_vision_pull_record_v5.py"

    @contextmanager
    def job_contract() -> Iterator[None]:
        calls.append("enter")
        yield
        calls.append("exit")

    def run_job(job_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(("run_job", job_path, kwargs))
        kwargs["popen"](["py", "-B", str(Path(run5).resolve()), "--sdk-child"], env={})
        return {"completed": True}

    previous = SimpleNamespace(
        __file__=run5,
        run_job=run_job,
        OWNED=SimpleNamespace(_process_identity="identify", _safe_cleanup="cleanup"),
    )
    return SimpleNamespace(PREVIOUS=previous, _job_contract=job_contract)


def test_run_job_with_sidecar_reuses_parent_runner_with_rewritten_child(tmp_path: Path) -> None:
    calls: list[Any] = []
    launched: list[Any] = []
    job = _job_file(tmp_path)
    result = S.run_job_with_sidecar(
        job, sidecar_dir=tmp_path / "sidecar", source_id="go2-f2-seed2033",
        expected_checkpoint_sha256=CK, popen=lambda command, **kwargs: launched.append((command, kwargs)),
        transferred=_fake_transferred_parent(calls),
    )
    assert result == {"completed": True}
    assert calls[0] == "enter" and calls[-1] == "exit"
    _, _, kwargs = calls[1]
    assert kwargs["identify"] == "identify" and kwargs["cleanup"] == "cleanup"
    command, launch_kwargs = launched[0]
    assert command[2] == str(Path(S.__file__).resolve())
    assert launch_kwargs["env"][S.ENV_SIDECAR_DIR] == str((tmp_path / "sidecar").absolute())
    assert launch_kwargs["env"][S.ENV_EXPECTED_CHECKPOINT] == CK


def test_run_job_with_sidecar_refuses_unbound_source_or_overlapping_dir(tmp_path: Path) -> None:
    calls: list[Any] = []
    with pytest.raises(S.SidecarError, match="source closure"):
        S.run_job_with_sidecar(
            _job_file(tmp_path, files_lf={}), sidecar_dir=tmp_path / "sidecar", source_id="x",
            popen=lambda command, **kwargs: None, transferred=_fake_transferred_parent(calls),
        )
    job = tmp_path / "stage" / "job.json"
    job.unlink()
    with pytest.raises(ValueError, match="겹친다"):
        S.run_job_with_sidecar(
            _job_file(tmp_path), sidecar_dir=tmp_path / "stage" / "sidecar", source_id="x",
            popen=lambda command, **kwargs: None, transferred=_fake_transferred_parent(calls),
        )
    assert calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows에서 기본 Popen 실행 거부를 확인한다")
def test_run_job_with_sidecar_refuses_default_popen_on_windows(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Linux"):
        S.run_job_with_sidecar(
            _job_file(tmp_path), sidecar_dir=tmp_path / "sidecar", source_id="x",
            transferred=_fake_transferred_parent([]),
        )


# --- 1-frame dry-run ----------------------------------------------------------------


class FakeSession:
    def __init__(self, base: Any, events: list[Any], *, fail_advance: bool = False) -> None:
        self.base = base
        self.events = events
        self.fail_advance = fail_advance

    def advance(self) -> None:
        if self.fail_advance:
            raise RuntimeError("env.step failed")
        self.base.common_step_counter += 1
        self.base.sim.current_time += 0.02

    def close(self) -> None:
        self.events.append(("session.close", sorted(path.name for path in self._out.rglob("*.json"))))

    _out: Path


def _dry_run(tmp_path: Path, events: list[Any], *, frames: int = 1, fail_advance: bool = False,
             checkpoint_sha256: str | None = None) -> int:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"policy")
    out = tmp_path / "dry"
    session = FakeSession(_base(FakeContact()), events, fail_advance=fail_advance)
    session._out = out

    class App:
        def close(self) -> None:
            events.append(("app.close", sorted(path.name for path in out.rglob("*.json"))))

    def app_factory() -> Any:
        events.append("app")
        return App()

    return S.run_state_dry_run(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256 or R.sha256_file(checkpoint),
        source_id="go2-f2-dry",
        out_dir=out,
        frames=frames,
        app_factory=app_factory,
        session_factory=lambda path: session,
    )


def test_state_dry_run_seals_before_session_and_app_close(tmp_path: Path) -> None:
    events: list[Any] = []
    assert _dry_run(tmp_path, events) == 0
    sealed = ["dry_run_result.json", "sidecar_manifest.json"]
    assert events[1:] == [("session.close", sealed), ("app.close", sealed)]
    result = R.read_object(tmp_path / "dry" / "dry_run_result.json")
    assert result["frames"] == 1 and result["native_recording_touched"] is False
    assert result["first_frame_after_reset_before_policy_step"] is True


def test_state_dry_run_failure_is_sealed_before_close(tmp_path: Path) -> None:
    events: list[Any] = []
    assert _dry_run(tmp_path, events, frames=2, fail_advance=True) == 1
    sealed = ["dry_run_failure.json", "sidecar_failure.json"]
    assert events[1:] == [("session.close", sealed), ("app.close", sealed)]
    assert R.read_object(tmp_path / "dry" / "dry_run_failure.json")["records_written"] == 1


def test_state_dry_run_checks_checkpoint_before_starting_app(tmp_path: Path) -> None:
    events: list[Any] = []
    with pytest.raises(ValueError, match="해시"):
        _dry_run(tmp_path, events, checkpoint_sha256="0" * 64)
    assert events == []


# --- CLI ------------------------------------------------------------------------------


def test_plan_is_marked_not_executed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert S.main(["plan", "--out", str(tmp_path / "plan.json")]) == 0
    plan = R.read_object(tmp_path / "plan.json")
    assert plan["executed"] is False
    assert plan["status"] == "draft_not_run_until_registered_gpu_protocol"
    assert "--frames" in plan["dry_run"]["command"]
    assert json.loads(capsys.readouterr().out) == plan


def test_cli_dispatches_child_run_job_and_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(S, "child_main", lambda flag, job: seen.append((flag, job)) or 0)
    assert S.main(["--sdk-child", "--job", str(tmp_path / "job.json")]) == 0
    assert S.main(["--aaba-child", "--job", str(tmp_path / "job.json")]) == 0
    assert seen == [("--sdk-child", tmp_path / "job.json"), ("--aaba-child", tmp_path / "job.json")]
    monkeypatch.setattr(S, "run_job_with_sidecar", lambda job, **kwargs: {"completed": kwargs["source_id"] == "s"})
    assert S.main(["run-job", "--job", "j.json", "--sidecar-dir", "d", "--source-id", "s"]) == 0
    reader = SyntheticReader()
    recorder = _started(tmp_path, reader)
    _drive(recorder, reader, 2)
    recorder.finalize()
    rows = [
        {"frame_index": index, "capture_ordinal": index, "simulation_step": index,
         "state": {"simulation_time_s": 0.01 + 0.02 * index}}
        for index in range(2)
    ]
    trace = _write_trace(tmp_path / "trace.json", rows)
    assert S.main(["validate", "--sidecar", str(tmp_path / "sidecar"), "--trace", str(trace),
                   "--out", str(tmp_path / "join.json")]) == 0
    assert R.read_object(tmp_path / "join.json")["joined"] is True
