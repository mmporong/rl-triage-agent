"""같은 Play 실행의 로봇 관절·토크·접촉 상태를 native 녹화의 capture 콜백 안에서 sidecar로 남긴다.

- 시간축을 새로 만들지 않는다. ``vision_play_worker_v1.run_real``이 프레임마다 부르는
  ``camera_capture(base, rep, annotators, camera_profile, eye, lookat, ordinal)`` 안에서
  같은 ``base.sim.current_time``과 ``base.common_step_counter``를 읽고, trace와 1:1로 대조한다.
- 원본에 있는 필드만 기록한다. 없는 필드는 이유와 함께 manifest에 남기고 0으로 채우지 않는다.
- 행동 정답·임계값·모델 설명을 만들거나 쓰지 않는다. 고정 평가 라벨을 관절·행동 정답으로 쓰지 않는다.
- Isaac·Kit·torch는 child 함수 안에서만 import한다. CPU에서 이 모듈을 import해도 앱·GPU가 켜지지 않는다.
- 기존 캡처·물리·프레임 설정, V01 스키마, 기존 녹화 산출물 폴더는 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import record_vision_evidence_v1 as R

SCHEMA = "agentic_state_sidecar_v1"
MANIFEST_SCHEMA = "agentic_state_sidecar_manifest_v1"
RECORD_SCHEMA = "agentic_state_sidecar_record_v1"
FAILURE_SCHEMA = "agentic_state_sidecar_failure_v1"
JOIN_SCHEMA = "agentic_state_sidecar_join_v1"
PLAN_SCHEMA = "agentic_state_sidecar_plan_v1"
DRY_RUN_SCHEMA = "agentic_state_sidecar_dry_run_v1"
RECORDS_NAME = "records.jsonl"
MANIFEST_NAME = "sidecar_manifest.json"
FAILURE_NAME = "sidecar_failure.json"
TIME_BASIS = "isaac_sim_current_time_and_common_step_counter_read_inside_native_capture_callback"
READER_ID = "isaaclab_v2.1.1_articulation_and_contact_sensor_v1"
ENV_SIDECAR_DIR = "RL_TRIAGE_AGENTIC_SIDECAR_DIR"
ENV_SOURCE_ID = "RL_TRIAGE_AGENTIC_SIDECAR_SOURCE_ID"
ENV_EXPECTED_CHECKPOINT = "RL_TRIAGE_AGENTIC_SIDECAR_EXPECTED_CHECKPOINT_SHA256"
ENV_ERROR_POLICY = "RL_TRIAGE_AGENTIC_SIDECAR_ERROR_POLICY"
ISOLATE = "isolate_sidecar"
ABORT = "abort_recording"
ERROR_POLICIES = (ISOLATE, ABORT)
SOURCE_PATH = "bench/record_agentic_state_sidecar_v1.py"
TEST_PATH = "tests/test_agentic_state_sidecar_v1.py"
# 실제 수집 protocol의 source_sha256_lf에 함께 넣어야 하는 파일. 기존 v3 closure에는 없다.
SOURCE_FILES = (SOURCE_PATH, TEST_PATH)

# (record field, ArticulationData attribute, unit) — Isaac Lab v2.1.1 articulation_data.py
ARTICULATION_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("joint_pos", "joint_pos", "rad"),
    ("joint_vel", "joint_vel", "rad/s"),
    ("joint_pos_target", "joint_pos_target", "rad"),
    ("computed_torque", "computed_torque", "N*m"),
    ("applied_torque", "applied_torque", "N*m"),
)
# (record field, ContactSensorData attribute, unit, components per body) — contact_sensor_data.py
CONTACT_FIELDS: tuple[tuple[str, str, str, int], ...] = (
    ("contact_net_force_w", "net_forces_w", "N", 3),
    ("contact_current_contact_time", "current_contact_time", "s", 1),
    ("contact_current_air_time", "current_air_time", "s", 1),
)
AIR_TIME_FIELDS = frozenset({"contact_current_contact_time", "contact_current_air_time"})
NOT_RECORDED = {
    "joint_acc": "읽을 때 _previous_joint_vel을 갱신하는 유한차분 속성이라 보상 계산과 간섭할 수 있어 읽지 않는다",
}
# 센서가 아직 현재 시각으로 갱신되지 않았으면 data 접근이 lazy update와 history roll을 일으키므로 읽지 않는다.
FRAME_SKIP_REASONS = frozenset({"contact_sensor_outdated_read_skipped", "contact_sensor_outdated_flag_unknown"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MISSING = object()
T = TypeVar("T")


class SidecarError(RuntimeError):
    """sidecar 계약 위반. 기본 정책에서는 native 녹화와 분리해 sidecar만 실패로 봉인한다."""


class StateReader(Protocol):
    """같은 env에서 한 프레임의 상태를 읽는다. 실제 구현은 IsaacStateReader, 테스트는 synthetic reader."""

    def describe(self) -> dict[str, Any]: ...

    def clock(self) -> tuple[float, int]: ...

    def read(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    checkpoint_sha256: str
    task: str
    env_index: int = 0
    job_sha256: str | None = None
    source_commit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not _IDENTIFIER.fullmatch(self.source_id):
            raise ValueError("source_id 형식이 계약과 다르다")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256는 소문자 SHA-256이어야 한다")
        if not isinstance(self.task, str) or not _IDENTIFIER.fullmatch(self.task):
            raise ValueError("task 형식이 계약과 다르다")
        if isinstance(self.env_index, bool) or not isinstance(self.env_index, int) or self.env_index < 0:
            raise ValueError("env_index는 0 이상의 정수여야 한다")
        if self.job_sha256 is not None and not _is_sha256(self.job_sha256):
            raise ValueError("job_sha256는 소문자 SHA-256이어야 한다")
        if self.source_commit is not None and (
            not isinstance(self.source_commit, str) or not _COMMIT.fullmatch(self.source_commit)
        ):
            raise ValueError("source_commit은 40자리 소문자 git SHA여야 한다")

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "task": self.task,
            "env_index": self.env_index,
            "job_sha256": self.job_sha256,
            "source_commit": self.source_commit,
        }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _integer(value: Any, anchor: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SidecarError(f"{anchor}는 정수여야 한다")
    return int(value)


def _finite(value: Any, anchor: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise SidecarError(f"{anchor}는 유한 수여야 한다(NaN·inf·bool 거부)")
    return float(value)


def _clock(value: Any) -> tuple[float, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise SidecarError("clock은 (simulation_time_s, simulation_step) 쌍이어야 한다")
    time_s = _finite(value[0], "simulation_time_s")
    step = _integer(value[1], "simulation_step")
    if step < 0:
        raise SidecarError("simulation_step은 음수일 수 없다")
    return time_s, step


def _names(value: Any, anchor: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise SidecarError(f"{anchor}는 비어 있지 않은 문자열 목록이어야 한다")
    if len(set(value)) != len(value):
        raise SidecarError(f"{anchor}에 중복 이름이 있다")
    return list(value)


def _validate_description(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SidecarError("reader describe 결과가 mapping이 아니다")
    expected = {"reader", "joint_names", "contact_body_names", "available", "unavailable",
                "not_recorded", "contact_force_threshold_n", "actuators"}
    if set(value) != expected:
        raise SidecarError("reader describe 필드가 계약과 다르다")
    actuators = value["actuators"]
    if actuators is not None and (
        not isinstance(actuators, Mapping) or not all(isinstance(item, Mapping) for item in actuators.values())
    ):
        raise SidecarError("actuators는 None 또는 그룹별 mapping이어야 한다")
    joint_names = _names(value["joint_names"], "joint_names")
    body_names = value["contact_body_names"]
    if body_names is not None:
        body_names = _names(body_names, "contact_body_names")
    available = value["available"]
    unavailable = value["unavailable"]
    if not isinstance(available, Mapping) or not isinstance(unavailable, Mapping):
        raise SidecarError("available·unavailable은 mapping이어야 한다")
    known = {name for name, _, _ in ARTICULATION_FIELDS} | {name for name, _, _, _ in CONTACT_FIELDS}
    if set(available) & set(unavailable) or set(available) | set(unavailable) != known:
        raise SidecarError("모든 등록 필드가 available 또는 unavailable 중 정확히 한 곳에 있어야 한다")
    if not all(isinstance(reason, str) and reason for reason in unavailable.values()):
        raise SidecarError("unavailable 필드에는 이유 문자열이 있어야 한다")
    contact_available = [name for name, _, _, _ in CONTACT_FIELDS if name in available]
    if contact_available and body_names is None:
        raise SidecarError("접촉 필드가 있으면 contact_body_names가 필요하다")
    threshold = value["contact_force_threshold_n"]
    if threshold is not None:
        threshold = _finite(threshold, "contact_force_threshold_n")
    return {
        "reader": str(value["reader"]),
        "joint_names": joint_names,
        "contact_body_names": body_names,
        "available": {name: dict(spec) for name, spec in available.items()},
        "unavailable": dict(unavailable),
        "not_recorded": dict(value["not_recorded"]),
        "contact_force_threshold_n": threshold,
        "actuators": None if actuators is None else {str(key): dict(item) for key, item in actuators.items()},
    }


def _field_shape(name: str, description: Mapping[str, Any]) -> tuple[int, int]:
    for field, _, _ in ARTICULATION_FIELDS:
        if field == name:
            return len(description["joint_names"]), 1
    for field, _, _, components in CONTACT_FIELDS:
        if field == name:
            return len(description["contact_body_names"]), components
    raise SidecarError(f"등록되지 않은 필드 {name}")


def _values(value: Any, count: int, components: int, anchor: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != count:
        raise SidecarError(f"{anchor} 길이가 이름 목록({count})과 다르다")
    if components == 1:
        return [_finite(item, f"{anchor}[{index}]") for index, item in enumerate(value)]
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != components:
            raise SidecarError(f"{anchor}[{index}]는 길이 {components} 벡터여야 한다")
        rows.append([_finite(part, f"{anchor}[{index}][{axis}]") for axis, part in enumerate(item)])
    return rows


def _validate_sample(value: Any, description: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != {"fields", "frame_unavailable"}:
        raise SidecarError("reader read 결과 필드가 계약과 다르다")
    fields = value["fields"]
    skipped = value["frame_unavailable"]
    if not isinstance(fields, Mapping) or not isinstance(skipped, Mapping):
        raise SidecarError("fields·frame_unavailable은 mapping이어야 한다")
    available = set(description["available"])
    contact = {name for name, _, _, _ in CONTACT_FIELDS}
    if set(fields) & set(skipped) or set(fields) | set(skipped) != available:
        raise SidecarError("available 필드가 이 frame에서 사라졌거나 등록 밖 필드가 들어왔다")
    for name, reason in skipped.items():
        if name not in contact or reason not in FRAME_SKIP_REASONS:
            raise SidecarError(f"{name}의 frame 단위 미가용 이유가 등록 밖이다")
    clean: dict[str, Any] = {}
    for name in sorted(fields):
        count, components = _field_shape(name, description)
        clean[name] = _values(fields[name], count, components, name)
    return clean, dict(skipped)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def self_sha256_lf() -> str:
    return R.sha256_file(Path(__file__), lf=True)


def require_source_binding(files_lf: Any) -> str:
    """job source closure가 지금 실행되는 이 파일을 같은 LF SHA로 묶었는지 확인한다."""
    if not isinstance(files_lf, Mapping):
        raise SidecarError("job source files_lf가 mapping이 아니다")
    current = self_sha256_lf()
    if files_lf.get(SOURCE_PATH) != current:
        raise SidecarError(f"protocol source closure에 {SOURCE_PATH}가 현재 SHA로 묶여 있지 않다")
    return current


class SidecarRecorder:
    """한 Play 실행의 sidecar를 쓰기 1회 폴더에 기록한다. 상태: created → recording → sealed | failed."""

    def __init__(
        self,
        directory: Path,
        source: SourceIdentity,
        *,
        expected_frames: int | None = None,
        error_policy: str = ISOLATE,
    ) -> None:
        if not isinstance(source, SourceIdentity):
            raise TypeError("source는 SourceIdentity여야 한다")
        if error_policy not in ERROR_POLICIES:
            raise ValueError("error_policy가 등록 밖이다")
        if expected_frames is not None and (
            isinstance(expected_frames, bool) or not isinstance(expected_frames, int) or expected_frames < 1
        ):
            raise ValueError("expected_frames는 1 이상의 정수여야 한다")
        self.directory = Path(directory).absolute()
        self.source = source
        self.expected_frames = expected_frames
        self.error_policy = error_policy
        self.status = "created"
        self.error: BaseException | None = None
        self.failure_stage: str | None = None
        self._reader: StateReader | None = None
        self._description: dict[str, Any] | None = None
        self._handle: Any = None
        self._count = 0
        self._first: tuple[float, int] | None = None
        self._last: tuple[float, int] | None = None
        self._frame_skips: dict[str, dict[str, int]] = {}

    @property
    def records_written(self) -> int:
        return self._count

    def start(self, reader: StateReader) -> None:
        if self.status != "created":
            raise SidecarError("sidecar는 한 번만 시작한다")
        if self.directory.exists() or self.directory.is_symlink():
            raise FileExistsError(f"sidecar 출력이 이미 있다: {self.directory}")
        description = _validate_description(reader.describe())
        self.directory.mkdir(parents=True)
        self._handle = (self.directory / RECORDS_NAME).open("x", encoding="utf-8", newline="\n")
        self._reader = reader
        self._description = description
        self.status = "recording"

    def _active_reader(self) -> StateReader:
        if self.status != "recording" or self._reader is None:
            raise SidecarError(f"sidecar가 기록 상태가 아니다({self.status})")
        return self._reader

    def clock(self) -> tuple[float, int]:
        return _clock(self._active_reader().clock())

    def record(
        self, frame_index: int, *, capture_ordinal: int, clock_before: tuple[float, int] | None = None
    ) -> dict[str, Any]:
        reader = self._active_reader()
        description = self._description
        if description is None:
            raise SidecarError("reader 설명이 없다")
        index = _integer(frame_index, "frame_index")
        ordinal = _integer(capture_ordinal, "capture_ordinal")
        if index != self._count or ordinal != index:
            raise SidecarError("frame index·capture 순서가 sidecar 기록 순서와 다르다")
        now = _clock(reader.clock())
        if clock_before is not None and _clock(clock_before) != now:
            raise SidecarError("render 전후 simulation 시각·step이 달라졌다")
        if self._last is not None:
            if not now[0] > self._last[0]:
                raise SidecarError("simulation 시각이 앞 frame보다 커지지 않았다(시간 역전·정지)")
            if not now[1] > self._last[1]:
                raise SidecarError("simulation step이 앞 frame보다 커지지 않았다")
        fields, skipped = _validate_sample(reader.read(), description)
        row = {
            "schema": RECORD_SCHEMA,
            "frame_index": index,
            "capture_ordinal": ordinal,
            "simulation_step": now[1],
            "simulation_time_s": now[0],
            "fields": fields,
            "frame_unavailable": skipped,
        }
        line = R.canonical(row).decode("utf-8")
        self._handle.write(line + "\n")
        self._handle.flush()
        self._count += 1
        self._last = now
        if self._first is None:
            self._first = now
        for name, reason in skipped.items():
            counts = self._frame_skips.setdefault(name, {})
            counts[reason] = counts.get(reason, 0) + 1
        return row

    def around_capture(self, frame_index: int, capture: Callable[[], T]) -> T:
        """native capture를 감싸 같은 시각의 상태를 기록한다. 격리 정책에서는 sidecar 오류가 녹화를 멈추지 않는다."""
        before: tuple[float, int] | None = None
        if self.status == "recording":
            try:
                before = self.clock()
            except BaseException as error:  # noqa: BLE001 - 정책에 따라 격리하거나 녹화를 멈춘다.
                self.handle_error(error, stage="clock_before_capture")
        result = capture()
        if self.status == "recording":
            try:
                self.record(frame_index, capture_ordinal=frame_index, clock_before=before)
            except BaseException as error:  # noqa: BLE001 - 정책에 따라 격리하거나 녹화를 멈춘다.
                self.handle_error(error, stage="record_frame")
        return result

    def handle_error(self, error: BaseException, *, stage: str) -> None:
        self.fail(error, stage=stage)
        if self.error_policy == ABORT:
            raise error

    def _close_records(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None and not handle.closed:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    def finalize(self) -> dict[str, Any]:
        self._active_reader()
        description = self._description
        if description is None or self._first is None or self._last is None:
            raise SidecarError("기록된 frame이 없다")
        if self.expected_frames is not None and self._count != self.expected_frames:
            raise SidecarError(f"기록 frame 수 {self._count}가 기대값 {self.expected_frames}와 다르다")
        self._close_records()
        records = self.directory / RECORDS_NAME
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "sidecar_schema": SCHEMA,
            "status": "sealed",
            "source": self.source.to_json(),
            "sidecar_source": {"path": SOURCE_PATH, "sha256_lf": self_sha256_lf()},
            "reader": description["reader"],
            "time_basis": TIME_BASIS,
            "frame_count": self._count,
            "first": {"simulation_time_s": self._first[0], "simulation_step": self._first[1]},
            "last": {"simulation_time_s": self._last[0], "simulation_step": self._last[1]},
            "joint_names": description["joint_names"],
            "contact_body_names": description["contact_body_names"],
            "available_fields": description["available"],
            "unavailable_fields": description["unavailable"],
            "not_recorded_fields": description["not_recorded"],
            "frame_unavailable_counts": self._frame_skips,
            "contact_force_threshold_n_from_sensor_cfg": description["contact_force_threshold_n"],
            "actuators": description["actuators"],
            "records": {"file": RECORDS_NAME, "sha256": _sha256_path(records), "bytes": records.stat().st_size},
            "behavior_label": None,
            "thresholds_created": False,
            "fixed_evaluation_used_as_behavior_ground_truth": False,
            "model_text_used_as_ground_truth": False,
            "used_for_decision": False,
        }
        R.write_new(self.directory / MANIFEST_NAME, manifest)
        self.status = "sealed"
        return manifest

    def fail(self, error: BaseException, *, stage: str) -> None:
        """첫 실패만 봉인한다. 시작 전 실패는 폴더가 아직 없을 때만 새로 만들어 남긴다(기존 폴더에 쓰지 않음)."""
        if self.status in ("sealed", "failed"):
            return
        started = self.status == "recording"
        self.status = "failed"
        self.error = error
        self.failure_stage = stage
        if started:
            try:
                self._close_records()
            except BaseException as close_error:  # noqa: BLE001 - 실패 증거 저장을 이어간다.
                traceback.print_exception(close_error)
        else:
            try:
                self.directory.mkdir(parents=True)
            except OSError:
                return
        records = self.directory / RECORDS_NAME
        R.write_new(
            self.directory / FAILURE_NAME,
            {
                "schema": FAILURE_SCHEMA,
                "status": "failed",
                "stage": stage,
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error_message": str(error),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__),
                "source": self.source.to_json(),
                "records_written": self._count,
                "partial_records": (
                    {"file": RECORDS_NAME, "sha256": _sha256_path(records), "bytes": records.stat().st_size}
                    if records.exists() else None
                ),
                "behavior_label": None,
                "used_for_decision": False,
            },
        )

    def seal(self) -> dict[str, Any] | None:
        """녹화 뒤 호출한다. 성공이면 manifest, 이미 실패했거나 봉인에 실패하면 None을 돌려준다."""
        if self.status != "recording":
            return None
        try:
            return self.finalize()
        except BaseException as error:  # noqa: BLE001 - 정책에 따라 격리하거나 녹화를 실패로 만든다.
            self.handle_error(error, stage="finalize")
            return None


def _row(tensor: Any, env_index: int) -> Any:
    return tensor[env_index].detach().cpu().tolist()


def _config_value(value: Any) -> Any:
    if value is None or (isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)):
        return value
    return repr(value)


def _actuator_metadata(robot: Any) -> dict[str, Any] | None:
    """actuator 그룹의 클래스·_clip_effort 함수 출처·cfg 한계값. 녹화 시점의 토크 곡선 패치 여부를 남긴다(판정 없음)."""
    actuators = getattr(robot, "actuators", None)
    if not isinstance(actuators, Mapping):
        return None
    result: dict[str, Any] = {}
    for group, actuator in actuators.items():
        kind = type(actuator)
        clip = getattr(kind, "_clip_effort", None)
        cfg = getattr(actuator, "cfg", None)
        result[str(group)] = {
            "class": f"{kind.__module__}.{kind.__qualname__}",
            "clip_effort_function": (
                f"{getattr(clip, '__module__', '?')}.{getattr(clip, '__qualname__', '?')}" if clip is not None else None
            ),
            "joint_names": [str(name) for name in (getattr(actuator, "joint_names", None) or [])],
            "cfg": {
                key: _config_value(getattr(cfg, key, None))
                for key in ("effort_limit", "saturation_effort", "velocity_limit", "stiffness", "damping")
            },
        }
    return result


class IsaacStateReader:
    """Isaac Lab v2.1.1 ``env.unwrapped``에서 한 env의 관절·토크·접촉 상태를 읽는다.

    torch를 import하지 않고 텐서 메서드만 쓴다. ``joint_acc``는 읽지 않는다. 접촉 센서는 이미 현재
    시각으로 갱신된 경우에만 ``data``에 접근한다(``SensorBase._is_outdated``, v2.1.1에서 확인).
    """

    def __init__(
        self, base: Any, *, env_index: int = 0, robot_key: str = "robot", contact_key: str = "contact_forces"
    ) -> None:
        self._base = base
        self._env = env_index
        self._robot = base.scene[robot_key]
        self._contact: Any
        try:
            self._contact = base.scene[contact_key]
        except KeyError:
            self._contact = None
        self._description: dict[str, Any] | None = None

    def describe(self) -> dict[str, Any]:
        if self._description is not None:
            return dict(self._description)
        data = self._robot.data
        joint_names = getattr(data, "joint_names", None) or getattr(self._robot, "joint_names", None)
        available: dict[str, Any] = {}
        unavailable: dict[str, str] = {}
        for name, attribute, unit in ARTICULATION_FIELDS:
            value = getattr(data, attribute, _MISSING)
            if value is _MISSING:
                unavailable[name] = "articulation_attribute_missing"
            elif value is None:
                unavailable[name] = "articulation_attribute_none"
            else:
                available[name] = {"source": f"robot.data.{attribute}", "unit": unit, "per": "joint"}
        body_names: list[str] | None = None
        threshold: float | None = None
        if self._contact is None:
            for name, _, _, _ in CONTACT_FIELDS:
                unavailable[name] = "scene_contact_sensor_missing"
        else:
            cfg = self._contact.cfg
            body_names = [str(item) for item in self._contact.body_names]
            threshold = float(cfg.force_threshold)
            track_air_time = bool(getattr(cfg, "track_air_time", False))
            for name, attribute, unit, components in CONTACT_FIELDS:
                if name in AIR_TIME_FIELDS and not track_air_time:
                    unavailable[name] = "contact_sensor_track_air_time_disabled"
                else:
                    available[name] = {
                        "source": f"scene.contact_forces.data.{attribute}",
                        "unit": unit,
                        "per": "contact_body",
                        "components": components,
                    }
        self._description = {
            "reader": READER_ID,
            "joint_names": [str(item) for item in joint_names] if joint_names is not None else None,
            "contact_body_names": body_names,
            "available": available,
            "unavailable": unavailable,
            "not_recorded": dict(NOT_RECORDED),
            "contact_force_threshold_n": threshold,
            "actuators": _actuator_metadata(self._robot),
        }
        return dict(self._description)

    def clock(self) -> tuple[float, int]:
        return float(self._base.sim.current_time), int(self._base.common_step_counter)

    def read(self) -> dict[str, Any]:
        description = self.describe()
        available = description["available"]
        data = self._robot.data
        fields: dict[str, Any] = {}
        skipped: dict[str, str] = {}
        for name, attribute, _ in ARTICULATION_FIELDS:
            if name in available:
                fields[name] = _row(getattr(data, attribute), self._env)
        contact_names = [name for name, _, _, _ in CONTACT_FIELDS if name in available]
        if contact_names:
            outdated = getattr(self._contact, "_is_outdated", None)
            if outdated is None:
                skipped = {name: "contact_sensor_outdated_flag_unknown" for name in contact_names}
            elif bool(outdated.any().item()):  # data 접근은 outdated인 모든 env를 갱신한다
                skipped = {name: "contact_sensor_outdated_read_skipped" for name in contact_names}
            else:
                contact_data = self._contact.data  # 이미 갱신된 센서라 lazy update가 일어나지 않는다
                for name, attribute, _, _ in CONTACT_FIELDS:
                    if name in contact_names:
                        fields[name] = _row(getattr(contact_data, attribute), self._env)
        return {"fields": fields, "frame_unavailable": skipped}


def _default_reader_factory(base: Any, env_index: int) -> StateReader:
    return IsaacStateReader(base, env_index=env_index)


def wrap_capture_factory(
    factory: Callable[..., Any],
    recorder: SidecarRecorder,
    *,
    reader_factory: Callable[[Any, int], StateReader] = _default_reader_factory,
) -> Callable[..., tuple[Callable[..., Any], Callable[[], None]]]:
    """``PLAY.CameraCaptureFactory``와 같은 형태로 감싼다. capture 결과와 cleanup은 그대로 돌려준다."""

    def wrapped(base: Any, rep: Any, render_product: Any, profile: Mapping[str, Any]) -> tuple[Any, Any]:
        capture, cleanup = factory(base, rep, render_product, profile)
        if not callable(capture) or not callable(cleanup):
            raise TypeError("capture factory는 callback과 cleanup을 반환해야 한다")
        if recorder.status == "created":
            try:
                recorder.start(reader_factory(base, recorder.source.env_index))
            except BaseException as error:  # noqa: BLE001 - 정책에 따라 격리하거나 녹화를 멈춘다.
                recorder.handle_error(error, stage="start")

        def sidecar_capture(
            capture_base: Any,
            capture_rep: Any,
            annotators: list[Any],
            camera_profile: Mapping[str, Any],
            eye: list[float],
            lookat: list[float],
            ordinal: int,
        ) -> Any:
            if capture_base is not base and recorder.status == "recording":
                recorder.handle_error(SidecarError("capture base가 factory의 env와 다르다"), stage="bind_env")

            def native() -> Any:
                return capture(capture_base, capture_rep, annotators, camera_profile, eye, lookat, ordinal)

            return recorder.around_capture(ordinal, native)

        return sidecar_capture, cleanup

    return wrapped


def check_destination(sidecar_dir: Path, *protected: Path) -> Path:
    """sidecar 폴더는 새 경로이고 native job stage·출력과 서로 포함하지 않아야 한다(기존 산출물 검증과 비간섭)."""
    target = Path(sidecar_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"sidecar 출력이 이미 있다: {target}")
    resolved_target = target.resolve(strict=False)
    for folder in protected:
        resolved = Path(folder).absolute().resolve(strict=False)
        if (
            resolved_target == resolved
            or resolved in resolved_target.parents
            or resolved_target in resolved.parents
        ):
            raise ValueError(f"sidecar 폴더가 native 녹화 경로와 겹친다: {resolved}")
    return target


def with_sidecar_worker(
    sidecar_dir: Path,
    *,
    source_id: str,
    expected_checkpoint_sha256: str | None = None,
    error_policy: str = ISOLATE,
    expected_frames: int = R.FRAME_COUNT,
    run_worker: Callable[..., dict[str, Any]] | None = None,
    capture_factory_maker: Callable[[Mapping[str, Any], Any], Callable[..., Any]] | None = None,
    reader_factory: Callable[[Any, int], StateReader] = _default_reader_factory,
) -> Callable[..., dict[str, Any]]:
    """``sdk_child(job, run_worker=...)``에 넣는 worker. sidecar 봉인은 sdk_child가 app을 닫기 전에 끝난다."""
    if expected_checkpoint_sha256 is not None and not _is_sha256(expected_checkpoint_sha256):
        raise ValueError("expected_checkpoint_sha256는 소문자 SHA-256이어야 한다")

    def worker(validated: Mapping[str, Any], app: Any, **kwargs: Any) -> dict[str, Any]:
        if "capture_factory" in kwargs:
            raise TypeError("capture_factory는 sidecar worker만 결합한다(외부 factory 전달 거부)")
        checkpoint = validated["checkpoint_sha256"]
        if expected_checkpoint_sha256 is not None and checkpoint != expected_checkpoint_sha256:
            raise SidecarError("job checkpoint가 sidecar 등록 checkpoint와 다르다")
        job_source = validated["job"]["source"]
        require_source_binding(job_source["files_lf"])
        target = check_destination(
            sidecar_dir, Path(validated["output_directory"]), Path(validated["job_path"]).parent
        )
        source = SourceIdentity(
            source_id=source_id,
            checkpoint_sha256=checkpoint,
            task=R.TASK,
            env_index=0,
            job_sha256=validated.get("job_sha256"),
            source_commit=job_source.get("commit"),
        )
        recorder = SidecarRecorder(target, source, expected_frames=expected_frames, error_policy=error_policy)
        if run_worker is None or capture_factory_maker is None:
            worker_module = importlib.import_module("vision_pull_record_worker_v3")
            selected = run_worker or worker_module.run_policy_record
            maker = capture_factory_maker or worker_module.CAPTURE.make_capture_factory
        else:
            selected, maker = run_worker, capture_factory_maker

        def factory_maker(job: Mapping[str, Any], job_app: Any) -> Callable[..., Any]:
            if job is not validated:
                raise SidecarError("capture factory에 다른 job이 전달됐다")
            return wrap_capture_factory(maker(job, job_app), recorder, reader_factory=reader_factory)

        try:
            report = selected(validated, app, capture_factory=factory_maker, **kwargs)
        except BaseException as error:
            recorder.fail(error, stage="native_record")
            raise
        if recorder.status == "created":
            recorder.handle_error(SidecarError("capture factory가 호출되지 않아 sidecar가 시작되지 않았다"),
                                  stage="not_started")
        recorder.seal()
        if recorder.status == "failed" and recorder.error is not None:
            print(f"sidecar 실패 격리: {recorder.failure_stage}: {recorder.error}", file=sys.stderr)
        return report

    return worker


def sidecar_config_from_env(environ: Mapping[str, str]) -> dict[str, Any]:
    directory = environ.get(ENV_SIDECAR_DIR)
    source_id = environ.get(ENV_SOURCE_ID)
    if not directory or not Path(directory).is_absolute():
        raise ValueError(f"{ENV_SIDECAR_DIR}에 절대 경로가 필요하다")
    if not source_id or not _IDENTIFIER.fullmatch(source_id):
        raise ValueError(f"{ENV_SOURCE_ID} 형식이 계약과 다르다")
    expected = environ.get(ENV_EXPECTED_CHECKPOINT) or None
    if expected is not None and not _is_sha256(expected):
        raise ValueError(f"{ENV_EXPECTED_CHECKPOINT}는 소문자 SHA-256이어야 한다")
    policy = environ.get(ENV_ERROR_POLICY) or ISOLATE
    if policy not in ERROR_POLICIES:
        raise ValueError(f"{ENV_ERROR_POLICY}가 등록 밖이다")
    return {
        "sidecar_dir": Path(directory),
        "source_id": source_id,
        "expected_checkpoint_sha256": expected,
        "error_policy": policy,
    }


def child_main(
    flag: str,
    job_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    transferred: Any = None,
    **worker_options: Any,
) -> int:
    """부모 실행기가 띄우는 child. AABA 진단은 그대로 위임하고 본 녹화에만 sidecar를 붙인다."""
    module = transferred or importlib.import_module("run_vision_transferred_pull_record_v3")
    if flag == "--aaba-child":
        return module.aaba_child(job_path)
    if flag != "--sdk-child":
        raise ValueError("child flag가 등록 밖이다")
    config = sidecar_config_from_env(os.environ if environ is None else environ)
    worker = with_sidecar_worker(
        config["sidecar_dir"],
        source_id=config["source_id"],
        expected_checkpoint_sha256=config["expected_checkpoint_sha256"],
        error_policy=config["error_policy"],
        **worker_options,
    )
    return module.sdk_child(job_path, run_worker=worker)


def sidecar_child_launcher(
    popen: Callable[..., Any], *, old_entry: str, sidecar_env: Mapping[str, str]
) -> Callable[..., Any]:
    """RUN5 진입점을 이 모듈로 바꾸고 sidecar 설정을 환경변수로 넘긴다(transferred v3 launcher와 같은 방식)."""
    current = str(Path(__file__).resolve())

    def launch(command: Sequence[str], **kwargs: Any) -> Any:
        rewritten = [current if value == old_entry else value for value in command]
        if rewritten == list(command):
            raise ValueError("sidecar 자식 명령에 RUN5 진입점이 없다")
        base_env = kwargs.pop("env", None)
        env = dict(os.environ if base_env is None else base_env)
        env.update(sidecar_env)
        return popen(rewritten, env=env, **kwargs)

    return launch


def run_job_with_sidecar(
    job_path: Path,
    *,
    sidecar_dir: Path,
    source_id: str,
    expected_checkpoint_sha256: str | None = None,
    error_policy: str = ISOLATE,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    transferred: Any = None,
) -> dict[str, Any]:
    """transferred v3 부모 실행을 그대로 쓰고 child만 이 모듈로 바꾼다. GPU 등록·lease·입장은 호출자가 소유한다."""
    module = transferred or importlib.import_module("run_vision_transferred_pull_record_v3")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError("sidecar native 실행은 소유 Linux group에서 수행한다")
    job = R.read_object(Path(job_path))
    require_source_binding(job["source"]["files_lf"])
    target = check_destination(sidecar_dir, Path(job["output_directory"]), Path(job_path).absolute().parent)
    sidecar_env = {
        ENV_SIDECAR_DIR: str(target),
        ENV_SOURCE_ID: source_id,
        ENV_ERROR_POLICY: error_policy,
    }
    if expected_checkpoint_sha256 is not None:
        sidecar_env[ENV_EXPECTED_CHECKPOINT] = expected_checkpoint_sha256
    sidecar_config_from_env(sidecar_env)
    old_entry = str(Path(module.PREVIOUS.__file__).resolve())
    launcher = sidecar_child_launcher(popen, old_entry=old_entry, sidecar_env=sidecar_env)
    with module._job_contract():
        return module.PREVIOUS.run_job(
            job_path,
            popen=launcher,
            gpu_reader=gpu_reader,
            identify=module.PREVIOUS.OWNED._process_identity,
            cleanup=module.PREVIOUS.OWNED._safe_cleanup,
        )


def _parse_line(line: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SidecarError(f"비유한 JSON 수 {value}는 허용되지 않는다")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise SidecarError(f"중복 JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(line, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise SidecarError("sidecar record는 JSON object여야 한다")
    return value


def load_sidecar(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """봉인된 sidecar를 읽고 records 해시·순서·시각을 다시 검증한다."""
    root = Path(directory)
    manifest = R.read_object(root / MANIFEST_NAME)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "sealed":
        raise SidecarError("sidecar manifest가 봉인된 계약이 아니다")
    descriptor = manifest["records"]
    records_path = root / descriptor["file"]
    if descriptor["file"] != RECORDS_NAME or _sha256_path(records_path) != descriptor["sha256"]:
        raise SidecarError("sidecar records 해시가 manifest와 다르다")
    records = [
        _parse_line(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if len(records) != manifest["frame_count"]:
        raise SidecarError("sidecar record 수가 manifest와 다르다")
    previous: tuple[float, int] | None = None
    for index, row in enumerate(records):
        if row.get("schema") != RECORD_SCHEMA or row.get("frame_index") != index or row.get("capture_ordinal") != index:
            raise SidecarError(f"sidecar record {index}의 순서·스키마가 다르다")
        now = _clock((row.get("simulation_time_s"), row.get("simulation_step")))
        if previous is not None and not (now[0] > previous[0] and now[1] > previous[1]):
            raise SidecarError(f"sidecar record {index}의 시각·step이 증가하지 않는다")
        previous = now
    return manifest, records


def join_with_trace(
    sidecar_dir: Path, trace_path: Path, native_manifest_path: Path | None = None
) -> dict[str, Any]:
    """sidecar를 같은 실행의 raw trace와 frame·step·시각·checkpoint로 1:1 대조한다."""
    manifest, records = load_sidecar(sidecar_dir)
    trace = R.read_object(Path(trace_path))
    checkpoint = manifest["source"]["checkpoint_sha256"]
    if trace.get("schema") != R.TRACE_SCHEMA:
        raise SidecarError("trace 스키마가 native raw trace가 아니다")
    if trace.get("checkpoint_sha256") != checkpoint:
        raise SidecarError("sidecar와 trace의 checkpoint가 다르다")
    rows = trace.get("records")
    if not isinstance(rows, list) or trace.get("frame_count") != len(records) or len(rows) != len(records):
        raise SidecarError("sidecar와 trace의 frame 수가 다르다")
    for index, (sidecar, native) in enumerate(zip(records, rows, strict=True)):
        if not isinstance(native, Mapping) or not isinstance(native.get("state"), Mapping):
            raise SidecarError(f"trace frame {index}의 형식이 다르다")
        if (
            sidecar["frame_index"] != native.get("frame_index")
            or sidecar["capture_ordinal"] != native.get("capture_ordinal")
            or sidecar["simulation_step"] != native.get("simulation_step")
            or sidecar["simulation_time_s"] != native.get("state", {}).get("simulation_time_s")
        ):
            raise SidecarError(f"frame {index}의 index·step·simulation 시각이 trace와 다르다")
    native_digest = None
    if native_manifest_path is not None:
        native_manifest = R.read_object(Path(native_manifest_path))
        if native_manifest.get("checkpoint_sha256") != checkpoint or native_manifest.get("frame_count") != len(records):
            raise SidecarError("native capture manifest의 checkpoint·frame 수가 sidecar와 다르다")
        native_digest = _sha256_path(Path(native_manifest_path))
    return {
        "schema": JOIN_SCHEMA,
        "joined": True,
        "frame_count": len(records),
        "checkpoint_sha256": checkpoint,
        "video_sha256": trace.get("video_sha256"),
        "sidecar_manifest_sha256": _sha256_path(Path(sidecar_dir) / MANIFEST_NAME),
        "trace_sha256": _sha256_path(Path(trace_path)),
        "native_manifest_sha256": native_digest,
        "join_keys": ["frame_index", "capture_ordinal", "simulation_step", "simulation_time_s"],
        "behavior_label": None,
        "used_for_decision": False,
    }


class PlaySession(Protocol):
    base: Any

    def advance(self) -> None: ...

    def close(self) -> None: ...


class _IsaacPlaySession:
    def __init__(self, env: Any, policy: Any, command_buffer: Any, fixed_command: Any, torch: Any) -> None:
        self._env = env
        self._policy = policy
        self._command = command_buffer
        self._fixed = fixed_command
        self._torch = torch
        self.base = env.unwrapped
        self._obs, _ = env.get_observations()

    def advance(self) -> None:
        self._command.copy_(self._fixed)
        with self._torch.no_grad():
            actions = self._policy(self._obs)
        self._obs, _, _, _ = self._env.step(actions)
        self._command.copy_(self._fixed)

    def close(self) -> None:
        self._env.close()


def _open_play_session(checkpoint_path: Path) -> PlaySession:
    """dry-run 전용 Play 환경. 설정은 ``vision_play_worker_v1.run_real``의 env·command 줄과 같게 둔다(카메라 없음)."""
    import gymnasium as gym  # pyright: ignore[reportMissingImports]
    import isaaclab_tasks  # noqa: F401  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # pyright: ignore[reportMissingImports]
    from isaaclab_tasks.utils import (  # pyright: ignore[reportMissingImports]
        load_cfg_from_registry,
        parse_env_cfg,
    )
    from rsl_rl.runners import OnPolicyRunner  # pyright: ignore[reportMissingImports]

    profile = R.fixed_profile()
    env_cfg = parse_env_cfg(profile["task"], device="cuda:0", num_envs=1)
    env_cfg.seed = profile["seed"]
    env_cfg.actions.joint_pos.scale = profile["action_scale"]
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.resampling_time_range = tuple(profile["command"]["resampling_time_range_s"])
    command_cfg.ranges.lin_vel_x = (1.0, 1.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    if abs(float(env_cfg.sim.dt * env_cfg.decimation) - R.FRAME_PERIOD_S) > 1e-12:
        raise ValueError("Go2 Play control step이 고정 50fps time basis와 다르다")
    agent_cfg = load_cfg_from_registry(profile["task"], "rsl_rl_cfg_entry_point")
    agent_cfg.device = "cuda:0"
    env = RslRlVecEnvWrapper(gym.make(profile["task"], cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint_path), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped
        base.reset(seed=profile["seed"])
        command_buffer = base.command_manager.get_command("base_velocity")
        fixed_command = torch.tensor([[1.0, 0.0, 0.0]], device=base.device)
        command_buffer.copy_(fixed_command)
        return _IsaacPlaySession(env, policy, command_buffer, fixed_command, torch)
    except BaseException:
        env.close()
        raise


def _default_app_factory() -> Any:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    return AppLauncher(headless=True, enable_cameras=False, fast_shutdown=False).app


def run_state_dry_run(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_id: str,
    out_dir: Path,
    frames: int = 1,
    app_factory: Callable[[], Any] | None = None,
    session_factory: Callable[[Path], PlaySession] | None = None,
    reader_factory: Callable[[Any, int], StateReader] = _default_reader_factory,
) -> int:
    """GPU 등록 뒤 메인이 돌릴 최소 dry-run. 결과·실패 증거는 세션·앱 종료 전에 봉인한다."""
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= R.FRAME_COUNT:
        raise ValueError("frames는 1 이상 고정 frame 수 이하여야 한다")
    checkpoint = Path(checkpoint_path).absolute()
    if R.sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint 파일 해시가 등록값과 다르다")
    source = SourceIdentity(source_id=source_id, checkpoint_sha256=checkpoint_sha256, task=R.TASK)
    out = Path(out_dir).absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"dry-run 출력이 이미 있다: {out}")
    recorder = SidecarRecorder(out / "sidecar", source, expected_frames=frames, error_policy=ABORT)
    app: Any = None
    session: PlaySession | None = None
    code = 1
    try:
        app = (app_factory or _default_app_factory)()
        session = (session_factory or _open_play_session)(checkpoint)
        recorder.start(reader_factory(session.base, source.env_index))
        for index in range(frames):
            if index:
                session.advance()
            recorder.record(index, capture_ordinal=index)
        manifest = recorder.finalize()
        R.write_new(
            out / "dry_run_result.json",
            {
                "schema": DRY_RUN_SCHEMA,
                "completed": True,
                "frames": frames,
                "first_frame_after_reset_before_policy_step": True,
                "sidecar_manifest_sha256": _sha256_path(out / "sidecar" / MANIFEST_NAME),
                "available_fields": sorted(manifest["available_fields"]),
                "unavailable_fields": manifest["unavailable_fields"],
                "frame_unavailable_counts": manifest["frame_unavailable_counts"],
                "native_recording_touched": False,
                "used_for_decision": False,
            },
        )
        code = 0
    except BaseException as error:  # noqa: BLE001 - 세션·앱 종료 전에 첫 실패를 봉인한다.
        recorder.fail(error, stage="dry_run")
        out.mkdir(parents=True, exist_ok=True)
        failure = out / "dry_run_failure.json"
        if not failure.exists():
            R.write_new(
                failure,
                {
                    "schema": DRY_RUN_SCHEMA,
                    "completed": False,
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "error_message": str(error),
                    "traceback": traceback.format_exception(type(error), error, error.__traceback__),
                    "records_written": recorder.records_written,
                },
            )
        traceback.print_exc()
    finally:
        if session is not None:
            try:
                session.close()
            except BaseException:  # noqa: BLE001 - 봉인된 증거를 유지하고 종료를 이어간다.
                traceback.print_exc()
                code = 1
        if app is not None:
            try:
                app.close()
            except BaseException:  # noqa: BLE001 - SDK 종료 실패도 nonzero로 남긴다.
                traceback.print_exc()
                code = 1
    return code


def plan_commands() -> dict[str, Any]:
    """다음 단계 명령 초안. 새 GPU protocol·lease·입장·소스 결합·비용 gate 등록 전에는 실행하지 않는다."""
    script = "bench/record_agentic_state_sidecar_v1.py"
    return {
        "schema": PLAN_SCHEMA,
        "executed": False,
        "status": "draft_not_run_until_registered_gpu_protocol",
        "preconditions": [
            "새 GPU protocol·tag를 사전등록하고 커밋했다(source_sha256_lf에 SOURCE_FILES 두 파일 포함)",
            "등록 GPU의 lease·입장 검사를 통과했다(Use-GpuLease 또는 등록 클라우드 잠금)",
            "checkpoint 파일 SHA가 protocol 값과 같다",
            "실행 시간·비용 상한과 정지 조건을 고정했다",
            "Isaac Lab v2.1.1 설치본에서 실행한다(IsaacStateReader가 이 판의 속성을 가정)",
            "sidecar 폴더는 새 경로이며 native job output과 겹치지 않는다",
        ],
        "dry_run": {
            "purpose": "녹화 없이 1 frame 상태 읽기로 필드 가용성과 shape만 확인",
            "command": [
                "<isaaclab_python>", "-B", script, "state-dry-run",
                "--checkpoint", "<checkpoint.pt>", "--checkpoint-sha256", "<sha256>",
                "--source-id", "<source-id>", "--out", "<new-dry-run-dir>", "--frames", "1",
            ],
            "cwd": "<IsaacLab v2.1.1 root>",
            "env": {"OMNI_KIT_ACCEPT_EULA": "YES", "CUDA_VISIBLE_DEVICES": "0"},
        },
        "native_record": {
            "purpose": "기존 transferred v3 native 녹화에 같은 콜백 sidecar를 붙인 625 frame 수집",
            "command": [
                "<python>", "-B", script, "run-job", "--job", "<transferred-v3-job.json>",
                "--sidecar-dir", "<new-sidecar-dir>", "--source-id", "<source-id>",
                "--expected-checkpoint-sha256", "<sha256>", "--error-policy", ISOLATE,
            ],
            "host": "소유 Linux group(posix)에서만 실행된다",
        },
        "validate": {
            "command": [
                "<python>", "-B", script, "validate", "--sidecar", "<new-sidecar-dir>",
                "--trace", "<job-output>/evidence/raw_trace.json",
                "--native-manifest", "<job-output>/native_capture_manifest.json", "--out", "<join.json>",
            ],
        },
        "non_interference": "CPU 계약만 확인했다. 기존 녹화와의 비간섭은 실제 실행 전 실측으로 주장하지 않는다",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--sdk-child" in args or "--aaba-child" in args:
        child = argparse.ArgumentParser(description=__doc__)
        mode = child.add_mutually_exclusive_group(required=True)
        mode.add_argument("--sdk-child", action="store_true")
        mode.add_argument("--aaba-child", action="store_true")
        child.add_argument("--job", required=True, type=Path)
        parsed = child.parse_args(args)
        return child_main("--sdk-child" if parsed.sdk_child else "--aaba-child", parsed.job)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-job")
    run.add_argument("--job", required=True, type=Path)
    run.add_argument("--sidecar-dir", required=True, type=Path)
    run.add_argument("--source-id", required=True)
    run.add_argument("--expected-checkpoint-sha256")
    run.add_argument("--error-policy", choices=ERROR_POLICIES, default=ISOLATE)
    dry = commands.add_parser("state-dry-run")
    dry.add_argument("--checkpoint", required=True, type=Path)
    dry.add_argument("--checkpoint-sha256", required=True)
    dry.add_argument("--source-id", required=True)
    dry.add_argument("--out", required=True, type=Path)
    dry.add_argument("--frames", type=int, default=1)
    validate = commands.add_parser("validate")
    validate.add_argument("--sidecar", required=True, type=Path)
    validate.add_argument("--trace", required=True, type=Path)
    validate.add_argument("--native-manifest", type=Path)
    validate.add_argument("--out", type=Path)
    plan = commands.add_parser("plan")
    plan.add_argument("--out", type=Path)
    parsed = parser.parse_args(args)
    if parsed.command == "run-job":
        result = run_job_with_sidecar(
            parsed.job,
            sidecar_dir=parsed.sidecar_dir,
            source_id=parsed.source_id,
            expected_checkpoint_sha256=parsed.expected_checkpoint_sha256,
            error_policy=parsed.error_policy,
        )
        return 0 if result.get("completed") is True else 1
    if parsed.command == "state-dry-run":
        return run_state_dry_run(
            checkpoint_path=parsed.checkpoint,
            checkpoint_sha256=parsed.checkpoint_sha256,
            source_id=parsed.source_id,
            out_dir=parsed.out,
            frames=parsed.frames,
        )
    value = (
        join_with_trace(parsed.sidecar, parsed.trace, parsed.native_manifest)
        if parsed.command == "validate"
        else plan_commands()
    )
    if parsed.out is not None:
        R.write_new(parsed.out, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
