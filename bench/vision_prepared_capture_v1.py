"""한 Writer attach 구간의 prepare와 네 smoke callback을 detach 뒤 결합한다."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import vision_frame_sync_v1 as FRAME
import vision_writer_payload_v1 as PAYLOAD

SCHEMA = "vision_prepared_capture_v1"
PREPARE_MARKER_CODE = 3
ARM_COUNT = 4
PREPARE_LOGICAL_ORDINAL = 4
MAX_RAW_CALLBACKS = 16
MAX_SNAPSHOT_TOTAL_ARRAY_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOT_NODES = 100_000
MAX_SNAPSHOT_ENTRIES = 100_000
MAX_SNAPSHOT_DEPTH = 64


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


class _SnapshotBudget:
    def __init__(self) -> None:
        self.array_bytes = 0
        self.nodes = 0
        self.entries = 0

    def node(self, depth: int) -> None:
        if depth > MAX_SNAPSHOT_DEPTH or self.nodes >= MAX_SNAPSHOT_NODES:
            raise ValueError("prepared capture callback node/depth 상한을 넘었다")
        self.nodes += 1

    def reserve_entries(self, count: int) -> None:
        if count < 0 or count > MAX_SNAPSHOT_ENTRIES - self.entries:
            raise ValueError("prepared capture callback entry 상한을 넘었다")
        self.entries += count

    def reserve_array(self, count: int) -> None:
        if count < 0 or count > MAX_SNAPSHOT_TOTAL_ARRAY_BYTES - self.array_bytes:
            raise ValueError("prepared capture callback array byte 상한을 넘었다")
        self.array_bytes += count


def _snapshot(value: Any, budget: _SnapshotBudget | None = None, depth: int = 0) -> Any:
    selected = budget if budget is not None else _SnapshotBudget()
    selected.node(depth)
    if isinstance(value, np.ndarray):
        selected.reserve_array(int(value.nbytes))
        return value.copy(order="K")
    if isinstance(value, Mapping):
        selected.reserve_entries(len(value))
        return {
            key: _snapshot(item, selected, depth + 1) for key, item in value.items()
        }
    if isinstance(value, list):
        selected.reserve_entries(len(value))
        return [_snapshot(item, selected, depth + 1) for item in value]
    if isinstance(value, tuple):
        selected.reserve_entries(len(value))
        return tuple(_snapshot(item, selected, depth + 1) for item in value)
    return deepcopy(value)


def _physics(value: Any, simulation_time_s: Any) -> tuple[dict[str, Any], float, str]:
    if not isinstance(value, Mapping):
        raise TypeError("prepared capture physics snapshot은 mapping이어야 한다")
    if isinstance(simulation_time_s, bool) or not isinstance(
        simulation_time_s, (int, float)
    ):
        raise TypeError("prepared capture simulation time은 수치여야 한다")
    time = float(simulation_time_s)
    if not math.isfinite(time):
        raise ValueError("prepared capture simulation time은 finite여야 한다")
    copied = _snapshot(dict(value))
    raw = _canonical(copied)
    return copied, time, hashlib.sha256(raw).hexdigest()


def _single_row(value: Any) -> tuple[str, Any] | None:
    if not isinstance(value, Mapping) or len(value) != 1:
        return None
    key, item = next(iter(value.items()))
    if not isinstance(key, str) or not key:
        return None
    return key, item


def _bootstrap_candidate(payload: Any, budget: _SnapshotBudget) -> Any | None:
    if not isinstance(payload, Mapping):
        return None
    annotators = payload.get("annotators")
    if not isinstance(annotators, Mapping):
        return None
    semantic_outer = annotators.get("semantic_segmentation")
    semantic_row = _single_row(semantic_outer)
    resolution_row = _single_row(annotators.get("resolution"))
    rgb_row = _single_row(annotators.get("rgb"))
    camera_params_row = _single_row(annotators.get("CameraParams"))
    if (
        semantic_row is None
        or resolution_row is None
        or rgb_row is None
        or camera_params_row is None
        or semantic_row[0] != resolution_row[0]
        or semantic_row[0] != rgb_row[0]
        or semantic_row[0] != camera_params_row[0]
    ):
        return None
    semantic = semantic_row[1]
    empty = semantic.get("data") if isinstance(semantic, Mapping) else None
    if (
        not isinstance(semantic_outer, Mapping)
        or not isinstance(semantic, Mapping)
        or set(semantic) != {"data", "idToLabels"}
        or not isinstance(empty, np.ndarray)
        or empty.dtype != np.float64
        or empty.shape != (0,)
    ):
        return None
    resolution = resolution_row[1]
    if (
        not isinstance(resolution, (list, tuple, np.ndarray))
        or len(resolution) != 2
        or any(
            isinstance(item, (bool, np.bool_))
            or not isinstance(item, (int, np.integer))
            or int(item) <= 0
            for item in resolution
        )
    ):
        return None
    width, height = int(resolution[0]), int(resolution[1])
    rgb_payload = rgb_row[1]
    rgba = rgb_payload.get("data") if isinstance(rgb_payload, Mapping) else None
    camera_params = camera_params_row[1]
    camera_resolution = (
        camera_params.get("renderProductResolution")
        if isinstance(camera_params, Mapping)
        else None
    )
    if (
        not isinstance(rgb_payload, Mapping)
        or set(rgb_payload) != {"data"}
        or not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.shape != (height, width, 4)
        or not isinstance(camera_resolution, (list, tuple, np.ndarray))
        or len(camera_resolution) != 2
        or list(camera_resolution) != [width, height]
    ):
        return None
    dummy_bytes = height * width * np.dtype(np.uint32).itemsize
    budget.reserve_array(dummy_bytes)
    candidate: dict[Any, Any] = dict(payload)
    candidate_annotators: dict[Any, Any] = dict(annotators)
    candidate_semantics: dict[Any, Any] = dict(semantic_outer)
    candidate_semantic: dict[Any, Any] = dict(semantic)
    candidate_semantic["data"] = np.zeros((height, width), dtype=np.uint32)
    candidate_semantics[semantic_row[0]] = candidate_semantic
    candidate_annotators["semantic_segmentation"] = candidate_semantics
    candidate["annotators"] = candidate_annotators
    return candidate


def _classify(
    payload: Any, budget: _SnapshotBudget
) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        decoded = PAYLOAD.decode_writer_payload(payload)
    except Exception as strict_error:  # noqa: BLE001 - payload 오류 종류를 봉인한다.
        candidate = _bootstrap_candidate(payload, budget)
        if candidate is None:
            return "invalid", None, _type_name(strict_error)
        try:
            PAYLOAD.decode_writer_payload(candidate)
        except Exception as remaining_error:  # noqa: BLE001 - 반사실 schema 검사다.
            return "invalid", None, _type_name(remaining_error)
        return "bootstrap_empty", None, _type_name(strict_error)
    return "valid", decoded, None


def _frame_payload(decoded: Mapping[str, Any]) -> dict[str, Any]:
    callback_ref = decoded["callback_ref"]
    values = {
        "rgb": decoded["rgb"],
        "semantic": decoded["semantic"],
        "camera": decoded["camera"],
        "read_times": decoded["read_times"],
        "reference_time": decoded["reference_time"],
    }
    return {
        "adapter_callback_ref": callback_ref,
        "render_product_ref": decoded["render_product_ref"],
        **{
            name: {"adapter_callback_ref": callback_ref, "data": item}
            for name, item in values.items()
        },
    }


def _prepare_as_code_zero(decoded: Mapping[str, Any]) -> dict[str, Any]:
    """code 3 label을 role 보존 code 0 label로 바꿔 AtomicFrameWriter에 검증시킨다."""
    copied = _snapshot(decoded)
    labels = copied["semantic"]["idToLabels"]
    changed: dict[str, Any] = {}
    for runtime_id, raw_label in labels.items():
        label = raw_label.get("class") if isinstance(raw_label, Mapping) else raw_label
        if not isinstance(label, str) or not label.startswith(FRAME.MARKER_PREFIX):
            changed[runtime_id] = raw_label
            continue
        suffix = label.removeprefix(FRAME.MARKER_PREFIX)
        if suffix == "sentinel":
            changed[runtime_id] = raw_label
            continue
        parts = suffix.split("/")
        if (
            len(parts) != 3
            or parts[0] not in {"bit0", "bit1"}
            or parts[1] not in {"rail0", "rail1"}
            or parts[2] not in {"active", "inactive"}
        ):
            changed[runtime_id] = raw_label
            continue
        swapped = 1 - int(parts[1][-1])
        mapped = f"{FRAME.MARKER_PREFIX}{parts[0]}/rail{swapped}/{parts[2]}"
        if isinstance(raw_label, Mapping):
            mapped_label = dict(raw_label)
            mapped_label["class"] = mapped
            changed[runtime_id] = mapped_label
        else:
            changed[runtime_id] = mapped
    copied["semantic"]["idToLabels"] = changed
    return copied


class PreparedCaptureMux:
    """한 attached Writer의 PREPARE→ARM0..3→DETACH→RESOLVE→SEAL 상태를 소유한다.

    callback 함수는 ``observe`` 하나만 사용한다. 배열은 호출 안에서 복사된다.
    실패 뒤에는 새 prepare/arm 단계가 열리지 않으며 detach와 실패 seal만 허용된다.
    """

    def __init__(
        self,
        *,
        render_product_ref: str,
        camera_path: str,
        resolution: Sequence[int],
    ) -> None:
        if not isinstance(render_product_ref, str) or not render_product_ref:
            raise ValueError("prepared capture RenderProduct reference가 없다")
        if not isinstance(camera_path, str) or not camera_path:
            raise ValueError("prepared capture camera path가 없다")
        if (
            isinstance(resolution, (str, bytes))
            or len(resolution) != 2
            or any(type(item) is not int or item <= 0 for item in resolution)
        ):
            raise ValueError("prepared capture resolution이 잘못됐다")
        self.render_product_ref = render_product_ref
        self.camera_path = camera_path
        self.resolution = list(resolution)
        self.state = "new"
        self.raw_callback_count = 0
        self.callbacks: list[dict[str, Any]] = []
        self._phase: dict[str, Any] | None = None
        self._prepare: dict[str, Any] | None = None
        self._arms: list[dict[str, Any]] = []
        self._failure: dict[str, Any] | None = None
        self._bundles: list[dict[str, Any]] = []
        self._sealed: dict[str, Any] | None = None

    def _fail(self, code: str, *, error: BaseException | None = None) -> None:
        if self._failure is None:
            self._failure = {
                "code": code,
                "state": self.state,
                "raw_callback_index": (
                    self.raw_callback_count - 1 if self.raw_callback_count else None
                ),
                "error_type": _type_name(error) if error is not None else None,
            }

    def _guard_capture(self) -> None:
        if self._failure is not None:
            raise RuntimeError("prepared capture는 실패 뒤 capture를 재개할 수 없다")

    def _begin_phase(
        self,
        kind: str,
        ordinal: int | None,
        marker_code: int,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        try:
            before, time, state_sha = _physics(state_before, simulation_time_s)
        except Exception as error:
            self._fail("physics_before_invalid", error=error)
            raise
        self._phase = {
            "kind": kind,
            "ordinal": ordinal,
            "marker_code": marker_code,
            "before": before,
            "time": time,
            "state_sha256": state_sha,
            "callbacks": [],
        }

    def begin_prepare(
        self,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
        *,
        marker_code: int,
    ) -> None:
        self._guard_capture()
        if self.state != "new" or marker_code != PREPARE_MARKER_CODE:
            self._fail("prepare_begin_invalid")
            raise ValueError("prepare는 new 상태와 marker code 3이 필요하다")
        self.state = "preparing"
        self._begin_phase("prepare", None, marker_code, state_before, simulation_time_s)

    def begin_arm(
        self,
        ordinal: int,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
        *,
        marker_code: int,
    ) -> None:
        self._guard_capture()
        if type(ordinal) is not int:
            self._fail("arm_begin_invalid")
            raise ValueError("arm ordinal은 0..3 정수여야 한다")
        expected_state = "prepared" if ordinal == 0 else f"arm{ordinal - 1}_done"
        if (
            ordinal != len(self._arms)
            or ordinal not in range(ARM_COUNT)
            or marker_code != ordinal
            or self.state != expected_state
        ):
            self._fail("arm_begin_invalid")
            raise ValueError("arm은 0..3 순서와 같은 marker code가 필요하다")
        self.state = f"arming{ordinal}"
        self._begin_phase("arm", ordinal, marker_code, state_before, simulation_time_s)

    def source_error(self, error: Exception) -> None:
        if not isinstance(error, Exception):
            raise TypeError("prepared capture source error는 Exception이어야 한다")
        self._fail("source_error_before_callback", error=error)

    def observe(self, payload: Any) -> None:
        raw_index = self.raw_callback_count
        self.raw_callback_count += 1
        if raw_index >= MAX_RAW_CALLBACKS:
            self._fail("raw_callback_overflow")
            return
        window = self.state
        row: dict[str, Any] = {
            "raw_index": raw_index,
            "window": window,
            "classification": "invalid",
            "error_type": None,
            "descriptor": None,
        }
        self.callbacks.append(row)
        try:
            budget = _SnapshotBudget()
            frozen = _snapshot(payload, budget)
            row["descriptor"] = PAYLOAD.describe_writer_payload(frozen)
            classification, decoded, error_type = _classify(frozen, budget)
        except Exception as error:  # noqa: BLE001 - callback 실패를 sticky evidence로 남긴다.
            row["error_type"] = _type_name(error)
            self._fail("callback_snapshot_or_classification_failed", error=error)
            return
        row["classification"] = classification
        row["error_type"] = error_type
        row["frame_number"] = (
            decoded["read_times"]["frameNumber"] if decoded is not None else None
        )
        if self.state.startswith("arming") or self.state == "preparing":
            if self._phase is None:
                self._fail("callback_without_phase")
                return
            self._phase["callbacks"].append(
                {
                    "raw_index": raw_index,
                    "classification": classification,
                    "decoded": decoded,
                }
            )
            if self.state.startswith("arming") and classification != "valid":
                self._fail("arm_callback_invalid")
            if self.state == "preparing":
                classes = [item["classification"] for item in self._phase["callbacks"]]
                if classes not in (
                    ["valid"],
                    ["bootstrap_empty"],
                    ["bootstrap_empty", "valid"],
                ):
                    self._fail("prepare_callback_sequence_invalid")
            if len(self._phase["callbacks"]) > (2 if self.state == "preparing" else 1):
                self._fail("callback_cardinality_invalid")
            return
        self._fail(
            "callback_during_detach"
            if self.state == "detaching"
            else "unsolicited_callback"
        )

    def _end_phase(
        self, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> dict[str, Any]:
        phase = self._phase
        if phase is None:
            self._fail("phase_missing")
            raise ValueError("prepared capture phase가 없다")
        try:
            _, time, state_sha = _physics(state_after, simulation_time_s)
        except Exception as error:
            self._fail("physics_after_invalid", error=error)
            raise
        if state_sha != phase["state_sha256"] or time != phase["time"]:
            self._fail("physics_or_time_changed")
        return phase

    def end_prepare(
        self, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> None:
        if self.state != "preparing":
            self._fail("prepare_end_invalid")
            raise ValueError("prepare가 시작되지 않았다")
        phase = self._end_phase(state_after, simulation_time_s)
        classes = [item["classification"] for item in phase["callbacks"]]
        if classes not in (["valid"], ["bootstrap_empty", "valid"]):
            self._fail("prepare_callback_sequence_invalid")
        if self._failure is not None:
            raise RuntimeError("prepare callback 계약이 실패했다")
        valid = phase["callbacks"][-1]
        self._prepare = {
            **phase,
            "decoded": valid["decoded"],
            "bootstrap_empty_observed": classes == ["bootstrap_empty", "valid"],
            "valid_raw_index": valid["raw_index"],
        }
        self._phase = None
        self.state = "prepared"

    def end_arm(
        self,
        ordinal: int,
        state_after: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        if self.state != f"arming{ordinal}" or self._phase is None:
            self._fail("arm_end_invalid")
            raise ValueError("해당 arm이 시작되지 않았다")
        phase = self._end_phase(state_after, simulation_time_s)
        callbacks = phase["callbacks"]
        if len(callbacks) != 1 or callbacks[0]["classification"] != "valid":
            self._fail("arm_callback_cardinality_invalid")
        if self._failure is not None:
            raise RuntimeError("arm callback 계약이 실패했다")
        self._arms.append({**phase, "decoded": callbacks[0]["decoded"]})
        self._phase = None
        self.state = f"arm{ordinal}_done"

    def begin_detach(self) -> None:
        if self.state == "detaching" or self.state == "detached":
            self._fail("detach_begin_invalid")
            raise ValueError("detach가 이미 시작됐다")
        if self._failure is None and self.state != "arm3_done":
            self._fail("detach_before_arms_complete")
            raise ValueError("네 arm이 끝나기 전에 detach할 수 없다")
        self.state = "detaching"

    def end_detach(self) -> None:
        if self.state != "detaching":
            self._fail("detach_end_invalid")
            raise ValueError("detach가 시작되지 않았다")
        self.state = "detached"

    def _validate_profile_and_progression(self) -> None:
        if self._prepare is None or len(self._arms) != ARM_COUNT:
            raise ValueError("prepare와 네 arm이 모두 필요하다")
        decoded_rows = [self._prepare["decoded"]] + [
            row["decoded"] for row in self._arms
        ]
        if any(
            row["render_product_ref"] != self.render_product_ref for row in decoded_rows
        ):
            raise ValueError("prepared capture RenderProduct가 등록값과 다르다")
        if any(row["camera_path"] != self.camera_path for row in decoded_rows):
            raise ValueError("prepared capture camera path가 등록값과 다르다")
        if any(
            row["render_product_resolution"] != self.resolution for row in decoded_rows
        ):
            raise ValueError("prepared capture resolution이 등록값과 다르다")
        projections = [row["camera"]["projection"] for row in decoded_rows]
        if any(value != projections[0] for value in projections[1:]):
            raise ValueError("prepared capture projection이 바뀌었다")
        frames = [row["read_times"]["frameNumber"] for row in decoded_rows]
        if frames != list(range(frames[0], frames[0] + ARM_COUNT + 1)):
            raise ValueError("prepare frame 뒤 arm frame이 하나씩 진행하지 않았다")
        phase_rows = [self._prepare] + self._arms
        if any(
            (row["state_sha256"], row["time"])
            != (phase_rows[0]["state_sha256"], phase_rows[0]["time"])
            for row in phase_rows[1:]
        ):
            raise ValueError("prepare와 arm의 physics state/time이 다르다")

    def _validate_prepare_marker(self) -> dict[str, Any]:
        assert self._prepare is not None
        decoded = _prepare_as_code_zero(self._prepare["decoded"])
        writer = FRAME.AtomicFrameWriter(
            FRAME.SMOKE_MODE, ARM_COUNT, render_product_ref=self.render_product_ref
        )
        writer.arm(0, 0, self._prepare["before"], self._prepare["time"])
        writer.before_step(0)
        writer.callback(_frame_payload(decoded))
        return writer.close(0, self._prepare["before"], self._prepare["time"])

    @staticmethod
    def _validate_prepare_to_arm_timing(
        preparation: Mapping[str, Any], arm_zero: Mapping[str, Any]
    ) -> None:
        prepare_fraction = preparation.get("timing_fraction")
        arm_fraction = arm_zero.get("timing_fraction")
        if not isinstance(prepare_fraction, Mapping) or not isinstance(
            arm_fraction, Mapping
        ):
            raise TypeError("prepare와 ARM0 timing fraction이 mapping이 아니다")
        for prefix in ("rational", "reference"):
            numerator = f"{prefix}_numerator"
            denominator = f"{prefix}_denominator"
            prepare_numerator = prepare_fraction.get(numerator)
            prepare_denominator = prepare_fraction.get(denominator)
            arm_numerator = arm_fraction.get(numerator)
            arm_denominator = arm_fraction.get(denominator)
            if any(
                type(value) is not int
                for value in (
                    prepare_numerator,
                    prepare_denominator,
                    arm_numerator,
                    arm_denominator,
                )
            ):
                raise TypeError("prepare와 ARM0 timing fraction 값이 정수가 아니다")
            assert isinstance(prepare_numerator, int)
            assert isinstance(prepare_denominator, int)
            assert isinstance(arm_numerator, int)
            assert isinstance(arm_denominator, int)
            if (
                arm_denominator != prepare_denominator
                or arm_numerator != prepare_numerator + 1
            ):
                raise ValueError(
                    "prepare에서 ARM0 timing fraction이 한 frame 진행하지 않았다"
                )

    def resolve(self) -> None:
        if self.state != "detached":
            self._fail("resolve_before_detach")
            raise ValueError("detach가 끝난 뒤 resolve해야 한다")
        if self._failure is None:
            try:
                self._validate_profile_and_progression()
                preparation_bundle = self._validate_prepare_marker()
                writer = FRAME.AtomicFrameWriter(
                    FRAME.SMOKE_MODE,
                    ARM_COUNT,
                    render_product_ref=self.render_product_ref,
                )
                for ordinal, row in enumerate(self._arms):
                    writer.arm(ordinal, ordinal, row["before"], row["time"])
                    writer.before_step(ordinal)
                    writer.callback(_frame_payload(row["decoded"]))
                    writer.close(ordinal, row["before"], row["time"])
                bundles = writer.finish()
                self._validate_prepare_to_arm_timing(preparation_bundle, bundles[0])
                self._bundles = bundles
            except Exception as error:  # noqa: BLE001 - resolve 실패를 seal evidence로 바꾼다.
                self._fail("resolve_validation_failed", error=error)
        self.state = "resolved"

    def seal(self) -> dict[str, Any]:
        if self.state != "resolved" or self._sealed is not None:
            self._fail("seal_state_invalid")
            raise ValueError("resolve 뒤 한 번만 seal할 수 있다")
        approved = self._failure is None and len(self._bundles) == ARM_COUNT
        preparation = None
        if self._prepare is not None:
            preparation = {
                "logical_ordinal": PREPARE_LOGICAL_ORDINAL,
                "journal_written": False,
                "marker_code": PREPARE_MARKER_CODE,
                "bootstrap_empty_observed": self._prepare["bootstrap_empty_observed"],
                "valid_raw_index": self._prepare["valid_raw_index"],
                "frame_number": self._prepare["decoded"]["read_times"]["frameNumber"],
            }
        self._sealed = {
            "schema": SCHEMA,
            "approved": approved,
            "state": "sealed",
            "raw_callback_count": self.raw_callback_count,
            "callback_rows_dropped": self.raw_callback_count - len(self.callbacks),
            "preparation": preparation,
            "arm_sources": [
                {
                    "ordinal": ordinal,
                    "marker_code": ordinal,
                    "raw_index": row["callbacks"][0]["raw_index"],
                    "frame_number": row["decoded"]["read_times"]["frameNumber"],
                }
                for ordinal, row in enumerate(self._arms)
            ],
            "callbacks": deepcopy(self.callbacks),
            "bundles": deepcopy(self._bundles),
            "failure": deepcopy(self._failure),
        }
        self.state = "sealed"
        return deepcopy(self._sealed)
