"""준비 callback 뒤 RECORD frame을 bounded streaming으로 검증한다."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from typing import Any

import vision_frame_sync_v1 as FRAME
import vision_prepared_capture_v1 as PREPARED
import vision_writer_payload_v1 as PAYLOAD

SCHEMA = "vision_prepared_record_v1"
MAX_RECORD_FRAMES = 625
MAX_ERROR_ROWS = 4
MAX_RECEIPT_BYTES = 64 * 1024

CommitCallback = Callable[
    [int, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
]


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("record commit receipt는 mapping이어야 한다")
    copied = deepcopy(dict(value))
    raw = _canonical(copied)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("record commit receipt가 byte 상한을 넘었다")
    return copied


def _raw(decoded: Mapping[str, Any]) -> tuple[bytes, bytes]:
    rgb = decoded["rgb"]
    semantic = decoded["semantic"]["mask"]
    return rgb.tobytes(order="C"), semantic.tobytes(order="C")


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


def _advanced(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    for prefix in ("rational", "reference"):
        numerator = f"{prefix}_numerator"
        denominator = f"{prefix}_denominator"
        if (
            current[numerator] * previous[denominator]
            <= previous[numerator] * current[denominator]
        ):
            return False
    return True


class PreparedRecordMux:
    """한 Writer attach의 PREPARE→RECORD arms→DETACH 상태를 소유한다.

    ``observe``는 callback 안에서 payload를 복사한다. 각 arm은 ``end_arm``에서
    frozen frame writer와 원문으로 검증된 뒤 commit callback에 전달된다. mux는
    배열과 원문 bytes를 보존하지 않고 경량 manifest와 receipt만 유지한다.
    """

    def __init__(
        self,
        *,
        render_product_ref: str,
        camera_path: str,
        resolution: Sequence[int],
        expected_count: int,
        commit_callback: CommitCallback,
    ) -> None:
        if not isinstance(render_product_ref, str) or not render_product_ref:
            raise ValueError("record RenderProduct reference가 없다")
        if not isinstance(camera_path, str) or not camera_path:
            raise ValueError("record camera path가 없다")
        if (
            isinstance(resolution, (str, bytes))
            or len(resolution) != 2
            or any(type(item) is not int or item <= 0 for item in resolution)
        ):
            raise ValueError("record resolution이 잘못됐다")
        if (
            type(expected_count) is not int
            or expected_count <= 0
            or expected_count > MAX_RECORD_FRAMES
        ):
            raise ValueError("record frame 수가 등록 상한을 벗어났다")
        if not callable(commit_callback):
            raise TypeError("record commit callback이 callable이 아니다")
        self.render_product_ref = render_product_ref
        self.camera_path = camera_path
        self.resolution = list(resolution)
        self.expected_count = expected_count
        self.commit_callback = commit_callback
        self.state = "new"
        self.raw_callback_count = 0
        self.callbacks: list[dict[str, Any]] = []
        self.manifests: list[dict[str, Any]] = []
        self._phase: dict[str, Any] | None = None
        self._prepare: dict[str, Any] | None = None
        self._failure: dict[str, Any] | None = None
        self._sealed: dict[str, Any] | None = None
        self._callback_refs: set[str] = set()
        self._profile: dict[str, Any] | None = None
        self._last_checked: dict[str, Any] | None = None
        self._before_prepare: dict[str, Any] | None = None
        self._after_detach: dict[str, Any] | None = None

    @property
    def raw_limit(self) -> int:
        return self.expected_count + 2

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

    def _guard(self) -> None:
        if self._failure is not None:
            raise RuntimeError("record mux는 실패 뒤 capture를 재개할 수 없다")

    def _physics(self, value: Any, simulation_time_s: Any) -> dict[str, Any]:
        copied, time, digest = PREPARED._physics(value, simulation_time_s)
        return {"snapshot": copied, "time": time, "sha256": digest}

    def _begin(
        self,
        kind: str,
        ordinal: int | None,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        try:
            physics = self._physics(state_before, simulation_time_s)
        except Exception as error:
            self._fail("physics_before_invalid", error=error)
            raise
        self._phase = {
            "kind": kind,
            "ordinal": ordinal,
            "physics": physics,
            "callbacks": [],
        }

    def begin_prepare(
        self, state_before: Mapping[str, Any], simulation_time_s: float
    ) -> None:
        self._guard()
        if self.state != "new":
            self._fail("prepare_begin_invalid")
            raise ValueError("record prepare는 new 상태에서 한 번만 시작한다")
        self.state = "preparing"
        self._begin("prepare", None, state_before, simulation_time_s)
        assert self._phase is not None
        self._before_prepare = deepcopy(self._phase["physics"])

    def begin_arm(
        self,
        ordinal: int,
        state_before: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        self._guard()
        expected_state = "prepared" if ordinal == 0 else f"arm{ordinal - 1}_done"
        if (
            type(ordinal) is not int
            or ordinal != len(self.manifests)
            or ordinal >= self.expected_count
            or self.state != expected_state
        ):
            self._fail("arm_begin_invalid")
            raise ValueError("record arm ordinal/state가 등록 순서와 다르다")
        self.state = f"arming{ordinal}"
        self._begin("arm", ordinal, state_before, simulation_time_s)

    def source_error(self, error: Exception) -> None:
        if not isinstance(error, Exception):
            raise TypeError("record source error는 Exception이어야 한다")
        self._fail("source_error_before_callback", error=error)

    def observe(self, payload: Any) -> None:
        raw_index = self.raw_callback_count
        self.raw_callback_count += 1
        if raw_index >= self.raw_limit:
            if len(self.callbacks) < self.raw_limit + MAX_ERROR_ROWS:
                self.callbacks.append(
                    {
                        "raw_index": raw_index,
                        "window": self.state,
                        "classification": "not_processed_overflow",
                        "error_type": None,
                        "descriptor_sha256": None,
                        "frame_number": None,
                    }
                )
            self._fail("raw_callback_overflow")
            return
        row: dict[str, Any] = {
            "raw_index": raw_index,
            "window": self.state,
            "classification": "invalid",
            "error_type": None,
            "descriptor_sha256": None,
            "frame_number": None,
        }
        self.callbacks.append(row)
        try:
            budget = PREPARED._SnapshotBudget()
            frozen = PREPARED._snapshot(payload, budget)
            descriptor = PAYLOAD.describe_writer_payload(frozen)
            row["descriptor_sha256"] = _sha(descriptor)
            classification, decoded, error_type = PREPARED._classify(frozen, budget)
        except Exception as error:  # noqa: BLE001 - callback 오류를 sticky evidence로 남긴다.
            row["error_type"] = _type_name(error)
            self._fail("callback_snapshot_or_classification_failed", error=error)
            return
        row["classification"] = classification
        row["error_type"] = error_type
        row["frame_number"] = (
            decoded["read_times"]["frameNumber"] if decoded is not None else None
        )
        if self.state != "preparing" and not self.state.startswith("arming"):
            self._fail(
                "callback_during_detach"
                if self.state == "detaching"
                else "unsolicited_callback"
            )
            return
        phase = self._phase
        if phase is None:
            self._fail("callback_without_phase")
            return
        phase["callbacks"].append(
            {
                "raw_index": raw_index,
                "classification": classification,
                "decoded": decoded,
                "descriptor_sha256": row["descriptor_sha256"],
            }
        )
        classes = [item["classification"] for item in phase["callbacks"]]
        if self.state == "preparing":
            if classes not in (
                ["valid"],
                ["bootstrap_empty"],
                ["bootstrap_empty", "valid"],
            ):
                self._fail("prepare_callback_sequence_invalid")
        elif classes != ["valid"]:
            self._fail("arm_callback_invalid_or_extra")

    def _end(
        self, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> dict[str, Any]:
        phase = self._phase
        if phase is None:
            self._fail("phase_missing")
            raise ValueError("record phase가 없다")
        try:
            after = self._physics(state_after, simulation_time_s)
        except Exception as error:
            self._fail("physics_after_invalid", error=error)
            raise
        before = phase["physics"]
        if before["sha256"] != after["sha256"] or before["time"] != after["time"]:
            self._fail("physics_or_time_changed_during_capture")
        return phase

    def _validate_profile(self, decoded: Mapping[str, Any]) -> None:
        current = {
            "render_product_ref": decoded["render_product_ref"],
            "camera_path": decoded["camera_path"],
            "resolution": decoded["render_product_resolution"],
            "projection": decoded["camera"]["projection"],
        }
        expected = {
            "render_product_ref": self.render_product_ref,
            "camera_path": self.camera_path,
            "resolution": self.resolution,
            "projection": current["projection"],
        }
        if self._profile is None:
            if current != expected:
                raise ValueError("record prepare profile이 등록값과 다르다")
            self._profile = deepcopy(current)
        elif current != self._profile:
            raise ValueError("record callback profile/projection이 바뀌었다")

    def _one_frame(
        self,
        decoded: Mapping[str, Any],
        physics: Mapping[str, Any],
        ordinal: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        writer = FRAME.AtomicFrameWriter(
            FRAME.RECORD_MODE, 1, render_product_ref=self.render_product_ref
        )
        writer.arm(0, None, physics["snapshot"], physics["time"])
        writer.before_step(0)
        writer.callback(_frame_payload(decoded))
        bundle = writer.close(0, physics["snapshot"], physics["time"])
        numbered = deepcopy(bundle)
        numbered["ordinal"] = ordinal
        numbered["callback_sequence"] = ordinal
        checked = FRAME.verify_bundle(numbered, _raw(decoded), ordinal)
        return numbered, checked

    def end_prepare(
        self, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> None:
        if self.state != "preparing":
            self._fail("prepare_end_invalid")
            raise ValueError("record prepare가 시작되지 않았다")
        phase = self._end(state_after, simulation_time_s)
        classes = [item["classification"] for item in phase["callbacks"]]
        if classes not in (["valid"], ["bootstrap_empty", "valid"]):
            self._fail("prepare_callback_sequence_invalid")
        if self._failure is not None:
            raise RuntimeError("record prepare callback 계약이 실패했다")
        valid = phase["callbacks"][-1]
        decoded = valid["decoded"]
        assert isinstance(decoded, Mapping)
        try:
            self._validate_profile(decoded)
            bundle, checked = self._one_frame(decoded, phase["physics"], 0)
        except Exception as error:
            self._fail("prepare_validation_failed", error=error)
            raise
        self._prepare = {
            "bootstrap_empty_observed": classes == ["bootstrap_empty", "valid"],
            "valid_raw_index": valid["raw_index"],
            "frame_number": checked["frame"],
            "swh_frame_number": checked["swh_frame"],
            "timing_fraction": bundle["timing_fraction"],
            "descriptor_sha256": valid["descriptor_sha256"],
        }
        self._callback_refs.add(checked["callback_ref"])
        self._last_checked = checked
        self._phase = None
        self.state = "prepared"

    def _validate_progression(self, checked: Mapping[str, Any]) -> None:
        previous = self._last_checked
        if previous is None:
            raise ValueError("record 이전 timing evidence가 없다")
        if (
            checked["frame"] <= previous["frame"]
            or checked["swh_frame"] < previous["swh_frame"]
            or not _advanced(previous["fraction"], checked["fraction"])
        ):
            raise ValueError("record callback frame/timing이 진행하지 않았다")
        if (
            checked["render_product_ref"] != previous["render_product_ref"]
            or checked["projection"] != previous["projection"]
            or checked["resolution"] != previous["resolution"]
        ):
            raise ValueError("record callback RP/projection/resolution이 바뀌었다")
        if checked["callback_ref"] in self._callback_refs:
            raise ValueError("record callback reference가 replay됐다")

    def end_arm(
        self,
        ordinal: int,
        state_after: Mapping[str, Any],
        simulation_time_s: float,
    ) -> None:
        if self.state != f"arming{ordinal}" or self._phase is None:
            self._fail("arm_end_invalid")
            raise ValueError("record arm이 시작되지 않았다")
        phase = self._end(state_after, simulation_time_s)
        callbacks = phase["callbacks"]
        if len(callbacks) != 1 or callbacks[0]["classification"] != "valid":
            self._fail("arm_callback_cardinality_invalid")
        if self._failure is not None:
            raise RuntimeError("record arm callback 계약이 실패했다")
        decoded = callbacks[0]["decoded"]
        assert isinstance(decoded, Mapping)
        try:
            self._validate_profile(decoded)
            bundle, checked = self._one_frame(decoded, phase["physics"], ordinal)
            self._validate_progression(checked)
            receipt = _receipt(self.commit_callback(ordinal, decoded, bundle))
        except Exception as error:
            self._fail("arm_validation_or_commit_failed", error=error)
            raise
        self._callback_refs.add(checked["callback_ref"])
        self.manifests.append(
            {
                "ordinal": ordinal,
                "raw_index": callbacks[0]["raw_index"],
                "callback_ref": checked["callback_ref"],
                "frame_number": checked["frame"],
                "swh_frame_number": checked["swh_frame"],
                "timing_fraction": deepcopy(checked["fraction"]),
                "physics": deepcopy(bundle["physics"]),
                "rgb": deepcopy(bundle["rgb"]),
                "semantic_mask": deepcopy(bundle["semantic_mask"]),
                "id_to_labels_sha256": _sha(bundle["id_to_labels"]),
                "descriptor_sha256": callbacks[0]["descriptor_sha256"],
                "commit_receipt": receipt,
            }
        )
        self._last_checked = deepcopy(checked)
        self._phase = None
        self.state = f"arm{ordinal}_done"

    def begin_detach(self) -> None:
        if self.state in {"detaching", "detached"}:
            self._fail("detach_begin_invalid")
            raise ValueError("record detach가 이미 시작됐다")
        expected = f"arm{self.expected_count - 1}_done"
        if self._failure is None and self.state != expected:
            self._fail("detach_before_arms_complete")
            raise ValueError("record arms가 끝나기 전에 detach할 수 없다")
        self.state = "detaching"

    def end_detach(
        self, state_after: Mapping[str, Any], simulation_time_s: float
    ) -> None:
        if self.state != "detaching":
            self._fail("detach_end_invalid")
            raise ValueError("record detach가 시작되지 않았다")
        try:
            self._after_detach = self._physics(state_after, simulation_time_s)
        except Exception as error:
            self._fail("detach_evidence_invalid", error=error)
            raise
        self.state = "detached"

    def resolve(self) -> None:
        if self.state != "detached":
            self._fail("resolve_before_detach")
            raise ValueError("record detach 뒤 resolve해야 한다")
        if self._failure is None and len(self.manifests) != self.expected_count:
            self._fail("record_count_incomplete")
        if self._failure is None:
            frames = [row["frame_number"] for row in self.manifests]
            if any(right <= left for left, right in pairwise(frames)):
                self._fail("record_manifest_frame_order_invalid")
        self.state = "resolved"

    def seal(self) -> dict[str, Any]:
        if self.state != "resolved" or self._sealed is not None:
            self._fail("seal_state_invalid")
            raise ValueError("record resolve 뒤 한 번만 seal할 수 있다")
        approved = self._failure is None and len(self.manifests) == self.expected_count
        self._sealed = {
            "schema": SCHEMA,
            "approved": approved,
            "state": "sealed",
            "expected_count": self.expected_count,
            "committed_count": len(self.manifests),
            "raw_callback_count": self.raw_callback_count,
            "raw_callback_limit": self.raw_limit,
            "callback_rows_dropped": self.raw_callback_count - len(self.callbacks),
            "before_prepare_physics": deepcopy(self._before_prepare),
            "after_detach_physics": deepcopy(self._after_detach),
            "preparation": deepcopy(self._prepare),
            "callbacks": deepcopy(self.callbacks),
            "manifests": deepcopy(self.manifests),
            "failure": deepcopy(self._failure),
        }
        self.state = "sealed"
        return deepcopy(self._sealed)
