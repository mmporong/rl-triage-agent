"""Replicator semantic callback 구조를 capture 승인 없이 bounded 관찰한다."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import vision_writer_payload_v1 as PAYLOAD

SCHEMA = "vision_semantic_diagnostic_v1"
MAX_EVENTS = 2
MAX_CALLBACKS = 16
REPORT_SUFFIX = ".semantic-diagnostic.json"
ISAAC_TIMING_KEYS = (
    "durationDenominator",
    "durationNumerator",
    "externalTimeOfSimNs",
    "frameNumber",
    "rationalTimeOfSimDenominator",
    "rationalTimeOfSimNumerator",
    "sampleTimeOffsetInSimFrames",
    "simulationTime",
    "simulationTimeMonotonic",
    "swhFrameNumber",
    "systemTime",
)


class DiagnosticFailure(RuntimeError):
    """부분 진단 report와 원래 일반 예외를 함께 전달한다."""

    def __init__(self, report: dict[str, Any], primary: Exception) -> None:
        super().__init__(f"semantic diagnostic failed: {_type_name(primary)}")
        self.report = report
        self.primary = primary


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_point)


def _numeric_scalar(value: Any) -> dict[str, Any]:
    result = {"type": _type_name(value), "numeric": False}
    if isinstance(value, (bool, np.bool_)):
        return result
    if isinstance(value, (int, np.integer)):
        return {**result, "numeric": True, "value": int(value)}
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return {**result, "numeric": True, "value": float(value)}
    return result


def _single_mapping_item(value: Any) -> tuple[str, Any] | None:
    if not isinstance(value, Mapping) or len(value) != 1:
        return None
    key, item = next(iter(value.items()))
    if not isinstance(key, str):
        return None
    return key, item


def _raw_frame_time(payload: Any) -> dict[str, Any]:
    result = {
        "swhFrameNumber": None,
        "reference_time": None,
        "isaac_read_times": {},
        "unexpected_isaac_read_times_fields": [],
    }
    if not isinstance(payload, Mapping):
        return result
    result["swhFrameNumber"] = _numeric_scalar(payload.get("swhFrameNumber"))
    reference = payload.get("reference_time")
    if isinstance(reference, (list, tuple)) and len(reference) == 2:
        result["reference_time"] = [_numeric_scalar(item) for item in reference]
    annotators = payload.get("annotators")
    if not isinstance(annotators, Mapping):
        return result
    row = _single_mapping_item(annotators.get("IsaacReadTimes"))
    if row is None or not isinstance(row[1], Mapping):
        return result
    result["isaac_read_times"] = {
        key: _numeric_scalar(row[1][key]) for key in ISAAC_TIMING_KEYS if key in row[1]
    }
    result["unexpected_isaac_read_times_fields"] = [
        {"name": key, "type": _type_name(value)}
        for key, value in sorted(
            (
                (key, value)
                for key, value in row[1].items()
                if isinstance(key, str) and key not in ISAAC_TIMING_KEYS
            ),
            key=lambda item: item[0],
        )
    ]
    return result


def _empty_semantic_candidate(payload: Any) -> tuple[bool, Any]:
    if not isinstance(payload, Mapping):
        return False, None
    annotators = payload.get("annotators")
    if not isinstance(annotators, Mapping):
        return False, None
    semantic_outer = annotators.get("semantic_segmentation")
    semantic_row = _single_mapping_item(semantic_outer)
    resolution_row = _single_mapping_item(annotators.get("resolution"))
    if semantic_row is None or resolution_row is None:
        return False, None
    if semantic_row[0] != resolution_row[0]:
        return False, None
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
        return False, None
    resolution = resolution_row[1]
    if (
        not isinstance(resolution, (list, tuple, np.ndarray))
        or len(resolution) != 2
        or any(type(item) not in (int, np.int32, np.int64) for item in resolution)
    ):
        return False, None
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0:
        return False, None
    candidate: dict[Any, Any] = dict(payload)
    candidate_annotators: dict[Any, Any] = dict(annotators)
    candidate_semantics: dict[Any, Any] = dict(semantic_outer)
    candidate_semantic: dict[Any, Any] = dict(semantic)
    candidate_semantic["data"] = np.zeros((height, width), dtype=np.uint32)
    candidate_semantics[semantic_row[0]] = candidate_semantic
    candidate_annotators["semantic_segmentation"] = candidate_semantics
    candidate["annotators"] = candidate_annotators
    return True, candidate


def _classify(payload: Any) -> tuple[str, dict[str, Any]]:
    try:
        PAYLOAD.decode_writer_payload(payload)
    except Exception as strict_error:  # noqa: BLE001 - SDK decode 오류를 분류한다.
        try:
            is_empty, candidate = _empty_semantic_candidate(payload)
        except Exception as classification_error:  # noqa: BLE001 - 구조 오류를 기록한다.
            return "invalid", {
                "classification_error_type": _type_name(classification_error),
                "strict_error_type": _type_name(strict_error),
                "counterfactual_schema_check": {
                    "attempted": False,
                    "passed": False,
                    "reason": None,
                    "error_type": None,
                },
            }
        if is_empty:
            try:
                PAYLOAD.decode_writer_payload(candidate)
            except Exception as remaining_error:  # noqa: BLE001 - 반사실 검사 오류다.
                return "invalid", {
                    "classification_error_type": None,
                    "strict_error_type": _type_name(strict_error),
                    "counterfactual_schema_check": {
                        "attempted": True,
                        "passed": False,
                        "reason": "semantic_float64_empty_replaced_for_schema_check",
                        "error_type": _type_name(remaining_error),
                    },
                }
            return "invalid", {
                "classification_error_type": None,
                "strict_error_type": _type_name(strict_error),
                "counterfactual_schema_check": {
                    "attempted": True,
                    "passed": True,
                    "reason": "semantic_float64_empty_replaced_for_schema_check",
                    "error_type": None,
                },
            }
        return "invalid", {
            "classification_error_type": None,
            "strict_error_type": _type_name(strict_error),
            "counterfactual_schema_check": {
                "attempted": False,
                "passed": False,
                "reason": None,
                "error_type": None,
            },
        }
    return "valid", {
        "classification_error_type": None,
        "strict_error_type": None,
        "counterfactual_schema_check": {
            "attempted": False,
            "passed": False,
            "reason": None,
            "error_type": None,
        },
    }


class DiagnosticObserver:
    """Writer callback을 event 인과 주장 없이 bounded 구조 evidence로 모은다."""

    def __init__(self, max_callbacks: int = MAX_CALLBACKS) -> None:
        if type(max_callbacks) is not int or not 0 < max_callbacks <= MAX_CALLBACKS:
            raise ValueError("semantic diagnostic callback 상한이 잘못됐다")
        self.max_callbacks = max_callbacks
        self.total_callbacks = 0
        self.overflow = False
        self.window = "after_attach"
        self.callbacks: list[dict[str, Any]] = []
        self._collection_started = False
        self._events_reserved = 0

    def _begin_collection(self, event_count: int) -> None:
        if self._collection_started or self._events_reserved + event_count > MAX_EVENTS:
            raise ValueError("semantic diagnostic observer는 single-use다")
        self._collection_started = True
        self._events_reserved += event_count

    def observe(self, payload: Any) -> None:
        ordinal = self.total_callbacks
        self.total_callbacks += 1
        if len(self.callbacks) >= self.max_callbacks:
            self.overflow = True
            return
        callback = {
            "ordinal": ordinal,
            "window": self.window,
            "descriptor": PAYLOAD.describe_writer_payload(payload),
            "raw_frame_time": _raw_frame_time(payload),
            "classification": "invalid",
            "classification_error_type": None,
            "strict_error_type": None,
            "counterfactual_schema_check": {
                "attempted": False,
                "passed": False,
                "reason": None,
                "error_type": None,
            },
        }
        self.callbacks.append(callback)
        try:
            classification, errors = _classify(payload)
        except Exception as error:  # noqa: BLE001 - callback evidence에 종류를 남긴다.
            callback["classification_error_type"] = _type_name(error)
        else:
            callback["classification"] = classification
            callback.update(errors)


def create_diagnostic_writer(rep: Any, observer: DiagnosticObserver) -> Any:
    """주입된 Replicator SDK로만 진단 Writer를 만든다."""
    if not isinstance(observer, DiagnosticObserver):
        raise TypeError("semantic diagnostic observer가 다르다")

    class SemanticDiagnosticWriter(rep.Writer):
        data_structure = "annotator"

        def __init__(self) -> None:
            self.annotators = [
                rep.AnnotatorRegistry.get_annotator("rgb"),
                rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation", init_params={"colorize": False}
                ),
                rep.AnnotatorRegistry.get_annotator("CameraParams"),
                rep.AnnotatorRegistry.get_annotator("IsaacReadTimes"),
            ]

        def write(self, data: dict[str, Any]) -> None:
            observer.observe(data)

    return SemanticDiagnosticWriter()


def _physics_snapshot(
    source: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    observed = source()
    if not isinstance(observed, Mapping):
        raise TypeError("semantic diagnostic physics snapshot은 mapping이어야 한다")
    copied = deepcopy(dict(observed))
    digest = _sha(copied)
    return copied, digest


def _build_report(
    observer: DiagnosticObserver,
    events: list[dict[str, Any]],
    event_count: int,
) -> dict[str, Any]:
    classes = [row["classification"] for row in observer.callbacks]
    counterfactual_empty = [
        row["counterfactual_schema_check"]["passed"] for row in observer.callbacks
    ]
    valid = "valid" in classes
    empty = any(counterfactual_empty)
    invalid = any(
        row["classification"] == "invalid"
        and not row["counterfactual_schema_check"]["passed"]
        for row in observer.callbacks
    )
    physics_changed = any(row["physics_equal"] is False for row in events)
    source_error = any(row["diagnostic_error_type"] is not None for row in events)
    observations = {
        "valid_observed": valid,
        "empty_observed": empty,
        "mixed_valid_empty": valid and empty,
        "continued_empty": bool(classes) and all(counterfactual_empty) and not valid,
        "invalid_observed": invalid,
        "trigger_zero": all(
            row["post_callback_count"] == row["pre_callback_count"] for row in events
        ),
        "physics_changed": physics_changed,
        "overflow": observer.overflow,
        "source_error": source_error,
    }
    hold = [
        name
        for name in (
            "empty_observed",
            "mixed_valid_empty",
            "continued_empty",
            "invalid_observed",
            "trigger_zero",
            "physics_changed",
            "overflow",
            "source_error",
        )
        if observations[name]
    ]
    last_phase = events[-1]["diagnostic_error_phase"] if events else None
    if last_phase == "physics_before":
        scope_end = "before_run_event_after_failed_pre_physics_snapshot"
    elif last_phase in ("physics_after", "run_event+physics_after"):
        scope_end = "after_run_event_return_during_failed_post_physics_snapshot"
    else:
        scope_end = "immediately_after_last_run_event_return_and_post_physics_snapshot"
    return {
        "schema": SCHEMA,
        "bounds": {"max_events": MAX_EVENTS, "max_callbacks": observer.max_callbacks},
        "events_requested": event_count,
        "events_recorded": len(events),
        "events_attempted": sum(
            row["diagnostic_error_phase"] != "physics_before" for row in events
        ),
        "events_completed": sum(row["diagnostic_error_type"] is None for row in events),
        "events": deepcopy(events),
        "callbacks_observed": observer.total_callbacks,
        "callbacks_preserved": len(observer.callbacks),
        "callbacks": deepcopy(observer.callbacks),
        "observations": observations,
        "hold_observations": hold,
        "no_capture_approved": True,
        "step_return_is_complete_drain": False,
        "callback_event_causality_proven": False,
        "collection_scope": {
            "starts": "after_writer_attach_before_first_run_event",
            "ends": scope_end,
            "late_callbacks_after_returned_report_included": False,
            "detach_before_final_evidence_seal_required_by_runner": True,
            "all_future_callbacks_collected": False,
        },
    }


def _failed_event(
    *,
    index: int,
    callback_count: int,
    phase: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "event_index": index,
        "pre_callback_count": callback_count,
        "post_callback_count": callback_count,
        "physics_before": None,
        "physics_after": None,
        "physics_before_sha256": None,
        "physics_after_sha256": None,
        "physics_equal": None,
        "run_event_error_type": None,
        "physics_after_error_type": None,
        "diagnostic_error_phase": phase,
        "diagnostic_error_type": _type_name(error),
    }


def collect_diagnostic(
    observer: DiagnosticObserver,
    *,
    run_event: Callable[[int], None],
    physics_snapshot: Callable[[], Mapping[str, Any]],
    event_count: int = MAX_EVENTS,
) -> dict[str, Any]:
    """같은 attached Writer를 유지한 채 최대 두 event의 관찰 report를 만든다."""
    if not isinstance(observer, DiagnosticObserver):
        raise TypeError("semantic diagnostic observer가 다르다")
    if not callable(run_event) or not callable(physics_snapshot):
        raise TypeError("semantic diagnostic source call이 callable이 아니다")
    if type(event_count) is not int or not 0 < event_count <= MAX_EVENTS:
        raise ValueError("semantic diagnostic event 상한이 잘못됐다")
    observer._begin_collection(event_count)
    events: list[dict[str, Any]] = []
    for index in range(event_count):
        pre_count = observer.total_callbacks
        try:
            before, before_sha = _physics_snapshot(physics_snapshot)
        except Exception as error:
            events.append(
                _failed_event(
                    index=index,
                    callback_count=pre_count,
                    phase="physics_before",
                    error=error,
                )
            )
            observer.window = "collection_finished"
            report = _build_report(observer, events, event_count)
            raise DiagnosticFailure(report, error) from error
        observer.window = f"event_{index}_running"
        event_error: Exception | None = None
        try:
            run_event(index)
        except Exception as error:  # noqa: BLE001 - 부분 report와 함께 재전파한다.
            event_error = error
        finally:
            observer.window = f"after_event_{index}_return"
        after: dict[str, Any] | None = None
        after_sha: str | None = None
        after_error: Exception | None = None
        try:
            after, after_sha = _physics_snapshot(physics_snapshot)
        except Exception as error:  # noqa: BLE001 - 부분 report와 함께 재전파한다.
            after_error = error
        primary = event_error or after_error
        if event_error is not None and after_error is not None:
            error_phase = "run_event+physics_after"
        elif event_error is not None:
            error_phase = "run_event"
        elif after_error is not None:
            error_phase = "physics_after"
        else:
            error_phase = None
        events.append(
            {
                "event_index": index,
                "pre_callback_count": pre_count,
                "post_callback_count": observer.total_callbacks,
                "physics_before": before,
                "physics_after": after,
                "physics_before_sha256": before_sha,
                "physics_after_sha256": after_sha,
                "physics_equal": before_sha == after_sha
                if after_sha is not None
                else None,
                "run_event_error_type": (
                    _type_name(event_error) if event_error is not None else None
                ),
                "physics_after_error_type": (
                    _type_name(after_error) if after_error is not None else None
                ),
                "diagnostic_error_phase": error_phase,
                "diagnostic_error_type": (
                    _type_name(primary) if primary is not None else None
                ),
            }
        )
        if primary is not None:
            observer.window = "collection_finished"
            report = _build_report(observer, events, event_count)
            raise DiagnosticFailure(report, primary) from primary
        if observer.overflow:
            break
    observer.window = "collection_finished"
    return _build_report(observer, events, event_count)


def write_report(path: Path, report: Mapping[str, Any]) -> Path:
    """capture/atomic namespace와 분리한 diagnostic report를 write-once로 쓴다."""
    target = Path(path)
    if (
        not target.name.endswith(REPORT_SUFFIX)
        or ".capture-" in target.name
        or ".atomic-" in target.name
    ):
        raise ValueError("semantic diagnostic report 경로가 전용 namespace가 아니다")
    if not isinstance(report, Mapping) or report.get("schema") != SCHEMA:
        raise ValueError("semantic diagnostic report schema가 다르다")
    absolute = target.absolute()
    if any(
        _is_link_or_reparse(candidate) for candidate in (absolute, *absolute.parents)
    ):
        raise ValueError("semantic diagnostic report 경로에 symlink가 있다")
    parent = absolute.parent
    if not parent.is_dir():
        raise ValueError("semantic diagnostic report parent가 실제 directory가 아니다")
    raw = _canonical(dict(report)) + b"\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags, 0o600)
    except OSError as error:
        raise ValueError(
            "semantic diagnostic report를 write-once로 만들 수 없다"
        ) from error
    with os.fdopen(descriptor, "wb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("semantic diagnostic report가 단일 regular file이 아니다")
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return absolute.resolve()
