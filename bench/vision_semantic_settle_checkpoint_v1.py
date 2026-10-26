"""marker visibility 변경 뒤 semantic graph를 정착시키고 checkpoint smoke를 계측한다."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import vision_callback_boundary_diagnostic_v2 as BOUNDARY
import vision_checkpoint_smoke_sdk_v2 as CHECKPOINT
import vision_prepared_capture_v1 as CAPTURE
import vision_prepared_sdk_v1 as PREPARED

SCHEMA = "vision_semantic_settle_checkpoint_v1"
MODE = "checkpoint_semantic_settle_diagnostic_only"
VARIANT = "manual_semantic_settle"
TOP_KEYS = frozenset(
    {
        "schema",
        "variant",
        "boundary",
        "settles",
        "marker_states",
        "primary_error",
        "diagnostic_complete",
        "used_for_decision",
        "approved",
    }
)
SETTLE_KEYS = frozenset(
    {
        "ordinal",
        "code",
        "status",
        "monotonic_ns",
        "thread_id",
        "callbacks_before",
        "callbacks_after",
        "callback_delta",
        "physics_before_sha256",
        "physics_after_sha256",
        "simulation_time_before_s",
        "simulation_time_after_s",
        "physics_equal",
        "marker_before",
        "marker_after",
        "error_type",
        "error_message",
    }
)
MARKER_STATE_KEYS = frozenset(
    {
        "phase",
        "code",
        "path_count",
        "visible_count",
        "visibility_sha256",
        "semantic_attributes_sha256",
        "semantic_attribute_count",
    }
)


class SemanticSettleDiagnosticFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(
            f"semantic settle diagnostic failed: {type(primary).__name__}: {primary}"
        )
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _marker_state(marker: Any, stage: Any, code: int | None) -> dict[str, Any]:
    paths = getattr(marker, "_paths", None)
    if not isinstance(paths, Mapping) or len(paths) != 9:
        raise ValueError("semantic settle marker path가 정확히 아홉 개가 아니다")
    visibility: dict[str, str] = {}
    semantics: dict[str, list[tuple[str, str]]] = {}
    for key, path in sorted(paths.items()):
        if not isinstance(key, str) or not isinstance(path, str):
            raise TypeError("semantic settle marker path 형식이 다르다")
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError("semantic settle marker prim이 유효하지 않다")
        attribute = prim.GetAttribute("visibility")
        visibility[key] = str(attribute.Get())
        semantic_rows = []
        for item in prim.GetAttributes():
            name = str(item.GetName())
            if "semantic" in name.lower():
                semantic_rows.append((name, repr(item.Get())))
        semantics[key] = sorted(semantic_rows)
    visible = sum(value != "invisible" for value in visibility.values())
    return {
        "code": code,
        "path_count": len(paths),
        "visible_count": visible,
        "visibility_sha256": _canonical_sha(visibility),
        "semantic_attributes_sha256": _canonical_sha(semantics),
        "semantic_attribute_count": sum(len(value) for value in semantics.values()),
    }


class _MarkerProxy:
    def __init__(
        self,
        marker: Any,
        marker_states: list[dict[str, Any]],
    ) -> None:
        self._marker = marker
        self._marker_states = marker_states
        self._stage: Any | None = None
        self._semantic_sha: str | None = None
        self._set_count = 0
        self.pending: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._marker, name)

    def bind_and_validate(self, stage: Any) -> None:
        self._marker.bind_and_validate(stage)
        self._stage = stage
        state = _marker_state(self._marker, stage, None)
        self._semantic_sha = state["semantic_attributes_sha256"]
        self._marker_states.append({"phase": "bound", **state})

    def set_code(self, stage: Any, code: int) -> None:
        if self._stage is None or stage is not self._stage:
            raise ValueError("semantic settle marker stage가 binding과 다르다")
        self._marker.set_code(stage, code)
        state = _marker_state(self._marker, stage, code)
        if state["semantic_attributes_sha256"] != self._semantic_sha:
            raise RuntimeError("marker set_code가 semantic attribute를 바꿨다")
        if state["visible_count"] != 5:
            raise RuntimeError("marker code의 visible prim 수가 다르다")
        phase = "prepare" if self._set_count == 0 else f"arm_{self._set_count - 1}"
        self._marker_states.append({"phase": phase, **state})
        if self._set_count > 0:
            ordinal = self._set_count - 1
            if ordinal not in range(4) or code != ordinal:
                raise ValueError("semantic settle arm marker 순서가 다르다")
            if self.pending is not None:
                raise RuntimeError("이전 semantic settle이 끝나지 않았다")
            self.pending = {
                "ordinal": ordinal,
                "code": code,
                "marker_before": deepcopy(state),
            }
        self._set_count += 1


def _instrument_dependencies(
    selected: Mapping[str, Any],
    callbacks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    capture_on_play: dict[str, Any],
    settles: list[dict[str, Any]],
    marker_states: list[dict[str, Any]],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    raw = dict(selected)
    deps = BOUNDARY._instrument_dependencies(
        raw,
        "manual_schedule",
        callbacks,
        operations,
        capture_on_play,
    )
    marker_factory = deps["marker"]
    physics_snapshot = deps["physics_snapshot"]
    raw_render = raw["capture_render_step"]
    raw_orchestrator = raw["rep"].orchestrator
    holder: dict[str, _MarkerProxy] = {}

    def create_marker(rep: Any, root: Any) -> _MarkerProxy:
        marker = _MarkerProxy(marker_factory(rep, root), marker_states)
        if holder:
            raise RuntimeError("semantic settle marker가 두 번 생성됐다")
        holder["marker"] = marker
        return marker

    def snapshot(base: Any) -> Any:
        before = physics_snapshot(base)
        marker = holder.get("marker")
        if marker is None or marker.pending is None:
            return before
        pending = marker.pending
        ordinal = pending["ordinal"]
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "code": pending["code"],
            "status": "started",
            "monotonic_ns": time.monotonic_ns(),
            "thread_id": threading.get_ident(),
            "callbacks_before": len(callbacks),
            "callbacks_after": None,
            "callback_delta": None,
            "physics_before_sha256": None,
            "physics_after_sha256": None,
            "simulation_time_before_s": None,
            "simulation_time_after_s": None,
            "physics_equal": False,
            "marker_before": deepcopy(pending["marker_before"]),
            "marker_after": None,
            "error_type": None,
            "error_message": None,
        }
        settles.append(row)
        try:
            time_before = float(base.sim.current_time)
            _, _, before_sha = CAPTURE._physics(before, time_before)
            row["physics_before_sha256"] = before_sha
            row["simulation_time_before_s"] = time_before
            raw_render(raw_orchestrator, profile)
            after = physics_snapshot(base)
            time_after = float(base.sim.current_time)
            _, _, after_sha = CAPTURE._physics(after, time_after)
            row["physics_after_sha256"] = after_sha
            row["simulation_time_after_s"] = time_after
            row["callbacks_after"] = len(callbacks)
            row["callback_delta"] = len(callbacks) - row["callbacks_before"]
            row["marker_after"] = _marker_state(
                marker._marker, marker._stage, pending["code"]
            )
            row["physics_equal"] = before_sha == after_sha and time_before == time_after
            if row["callback_delta"] != 0:
                raise RuntimeError("unscheduled semantic settle 중 callback이 발생했다")
            if row["physics_equal"] is not True:
                raise RuntimeError(
                    "semantic settle이 physics 또는 simulation time을 바꿨다"
                )
            if row["marker_after"] != row["marker_before"]:
                raise RuntimeError("semantic settle 중 marker state가 바뀌었다")
            row["status"] = "complete"
            marker.pending = None
            return after
        except BaseException as error:
            row["callbacks_after"] = len(callbacks)
            row["callback_delta"] = len(callbacks) - row["callbacks_before"]
            row["error_type"] = _type_name(error)
            row["error_message"] = str(error)
            row["status"] = "failed"
            raise

    deps["marker"] = create_marker
    deps["physics_snapshot"] = snapshot
    return deps


def _settles_complete(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(
        isinstance(row, Mapping)
        and set(row) == SETTLE_KEYS
        and row.get("ordinal") == ordinal
        and row.get("code") == ordinal
        and row.get("status") == "complete"
        and row.get("callback_delta") == 0
        and row.get("physics_equal") is True
        and row.get("marker_before") == row.get("marker_after")
        and all(
            isinstance(row.get(key), str)
            and len(row[key]) == 64
            and all(char in "0123456789abcdef" for char in row[key])
            for key in ("physics_before_sha256", "physics_after_sha256")
        )
        and row.get("error_type") is None
        and row.get("error_message") is None
        for ordinal, row in enumerate(value)
    )


def _marker_states_complete(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 6:
        return False
    phases = ["bound", "prepare", "arm_0", "arm_1", "arm_2", "arm_3"]
    semantic_sha: str | None = None
    for index, (row, phase) in enumerate(zip(value, phases, strict=True)):
        if not isinstance(row, Mapping) or set(row) != MARKER_STATE_KEYS:
            return False
        candidate = row.get("semantic_attributes_sha256")
        visibility_sha = row.get("visibility_sha256")
        if (
            row.get("phase") != phase
            or row.get("path_count") != 9
            or type(row.get("semantic_attribute_count")) is not int
            or row["semantic_attribute_count"] <= 0
            or not isinstance(candidate, str)
            or len(candidate) != 64
            or any(char not in "0123456789abcdef" for char in candidate)
            or not isinstance(visibility_sha, str)
            or len(visibility_sha) != 64
            or any(char not in "0123456789abcdef" for char in visibility_sha)
        ):
            return False
        if index == 0:
            semantic_sha = candidate
        elif row.get("code") != index - 2 if index >= 2 else row.get("code") != 3:
            return False
        if candidate != semantic_sha:
            return False
        if index > 0 and row.get("visible_count") != 5:
            return False
    return True


def _report(
    boundary: Mapping[str, Any],
    settles: list[dict[str, Any]],
    marker_states: list[dict[str, Any]],
    primary: BaseException | None,
) -> dict[str, Any]:
    underlying = boundary.get("underlying")
    probe = underlying.get("policy_probe") if isinstance(underlying, Mapping) else None
    complete = (
        boundary.get("diagnostic_complete") is True
        and _settles_complete(settles)
        and _marker_states_complete(marker_states)
        and isinstance(probe, Mapping)
        and probe.get("environment_steps") == 1
        and probe.get("capture_completed_before_policy_step") is True
        and probe.get("writer_detached_before_policy_step") is True
    )
    return {
        "schema": SCHEMA,
        "variant": VARIANT,
        "boundary": deepcopy(dict(boundary)),
        "settles": deepcopy(settles),
        "marker_states": deepcopy(marker_states),
        "primary_error": (
            None
            if primary is None
            else {"error_type": _type_name(primary), "error_message": str(primary)}
        ),
        "diagnostic_complete": complete,
        "used_for_decision": False,
        "approved": False,
    }


def run_semantic_settle_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] = CHECKPOINT.run_checkpoint_smoke,
) -> dict[str, Any]:
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("semantic settle diagnostic job mode가 다르다")
    if job.get("callback_variant") != VARIANT:
        raise ValueError("semantic settle diagnostic variant가 다르다")
    if not callable(seal_report):
        raise TypeError("semantic settle seal_report가 callable이 아니다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("semantic settle checkpoint가 regular file이 아니다")
    selected = (
        dict(dependencies)
        if dependencies is not None
        else PREPARED._load_dependencies()
    )
    callbacks: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    capture_on_play: dict[str, Any] = {}
    settles: list[dict[str, Any]] = []
    marker_states: list[dict[str, Any]] = []
    deps = _instrument_dependencies(
        selected,
        callbacks,
        operations,
        capture_on_play,
        settles,
        marker_states,
        job["profile"],
    )
    inner = dict(validated)
    inner["job"] = {**dict(job), "mode": CHECKPOINT.MODE}
    underlying: dict[str, Any] | None = None
    primary: BaseException | None = None

    def capture_underlying(value: Mapping[str, Any]) -> None:
        nonlocal underlying
        underlying = deepcopy(dict(value))

    try:
        result = run_sdk(inner, app, seal_report=capture_underlying, dependencies=deps)
        underlying = deepcopy(result)
    except BaseException as error:
        raw = getattr(error, "envelope", getattr(error, "report", None))
        if isinstance(raw, Mapping):
            underlying = deepcopy(dict(raw))
        primary = getattr(error, "primary", error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            boundary = BOUNDARY._report(
                "manual_schedule",
                underlying,
                callbacks,
                operations,
                capture_on_play,
                primary,
            )
            report = _report(boundary, settles, marker_states, primary)
            error.__dict__["semantic_settle_diagnostic_envelope"] = report
            raise
    try:
        BOUNDARY._restore_capture_on_play(capture_on_play, operations)
    except Exception as restore_error:  # noqa: BLE001 - 최초 오류를 보존한다.
        if primary is None:
            primary = restore_error
        else:
            primary.__dict__["semantic_settle_restore_error"] = {
                "error_type": _type_name(restore_error),
                "error_message": str(restore_error),
            }
    boundary = BOUNDARY._report(
        "manual_schedule",
        underlying,
        callbacks,
        operations,
        capture_on_play,
        primary,
    )
    report = _report(boundary, settles, marker_states, primary)
    seal_report(report)
    if report["diagnostic_complete"] is not True:
        raise SemanticSettleDiagnosticFailure(
            report, primary or RuntimeError("semantic settle 진단 증거가 불완전하다")
        )
    return report
