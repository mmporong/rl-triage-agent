"""기본 OgnOnFrame Writer와 renderer step 하나로 checkpoint smoke를 계측한다."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import vision_callback_boundary_diagnostic_v2 as BOUNDARY
import vision_checkpoint_smoke_sdk_v2 as CHECKPOINT
import vision_prepared_sdk_v1 as PREPARED

SCHEMA = "vision_frame_trigger_checkpoint_v1"
MODE = "checkpoint_frame_trigger_diagnostic_only"
VARIANT = "default_frame_step"
TOP_KEYS = frozenset(
    {
        "schema",
        "variant",
        "underlying",
        "callbacks",
        "operations",
        "capture_on_play",
        "dispatch_counts",
        "primary_error",
        "diagnostic_complete",
        "used_for_decision",
        "approved",
    }
)


class FrameTriggerDiagnosticFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(
            f"frame trigger diagnostic failed: {type(primary).__name__}: {primary}"
        )
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


class _NoEventUtilsProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def send_og_event(self, name: str) -> None:
        self._operations.append(
            {
                "ordinal": len(self._operations),
                "name": "send_og_event_suppressed",
                "monotonic_ns": time.monotonic_ns(),
                "thread_id": threading.get_ident(),
                "status": "complete",
                "error_type": None,
                "evidence": {"event_name": name, "underlying_calls": 0},
            }
        )


class _NoEventTriggerProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def on_custom_event(self, name: str) -> object:
        self._operations.append(
            {
                "ordinal": len(self._operations),
                "name": "on_custom_event_suppressed",
                "monotonic_ns": time.monotonic_ns(),
                "thread_id": threading.get_ident(),
                "status": "complete",
                "error_type": None,
                "evidence": {"event_name": name, "underlying_calls": 0},
            }
        )
        return object()


class _RepProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self.utils = _NoEventUtilsProxy(value.utils, operations)
        self.trigger = _NoEventTriggerProxy(value.trigger, operations)
        self.orchestrator = BOUNDARY._OrchestratorProxy(value.orchestrator, operations)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)


class _DefaultFrameWriterProxy:
    def __init__(
        self,
        writer: Any,
        mux: Any,
        callbacks: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> None:
        self._observed = BOUNDARY._WriterProxy(
            writer, mux, "custom_event", callbacks, operations
        )
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._observed, name)

    def attach(self, render_product: Any, *, trigger: Any) -> Any:
        return BOUNDARY._record_operation(
            self._operations,
            "writer_attach",
            lambda: self._observed._writer.attach(render_product),
            evidence={
                "variant": VARIANT,
                "sdk_trigger_argument_received": _type_name(trigger),
                "underlying_trigger_argument_supplied": False,
                "default_trigger": "OgnOnFrame",
            },
        )

    def schedule_write(self) -> Any:
        raise RuntimeError(
            "default frame trigger 후보는 schedule_write를 허용하지 않는다"
        )


def _setting(settings: Any) -> Any:
    return settings.get(BOUNDARY.STARTUP.CAPTURE_ON_PLAY)


def _instrument_dependencies(
    dependencies: Mapping[str, Any],
    callbacks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    capture_on_play: dict[str, Any],
) -> dict[str, Any]:
    if set(dependencies) != PREPARED._DEPENDENCY_KEYS:
        raise ValueError("frame trigger dependency 표면이 다르다")
    deps = dict(dependencies)
    raw_rep = deps["rep"]
    deps["rep"] = _RepProxy(raw_rep, operations)
    raw_writer = deps["writer"]
    raw_settings = deps["settings"]
    raw_startup = deps["startup"]
    raw_render = deps["capture_render_step"]

    def settings_call() -> Any:
        selected = raw_settings()
        capture_on_play["before"] = _setting(selected)
        if type(capture_on_play["before"]) is not bool:
            raise TypeError("frame trigger 시작 captureOnPlay가 bool이 아니다")
        capture_on_play["settings"] = selected
        return selected

    def startup_call(selected_rep: Any, settings: Any) -> Any:
        result = raw_startup(selected_rep, settings)
        capture_on_play["after_startup"] = _setting(settings)
        if capture_on_play["after_startup"] is not False:
            raise RuntimeError("frame trigger captureOnPlay가 false가 아니다")
        return result

    def writer_call(selected_rep: Any, mux: Any) -> _DefaultFrameWriterProxy:
        return _DefaultFrameWriterProxy(
            raw_writer(selected_rep, mux), mux, callbacks, operations
        )

    def render_call(orchestrator: Any, profile: Any) -> Any:
        return raw_render(orchestrator, profile)

    deps["settings"] = settings_call
    deps["startup"] = startup_call
    deps["writer"] = writer_call
    deps["capture_render_step"] = render_call
    return deps


def _dispatch_counts(operations: list[dict[str, Any]]) -> dict[str, int]:
    names = [row.get("name") for row in operations]
    return {
        "on_custom_event_requests_suppressed": names.count(
            "on_custom_event_suppressed"
        ),
        "on_custom_event_underlying_calls": names.count("on_custom_event"),
        "send_og_event_requests_suppressed": names.count("send_og_event_suppressed"),
        "send_og_event_underlying_calls": names.count("send_og_event"),
        "schedule_write_calls": names.count("schedule_write"),
        "orchestrator_step_calls": names.count("orchestrator_step"),
    }


def _report(
    underlying: Mapping[str, Any] | None,
    callbacks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    capture_on_play: Mapping[str, Any],
    primary: BaseException | None,
) -> dict[str, Any]:
    attach_rows = [row for row in operations if row.get("name") == "writer_attach"]
    step_rows = [row for row in operations if row.get("name") == "orchestrator_step"]
    counts = _dispatch_counts(operations)
    probe = underlying.get("policy_probe") if isinstance(underlying, Mapping) else None
    expected_step = {
        "args": [],
        "kwargs": {
            "delta_time": 0.0,
            "pause_timeline": False,
            "rt_subframes": 1,
        },
    }
    complete = (
        primary is None
        and isinstance(underlying, Mapping)
        and underlying.get("approved") is True
        and isinstance(probe, Mapping)
        and probe.get("environment_steps") == 1
        and probe.get("capture_completed_before_policy_step") is True
        and probe.get("writer_detached_before_policy_step") is True
        and len(attach_rows) == 1
        and attach_rows[0].get("status") == "complete"
        and attach_rows[0]
        .get("evidence", {})
        .get("underlying_trigger_argument_supplied")
        is False
        and attach_rows[0].get("evidence", {}).get("default_trigger") == "OgnOnFrame"
        and counts
        == {
            "on_custom_event_requests_suppressed": 1,
            "on_custom_event_underlying_calls": 0,
            "send_og_event_requests_suppressed": 5,
            "send_og_event_underlying_calls": 0,
            "schedule_write_calls": 0,
            "orchestrator_step_calls": 5,
        }
        and len(step_rows) == 5
        and all(row.get("evidence") == expected_step for row in step_rows)
        and capture_on_play.get("after_startup") is False
        and capture_on_play.get("restored") == capture_on_play.get("before")
        and bool(callbacks)
        and all(BOUNDARY._callback_complete(row) for row in callbacks)
    )
    return {
        "schema": SCHEMA,
        "variant": VARIANT,
        "underlying": deepcopy(dict(underlying)) if underlying is not None else None,
        "callbacks": deepcopy(callbacks),
        "operations": {
            "timeline": deepcopy(operations),
            "counts": BOUNDARY._operation_counts(operations),
        },
        "capture_on_play": deepcopy(dict(capture_on_play)),
        "dispatch_counts": counts,
        "primary_error": (
            None
            if primary is None
            else {"error_type": _type_name(primary), "error_message": str(primary)}
        ),
        "diagnostic_complete": complete,
        "used_for_decision": False,
        "approved": False,
    }


def run_frame_trigger_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] = CHECKPOINT.run_checkpoint_smoke,
) -> dict[str, Any]:
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("frame trigger diagnostic job mode가 다르다")
    if job.get("callback_variant") != VARIANT:
        raise ValueError("frame trigger diagnostic variant가 다르다")
    if not callable(seal_report):
        raise TypeError("frame trigger seal_report가 callable이 아니다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("frame trigger checkpoint가 regular file이 아니다")
    selected = (
        dict(dependencies)
        if dependencies is not None
        else PREPARED._load_dependencies()
    )
    callbacks: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    capture_on_play: dict[str, Any] = {}
    deps = _instrument_dependencies(selected, callbacks, operations, capture_on_play)
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
            report = _report(
                underlying, callbacks, operations, capture_on_play, primary
            )
            error.__dict__["frame_trigger_diagnostic_envelope"] = report
            raise
    try:
        BOUNDARY._restore_capture_on_play(capture_on_play, operations)
    except Exception as restore_error:  # noqa: BLE001 - 최초 오류를 보존한다.
        if primary is None:
            primary = restore_error
        else:
            primary.__dict__["frame_trigger_restore_error"] = {
                "error_type": _type_name(restore_error),
                "error_message": str(restore_error),
            }
    report = _report(underlying, callbacks, operations, capture_on_play, primary)
    seal_report(report)
    if report["diagnostic_complete"] is not True:
        raise FrameTriggerDiagnosticFailure(
            report, primary or RuntimeError("frame trigger 진단 증거가 불완전하다")
        )
    return report
