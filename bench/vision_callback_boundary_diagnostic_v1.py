"""Replicator trigger와 render step 사이의 Writer callback 경계를 계측한다."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import vision_callback_fingerprint_v1 as FINGERPRINT
import vision_checkpoint_smoke_sdk_v2 as CHECKPOINT
import vision_prepared_sdk_v1 as PREPARED
import vision_replicator_startup_v1 as STARTUP

SCHEMA = "vision_callback_boundary_diagnostic_v1"
MODE = "checkpoint_callback_boundary_diagnostic_only"
VARIANTS = frozenset({"custom_event", "manual_schedule"})
TOP_KEYS = frozenset(
    {
        "schema",
        "variant",
        "underlying",
        "callbacks",
        "operations",
        "capture_on_play",
        "primary_error",
        "diagnostic_complete",
        "used_for_decision",
        "approved",
    }
)


class CallbackBoundaryDiagnosticFailure(RuntimeError):
    """진단 자체가 유효한 callback 경계 증거를 만들지 못했음을 보존한다."""

    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(
            f"callback boundary diagnostic failed: {type(primary).__name__}: {primary}"
        )
        self.report = report
        self.envelope = report
        self.primary = primary


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _small(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 8:
        return [_small(item) for item in value]
    if (
        isinstance(value, Mapping)
        and len(value) <= 8
        and all(isinstance(key, str) for key in value)
    ):
        return {key: _small(item) for key, item in sorted(value.items())}
    return {"type": _type_name(value)}


def _frame_number(payload: Any) -> Any:
    try:
        rows = payload["annotators"]["IsaacReadTimes"]
        if not isinstance(rows, Mapping) or len(rows) != 1:
            return None
        value = next(iter(rows.values()))
        return value.get("frameNumber") if isinstance(value, Mapping) else None
    except (KeyError, TypeError):
        return None


def _callback_row(
    payload: Any,
    mux: Any,
    ordinal: int,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    fingerprint = FINGERPRINT.fingerprint(payload)
    phase = getattr(mux, "_phase", None)
    marker_code = phase.get("marker_code") if isinstance(phase, Mapping) else None
    components = fingerprint["components"]
    return {
        "ordinal": ordinal,
        "monotonic_ns": time.monotonic_ns(),
        "thread_id": threading.get_ident(),
        "window": getattr(mux, "state", None),
        "marker_code": marker_code,
        "reference_time": _small(payload.get("reference_time")),
        "frame_number": _frame_number(payload),
        "raw_callback_count_before": getattr(mux, "raw_callback_count", None),
        "raw_payload_sha256": fingerprint["whole_payload_sha256"],
        "trigger_outputs": deepcopy(components["trigger_outputs"]),
        "og_node_metadata": {
            "distribution_outputs": deepcopy(components["distribution_outputs"]),
            "named_outputs": deepcopy(components["named_outputs"]),
            "writer_swh_frame_number": deepcopy(components["writer_swh_frame_number"]),
            "render_product_identity": deepcopy(components["render_product_identity"]),
        },
        "operation_counts_before": _operation_counts(operations),
    }


def _operation_counts(operations: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("send_og_event", "schedule_write", "orchestrator_step")
    return {
        name: {
            "started": sum(row["name"] == name for row in operations),
            "complete": sum(
                row["name"] == name and row["status"] == "complete"
                for row in operations
            ),
            "failed": sum(
                row["name"] == name and row["status"] == "failed" for row in operations
            ),
        }
        for name in names
    }


def _record_operation(
    operations: list[dict[str, Any]],
    name: str,
    call: Callable[[], Any],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> Any:
    row = {
        "ordinal": len(operations),
        "name": name,
        "monotonic_ns": time.monotonic_ns(),
        "thread_id": threading.get_ident(),
        "status": "started",
        "error_type": None,
        "evidence": deepcopy(dict(evidence)) if evidence is not None else {},
    }
    operations.append(row)
    try:
        result = call()
    except BaseException as error:
        row["status"] = "failed"
        row["error_type"] = _type_name(error)
        raise
    row["status"] = "complete"
    return result


class _UtilsProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def send_og_event(self, name: str) -> Any:
        return _record_operation(
            self._operations,
            "send_og_event",
            lambda: self._value.send_og_event(name),
            evidence={"event_name": name},
        )


class _OrchestratorProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self._operations = operations

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return _record_operation(
            self._operations,
            "orchestrator_step",
            lambda: self._value.step(*args, **kwargs),
            evidence={
                "args": [_small(item) for item in args],
                "kwargs": _small(kwargs),
            },
        )


class _RepProxy:
    def __init__(self, value: Any, operations: list[dict[str, Any]]) -> None:
        self._value = value
        self.utils = _UtilsProxy(value.utils, operations)
        self.orchestrator = _OrchestratorProxy(value.orchestrator, operations)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)


class _WriterProxy:
    def __init__(
        self,
        writer: Any,
        mux: Any,
        variant: str,
        callbacks: list[dict[str, Any]],
        operations: list[dict[str, Any]],
    ) -> None:
        self._writer = writer
        self._variant = variant
        self._operations = operations
        original_write = writer.write

        def observed_write(payload: Any) -> Any:
            try:
                row = _callback_row(payload, mux, len(callbacks), operations)
                row["instrumentation_error"] = None
            except Exception as error:  # noqa: BLE001 - 원 Writer 호출을 보존한다.
                row = {
                    "ordinal": len(callbacks),
                    "monotonic_ns": time.monotonic_ns(),
                    "thread_id": threading.get_ident(),
                    "window": getattr(mux, "state", None),
                    "instrumentation_error": {
                        "error_type": _type_name(error),
                        "error_message": str(error),
                    },
                    "operation_counts_before": _operation_counts(operations),
                }
            callbacks.append(row)
            try:
                result = original_write(payload)
            finally:
                row["raw_callback_count_after"] = getattr(
                    mux, "raw_callback_count", None
                )
                row["mux_failure"] = deepcopy(getattr(mux, "_failure", None))
                row["operation_counts_after"] = _operation_counts(operations)
            return result

        writer.write = observed_write

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)

    def attach(self, render_product: Any, *, trigger: Any) -> Any:
        selected_trigger = trigger if self._variant == "custom_event" else None
        return _record_operation(
            self._operations,
            "writer_attach",
            lambda: self._writer.attach(render_product, trigger=selected_trigger),
            evidence={
                "variant": self._variant,
                "trigger_is_none": selected_trigger is None,
                "trigger_type": _type_name(selected_trigger),
            },
        )

    def detach(self) -> Any:
        return _record_operation(self._operations, "writer_detach", self._writer.detach)

    def schedule_write(self) -> Any:
        return _record_operation(
            self._operations, "schedule_write", self._writer.schedule_write
        )


def _setting(settings: Any) -> Any:
    return settings.get(STARTUP.CAPTURE_ON_PLAY)


def _instrument_dependencies(
    dependencies: Mapping[str, Any],
    variant: str,
    callbacks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    capture_on_play: dict[str, Any],
) -> dict[str, Any]:
    if set(dependencies) != PREPARED._DEPENDENCY_KEYS:
        raise ValueError("callback diagnostic dependency 표면이 다르다")
    deps = dict(dependencies)
    original_rep = deps["rep"]
    rep = _RepProxy(original_rep, operations)
    deps["rep"] = rep
    original_writer = deps["writer"]
    original_settings = deps["settings"]
    original_startup = deps["startup"]
    original_render = deps["capture_render_step"]
    writer_holder: dict[str, _WriterProxy] = {}

    def settings_call() -> Any:
        selected = original_settings()
        capture_on_play["before"] = _setting(selected)
        if type(capture_on_play["before"]) is not bool:
            raise TypeError("callback diagnostic 시작 captureOnPlay가 bool이 아니다")
        capture_on_play["settings"] = selected
        return selected

    def startup_call(selected_rep: Any, settings: Any) -> Any:
        result = original_startup(selected_rep, settings)
        capture_on_play["after_startup"] = _setting(settings)
        if capture_on_play["after_startup"] is not False:
            raise RuntimeError("callback diagnostic captureOnPlay가 false가 아니다")
        return result

    def writer_call(selected_rep: Any, mux: Any) -> _WriterProxy:
        writer = original_writer(selected_rep, mux)
        proxy = _WriterProxy(writer, mux, variant, callbacks, operations)
        if writer_holder:
            raise RuntimeError("callback diagnostic Writer가 두 번 생성됐다")
        writer_holder["writer"] = proxy
        return proxy

    def render_call(orchestrator: Any, profile: Any) -> Any:
        if variant == "manual_schedule":
            writer = writer_holder.get("writer")
            if writer is None:
                raise RuntimeError("callback diagnostic schedule 대상 Writer가 없다")
            writer.schedule_write()
        return original_render(orchestrator, profile)

    deps["settings"] = settings_call
    deps["startup"] = startup_call
    deps["writer"] = writer_call
    deps["capture_render_step"] = render_call
    return deps


def _restore_capture_on_play(
    capture_on_play: dict[str, Any], operations: list[dict[str, Any]]
) -> None:
    settings = capture_on_play.pop("settings", None)
    if settings is None:
        capture_on_play["restored"] = None
        capture_on_play["restore_changed"] = False
        return
    current = _setting(settings)
    before = capture_on_play.get("before")
    capture_on_play["restore_changed"] = current != before
    if capture_on_play["restore_changed"]:

        def restore() -> None:
            settings.set_bool(STARTUP.CAPTURE_ON_PLAY, before)

        _record_operation(operations, "capture_on_play_restore", restore)
    capture_on_play["restored"] = _setting(settings)
    if capture_on_play["restored"] != before:
        raise RuntimeError("callback diagnostic captureOnPlay 복원이 실패했다")


def _report(
    variant: str,
    underlying: Mapping[str, Any] | None,
    callbacks: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    capture_on_play: Mapping[str, Any],
    primary: BaseException | None,
) -> dict[str, Any]:
    names = [row["name"] for row in operations]
    render_count = names.count("orchestrator_step")
    schedule_count = names.count("schedule_write")
    send_count = names.count("send_og_event")
    attach_rows = [row for row in operations if row["name"] == "writer_attach"]
    step_rows = [row for row in operations if row["name"] == "orchestrator_step"]
    expected_step = {
        "args": [],
        "kwargs": {
            "delta_time": 0.0,
            "pause_timeline": False,
            "rt_subframes": 1,
        },
    }
    manual_pairs = all(
        index > 0 and operations[index - 1]["name"] == "schedule_write"
        for index, row in enumerate(operations)
        if row["name"] == "orchestrator_step"
    )
    complete = (
        underlying is not None
        and bool(callbacks)
        and len(attach_rows) == 1
        and attach_rows[0]["status"] == "complete"
        and attach_rows[0]["evidence"].get("variant") == variant
        and attach_rows[0]["evidence"].get("trigger_is_none")
        is (variant == "manual_schedule")
        and render_count >= 1
        and send_count == render_count
        and all(row["evidence"] == expected_step for row in step_rows)
        and (
            variant != "manual_schedule"
            or (schedule_count == render_count and manual_pairs)
        )
        and (variant != "custom_event" or schedule_count == 0)
        and capture_on_play.get("after_startup") is False
        and capture_on_play.get("restored") == capture_on_play.get("before")
        and all(row.get("instrumentation_error") is None for row in callbacks)
    )
    return {
        "schema": SCHEMA,
        "variant": variant,
        "underlying": deepcopy(dict(underlying)) if underlying is not None else None,
        "callbacks": deepcopy(callbacks),
        "operations": {
            "timeline": deepcopy(operations),
            "counts": _operation_counts(operations),
        },
        "capture_on_play": deepcopy(dict(capture_on_play)),
        "primary_error": (
            None
            if primary is None
            else {"error_type": _type_name(primary), "error_message": str(primary)}
        ),
        "diagnostic_complete": complete,
        "used_for_decision": False,
        "approved": False,
    }


def run_boundary_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    variant: str,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] = CHECKPOINT.run_checkpoint_smoke,
) -> dict[str, Any]:
    """두 등록 trigger 변형을 strict mux 변경 없이 관찰한다."""
    job = validated.get("job")
    if not isinstance(job, Mapping) or job.get("mode") != MODE:
        raise ValueError("callback diagnostic job mode가 다르다")
    if variant not in VARIANTS:
        raise ValueError("callback diagnostic variant가 다르다")
    if not callable(seal_report):
        raise TypeError("callback diagnostic seal_report가 callable이 아니다")
    checkpoint = Path(validated.get("checkpoint_path", ""))
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("callback diagnostic checkpoint가 regular file이 아니다")

    selected = (
        dict(dependencies)
        if dependencies is not None
        else PREPARED._load_dependencies()
    )
    callbacks: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    capture_on_play: dict[str, Any] = {}
    deps = _instrument_dependencies(
        selected, str(variant), callbacks, operations, capture_on_play
    )
    inner = dict(validated)
    inner["job"] = {**dict(job), "mode": CHECKPOINT.MODE}
    underlying: dict[str, Any] | None = None
    primary: BaseException | None = None

    def capture_underlying(value: Mapping[str, Any]) -> None:
        nonlocal underlying
        underlying = deepcopy(dict(value))

    try:
        result = run_sdk(
            inner,
            app,
            seal_report=capture_underlying,
            dependencies=deps,
        )
        underlying = deepcopy(result)
    except BaseException as error:
        raw = getattr(error, "envelope", getattr(error, "report", None))
        if isinstance(raw, Mapping):
            underlying = deepcopy(dict(raw))
        primary = getattr(error, "primary", error)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            report = _report(
                str(variant),
                underlying,
                callbacks,
                operations,
                capture_on_play,
                primary,
            )
            error.__dict__["callback_boundary_diagnostic_envelope"] = report
            raise
    try:
        _restore_capture_on_play(capture_on_play, operations)
    except Exception as restore_error:  # noqa: BLE001 - 첫 SDK 오류를 보존한다.
        if primary is None:
            primary = restore_error
        else:
            primary.__dict__["callback_boundary_restore_error"] = {
                "error_type": _type_name(restore_error),
                "error_message": str(restore_error),
            }
    report = _report(
        str(variant), underlying, callbacks, operations, capture_on_play, primary
    )
    seal_report(report)
    if report["diagnostic_complete"] is not True:
        raise CallbackBoundaryDiagnosticFailure(
            report, primary or RuntimeError("callback boundary 진단 증거가 불완전하다")
        )
    return report
