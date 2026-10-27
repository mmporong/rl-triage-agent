"""Native Kit update counter를 요구하는 annotator pull checkpoint 경계."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import vision_annotator_pull_checkpoint_v1 as V1
import vision_annotator_pull_worker_v2 as WORKER

SCHEMA = "vision_annotator_pull_checkpoint_v2"
MODE = V1.MODE
VARIANT = V1.VARIANT
TOP_KEYS = V1.TOP_KEYS

_camera_profile = V1._camera_profile
_write_raw_artifacts = V1._write_raw_artifacts


class AnnotatorPullDiagnosticFailure(RuntimeError):
    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"annotator pull diagnostic failed: {type(primary).__name__}: {primary}")
        self.report = report
        self.envelope = report
        self.primary = primary


def _native_counter_receipt(report: Mapping[str, Any]) -> dict[str, Any]:
    worker = report.get("worker")
    setup = worker.get("setup") if isinstance(worker, Mapping) else None
    receipt = setup.get("app_update_counter") if isinstance(setup, Mapping) else None
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "source",
        "method",
        "initial_value",
        "after_extension_update",
        "fail_closed",
    }:
        raise ValueError("native app update counter receipt가 없다")
    initial = receipt["initial_value"]
    after_extension = receipt["after_extension_update"]
    if (
        receipt["source"] != "omni.kit.app.IApp.get_update_number"
        or receipt["method"] != "app.get_update_number()"
        or receipt["fail_closed"] is not True
        or isinstance(initial, bool)
        or not isinstance(initial, int)
        or isinstance(after_extension, bool)
        or not isinstance(after_extension, int)
        or initial < 0
        or after_extension <= initial
    ):
        raise ValueError("native app update counter receipt가 계약과 다르다")
    return deepcopy(dict(receipt))


def _upgrade(value: Mapping[str, Any]) -> dict[str, Any]:
    report = deepcopy(dict(value))
    report["schema"] = SCHEMA
    if report.get("diagnostic_complete") is True:
        _native_counter_receipt(report)
    return report


def run_annotator_pull_diagnostic(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
    run_worker: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """v1 raw 봉인을 재사용하고 worker v2를 명시적으로 연결한다."""
    if not callable(seal_report):
        raise TypeError("annotator pull seal_report가 callable이 아니다")
    emitted: dict[str, Any] | None = None

    def seal_v2(value: Mapping[str, Any]) -> None:
        nonlocal emitted
        emitted = _upgrade(value)
        seal_report(emitted)

    selected_worker = run_worker or WORKER.run_worker
    try:
        V1.run_annotator_pull_diagnostic(
            validated,
            app,
            seal_report=seal_v2,
            dependencies=dependencies,
            run_worker=selected_worker,
        )
    except V1.AnnotatorPullDiagnosticFailure as error:
        report = emitted or _upgrade(error.report)
        raise AnnotatorPullDiagnosticFailure(report, error.primary) from error
    if emitted is None:
        error = RuntimeError("annotator pull v2 report가 봉인되지 않았다")
        raise AnnotatorPullDiagnosticFailure(_upgrade({}), error) from error
    return emitted
