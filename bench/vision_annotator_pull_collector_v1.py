"""Writer 없이 Replicator annotator 네 개를 연속 pull하는 수집 경계 초안."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import vision_play_worker_v1 as PLAY
import vision_prepared_capture_v1 as PREPARED

SCHEMA = "vision_annotator_pull_capture_v1"
REPORT_SCHEMA = "vision_annotator_pull_collection_report_v1"
ANNOTATORS = ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes")
AABA = (("A", 0), ("A", 1), ("B", 2), ("A", 3))


class AnnotatorPullFailure(RuntimeError):
    """첫 수집 오류와 cleanup 결과를 함께 보존한다."""

    def __init__(self, report: dict[str, Any], primary: BaseException) -> None:
        super().__init__(f"annotator pull failed: {_type_name(primary)}")
        self.report = report
        self.primary = primary


def _type_name(value: BaseException) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _physics(
    physics_snapshot: Callable[[], Mapping[str, Any]], active_base: Any
) -> dict[str, Any]:
    snapshot, simulation_time_s, digest = PREPARED._physics(
        physics_snapshot(), active_base.sim.current_time
    )
    return {
        "snapshot": snapshot,
        "simulation_time_s": simulation_time_s,
        "sha256": digest,
    }


def _annotators(rep: Any) -> list[tuple[str, Any]]:
    return [
        ("rgb", rep.AnnotatorRegistry.get_annotator("rgb")),
        (
            "semantic_segmentation",
            rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"colorize": False}
            ),
        ),
        ("CameraParams", rep.AnnotatorRegistry.get_annotator("CameraParams")),
        ("IsaacReadTimes", rep.AnnotatorRegistry.get_annotator("IsaacReadTimes")),
    ]


def _arm_receipt(
    value: Any, *, ordinal: int, camera_kind: str, marker_code: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("arm 준비 receipt는 mapping이어야 한다")
    copied = PREPARED._snapshot(dict(value))
    if copied.get("ordinal") != ordinal:
        raise ValueError("arm 준비 receipt ordinal이 다르다")
    if copied.get("camera_kind") != camera_kind:
        raise ValueError("arm 준비 receipt camera kind가 다르다")
    if copied.get("marker_code") != marker_code:
        raise ValueError("arm 준비 receipt marker code가 다르다")
    return copied


def _failure_report(
    trace: Sequence[Mapping[str, Any]],
    captures: Sequence[Mapping[str, Any]],
    partial_capture: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "failed",
        "capture_count": len(captures),
        "captures": list(captures),
        "partial_capture": (
            PREPARED._snapshot(dict(partial_capture))
            if partial_capture is not None
            else None
        ),
        "operation_trace": list(trace),
        "cleanup_errors": [],
        "atomicity_claimed": False,
    }


def collect_aaba(
    *,
    rep: Any,
    render_product: Any,
    orchestrator: Any,
    camera_profile: Mapping[str, Any],
    active_base: Any,
    physics_snapshot: Callable[[], Mapping[str, Any]],
    prepare_arm: Callable[[int, str, int], Mapping[str, Any]],
    app_update_count: Callable[[], int],
) -> dict[str, Any]:
    """네 annotator를 attach한 뒤 AABA arm마다 step 1회와 연속 pull을 수행한다."""
    if not isinstance(camera_profile, Mapping):
        raise TypeError("camera profile은 mapping이어야 한다")
    for key in ("rt_subframes", "pause_timeline", "delta_time_s"):
        if key not in camera_profile:
            raise ValueError(f"camera profile에 {key}가 없다")
    if camera_profile["delta_time_s"] != 0.0:
        raise ValueError("annotator pull 후보는 delta_time_s=0.0만 허용한다")
    if not callable(app_update_count):
        raise TypeError("app update counter가 필요하다")

    pairs = _annotators(rep)
    attached: list[tuple[str, Any]] = []
    trace: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    primary: BaseException | None = None
    report: dict[str, Any] | None = None
    partial_capture: dict[str, Any] | None = None
    try:
        for name, annotator in pairs:
            if annotator is None:
                raise RuntimeError(f"{name} annotator 등록값이 없다")
            annotator.attach(render_product)
            attached.append((name, annotator))
            trace.append({"operation": "attach", "annotator": name})

        for ordinal, (camera_kind, marker_code) in enumerate(AABA):
            state = _arm_receipt(
                prepare_arm(ordinal, camera_kind, marker_code),
                ordinal=ordinal,
                camera_kind=camera_kind,
                marker_code=marker_code,
            )
            before = _physics(physics_snapshot, active_base)
            updates_before_render = app_update_count()
            trace.append({"operation": "step", "ordinal": ordinal})
            PLAY._capture_render_step(orchestrator, camera_profile)
            updates_after_render = app_update_count()
            raw: dict[str, Any] = {}
            partial_capture = {
                "ordinal": ordinal,
                "arm": state,
                "physics_before": before,
                "raw": raw,
            }
            pull_order: list[str] = []
            for name, annotator in pairs:
                value = annotator.get_data(do_array_copy=True)
                raw[name] = PREPARED._snapshot(value)
                pull_order.append(name)
                trace.append(
                    {"operation": "pull", "ordinal": ordinal, "annotator": name}
                )
            updates_after_pull = app_update_count()
            after = _physics(physics_snapshot, active_base)
            captures.append(
                {
                    "schema": SCHEMA,
                    "ordinal": ordinal,
                    "render_product": str(render_product),
                    "arm": state,
                    "step": {
                        "call_count": 1,
                        "delta_time_s": camera_profile["delta_time_s"],
                        "pause_timeline": camera_profile["pause_timeline"],
                        "rt_subframes": camera_profile["rt_subframes"],
                    },
                    "pull_order": pull_order,
                    "do_array_copy": True,
                    "app_update_counts": {
                        "before_render": updates_before_render,
                        "after_render_before_pull": updates_after_render,
                        "after_pull": updates_after_pull,
                    },
                    "render_window_app_updates": (
                        updates_after_render - updates_before_render
                    ),
                    "pull_window_app_updates": (
                        updates_after_pull - updates_after_render
                    ),
                    "physics_before": before,
                    "physics_after": after,
                    "raw": raw,
                    "atomicity_claimed": False,
                }
            )
            partial_capture = None
        report = {
            "schema": REPORT_SCHEMA,
            "status": "collected_unvalidated",
            "capture_count": len(captures),
            "captures": captures,
            "partial_capture": None,
            "operation_trace": trace,
            "cleanup_errors": [],
            "atomicity_claimed": False,
        }
    except BaseException as error:  # noqa: BLE001 - first failure must survive cleanup.
        primary = error
        report = _failure_report(trace, captures, partial_capture)
    finally:
        cleanup_errors: list[dict[str, str]] = []
        for name, annotator in reversed(attached):
            try:
                annotator.detach(render_product)
                trace.append({"operation": "detach", "annotator": name})
            except BaseException as error:  # noqa: BLE001 - preserve cleanup evidence.
                cleanup_errors.append(
                    {"annotator": name, "error_type": _type_name(error)}
                )
                if primary is None:
                    primary = error
        if report is None:
            report = _failure_report(trace, captures, partial_capture)
        report["operation_trace"] = trace
        report["cleanup_errors"] = cleanup_errors

    if primary is not None:
        report["status"] = "failed"
        raise AnnotatorPullFailure(report, primary) from primary
    return report
