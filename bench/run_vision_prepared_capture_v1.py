"""준비 callback과 네 측정 frame의 소유 실행·종료 증거를 검증한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD  # pyright: ignore[reportMissingImports]
import vision_atomic_smoke_v1 as SMOKE  # pyright: ignore[reportMissingImports]
import vision_brev_backend_v1 as OWNED  # pyright: ignore[reportMissingImports]
import vision_brev_render_backend_v2 as GATE  # pyright: ignore[reportMissingImports]
import vision_capture_readiness_v1 as READY  # pyright: ignore[reportMissingImports]
import vision_fxaa_policy_v1 as FXAA  # pyright: ignore[reportMissingImports]
import vision_prepared_job_v1 as JOB  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]
import vision_shutdown_policy_v1 as SHUTDOWN  # pyright: ignore[reportMissingImports]

REPORT_NAME = "prepared-capture.report.json"
EXECUTION_SCHEMA = "vision_prepared_capture_execution_v1"
POSTFLIGHT_RESERVE_S = 200.0
OWNED_CLEANUP_WORST_CASE_S = 40.0
LIFECYCLE_FLAGS = frozenset(
    {
        "setup_completed",
        "writer_attach_attempted",
        "writer_attached",
        "prepare_completed",
        "arms_completed",
        "writer_detach_attempted",
        "writer_detached",
        "render_product_destroyed",
        "environment_closed",
        "mux_resolved",
        "mux_sealed",
        "journals_written",
        "capture_report_written",
        "report_verified",
        "envelope_sealed",
    }
)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return JOB._read(path)


def _regular(path: Path) -> Path:
    return JOB._regular(path.absolute(), "prepared capture 증거")


def _descriptor(path: Path) -> dict[str, Any]:
    target = _regular(path)
    return {"file": target.name, "sha256": _sha(target), "bytes": target.stat().st_size}


def _leaf(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label}는 leaf 이름이어야 한다")
    return value


def _artifact_paths(
    validated: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> tuple[Path, dict[str, Path]]:
    expected = {"report", "journals", "audits"}
    if set(artifacts) != expected:
        raise ValueError("prepared artifact 필드가 다르다")
    output = Path(validated["output_directory"])
    stem = Path(validated["job_path"]).stem
    report_leaf = _leaf(artifacts["report"], "smoke report")
    if report_leaf != f"{stem}.smoke-report.json":
        raise ValueError("prepared smoke report leaf가 job과 다르다")
    report = _regular(output / report_leaf)
    audit_leaves = artifacts["audits"]
    if not isinstance(audit_leaves, Mapping) or set(audit_leaves) != {
        "audit_rgb_1",
        "audit_rgb_2",
        "audit_rgb_3",
    }:
        raise ValueError("prepared audit RGB 필드가 다르다")
    audits = {
        key: _regular(output / _leaf(audit_leaves[key], key))
        for key in ("audit_rgb_1", "audit_rgb_2", "audit_rgb_3")
    }
    for ordinal in (1, 2, 3):
        if audit_leaves[f"audit_rgb_{ordinal}"] != (
            f"{stem}.prepared-audit-rgb-{ordinal}.rgb"
        ):
            raise ValueError("prepared audit RGB leaf가 job·ordinal과 다르다")
    journals = artifacts["journals"]
    if not isinstance(journals, list) or len(journals) != 4:
        raise ValueError("prepared journal은 네 ordinal이어야 한다")
    stage = Path(validated["job_path"]).parent
    for ordinal, row in enumerate(journals):
        if not isinstance(row, Mapping) or set(row) != {"ordinal", "json", "npz"}:
            raise ValueError("prepared journal descriptor가 다르다")
        if row["ordinal"] != ordinal:
            raise ValueError("prepared journal ordinal이 다르다")
        expected_json = f"{stem}.atomic-{ordinal:06d}.json"
        expected_npz = f"{stem}.atomic-{ordinal:06d}.npz"
        if row["json"] != expected_json or row["npz"] != expected_npz:
            raise ValueError("prepared journal leaf가 ordinal과 다르다")
        _regular(stage / _leaf(row["json"], "journal json"))
        _regular(stage / _leaf(row["npz"], "journal npz"))
    return report, audits


def _vector3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _setup_structure(
    setup: Any, prepared: Mapping[str, Any], capture_report: Mapping[str, Any]
) -> bool:
    expected = {
        "phase_order",
        "setup_physics_before_after",
        "extension",
        "startup",
        "registrations",
        "camera_poses",
        "raw_callback_count",
        "complete_drain_proven",
        "all_future_callbacks_collected",
    }
    if not isinstance(setup, Mapping) or set(setup) != expected:
        return False
    physics = setup["setup_physics_before_after"]
    if (
        not isinstance(physics, Mapping)
        or set(physics) != {"before", "after", "equal"}
        or not isinstance(physics["before"], Mapping)
        or not physics["before"]
        or physics["before"] != physics["after"]
        or physics["equal"] is not True
        or setup["extension"]
        != {
            "name": "isaacsim.core.nodes",
            "enable_result": True,
            "app_update_completed": True,
        }
        or not isinstance(setup["startup"], Mapping)
        or set(setup["startup"]) != {"before", "after"}
        or not all(
            isinstance(setup["startup"][key], Mapping) for key in ("before", "after")
        )
        or setup["registrations"]
        != dict.fromkeys(
            ("rgb", "semantic_segmentation", "CameraParams", "IsaacReadTimes"), True
        )
        or setup["raw_callback_count"] != prepared.get("raw_callback_count")
        or type(setup["raw_callback_count"]) is not int
        or setup["complete_drain_proven"] is not False
        or setup["all_future_callbacks_collected"] is not False
    ):
        return False
    phase_order = setup["phase_order"]
    essential = [
        "profile",
        "settings",
        "extension_enable",
        "extension_app_update",
        "replicator_startup",
        "annotator_registrations",
        "usd_context",
        "environment_config",
        "environment_create",
        "environment_reset",
        "command_configure",
        "setup_physics_before",
        "camera_configure",
        "render_product_create",
        "marker_root_position",
        "marker_create",
        "renderer_readback",
        "capture_readiness",
        "marker_bind",
        "setup_physics_after",
        "writer_attach",
        "prepare_marker_3",
        "prepare_camera_A",
        "prepare_event",
        "prepare_render",
    ]
    for ordinal in range(4):
        essential.extend(
            (
                f"arm_{ordinal}_marker",
                f"arm_{ordinal}_camera",
                f"arm_{ordinal}_event",
                f"arm_{ordinal}_render",
            )
        )
    essential.extend(
        (
            "writer_detach",
            "render_product_destroy",
            "environment_close",
            "mux_resolve",
            "mux_seal",
        )
    )
    if (
        not isinstance(phase_order, list)
        or any(not isinstance(item, str) or not item for item in phase_order)
        or any(phase_order.count(name) != 1 for name in essential)
        or [phase_order.index(name) for name in essential]
        != sorted(phase_order.index(name) for name in essential)
    ):
        return False
    poses = setup["camera_poses"]
    captures = capture_report.get("captures")
    pose_phases = ["prepare", "arm_0", "arm_1", "arm_2", "arm_3"]
    if (
        not isinstance(poses, list)
        or len(poses) != 5
        or not isinstance(captures, list)
        or len(captures) != 4
        or any(not isinstance(row, Mapping) for row in captures)
    ):
        return False
    for index, (pose, phase) in enumerate(zip(poses, pose_phases, strict=True)):
        if (
            not isinstance(pose, Mapping)
            or set(pose) != {"phase", "eye_world_m", "lookat_world_m"}
            or pose["phase"] != phase
            or not _vector3(pose["eye_world_m"])
            or not _vector3(pose["lookat_world_m"])
        ):
            return False
        if index == 0:
            if {
                "eye_world_m": pose["eye_world_m"],
                "lookat_world_m": pose["lookat_world_m"],
            } != captures[0].get("camera"):
                return False
        elif {
            "eye_world_m": pose["eye_world_m"],
            "lookat_world_m": pose["lookat_world_m"],
        } != captures[index - 1].get("camera"):
            return False
    return True


def _renderer_structure(validated: Mapping[str, Any], renderer: Any) -> bool:
    if not isinstance(renderer, Mapping):
        return False
    use_fxaa = FXAA.required(validated["job"])
    expected_keys = {
        "attached_hydra_engine_names",
        "carb_settings",
        "expected_capture_profile",
        "renderer_accumulation_reset",
    } | ({"render_policy"} if use_fxaa else set())
    camera = RECORD.fixed_profile()["camera"]
    profile = {
        key: camera[key]
        for key in (
            "capture_primitive",
            "rt_subframes",
            "pause_timeline",
            "delta_time_s",
        )
    }
    if (
        set(renderer) != expected_keys
        or not isinstance(renderer["attached_hydra_engine_names"], list)
        or not renderer["attached_hydra_engine_names"]
        or any(
            not isinstance(name, str) or not name
            for name in renderer["attached_hydra_engine_names"]
        )
        or not isinstance(renderer["carb_settings"], Mapping)
        or set(renderer["carb_settings"]) != set(READY.CARB_SETTING_KEYS)
        or renderer["expected_capture_profile"] != profile
        or renderer["renderer_accumulation_reset"]
        != "omni.usd.UsdContext.reset_renderer_accumulation"
    ):
        return False
    if use_fxaa:
        try:
            FXAA.verify_receipt(renderer["render_policy"])
        except (TypeError, ValueError):
            return False
        return (
            renderer["carb_settings"].get("/rtx/post/aa/op") == 2
            and renderer["carb_settings"].get("/rtx/rendermode")
            == renderer["render_policy"]["after"]["/rtx/rendermode"]
        )
    return True


def _prepared_structure(prepared: Any, capture_report: Mapping[str, Any]) -> bool:
    expected_keys = {
        "schema",
        "approved",
        "state",
        "raw_callback_count",
        "callback_rows_dropped",
        "preparation",
        "arm_sources",
        "callbacks",
        "bundles",
        "failure",
    }
    if (
        not isinstance(prepared, Mapping)
        or set(prepared) != expected_keys
        or prepared.get("approved") is not True
        or prepared.get("state") != "sealed"
        or prepared.get("failure") is not None
    ):
        return False
    raw_count = prepared["raw_callback_count"]
    callbacks = prepared["callbacks"]
    bundles = prepared["bundles"]
    sources = prepared["arm_sources"]
    captures = capture_report.get("captures")
    if (
        type(raw_count) is not int
        or not 5 <= raw_count <= 16
        or type(prepared["callback_rows_dropped"]) is not int
        or prepared["callback_rows_dropped"] != 0
        or not isinstance(callbacks, list)
        or len(callbacks) != raw_count
        or not isinstance(bundles, list)
        or len(bundles) != 4
        or any(not isinstance(bundle, Mapping) for bundle in bundles)
        or not isinstance(sources, list)
        or len(sources) != 4
        or not isinstance(captures, list)
        or len(captures) != 4
        or any(not isinstance(row, Mapping) for row in captures)
        or bundles != [row.get("atomic_bundle") for row in captures]
    ):
        return False
    preparation = prepared["preparation"]
    if (
        not isinstance(preparation, Mapping)
        or set(preparation)
        != {
            "logical_ordinal",
            "journal_written",
            "marker_code",
            "bootstrap_empty_observed",
            "valid_raw_index",
            "frame_number",
        }
        or preparation["logical_ordinal"] != 4
        or type(preparation["logical_ordinal"]) is not int
        or preparation["journal_written"] is not False
        or preparation["marker_code"] != 3
        or type(preparation["marker_code"]) is not int
        or type(preparation["bootstrap_empty_observed"]) is not bool
        or type(preparation["valid_raw_index"]) is not int
        or type(preparation["frame_number"]) is not int
    ):
        return False
    bootstrap = preparation["bootstrap_empty_observed"]
    prepare_count = 2 if bootstrap else 1
    if (
        raw_count != prepare_count + 4
        or preparation["valid_raw_index"] != prepare_count - 1
    ):
        return False
    callback_keys = {
        "raw_index",
        "window",
        "classification",
        "error_type",
        "descriptor",
        "frame_number",
    }
    if any(
        not isinstance(row, Mapping)
        or set(row) != callback_keys
        or type(row["raw_index"]) is not int
        or row["raw_index"] != index
        or not isinstance(row["descriptor"], Mapping)
        for index, row in enumerate(callbacks)
    ):
        return False
    prepare_rows = callbacks[:prepare_count]
    expected_classes = ["bootstrap_empty", "valid"] if bootstrap else ["valid"]
    if (
        [row["classification"] for row in prepare_rows] != expected_classes
        or any(row["window"] != "preparing" for row in prepare_rows)
        or prepare_rows[-1]["frame_number"] != preparation["frame_number"]
        or prepare_rows[-1]["error_type"] is not None
        or (
            bootstrap
            and (
                prepare_rows[0]["frame_number"] is not None
                or not isinstance(prepare_rows[0]["error_type"], str)
                or not prepare_rows[0]["error_type"]
            )
        )
    ):
        return False
    source_raw_indexes: list[int] = []
    for ordinal, (source, bundle, callback) in enumerate(
        zip(sources, bundles, callbacks[prepare_count:], strict=True)
    ):
        timing = bundle.get("timing")
        read_times = timing.get("read_times") if isinstance(timing, Mapping) else None
        frame = (
            read_times.get("frameNumber") if isinstance(read_times, Mapping) else None
        )
        if (
            not isinstance(source, Mapping)
            or set(source) != {"ordinal", "marker_code", "raw_index", "frame_number"}
            or any(type(source[key]) is not int for key in source)
            or source["ordinal"] != ordinal
            or source["marker_code"] != ordinal
            or source["raw_index"] != prepare_count + ordinal
            or type(frame) is not int
            or source["frame_number"] != frame
            or callback["raw_index"] != source["raw_index"]
            or callback["window"] != f"arming{ordinal}"
            or callback["classification"] != "valid"
            or callback["error_type"] is not None
            or callback["frame_number"] != frame
        ):
            return False
        source_raw_indexes.append(source["raw_index"])
    return (
        source_raw_indexes == sorted(set(source_raw_indexes))
        and sources[0]["frame_number"] == preparation["frame_number"] + 1
    )


def _successful_sdk_envelope(
    validated: Mapping[str, Any], value: Mapping[str, Any], *, sealed: bool
) -> bool:
    try:
        import vision_prepared_capture_v1 as PREPARED
        import vision_prepared_sdk_v1 as SDK

        if set(value) != {
            "schema",
            "prepared",
            "capture_report",
            "artifacts",
            "lifecycle",
            "renderer",
            "setup_evidence",
            "renderer_smoke_only",
            "approved",
        }:
            return False
        if (
            value.get("schema") != SDK.SDK_REPORT_SCHEMA
            or value.get("renderer_smoke_only") is not True
            or value.get("approved") is not True
        ):
            return False
        lifecycle = value.get("lifecycle")
        if (
            not isinstance(lifecycle, Mapping)
            or set(lifecycle) != LIFECYCLE_FLAGS | {"errors"}
            or lifecycle.get("errors") != []
            or any(
                lifecycle.get(flag) is not True
                for flag in LIFECYCLE_FLAGS - {"arms_completed", "envelope_sealed"}
            )
            or lifecycle.get("arms_completed") != 4
            or lifecycle.get("envelope_sealed") is not sealed
        ):
            return False
        prepared = value.get("prepared")
        capture_report = value.get("capture_report")
        if (
            not isinstance(prepared, Mapping)
            or prepared.get("schema") != PREPARED.SCHEMA
            or not isinstance(capture_report, Mapping)
            or capture_report.get("schema") != SMOKE.SCHEMA
            or capture_report.get("status") != "pass"
            or capture_report.get("capture_order") != list(SMOKE.ORDER)
            or capture_report.get("policy_steps") != 0
            or capture_report.get("capture_phase_physics_steps") != 0
            or not _prepared_structure(prepared, capture_report)
        ):
            return False
        if not _renderer_structure(
            validated, value.get("renderer")
        ) or not _setup_structure(
            value.get("setup_evidence"), prepared, capture_report
        ):
            return False
        report_path, audit_paths = _artifact_paths(validated, value["artifacts"])
        verified = SMOKE.verify_report(
            report_path, Path(validated["job_path"]), audit_paths
        )
        return dict(capture_report) == verified
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _terminal_body(
    validated: Mapping[str, Any], envelope: Mapping[str, Any], report: Path
) -> dict[str, Any]:
    return {
        "schema": JOB.RESULT_SCHEMA,
        "completed": True,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "capture_smoke_completed": True,
        "renderer_smoke_only": True,
        "report": _descriptor(report),
        "sdk": dict(envelope),
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
    }


def verify_result(validated: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(validated["output_directory"])
    for name in ("failure.json", "shutdown_failure.json"):
        if (output / name).exists() or (output / name).is_symlink():
            raise ValueError("prepared 성공과 실패 증거가 함께 있다")
    report_path = _regular(output / REPORT_NAME)
    result = _read(_regular(output / "result.json"))
    envelope = result.get("sdk")
    if not isinstance(envelope, Mapping) or not _successful_sdk_envelope(
        validated, envelope, sealed=True
    ):
        raise ValueError("prepared SDK lifecycle 또는 네 frame 검증이 완료되지 않았다")
    sealed = _read(report_path)
    if not _successful_sdk_envelope(validated, sealed, sealed=False):
        raise ValueError("prepared 봉인 보고서가 cleanup 후 경계와 다르다")
    expected = dict(envelope)
    expected["lifecycle"] = dict(envelope["lifecycle"])
    expected["lifecycle"]["envelope_sealed"] = False
    if sealed != expected:
        raise ValueError("prepared 봉인 원문이 terminal SDK 결과와 다르다")
    if result != _terminal_body(validated, envelope, report_path):
        raise ValueError("prepared terminal result의 job·원문·모드 결합이 다르다")
    SHUTDOWN.verify(Path(validated["job_path"]), validated, output / "result.json")
    return result


def sdk_child(
    job_path: Path,
    *,
    app_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    run_sdk: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """SDK close 전에 cleanup 완료 원문과 terminal 증거를 쓴다."""
    validated = JOB.validate_job(job_path)
    output = Path(validated["output_directory"])
    output.mkdir(exist_ok=False)
    app = None
    primary: BaseException | None = None
    report_path = output / REPORT_NAME
    try:
        if app_factory is None:
            import vision_play_worker_v3 as V3

            app_factory = V3._sdk_app_factory
        if run_sdk is None:
            import vision_prepared_sdk_v1 as SDK

            run_sdk = SDK.run_prepared_smoke
        app = app_factory(validated)

        def seal_report(value: Mapping[str, Any]) -> None:
            JOB.write_new(report_path, value)

        envelope = run_sdk(validated, app, seal_report=seal_report)
        if not _successful_sdk_envelope(validated, envelope, sealed=True):
            raise ValueError("prepared SDK 수집·cleanup·원문 검증 경계가 실패했다")
        JOB.write_new(
            output / "result.json", _terminal_body(validated, envelope, report_path)
        )
    except BaseException as error:  # noqa: BLE001 - close 전에 실패 원문을 보존한다.
        primary = error
        partial = None
        try:
            if report_path.exists() or report_path.is_symlink():
                partial = _descriptor(report_path)
        except Exception as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
        failure = {
            "schema": JOB.FAILURE_SCHEMA,
            "completed": False,
            "mode": JOB.MODE,
            "job_sha256": validated["job_sha256"],
            "renderer_smoke_only": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "partial_report": partial,
            "sdk": getattr(
                error, "envelope", getattr(error, "prepared_sdk_envelope", None)
            ),
        }
        try:
            JOB.write_new(output / "failure.json", failure)
        except BaseException as evidence_error:  # noqa: BLE001
            traceback.print_exception(evidence_error)
    finally:
        if app is not None:
            try:
                app.close()
            except BaseException as error:  # noqa: BLE001
                try:
                    JOB.write_new(
                        output / "shutdown_failure.json",
                        {
                            "schema": "vision_prepared_capture_shutdown_failure_v1",
                            "completed": False,
                            "job_sha256": validated["job_sha256"],
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    )
                except BaseException as evidence_error:  # noqa: BLE001
                    traceback.print_exception(evidence_error)
                if primary is None:
                    primary = error
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return 0 if primary is None else 1


def _remaining(deadline: float, *, reserve: float = 0.0) -> float:
    value = deadline - time.monotonic() - reserve
    if value <= 0:
        raise TimeoutError("prepared 전체 시간 또는 후속 검증 예약이 소진됐다")
    return value


def _gpu_query_run(
    deadline: float, run: Callable[..., Any] = subprocess.run
) -> Callable[..., Any]:
    def bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = min(
            30.0, float(kwargs.get("timeout", 30.0)), _remaining(deadline)
        )
        result = run(*args, **kwargs)
        _remaining(deadline)
        return result

    return bounded


def _gpu_reference(reader: Any, validated: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = getattr(reader, "last_receipt", None)
    if not isinstance(reference, Mapping):
        raise TypeError("prepared GPU 관찰 원문 참조가 없다")
    REMOTE.verify_observation_reference(
        reference,
        validated["resource"]["resource_id"],
        validated["resource"]["gpu_uuid"],
    )
    return reference


def run_job(
    job_path: Path,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    gpu_reader: Callable[[], list[int]] | None = None,
    identify: Callable[[int], Mapping[str, Any]] = OWNED._process_identity,
    cleanup: Callable[..., bool] = OWNED._safe_cleanup,
) -> dict[str, Any]:
    validated = JOB.validate_job(job_path)
    stage = Path(validated["job_path"]).parent
    if (stage / "execution.json").exists() or (stage / "launch.json").exists():
        raise FileExistsError("실행 이력이 있는 prepared job을 반복하지 않는다")
    if os.name != "posix" and popen is subprocess.Popen:
        raise RuntimeError(
            "실제 prepared GPU 실행은 소유 Linux process group에서 수행한다"
        )
    started = time.monotonic()
    max_wall_s = float(JOB.LIMITS["max_wall_s"])
    deadline = started + max_wall_s
    if gpu_reader is None:
        resource = validated["resource"]
        gpu_reader = REMOTE.StrictRemoteGpuReader(
            {key: resource[key] for key in REMOTE.CAPTURE_KEYS},
            stage / "gpu-observations",
            run=_gpu_query_run(deadline),
        )
    if gpu_reader is None:
        raise TypeError("prepared GPU 관찰 함수가 없다")
    before_bytes = Path(job_path).read_bytes()
    receipt: dict[str, Any] = {
        "schema": EXECUTION_SCHEMA,
        "completed": False,
        "mode": JOB.MODE,
        "job_sha256": validated["job_sha256"],
        "exit_code": None,
        "timed_out": False,
        "cleanup_verified": False,
        "launch": None,
        "gpu_before": None,
        "gpu_after": None,
        "gpu_before_observation": None,
        "gpu_after_observation": None,
        "result_sha256": None,
        "error_type": None,
        "error_message": None,
        "wall_s": None,
        "renderer_smoke_only": True,
        "training_actions": 0,
        "model_api_requests": 0,
        "full_recordings": 0,
        "timing": {
            "max_wall_s": max_wall_s,
            "child_max_s": max_wall_s - POSTFLIGHT_RESERVE_S,
            "postflight_reserve_s": POSTFLIGHT_RESERVE_S,
            "cleanup_worst_case_s": OWNED_CLEANUP_WORST_CASE_S,
            "child_wait_timeout_s": None,
            "cleanup_wall_s": None,
            "postflight_wall_s": None,
            "deadline_overrun_s": 0.0,
        },
    }
    process = None
    primary: BaseException | None = None
    cleanup_attempted = False
    postflight_started: float | None = None

    def clean_owned() -> bool:
        nonlocal cleanup_attempted
        cleanup_attempted = True
        begin = time.monotonic()
        try:
            return cleanup(receipt["launch"], process)
        finally:
            receipt["timing"]["cleanup_wall_s"] = time.monotonic() - begin

    command = [
        str(validated["job"]["python"]),
        "-I",
        "-S",
        "-c",
        GATE.OWNED_WORKER_GATE,
        str(validated["job"]["python"]),
        "-B",
        str(Path(__file__).resolve()),
        "--sdk-child",
        "--job",
        str(Path(job_path).absolute()),
    ]
    try:
        receipt["gpu_before"] = gpu_reader()
        if receipt["gpu_before"]:
            raise RuntimeError("등록 GPU에 다른 compute process가 있다")
        receipt["gpu_before_observation"] = dict(_gpu_reference(gpu_reader, validated))
        _remaining(deadline, reserve=POSTFLIGHT_RESERVE_S)
        JOB.write_new(
            stage / "launch.json",
            {"command": command, "cwd": validated["job"]["isaaclab"]},
        )
        with (
            (stage / "stdout.log").open("xb") as stdout,
            (stage / "stderr.log").open("xb") as stderr,
        ):
            process = popen(
                command,
                cwd=validated["job"]["isaaclab"],
                env={
                    **os.environ,
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": "0",
                },
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.PIPE,
                start_new_session=True,
                shell=False,
            )
            try:
                receipt["launch"] = dict(identify(process.pid))
            except BaseException:
                receipt["cleanup_verified"] = GATE._cleanup_waiting_gate(process)
                process = None
                raise
            if process.stdin is None:
                raise RuntimeError("소유 prepared gate에 stdin이 없다")
            remaining = min(
                max_wall_s - POSTFLIGHT_RESERVE_S,
                _remaining(deadline, reserve=POSTFLIGHT_RESERVE_S),
            )
            receipt["timing"]["child_wait_timeout_s"] = remaining
            process.stdin.write(b"GO\n")
            process.stdin.flush()
            process.stdin.close()
            try:
                receipt["exit_code"] = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
                raise
            finally:
                postflight_started = time.monotonic()
                receipt["cleanup_verified"] = clean_owned()
        _remaining(deadline)
        receipt["gpu_after"] = gpu_reader()
        receipt["gpu_after_observation"] = dict(_gpu_reference(gpu_reader, validated))
        _remaining(deadline)
        if Path(job_path).read_bytes() != before_bytes:
            raise ValueError("prepared SDK child 실행 중 job 원문이 바뀌었다")
        with JOB.validation_deadline(deadline):
            JOB.validate_job(job_path, require_fresh=False)
        _remaining(deadline)
        if (
            receipt["exit_code"] != 0
            or receipt["cleanup_verified"] is not True
            or receipt["gpu_after"] != []
        ):
            raise RuntimeError(
                "prepared child 종료·소유 group 정리·GPU idle 증거가 실패했다"
            )
        verify_result(validated)
        _remaining(deadline)
        if receipt["timing"]["cleanup_wall_s"] > OWNED_CLEANUP_WORST_CASE_S:
            raise TimeoutError("소유 process group 정리가 예약한 시간을 넘었다")
        receipt["result_sha256"] = _sha(
            Path(validated["output_directory"]) / "result.json"
        )
        receipt["completed"] = True
    except BaseException as error:  # noqa: BLE001
        primary = error
        receipt["error_type"], receipt["error_message"] = (
            type(error).__name__,
            str(error),
        )
    finally:
        if process is not None and not cleanup_attempted:
            try:
                postflight_started = time.monotonic()
                receipt["cleanup_verified"] = clean_owned()
                receipt["exit_code"] = process.returncode
            except BaseException as error:  # noqa: BLE001
                receipt["cleanup_verified"] = False
                if primary is None:
                    primary = error
                    receipt["error_type"], receipt["error_message"] = (
                        type(error).__name__,
                        str(error),
                    )
        if receipt["gpu_after"] is None:
            try:
                _remaining(deadline)
                receipt["gpu_after"] = gpu_reader()
                receipt["gpu_after_observation"] = getattr(
                    gpu_reader, "last_receipt", None
                )
            except BaseException as error:  # noqa: BLE001
                if primary is None:
                    primary = error
                    receipt["error_type"], receipt["error_message"] = (
                        type(error).__name__,
                        str(error),
                    )
        receipt["wall_s"] = time.monotonic() - started
        receipt["timing"]["postflight_wall_s"] = (
            time.monotonic() - postflight_started
            if postflight_started is not None
            else None
        )
        receipt["timing"]["deadline_overrun_s"] = max(
            0.0, receipt["wall_s"] - max_wall_s
        )
        if receipt["wall_s"] >= max_wall_s:
            receipt["completed"] = False
            if primary is None:
                receipt["error_type"] = "TimeoutError"
                receipt["error_message"] = "prepared 전체 시간 상한이 끝났다"
        if primary is not None:
            receipt["completed"] = False
        JOB.write_new(stage / "execution.json", receipt)
    if isinstance(primary, (KeyboardInterrupt, SystemExit)):
        raise primary
    return receipt


def prepare_job(
    protocol_path: Path,
    resource_path: Path,
    resource_sha: str,
    isaaclab: Path,
    python: Path,
    stage: Path,
) -> Path:
    protocol = JOB.validate_protocol(protocol_path)
    _regular(resource_path)
    if _sha(resource_path) != resource_sha:
        raise ValueError("prepared resource proof SHA가 다르다")
    stage = stage.absolute()
    JOB._check_ancestors(stage, "prepared stage")
    if stage.exists():
        raise FileExistsError("prepared stage는 symlink가 없는 새 폴더여야 한다")
    stage.mkdir(parents=True, exist_ok=False)
    path = stage / "job.json"
    JOB.write_new(
        path,
        {
            "schema": JOB.JOB_SCHEMA,
            "mode": JOB.MODE,
            "job_id": protocol["protocol_id"],
            "source": {
                "repo_root": str(ROOT),
                "commit": protocol["source_commit"],
                "files_lf": protocol["source_sha256_lf"],
            },
            "protocol": {
                "path": str(protocol_path.absolute()),
                "sha256_lf": _sha(protocol_path, lf=True),
            },
            "resource_proof": {
                "path": str(resource_path.absolute()),
                "sha256": resource_sha,
            },
            "isaaclab": str(isaaclab.absolute()),
            "python": str(python.absolute()),
            "output_directory": str(stage / "output"),
        },
    )
    JOB.validate_job(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-child", action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args(argv)
    if args.sdk_child:
        if args.job is None:
            parser.error("--sdk-child에는 --job이 필요하다")
        return sdk_child(args.job)
    if args.job is None:
        if any(
            value is None
            for value in (
                args.protocol,
                args.resource_proof,
                args.resource_sha256,
                args.isaaclab,
                args.python,
                args.stage,
            )
        ):
            parser.error(
                "새 실행에는 protocol/resource/isaaclab/python/stage가 필요하다"
            )
        args.job = prepare_job(
            args.protocol,
            args.resource_proof,
            args.resource_sha256,
            args.isaaclab,
            args.python,
            args.stage,
        )
    result = run_job(args.job)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["completed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
