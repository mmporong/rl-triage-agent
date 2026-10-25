"""동일 Writer callback 원문으로 A-A-B-A 정지 화면을 검증한다."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R
import vision_frame_sync_v1 as SYNC

SCHEMA = "vision_render_sync_smoke_v2"
METHOD = "atomic_writer_dual_rail_frame_code_v1"
ORDER = ("warmup", "A", "B", "A")


def _journal(job_path: Path, ordinal: int) -> tuple[dict, dict, tuple[bytes, bytes]]:
    from vision_writer_payload_v1 import load_capture_journal

    return load_capture_journal(job_path, ordinal)


def _assess(rows: Sequence[Mapping[str, Any]], job_path: Path) -> dict[str, Any]:
    if len(rows) != 4 or [row.get("label") for row in rows] != list(ORDER):
        raise ValueError("atomic smoke는 A-A-B-A 네 capture여야 한다")
    bundles, raw = [], []
    fixed_state: Mapping[str, Any] | None = None
    for ordinal, row in enumerate(rows):
        if set(row) != {
            "label",
            "state_before",
            "state_after",
            "simulation_time_before_s",
            "simulation_time_after_s",
            "camera",
            "atomic_bundle",
        }:
            raise ValueError("atomic smoke capture 필드가 다르다")
        if not isinstance(row["state_before"], Mapping) or not isinstance(row["state_after"], Mapping):
            raise TypeError("smoke physics state가 mapping이 아니다")
        R.validate_render_state_unchanged(row["state_before"], row["state_after"])
        if fixed_state is None:
            fixed_state = row["state_before"]
        assert fixed_state is not None
        R.validate_render_state_unchanged(fixed_state, row["state_before"])
        _, bundle, pixels = _journal(job_path, ordinal)
        if row["atomic_bundle"] != bundle:
            raise ValueError("smoke bundle과 callback journal 원문이 다르다")
        before = R._finite(row["simulation_time_before_s"], "smoke time")
        if (
            before != row["simulation_time_after_s"]
            or before != bundle["physics"]["simulation_time_before_s"]
            or bundle["mode"] != SYNC.SMOKE_MODE
        ):
            raise ValueError("smoke state와 callback physics 시각이 다르다")
        camera = R._exact(row["camera"], {"eye_world_m", "lookat_world_m"}, "camera")
        eye = R._vector3(camera["eye_world_m"], "eye")
        lookat = R._vector3(camera["lookat_world_m"], "lookat")
        offset = [-3.0, 3.0, 2.0] if ordinal == 2 else [3.0, 3.0, 2.0]
        target = R.fixed_profile()["camera"]["lookat_offset_m"]
        root = R._vector3(row["state_before"]["root_position_world_m"], "root")
        if any(
            abs(eye[i] - root[i] - offset[i]) > 1e-9
            or abs(lookat[i] - root[i] - target[i]) > 1e-9
            for i in range(3)
        ):
            raise ValueError("smoke camera가 등록 A/B pose와 다르다")
        bundles.append(bundle)
        raw.append(pixels)
    SYNC.verify_sequence(bundles, raw)
    return {
        "schema": SCHEMA,
        "status": "pass",
        "method": METHOD,
        "capture_order": list(ORDER),
        "policy_steps": 0,
        "capture_phase_physics_steps": 0,
        "environment_reset_performed": True,
        "delta_time_s": 0.0,
        "rt_subframes": 1,
        "pause_timeline": False,
        "renderer_combination_only": True,
        "universal_zero_lag_claim": False,
        "captures": [dict(row) for row in rows],
    }


def collect(
    capture: Callable[..., Mapping[str, Any]], job_path: Path
) -> tuple[list[dict], dict]:
    camera = list(R.fixed_profile()["camera"]["eye_offset_m"])
    offsets = (camera, camera, [-camera[0], camera[1], camera[2]], camera)
    captures = [
        dict(capture(label, list(offset)))
        for label, offset in zip(ORDER, offsets, strict=True)
    ]
    rows = []
    for ordinal, row in enumerate(captures):
        _, bundle, pixels = _journal(job_path, ordinal)
        if (
            row["rgb"] != pixels[0]
            or row["atomic_bundle"] != bundle
            or {
                "read_times": row["read_times"],
                "reference_time": row["reference_time"],
            }
            != bundle["timing"]
        ):
            raise ValueError("반환한 RGB/timing과 Writer journal이 다르다")
        rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"rgb", "read_times", "reference_time"}
            }
        )
    return captures, _assess(rows, job_path)


def verify_report(
    report_path: Path, job_path: Path, audit_paths: Mapping[str, Path]
) -> dict:
    report = R.read_object(report_path)
    rows = report.get("captures")
    if not isinstance(rows, list) or report != _assess(rows, job_path):
        raise ValueError("atomic smoke report가 callback 원문과 다르다")
    for ordinal in (1, 2, 3):
        _, bundle, raw = _journal(job_path, ordinal)
        path = audit_paths[f"audit_rgb_{ordinal}"]
        if (
            path.read_bytes() != raw[0]
            or hashlib.sha256(raw[0]).hexdigest() != bundle["rgb"]["sha256"]
        ):
            raise ValueError("atomic smoke audit RGB가 Writer 원문과 다르다")
    return report
