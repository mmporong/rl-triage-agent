"""비연속 카메라 캡처의 FXAA 설정과 실제 readback을 결합한다."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SOURCE_FILE = "bench/vision_fxaa_policy_v1.py"
SCHEMA = "vision_fxaa_policy_v1"
METHOD = "omni.replicator.core.settings.set_render_rtx_realtime"
SETTING_KEYS = ("/rtx/rendermode", "/rtx/post/aa/op", "/rtx-transient/post/aa/limitedOps")


def required(job: Mapping[str, Any]) -> bool:
    source = job.get("source", {})
    return SOURCE_FILE in source.get("files_lf", {})


def readback(settings: Any) -> dict[str, Any]:
    return {key: settings.get(key) for key in SETTING_KEYS}


def verify_readback(value: Any) -> None:
    if (not isinstance(value, dict) or set(value) != set(SETTING_KEYS)
            or not isinstance(value[SETTING_KEYS[0]], str)
            or value[SETTING_KEYS[0]].casefold() != "raytracedlighting"
            or type(value[SETTING_KEYS[1]]) is not int or value[SETTING_KEYS[1]] != 2
            or value[SETTING_KEYS[2]] is not False):
        raise ValueError("FXAA renderer 설정 readback이 등록 정책과 다르다")


def apply(rep: Any, settings: Any) -> dict[str, Any]:
    before = readback(settings)
    rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
    after = readback(settings)
    verify_readback(after)
    return {"schema": SCHEMA, "method": METHOD, "antialiasing": "FXAA",
            "before": before, "after": after}


def verify_receipt(value: Any) -> None:
    if (not isinstance(value, dict)
            or set(value) != {"schema", "method", "antialiasing", "before", "after"}
            or value.get("schema") != SCHEMA or value.get("method") != METHOD
            or value.get("antialiasing") != "FXAA"
            or not isinstance(value.get("before"), dict)
            or set(value["before"]) != set(SETTING_KEYS)):
        raise ValueError("FXAA renderer 정책 receipt가 다르다")
    verify_readback(value["after"])
