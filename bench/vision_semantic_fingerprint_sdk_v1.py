"""Semantic 진단 callback의 본문 비노출 지문을 sidecar로 봉인한다."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import vision_callback_fingerprint_v1 as FINGERPRINT
import vision_semantic_diagnostic_v1 as DIAGNOSTIC
import vision_semantic_job_v1 as JOB
import vision_semantic_sdk_v1 as BASE

SIDECAR_NAME = "one-view.callback-fingerprints.json"
SIDECAR_SCHEMA = "vision_semantic_callback_fingerprints_v1"
MAX_RAW_CALLBACKS = 4
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SIDECAR_KEYS = {
    "schema",
    "job_sha256",
    "callbacks_observed",
    "callbacks_preserved",
    "callback_limit",
    "overflow",
    "callbacks",
    "comparison",
    "used_for_decision",
    "sealed_before_base_report",
    "no_capture_approved",
}
CALLBACK_KEYS = {
    "ordinal", "window", "comparison_group_sha256", "fingerprint", "error_type"
}
FINGERPRINT_KEYS = {
    "schema", "whole_payload_sha256", "components", "canonical_limits",
    "no_capture_approved",
}
COMPONENT_KEYS = {
    "distribution_outputs",
    "trigger_outputs",
    "named_outputs",
    "reference_time",
    "writer_swh_frame_number",
    "render_product_identity",
    "camera_identity",
    "resolution",
    "rgb_data",
    "semantic_segmentation",
    "semantic_mask",
    "semantic_labels",
    "camera_params",
    "isaac_read_times",
}


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


class _FingerprintObserver(DIAGNOSTIC.DiagnosticObserver):
    """기존 observer로 전달하면서 원문을 보존하지 않는 지문만 수집한다."""

    def __init__(self, delegate: DIAGNOSTIC.DiagnosticObserver) -> None:
        self._delegate = delegate
        self.total_callbacks = 0
        self.overflow = False
        self.callbacks: list[dict[str, Any]] = []

    def observe(self, payload: Any) -> None:
        ordinal = self.total_callbacks
        self.total_callbacks += 1
        window = self._delegate.window
        if len(self.callbacks) >= MAX_RAW_CALLBACKS:
            self.overflow = True
        else:
            row: dict[str, Any] = {
                "ordinal": ordinal,
                "window": window,
                "comparison_group_sha256": None,
                "fingerprint": None,
                "error_type": None,
            }
            try:
                result = FINGERPRINT.fingerprint(payload)
                digest = result.get("whole_payload_sha256")
                if (
                    result.get("schema") != FINGERPRINT.SCHEMA
                    or result.get("no_capture_approved") is not True
                    or not isinstance(digest, str)
                    or HEX64.fullmatch(digest) is None
                ):
                    raise ValueError("callback fingerprint 반환 계약이 잘못됐다")
                row["fingerprint"] = result
                row["comparison_group_sha256"] = _group_sha(
                    window, DIAGNOSTIC._raw_frame_time(payload), result
                )
            except (TypeError, ValueError) as error:
                row["error_type"] = _type_name(error)
            self.callbacks.append(row)
        self._delegate.observe(payload)

    def sidecar(self, job_sha256: str) -> dict[str, Any]:
        comparison = _comparison(self.callbacks)
        return {
            "schema": SIDECAR_SCHEMA,
            "job_sha256": job_sha256,
            "callbacks_observed": self.total_callbacks,
            "callbacks_preserved": len(self.callbacks),
            "callback_limit": MAX_RAW_CALLBACKS,
            "overflow": self.overflow,
            "callbacks": [dict(row) for row in self.callbacks],
            "comparison": comparison,
            "used_for_decision": False,
            "sealed_before_base_report": True,
            "no_capture_approved": True,
        }


def _group_sha(
    window: str, raw_frame_time: Mapping[str, Any], fingerprint: Mapping[str, Any]
) -> str:
    components = fingerprint.get("components")
    if not isinstance(components, Mapping):
        raise TypeError("callback fingerprint components가 mapping이 아니다")
    isaac_times = raw_frame_time.get("isaac_read_times")
    if not isinstance(isaac_times, Mapping):
        raise TypeError("callback raw Isaac frame time이 mapping이 아니다")
    evidence = {
        "window": window,
        "render_product_identity": components.get("render_product_identity"),
        "raw_swh_frame_number": raw_frame_time.get("swhFrameNumber"),
        "raw_reference_time": raw_frame_time.get("reference_time"),
        "raw_isaac_frame_identity": {
            key: isaac_times.get(key)
            for key in (
                "frameNumber",
                "rationalTimeOfSimDenominator",
                "rationalTimeOfSimNumerator",
                "swhFrameNumber",
            )
        },
        "reference_time": components.get("reference_time"),
        "writer_swh_frame_number": components.get("writer_swh_frame_number"),
    }
    raw = json.dumps(
        evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_fingerprint(value: Mapping[str, Any]) -> str:
    if set(value) != FINGERPRINT_KEYS:
        raise ValueError("callback fingerprint 필드가 다르다")
    digest = value.get("whole_payload_sha256")
    components = value.get("components")
    if (
        value.get("schema") != FINGERPRINT.SCHEMA
        or value.get("no_capture_approved") is not True
        or not isinstance(digest, str)
        or HEX64.fullmatch(digest) is None
        or not isinstance(components, Mapping)
        or set(components) != COMPONENT_KEYS
        or value.get("canonical_limits")
        != {
            "max_bytes": FINGERPRINT.MAX_CANONICAL_BYTES,
            "max_nodes": FINGERPRINT.MAX_CANONICAL_NODES,
            "max_depth": FINGERPRINT.MAX_CANONICAL_DEPTH,
        }
    ):
        raise ValueError("callback fingerprint 본문 계약이 다르다")
    for component in components.values():
        if (
            not isinstance(component, Mapping)
            or set(component) != {"sha256", "metadata"}
            or not isinstance(component.get("sha256"), str)
            or HEX64.fullmatch(component["sha256"]) is None
            or not isinstance(component.get("metadata"), Mapping)
        ):
            raise ValueError("callback fingerprint component 계약이 다르다")
    return digest


def _comparison(rows: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        group = row.get("comparison_group_sha256")
        fingerprint = row.get("fingerprint")
        if not isinstance(group, str) or not isinstance(fingerprint, Mapping):
            continue
        digest = fingerprint.get("whole_payload_sha256")
        if isinstance(digest, str):
            grouped.setdefault(group, []).append(digest)
    comparable = [hashes for hashes in grouped.values() if len(hashes) >= 2]
    if not comparable:
        return "insufficient"
    return "equal" if all(len(set(hashes)) == 1 for hashes in comparable) else "unequal"


def _validate_counterparts(
    sidecar: Mapping[str, Any], envelope: Mapping[str, Any]
) -> None:
    if set(sidecar) != SIDECAR_KEYS:
        raise ValueError("fingerprint sidecar 필드가 다르다")
    diagnostic = envelope.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise TypeError("fingerprint counterpart diagnostic이 mapping이 아니다")
    callbacks = diagnostic.get("callbacks")
    side_callbacks = sidecar.get("callbacks")
    observations = diagnostic.get("observations")
    if (
        not isinstance(callbacks, list)
        or not isinstance(side_callbacks, list)
        or not isinstance(observations, Mapping)
    ):
        raise TypeError("fingerprint callback 목록이 아니다")
    if (
        sidecar.get("overflow") is not False
        or sidecar.get("callbacks_observed") != len(callbacks)
        or sidecar.get("callbacks_preserved") != len(callbacks)
        or diagnostic.get("callbacks_observed") != len(callbacks)
        or diagnostic.get("callbacks_preserved") != len(callbacks)
        or observations.get("overflow") is not False
        or len(callbacks) > MAX_RAW_CALLBACKS
        or len(side_callbacks) != len(callbacks)
    ):
        raise ValueError("fingerprint callback 수 또는 상한이 다르다")
    for index, (observed, fingerprinted) in enumerate(
        zip(callbacks, side_callbacks, strict=True)
    ):
        if not isinstance(observed, Mapping) or not isinstance(fingerprinted, Mapping):
            raise TypeError("fingerprint counterpart callback이 mapping이 아니다")
        if set(fingerprinted) != CALLBACK_KEYS:
            raise ValueError("fingerprint callback 필드가 다르다")
        if (
            observed.get("ordinal") != index
            or fingerprinted.get("ordinal") != index
            or observed.get("window") != fingerprinted.get("window")
        ):
            raise ValueError("fingerprint callback ordinal/window가 다르다")
        fingerprint = fingerprinted.get("fingerprint")
        error_type = fingerprinted.get("error_type")
        if observed.get("classification") == "valid":
            if not isinstance(fingerprint, Mapping):
                raise ValueError("유효 callback의 whole payload 지문이 없다")
            _validate_fingerprint(fingerprint)
            if error_type is not None:
                raise ValueError("유효 callback에 fingerprint 오류가 있다")
            expected_group = _group_sha(
                str(observed["window"]), observed["raw_frame_time"], fingerprint
            )
            if fingerprinted.get("comparison_group_sha256") != expected_group:
                raise ValueError("fingerprint callback 비교 group 결합이 다르다")
        else:
            if (
                fingerprint is not None
                or fingerprinted.get("comparison_group_sha256") is not None
                or not isinstance(error_type, str)
            ):
                raise ValueError("거절 callback의 fingerprint 오류 증거가 없다")
    if sidecar.get("comparison") != _comparison(side_callbacks):
        raise ValueError("fingerprint 비교 결과가 callback 지문과 다르다")


def _dependencies_with_fingerprint(
    deps: Mapping[str, Any], holder: dict[str, _FingerprintObserver]
) -> dict[str, Any]:
    selected = dict(deps)
    original_writer = selected["writer"]

    def writer(rep: Any, observer: DIAGNOSTIC.DiagnosticObserver) -> Any:
        if holder:
            raise RuntimeError("fingerprint writer는 한 번만 만들 수 있다")
        wrapped = _FingerprintObserver(observer)
        holder["observer"] = wrapped
        return original_writer(rep, wrapped)

    selected["writer"] = writer
    return selected


def run_one_view(
    validated: Mapping[str, Any],
    app: Any,
    *,
    seal_report: Callable[[Mapping[str, Any]], None],
    dependencies: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """기존 SDK lifecycle에 callback 지문 sidecar 봉인만 추가한다."""
    deps = dict(dependencies) if dependencies is not None else BASE._load_dependencies()
    holder: dict[str, _FingerprintObserver] = {}
    output = Path(validated["output_directory"])
    sidecar_path = output / SIDECAR_NAME
    job_sha256 = validated.get("job_sha256")
    if not isinstance(job_sha256, str) or HEX64.fullmatch(job_sha256) is None:
        raise ValueError("fingerprint job SHA-256이 잘못됐다")

    def seal_with_fingerprint(envelope: Mapping[str, Any]) -> None:
        observer = holder.get("observer")
        if observer is None:
            raise RuntimeError("fingerprint observer가 생성되지 않았다")
        sidecar = observer.sidecar(job_sha256)
        JOB.write_new(sidecar_path, sidecar)
        _validate_counterparts(sidecar, envelope)
        seal_report(envelope)

    return BASE.run_one_view(
        validated,
        app,
        seal_report=seal_with_fingerprint,
        dependencies=_dependencies_with_fingerprint(deps, holder),
    )
