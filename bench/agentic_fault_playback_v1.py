"""등록된 Go2 Play 환경에 학습 당시 F2를 적용하고 원본 함수로 복원한다.

모듈 import에서는 Isaac·torch를 불러오지 않는다. App을 만든 child가 env 생성
전에 사용하는 run_real callback이며, 영상과 정상 환경 고정 평가의 의미를 구분한다.
"""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import record_vision_evidence_v1 as R

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "agentic_fault_playback_registration_v1"
SOURCE_FILES = {"bench/agentic_fault_playback_v1.py", "bench/hidden_faults.py"}


def registration(logical_id: str, checkpoint_sha256: str, fault_id: str) -> dict:
    """호출 전 protocol에 고정할 공개 등록 객체를 만든다."""
    if (not isinstance(logical_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", logical_id) is None
            or re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256) is None or fault_id not in {"NONE", "F2"}):
        raise ValueError("등록 playback 식별자·checkpoint·fault 형식")
    return {"schema": SCHEMA, "logical_id": logical_id, "checkpoint_sha256": checkpoint_sha256,
            "fault_id": fault_id, "task": R.TASK,
            "hidden_faults_sha256_lf": R.sha256_file(ROOT / "bench/hidden_faults.py", lf=True),
            "interpretation_scope": "same_play_state_not_pristine_fixed_evaluation"}


def _verify(value: dict, validated: dict) -> dict:
    expected = registration(value["logical_id"], value["checkpoint_sha256"], value["fault_id"])
    job, protocol = validated["job"], validated["protocol"]
    if (value != expected or protocol.get("playback_registrations", {}).get(value["logical_id"]) != expected
            or job["checkpoint"]["logical_id"] != value["logical_id"]
            or validated["checkpoint_sha256"] != value["checkpoint_sha256"]
            or job["checkpoint"]["sha256"] != value["checkpoint_sha256"]
            or job["profile"] != R.fixed_profile() or Path(validated["repo"]).resolve() != ROOT):
        raise ValueError("playback protocol·checkpoint·Go2 profile 결합 불일치")
    for name in SOURCE_FILES:
        actual = R.sha256_file(ROOT / name, lf=True)
        if (protocol["source_sha256_lf"].get(name) != actual
                or job["source"]["files_lf"].get(name) != actual):
            raise ValueError("playback source closure 불일치")
    return expected


def _install(value: dict) -> tuple[dict, Callable[[], None]]:
    if value["fault_id"] == "NONE":
        return {"fault_id": "NONE", "class_method_changed": False}, lambda: None
    # App 생성 뒤 호출된다. 기존 frozen 주입 함수를 사용하고 식을 복제하지 않는다.
    actuator = importlib.import_module("isaaclab.actuators.actuator_pd").DCMotor
    hidden = importlib.import_module("hidden_faults")
    hidden_file = getattr(hidden, "__file__", None)
    if not isinstance(hidden_file, str) or Path(hidden_file).resolve() != ROOT / "bench/hidden_faults.py":
        raise ValueError("다른 hidden_faults 모듈")
    old = actuator._clip_effort
    if old.__module__ != "isaaclab.actuators.actuator_pd":
        raise ValueError("playback 시작 전에 모터 함수가 이미 바뀌었다")
    try:
        hidden._before_env("F2")
        changed = actuator._clip_effort
        if changed is old:
            raise ValueError("F2가 모터 함수를 바꾸지 않았다")
    except BaseException:
        actuator._clip_effort = old
        raise

    def restore() -> None:
        actuator._clip_effort = old

    return {"fault_id": "F2", "class_method_changed": True,
            "before": {"module": old.__module__, "qualname": old.__qualname__},
            "after": {"module": changed.__module__, "qualname": changed.__qualname__}}, restore


def make_run_real(value: dict, *, runner: Callable | None = None,
                  installer: Callable | None = None) -> Callable:
    """기존 native worker의 run_real 인자로 쓰는 callback을 반환한다."""
    registered = deepcopy(value)

    def run_real(validated: dict, app: Any, **kwargs: Any) -> Any:
        checked = _verify(registered, validated)
        if app is None:
            raise ValueError("playback은 App 생성 뒤에 실행해야 한다")
        folder = Path(validated["job_path"]).parent
        intent = {"schema": "agentic_fault_playback_intent_v1", "registration": checked,
                  "job_sha256": validated["job_sha256"], "used_for_recovery_decision": False}
        R.write_new(folder / "playback-intent.json", intent)
        selected = runner
        if selected is None:
            import vision_play_worker_v1 as PLAY

            selected = PLAY.run_real
        primary: BaseException | None = None
        cleanup_error: BaseException | None = None
        restore: Callable[[], None] | None = None
        evidence: dict = {}
        result = None
        try:
            evidence, restore = (installer or _install)(checked)
            result = selected(validated, app, **kwargs)
        except BaseException as error:  # noqa: BLE001 - App close 전에 원래 실패를 기록한다.
            primary = error
        if restore is not None:
            try:
                restore()
            except BaseException as error:  # noqa: BLE001 - 첫 실패와 복원 실패를 분리한다.
                cleanup_error = error
        receipt = {**intent, "schema": "agentic_fault_playback_receipt_v1",
                   "status": "failed" if primary or cleanup_error else "completed", "installation": evidence,
                   "restore_completed": restore is not None and cleanup_error is None,
                   "primary_error": type(primary).__name__ if primary else None,
                   "restore_error": type(cleanup_error).__name__ if cleanup_error else None}
        R.write_new(folder / "playback-receipt.json", receipt)
        if primary is not None:
            raise primary
        if cleanup_error is not None:
            raise cleanup_error
        return result

    return run_real
