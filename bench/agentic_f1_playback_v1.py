"""F1(보상 결함) 학습 정책을 정상 Play 환경에서 재생하는 등록 객체와 run_real callback.

F1은 학습 중 ``track_lin_vel_xy_exp`` 보상 함수만 바꾼다. Play 동역학과 관측에는 영향이 없으므로 재생 때 다시
주입하지 않는다(옛 transferred Play 의미). F2처럼 모터 함수를 재주입하는 계약과 섞지 않는다.

- NONE: ``agentic_fault_playback_v1``의 등록 객체·callback을 그대로 쓴다.
- F1: 새 schema의 등록 객체. ``fault_id``는 학습 결함 F1이고 ``playback_injection``은 ``none``이다.
  callback은 App 생성 뒤 보상 함수와 모터 함수가 원본인지 관측해 receipt에 남기고, 바뀌어 있으면 재생 전에 멈춘다.

모듈 import에서는 Isaac·torch를 불러오지 않는다.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import agentic_fault_playback_v1 as V1
import record_vision_evidence_v1 as R

SCHEMA = "agentic_training_fault_playback_registration_v1"
FAULT_IDS = ("NONE", "F1")
PLAYBACK_INJECTION = "none"
TRAINING_FAULT_CATEGORY = "reward"
SOURCE_FILES = V1.SOURCE_FILES | {"bench/agentic_f1_playback_v1.py"}
REWARD_FUNCTION = ("isaaclab.envs.mdp.rewards", "track_lin_vel_xy_exp")
CLIP_FUNCTION = ("isaaclab.actuators.actuator_pd", "DCMotor._clip_effort")
_V1_MAKE_RUN_REAL = V1.make_run_real  # import 시점 원본. 실행 중 교체와 무관하게 NONE은 v1 그대로 간다.


def _hidden_category() -> Any:
    hidden = importlib.import_module("hidden_faults")
    hidden_file = getattr(hidden, "__file__", None)
    if not isinstance(hidden_file, str) or Path(hidden_file).resolve() != Path(V1.ROOT) / "bench/hidden_faults.py":
        raise ValueError("다른 hidden_faults 모듈")
    return hidden.FAULTS.get("F1", {}).get("category")


def registration(logical_id: str, checkpoint_sha256: str, fault_id: str) -> dict:
    """호출 전 protocol에 고정할 공개 등록 객체. NONE은 v1 객체와 같다."""
    if fault_id not in FAULT_IDS:
        raise ValueError("F1 재생 계약은 NONE·F1만 등록한다")
    base = V1.registration(logical_id, checkpoint_sha256, "NONE")  # 식별자·SHA·task·hidden_faults 결합은 v1 규칙
    if fault_id == "NONE":
        return base
    if _hidden_category() != TRAINING_FAULT_CATEGORY:
        raise ValueError("hidden_faults의 F1이 보상 결함이 아니다")
    return {**base, "schema": SCHEMA, "fault_id": "F1", "training_fault_category": TRAINING_FAULT_CATEGORY,
            "playback_injection": PLAYBACK_INJECTION}


def _verify(value: dict, validated: dict) -> dict:
    expected = registration(value["logical_id"], value["checkpoint_sha256"], value["fault_id"])
    job, protocol = validated["job"], validated["protocol"]
    if (value != expected or expected["schema"] != SCHEMA
            or protocol.get("playback_registrations", {}).get(value["logical_id"]) != expected
            or job["checkpoint"]["logical_id"] != value["logical_id"]
            or validated["checkpoint_sha256"] != value["checkpoint_sha256"]
            or job["checkpoint"]["sha256"] != value["checkpoint_sha256"]
            or job["profile"] != R.fixed_profile() or Path(validated["repo"]).resolve() != Path(V1.ROOT)):
        raise ValueError("F1 playback protocol·checkpoint·Go2 profile 결합 불일치")
    for name in SOURCE_FILES:
        actual = R.sha256_file(Path(V1.ROOT) / name, lf=True)
        if protocol["source_sha256_lf"].get(name) != actual or job["source"]["files_lf"].get(name) != actual:
            raise ValueError("F1 playback source closure 불일치")
    return expected


def _function_identity(module_name: str, qualname: str) -> dict:
    owner: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        owner = getattr(owner, part)
    code = getattr(owner, "__code__", None)
    return {"module": getattr(owner, "__module__", None), "qualname": getattr(owner, "__qualname__", None),
            "code_name": None if code is None else code.co_name}


def observe_no_injection(checked: dict) -> dict:
    """App 생성 뒤 호출한다. F1 보상 함수와 F2 모터 함수가 원본이어야 한다(아무것도 바꾸지 않는다)."""
    reward = _function_identity(*REWARD_FUNCTION)
    clip = _function_identity(*CLIP_FUNCTION)
    if (reward["module"] != REWARD_FUNCTION[0] or reward["code_name"] != REWARD_FUNCTION[1]
            or clip["module"] != CLIP_FUNCTION[0] or clip["qualname"] != CLIP_FUNCTION[1]):
        raise ValueError("F1 재생 전에 보상 또는 모터 함수가 이미 바뀌었다")
    return {"fault_id": checked["fault_id"], "class_method_changed": False,
            "playback_injection": PLAYBACK_INJECTION, "reward_function": reward, "clip_function": clip}


def make_run_real(value: dict, *, runner: Callable | None = None,
                  installer: Callable | None = None) -> Callable:
    """기존 native worker의 run_real 인자로 쓰는 callback. NONE은 v1 callback이다."""
    if value.get("schema") == V1.SCHEMA and value.get("fault_id") == "NONE":
        return _V1_MAKE_RUN_REAL(value, runner=runner, installer=installer)
    if value.get("schema") != SCHEMA or value.get("fault_id") != "F1":
        raise ValueError("F1 재생 계약 밖의 등록 객체다")
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
        evidence: dict = {}
        result = None
        try:
            evidence = (installer or observe_no_injection)(checked)
            result = selected(validated, app, **kwargs)
        except BaseException as error:  # noqa: BLE001 - App close 전에 원래 실패를 기록한다.
            primary = error
        receipt = {**intent, "schema": "agentic_fault_playback_receipt_v1",
                   "status": "failed" if primary else "completed", "installation": evidence,
                   "restore_completed": True, "primary_error": type(primary).__name__ if primary else None,
                   "restore_error": None}
        R.write_new(folder / "playback-receipt.json", receipt)
        if primary is not None:
            raise primary
        return result

    return run_real
