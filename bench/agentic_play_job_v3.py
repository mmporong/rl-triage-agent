"""agentic play job v3: F1(보상 결함)·정상 같은 seed 쌍을 v1/v2 계약 그대로 녹화하게 하는 얇은 scope adapter.

v1 검증 함수·lease·입장·sidecar·job 규칙은 그대로 쓴다. ``f1_play_scope()`` 안에서만 v1 전역 일곱 개를 바꾸고
바깥으로 나가면 원래 값으로 되돌린다. F2 계약(v1·v2 protocol)은 이 scope 안에서 schema가 달라 거부된다.

- ``PROTOCOL_SCHEMA``·``JOB_SCHEMA``: v3 이름. F1 job은 scope 밖에서 검증되지 않는다.
- ``FAULT_IDS``·``PLAYBACK``: NONE·F1과 ``agentic_f1_playback_v1`` 등록 객체(F1은 재생 주입 없음).
- ``INTERPRETATION``: F1 Play 의미. ``playback_scope``는 v1과 같다.
- ``_registrations``: v1 검사 뒤 checkpoint 행이 historical base catalog(``vision_prepared_record_20261026_v4``,
  transferred v1이 SHA로 고정한 파일)의 같은 logical_id 행과 정확히 같아야 한다. 이름 위장·새 checkpoint를 막는다.
- ``_pairs``: case F1 하나와 같은 seed NONE reference 하나.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import agentic_f1_playback_v1 as PLAYBACK
import agentic_play_job_v1 as JOB1
import agentic_play_job_v2 as JOB2
import vision_transferred_pull_job_v1 as TRANSFERRED

PROTOCOL_SCHEMA = "agentic_play_protocol_v3"
JOB_SCHEMA = "agentic_play_job_v3"
FAULT_IDS = PLAYBACK.FAULT_IDS
HISTORICAL_CATALOG = dict(TRANSFERRED.BASE_PROTOCOL)
SOURCE_FILES = frozenset(set(PLAYBACK.SOURCE_FILES) | {
    "bench/agentic_play_job_v3.py",
    "tests/test_agentic_f1_playback_v1.py",
    "tests/test_agentic_play_job_v3.py",
})
INTERPRETATION = {
    "f1_scope": "reward_only_training_fault_track_lin_vel_xy_exp_world_frame",
    "playback_injection": PLAYBACK.PLAYBACK_INJECTION,
    "single_leg_fault": False,
    "normal_control": "same_seed_none_checkpoint_same_play",
    "playback_scope": JOB1.INTERPRETATION["playback_scope"],
    "fixed_eval_relation": "separate_pristine_env_execution_not_behavior_ground_truth",
    "f2_relation": "separate_contract_motor_reinjection_not_used",
    "checkpoint_lineage": "historical_base_catalog_rows_exact",
    "behavior_labels_created": False,
    "thresholds_created": False,
}
PREDICTION = "등록한 F1·정상 같은 seed checkpoint 쌍을 새 idle GPU에서 같은 Play 설정·상태 sidecar·렌더 sync로 재생한다."
OUTCOMES = {
    "recording_complete": "두 정책의 native 녹화·sidecar 봉인·trace 대조·render-sync가 완료된다.",
    "capture_hold": "AABA·terminal·sidecar·playback·render-sync 조건이 어긋나면 HOLD한다.",
    "infrastructure_failure": "자원·복원·실행 계층 실패를 분리 기록한다.",
}
_LOCK = threading.RLock()
_BASE: dict[str, Any] = {
    "PROTOCOL_SCHEMA": JOB1.PROTOCOL_SCHEMA, "JOB_SCHEMA": JOB1.JOB_SCHEMA, "FAULT_IDS": JOB1.FAULT_IDS,
    "PLAYBACK": JOB1.PLAYBACK, "INTERPRETATION": JOB1.INTERPRETATION,
    "_registrations": JOB1._registrations, "_pairs": JOB1._pairs,
}
_STATE: dict[str, Any] = {"depth": 0}


def historical_rows(repo: Path | None = None) -> dict[str, dict[str, Any]]:
    """transferred v1이 고정한 base protocol(path·LF SHA)의 checkpoint catalog를 logical_id별로 읽는다."""
    root = Path(repo or JOB1.ROOT)
    path = JOB1._regular(root / HISTORICAL_CATALOG["path"], "historical base catalog")
    if JOB1._sha(path, lf=True) != HISTORICAL_CATALOG["sha256_lf"]:
        raise ValueError("historical base catalog SHA가 다르다")
    rows = JOB1.RECORD_JOB._checkpoints(JOB1._read(path).get("checkpoints"))
    return {row["logical_id"]: dict(row) for row in rows}


def _registrations(
    value: Any, checkpoints: Sequence[Mapping[str, Any]], sources: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    result = _BASE["_registrations"](value, checkpoints, sources)
    historical = historical_rows()
    for row in checkpoints:
        if historical.get(row["logical_id"]) != dict(row):
            raise ValueError("checkpoint가 historical base catalog의 같은 이름 행과 다르다")
    return result


def _pairs(value: Any, registrations: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("F1·정상 pair가 비었다")
    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    result = []
    for raw in value:
        row = JOB1.COMMON._exact(raw, JOB1.PAIR_KEYS, "pair")
        pair_id, seed = row["pair_id"], row["seed"]
        if not isinstance(pair_id, str) or JOB1.COMMON.SLUG.fullmatch(pair_id) is None or pair_id in seen_pairs:
            raise ValueError("pair_id가 잘못됐거나 중복됐다")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("pair seed는 0 이상의 정수여야 한다")
        case, reference = row["case"], row["reference"]
        if case not in registrations or reference not in registrations or case == reference:
            raise ValueError("pair checkpoint가 catalog 밖이거나 같다")
        if registrations[case]["fault_id"] != "F1" or registrations[reference]["fault_id"] != "NONE":
            raise ValueError("pair는 F1 case 하나와 NONE reference 하나여야 한다")
        if not case.endswith(f"s{seed}-f1") or not reference.endswith(f"s{seed}-none"):
            raise ValueError("pair checkpoint 이름의 seed가 pair seed와 다르다")
        if {case, reference} & seen_ids:
            raise ValueError("checkpoint가 두 pair에 들어갔다")
        seen_pairs.add(pair_id)
        seen_ids |= {case, reference}
        result.append(dict(row))
    if seen_ids != set(registrations):
        raise ValueError("pair에 들어가지 않은 checkpoint가 있다")
    return result


_SCOPED = {
    "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA, "JOB_SCHEMA": JOB_SCHEMA, "FAULT_IDS": FAULT_IDS, "PLAYBACK": PLAYBACK,
    "INTERPRETATION": INTERPRETATION, "_registrations": _registrations, "_pairs": _pairs,
}


@contextmanager
def f1_play_scope() -> Iterator[None]:
    """process-global 교체 scope. 같은 thread 재진입은 깊이만 센다. 이미 다른 값이면 거부한다."""
    with _LOCK:
        if _STATE["depth"] == 0:
            if any(getattr(JOB1, name) is not value for name, value in _BASE.items()):
                raise RuntimeError("JOB1 계약 전역이 이미 교체됐다")
            for name, value in _SCOPED.items():
                setattr(JOB1, name, value)
        _STATE["depth"] += 1
        try:
            yield
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] == 0:
                for name, value in _BASE.items():
                    setattr(JOB1, name, value)


def scope_active() -> bool:
    return _STATE["depth"] > 0


def _scoped(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with f1_play_scope():
            return function(*args, **kwargs)

    return wrapper


def build_protocol(*, prediction: str = PREDICTION, **kwargs: Any) -> dict[str, Any]:
    """v1 build_protocol 인자 그대로. F1 사전등록 문장으로 바꾼 뒤 같은 본문 검사를 다시 한다."""
    with f1_play_scope(), JOB2.canonical_lease_scope():
        value = JOB1.build_protocol(**kwargs)
        value["preregistration"]["prediction"] = prediction
        value["preregistration"]["outcomes"] = dict(OUTCOMES)
        JOB1._check_body(value, now=JOB1._utc(value["registered_at_utc"], "registered"), require_current=False)
    return value


# lease root·source closure를 읽는 v2 API를 F1 scope 안에서 실행한다(서명은 v1과 같다).
validate_protocol = _scoped(JOB2.validate_protocol)
validate_resource_proof = _scoped(JOB2.validate_resource_proof)
build_resource_proof = _scoped(JOB2.build_resource_proof)
validate_job = _scoped(JOB2.validate_job)
create_job = _scoped(JOB2.create_job)
lease_paths = _scoped(JOB2.lease_paths)
verify_lease = _scoped(JOB2.verify_lease)


@contextmanager
def resource_lease(protocol_path: Path, **kwargs: Any) -> Iterator[dict[str, str]]:
    with f1_play_scope(), JOB2.resource_lease(protocol_path, **kwargs) as reference:
        yield reference


def __getattr__(name: str) -> Any:
    """나머지 이름은 v2(→v1)로 넘긴다."""
    try:
        return getattr(JOB2, name)
    except AttributeError as error:
        raise AttributeError(f"agentic_play_job_v3에 {name}이 없다") from error
