"""검증된 probe receipt만으로 등록 원인을 확인한다.

측정기와 실행 backend를 가져오지 않는다. 호출자가 제공한 판정 boolean 대신
검증 함수를 실행하고, 그 결과를 이 모듈만 발급하는 불투명 객체로 감싼다.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from rl_triage import mechanism_registry as registry_tools

ACTUATOR_PROBE = "P_actuator_mapping"
COUNT_PROBE = "P_optimizer_count"
GRADIENT_PROBE = "P_optimizer_gradient"
TORQUE_PROBE = "P_torque"
COUNT_CAUSE = "optimizer_update_count"
GRADIENT_CAUSE = "optimizer_surrogate"
_OPAQUE_MARKER = object()


def _canonical(value: object) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("value must be finite canonical JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def mechanism_registry(profile: str = "full") -> dict[str, Any]:
    """전체 계약 또는 실제 실행기가 지원하는 세 확인 검사 계약을 반환한다."""
    if profile not in {"full", "runtime_confirmation_v1"}:
        raise ValueError("unregistered mechanism registry profile")
    value = registry_tools.without_torque_signal()
    value["mechanisms"].extend([COUNT_CAUSE, GRADIENT_CAUSE])
    for spec in value["probes"].values():
        spec["expect"].update({COUNT_CAUSE: None, GRADIENT_CAUSE: None})

    def expectations(target: str) -> dict[str, str]:
        return {name: "abnormal" if name == target else "normal" for name in value["mechanisms"]}

    value["probes"].update({
        ACTUATOR_PROBE: {
            "measures": "검증된 runtime DCMotor 토크-속도 mapping의 등록 격자 일치 상태",
            "cost_gpu_s": 0,
            "args": {"input_grid": "actuator_envelope.VELOCITY_GRID", "policy_rollout": False},
            "expect": expectations("actuator"),
        },
        COUNT_PROBE: {
            "measures": "완료 PPO update의 optimizer.step 및 minibatch count 상태",
            "cost_gpu_s": 0,
            "args": {"updates": 3, "observation": "recorded_optimizer_counts"},
            "expect": expectations(COUNT_CAUSE),
        },
        GRADIENT_PROBE: {
            "measures": "저장 clipped-surrogate gradient와 기대 gradient의 일치 상태",
            "cost_gpu_s": 0,
            "args": {"observation": "recorded_ppo_gradient"},
            "expect": expectations(GRADIENT_CAUSE),
        },
    })
    value["fixed_order"].extend([ACTUATOR_PROBE, COUNT_PROBE, GRADIENT_PROBE])
    if profile == "runtime_confirmation_v1":
        # 지원 밖 원인은 가설에 남겨 정상 세 검사만으로 건강을 확정하지 않는다.
        order = [COUNT_PROBE, GRADIENT_PROBE, ACTUATOR_PROBE]
        value["probes"] = {probe: value["probes"][probe] for probe in order}
        value["fixed_order"] = order
        value["probes"][COUNT_PROBE]["args"]["observation"] = "registered_optimizer_counts"
        value["probes"][GRADIENT_PROBE]["args"]["observation"] = "registered_ppo_gradient"
        value["probes"][GRADIENT_PROBE]["measures"] = (
            "등록 PPO update의 clipped-surrogate gradient와 기대 gradient의 일치 상태"
        )
    return registry_tools.normalize(value)


def selector_probe_metadata() -> dict[str, dict[str, Any]]:
    """selector가 probe를 안전하게 설명하고 후속 가설을 고를 때 쓰는 메타데이터."""
    return deepcopy({
        TORQUE_PROBE: {
            "role": "symptom_only",
            "prerequisites": ["same-seed reference measurement"],
            "follow_up_hypotheses": ["actuator", "physics", "policy_or_command_distribution"],
            "claim_limit": "abnormal은 actuator 원인 확인이 아니며 outcome은 원인 갱신에 쓰지 않는다",
        },
        ACTUATOR_PROBE: {
            "role": "cause_confirmation",
            "prerequisites": ["verified source/config/runtime binding", "registered signed velocity grid"],
            "follow_up_hypotheses": ["actuator"],
            "claim_limit": "normal은 등록 격자의 DCMotor mapping만 제외하며 모든 actuator의 정상을 뜻하지 않는다",
        },
        COUNT_PROBE: {
            "role": "cause_confirmation",
            "prerequisites": ["verified case/reference receipts", "reference count normal"],
            "follow_up_hypotheses": [COUNT_CAUSE],
            "claim_limit": "count abnormal만 optimizer update-count 원인을 확인한다",
        },
        GRADIENT_PROBE: {
            "role": "cause_confirmation",
            "prerequisites": ["verified case/reference receipts", "count normal", "reference gradient normal"],
            "follow_up_hypotheses": [GRADIENT_CAUSE],
            "claim_limit": "gradient abnormal은 actuator 원인이 아니라 surrogate 원인만 확인한다",
        },
    })


@dataclass(frozen=True)
class _VerifiedProbeResult:
    probe: str
    raw: dict[str, Any]
    raw_sha256: str
    source: dict[str, Any]
    source_sha256: str
    verifier: str
    verification_errors: tuple[str, ...]
    _marker: object


def verify_probe_result(
    probe: str,
    raw_result: Mapping[str, Any],
    *,
    verifier: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[str]],
    source: Mapping[str, Any],
) -> _VerifiedProbeResult:
    """검증기를 실제 호출하고 원문·출처 digest와 함께 불투명 결과를 발급한다."""
    if probe not in mechanism_registry()["probes"]:
        raise ValueError("probe is not registered")
    if not isinstance(raw_result, Mapping) or not isinstance(source, Mapping):
        raise TypeError("raw_result and source must be mappings")
    if not callable(verifier):
        raise TypeError("verifier must be callable")
    raw = json.loads(_canonical(dict(raw_result)))
    provenance = json.loads(_canonical(dict(source)))
    errors = verifier(deepcopy(raw), deepcopy(provenance))
    if (not isinstance(errors, Sequence) or isinstance(errors, (str, bytes))
            or any(not isinstance(item, str) or not item for item in errors)):
        raise TypeError("verifier must return a sequence of non-empty error strings")
    identity = f"{getattr(verifier, '__module__', '<unknown>')}.{getattr(verifier, '__qualname__', type(verifier).__qualname__)}"
    return _VerifiedProbeResult(
        probe=probe,
        raw=raw,
        raw_sha256=_digest(raw),
        source=provenance,
        source_sha256=_digest(provenance),
        verifier=identity,
        verification_errors=tuple(errors),
        _marker=_OPAQUE_MARKER,
    )


def _checked(result: _VerifiedProbeResult) -> _VerifiedProbeResult:
    if not isinstance(result, _VerifiedProbeResult) or result._marker is not _OPAQUE_MARKER:
        raise TypeError("a result issued by verify_probe_result is required")
    if _digest(result.raw) != result.raw_sha256 or _digest(result.source) != result.source_sha256:
        raise ValueError("verified result content changed")
    return result


def normalize_observation(result: _VerifiedProbeResult) -> dict[str, Any]:
    """불투명 검증 결과를 registry outcome으로 정규화한다."""
    checked = _checked(result)
    outcome = "unknown"
    if not checked.verification_errors:
        if checked.probe == ACTUATOR_PROBE:
            if checked.raw.get("status") == "valid" and type(checked.raw.get("curve_matches_expected")) is bool:
                outcome = "normal" if checked.raw["curve_matches_expected"] else "abnormal"
        elif checked.probe == COUNT_PROBE:
            candidate = checked.raw.get("count_status")
            if candidate in ("normal", "abnormal", "unknown"):
                outcome = candidate
        elif checked.probe == GRADIENT_PROBE:
            candidate = checked.raw.get("gradient_status", checked.raw.get("outcome"))
            if candidate in ("normal", "abnormal", "unknown"):
                outcome = candidate
        elif checked.probe == TORQUE_PROBE:
            candidate = checked.raw.get("outcome")
            if candidate in ("normal", "abnormal", "unknown"):
                outcome = candidate
    return {
        "schema": "mechanism_observation_v1",
        "probe": checked.probe,
        "outcome": outcome,
        "raw_sha256": checked.raw_sha256,
        "source": deepcopy(checked.source),
        "source_sha256": checked.source_sha256,
        "verifier": checked.verifier,
        "verification_errors": list(checked.verification_errors),
        "claim_limit": selector_probe_metadata().get(checked.probe, {}).get("claim_limit"),
    }


def confirm_registered_cause(
    registry: dict[str, Any],
    hypotheses: Sequence[str],
    results: Sequence[_VerifiedProbeResult],
) -> dict[str, Any]:
    """검증 receipt가 등록된 단독 원인 조건을 충족하는지 판정한다."""
    checked_registry = registry_tools.normalize(registry)
    if (not isinstance(hypotheses, Sequence) or isinstance(hypotheses, (str, bytes))
            or any(name not in checked_registry["mechanisms"] for name in hypotheses)):
        raise ValueError("hypotheses must be registered")
    remaining = list(hypotheses)
    observations = []
    seen: set[str] = set()
    for result in results:
        observation = normalize_observation(result)
        probe, outcome = observation["probe"], observation["outcome"]
        if probe in seen:
            raise ValueError("a probe may be consumed only once")
        seen.add(probe)
        remaining, dropped = registry_tools.update(checked_registry, remaining, probe, outcome)
        observations.append({**observation, "dropped": dropped, "remaining": list(remaining)})

    confirmed = None
    if len(remaining) == 1:
        candidate = remaining[0]
        for row in observations:
            expected = checked_registry["probes"][row["probe"]]["expect"]
            exclusive = (expected.get(candidate) == "abnormal" and all(
                value != "abnormal" for name, value in expected.items() if name != candidate
            ))
            if row["outcome"] == "abnormal" and exclusive:
                confirmed = candidate
                break
    return {
        "schema": "mechanism_confirmation_v1",
        "status": "confirmed" if confirmed else ("none_supported" if not remaining else "unresolved"),
        "confirmed_cause": confirmed,
        "remaining_hypotheses": remaining,
        "observations": observations,
        "registry_digest": registry_tools.registry_digest(checked_registry),
    }


__all__ = [
    "ACTUATOR_PROBE",
    "COUNT_CAUSE",
    "COUNT_PROBE",
    "GRADIENT_CAUSE",
    "GRADIENT_PROBE",
    "TORQUE_PROBE",
    "confirm_registered_cause",
    "mechanism_registry",
    "normalize_observation",
    "selector_probe_metadata",
    "verify_probe_result",
]
