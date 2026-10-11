"""Blind active-diagnosis model contract and deterministic validation.

This module has no model, GPU, ledger, or filesystem side effects.  The model
may rank mechanisms and request a public probe; frozen probe-loop rules alone
decide whether evidence is sufficient to confirm a diagnosis.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

from . import probe_loop as L


SYSTEM_PROMPT = """You are a blind RL failure-diagnosis policy.
Use only the supplied evidence facts, public probe catalog, and observation receipts.
Return one JSON object with exactly these fields:
- ranking: a permutation of all six public mechanisms
- probe: one unobserved public probe for action=continue, otherwise null
- action: continue, hold, or confirm
- reason: a non-empty concise explanation
- claims: one or more {"id": evidence_fact_id, "value": exact_fact_value} objects
- expected_outcomes: for continue, the selected probe's public outcomes for the
  top three ranked mechanisms; for hold/confirm, an empty object

Every claim must copy an available evidence fact exactly. After an observation, cite
at least one fact from the latest receipt. A receipt may support keeping the same
ranking; never change a judgment merely because a new observation arrived.
Treat configured or reference values as configuration/reference facts, not proof of
runtime behavior. Use causal wording only when an actual runtime observation supports
it. Never infer hidden case labels. Confirm only when the deterministic evidence state
is confirmed and its sole hypothesis is ranked first. Output JSON only.
"""

_ACTIONS = frozenset({"continue", "hold", "confirm"})
_SCALAR = (str, bool, int, float)


def _is_scalar(value: object) -> bool:
    return isinstance(value, _SCALAR) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _receipt_id(receipt: Mapping) -> str:
    rid = receipt.get("request_id", receipt.get("rid", receipt.get("id")))
    if not isinstance(rid, str) or not rid.strip() or ":" in rid:
        raise ValueError("receipt에는 콜론 없는 request_id가 필요하다")
    return rid


def _flatten_measurement(value: object, prefix: str, out: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key or ":" in key:
                raise ValueError("measurement 키는 콜론 없는 문자열이어야 한다")
            field = f"{prefix}.{key}" if prefix else key
            _flatten_measurement(child, field, out)
    elif _is_scalar(value):
        out[prefix] = value


def _receipt_is_usable(receipt: Mapping) -> bool:
    return (
        receipt.get("outcome") in {"normal", "abnormal"}
        and receipt.get("over_budget") is not True
        and receipt.get("exit_code", 0) == 0
    )


def receipt_facts(receipt: Mapping) -> dict[str, object]:
    """Return citable scalar facts from one receipt without mutating it."""
    if not isinstance(receipt, Mapping):
        raise ValueError("receipt는 JSON 객체여야 한다")
    rid = _receipt_id(receipt)
    if not _receipt_is_usable(receipt):
        return {}
    facts: dict[str, object] = {}
    measurement = receipt.get("measurement", {})
    if measurement is not None and not isinstance(measurement, Mapping):
        raise ValueError("receipt.measurement는 JSON 객체여야 한다")
    flattened: dict[str, object] = {}
    _flatten_measurement(measurement or {}, "", flattened)
    for field, value in flattened.items():
        facts[f"receipt:{rid}:{field}"] = value
    return facts


def _observed_probe(receipt: Mapping) -> tuple[str, str]:
    probe, outcome = receipt.get("probe"), receipt.get("outcome")
    if probe not in L.PROBES:
        raise ValueError(f"모르는 observed probe {probe!r}")
    if outcome not in L.OUTCOMES:
        raise ValueError(f"모르는 observed outcome {outcome!r}")
    return probe, outcome


def evidence_state(observed: Sequence[Mapping], budget_left: int | None = None) -> dict:
    """Replay frozen elimination rules over receipts, starting from all mechanisms."""
    if not isinstance(observed, Sequence) or isinstance(observed, (str, bytes, bytearray)):
        raise ValueError("observed는 receipt 목록이어야 한다")
    if budget_left is None:
        budget_left = max(0, len(L.PROBES) - len(observed))
    if isinstance(budget_left, bool) or not isinstance(budget_left, int) or budget_left < 0:
        raise ValueError("budget_left는 0 이상의 정수여야 한다")

    hypotheses: list[str] = list(L.MECHANISMS)
    dropped: list[str] = []
    outcomes: dict[str, str] = {}
    seen_ids: set[str] = set()
    for receipt in observed:
        rid = _receipt_id(receipt)
        if rid in seen_ids:
            raise ValueError(f"중복 receipt id {rid!r}")
        seen_ids.add(rid)
        probe, outcome = _observed_probe(receipt)
        if not _receipt_is_usable(receipt):
            outcome = "unknown"
        hypotheses, newly_dropped = L.update(hypotheses, probe, outcome)
        dropped.extend(newly_dropped)
        outcomes[probe] = outcome
    return {
        "hypotheses": hypotheses,
        "dropped": dropped,
        "status": L.status(hypotheses, outcomes, budget_left),
    }


def _public_receipt(receipt: Mapping) -> dict:
    probe, outcome = _observed_probe(receipt)
    usable = _receipt_is_usable(receipt)
    return {
        "request_id": _receipt_id(receipt),
        "probe": probe,
        "outcome": outcome if usable else "unknown",
        "measurement": deepcopy(receipt.get("measurement") or {}) if usable else {},
    }


def _public_packet(packet: Mapping) -> dict:
    if not isinstance(packet, Mapping):
        raise ValueError("packet은 JSON 객체여야 한다")
    facts = packet.get("facts", {})
    observed = packet.get("observed", [])
    if not isinstance(facts, Mapping) or not isinstance(observed, list):
        raise ValueError("packet facts/observed 형식이 잘못되었다")
    clean_facts = {}
    for fact_id, value in facts.items():
        if not isinstance(fact_id, str) or not fact_id or not _is_scalar(value):
            raise ValueError("facts는 유한 JSON scalar를 가진 문자열 ID 객체여야 한다")
        clean_facts[fact_id] = value
    previous = packet.get("previous_decision")
    if previous is not None and not isinstance(previous, Mapping):
        raise ValueError("previous_decision은 JSON 객체 또는 null이어야 한다")
    decision_context = None if previous is None else {
        key: deepcopy(previous.get(key))
        for key in ("ranking", "probe", "action", "reason", "claims", "expected_outcomes")
    }
    return {
        "evidence_facts": clean_facts,
        "observations": [_public_receipt(r) for r in observed],
        "budget_left": packet.get("budget_left"),
        "execution_budget_s": packet.get("execution_budget_s"),
        "decision_context": decision_context,
        "mechanisms": list(L.MECHANISMS),
        "probe_catalog": {
            name: {
                "measures": spec["measures"],
                "cost_gpu_s": spec["cost_gpu_s"],
                "args": deepcopy(spec["args"]),
                "expect": deepcopy(spec["expect"]),
            }
            for name, spec in L.PROBES.items()
        },
        "thresholds": {name: list(rule) for name, rule in L.THRESHOLDS.items()},
    }


def build_prompt(packet: Mapping) -> str:
    """Build a deterministic user prompt containing only blind public evidence."""
    if not isinstance(packet, Mapping):
        raise ValueError("packet은 JSON 객체여야 한다")
    phase = packet.get("phase", "initial")
    if not isinstance(phase, str) or not phase:
        raise ValueError("packet.phase는 비어 있지 않은 문자열이어야 한다")
    return (
        f"Decision phase: {phase}. Sampling temperature: 0.\n"
        + json.dumps(_public_packet(packet), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def decision_messages(packet: Mapping) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(packet)},
    ]


def _known_facts(packet: Mapping) -> dict[str, object]:
    facts = dict(packet.get("facts", {}))
    for receipt in packet.get("observed", []):
        for fact_id, value in receipt_facts(receipt).items():
            if fact_id in facts and not _same_value(facts[fact_id], value):
                raise ValueError(f"충돌하는 fact id {fact_id!r}")
            facts[fact_id] = value
    return facts


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def validate_decision(decision: Mapping, packet: Mapping) -> dict:
    """Validate untrusted model JSON and return a fresh canonical object."""
    if not isinstance(decision, Mapping):
        raise ValueError("decision은 JSON 객체여야 한다")
    if set(decision) != {"ranking", "probe", "action", "reason", "claims", "expected_outcomes"}:
        raise ValueError("decision 필드는 공개 계약과 정확히 일치해야 한다")
    ranking = decision.get("ranking")
    if (
        not isinstance(ranking, list)
        or len(ranking) != len(L.MECHANISMS)
        or any(not isinstance(item, str) for item in ranking)
        or len(set(ranking)) != len(ranking)
        or set(ranking) != set(L.MECHANISMS)
    ):
        raise ValueError("ranking은 여섯 mechanism의 중복 없는 순열이어야 한다")

    action, probe, reason = decision.get("action"), decision.get("probe"), decision.get("reason")
    if action not in _ACTIONS:
        raise ValueError("action은 continue/hold/confirm 중 하나여야 한다")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason은 비어 있지 않은 문자열이어야 한다")
    observed = packet.get("observed", [])
    state = evidence_state(observed, packet.get("budget_left"))
    seen_probes = {_observed_probe(receipt)[0] for receipt in observed}
    expected_outcomes = decision.get("expected_outcomes")
    if not isinstance(expected_outcomes, dict):
        raise ValueError("expected_outcomes는 JSON 객체여야 한다")
    if action == "continue":
        if probe not in L.PROBES or probe in seen_probes:
            raise ValueError("continue probe는 미관측 public probe여야 한다")
        if packet.get("budget_left", 0) <= 0:
            raise ValueError("남은 예산 없이 continue할 수 없다")
        expected = {mechanism: L.PROBES[probe]["expect"][mechanism] for mechanism in ranking[:3]}
        if expected_outcomes != expected:
            raise ValueError("expected_outcomes는 선택 probe의 공개 카탈로그와 정확히 일치해야 한다")
    elif probe is not None:
        raise ValueError("hold/confirm의 probe는 null이어야 한다")
    elif expected_outcomes != {}:
        raise ValueError("hold/confirm의 expected_outcomes는 빈 객체여야 한다")
    if observed and not _receipt_is_usable(observed[-1]) and action != "hold":
        raise ValueError("unknown/실패/예산 초과 receipt 뒤에는 hold만 허용한다")
    if action == "confirm":
        if not observed:
            raise ValueError("초기 판단은 confirm할 수 없다")
        if state["status"] != "confirmed" or state["hypotheses"] != [ranking[0]]:
            raise ValueError("confirm은 frozen evidence의 유일한 가설이 raw ranking 1위일 때만 가능하다")

    claims = decision.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims는 하나 이상의 근거 claim이어야 한다")
    known = _known_facts(packet)
    clean_claims, claim_ids = [], set()
    for claim in claims:
        if not isinstance(claim, Mapping) or set(claim) != {"id", "value"}:
            raise ValueError("claim은 id와 value만 가진 객체여야 한다")
        fact_id, value = claim["id"], claim["value"]
        if not isinstance(fact_id, str) or fact_id in claim_ids or fact_id not in known:
            raise ValueError(f"알 수 없거나 중복된 claim id {fact_id!r}")
        if not _is_scalar(value) or not _same_value(value, known[fact_id]):
            raise ValueError(f"claim 값이 evidence fact와 다르다: {fact_id!r}")
        claim_ids.add(fact_id)
        clean_claims.append({"id": fact_id, "value": value})
    if observed:
        latest_prefix = f"receipt:{_receipt_id(observed[-1])}:"
        latest_facts = receipt_facts(observed[-1])
        if latest_facts and not any(fact_id.startswith(latest_prefix) for fact_id in claim_ids):
            raise ValueError("post decision은 최신 receipt fact를 인용해야 한다")

    return {
        "ranking": list(ranking),
        "probe": probe,
        "action": action,
        "reason": reason.strip(),
        "claims": clean_claims,
        "expected_outcomes": dict(expected_outcomes),
    }


def contrast(before: Mapping, after: Mapping, without_receipt: Mapping, observed: Sequence[Mapping]) -> dict[str, bool]:
    """Compute causal-comparison flags without changing or revalidating decisions."""
    before_ranking, after_ranking = before.get("ranking", []), after.get("ranking", [])
    ablated_ranking = without_receipt.get("ranking", [])
    before_leader = before_ranking[0] if before_ranking else None
    after_leader = after_ranking[0] if after_ranking else None
    ablated_leader = ablated_ranking[0] if ablated_ranking else None
    latest_prefix = f"receipt:{_receipt_id(observed[-1])}:" if observed else None
    latest_facts = receipt_facts(observed[-1]) if observed else {}
    cited = bool(latest_prefix) and any(
        isinstance(c, Mapping)
        and c.get("id") in latest_facts
        and _same_value(c.get("value"), latest_facts[c["id"]])
        for c in after.get("claims", [])
    )
    rejected = False
    if observed and before_leader in L.MECHANISMS:
        probe, outcome = _observed_probe(observed[-1])
        if not _receipt_is_usable(observed[-1]):
            outcome = "unknown"
        kept, _ = L.update([before_leader], probe, outcome)
        rejected = not kept
    state = evidence_state(observed, 0)
    after_survives = after_leader in state["hypotheses"] if after_leader is not None else False
    safe_confirm = (
        after.get("action") == "confirm"
        and state["status"] == "confirmed"
        and state["hypotheses"] == [after_leader]
    )
    return {
        "leader_changed": before_leader != after_leader,
        "ranking_changed": list(before_ranking) != list(after_ranking),
        "latest_receipt_cited": cited,
        "changed_leader_rejected_by_actual_observation": bool(before_leader != after_leader and rejected),
        "after_leader_survives_actual_observation": after_survives,
        "ablation_leader_diff": after_leader != ablated_leader,
        "ablation_preserves_before_leader": ablated_leader == before_leader,
        "safe_confirm": safe_confirm,
    }
