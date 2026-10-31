"""검증된 runtime facts를 읽고 관측 뒤 다시 계획하는 순수 진단 선택기."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import diagnostic_fact_ledger_v1 as LEDGER
import recorded_optimizer_probe_v1 as RECORDED
import residual_selection_v2 as S
import runtime_observation_probe_v1 as OBS

SCHEMA = "evidence_guided_selector_state_v1"
RESOURCE = "evidence:optimizer-counts"
NEW_PROBE = RECORDED.PROBE
ACTIONS = {"read_evidence", "run_probe", "hold"}
OUTPUT_FIELDS = {"action", "target", "fact_ids"}
SYSTEM = (
    "Choose one registered action from the supplied context. Return one JSON object with exactly "
    "action, target, fact_ids. Do not provide hidden reasoning, causes, request arguments, repairs, "
    "probabilities, or posterior claims. Use only exact currently visible non-null fact IDs. "
    "A run_probe action requires at least one fact ID. For P_recorded_optimizer_updates, every "
    "cited role/count group must include all three IDs: expected_optimizer_steps, "
    "optimizer_steps_min, optimizer_steps_max; or expected_minibatches, minibatches_min, "
    "minibatches_max. A read_evidence action requires fact_ids=[]; a hold action requires "
    "target=null. Observations may change the next action."
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _review(status: str, codes: list[str], reasons: list[str], action: dict | None = None) -> dict:
    value: dict[str, Any] = {"status": status, "codes": codes, "reasons": reasons}
    if action is not None:
        value["action"] = deepcopy(action)
    return value


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "state_digest"}
    return {**body, "state_digest": _digest(body)}


def _state(value: object) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("schema") == SCHEMA, "selector state schema")
    state = cast(dict[str, Any], value)
    body = {key: item for key, item in state.items() if key != "state_digest"}
    require(state.get("state_digest") == _digest(body), "selector state digest/tamper")
    require(type(state.get("budget_left")) is int and 0 <= state["budget_left"] <= 3
            and isinstance(state.get("hypotheses"), list)
            and isinstance(state.get("observed"), dict)
            and isinstance(state.get("visible_facts"), dict)
            and type(state.get("evidence_read")) is bool,
            "selector state shape")
    _replay_state(state)
    return state


def initialize(packet: dict, fact_ledger_path: Path, *, budget_left: int = 3) -> dict[str, Any]:
    """기존 packet 후보를 보존하고 runtime count는 read action 전까지 감춘다."""
    value = S.validate_packet(packet)
    require(type(budget_left) is int and budget_left == value["budget_left"]
            and 0 <= budget_left <= 3, "selector budget은 packet budget과 같아야 한다")
    fact_path = Path(fact_ledger_path).resolve()
    checked = LEDGER.checked(fact_path)
    source_binding = checked.get("source_binding")
    input_bindings = value.get("input_bindings")
    if not isinstance(source_binding, dict) or not isinstance(input_bindings, dict):
        raise TypeError("packet/fact ledger source binding이 없다")
    for role in ("case", "reference"):
        ledger_role = source_binding.get(role)
        packet_role = input_bindings.get(role)
        require(isinstance(ledger_role, dict) and isinstance(packet_role, dict)
                and ledger_role.get("checkpoint_sha256") == packet_role.get("checkpoint_sha256"),
                f"{role} packet checkpoint와 fact ledger source가 다르다")
    visible = {key: deepcopy(item) for key, item in value["facts"].items()
               if not key.startswith(("runtime:", "derived:"))}
    return _seal({
        "schema": SCHEMA, "hypotheses": list(value["remaining_candidates"]),
        "observed": dict(value["observed"]), "visible_facts": visible,
        "evidence_read": False, "budget_left": budget_left, "status": "open",
        "fact_ledger": {"path": str(fact_path), "sha256": LEDGER.sha(fact_path),
                        "source_binding": deepcopy(checked["source_binding"])},
        "initial_packet": deepcopy(value),
        "history": [],
        "scope": {"model_hidden_paths": True, "fault_labels_exposed": False,
                  "posterior_or_eig": False, "host_generates_args_and_repairs": True},
    })


def _eligible(state: dict[str, Any]) -> list[str]:
    if state["status"] != "open" or state["budget_left"] == 0:
        return []
    registry = S.C1.registry()
    legacy = [probe for probe in registry["probes"]
              if S._probe_semantic(probe, state["hypotheses"], state["observed"])[0]]
    count_allowed = (NEW_PROBE not in state["observed"]
                     and "optimizer_update_count" in state["hypotheses"])
    return [*legacy, *([NEW_PROBE] if count_allowed else [])]


def _catalog() -> list[dict[str, Any]]:
    registry = S.C1.registry()
    rows = [{"target": name, "measures": spec["measures"],
             "cost_gpu_s": spec["cost_gpu_s"], "registry_order": index}
            for index, (name, spec) in enumerate(registry["probes"].items())]
    rows.append({"target": NEW_PROBE,
                 "measures": "verified configured and observed optimizer/minibatch counts",
                 "cost_gpu_s": 0, "registry_order": len(rows),
                 "outcome_contract": "normal iff both role-bound recorded counts match config; else abnormal/unknown"})
    return rows


def context(state: dict[str, Any]) -> dict[str, Any]:
    value = _state(state)
    resources = [] if value["evidence_read"] else [{"target": RESOURCE,
        "description": "verified optimizer step and minibatch expected/observed facts"}]
    return {
        "schema": "evidence_guided_selector_context_v1",
        "evidence_facts": deepcopy(value["visible_facts"]),
        "remaining_candidates": list(value["hypotheses"]),
        "observations": [{"target": target, "outcome": outcome}
                         for target, outcome in value["observed"].items()],
        "available_evidence": resources, "probe_catalog": _catalog(),
        "eligible_probes": _eligible(value), "budget_left": value["budget_left"],
        "status": value["status"],
        "tie_break": "equal registered choices use ascending registry_order",
        "action_contract": {
            "fact_ids": "exact currently visible non-null IDs only",
            "run_probe": "at least one fact ID is required",
            "P_recorded_optimizer_updates": (
                "for each cited role/count group include expected plus observed min and max IDs"),
            "optimizer_triplet": ["expected_optimizer_steps", "optimizer_steps_min",
                                  "optimizer_steps_max"],
            "minibatch_triplet": ["expected_minibatches", "minibatches_min", "minibatches_max"],
            "read_evidence": {"fact_ids": []}, "hold": {"target": None},
        },
    }


def messages(state: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(
                context(state), ensure_ascii=False, sort_keys=True, allow_nan=False)}]


def _format(content: object) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return _review("invalid", ["F_EMPTY"], ["응답이 비어 있다"])
    try:
        action = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _review("invalid", ["F_JSON"], ["JSON 객체 하나여야 한다"])
    if not isinstance(action, dict) or set(action) != OUTPUT_FIELDS:
        return _review("invalid", ["F_FIELDS"], ["action,target,fact_ids만 허용한다"])
    codes, reasons = [], []
    if action.get("action") not in ACTIONS:
        codes.append("F_ACTION")
        reasons.append("등록 action이 아니다")
    target, kind = action.get("target"), action.get("action")
    if ((kind == "hold" and target is not None)
            or (kind != "hold" and (not isinstance(target, str) or not target))):
        codes.append("F_TARGET")
        reasons.append("action과 target 형식이 다르다")
    fact_ids = action.get("fact_ids")
    if (not isinstance(fact_ids, list)
            or any(not isinstance(item, str) or not item for item in fact_ids)
            or len(set(fact_ids)) != len(fact_ids)):
        codes.append("F_FACT_IDS")
        reasons.append("fact_ids는 중복 없는 문자열 목록이어야 한다")
    return (_review("invalid", codes, reasons) if codes
            else _review("valid", [], [], cast(dict[str, Any], action)))


def _semantic(action: dict[str, Any], state: dict[str, Any], *, model: bool) -> dict[str, Any]:
    value = _state(state)
    kind, target, fact_ids = action["action"], action["target"], action["fact_ids"]
    codes, reasons = [], []
    unknown = [item for item in fact_ids if item not in value["visible_facts"]]
    if unknown:
        codes.append("S_FACT_UNREAD")
        reasons.append("조회하지 않았거나 등록되지 않은 fact ID다")
    elif any(value["visible_facts"][item] is None for item in fact_ids):
        codes.append("S_FACT_NULL")
        reasons.append("값이 없는 fact는 근거로 인용할 수 없다")
    if kind == "read_evidence":
        if target != RESOURCE:
            codes.append("S_EVIDENCE_UNKNOWN")
            reasons.append("등록 evidence resource가 아니다")
        if value["evidence_read"]:
            codes.append("S_EVIDENCE_DUPLICATE")
            reasons.append("이미 읽은 evidence resource다")
        if fact_ids:
            codes.append("S_FACT_UNREAD")
            reasons.append("read 전 count fact를 인용할 수 없다")
    elif kind == "run_probe":
        if target not in _eligible(value):
            codes.append("S_PROBE_INELIGIBLE")
            reasons.append("미등록·중복·비정보 probe다")
        if model and not fact_ids:
            codes.append("S_FACT_IDS")
            reasons.append("모델 probe 제안은 조회한 근거 ID가 필요하다")
        if target == NEW_PROBE and not codes:
            try:
                LEDGER.validate_proposal_fact_ids(value["visible_facts"], fact_ids)
            except ValueError as error:
                codes.append("S_COUNT_FACT_PAIR")
                reasons.append(str(error))
    elif kind == "hold" and target is not None:
        codes.append("S_HOLD_TARGET")
        reasons.append("hold target은 null이어야 한다")
    return (_review("invalid", codes, reasons) if codes
            else _review("valid", [], [], action))


def assess(content: object, state: dict[str, Any]) -> dict[str, Any]:
    formatted = _format(content)
    if formatted["status"] != "valid":
        return {**formatted, "format_status": "invalid", "semantic_status": "not_run"}
    semantic = _semantic(formatted["action"], state, model=True)
    return {**semantic, "format_status": "valid", "semantic_status": semantic["status"]}


def select_model(review: dict, state: dict[str, Any]) -> dict[str, Any]:
    _state(state)
    if review.get("status") != "valid":
        return {"action": "hold", "target": None, "fact_ids": []}
    action = review.get("action")
    if not isinstance(action, dict):
        raise TypeError("저장된 valid model action 형식")
    semantic = _semantic(action, state, model=True)
    require(semantic["status"] == "valid", "저장된 valid model action 재검증 실패")
    return deepcopy(action)


def _count_ids() -> list[str]:
    names = ("expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max",
             "expected_minibatches", "minibatches_min", "minibatches_max")
    return [f"runtime:{role}:{name}" for role in ("case", "reference") for name in names]


def _request_scope(state: dict[str, Any], history: list[dict[str, Any]] | None = None) -> str:
    return _digest({"initial_packet": state["initial_packet"],
                    "fact_ledger": state["fact_ledger"],
                    "history": state["history"] if history is None else history})


def _request_ids(scope: str, target: str) -> dict[str, str]:
    return {role: "selector-" + hashlib.sha256(
        f"{scope}:{target}:{role}".encode()).hexdigest()[:32]
            for role in ("case", "reference")}


def select_strong_rule(state: dict[str, Any]) -> dict[str, Any]:
    """count evidence 조건 외에는 기존 registry next를 그대로 쓴다."""
    value = _state(state)
    if value["status"] != "open" or value["budget_left"] == 0:
        return {"action": "hold", "target": None, "fact_ids": []}
    if not value["evidence_read"]:
        return {"action": "read_evidence", "target": RESOURCE, "fact_ids": []}
    facts = value["visible_facts"]
    case_mismatch = (facts.get("derived:case:step_count_matches_config") is False
                     or facts.get("derived:case:minibatch_count_matches_config") is False)
    reference_normal = (facts.get("derived:reference:step_count_matches_config") is True
                        and facts.get("derived:reference:minibatch_count_matches_config") is True)
    if case_mismatch and reference_normal and NEW_PROBE in _eligible(value):
        return {"action": "run_probe", "target": NEW_PROBE, "fact_ids": _count_ids()}
    registry = S.C1.registry()
    legacy_observed = {key: item for key, item in value["observed"].items()
                       if key in registry["probes"]}
    probe = S.M._next(registry, value["hypotheses"], legacy_observed)
    if probe is None or probe not in _eligible(value):
        return {"action": "hold", "target": None, "fact_ids": []}
    return {"action": "run_probe", "target": probe, "fact_ids": []}


def request_spec(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """root adapter가 승인 전에 결합할 probe args와 source snapshot을 반환한다."""
    value = _state(state)
    require(set(action) == OUTPUT_FIELDS and _semantic(action, value, model=False)["status"] == "valid",
            "request spec action이 현재 state와 다르다")
    kind, target = action["action"], action["target"]
    if kind == "read_evidence":
        args: dict[str, Any] = {"resource": RESOURCE}
    elif kind == "run_probe" and target == NEW_PROBE:
        args = {"roles": ["case", "reference"], "params": deepcopy(RECORDED.PARAMS)}
    elif kind == "run_probe":
        args = {"roles": ["case", "reference"],
                "params": deepcopy(S.C1.registry()["probes"][target]["args"])}
    else:
        args = {}
    scope = _request_scope(value)
    request_ids = (_request_ids(scope, str(target))
                   if kind == "run_probe" and target == NEW_PROBE else None)
    return {"schema": "evidence_guided_selector_request_spec_v1",
            "state_digest": value["state_digest"], "action": deepcopy(action),
            "request_scope_digest": scope, "request_ids": request_ids,
            "source_binding": deepcopy(value["fact_ledger"]["source_binding"]),
            "host_args": args, "approval_granted": False,
            "model_generated_args": False, "model_generated_cause": False,
            "model_generated_repair": False}


def _source_match(
    role: str, checkpoint: object, binding_digest: object, request_id: object,
    source_binding: dict[str, Any], expected_request_id: str | None, request_ids: set[str],
) -> str:
    expected = source_binding.get(role)
    require(isinstance(expected, dict)
            and checkpoint == expected.get("checkpoint_sha256")
            and binding_digest == expected.get("binding_digest"),
            f"{role} receipt checkpoint/runtime binding이 fact ledger와 다르다")
    if not isinstance(request_id, str):
        raise TypeError("paired receipt request ID 형식")
    require((expected_request_id is None or request_id == expected_request_id)
            and request_id not in request_ids,
            "paired receipt request ID가 selector request와 다르거나 중복이다")
    request_ids.add(request_id)
    return request_id


def _file_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), "proof 파일이 없다")
    return {"path": str(path), "sha256": RECORDED.sha(path)}


def _legacy_source(
    role: str, plan: dict[str, Any], receipt: dict[str, Any],
    fact_ledger: dict[str, Any],
) -> dict[str, Any]:
    ledger_path = Path(fact_ledger["path"])
    require(ledger_path.is_file() and LEDGER.sha(ledger_path) == fact_ledger["sha256"],
            "legacy fact ledger proof가 바뀌었다")
    checked = LEDGER.checked(ledger_path)
    require(checked.get("source_binding") == fact_ledger["source_binding"],
            "legacy fact ledger source가 바뀌었다")
    runtime_ref = checked.get("private_source_refs", {}).get(role)
    collected_ref = plan.get("collected_input")
    require(isinstance(runtime_ref, dict) and isinstance(collected_ref, dict),
            f"{role} legacy typed source ref가 없다")
    runtime_ref = cast(dict[str, Any], runtime_ref)
    collected_ref = cast(dict[str, Any], collected_ref)
    runtime_path, collected_path = Path(runtime_ref["path"]), Path(collected_ref["path"])
    require(runtime_path.is_file() and RECORDED.sha(runtime_path) == runtime_ref.get("sha256")
            and collected_path.is_file()
            and RECORDED.sha(collected_path) == collected_ref.get("sha256"),
            f"{role} legacy typed source bytes가 바뀌었다")
    source = RECORDED._source(collected_path, runtime_path, role)
    expected = fact_ledger["source_binding"].get(role)
    require(isinstance(expected, dict)
            and source["checkpoint_sha256"] == receipt.get("checkpoint_sha256")
            == expected.get("checkpoint_sha256")
            and source["runtime_binding_digest"] == expected.get("binding_digest")
            and source["collected_binding_digest"] == receipt.get("binding_digest")
            and source["telemetry_sha256"] == receipt.get("telemetry_sha256"),
            f"{role} legacy collected/runtime typed binding이 다르다")
    return source


def _legacy_approval(
    paths: tuple[Path, Path, Path, Path], *, state_digest: str, action: dict[str, Any],
    scope: str, source_binding: dict[str, Any], expected_ref: dict[str, Any] | None,
) -> dict[str, str]:
    case_plan, _case_receipt, reference_plan, _reference_receipt = paths
    root = case_plan.parent.parent
    require(reference_plan.parent.parent == root, "legacy paired plan root가 다르다")
    path = root / "root-approval.json"
    proof = _file_ref(path)
    if expected_ref is not None:
        require(expected_ref == proof, "history root approval proof가 바뀌었다")
    value = RECORDED.read(path)
    spec = value.get("request_spec")
    expected_plans = {"case": _file_ref(case_plan), "reference": _file_ref(reference_plan)}
    require(value.get("schema") == "evidence_guided_probe_host_approval_v1"
            and value.get("selector_contract")
            == "new_evidence_guided_selector_not_legacy_model_dialogue"
            and value.get("state_digest") == state_digest and value.get("action") == action
            and value.get("plans") == expected_plans and value.get("approval_granted") is True
            and isinstance(value.get("actor"), str) and bool(value["actor"].strip()),
            "legacy root approval state/action/plan 결합이 다르다")
    require(isinstance(spec, dict) and spec.get("schema")
            == "evidence_guided_selector_request_spec_v1"
            and spec.get("state_digest") == state_digest and spec.get("action") == action
            and spec.get("request_scope_digest") == scope and spec.get("request_ids") is None
            and spec.get("source_binding") == source_binding
            and spec.get("host_args", {}).get("params")
            == S.C1.registry()["probes"][action["target"]]["args"]
            and spec.get("approval_granted") is False
            and spec.get("model_generated_args") is False
            and spec.get("model_generated_cause") is False
            and spec.get("model_generated_repair") is False,
            "legacy root approval selector request scope가 다르다")
    return proof


def _legacy_outcome(
    target: str, paths: tuple[Path, Path, Path, Path], fact_ledger: dict[str, Any],
    *, request_scope: str, state_digest: str, action: dict[str, Any],
    expected_approval: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str], dict[str, str]]:
    case_plan, case_receipt, reference_plan, reference_receipt = paths
    case = OBS.verify_receipt(case_plan, case_receipt)
    reference = OBS.verify_receipt(reference_plan, reference_receipt)
    require(case.get("role") == "case" and reference.get("role") == "reference"
            and case.get("probe") == reference.get("probe") == target,
            "paired legacy receipt role/probe 결합")
    case_plan_value, reference_plan_value = OBS.read(case_plan), OBS.read(reference_plan)
    _legacy_source("case", case_plan_value, case, fact_ledger)
    _legacy_source("reference", reference_plan_value, reference, fact_ledger)
    case_raw, reference_raw = OBS.read(case_receipt), OBS.read(reference_receipt)
    case_auth, reference_auth = case_raw.get("authorization"), reference_raw.get("authorization")
    require(isinstance(case_auth, dict) and case_auth == reference_auth
            and set(case_auth) == {"path", "sha256", "request_id"},
            "paired legacy receipt는 동일 authorization ref를 공유해야 한다")
    case_auth = cast(dict[str, Any], case_auth)
    request_id = case_auth.get("request_id")
    require(isinstance(request_id, str) and bool(request_id), "legacy shared request ID 형식")
    request_id = cast(str, request_id)
    actual_requests = {"case": request_id, "reference": request_id}
    approval = _legacy_approval(
        paths, state_digest=state_digest, action=action, scope=request_scope,
        source_binding=fact_ledger["source_binding"], expected_ref=expected_approval)
    if case["status"] != "complete" or reference["status"] != "complete":
        return "unknown", actual_requests, approval
    return (OBS.classify_raw(target, case["measurement"], reference["measurement"]),
            actual_requests, approval)


def _recorded_outcome(
    paths: tuple[Path, Path, Path, Path], source_binding: dict[str, Any],
    expected_requests: dict[str, str],
) -> tuple[str, dict[str, str]]:
    case_plan, case_receipt, reference_plan, reference_receipt = paths
    case = RECORDED.checked_receipt(case_plan, case_receipt)
    reference = RECORDED.checked_receipt(reference_plan, reference_receipt)
    require(case.get("role") == "case" and reference.get("role") == "reference"
            and case.get("probe") == reference.get("probe") == NEW_PROBE,
            "paired recorded receipt role/probe 결합")
    requests: set[str] = set()
    actual_requests = {}
    for role, receipt in (("case", case), ("reference", reference)):
        receipt_binding = receipt.get("source_binding", {})
        actual_requests[role] = _source_match(
            role, receipt_binding.get("checkpoint_sha256"),
            receipt_binding.get("runtime_binding_digest"), receipt.get("request_id"),
            source_binding, expected_requests[role], requests)
    outcome = case["outcome"] if reference["outcome"] == "normal" else "unknown"
    return outcome, actual_requests


def _proof_paths(row: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    proof = row.get("proof")
    if not isinstance(proof, dict):
        raise TypeError("history paired receipt proof 형식")
    required = {"case_plan", "case_receipt", "reference_plan", "reference_receipt"}
    require(set(proof) in (required, required | {"root_approval"}),
            "history paired receipt proof가 없다")
    for ref in proof.values():
        require(isinstance(ref, dict), "history receipt ref 형식")
        path = Path(ref["path"])
        require(path.is_file() and RECORDED.sha(path) == ref["sha256"],
                "history receipt/plan proof bytes가 바뀌었다")
    return cast(tuple[Path, Path, Path, Path], tuple(Path(proof[name]["path"]) for name in (
        "case_plan", "case_receipt", "reference_plan", "reference_receipt")))


def _apply_outcome(
    hypotheses: list[str], observed: dict[str, str], target: str, outcome: str,
) -> tuple[list[str], dict[str, str]]:
    registry = S.C1.registry()
    if target == NEW_PROBE:
        if outcome != "unknown":
            expected = {name: ("abnormal" if name == "optimizer_update_count" else "normal")
                        for name in registry["mechanisms"]}
            hypotheses = [name for name in hypotheses if expected[name] == outcome]
    else:
        hypotheses, _dropped = S.M.update(registry, hypotheses, target, outcome)
    observed = {**observed, target: outcome}
    return hypotheses, observed


def _status(
    hypotheses: list[str], observed: dict[str, str], target: str, outcome: str, budget: int,
) -> str:
    registry = S.C1.registry()
    if outcome == "unknown":
        return "unidentifiable"
    if target == NEW_PROBE and outcome == "abnormal" \
            and hypotheses == ["optimizer_update_count"]:
        return "confirmed"
    legacy = {key: item for key, item in observed.items() if key in registry["probes"]}
    return S.M._status(registry, hypotheses, legacy, budget)


def _replay_state(state: dict[str, Any]) -> None:
    initial_packet = state.get("initial_packet")
    if not isinstance(initial_packet, dict):
        raise TypeError("initial packet 형식")
    packet = S.validate_packet(initial_packet)
    source_binding = state["fact_ledger"]["source_binding"]
    for role in ("case", "reference"):
        require(packet["input_bindings"][role]["checkpoint_sha256"]
                == source_binding[role]["checkpoint_sha256"],
                f"replay {role} packet/source checkpoint가 다르다")
    hypotheses, observed = list(packet["remaining_candidates"]), dict(packet["observed"])
    visible = {key: deepcopy(item) for key, item in packet["facts"].items()
               if not key.startswith(("runtime:", "derived:"))}
    budget, status, evidence_read = packet["budget_left"], "open", False
    prior_history: list[dict[str, Any]] = []
    for row in state["history"]:
        require(status == "open", "종료 state 뒤 history가 있다")
        if row.get("action") == "read_evidence":
            require(not evidence_read and row.get("target") == RESOURCE, "history evidence read 중복/target")
            ref = state["fact_ledger"]
            path = Path(ref["path"])
            require(path.is_file() and LEDGER.sha(path) == ref["sha256"],
                    "history fact ledger proof가 바뀌었다")
            checked = LEDGER.checked(path)
            require(checked["source_binding"] == source_binding, "history fact ledger source 변경")
            visible.update(deepcopy(checked["model_view"]["facts"]))
            evidence_read = True
            prior_history.append(row)
            continue
        if row.get("action") == "hold":
            status = "hold"
            prior_history.append(row)
            continue
        require(row.get("action") == "run_probe" and budget > 0, "history probe action/budget")
        paths = _proof_paths(row)
        target = row["target"]
        scope = _request_scope(state, prior_history)
        expected_requests = _request_ids(scope, target) if target == NEW_PROBE else None
        require(row.get("request_scope_digest") == scope
                and (target != NEW_PROBE or row.get("request_ids") == expected_requests),
                "history selector request scope가 다르다")
        if target == NEW_PROBE:
            outcome, actual_requests = _recorded_outcome(
                paths, source_binding, _request_ids(scope, target))
        else:
            replay_state = deepcopy(state)
            replay_state.update(hypotheses=deepcopy(hypotheses), observed=deepcopy(observed),
                                visible_facts=deepcopy(visible), budget_left=budget,
                                status=status, evidence_read=evidence_read,
                                history=deepcopy(prior_history))
            replay_state = _seal(replay_state)
            replay_action = {name: deepcopy(row[name]) for name in OUTPUT_FIELDS}
            proof = cast(dict[str, Any], row["proof"])
            outcome, actual_requests, approval = _legacy_outcome(
                target, paths, state["fact_ledger"], request_scope=scope,
                state_digest=replay_state["state_digest"], action=replay_action,
                expected_approval=cast(dict[str, Any], proof.get("root_approval")))
            require(proof.get("root_approval") == approval,
                    "history legacy root approval 재검증이 다르다")
        require(outcome == row.get("outcome") and row.get("request_ids") == actual_requests
                and target not in observed,
                "history receipt outcome/중복 재검증이 다르다")
        hypotheses, observed = _apply_outcome(hypotheses, observed, target, outcome)
        budget -= 1
        status = _status(hypotheses, observed, target, outcome, budget)
        prior_history.append(row)
    require(state["hypotheses"] == hypotheses and state["observed"] == observed
            and state["visible_facts"] == visible and state["budget_left"] == budget
            and state["status"] == status and state["evidence_read"] == evidence_read,
            "selector state가 initial packet+history 재생과 다르다")


def advance(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    case_plan: Path | None = None,
    case_receipt: Path | None = None,
    reference_plan: Path | None = None,
    reference_receipt: Path | None = None,
) -> dict[str, Any]:
    """검증된 read 또는 paired receipt 관측 뒤 후보를 갱신해 다시 계획 가능하게 한다."""
    value = deepcopy(_state(state))
    require(set(action) == OUTPUT_FIELDS, "advance action shape")
    semantic = _semantic(action, value, model=False)
    require(semantic["status"] == "valid", "advance action semantic")
    if action["action"] == "hold":
        value["status"] = "hold"
        value["history"].append(deepcopy(action))
        return _seal(value)
    if action["action"] == "read_evidence":
        ref = value["fact_ledger"]
        path = Path(ref["path"])
        require(path.is_file() and LEDGER.sha(path) == ref["sha256"], "fact ledger bytes 변경")
        checked = LEDGER.checked(path)
        require(checked["source_binding"] == ref["source_binding"], "fact ledger source binding 변경")
        value["visible_facts"].update(deepcopy(checked["model_view"]["facts"]))
        value["evidence_read"] = True
        value["history"].append(deepcopy(action))
        return _seal(value)
    paths = (case_plan, case_receipt, reference_plan, reference_receipt)
    require(all(isinstance(path, Path) for path in paths), "run_probe에는 paired receipt 경로가 필요하다")
    typed_paths = cast(tuple[Path, Path, Path, Path], paths)
    typed_case_plan, typed_case_receipt, typed_reference_plan, typed_reference_receipt = typed_paths
    target = action["target"]
    source_binding = value["fact_ledger"]["source_binding"]
    request_scope = _request_scope(value)
    if target == NEW_PROBE:
        outcome, actual_requests = _recorded_outcome(
            typed_paths, source_binding, _request_ids(request_scope, target))
        approval = None
    else:
        outcome, actual_requests, approval = _legacy_outcome(
            target, typed_paths, value["fact_ledger"], request_scope=request_scope,
            state_digest=value["state_digest"], action=action)
    require(outcome in {"normal", "abnormal", "unknown"}, "등록 outcome이 아니다")
    value["hypotheses"], value["observed"] = _apply_outcome(
        value["hypotheses"], value["observed"], target, outcome)
    value["budget_left"] -= 1
    proof = {
        "case_plan": {"path": str(typed_case_plan.resolve()),
                      "sha256": RECORDED.sha(typed_case_plan)},
        "case_receipt": {"path": str(typed_case_receipt.resolve()),
                         "sha256": RECORDED.sha(typed_case_receipt)},
        "reference_plan": {"path": str(typed_reference_plan.resolve()),
                           "sha256": RECORDED.sha(typed_reference_plan)},
        "reference_receipt": {"path": str(typed_reference_receipt.resolve()),
                              "sha256": RECORDED.sha(typed_reference_receipt)},
    }
    if approval is not None:
        proof["root_approval"] = approval
    value["history"].append({**deepcopy(action), "outcome": outcome, "proof": proof,
                             "request_scope_digest": request_scope,
                             "request_ids": actual_requests})
    value["status"] = _status(
        value["hypotheses"], value["observed"], target, outcome, value["budget_left"])
    return _seal(value)
