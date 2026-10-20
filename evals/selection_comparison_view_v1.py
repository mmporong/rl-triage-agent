"""동일 정보의 검사 선택과 shadow 대조를 위한 읽기 전용 CPU 계약."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import residual_selection_v2 as S  # noqa: E402

C, M, L = S.C1, S.M, S.C1.L
POLICIES = ("model_first_then_rule", "model_adaptive", "registry_next", "fixed_eligible")
PROTOCOL = ROOT / "bench/protocols/selection_comparison_view_v1.json"
SOURCE_FILES = (
    "evals/selection_comparison_view_v1.py", "tests/test_selection_comparison_view_v1.py",
    "evals/residual_selection_v2.py", "evals/residual_contract_v1.py",
    "evals/r1_repair_contract.py", "evals/correction_model_call.py",
    "src/rl_triage/__init__.py", "src/rl_triage/active_diagnosis.py",
    "src/rl_triage/correction_registry.py", "src/rl_triage/mechanism_registry.py",
    "src/rl_triage/probe_loop.py",
)
TELEMETRY_FIELDS = (
    "Episode_Reward/action_rate_l2", "Episode_Reward/ang_vel_xy_l2", "Episode_Reward/dof_acc_l2",
    "Episode_Reward/dof_pos_limits", "Episode_Reward/dof_torques_l2", "Episode_Reward/feet_air_time",
    "Episode_Reward/flat_orientation_l2", "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/track_ang_vel_z_exp", "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Termination/base_contact", "Episode_Termination/time_out", "Loss/entropy",
    "Loss/learning_rate", "Loss/surrogate", "Loss/value_function", "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw", "Perf/collection time", "Perf/learning_time", "Perf/total_fps",
    "Policy/mean_noise_std", "Train/mean_episode_length", "Train/mean_reward",
)
BASE_FACT_IDS = frozenset({"config:action_scale"} | {
    f"telemetry:{role}:{field}" for role in ("case", "reference") for field in TELEMETRY_FIELDS
})
PUBLIC_NUMBERS = {
    "P_value": ("critic_change", "actor_change", "case/critic_change", "case/actor_change",
                "reference/critic_change", "reference/actor_change"),
    "P_noise": ("noise_ratio", "logged_std_mean"),
    "P_reward": ("track_lin_vel_xy_exp_rel_error", "track_ang_vel_z_exp_rel_error"),
    "P_torque": ("low_speed_saturation", "velocity_limit", "effort_limit"),
    "P_physics": ("mismatch_count", "static_friction_mean", "dynamic_friction_mean",
                  "mass_delta_min", "mass_delta_max", "physics_dt"),
    "P_episode": ("timeout_count", "median_timeout_step", "timeout_ratio"),
    C.NEW_PROBE: tuple(f"{role}/{field}/{index}" for role in ("case", "control")
                       for field in ("optimizer_steps", "minibatches") for index in range(3)),
}
SYSTEM = """Select a registered diagnostic probe using only the shared view. Return one JSON object
with exactly action, probe, fact_ids. action is continue or hold. continue requires an eligible probe;
hold requires probe=null. fact_ids is a nonempty list of distinct IDs in evidence_facts.
Each expect entry describes a registered subtype's predicted normal/abnormal outcome. null means
no prediction: it neither rejects nor supports that subtype. Reject a subtype only when a non-null
expectation contradicts the observed outcome. unknown stops execution without a confirmation.
One remaining subtype is confirmed only by its uniquely abnormal registered confirmation probe.
The host computes candidates, eligibility, arguments, thresholds and execution authorization.
Catalog cost_gpu_s is a registered planning cost, not a measured GPU-kernel duration.
An executed probe with outcome_hidden has consumed budget but supplies no outcome or measurement.
Do not invent fact values, arguments, causes or execution approval. Output JSON only."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_protocol(expected: str) -> dict:
    """로컬 파일 결합 검사다. 커밋의 조상 관계 인증은 향후 controller가 수행한다."""
    raw = PROTOCOL.read_bytes().replace(b"\r\n", b"\n")
    require(S._valid_sha(expected) and hashlib.sha256(raw).hexdigest() == expected, "프로토콜 SHA 불일치")
    value = json.loads(raw)
    require(value.get("schema") == "selection_comparison_view_protocol_v1"
            and value.get("measurement_kind") == "synthetic_cpu_contract_only"
            and set(value.get("source_sha256_lf", {})) == set(SOURCE_FILES), "프로토콜 범위 또는 source 목록")
    require(all(file_digest(ROOT / path) == sha for path, sha in value["source_sha256_lf"].items()),
            "결합 source 변경")
    return value


def epoch_binding(packet: dict, run_id: str, policy: str, protocol_sha256: str,
                  parent: dict | None = None) -> dict:
    """부모의 구조적 결합이다. 실제 교정 receipt 인증은 controller 책임이다."""
    packet = S.validate_packet(packet)
    verify_protocol(protocol_sha256)
    require(isinstance(run_id, str) and bool(re.fullmatch(r"[a-z0-9_]{1,80}", run_id)), "비교 tag 불일치")
    require(policy in POLICIES and S._valid_sha(protocol_sha256), "정책 또는 프로토콜 SHA 불일치")
    require(type(packet["execution_budget_s"]) is int, "실행 시간 상한은 정수여야 함")
    for key, value in packet["facts"].items():
        require(key in BASE_FACT_IDS
                and (value is None or isinstance(value, bool) or number(value)), "공개 base 수치 fact가 아님")
    if packet["epoch"] == 1:
        require(parent is None, "첫 epoch의 부모 입력")
    else:
        prior = packet["prior_receipt"]
        expected = {"run_id": run_id, "policy": policy, "epoch": 1, **prior}
        require(parent == expected and prior["checkpoint_sha256"]
                == packet["input_bindings"]["case"]["checkpoint_sha256"], "다른 arm 또는 출력 checkpoint의 부모")
    return {"schema": "selection_comparison_epoch_v1", "run_id": run_id, "policy": policy,
            "packet_digest": L.digest(packet), "protocol_sha256": protocol_sha256,
            "parent": deepcopy(parent)}


def eligible(registry: dict, candidates: list, observed: dict, probe: str) -> bool:
    if probe not in registry["probes"] or probe in observed or not candidates:
        return False
    if len(candidates) == 1:
        return M._confirming(registry, candidates[0]) == probe
    predictions = {registry["probes"][probe]["expect"][item] for item in candidates}
    return {"normal", "abnormal"} <= predictions


def read_history(path: Path, binding: dict, maximum: int | float) -> tuple[list, str, str | None]:
    """원장을 쓰지 않고 context·인자·생명주기를 검사한다."""
    ledger = M.RegistryLedger(path, C.registry())
    events = ledger.events()
    require(all(isinstance(event, dict) and event.get("event") in
                ("proposed", "approved", "consumed", "receipt")
                and isinstance(event.get("request_id"), str) and bool(event["request_id"])
                for event in events), "미등록 원장 event")
    requests = ledger.requests()
    rows, expected_events, pending = [], [], None
    require(sum(event["event"] == "proposed" for event in events) == len(requests), "중복 요청 ID")
    order = {"pending": ["proposed"], "approved": ["proposed", "approved"],
             "running": ["proposed", "approved", "consumed"],
             "done": ["proposed", "approved", "consumed", "receipt"]}
    for request_id, request in requests.items():
        require(pending is None, "미완결 요청 뒤 다른 요청")
        probe = request["probe"]
        require(probe in C.registry()["probes"] and request["prereg_digest"] == L.digest(binding)
                and request["registry_digest"] == ledger.registry_digest
                and request["args_digest"] == L.digest(C.registry()["probes"][probe]["args"])
                and type(request["budget_gpu_s"]) is int
                and request["budget_gpu_s"] == maximum, "다른 context 또는 요청 인자·상한")
        effect = ledger.effect_key(request["prereg_digest"], probe, request["args_digest"])
        items = [event for event in events if event.get("request_id") == request_id]
        require([event["event"] for event in items] == order.get(request["state"]), "원장 생명주기 불일치")
        for event in items:
            require(event.get("registry_digest") == ledger.registry_digest
                    and event.get("effect_key") == effect, "중간 원장 결합 불일치")
            if event["event"] == "approved":
                require(isinstance(event.get("approver"), str) and bool(event["approver"].strip()), "승인 주체 없음")
            else:
                require(event.get("args_digest") == request["args_digest"], "중간 원장 인자 불일치")
        expected_events.extend(items)
        if request["state"] != "done":
            pending = probe
            continue
        receipt = request["receipt"]
        require(receipt["outcome"] in L.OUTCOMES and number(receipt["gpu_s"]) and receipt["gpu_s"] >= 0
                and (receipt["exit_code"] is None or type(receipt["exit_code"]) is int)
                and type(receipt["over_budget"]) is bool
                and receipt["over_budget"] == (receipt["gpu_s"] > request["budget_gpu_s"]), "잘못된 실행 receipt")
        if receipt["exit_code"] != 0 or receipt["over_budget"]:
            require(receipt["outcome"] == "unknown", "실패 실행을 유효 판정으로 사용")
        require(isinstance(receipt["measurement"], dict), "측정 객체 아님")
        canonical(receipt["measurement"])
        rows.append({"probe": probe, "outcome": receipt["outcome"],
                     "measurement": deepcopy(receipt["measurement"]), "gpu_s": receipt["gpu_s"]})
    require(events == expected_events == ledger.events(), "원장 교차 실행 또는 읽기 중 변경")
    return rows, L.digest(events), pending


def _measurement_facts(index: int, row: dict) -> dict:
    prefix = f"observation:{index}:{row['probe']}:"
    result = {prefix + "outcome": row["outcome"]}
    for field in PUBLIC_NUMBERS[row["probe"]]:
        value = row["measurement"]
        try:
            for key in field.split("/"):
                value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError):
            continue
        except (TypeError, ValueError) as error:
            raise ValueError("등록 공개 측정 구조가 아님") from error
        require(value is None or number(value), "등록 공개 측정 수치가 아님")
        result[prefix + field] = value
    return result


def _view(packet: dict, rows: list, pending: str | None, masked: bool) -> dict:
    registry = C.registry()
    candidates, observed, states = list(packet["remaining_candidates"]), {}, []
    for index, row in enumerate(rows):
        require(M._status(registry, candidates, observed, packet["budget_left"] - index) == "open"
                and not any(item["outcome"] == "unknown" for item in rows[:index]), "종료된 상태 뒤 관측")
        require(eligible(registry, candidates, observed, row["probe"]), "중복 또는 의미 없는 검사 관측")
        states.append(list(candidates))
        candidates, _ = M.update(registry, candidates, row["probe"], row["outcome"])
        observed[row["probe"]] = row["outcome"]
    left = packet["budget_left"] - len(rows)
    require(left >= 0, "검사 수 예산 초과")
    status = M._status(registry, candidates, observed, left)
    if any(row["outcome"] == "unknown" for row in rows):
        status = "unidentifiable"
    if pending is not None:
        require(status == "open" and eligible(registry, candidates, observed, pending), "종료 뒤 미완결 검사")
        left -= 1
        status = "pending"
    visible_rows = rows
    public_observed = [{"probe": row["probe"], "outcome": row["outcome"]} for row in rows]
    if masked:
        require(bool(rows) and pending is None, "shadow에는 완료된 최신 관측이 필요함")
        candidates, visible_rows = states[-1], rows[:-1]
        observed = {row["probe"]: row["outcome"] for row in visible_rows}
        # 검사 실행 사실은 남지만 outcome은 후보 갱신이나 확정에 쓰지 않는다.
        observed[rows[-1]["probe"]] = None
        status = M._status(registry, candidates, observed, left)
        public_observed[-1] = {"probe": rows[-1]["probe"], "outcome_hidden": True}
    facts = deepcopy(packet["facts"])
    for index, row in enumerate(visible_rows):
        facts.update(_measurement_facts(index, row))
    return {"schema": "selection_shared_view_v1", "epoch": packet["epoch"], "evidence_facts": facts,
            "candidate_categories": [{"subtype": item, "public_category": S.R.category(item)}
                                     for item in registry["mechanisms"]],
            "remaining_candidates": candidates, "observations": public_observed, "status": status,
            "probe_catalog": [{"probe": name, **{key: deepcopy(spec[key])
                               for key in ("measures", "cost_gpu_s", "expect")}}
                              for name, spec in registry["probes"].items()],
            "eligible_probes": [name for name in registry["probes"]
                                if status == "open" and eligible(registry, candidates, observed, name)],
            "fixed_order": list(registry["fixed_order"]),
            "budget_left": left, "execution_budget_s": packet["execution_budget_s"],
            "execution_budget_scope": "per_request_legacy_receipt_wall_field",
            "recorded_cost_scope": "sum_of_completed_legacy_receipt_gpu_s_not_kernel_time",
            "recorded_probe_gpu_s": sum(row["gpu_s"] for row in rows)}


def build_snapshot(packet: dict, run_id: str, policy: str, protocol_sha256: str, ledger_path: str | Path,
                   parent: dict | None = None, masked: bool = False) -> dict:
    packet = S.validate_packet(packet)
    require(type(masked) is bool, "shadow 플래그 불일치")
    binding = epoch_binding(packet, run_id, policy, protocol_sha256, parent)
    rows, ledger_digest, pending = read_history(Path(ledger_path), binding, packet["execution_budget_s"])
    view = _view(packet, rows, pending, masked)
    inputs = {"packet": packet, "run_id": run_id, "policy": policy, "protocol_sha256": protocol_sha256,
              "ledger_path": str(Path(ledger_path).resolve()), "parent": deepcopy(parent), "masked": masked}
    pre = {"binding": binding, "prior_observations": rows[:-1],
           "executed_probe": rows[-1]["probe"] if rows else None}
    return {"schema": "selection_snapshot_v1", "inputs": inputs, "binding": binding,
            "ledger_digest": ledger_digest, "pre_receipt_anchor": L.digest(pre),
            "view": view, "view_digest": L.digest(view), "observation_masked_control": masked,
            "proposal_allowed": not masked and view["status"] == "open", "execution_authorized": False,
            "approval_granted": False}


def checked(snapshot: dict) -> dict:
    require(isinstance(snapshot, dict) and snapshot.get("schema") == "selection_snapshot_v1", "snapshot 형식")
    fresh = build_snapshot(**snapshot["inputs"])
    require(snapshot == fresh, "변조 또는 오래된 snapshot")
    return fresh


def messages(snapshot: dict) -> list:
    value = checked(snapshot)
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(value["view"]).decode("utf-8")}]


def registry_from_view(view: dict) -> dict:
    """선택 규칙은 이 adapter의 공개 표만 읽는다. args는 사용하지 않는다."""
    return M.normalize({"mechanisms": [row["subtype"] for row in view["candidate_categories"]],
                        "outcomes": list(L.OUTCOMES),
                        "probes": {row["probe"]: {key: deepcopy(row[key]) for key in ("measures", "cost_gpu_s", "expect")}
                                   | {"args": {}} for row in view["probe_catalog"]},
                        "fixed_order": list(view["fixed_order"])})


def rule_probe(view: dict, policy: str) -> str | None:
    require(policy in ("registry_next", "fixed_eligible"), "미등록 규칙")
    if view["status"] != "open" or view["budget_left"] <= 0:
        return None
    if policy == "fixed_eligible":
        return next((probe for probe in view["fixed_order"] if probe in view["eligible_probes"]), None)
    registry = registry_from_view(view)
    observed = {row["probe"]: row.get("outcome") for row in view["observations"]}
    return M._next(registry, view["remaining_candidates"], observed)


def select(snapshot: dict, reply: dict | None = None) -> dict:
    value = checked(snapshot)
    view, policy = value["view"], value["inputs"]["policy"]
    result = {"schema": "selection_comparison_choice_v1", "context_digest": L.digest(value["binding"]),
              "view_digest": value["view_digest"], "policy": policy, "selected_by": "hold",
              "action": "hold", "probe": None, "fact_ids": [], "model_status": "not_used", "codes": [],
              "observation_masked_control": value["observation_masked_control"], "approval_granted": False}
    if view["status"] != "open" or view["budget_left"] <= 0:
        return {**result, "reason": "state_" + view["status"]}
    uses_model = policy == "model_adaptive" or (policy == "model_first_then_rule" and not view["observations"])
    if uses_model:
        if reply is None:
            return {**result, "reason": "model_unavailable"}
        require(isinstance(reply, dict) and set(reply) == {"context_digest", "view_digest", "status", "content"}
                and reply["context_digest"] == result["context_digest"]
                and reply["view_digest"] == result["view_digest"], "다른 context 또는 view의 모델 응답")
        require(reply["status"] in ("completed", "infra_error", "unknown"), "모델 transport 상태 불일치")
        if reply["status"] != "completed":
            return {**result, "model_status": reply["status"], "reason": "model_unavailable"}
        format_packet = {**value["inputs"]["packet"], "facts": view["evidence_facts"]}
        review = S.format_review(reply["content"], format_packet)
        if review["status"] == "valid":
            decision = review["decision"]
            codes = ([] if all(key in view["evidence_facts"] for key in decision["fact_ids"]) else ["S_FACT_UNKNOWN"])
            if decision["action"] == "continue" and decision["probe"] not in view["eligible_probes"]:
                codes.append("S_PROBE_INELIGIBLE")
            review = {**review, "status": "invalid" if codes else "valid", "codes": codes}
        result.update(model_status=review["status"], codes=review["codes"])
        if review["status"] == "valid":
            decision = review["decision"]
            return {**result, **decision, "selected_by": "model" if decision["action"] == "continue" else "hold",
                    "reason": "model_valid"}
    if not uses_model:
        require(reply is None, "규칙 단계에 불필요한 모델 응답")
    fallback_policy = policy if policy in ("registry_next", "fixed_eligible") else "registry_next"
    probe = rule_probe(view, fallback_policy)
    return {**result, "probe": probe, "action": "continue" if probe else "hold",
            "selected_by": ("fixed_order" if fallback_policy == "fixed_eligible" else "registered_rule") if probe else "hold",
            "reason": "model_invalid" if uses_model else "registered_policy"}


def request_spec(snapshot: dict, selection: dict, reply: dict | None = None) -> dict:
    """RegistryLedger.propose에 넣을 후보다. 승인·소비·실행은 수행하지 않는다."""
    value = checked(snapshot)
    require(value["proposal_allowed"] and not value["observation_masked_control"], "shadow 또는 닫힌 상태 실행 거부")
    require(selection == select(value, reply) and selection["action"] == "continue", "선택 결합 또는 실행 후보 불일치")
    probe = selection["probe"]
    return {"prereg": deepcopy(value["binding"]), "probe": probe,
            "args": deepcopy(C.registry()["probes"][probe]["args"]),
            "budget_gpu_s": value["inputs"]["packet"]["execution_budget_s"], "approval_granted": False}
