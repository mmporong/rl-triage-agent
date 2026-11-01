"""새 학습 계보에서 다섯 선택 정책을 같은 검사·원장 경계로 실행한다."""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "src")]

import lineage_five_policy_nim_transport_v1 as TRANSPORT
import lineage_source_inputs_v2 as INPUT
import local_pair_four_arm_v1 as BASE
import local_pair_input_runtime_guard_v1 as GUARD
import local_pair_probe_host_v1 as HOST
import registered_optimizer_first_policy_v1 as OPT

C, A, CALL = BASE.C, BASE.A, BASE.CALL
SCHEMA = "lineage_five_policy_protocol_v1"
SOURCE_SCHEMA = "lineage_five_policy_source_v1"
RESULT_SCHEMA = "lineage_five_policy_result_v1"
ARMS = (*BASE.ARMS, OPT.POLICY)
MODEL_ARMS = BASE.MODEL_ARMS
SOURCE_FILES = tuple(
    sorted(
        {
            *BASE.SOURCE_FILES,
            *HOST.SOURCE_FILES,
            *TRANSPORT.SOURCE_FILES,
            *OPT.SOURCE_FILES,
            *INPUT.BATCH.source_files(),
            "evals/lineage_five_policy_driver_v1.py",
            "tests/test_lineage_five_policy_driver_v1.py",
            "evals/lineage_source_inputs_v2.py",
            "tests/test_lineage_source_inputs_v2.py",
            "evals/local_pair_input_runtime_guard_v1.py",
            "tests/test_local_pair_input_runtime_guard_v1.py",
            "evals/lineage_evaluation_gate_v1.py",
            "tests/test_lineage_evaluation_gate_v1.py",
            "evals/lineage_selection_inputs_v1.py",
            "tests/test_lineage_selection_inputs_v1.py",
            "evals/optimizer_runtime_facts_v1.py",
            "tests/test_optimizer_runtime_facts_v1.py",
        }
    )
)


def prepare_source(
    index_path: Path,
    case_id: str,
    *,
    providers: dict | None = None,
    visual_evidence: dict | None = None,
) -> dict:
    """정답은 private qualification에 두고 모델에는 training48만 전달한다."""
    envelope = INPUT.make_case_packet(index_path, case_id, budget_left=3)
    backend = envelope["packet"]
    packet = deepcopy(backend)
    packet["facts"] = {
        key: value
        for key, value in backend["facts"].items()
        if not key.startswith("runtime:")
    }
    HOST._packets(packet, backend)
    result = {
        "schema": SOURCE_SCHEMA,
        "case_id": case_id,
        "index": BASE.ref(index_path),
        "packet": packet,
        "backend_packet": backend,
        "qualification": envelope["private_metadata"]["classification"],
        "source_binding": envelope["source_binding"],
        "providers": deepcopy(providers),
        "visual_evidence": deepcopy(visual_evidence),
        "approval_granted": False,
        "api_authorized": False,
    }
    checked_source(result)
    return result


def checked_source(source: dict, *, fixture: bool = False) -> dict:
    BASE.require(
        source.get("schema") == SOURCE_SCHEMA
        and re.fullmatch(r"case-(?:0[1-9]|10)", source.get("case_id", "")) is not None
        and source.get("approval_granted") is False
        and source.get("api_authorized") is False,
        "five-policy source contract",
    )
    packet, backend = HOST._packets(source["packet"], source["backend_packet"])
    BASE.require(packet["budget_left"] == 3, "same paired-probe budget must be three")
    if not fixture:
        index = BASE._bound_file(source["index"])
        current = INPUT.make_case_packet(index, source["case_id"], budget_left=3)
        BASE.require(
            current["packet"] == backend
            and current["source_binding"] == source["source_binding"]
            and current["private_metadata"]["classification"]
            == source["qualification"],
            "source index/checkpoint/telemetry/qualification changed",
        )
    providers = source.get("providers")
    if providers is not None:
        BASE.require(
            isinstance(providers, dict) and set(providers) == set(MODEL_ARMS),
            "paired provider arms differ",
        )
        for name, provider in providers.items():
            context = A.checked(provider["context"])
            messages = CALL._messages(context, provider["messages"])
            BASE.require(
                context["snapshot"]["inputs"]["packet"] == packet
                and provider["params"] == CALL.params_for_messages(messages),
                "provider input is not this public48 source",
            )
            extra = context["extra_facts"]
            common = {
                key: value
                for key, value in extra.items()
                if not key.startswith("vision:frame:")
            }
            BASE.require(
                len(common) == 54
                and all(key.startswith(BASE.P.PAIRED.PLAY_PREFIX) for key in common),
                "same-Play facts must be the registered 54 fields",
            )
            image_count = (
                sum(
                    block.get("type") == "image_url"
                    for block in messages[1]["content"]
                    if isinstance(block, dict)
                )
                if isinstance(messages[1]["content"], list)
                else 0
            )
            BASE.require(
                image_count == (6 if name == "logs_plus_frames" else 0),
                "image delta is 0/6",
            )
        BASE.require(
            BASE.P.PAIRED._context_common(providers["logs_only"]["context"])
            == BASE.P.PAIRED._context_common(providers["logs_plus_frames"]["context"]),
            "paired numeric facts differ",
        )
    return deepcopy(source)


def verify_protocol(path: Path, anchor: str) -> dict:
    value = TRANSPORT._verify_protocol(path, anchor)
    BASE.require(
        value.get("schema") == SCHEMA
        and value.get("arms") == list(ARMS)
        and value.get("max_probes_per_arm") == 3
        and value.get("normal_shadow_max_decisions") == 1
        and value.get("allow_source_replacement") is False
        and set(value["sources_lf"]) >= set(SOURCE_FILES),
        "five-policy execution contract",
    )
    design_ref = value["study_design"]
    design_path = ROOT / design_ref["path"]
    BASE.require(
        BASE.sha(design_path, lf=True) == design_ref["sha256_lf"],
        "study design bytes changed",
    )
    design = BASE.read(design_path)
    BASE.require(
        design.get("schema") == "agentic_five_type_evaluation_design_v1"
        and design["primary_policies"] + design["auxiliary_policies"] == list(ARMS)
        and design["model"] == value["base_params"]
        and value["case_order"] == [row["case_id"] for row in design["case_plan"]],
        "execution diverges from preregistered study",
    )
    expected_order = {
        case_id: list(ARMS[index % len(ARMS) :] + ARMS[: index % len(ARMS)])
        for index, case_id in enumerate(value["case_order"])
    }
    BASE.require(value.get("arm_order") == expected_order, "balanced arm order changed")
    return value


def protocol_value(
    study_path: Path, sources: list[dict], *, extra_sources: tuple[str, ...] = ()
) -> dict:
    """실행 전 공개 등록용 값. 개인 경로는 source digest 안으로만 결합한다."""
    study_path = Path(study_path).resolve(strict=True)
    design = BASE.read(study_path)
    BASE.require(
        design.get("schema") == "agentic_five_type_evaluation_design_v1"
        and design["primary_policies"] + design["auxiliary_policies"] == list(ARMS),
        "registered study design required",
    )
    case_order = [row["case_id"] for row in design["case_plan"]]
    BASE.require(
        len(sources) == 10 and [row["case_id"] for row in sources] == case_order,
        "all ten cases, including excluded cases, must remain in order",
    )
    for source in sources:
        checked_source(source)
    pins = {}
    for name in sorted(set(SOURCE_FILES) | set(extra_sources)):
        path = TRANSPORT.BASE._source_path(ROOT, name)
        pins[name] = BASE.sha(path, lf=True)
    relative_design = study_path.relative_to(ROOT).as_posix()
    pins[relative_design] = BASE.sha(study_path, lf=True)
    return {
        "schema": SCHEMA,
        "api_cap": 72,
        "backend0prefix": 0,
        "model": CALL.PARAMS["model"],
        "base_params": deepcopy(CALL.PARAMS),
        "accepted_provider_models": sorted(
            TRANSPORT.MODEL_CALL.ACCEPTED_PROVIDER_MODELS
        ),
        "authority": deepcopy(TRANSPORT.AUTHORITY),
        "automatic_retries": 0,
        "sources_lf": pins,
        "arms": list(ARMS),
        "max_probes_per_arm": 3,
        "normal_shadow_max_decisions": 1,
        "allow_source_replacement": False,
        "study_design": {
            "path": relative_design,
            "sha256_lf": BASE.sha(study_path, lf=True),
        },
        "case_order": case_order,
        "arm_order": {
            case: list(ARMS[index % len(ARMS) :] + ARMS[: index % len(ARMS)])
            for index, case in enumerate(case_order)
        },
        "source_digests": {row["case_id"]: BASE.digest(row) for row in sources},
        "measurement_scope": deepcopy(design["measurement_scope"]),
        "diagnosis_sdk_upper_bound": deepcopy(design["diagnosis_sdk_upper_bound"]),
    }


def context_for(
    source: dict, snapshot: dict, arm: str, remaining: int
) -> tuple[dict, list]:
    if source.get("providers") is not None:
        return BASE._context(source, snapshot, arm, remaining)
    BASE.require(
        arm != "logs_plus_frames", "frames are unavailable, no substitute images"
    )
    context = A.prepare(snapshot, goal=BASE.P.PAIR_GOAL, decisions_left=remaining)
    return context, CALL._messages(context, A.messages(context))


def _request(
    source: dict, context: dict, messages: list, run_id: str, arm: str, step: int
) -> dict:
    # 기존 transport request ABI만 재사용하며 새로운 source binding digest를 명시한다.
    material = {**source, "pair": {"sha256": BASE.digest(source["source_binding"])}}
    return BASE._request(material, context, messages, run_id, arm, step)


def verify_transport(response: dict, request: dict, verifier: Callable) -> dict:
    """미완료 인프라 시도와 실제 모델 응답을 구분해 원문 receipt에 결합한다."""
    reference = response["call_ref"]
    call = Path(reference["path"]).resolve(strict=True)
    BASE.require(
        call.is_dir() and BASE.sha(call / "receipt.json") == reference["sha256"],
        "transport receipt SHA mismatch",
    )
    checked = verifier(call)
    BASE.require(
        checked.get("request") == request
        and all(
            checked.get(key) == response.get(key)
            for key in ("status", "content", "usage", "wall_s", "call_ref")
        ),
        "transport request/response changed",
    )
    BASE.require(
        checked.get("live") is True
        or (
            checked.get("live") is False
            and checked.get("status") == "infra_error"
            and checked.get("outcome") == "infrastructure_unknown"
        ),
        "non-live response cannot become a completed model decision",
    )
    return reference


def _storage_policy(arm: str) -> str:
    return (
        "registry_next"
        if arm == OPT.POLICY
        else "model_adaptive"
        if arm in MODEL_ARMS
        else arm
    )


def _input_roots(value: object) -> tuple[Path, ...]:
    BASE.require(isinstance(value, (list, tuple)), "registered input roots contract")
    values = cast(list[object] | tuple[object, ...], value)
    forbidden = {ROOT.resolve(), Path.home().resolve()}
    roots: list[Path] = []
    for raw in values:
        BASE.require(isinstance(raw, (str, Path)), "registered input root type")
        root = Path(cast(str | Path, raw)).resolve()
        BASE.require(
            root not in forbidden and root != Path(root.anchor),
            "broad registered input root is forbidden",
        )
        roots.append(root)
    return tuple(roots)


def source_registration_key(source: dict, arm: str) -> str:
    return source["case_id"]


def _score(qualification: dict, confirmed: str | None) -> dict:
    expected = {
        "F1": {"reward"},
        "F2": {"actuator"},
        "COMPOUND": {"reward", "optimizer_update_count"},
        "F5": {"physics"},
        "NONE": set(),
    }[qualification["arm"]]
    correct = confirmed in expected if confirmed is not None else False
    return {
        "expected_families": sorted(expected),
        "confirmed_family": confirmed,
        "correct_any_family": correct,
        "wrong_confirmation": confirmed is not None and not correct,
        "compound_partial": qualification["arm"] == "COMPOUND" and correct,
        "full_cause_confirmation": correct and len(expected) == 1,
        "repair_unsupported": qualification["arm"] == "F5" and correct,
    }


def run_arm(
    source: dict,
    *,
    arm: str,
    run_id: str,
    output: Path,
    protocol_path: Path,
    execution_anchor: str,
    complete: Callable | None,
    execute_probe: Callable,
    verify_probe: Callable,
    verify_call: Callable | None = None,
    busy_reader: Callable | None = None,
    allow_fixture: bool = False,
    registered_input_roots: tuple[Path | str, ...] | list[Path | str] = (),
) -> dict:
    """형식 무효·미확정은 HOLD하며 승인된 검사 receipt만 후보를 갱신한다."""
    BASE.require(
        arm in ARMS and bool(re.fullmatch(r"[a-z0-9_]{1,60}", run_id)), "arm/run_id"
    )
    roots = _input_roots(registered_input_roots)
    raw_source = deepcopy(source)
    raw_digest = BASE.digest(raw_source)
    matching_keys: list[str] = []
    if allow_fixture:
        protocol = BASE.read(protocol_path)
        BASE.require(protocol.get("fixture") is True, "explicit fixture required")
        source = checked_source(raw_source, fixture=True)
    else:
        protocol = verify_protocol(protocol_path, execution_anchor)
        matching_keys = [
            key
            for key, digest in protocol["source_digests"].items()
            if digest == raw_digest
        ]
        BASE.require(len(matching_keys) == 1, "source not preregistered")
        BASE.require(
            callable(busy_reader)
            and callable(verify_probe)
            and (arm not in MODEL_ARMS or callable(verify_call)),
            "live boundary callbacks missing",
        )
    output = Path(output).resolve()
    BASE.require(
        not output.exists(), "new output required; pending is never auto-restarted"
    )
    started = time.perf_counter()
    if not allow_fixture:
        checked = GUARD.invoke_guarded(
            "input",
            checked_source,
            raw_source,
            fixture=False,
            registered_input_roots=roots,
        )
        if isinstance(checked, dict) and checked.get("schema") == GUARD.SCHEMA:
            output.mkdir(parents=True)
            result = {
                "schema": RESULT_SCHEMA,
                "arm": arm,
                "case_id": raw_source.get("case_id"),
                "run_id": run_id,
                "scope": "actual_backend",
                "selector_policy": arm,
                "host_storage_policy": _storage_policy(arm),
                "host_storage_alias": arm != _storage_policy(arm),
                "source_digest": raw_digest,
                "source_index": deepcopy(raw_source.get("index")),
                "checkpoint_bindings": None,
                "steps": [],
                "model_calls": 0,
                "native_normal_intervention_proposal": None,
                "hold_reason": "registered_input_unavailable",
                "input_unavailable": 1,
                "input_guard": deepcopy(checked),
                "semantic_invalid": 0,
                "infra_unknown": 0,
                "provider_unknown": 0,
                "frame_fact_citations": 0,
                "repair": False,
                "recovery": False,
                "heldout": False,
                "qualification": None,
                "end_state": "HOLD",
                "confirmed_subtype": None,
                "final_view_digest": None,
                "remaining_candidates": None,
                "scoring": None,
                "parent_wall_s": time.perf_counter() - started,
            }
            BASE.write(output / "input-guard.json", checked)
            BASE.write(output / "result.json", result)
            return result
        source = cast(dict[str, Any], checked)
        BASE.require(
            source_registration_key(source, arm) == matching_keys[0],
            "source registration key changed after input verification",
        )
    output.mkdir(parents=True)
    ledger, policy = output / "probes.ledger.jsonl", _storage_policy(arm)
    qualification = source["qualification"]
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "arm": arm,
        "case_id": source["case_id"],
        "run_id": run_id,
        "scope": "cpu_fixture" if allow_fixture else "actual_backend",
        "selector_policy": arm,
        "host_storage_policy": policy,
        "host_storage_alias": arm != policy,
        "source_digest": BASE.digest(source),
        "source_index": deepcopy(source["index"]),
        "checkpoint_bindings": deepcopy(source["packet"]["input_bindings"]),
        "steps": [],
        "model_calls": 0,
        "native_normal_intervention_proposal": None,
        "hold_reason": None,
        "input_unavailable": 0,
        "semantic_invalid": 0,
        "infra_unknown": 0,
        "provider_unknown": 0,
        "frame_fact_citations": 0,
        "repair": False,
        "recovery": False,
        "heldout": False,
        "qualification": deepcopy(qualification),
        "end_state": "HOLD",
    }
    packet = source["packet"]
    normal_shadow = (
        qualification["arm"] == "NONE" and qualification["label"] == "eligible"
    )
    if not qualification["diagnosis_eligible"] and not normal_shadow:
        result["hold_reason"] = "source_" + qualification["label"]
    elif normal_shadow and arm not in MODEL_ARMS:
        result["hold_reason"] = "normal_host_gate"
    elif arm == "logs_plus_frames" and source.get("providers") is None:
        result["hold_reason"] = "same_play_frames_unavailable"
    elif arm == "logs_plus_frames" and source.get("visual_evidence") is None:
        result["hold_reason"] = "visual_quality_proof_missing"
    maximum = 1 if normal_shadow else 3
    for step in range(1, maximum + 1):
        if result["hold_reason"] is not None:
            break
        snapshot = C.V.build_snapshot(
            packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger
        )
        if snapshot["view"]["status"] != "open":
            break
        decision_dir = output / f"step{step:02d}"
        context, messages = context_for(source, snapshot, arm, maximum - step + 1)
        BASE.write(decision_dir / "context.json", context)
        model_ref = None
        if arm in MODEL_ARMS:
            if complete is None:
                raise ValueError("model callback missing")
            if arm == "logs_plus_frames":
                gate = GUARD.pre_dispatch_gate(**source["visual_evidence"])
                BASE.write(decision_dir / "visual-gate.json", gate)
                if gate["decision"] == "HOLD":
                    result["hold_reason"] = gate["reason"]
                    break
            request = _request(source, context, messages, run_id, arm, step)
            BASE.write(decision_dir / "request.json", request)
            response = GUARD.invoke_guarded("transport", complete, deepcopy(request))
            result["model_calls"] += 1
            BASE.write(decision_dir / "model-result.json", response)
            if (
                response.get("schema") == GUARD.SCHEMA
                and response.get("decision") == "HOLD"
            ):
                result["hold_reason"] = response["reason"]
                result["infra_unknown"] += 1
                break
            if not allow_fixture:
                if verify_call is None:
                    raise ValueError("live model verifier missing")
                model_ref = verify_transport(response, request, verify_call)
                verified_call = verify_call(Path(model_ref["path"]))
                gate = GUARD.gate_transport(verified_call)
                BASE.write(decision_dir / "transport-gate.json", gate)
                if gate["decision"] == "HOLD":
                    result["hold_reason"] = gate["reason"]
            if response.get("status") != "completed":
                field = (
                    "provider_unknown"
                    if response.get("status") == "unknown"
                    else "infra_unknown"
                )
                result[field] += 1
                result["hold_reason"] = "model_" + field
                break
            review = A.assess(response.get("content"), context)
            BASE.write(
                decision_dir / "model-review.json",
                {"transport": model_ref, "review": review},
            )
            if review["status"] != "valid":
                result["semantic_invalid"] += 1
                result["hold_reason"] = "model_contract_invalid"
                break
            choice = (
                A.probe_choice(context, review)
                if review["decision"]["action"] == "run_probe"
                else None
            )
            if normal_shadow:
                result["native_normal_intervention_proposal"] = choice is not None
                BASE.write(
                    decision_dir / "normal-shadow.json",
                    {
                        "review": review,
                        "choice": choice,
                        "approval_granted": False,
                        "request_spec_created": False,
                        "dispatch": False,
                    },
                )
                result["hold_reason"] = "normal_shadow_only_host_gate"
                break
            if choice is None:
                result["hold_reason"] = "model_hold"
                break
            result["frame_fact_citations"] += sum(
                key.startswith("vision:frame:") for key in choice["fact_ids"]
            )
        else:
            choice = (
                OPT.choose(snapshot) if arm == OPT.POLICY else C.choose(snapshot, arm)
            )
        if choice["action"] != "continue":
            result["hold_reason"] = choice["reason"]
            break
        spec = (
            OPT.request_spec(snapshot, choice)
            if arm == OPT.POLICY
            else C.request_spec(snapshot, choice)
        )
        BASE.write(decision_dir / "choice.json", choice)
        BASE.write(decision_dir / "request-spec.json", spec)
        if busy_reader is not None:
            gate = GUARD.pre_dispatch_gate(
                busy_reader=busy_reader, operation_id=f"{run_id}_{step}", intent=spec
            )
            BASE.write(decision_dir / "busy-gate.json", gate)
            if gate["decision"] == "HOLD":
                result["hold_reason"] = gate["reason"]
                break
        dispatched = GUARD.invoke_guarded(
            "input",
            execute_probe,
            snapshot,
            choice,
            out=decision_dir / "probe",
            registered_input_roots=roots,
        )
        if isinstance(dispatched, dict) and dispatched.get("schema") == GUARD.SCHEMA:
            BASE.write(decision_dir / "input-guard.json", dispatched)
            result["hold_reason"] = "registered_input_unavailable"
            result["input_unavailable"] += 1
            break
        receipt_ref = dispatched["result_ref"]
        verified = verify_probe(BASE._bound_file(receipt_ref), snapshot, choice)
        BASE.require(
            verified.get("probe") == spec["probe"]
            and verified.get("outcome") in {"normal", "abnormal", "unknown"},
            "verified probe mismatch",
        )
        fresh = C.V.build_snapshot(
            packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger
        )
        BASE.require(
            fresh["view"]["observations"]
            == snapshot["view"]["observations"]
            + [{"probe": spec["probe"], "outcome": verified["outcome"]}],
            "probe receipt ledger not settled",
        )
        if not allow_fixture:
            BASE.require(
                verified.get("evidence_kind") == "actual_runtime_observation",
                "fixture receipt not live",
            )
        result["steps"].append(
            {
                "step": step,
                "selected_by": choice["selected_by"],
                "selector_policy": arm,
                "host_storage_policy": policy,
                "probe": spec["probe"],
                "outcome": verified["outcome"],
                "fact_ids": choice["fact_ids"],
                "probe_result": receipt_ref,
                "model_call": model_ref,
                "before_view_digest": snapshot["view_digest"],
                "after_view_digest": fresh["view_digest"],
                "owned_child_wall_s": verified.get("owned_child_wall_s"),
            }
        )
        if verified["outcome"] == "unknown":
            result["hold_reason"] = "probe_outcome_unknown"
            break
    final = C.V.build_snapshot(
        packet, run_id, policy, C.V.file_digest(C.V.PROTOCOL), ledger
    )
    confirmed = (
        final["view"]["remaining_candidates"][0]
        if final["view"]["status"] == "confirmed"
        else None
    )
    result.update(
        confirmed_subtype=confirmed,
        final_view_digest=final["view_digest"],
        remaining_candidates=final["view"]["remaining_candidates"],
        scoring=_score(qualification, confirmed),
    )
    if confirmed is not None:
        result["end_state"] = "confirmed"
        if result["scoring"]["repair_unsupported"]:
            result["end_state"] = "HOLD"
            result["hold_reason"] = "repair_unsupported"
    elif result["hold_reason"] is None:
        result["hold_reason"] = "probe_budget_or_no_informative_probe"
    result["parent_wall_s"] = time.perf_counter() - started
    BASE.write(output / "result.json", result)
    return result
