"""새 source에서 none/low/등록 규칙을 첫 행동부터 비교한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import evidence_guided_probe_host_v1 as HOST
import prospective_agentic_comparison_v1 as COMP
import prospective_diagnosis_inputs_v1 as INPUT
import prospective_diagnosis_training_v1 as TRAIN
import reasoning_diagnostic_loop_v1 as RLOOP
import registered_reasoning_bridge_v2 as BRIDGE
import registry_action_options_v2 as O2
import run_prospective_diagnosis_training_v1 as SOURCE
import run_registry_action_study_v1 as BASE

SEL = COMP.SEL
DESIGN_SCHEMA = "prospective_reasoning_confirmation_design_v1"
EXECUTION_SCHEMA = "prospective_reasoning_confirmation_execution_v1"
RESULT_SCHEMA = "prospective_reasoning_confirmation_result_v1"
POLICIES = ("legacy_none", "enabled_low", "strong_registered_rule")
FAULT_CASES = BRIDGE.FAULT_CASES
NORMAL_CASES = ("c01", "c05")
MAX_API = 96
MAX_PAIRED = 54
MAX_SDK_ROLES = 108
MAX_WALL_S = 14400
BASE_PROTOCOL = ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json"
_LOCAL_SOURCE_FILES = (
    "evals/registered_reasoning_bridge_v2.py",
    "tests/test_registered_reasoning_bridge_v2.py",
    "evals/run_prospective_reasoning_confirmation_v1.py",
    "tests/test_run_prospective_reasoning_confirmation_v1.py",
    "evals/reasoning_diagnostic_loop_v1.py",
    "tests/test_reasoning_diagnostic_loop_v1.py",
    "evals/registry_action_options_v2.py",
    "evals/registry_action_loop_v2.py",
    "evals/evidence_guided_selector_v1.py",
    "evals/evidence_guided_probe_host_v1.py",
    "evals/prospective_agentic_comparison_v1.py",
    "evals/immutable_git_proof_cache_v1.py",
    "tests/test_immutable_git_proof_cache_v1.py",
    "evals/prospective_diagnosis_inputs_v1.py",
    "evals/core_seed_guard_v1.py",
    "evals/lineage_lease_context_v2.py",
    "evals/runtime_observation_view_v1.py",
    "tests/test_runtime_observation_view_v1.py",
    "bench/prospective_diagnosis_training_v1.py",
    "bench/run_prospective_diagnosis_training_v1.py",
)
SOURCE_FILES = tuple(sorted(set(_LOCAL_SOURCE_FILES)
                            | set(BRIDGE.MODE.SOURCE_FILES)
                            | set(BRIDGE.MODE.CALL.SOURCE_FILES)
                            | set(RLOOP.SOURCE_FILES)
                            | set(BASE.SOURCE_FILES)
                            | set(BASE.COMP.SOURCE_FILES)))


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체가 아니다: {Path(path).name}")
    return value


def _write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _public_ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file() and path.is_relative_to(ROOT), "공개 파일 ref가 아니다")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}


def _mode_contract() -> dict[str, dict[str, Any]]:
    return BRIDGE._mode_contract()


def _slots() -> list[dict[str, str]]:
    return BRIDGE._expected_slots()


def _checked_source_protocol(path: Path) -> dict[str, Any]:
    value = _read(path)
    require(value.get("schema") == TRAIN.SCHEMA
            and value.get("categories") == list(TRAIN.CATEGORIES)
            and len(value.get("seeds", [])) == 2
            and len(set(value["seeds"])) == 2
            and value.get("training") == {
                "task": TRAIN.TRAIN.RESIDUAL.TASK, "num_envs": 4096, "updates": 300,
                "configured_learning_epochs": 5, "configured_mini_batches": 4,
                "runtime_updates_to_observe": 3, "fresh_initialization": True,
                "checkpoint_loaded": False}
            and value.get("evaluation") == {
                "fixed_conditions": 26, "unchanged_existing_evaluator": True},
            "source protocol 학습·범주 계약 불일치")
    return value


def prepare_design(source_protocol: Path) -> dict[str, Any]:
    """source 실행 전 모델·규칙·예측·예산을 고정한다."""
    source_protocol = Path(source_protocol).resolve()
    source = _checked_source_protocol(source_protocol)
    base = _read(BASE_PROTOCOL)
    protocol_refs = base.get("protocols")
    require(isinstance(protocol_refs, dict), "기준 비교 protocol refs 불일치")
    protocol_refs = cast(dict[str, dict[str, str]], protocol_refs)
    tool_refs = {name: deepcopy(protocol_refs[name]) for name in ("paired_probe", "view")}
    for name, row in tool_refs.items():
        target = ROOT / row["path"]
        require(target.is_file() and _sha(target, lf=True) == row["sha256_lf"],
                f"{name} protocol ref 불일치")
    closure = deepcopy(source["source_sha256_lf"])
    closure.update({name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES})
    closure[source_protocol.relative_to(ROOT).as_posix()] = _sha(source_protocol, lf=True)
    return {
        "schema": DESIGN_SCHEMA, "status": "registered_only_after_commit",
        "purpose": "new_seed_reasoning_mode_confirmation_before_source_execution",
        "source_protocol": _public_ref(source_protocol), "seeds": deepcopy(source["seeds"]),
        "categories": deepcopy(source["categories"]), "model": BRIDGE.MODE.MODEL,
        "tool_protocols": tool_refs,
        "accepted_provider_models": sorted(BRIDGE.MODE.CALL.ACCEPTED_PROVIDER_MODELS),
        "policies": list(POLICIES),
        "bridge": {"modes": _mode_contract(), "logical_slots": _slots(),
                   "max_requests": MAX_API, "fallback": False, "retry": 0},
        "budgets": {"model_actions_per_row": 4, "paired_probes_per_row": 3,
                    "api_requests": MAX_API, "paired_probes": MAX_PAIRED,
                    "sdk_role_executions": MAX_SDK_ROLES, "wall_s": MAX_WALL_S},
        "rule_policy": ["read_evidence", "P_optimizer_updates",
                        "if_count_normal_then_P_reward"],
        "analysis": {
            "post_exposure_design": True,
            "prediction": "enabled_low may preserve more correct confirmations than legacy_none; confirmation is not guaranteed",
            "candidate_rule": {"complete_rows": 24, "complete_fault_model_rows": 12,
                               "normal_host_gate_api_and_probe_zero": True,
                               "wrong_family_confirmations": 0,
                               "enabled_low_correct_confirmed_strictly_above_none": True},
            "rule_result_reported_separately": True,
            "incomplete_or_infrastructure": "indeterminate",
            "failed_candidate": "not_confirmed",
        },
        "claims": {"new_fault_family_generalization": False, "llm_superiority": False,
                   "default_mode_adoption": False, "repair": False, "recovery": False},
        "source_sha256_lf": closure,
    }


def _source_rows(source_batch: Path, design: dict[str, Any]) -> list[dict[str, Any]]:
    value = _read(source_batch)
    require(value.get("schema") == SOURCE.BATCH_SCHEMA and value.get("status") == "complete"
            and value.get("seeds") == design["seeds"]
            and value.get("categories") == design["categories"], "source batch identity 불일치")
    stages = [row for row in value.get("stages", []) if row.get("stage") != "smoke"]
    require(len(stages) == 8, "source batch 8개 stage 불일치")
    rows = []
    expected = [(seed, category) for seed in design["seeds"] for category in design["categories"]]
    for index, (stage, (seed, category)) in enumerate(zip(stages, expected, strict=True), 1):
        require(stage.get("stage") == f"s{seed}-{category.lower()}", "source stage 순서 불일치")
        input_ref, checkpoint = stage.get("input"), stage.get("checkpoint")
        require(isinstance(input_ref, dict) and isinstance(checkpoint, dict), "source refs 불일치")
        input_path = Path(input_ref["path"])
        require(input_path.is_file() and _sha(input_path) == input_ref["sha256"], "input SHA 불일치")
        public, metadata = INPUT.checked(input_path)
        result_ref = stage.get("result")
        require(isinstance(result_ref, dict), "source result ref 불일치")
        result_ref = cast(dict[str, Any], result_ref)
        result_path = Path(result_ref["path"])
        require(result_path.is_file() and _sha(result_path) == result_ref["sha256"],
                "source result SHA 불일치")
        result = _read(result_path)
        fixed_ref = result.get("fixed_report")
        require(isinstance(fixed_ref, dict), "fixed report ref 불일치")
        fixed_ref = cast(dict[str, Any], fixed_ref)
        fixed_path = Path(fixed_ref["path"])
        require(fixed_path.is_file() and _sha(fixed_path) == fixed_ref["sha256"],
                "fixed report SHA 불일치")
        fixed_report = _read(fixed_path)
        require(public["source_binding"]["checkpoint_sha256"] == checkpoint["sha256"],
                "input/checkpoint 불일치")
        rows.append({"case_id": f"c{index:02d}", "seed": seed, "category": category,
                     "input": input_path, "checkpoint": checkpoint, "metadata": metadata,
                     "result": result_path, "fixed": fixed_path,
                     "fixed_report": fixed_report})
    return rows


def _corpus(corpus: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value, checked = BRIDGE.MODE._corpus_ref(Path(corpus).resolve(), load_payloads=True)
    require([row["case_id"] for row in checked] == [f"c{i:02d}" for i in range(1, 9)],
            "corpus 8 case 순서 불일치")
    return value, {row["case_id"]: row for row in checked}


def _bind_rows(rows: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> None:
    references = {row["seed"]: row for row in rows if row["category"] == "NONE"}
    require(set(references) == {row["seed"] for row in rows}, "same-seed NONE 불완전")
    for row in rows:
        state = corpus[row["case_id"]]["state"]
        expected = {"case": row, "reference": references[row["seed"]]}
        for role, source in expected.items():
            checkpoint = state["initial_packet"]["input_bindings"][role]["checkpoint_sha256"]
            require(checkpoint == source["checkpoint"]["sha256"]
                    == state["fact_ledger"]["source_binding"][role]["checkpoint_sha256"],
                    f"corpus/source {row['case_id']} {role} checkpoint 불일치")
        gate = corpus[row["case_id"]]["evaluation_role"]
        require((row["category"] == "NONE") == (gate == "normal_model_only"),
                "corpus host gate/category 불일치")


def _verify_source_corpus_binding(design: dict[str, Any], batch: dict[str, Any],
                                  corpus: dict[str, Any], source_batch: Path) -> None:
    batch_protocol = batch.get("protocol")
    require(isinstance(batch_protocol, dict)
            and set(batch_protocol) == {"path", "sha256"}
            and Path(batch_protocol["path"]).resolve()
            == (ROOT / design["source_protocol"]["path"]).resolve()
            and batch_protocol["sha256"] == design["source_protocol"]["sha256"],
            "source batch protocol ref가 design과 다르다")
    require(corpus["source_batch_sha256"] == _sha(source_batch),
            "corpus가 다른 source batch를 가리킨다")
    require(corpus["source_protocol_sha256_lf"]
            == _sha(ROOT / design["source_protocol"]["path"], lf=True),
            "corpus source protocol SHA가 다르다")
    require(corpus["source_anchor"] == batch.get("anchor"),
            "corpus/source batch anchor가 다르다")


def build_corpus(design_protocol: Path, source_batch: Path, out: Path) -> dict[str, Any]:
    """새 source의 8개 state/messages를 같은-seed NONE 기준으로 생성한다."""
    design_protocol, source_batch, out = (Path(design_protocol).resolve(),
                                          Path(source_batch).resolve(), Path(out).resolve())
    require(not out.exists(), "corpus out은 fresh여야 한다")
    design = _read(design_protocol)
    require(design == prepare_design(ROOT / design["source_protocol"]["path"]),
            "corpus design이 현재 source와 다르다")
    rows = _source_rows(source_batch, design)
    batch = _read(source_batch)
    references = {row["seed"]: row for row in rows if row["category"] == "NONE"}
    require(set(references) == set(design["seeds"]), "same-seed NONE reference 불완전")
    require(_sha(BASE.RUN.BANDS) == BASE.RUN.BANDS_SHA256, "fixed26 bands 변경")
    bands = _read(BASE.RUN.BANDS)["bands"]
    gates = []
    for row in rows:
        gate = BASE.RUN._gate(row, references[row["seed"]], bands)
        gates.append({"case_id": row["case_id"], "category": row["category"],
                      "gate": deepcopy(gate)})
    qualified = (sum(item["gate"].get("reason") == "healthy_gate" for item in gates) == 2
                 and sum(item["gate"].get("run") is True for item in gates) == 6
                 and all((item["category"] == "NONE")
                         == (item["gate"].get("reason") == "healthy_gate") for item in gates))
    out.mkdir(parents=True, exist_ok=False)
    qualification = {"schema": "prospective_reasoning_source_qualification_v1",
                     "status": "qualified" if qualified else "source_not_qualified",
                     "gates": gates, "api_requests": 0, "replacement_seed": False,
                     "comparison_outcome": None if qualified else "indeterminate"}
    _write_new(out / "source-qualification.json", qualification)
    if not qualified:
        return qualification

    entries = []
    for row in rows:
        case_root = out / "cases" / row["case_id"]
        built = BASE._inputs(row, references[row["seed"]], case_root / "work", eager=False)
        state = built["state"]
        messages = O2.messages(state)
        encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).lower()
        require(not (BRIDGE.MODE._walk_keys(state) & BRIDGE.MODE.FORBIDDEN_STATE_KEYS),
                "corpus state에 정답 key가 노출됐다")
        require(all(token not in encoded for token in ("c:\\", "/home/", "/mnt/",
                                                        '"seed"', '"fault"')),
                "provider messages에 경로·seed·정답이 노출됐다")
        state_path, messages_path = case_root / "state.json", case_root / "messages.json"
        _write_new(state_path, state)
        _write_new(messages_path, messages)
        entries.append({"case_id": row["case_id"],
                        "host_gate": deepcopy(gates[len(entries)]["gate"]),
                        "state": {"path": state_path.relative_to(out).as_posix(),
                                  "sha256": _sha(state_path)},
                        "messages": {"path": messages_path.relative_to(out).as_posix(),
                                     "sha256": _sha(messages_path)},
                        "messages_digest": O2.O1._digest(messages)})
    corpus = {"schema": BRIDGE.MODE.CORPUS_SCHEMA, "source_anchor": batch["anchor"],
              "source_batch_sha256": _sha(source_batch),
              "source_protocol_sha256_lf": _sha(
                  ROOT / design["source_protocol"]["path"], lf=True),
              "same_source_measurements": True, "post_exposure_development": True,
              "approval_granted": False, "GPU_executed": False, "entries": entries}
    corpus_path = out / "corpus.json"
    _write_new(corpus_path, corpus)
    checked_corpus, checked_rows = _corpus(corpus_path)
    _verify_source_corpus_binding(design, batch, checked_corpus, source_batch)
    _bind_rows(rows, checked_rows)
    result = {"schema": "prospective_reasoning_corpus_build_result_v1",
              "status": "complete", "corpus": {"path": str(corpus_path),
                                                   "sha256": _sha(corpus_path)},
              "qualification": {"path": str(out / "source-qualification.json"),
                                  "sha256": _sha(out / "source-qualification.json")},
              "cases": 8, "api_requests": 0, "model_responses_reused": 0}
    _write_new(out / "result.json", result)
    return result


def prepare_execution(design_protocol: Path, bridge_protocol: Path, source_batch: Path,
                      corpus: Path, *, design_anchor: str) -> dict[str, Any]:
    design_protocol, bridge_protocol = map(Path.resolve, (Path(design_protocol), Path(bridge_protocol)))
    design = _read(design_protocol)
    require(design == prepare_design(ROOT / design["source_protocol"]["path"]),
            "design protocol이 현재 source와 다르다")
    bridge = _read(bridge_protocol)
    require(bridge == BRIDGE.prepare_protocol(design_protocol), "bridge protocol 불일치")
    require(re.fullmatch(r"[0-9a-f]{40}", design_anchor) is not None, "design anchor 형식")
    rows = _source_rows(source_batch, design)
    corpus_value, corpus_rows = _corpus(corpus)
    batch_value = _read(source_batch)
    _verify_source_corpus_binding(design, batch_value, corpus_value, source_batch)
    _bind_rows(rows, corpus_rows)
    return {
        "schema": EXECUTION_SCHEMA, "status": "registered_only_after_commit",
        "design_protocol": _public_ref(design_protocol), "design_anchor": design_anchor,
        "bridge_protocol": _public_ref(bridge_protocol),
        "source_batch_sha256": _sha(source_batch), "corpus_sha256": _sha(corpus),
        "corpus_source_anchor": corpus_value["source_anchor"],
        "case_bindings": [{"case_id": row["case_id"], "seed": row["seed"],
                           "category": row["category"],
                           "checkpoint_sha256": row["checkpoint"]["sha256"]}
                          for row in rows],
        "planned_rows": 24, "fault_rows": 18, "normal_host_gate_rows": 6,
        "claims": deepcopy(design["claims"]),
    }


def _transport_ref_v2(value: object, *, slot: str, mode: str, messages: list,
                      params: dict[str, Any], returned: dict[str, Any]) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"},
            "bridge v2 transport ref 형식")
    ref = cast(dict[str, str], value)
    response_path = Path(ref["path"])
    require(response_path.is_absolute() and response_path.is_file()
            and _sha(response_path) == ref["sha256"]
            and response_path.parent.name == "responses", "bridge v2 response ref 불일치")
    request_path = response_path.parent.parent / "requests" / response_path.name
    require(request_path.is_file(), "bridge v2 request 원문 없음")
    response, intent = _read(response_path), _read(request_path)
    raw_returned = deepcopy(returned)
    raw_returned.pop("transport_ref", None)
    match = re.fullmatch(r"(c0[2-8])\.(legacy_none|enabled_low)\.d([1-4])\.a([12])", slot)
    require(match is not None and match.group(2) == mode, "bridge v2 slot/mode 불일치")
    if match is None:
        raise ValueError("bridge v2 slot 형식")
    require(response == {"schema": BRIDGE.RESPONSE_SCHEMA,
                         "request_sha256": _sha(request_path), "result": raw_returned},
            "bridge v2 response 원문 결합 불일치")
    require(intent.get("schema") == BRIDGE.REQUEST_SCHEMA
            and intent.get("slot") == slot and intent.get("case_id") == match.group(1)
            and intent.get("mode") == mode and intent.get("messages") == messages
            and intent.get("messages_sha256") == BRIDGE.digest(messages)
            and intent.get("params") == params
            and intent.get("params_sha256") == BRIDGE.digest(params)
            and intent.get("context_sha256") == BRIDGE.digest({
                "slot": slot, "messages": messages, "params": params}),
            "bridge v2 request 원문 결합 불일치")
    return deepcopy(ref)


def _dialogue_v2(state: dict[str, Any], *, run_id: str, action_index: int, mode: str,
                 verify_context: Callable[[], object], complete_factory: Callable) -> dict[str, Any]:
    sent, calls = O2.messages(state), []
    provider_ok = False
    for attempt in range(1, 3):
        slot = f"{run_id}.d{action_index}.a{attempt}"
        params = BRIDGE.MODE.build_request(sent, mode)
        require(params.pop("messages") == sent, "mode request가 messages를 바꿨다")
        verify_context()
        result = complete_factory(slot, deepcopy(state))(deepcopy(sent), deepcopy(params))
        verify_context()
        require(isinstance(result, dict) and result.get("request_params") == params,
                "provider result/params 불일치")
        result = deepcopy(result)
        transport = _transport_ref_v2(result.get("transport_ref"), slot=slot, mode=mode,
                                      messages=sent, params=params, returned=result)
        provider_ok = (result.get("provider_model") in BRIDGE.MODE.CALL.ACCEPTED_PROVIDER_MODELS
                       and result.get("finish_reason") == "stop"
                       and not result.get("error_type") and not result.get("infra_error")
                       and isinstance(result.get("content"), str))
        review = (O2.assess(result.get("content"), state) if provider_ok else {
            "status": "invalid", "format_status": "not_run", "semantic_status": "not_run",
            "codes": ["PROVIDER_UNAVAILABLE"]})
        calls.append({"attempt": attempt, "slot": slot, "messages": deepcopy(sent),
                      "messages_sha256": O2.O1._digest(sent),
                      "request_params": deepcopy(params), "transport_ref": transport,
                      "result": result, "review": deepcopy(review)})
        if not provider_ok or review["status"] == "valid" or attempt == 2:
            break
        current = O2.context(state)
        sent.extend([{"role": "assistant", "content": result.get("content") or ""},
                     {"role": "user", "content": json.dumps({
                         "validation_errors": review["codes"],
                         "output_schema": current["output_schema"],
                         "action_options": current["action_options"],
                         "count_evidence_groups": current["count_evidence_groups"],
                         "required": "Use one current option_id and exact visible fact_ids only."},
                         ensure_ascii=False, sort_keys=True)}])
    final = calls[-1]["review"]
    action = (O2.select_model(final, state) if provider_ok else
              {"action": "hold", "target": None, "fact_ids": []})
    return {"schema": "prospective_reasoning_mode_dialogue_v1",
            "state_digest": state["state_digest"], "mode": mode, "calls": calls,
            "action": action, "final_review": deepcopy(final),
            "model_valid": final["status"] == "valid",
            "semantic_feedback_used": len(calls) == 2,
            "automatic_retry": False, "provider_incomplete_hold": not provider_ok}


def _run_model_epoch(state: dict[str, Any], *, run_id: str, mode: str, out: Path,
                     verify_context: Callable[[], object], complete_factory: Callable,
                     execute_pair: Callable) -> dict[str, Any]:
    require(mode in BRIDGE.MODE.MODES, "reasoning mode 불일치")
    out = Path(out).resolve()
    require(not out.exists(), "model epoch out은 fresh여야 한다")
    verify_context()
    O2.context(state)
    require(state["budget_left"] <= 3, "paired probe 예산 초과")
    out.mkdir(parents=True, exist_ok=False)
    current = deepcopy(state)
    steps: list[dict[str, Any]] = []
    actions = probes = requests = 0
    provider_unusable = semantic_invalid = 0
    parent_wall = child_wall = 0.0
    stop_reason = None
    started = time.monotonic()
    while current["status"] == "open" and current["budget_left"] > 0:
        if actions >= 4:
            stop_reason = "action_budget_exhausted"
            break
        actions += 1
        stage = out / f"step-{actions:02d}"
        before = deepcopy(current)
        dialogue = _dialogue_v2(before, run_id=run_id, action_index=actions, mode=mode,
                                verify_context=verify_context,
                                complete_factory=complete_factory)
        RLOOP.L1.write_new(stage / "dialogue.json", dialogue)
        requests += len(dialogue["calls"])
        provider_unusable += int(dialogue["provider_incomplete_hold"])
        semantic_invalid += int(not dialogue["provider_incomplete_hold"]
                                and not dialogue["model_valid"])
        current, probes, parent_delta, child_delta, host = RLOOP._apply_action(
            before, dialogue["action"], stage, verify_context=verify_context,
            execute_pair=execute_pair, paired_probes=probes)
        parent_wall += parent_delta
        child_wall += child_delta
        RLOOP.L1.write_new(stage / "host-action.json", host)
        RLOOP.L1.write_new(stage / "state-after.json", current)
        steps.append({"step": actions, "action": deepcopy(dialogue["action"]),
                      "model_valid": dialogue["model_valid"], "calls": len(dialogue["calls"]),
                      "state_before_digest": before["state_digest"],
                      "state_after_digest": current["state_digest"]})
        if dialogue["action"]["action"] == "hold":
            break
    result = {"schema": "prospective_reasoning_mode_epoch_v1", "status": "complete",
              "run_id": run_id, "mode": mode, "model_actions": actions,
              "model_requests": requests, "paired_probes": probes,
              "provider_unusable_count": provider_unusable,
              "semantic_invalid_count": semantic_invalid,
              "decision_outcome": "confirmed" if current["status"] == "confirmed" else "HOLD",
              "confirmed_family": (current["hypotheses"][0]
                                   if current["status"] == "confirmed" else None),
              "final_status": current["status"], "steps": steps,
              "stop_reason": stop_reason,
              "parent_probe_wall_s": parent_wall, "owned_child_wall_s": child_wall,
              "epoch_wall_s": time.monotonic() - started,
              "automatic_retry": False, "repair": False, "recovery": False}
    RLOOP.L1.write_new(out / "result.json", result)
    return result


def _score(category: str, epoch: dict[str, Any]) -> dict[str, bool]:
    confirmed = epoch.get("confirmed_family")
    expected = {"F1": {"reward"}, "R1_ONLY": {"optimizer_update_count"},
                "COMPOUND": {"reward", "optimizer_update_count"}}.get(category, set())
    correct = confirmed in expected and epoch.get("decision_outcome") == "confirmed"
    partial = category == "COMPOUND" and correct
    return {"correct_confirmed_family": correct,
            "wrong_family_confirmation": confirmed is not None and confirmed not in expected,
            "compound_partial": partial,
            "compound_complete": False}


def _assessment(rows: list[dict[str, Any]]) -> str:
    if len(rows) != 24 or any(row["status"] != "complete" for row in rows):
        return "indeterminate"
    if any(row.get("provider_unusable_count", 0) for row in rows):
        return "indeterminate"
    normal = [row for row in rows if row["case_id"] in NORMAL_CASES]
    fault = [row for row in rows if row["case_id"] in FAULT_CASES]
    if any(row["model_requests"] or row["paired_probes"] for row in normal):
        return "not_confirmed"
    models = [row for row in fault if row["policy"] in BRIDGE.MODE.MODES]
    if any(row["wrong_family_confirmation"] for row in models):
        return "not_confirmed"
    counts = {mode: sum(row["correct_confirmed_family"] for row in models
                        if row["policy"] == mode) for mode in BRIDGE.MODE.MODES}
    return ("post_exposure_mode_candidate_confirmed_on_new_seeds"
            if counts["enabled_low"] > counts["legacy_none"] else "not_confirmed")


def _git(*args: str) -> bytes:
    run = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    require(run.returncode == 0, f"git {' '.join(args)} 실패")
    return run.stdout


def _verify_registered_file(path: Path, anchor: str, expected: dict[str, Any],
                            *, source_closure: dict[str, str] | None = None) -> None:
    path = Path(path).resolve()
    require(re.fullmatch(r"[0-9a-f]{40}", anchor) is not None, "anchor 형식")
    require(path.is_file() and path.is_relative_to(ROOT / "bench/protocols"),
            "등록 protocol 공개 경로 불일치")
    require(_read(path) == expected, "등록 protocol 현재 bytes 불일치")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    relative = path.relative_to(ROOT).as_posix()
    blob = _git("show", f"{anchor}:{relative}").replace(b"\r\n", b"\n")
    require(hashlib.sha256(blob).hexdigest() == _sha(path, lf=True),
            "등록 protocol anchor blob 불일치")
    for name, digest in (source_closure or {}).items():
        raw = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n")
        require(hashlib.sha256(raw).hexdigest() == digest
                and _sha(ROOT / name, lf=True) == digest,
                f"등록 source blob 불일치: {name}")


def verify_design(path: Path, anchor: str) -> dict[str, Any]:
    value = _read(path)
    source_protocol = ROOT / value.get("source_protocol", {}).get("path", "")
    expected = prepare_design(source_protocol)
    require(value == expected, "design protocol이 현재 source와 다르다")
    _verify_registered_file(path, anchor, expected,
                            source_closure=value["source_sha256_lf"])
    return value


def verify_execution(protocol: Path, anchor: str, design: Path, bridge: Path,
                     source_batch: Path, corpus: Path) -> dict[str, Any]:
    value = _read(protocol)
    require(value == prepare_execution(design, bridge, source_batch, corpus,
                                       design_anchor=value.get("design_anchor", "")),
            "execution protocol이 현재 입력과 다르다")
    design_value = verify_design(design, value["design_anchor"])
    bridge_value = BRIDGE._verify_registered(bridge, value["design_anchor"])
    require(bridge_value == BRIDGE.prepare_protocol(design), "bridge 등록 계약 불일치")
    _verify_registered_file(protocol, anchor, value)
    for name, row in design_value["tool_protocols"].items():
        target = ROOT / row["path"]
        require(_sha(target, lf=True) == row["sha256_lf"],
                f"실행 중 {name} protocol 변경")
    return value


def preflight(protocol: Path, anchor: str, design: Path, bridge: Path,
              source_batch: Path, corpus: Path, out: Path) -> dict[str, Any]:
    spec = verify_execution(protocol, anchor, design, bridge, source_batch, corpus)
    value = {"schema": "prospective_reasoning_confirmation_preflight_v1",
             "status": "complete", "execution_protocol": _public_ref(protocol),
             "design_anchor": spec["design_anchor"], "source_batch_sha256": _sha(source_batch),
             "corpus_sha256": _sha(corpus), "api_requests": 0, "gpu_started": False}
    _write_new(out, value)
    return value


def _verify_current_bytes(protocol_path: Path, spec: dict[str, Any], design_path: Path,
                          design: dict[str, Any], bridge_path: Path,
                          bridge: dict[str, Any]) -> None:
    require(_read(protocol_path) == spec, "실행 중 execution protocol 변경")
    require(_read(design_path) == design, "실행 중 design protocol 변경")
    require(_read(bridge_path) == bridge, "실행 중 bridge protocol 변경")
    require(_sha(ROOT / design["source_protocol"]["path"])
            == design["source_protocol"]["sha256"], "실행 중 source protocol 변경")
    for name, digest in design["source_sha256_lf"].items():
        require(_sha(ROOT / name, lf=True) == digest, f"실행 중 source 변경: {name}")
    for name, row in design["tool_protocols"].items():
        require(_sha(ROOT / row["path"], lf=True) == row["sha256_lf"],
                f"실행 중 {name} protocol 변경")


def _execute(protocol: Path, anchor: str, design: Path, bridge: Path, source_batch: Path,
             corpus: Path, lab: Path, bridge_root: Path, out: Path, *,
             complete_factory: Callable | None = None,
             execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    spec = verify_execution(protocol, anchor, design, bridge, source_batch, corpus)
    design_value = _read(design)
    bridge_value = _read(bridge)
    rows = _source_rows(source_batch, design_value)
    _corpus_value, corpus_rows = _corpus(corpus)
    _bind_rows(rows, corpus_rows)
    out = Path(out).resolve()
    require(not out.exists(), "confirmation out은 fresh여야 한다")
    out.mkdir(parents=True, exist_ok=False)
    lease = SOURCE.LEASE.lease_context(out / "lease-initial")
    require(lease.get("owner") == SOURCE.OWNER, "공용 Use-GpuLease owner 불일치")
    lease_id = lease.get("lease_id")
    if complete_factory is None:
        BRIDGE.initialize(bridge_root, bridge, spec["design_anchor"])
        complete_factory = lambda slot, _state: (
            lambda messages, params: BRIDGE.complete(bridge_root, slot, messages, params))
    started = time.monotonic()
    result_rows: list[dict[str, Any]] = []
    probe_count = 0
    references = {row["seed"]: row for row in rows if row["category"] == "NONE"}

    def verify_context() -> object:
        require(time.monotonic() - started < MAX_WALL_S, "confirmation wall 상한 종료")
        require(_sha(source_batch) == spec["source_batch_sha256"]
                and _sha(corpus) == spec["corpus_sha256"], "실행 중 source/corpus 변경")
        _verify_current_bytes(protocol, spec, design, design_value, bridge, bridge_value)
        current = SOURCE.LEASE.lease_context()
        require(current.get("owner") == SOURCE.OWNER
                and current.get("lease_id") == lease_id, "실행 중 lease identity 변경")
        return True

    for row in rows:
        reference = references[row["seed"]]
        corpus_row = corpus_rows[row["case_id"]]
        host_gate = corpus_row["evaluation_role"] == "normal_model_only"
        for policy in POLICIES:
            base = {"case_id": row["case_id"], "policy": policy, "status": "complete"}
            if host_gate:
                result_rows.append({**base, "decision_outcome": "HOLD", "model_requests": 0,
                                    "paired_probes": 0, "correct_confirmed_family": False,
                                    "wrong_family_confirmation": False,
                                    "provider_unusable_count": 0,
                                    "semantic_invalid_count": 0,
                                    "compound_partial": False, "compound_complete": False})
                continue

            def pair(state: dict, action: dict, stage: Path, *, _row=row,
                     _reference=reference) -> dict[str, Any]:
                nonlocal probe_count
                require(probe_count < MAX_PAIRED, "paired probe 전역 상한 초과")
                result = execute_host(
                    state, action, case_input=_row["input"], reference_input=_reference["input"],
                    case_runtime=Path(_row["metadata"]["runtime_facts"]["case"]["path"]),
                    reference_runtime=Path(_reference["metadata"]["runtime_facts"]["reference"]["path"]),
                    probe_protocol=ROOT / design_value["tool_protocols"]["paired_probe"]["path"],
                    view_protocol=ROOT / design_value["tool_protocols"]["view"]["path"],
                    anchor=anchor, lab=Path(lab).resolve(), out=stage,
                    verify_context=lambda _context=None: verify_context())
                probe_count += 1
                return result

            row_out = out / row["case_id"] / policy
            if policy == "strong_registered_rule":
                epoch = COMP.run_epoch(row["input"], reference["input"], policy=policy,
                                       run_id=f"{row['case_id']}.{policy}", out=row_out,
                                       verify_context=verify_context, complete_factory=None,
                                       execute_pair=pair)
                model_requests = epoch["request_count"]
                paired = epoch["probe_count"]
                provider_unusable = semantic_invalid = 0
            else:
                epoch = _run_model_epoch(corpus_row["state"],
                                         run_id=f"{row['case_id']}.{policy}", mode=policy,
                                         out=row_out, verify_context=verify_context,
                                         complete_factory=complete_factory, execute_pair=pair)
                model_requests = epoch["model_requests"]
                paired = epoch["paired_probes"]
                provider_unusable = epoch["provider_unusable_count"]
                semantic_invalid = epoch["semantic_invalid_count"]
            result_rows.append({**base, "decision_outcome": epoch["decision_outcome"],
                                "confirmed_family": epoch.get("confirmed_family"),
                                "model_requests": model_requests, "paired_probes": paired,
                                "provider_unusable_count": provider_unusable,
                                "semantic_invalid_count": semantic_invalid,
                                **_score(row["category"], epoch)})
    requests = sum(row["model_requests"] for row in result_rows)
    probes = sum(row["paired_probes"] for row in result_rows)
    require(requests <= MAX_API and probes <= MAX_PAIRED and probes * 2 <= MAX_SDK_ROLES,
            "전역 API/probe/SDK 상한 초과")
    result = {"schema": RESULT_SCHEMA, "status": "complete", "rows": result_rows,
              "planned_rows": 24, "model_requests": requests, "paired_probes": probes,
              "sdk_role_executions_upper_bound": probes * 2,
              "assessment": _assessment(result_rows),
              "claims": deepcopy(design_value["claims"])}
    _write_new(out / "result.json", result)
    return result


def execute(protocol: Path, anchor: str, design: Path, bridge: Path, source_batch: Path,
            corpus: Path, lab: Path, bridge_root: Path, out: Path, *,
            complete_factory: Callable | None = None,
            execute_host: Callable = HOST.execute_pair) -> dict[str, Any]:
    """Immutable ``git show``만 한 실행 구간에서 재사용하고 나머지 검증은 유지한다."""
    with COMP.CACHE.proof_scope():
        return _execute(protocol, anchor, design, bridge, source_batch, corpus, lab,
                        bridge_root, out, complete_factory=complete_factory,
                        execute_host=execute_host)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    design = sub.add_parser("prepare-design")
    design.add_argument("--source-protocol", type=Path, required=True)
    design.add_argument("--out", type=Path, required=True)
    corpus_cmd = sub.add_parser("prepare-corpus")
    corpus_cmd.add_argument("--design-protocol", type=Path, required=True)
    corpus_cmd.add_argument("--source-batch", type=Path, required=True)
    corpus_cmd.add_argument("--out", type=Path, required=True)
    execution = sub.add_parser("prepare-execution")
    for name in ("design-protocol", "bridge-protocol", "source-batch", "corpus", "out"):
        execution.add_argument(f"--{name}", type=Path, required=True)
    execution.add_argument("--design-anchor", required=True)
    check = sub.add_parser("preflight")
    for name in ("protocol", "design-protocol", "bridge-protocol", "source-batch",
                 "corpus", "out"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--anchor", required=True)
    run = sub.add_parser("execute")
    for name in ("protocol", "design-protocol", "bridge-protocol", "source-batch",
                 "corpus", "lab", "bridge-root", "out"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--anchor", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-design":
        _write_new(args.out, prepare_design(args.source_protocol))
    elif args.command == "prepare-corpus":
        build_corpus(args.design_protocol, args.source_batch, args.out)
    elif args.command == "prepare-execution":
        _write_new(args.out, prepare_execution(
            args.design_protocol, args.bridge_protocol, args.source_batch, args.corpus,
            design_anchor=args.design_anchor))
    elif args.command == "preflight":
        preflight(args.protocol, args.anchor, args.design_protocol, args.bridge_protocol,
                  args.source_batch, args.corpus, args.out)
    else:
        execute(args.protocol, args.anchor, args.design_protocol, args.bridge_protocol,
                args.source_batch, args.corpus, args.lab, args.bridge_root, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DESIGN_SCHEMA", "EXECUTION_SCHEMA", "RESULT_SCHEMA", "build_corpus", "execute", "preflight",
           "prepare_design", "prepare_execution", "verify_design", "verify_execution"]
