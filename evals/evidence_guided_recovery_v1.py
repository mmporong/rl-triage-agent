"""새 option-id recorded-count 선택을 등록 R1 교정과 독립 seed 재학습에 연결한다.

모델은 검사를 선택할 뿐 교정과 회복 방식을 생성하지 않는다. 이 모듈은
새 option 계약 study의 seed 2320 R1_ONLY/c03/eager_model 결과 하나만 받아,
등록 규칙 restore_configured_ppo_epochs와 fresh restart 300회를 seed 3332에 연결한다.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import core_seed_guard_v1 as SEEDS
import evidence_guided_probe_host_v1 as HOST
import evidence_guided_selector_v1 as SEL
import lineage_compound_first_recovery_v1 as FIRST
import optimizer_runtime_facts_v1 as FACT
import recorded_optimizer_probe_v1 as RECORDED
import registry_action_loop_v1 as ACTION_LOOP
import registry_action_options_v1 as OPTIONS
import run_prospective_agentic_comparison_v1 as RUN
import run_registry_action_study_v1 as STUDY
import runtime_observation_training_v1 as TRAIN
import selection_comparison_call_v2 as CALL
import selection_comparison_controller_v1 as C

from rl_triage import probe_loop as LOOP
from rl_triage.correction_ledger import CorrectionLedger

PROTOCOL_SCHEMA = "evidence_guided_recovery_protocol_v2"
CANDIDATE_SCHEMA = "evidence_guided_recovery_candidate_v2"
APPROVAL_SCHEMA = "evidence_guided_recovery_approval_v2"
RESULT_SCHEMA = "evidence_guided_recovery_result_v2"
SOURCE_SEED = 2320
RECOVERY_SEED = 3332
SMOKE_SEED = 4332
CASE_ID = "c03"
POLICY = "eager_model"
COMPONENT = "optimizer_update_count"
PROBE = RECORDED.PROBE
OPTION_ID = f"probe:{PROBE}"
CORRECTION = "restore_configured_ppo_epochs"
RECOVERY_MODE = "restart"
MAX_WALL_S = 5400
OWNER = RUN.OWNER
COMPARISON_PROTOCOL = ROOT / "bench/protocols/prospective_agentic_comparison_20261101_v5.json"
COMPARISON_ANCHOR = "164348c2b8df8f275371193306df6aa91d5d312b"
COMPARISON_SHA256_LF = "8cd83d6389cde2d5e9eb2972c0fa6fa4ed50cd40a7154bede3fd8baeadc48b0b"
STUDY_PROTOCOL_NAME = "registry_action_contract_development_20261101_v1.json"
PRIOR_COMPARISON_RESULT_SHA256 = "9aa3a919ca878e941f69fa2e1e8fe2c7aaa0c5ca4131d2b9b35dd42188577ed3"
SOURCE_FILES = tuple(sorted({
    *TRAIN.TRAIN.BOUND_SOURCES,
    "evals/evidence_guided_recovery_v1.py",
    "tests/test_evidence_guided_recovery_v1.py",
    "evals/runtime_observation_training_v1.py",
    "evals/runtime_training_inputs_v1.py",
    "evals/optimizer_runtime_facts_v1.py",
    "evals/run_prospective_agentic_comparison_v1.py",
    "evals/prospective_agentic_comparison_v1.py",
    "evals/evidence_guided_selector_v1.py",
    "evals/evidence_guided_probe_host_v1.py",
    "evals/recorded_optimizer_probe_v1.py",
    "evals/registry_action_options_v1.py",
    "tests/test_registry_action_options_v1.py",
    "evals/registry_action_loop_v1.py",
    "tests/test_registry_action_loop_v1.py",
    "evals/run_registry_action_study_v1.py",
    "tests/test_run_registry_action_study_v1.py",
    "evals/selection_comparison_controller_v1.py",
    "evals/residual_contract_v1.py",
    "evals/r1_repair_contract.py",
    "src/rl_triage/correction_ledger.py",
}))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TAG = re.compile(r"[a-z0-9][a-z0-9_]{0,79}\Z")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 객체여야 한다")
    return value


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": sha(path)}


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 저장소 내부여야 한다") from error


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    require(result.returncode == 0, "Git anchor/blob 결합을 확인하지 못했다")
    return result.stdout


def _registry() -> dict[str, Any]:
    return C.RC.correction_registry()


def _lease_admission(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """공용 lease 원검증 뒤 이 실행기의 등록 owner를 한 번 더 고정한다."""
    require(metadata.get("owner") == OWNER and metadata.get("state") == "running"
            and metadata.get("completedUtc") is None,
            "공용 GPU lease owner/state가 recovery 등록값과 다르다")
    return deepcopy(dict(metadata))


def prepare_protocol(study_protocol: Path, runtime_source_sha256_lf: Mapping[str, str]) -> dict[str, Any]:
    """study 결과를 보기 전에 option 선택·교정·seed·예산 계약을 고정한다."""
    study_protocol = Path(study_protocol).resolve()
    require(study_protocol.is_relative_to(ROOT)
            and study_protocol.name == STUDY_PROTOCOL_NAME,
            "고정 option study protocol 경로가 아니다")
    study = read(study_protocol)
    require(study.get("schema") == STUDY.SCHEMA
            and study.get("purpose") == "post_exposure_registry_action_contract_development_study",
            "고정 option study protocol schema/purpose가 다르다")
    require(study == STUDY.prepare_protocol(COMPARISON_PROTOCOL, PRIOR_COMPARISON_RESULT_SHA256),
            "option study protocol이 고정 prior/source closure와 다르다")
    base = study.get("base_comparison_protocol")
    require(isinstance(base, dict)
            and base.get("path") == _relative(COMPARISON_PROTOCOL)
            and base.get("sha256_lf") == COMPARISON_SHA256_LF
            and study.get("base_comparison_anchor") == COMPARISON_ANCHOR,
            "option study의 base comparison 결합이 다르다")
    runtime = dict(runtime_source_sha256_lf)
    require(len(runtime) == 9 and all(isinstance(name, str) and HEX64.fullmatch(value)
                                      for name, value in runtime.items()),
            "runtime pin은 정확한 SHA 9개여야 한다")
    return {
        "schema": PROTOCOL_SCHEMA,
        "purpose": "option_selected_recorded_count_registered_r1_recovery",
        "preselected_case": {"case_id": CASE_ID, "source_seed": SOURCE_SEED,
                             "category": "R1_ONLY", "policy": POLICY},
        "preselected_recovery_seed": RECOVERY_SEED,
        "smoke_seed": SMOKE_SEED,
        "selection_gate": {"selection_contract": "registry_action_options_v1",
                           "selected_by": "model_option_id", "option_id": OPTION_ID,
                           "probe": PROBE, "outcome": "abnormal",
                           "confirmed_component": COMPONENT, "model_must_select": True,
                           "provider_models": sorted(CALL.ACCEPTED_PROVIDER_MODELS)},
        "repair": {"correction_id": CORRECTION, "selected_by": "registered_host_rule",
                   "mode": "fresh_restart", "arm": "FIRST", "original_faults": ["R1"],
                   "applied_corrections": [CORRECTION], "checkpoint_loaded": False,
                   "source_checkpoint_provenance_only": True},
        "training": {"num_envs": 4096, "updates": 300,
                     "configured_learning_epochs": 5, "configured_mini_batches": 4,
                     "runtime_updates_to_observe": 3},
        "evaluation": {"conditions": 26, "reference": "same_source_seed_2320_NONE",
                       "corrected_seed": RECOVERY_SEED},
        "bounds": {"sdk_jobs": 3, "api_requests": 0, "corrections": 1,
                   "wall_s": MAX_WALL_S, "automatic_retry": False},
        "selection_study": {"path": _relative(study_protocol),
                            "sha256_lf": sha(study_protocol, lf=True),
                            "schema": STUDY.SCHEMA,
                            "result_schema": STUDY.RESULT_SCHEMA,
                            "base_comparison_anchor": COMPARISON_ANCHOR},
        "source_sha256_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "runtime_source_sha256_lf": deepcopy(runtime),
        "predictions": {"candidate": "eligible_only_if_model_selected_recorded_abnormal",
                        "recovery": "measured_not_assumed", "otherwise": "HOLD"},
        "claims": {"new_model_call": False, "model_selected_correction": False,
                   "recorded_probe_is_live_probe": False, "recovery": False},
    }


def verify_protocol(path: Path, anchor: str, lab: Path) -> dict[str, Any]:
    path, lab = Path(path).resolve(), Path(lab).resolve()
    value = read(path)
    study_ref = value.get("selection_study")
    require(isinstance(study_ref, dict) and isinstance(study_ref.get("path"), str),
            "option study protocol ref가 없다")
    study_path = ROOT / cast(dict[str, Any], study_ref)["path"]
    require(value == prepare_protocol(study_path, value.get("runtime_source_sha256_lf", {})),
            "recovery protocol 내용 또는 source closure가 다르다")
    require(HEX40.fullmatch(anchor) is not None, "anchor 형식이 다르다")
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    STUDY.verify_protocol(study_path, anchor, lab)
    for name, expected in {_relative(path): sha(path, lf=True), **value["source_sha256_lf"]}.items():
        current = ROOT / name
        require(current.is_file() and sha(current, lf=True) == expected,
                f"현재 source가 protocol과 다르다: {name}")
        blob = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(blob).hexdigest() == expected,
                f"anchor blob이 protocol과 다르다: {name}")
    for name, expected in value["runtime_source_sha256_lf"].items():
        target = lab / name
        require(target.is_file() and sha(target, lf=True) == expected,
                f"runtime source가 protocol과 다르다: {name}")
    return deepcopy(value)


def _hold(out: Path, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    value = {"schema": CANDIDATE_SCHEMA, "status": "HOLD", "reason": reason,
             "eligible": False, "evidence": evidence, "approval_granted": False}
    write_new(Path(out) / "candidate.json", value)
    return value


def _recorded_row(state: dict[str, Any]) -> dict[str, Any]:
    SEL.context(state)  # _state와 전체 history 원문을 재생한다.
    rows = [row for row in state["history"]
            if row.get("action") == "run_probe" and row.get("target") == PROBE]
    require(len(rows) == 1 and rows[0].get("outcome") == "abnormal",
            "recorded count abnormal receipt가 정확히 하나가 아니다")
    proof = rows[0].get("proof")
    require(isinstance(proof, dict), "recorded proof가 없다")
    for role in ("case", "reference"):
        plan = Path(proof[f"{role}_plan"]["path"])
        receipt = Path(proof[f"{role}_receipt"]["path"])
        require(ref(plan) == proof[f"{role}_plan"] and ref(receipt) == proof[f"{role}_receipt"],
                f"{role} recorded proof SHA가 바뀌었다")
        checked = RECORDED.checked_receipt(plan, receipt)
        require(checked["probe"] == PROBE
                and checked["outcome"] == ("abnormal" if role == "case" else "normal"),
                f"{role} recorded count outcome이 다르다")
    return rows[0]


def _model_selection(epoch_root: Path, result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for index, step in enumerate(result.get("steps", []), 1):
        if (step.get("selected_by") == "model_option_id"
                and step.get("selected_option_id") == OPTION_ID
                and step.get("action", {}).get("target") == PROBE):
            selected.append((index, cast(dict[str, Any], step)))
    require(len(selected) == 1, "모델이 option-id recorded count 검사를 고른 원문이 정확히 하나가 아니다")
    index, step = selected[0]
    stage = Path(epoch_root) / f"step-{index:02d}"
    paths = {name: stage / name for name in (
        "state-before.json", "option-context.json", "dialogue.json", "selected-action.json",
        "request-spec.json", "paired-proof.json", "state-after.json")}
    require(all(path.is_file() for path in paths.values()), "option 선택 단계 원문이 완전하지 않다")
    before, dialogue = read(paths["state-before.json"]), read(paths["dialogue.json"])
    option_context = OPTIONS.context(before)
    require(read(paths["option-context.json"]) == option_context,
            "저장 option context가 현재 state와 다르다")
    require(dialogue.get("schema") == "registry_action_options_dialogue_v1"
            and dialogue.get("model_valid") is True
            and dialogue.get("state_digest") == before.get("state_digest")
            and dialogue.get("action") == step.get("action")
            and dialogue.get("final_review", {}).get("selected_option_id") == OPTION_ID,
            "option-id recorded count 모델 action이 유효하지 않다")
    require(step.get("dialogue") == ref(paths["dialogue.json"])
            and step.get("state_after") == ref(paths["state-after.json"]),
            "epoch step이 option dialogue/state 원문과 다르다")
    calls = dialogue.get("calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("최종 모델 call 원문이 없다")
    calls = cast(list[dict[str, Any]], calls)
    review = OPTIONS.assess(calls[-1].get("result", {}).get("content"), before)
    require(calls[-1].get("review") == review == dialogue.get("final_review")
            and review.get("status") == "valid"
            and review.get("selected_option_id") == OPTION_ID,
            "최종 option review가 raw 응답·state와 다르다")
    provider = calls[-1].get("result", {}).get("provider_model")
    require(provider in CALL.ACCEPTED_PROVIDER_MODELS, "등록된 3.5 provider 응답이 아니다")
    selected_action = read(paths["selected-action.json"])
    require(selected_action.get("selected_by") == "model_option_id"
            and selected_action.get("selected_option_id") == OPTION_ID
            and selected_action.get("dialogue") == ref(paths["dialogue.json"])
            and selected_action.get("action") == dialogue["action"],
            "selected-action이 option dialogue와 다르다")
    require(read(paths["request-spec.json"]) == SEL.request_spec(before, dialogue["action"]),
            "option request spec이 현재 state/action과 다르다")
    proof = read(paths["paired-proof.json"])
    root_approval = stage / "probe/root-approval.json"
    require(proof.get("root_approval") == ref(root_approval),
            "paired proof와 root approval ref가 다르다")
    require(state.get("status") == "confirmed" and state.get("hypotheses") == [COMPONENT],
            "최종 selector가 optimizer update count를 확정하지 않았다")
    return {"dialogue": ref(paths["dialogue.json"]),
            "selected_action": ref(paths["selected-action.json"]),
            "root_approval": ref(root_approval), "provider_model": provider,
            "option_id": OPTION_ID, "selection_subject": "model_option_id",
            "state_digest": dialogue.get("state_digest"),
            "messages_sha256": calls[-1].get("messages_sha256"),
            "response_id": calls[-1].get("result", {}).get("response_id")}


def prepare_candidate(
    protocol_path: Path, anchor: str, source_batch_result: Path, study_result: Path,
    study_root: Path, *, lab: Path, out: Path,
) -> dict[str, Any]:
    """새 option study의 c03/eager_model 원문을 재검증하고 승인 전 후보만 봉인한다."""
    out = Path(out).resolve()
    require(not out.exists(), "candidate out은 fresh 폴더여야 한다")
    protocol = verify_protocol(protocol_path, anchor, lab)
    out.mkdir(parents=True)
    evidence = {"protocol": ref(protocol_path), "source_batch": None, "study_result": None}
    if not Path(source_batch_result).is_file() or not Path(study_result).is_file():
        return _hold(out, "required_result_missing", evidence)
    evidence.update(source_batch=ref(source_batch_result), study_result=ref(study_result))
    study = read(study_result)
    if study.get("schema") != STUDY.RESULT_SCHEMA or study.get("status") != "complete":
        return _hold(out, "selection_study_not_complete", evidence)
    study_protocol = ROOT / protocol["selection_study"]["path"]
    require(study.get("protocol") == ref(study_protocol)
            and study.get("source_batch") == ref(source_batch_result)
            and study.get("planned_rows") == 16,
            "option study 등록판/source/분모가 다르다")
    rows = study.get("rows")
    require(isinstance(rows, list) and len(rows) == 16, "option study 16행 분모가 다르다")
    rows = cast(list[dict[str, Any]], rows)
    matched = [row for row in rows if isinstance(row, dict)
               if row.get("case_id") == CASE_ID and row.get("policy") == POLICY]
    require(len(matched) <= 1, "고정 option study 행이 중복됐다")
    row = matched[0] if matched else None
    if not isinstance(row, dict) or row.get("status") != "complete" \
            or row.get("decision_outcome") != "confirmed" \
            or row.get("confirmed_family") != COMPONENT \
            or row.get("model_actions", 0) < 1 \
            or row.get("paired_probes", 0) < 1:
        return _hold(out, "model_selection_not_confirmed", evidence)
    cases = RUN._source_cases(Path(source_batch_result), read(COMPARISON_PROTOCOL), COMPARISON_ANCHOR)
    case = next((source for source in cases if source["case_id"] == CASE_ID), None)
    reference = next((source for source in cases if source["seed"] == SOURCE_SEED
                      and source["category"] == "NONE"), None)
    if case is None or reference is None:
        raise ValueError("고정 source/reference case가 없다")
    require(case["seed"] == SOURCE_SEED and case["category"] == "R1_ONLY",
            "고정 source/reference case가 다르다")
    epoch_root = Path(study_root).resolve() / CASE_ID / POLICY / "epoch"
    epoch_result_path, state_path = epoch_root / "result.json", epoch_root / "final-state.json"
    require(epoch_result_path.is_file() and state_path.is_file(), "epoch final 원문이 없다")
    epoch, state = read(epoch_result_path), read(state_path)
    require(epoch.get("schema") == ACTION_LOOP.SCHEMA and epoch.get("status") == "complete"
            and epoch.get("run_id") == f"{CASE_ID}.{POLICY}"
            and epoch.get("selection_subject") == "model_option_id"
            and epoch.get("decision_outcome") == "confirmed"
            and epoch.get("confirmed_family") == COMPONENT,
            "epoch result가 고정 진단과 다르다")
    recorded = _recorded_row(state)
    model = _model_selection(epoch_root, epoch, state)
    proof = cast(dict[str, dict[str, str]], recorded["proof"])
    root_approval = Path(model["root_approval"]["path"])
    approval = read(root_approval)
    require(approval.get("schema") == HOST.APPROVAL_SCHEMA
            and approval.get("selector_contract") == "new_evidence_guided_selector_not_legacy_model_dialogue"
            and approval.get("state_digest") == model["state_digest"]
            and approval.get("action", {}).get("target") == PROBE
            and approval.get("request_spec", {}).get("action") == approval.get("action")
            and approval.get("plans") == {
                "case": proof["case_plan"], "reference": proof["reference_plan"]}
            and isinstance(approval.get("actor"), str) and bool(approval["actor"].strip())
            and approval.get("approval_granted") is True,
            "recorded probe root approval이 유효하지 않다")
    source_cp, reference_cp = case["checkpoint"]["sha256"], reference["checkpoint"]["sha256"]
    require(row.get("checkpoint_sha256") == source_cp
            and row.get("reference_checkpoint_sha256") == reference_cp,
            "comparison row와 source checkpoint가 다르다")
    candidate = {
        "schema": CANDIDATE_SCHEMA, "status": "eligible", "reason": None, "eligible": True,
        "case": {"case_id": CASE_ID, "seed": SOURCE_SEED, "category": "R1_ONLY",
                 "policy": POLICY, "checkpoint_sha256": source_cp,
                 "reference_checkpoint_sha256": reference_cp},
        "diagnosis": {"component": COMPONENT, "probe": PROBE, "outcome": "abnormal",
                      "selection_contract": "registry_action_options_v1",
                      "selection_subject": "model_option_id", "option_id": OPTION_ID,
                      "selector_state": ref(state_path), "epoch_result": ref(epoch_result_path),
                      "model_selection": model, "root_approval": ref(root_approval),
                      "paired_receipts": deepcopy(proof)},
        "repair": deepcopy(protocol["repair"]),
        "source": {"case_result": ref(case["result"]), "case_fixed": ref(case["fixed"]),
                   "reference_result": ref(reference["result"]),
                   "reference_fixed": ref(reference["fixed"])},
        "approval_granted": False,
        "claims": {"recorded_probe_is_live_probe": False, "model_selected_correction": False,
                   "recovery": False},
    }
    write_new(out / "candidate.json", candidate)
    return candidate


def verify_candidate(candidate_path: Path) -> dict[str, Any]:
    """승인·실행 직전에 selector, 모델 원문, paired receipt를 다시 계산한다."""
    candidate = read(candidate_path)
    require(candidate.get("schema") == CANDIDATE_SCHEMA and candidate.get("eligible") is True
            and candidate.get("status") == "eligible" and candidate.get("approval_granted") is False,
            "eligible 후보가 아니다")
    diagnosis = candidate.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise TypeError("후보 diagnosis가 없다")
    diagnosis = cast(dict[str, Any], diagnosis)
    state_ref, epoch_ref = diagnosis.get("selector_state"), diagnosis.get("epoch_result")
    if not isinstance(state_ref, dict) or not isinstance(epoch_ref, dict):
        raise TypeError("selector/epoch ref 형식")
    state_ref = cast(dict[str, str], state_ref)
    epoch_ref = cast(dict[str, str], epoch_ref)
    state_path, epoch_path = Path(state_ref["path"]), Path(epoch_ref["path"])
    require(ref(state_path) == state_ref and ref(epoch_path) == epoch_ref,
            "selector/epoch 원문 SHA가 바뀌었다")
    state, epoch = read(state_path), read(epoch_path)
    recorded = _recorded_row(state)
    model_ref = diagnosis.get("model_selection")
    if not isinstance(model_ref, dict) or not isinstance(model_ref.get("dialogue"), dict):
        raise TypeError("모델 선택 ref 형식")
    model_ref = cast(dict[str, Any], model_ref)
    dialogue_ref = cast(dict[str, str], model_ref["dialogue"])
    dialogue_path = Path(dialogue_ref["path"])
    model = _model_selection(dialogue_path.parents[1], epoch, state)
    require(model == model_ref, "모델 선택 원문이 후보와 다르다")
    require(diagnosis.get("selection_contract") == "registry_action_options_v1"
            and diagnosis.get("selection_subject") == "model_option_id"
            and diagnosis.get("option_id") == OPTION_ID,
            "후보 option 선택 계약이 다르다")
    root_ref = diagnosis.get("root_approval")
    if not isinstance(root_ref, dict):
        raise TypeError("root approval ref 형식")
    root_ref = cast(dict[str, str], root_ref)
    require(ref(Path(root_ref["path"])) == root_ref == model["root_approval"],
            "root approval 원문이 바뀌었다")
    approval = read(Path(root_ref["path"]))
    require(approval.get("schema") == HOST.APPROVAL_SCHEMA
            and approval.get("selector_contract") == "new_evidence_guided_selector_not_legacy_model_dialogue"
            and approval.get("state_digest") == model["state_digest"]
            and approval.get("action", {}).get("target") == PROBE
            and approval.get("request_spec", {}).get("action") == approval.get("action")
            and approval.get("plans") == {
                "case": recorded["proof"]["case_plan"],
                "reference": recorded["proof"]["reference_plan"]}
            and isinstance(approval.get("actor"), str) and bool(approval["actor"].strip())
            and approval.get("approval_granted") is True,
            "root approval이 모델 선택과 다르다")
    require(diagnosis.get("paired_receipts") == recorded.get("proof"),
            "후보 paired receipt proof가 selector history와 다르다")
    for source_ref in candidate.get("source", {}).values():
        require(isinstance(source_ref, dict) and ref(Path(source_ref["path"])) == source_ref,
                "source case/reference 원문 SHA가 바뀌었다")
    return deepcopy(candidate)


def approve(candidate_path: Path, *, actor: str, out: Path) -> dict[str, Any]:
    verify_candidate(candidate_path)
    require(isinstance(actor, str) and bool(actor.strip()), "승인 주체가 없다")
    value = {"schema": APPROVAL_SCHEMA, "actor": actor, "candidate": ref(candidate_path),
             "correction_id": CORRECTION, "recovery_seed": RECOVERY_SEED,
             "approval_granted": True}
    write_new(Path(out) / "approval.json", value)
    return value


def _runtime_20(path: Path, checkpoint_sha256: str) -> dict[str, Any]:
    envelope, facts = FACT.checked(Path(path), expected_checkpoint_sha256=checkpoint_sha256)
    for suffix in ("expected_optimizer_steps", "optimizer_steps_min", "optimizer_steps_max",
                   "expected_minibatches", "minibatches_min", "minibatches_max"):
        require(facts[f"runtime:case:{suffix}"] == 20, "교정 학습 첫 3 update가 20/20이 아니다")
    require(facts["runtime:case:updates_observed"] == 3, "교정 학습 runtime update가 3개가 아니다")
    return envelope


def execute(
    protocol_path: Path, anchor: str, candidate_path: Path, approval_path: Path,
    source_batch_result: Path, *, lab: Path, isaac_python: Path, out: Path,
    runner: Callable, wait: Callable,
) -> dict[str, Any]:
    """승인된 후보 하나를 smoke→fresh300→fixed26 순서로 실행한다."""
    out = Path(out).resolve()
    require(not out.exists(), "execute out은 fresh 폴더여야 한다")
    protocol = verify_protocol(protocol_path, anchor, lab)
    candidate, approval = verify_candidate(candidate_path), read(approval_path)
    require(approval.get("schema") == APPROVAL_SCHEMA and approval.get("approval_granted") is True
            and approval.get("candidate") == ref(candidate_path)
            and candidate.get("eligible") is True and approval.get("recovery_seed") == RECOVERY_SEED,
            "후보/승인 결합이 다르다")
    cases = RUN._source_cases(Path(source_batch_result), read(COMPARISON_PROTOCOL), COMPARISON_ANCHOR)
    case = next(row for row in cases if row["case_id"] == CASE_ID)
    reference = next(row for row in cases if row["seed"] == SOURCE_SEED and row["category"] == "NONE")
    require(candidate["case"]["checkpoint_sha256"] == case["checkpoint"]["sha256"]
            and candidate["case"]["reference_checkpoint_sha256"] == reference["checkpoint"]["sha256"],
            "실행 직전 source checkpoint가 후보와 다르다")
    fresh = SEEDS.assert_fresh_seeds([RECOVERY_SEED, SMOKE_SEED], Path(lab))
    out.mkdir(parents=True)
    write_new(out / "freshness-before-execution.json", fresh)
    _lease_admission(FIRST.LEASE.lease_context(out / "lease-initial"))
    started = time.monotonic()

    def budget(reserve: float) -> None:
        elapsed = time.monotonic() - started
        require(math.isfinite(elapsed) and elapsed >= 0 and elapsed + reserve <= MAX_WALL_S,
                "회복 실행 wall 상한 예약이 부족하다")

    registry = _registry()
    entry = registry[COMPONENT]
    ledger_path = out / "corrections.ledger.jsonl"
    ledger = CorrectionLedger(ledger_path, registry)
    state_digest = LOOP.digest({"candidate": ref(candidate_path),
                                "selector_state": candidate["diagnosis"]["selector_state"]})
    receipts_digest = LOOP.digest(candidate["diagnosis"]["paired_receipts"])
    request = ledger.propose(state_digest, COMPONENT, receipts_digest, CORRECTION, RECOVERY_MODE,
                             {"case": entry["protocol_case"], "arm": "corrected"},
                             MAX_WALL_S, evidence={"selection_source": "validated_model_option_id",
                                                   "repair_source": "registered_host_rule",
                                                   "probe": PROBE, "evidence_kind": "recorded_training_reuse"})
    ledger.approve(request["request_id"], approval["actor"])
    ledger.consume(request["request_id"], state_digest)
    stages: list[dict[str, Any]] = []
    result = {"schema": RESULT_SCHEMA, "status": "running", "outcome": "HOLD",
              "source_seed": SOURCE_SEED, "recovery_seed": RECOVERY_SEED,
              "stages": stages, "repairs": [], "model_requests": 0,
              "claims": {"recorded_probe_is_live_probe": False,
                         "model_selected_correction": False, "recovery": False}}
    try:
        budget(600)
        smoke_stage = out / "smoke/train"
        smoke_manifest = TRAIN.build_manifest(
            protocol_path, protocol, anchor, stage=smoke_stage, seed=SMOKE_SEED, arm="NONE",
            original_faults=[], applied_corrections=[],
            source_checkpoint_sha256=case["checkpoint"]["sha256"],
            run_name=f"egr_smoke_s{SMOKE_SEED}", mode="inspection")
        smoke = TRAIN.run_training(smoke_stage, smoke_manifest, lab=lab,
                                   isaac_python=isaac_python, runner=runner, wait=wait)
        write_new(out / "smoke/training-result.json", smoke)
        stages.append({"stage": "smoke", "training": ref(out / "smoke/training-result.json")})

        budget(2400)
        repair_stage = out / "repair/train"
        manifest = TRAIN.build_manifest(
            protocol_path, protocol, anchor, stage=repair_stage, seed=RECOVERY_SEED, arm="FIRST",
            original_faults=["R1"], applied_corrections=[CORRECTION],
            source_checkpoint_sha256=case["checkpoint"]["sha256"],
            run_name=f"egr_repair_s{RECOVERY_SEED}", mode="train")
        require(manifest["arm"] == "FIRST" and manifest["original_faults"] == ["R1"]
                and manifest["applied_corrections"] == [CORRECTION],
                "R1-only 교정 manifest 의미가 다르다")
        training = TRAIN.run_training(repair_stage, manifest, lab=lab,
                                      isaac_python=isaac_python, runner=runner, wait=wait)
        write_new(out / "repair/training-result.json", training)
        _runtime_20(Path(training["runtime_facts"]["case"]["path"]),
                    training["checkpoint"]["sha256"])
        stages.append({"stage": "repair", "training": ref(out / "repair/training-result.json")})

        budget(1200)
        evidence = read(Path(training["evidence"]["path"]))
        corrected, fixed_execution = TRAIN.run_fixed(
            out / "repair/fixed", evidence, name=f"egr_repair_s{RECOVERY_SEED}",
            lab=lab, isaac_python=isaac_python, runner=runner, wait=wait)
        corrected_path = out / "repair/fixed/corrected.json"
        reference_report = read(Path(reference["fixed"]))
        bands = read(RUN.BANDS)["bands"]
        recovery = C.D1.recovery(corrected, reference_report, bands)
        receipt_value = {
            "schema": "evidence_guided_recovery_receipt_v2", "request_id": request["request_id"],
            "source_seed": SOURCE_SEED, "recovery_seed": RECOVERY_SEED,
            "source_checkpoint_sha256": case["checkpoint"]["sha256"],
            "output_checkpoint_sha256": training["checkpoint"]["sha256"],
            "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
            "correction_id": CORRECTION, "recovery_mode": "fresh_restart",
            "checkpoint_loaded": False, "corrected_report": ref(corrected_path),
            "reference_report": ref(Path(reference["fixed"])), "recovery": recovery,
            "fixed_execution": fixed_execution,
            "runtime_first3_expected_observed": "20/20", "gpu_s": time.monotonic() - started,
        }
        receipt_path = out / "repair/receipt.json"
        write_new(receipt_path, receipt_value)
        verdict = recovery["outcome"]
        ledger.receipt(request["request_id"], f"egr_repair_s{RECOVERY_SEED}", verdict,
                       sha(receipt_path), receipt_value["gpu_s"], 0)
        result.update(status="complete", outcome=verdict,
                      repairs=[ref(receipt_path)], correction_ledger=ref(ledger_path),
                      corrected_seed_is_independent=True,
                      baseline_reused_only_as_reference=True,
                      model_requests=0,
                      claims={"recorded_probe_is_live_probe": False,
                              "model_selected_correction": False,
                              "recovery": verdict == "recovered"})
        write_new(out / "result.json", result)
        return result
    except BaseException as error:
        write_new(out / "failure.json", {**result, "status": "invalid",
                  "error_type": type(error).__name__, "error": str(error),
                  "automatic_retry": False})
        raise


__all__ = ["approve", "execute", "prepare_candidate", "prepare_protocol", "verify_candidate",
           "verify_protocol"]
