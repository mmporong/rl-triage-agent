"""F1+R1 개발 시연: 검증한 비회복 receipt 뒤에만 두 번째 진단을 연다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))
import residual_contract_v1 as C
import residual_execution_v1 as E
import residual_model_call_v1 as MC
import r1_repair_demo_v2 as V
from preregistered_seed_guard import assert_fresh_seeds
from rl_triage import grounded_recovery as G
from rl_triage import recovery as REC
from rl_triage.correction_ledger import CorrectionLedger

PROTOCOL = ROOT / "bench/protocols/residual_demo_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task29/runs"
SEEDS = (2110, 2111)
TAG = re.compile(r"residual_v1_[0-9]{8}[a-z0-9_]{0,8}")
TASK = V.TASK
PROBE_BUDGET_S, CORRECTION_BUDGET_S = 1200, 3600
BANDS = "evals/results/p0c_fixed_eval_20261009/summary.json"
NEW_FILES = (
    "evals/residual_contract_v1.py", "evals/residual_execution_v1.py", "evals/residual_demo_v1.py",
    "evals/residual_model_call_v1.py", "bench/residual_faults_v1.py", "bench/residual_training_v1.py",
    "bench/residual_probe_v1.py", "bench/run_residual_demo_v1.ps1",
    "tests/test_residual_contract_v1.py", "tests/test_residual_execution_v1.py",
    "tests/test_residual_demo_v1.py", "tests/test_residual_model_call_v1.py", "tests/test_residual_training_v1.py",
)
CODE = V.CODE | set(NEW_FILES)
EVIDENCE = V.EVIDENCE
OUTCOMES = {
    "recovered": "서로 다른 두 등록 교정과 두 고리를 거친 정책이 고정 평가를 통과했다.",
    "recovered_after_one": "첫 교정 정책이 고정 평가를 통과해 두 번째 요청을 만들지 않았다.",
    "not_recovered": "두 교정 뒤에도 고정 평가를 통과하지 못해 멈췄다.",
    "source_failure_absent": "복합 정책이 고정 평가를 통과해 진단할 실패가 없었다.",
    "baseline_invalid": "정상 대조가 걷기 조건을 통과하지 못해 멈췄다.",
    "HOLD": "모델·검사·승인·평가 중 필요한 근거가 확정되지 않아 다음 요청을 만들지 않았다.",
    "invalid": "출처 또는 산출물 결합을 검증하지 못해 결과 판정을 멈췄다.",
}
read, write_new, sha = V.read, V.E.write_new, V.E.sha


def fixed_protocol() -> dict:
    return {
        "schema": "residual_demo_protocol_v1", "seeds": list(SEEDS), "task": TASK,
        "design_coupling": "설계자가 F1·R1 단일 결함과 앞선 수리 결과를 본 뒤 만든 복합 개발 측정이다. 새 seed는 이 결합을 없애지 않는다.",
        "hypothesis": "두 결함이 함께 남을 때 한 성분의 교정 뒤 유효한 비회복 평가가 두 번째 진단의 필요를 알려 준다.",
        "prediction": {"NONE": "walks", "COMPOUND": "unhealthy", "FIRST": "unhealthy", "BOTH": "healthy",
                       "stop_on_counterexample": "복합 실패 부재·첫 교정 회복·미확정이면 중단한다. 결함 강도나 seed를 결과에 맞춰 교체하지 않는다"},
        "arms": ["NONE", "COMPOUND", "FIRST", "BOTH"], "original_faults": ["F1", "R1"],
        "training": {"iterations": 300, "num_envs": 4096, "same_case_seed_all_arms": True,
                     "mode": "restart", "checkpoint_loaded": False,
                     "source_checkpoint_sha256": "앞선 정책 provenance만 결합; 가중치·optimizer 상태는 읽어 학습하지 않음",
                     "arm_order": "NONE → COMPOUND → 모델과 검사로 선택한 FIRST → 다른 등록 교정을 더한 BOTH",
                     "equal_checkpoint": "같은 seed·복원 설정이면 NONE/BOTH가 같을 수 있으나 SHA 동일성은 성공 조건이 아님"},
        "loop_registry": C.registry(), "correction_registry": C.correction_registry(),
        "confirmation_scope": "epoch 지역에서 이번에 수리할 한 성분의 확정. 다른 성분의 전역 부재를 뜻하지 않음",
        "epoch2_gate": ["첫 승인 원장 proposed/approved/consumed/receipt 각 1회",
                       "교정 전후 checkpoint·manifest·readback·평가 SHA 결합",
                       "정상 기준과 동결 고정 평가를 다시 계산한 not_recovered", "성공한 등록 교정 한 개",
                       "새 checkpoint로 빈 관측과 새 지역 가설; 앞선 지역 기각은 전역 후보에서 빼지 않음"],
        "limits": {"epochs": 2, "distinct_corrections": 2, "probes_per_epoch": 7,
                   "api_requests_total": 20, "probe_budget_s": PROBE_BUDGET_S,
                   "correction_budget_s": CORRECTION_BUDGET_S},
        "model": {"id": C.MODEL, "feedback_retries": 1, "infrastructure_dialogues": 2,
                  "budget": "두 seed가 공유하는 요청 전 배타 예약 원장. 피드백·인프라 재시도·미응답 예약 포함, 21번째 HTTP 금지",
                  "pending": "미완료 예약이 있으면 후속 요청과 완료 판정을 막음"},
        "inspection": {"runtime_updates": 3, "configured_epochs": 5, "mini_batches": 4,
                       "normal": "3개 update 모두 실제 optimizer.step와 minibatch가 각각 20",
                       "abnormal": "완결된 update 중 하나라도 등록 횟수와 다름", "incomplete": "unknown → HOLD",
                       "checkpoint_loaded": False, "scope": "현재 누적 교정 설정으로 새로 시작한 학습 루프 계측"},
        "evaluation": {"protocol": "bench/protocols/fixed_eval_v1.json", "conditions": 26,
                       "bands": BANDS, "reference": "같은 seed의 최초 NONE; epoch 사이에 바꾸지 않음",
                       "healthy": "recovered", "unhealthy": "not_recovered", "undetermined": "HOLD"},
        "recovery_selection_basis": "보상 v4와 R1의 재학습 결과를 사용하여 모든 arm을 같은 초기화·300회로 비교한다. 복합 결함에서 이어 학습의 효과는 측정하지 않는다.",
        "success_conditions": ["two_fresh_seeds", "both_source_failures", "both_first_not_recovered",
                               "both_second_receipts_recovered", "two_distinct_registered_repairs",
                               "no_unnecessary_epoch2_on_cpu_replay", "api_reserved_at_most_20_no_pending"],
        "cpu_replay": "NONE·단일 F1·단일 R1은 두 번째 요청 0. 두 교정 순서·위조 receipt·unknown·중복 교정·예산을 테스트",
        "outcome_wording": OUTCOMES,
        "aggregate_wording": {"both": "두 개발 seed에서 두 고리와 두 교정 뒤 회복했다.",
                              "mixed": "개발 seed 간 결과가 달라 후속 사용 규칙을 바꾸지 않는다.",
                              "none": "두 번째 고리의 등록 성공 조건을 충족하지 못했다."},
        "failure_policy": "미확정은 HOLD, 예외는 invalid로 기록하고 배치를 멈춤. 기존 tag·결과 재사용 금지",
        "approval": "운영자가 등록 검사·교정을 실행 전에 위임하고 각 요청의 승인을 한 번 소비한다. 사용자 신원 인증을 주장하지 않음",
        "gpu": "공용 Use-GpuLease 취득·해제 + 각 stage GPU 입장 검사; GPU와 커밋은 메인",
    }


def dynamic(isaaclab: Path) -> dict:
    runtime = V.dynamic(isaaclab)["runtime_source_sha256_lf"]
    rewards = "source/isaaclab/isaaclab/envs/mdp/rewards.py"
    runtime[rewards] = sha(isaaclab / rewards, True)
    return {"source_sha256_lf": {p: sha(ROOT / p, True) for p in sorted(CODE)},
            "evidence_sha256_lf": {p: sha(ROOT / p, True) for p in EVIDENCE},
            "runtime_source_sha256_lf": runtime}


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**fixed_protocol(), **dynamic(isaaclab)})
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, anchor: str | None = None, *, committed: bool = True) -> tuple[dict, str]:
    value = read(PROTOCOL)
    if value != {**fixed_protocol(), **dynamic(isaaclab)}:
        raise ValueError("사전등록·소스·입력·런타임 결합 불일치")
    head = V.git("rev-parse", "HEAD")
    anchor = anchor or head
    if committed:
        paths = sorted(CODE | set(EVIDENCE) | {PROTOCOL.relative_to(ROOT).as_posix()})
        V.git("merge-base", "--is-ancestor", anchor, head)
        V.git("ls-files", "--error-unmatch", "--", *paths)
        if V.git("status", "--porcelain", "--", *paths):
            raise ValueError("결합 파일을 실행 전에 커밋해야 한다")
        for path in paths:
            if V.git("rev-parse", f"{anchor}:{path}") != V.git("rev-parse", f"{head}:{path}"):
                raise ValueError("실험 commit 뒤 결합 blob이 바뀌었다")
    return value, anchor


def binding(protocol: dict, anchor: str, *, applied=None, faults=None, checkpoint=None, updates=300) -> dict:
    return {"source_commit_sha": anchor, "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256_lf": sha(PROTOCOL, True), "source_sha256_lf": protocol["source_sha256_lf"],
            "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"], "task": TASK,
            "num_envs": 4096, "max_iterations": updates, "configured_learning_epochs": 5,
            "configured_mini_batches": 4, "runtime_updates_to_observe": 3,
            "source_checkpoint_sha256": checkpoint, "original_faults": list(faults if faults is not None else ["F1", "R1"]),
            "applied_corrections": C.validate_applied(list(applied or []))}


def recovery(report: dict, reference: dict, bands: dict) -> dict:
    if report["versions"] != reference["versions"]:
        raise ValueError("고정 평가 런타임 버전이 다르다")
    verdict = REC.verdict(REC.compare(G._behavior(report), G._behavior(reference)), bands)
    return {"verdict": verdict, "outcome": {"healthy": "recovered", "unhealthy": "not_recovered",
                                           "undetermined": "unknown"}[verdict["label"]]}


def diagnosis_packet(current: dict, reference: dict, epoch: int, applied: list, prior: dict | None) -> dict:
    facts = {}
    for artifact, label in ((current, "case"), (reference, "reference")):
        path = Path(artifact["telemetry"]["path"])
        if sha(path) != artifact["telemetry"]["sha256"]:
            raise ValueError("진단 telemetry SHA 불일치")
        for name, row in sorted(read(path)["summary"].items()):
            if isinstance(row, dict) and "mean_last_20pct" in row:
                facts[f"telemetry:{label}:{name}"] = row["mean_last_20pct"]
    return {"schema": C.SCHEMA, "epoch": epoch, "applied_corrections": applied,
            "prior_receipt": prior, "facts": facts, "observed": [], "budget_left": 7,
            "execution_budget_s": PROBE_BUDGET_S}


def _files(paths: list[Path]) -> dict:
    return {str(path.resolve()): sha(path) for path in sorted(set(paths))}


def accepted_correction_evidence(model: dict, packet_path: Path, api_root: Path, expected: dict) -> dict:
    """저장 응답을 같은 계약으로 재검토하고 실제 HTTP 예약·응답에 연결한다."""
    data = packet_path.read_bytes()
    if read(packet_path) != expected or data != MC._packet_bytes(expected):
        raise ValueError("교정 모델 packet이 현재 근거와 다르다")
    attempts = model.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise ValueError("교정 모델 대화 횟수 불일치")
    for index, attempt in enumerate(attempts, 1):
        name = f"correction.dialogue{index}.json"
        path = packet_path.parent / name
        raw = read(path)
        verdict = MC._review_record(raw, data, "correction", api_root)
        recomputed = {"record": name, "record_sha256": sha(path), "review": verdict,
                      "request_receipts": raw["request_receipts"]}
        if attempt != recomputed or read(packet_path.parent / f"correction.review{index}.json") != recomputed:
            raise ValueError("교정 모델 저장 검토 결과 불일치")
        if index < len(attempts) and verdict["status"] != "infra_error":
            raise ValueError("인프라 오류 외 교정 재시도")
    review = attempts[-1]["review"]
    if (model.get("review") != review or review["status"] != "valid"
            or review["decision"]["decision"] != "propose" or not review["decision"]["agrees_with_rule"]):
        raise ValueError("수락된 교정 모델 제안이 없다")
    if MC.audit_budget(api_root)["pending"]:
        raise ValueError("모델 요청 예약이 완결되지 않았다")
    return {"packet_sha256": sha(packet_path), "model_review_digest": C.L.digest(model),
            "accepted_record_sha256": attempts[-1]["record_sha256"],
            "request_receipts": [r for a in attempts for r in a["request_receipts"]]}


def seal_repair(path: Path, *, request_id: str, epoch: int, previous: str | None,
                applied_before: list, applied_after: list, source: dict, artifact: dict,
                evaluation: dict, reference: dict, reference_path: Path,
                train_binding: dict, seed: int, outcome: dict, state: dict, probe_steps: list,
                decision: dict, probe_prereg: dict, model: dict, api_root: Path,
                source_report: dict, source_report_path: Path) -> dict:
    # 새 private 산출물과 checkpoint의 SHA를 묶으며 원본 파일은 수정하지 않는다.
    paths = [Path(artifact["checkpoint"]["path"]), Path(source["checkpoint"]["path"]),
             Path(artifact["telemetry"]["path"]), reference_path, source_report_path,
             *[Path(p) for p in evaluation["report_paths"].values()]]
    for key in ("manifest_path", "readback_path", "records_path"):
        paths.append(Path(artifact[key]))
    paths.append(path.parent / "probes.ledger.jsonl")
    paths.extend(p for p in (path.parent / "model").rglob("*") if p.is_file())
    value = {"schema": "residual_repair_receipt_bundle_v1", "request_id": request_id, "epoch": epoch,
             "previous_result_sha256": previous, "applied_before": applied_before, "applied_after": applied_after,
             "input_checkpoint_sha256": source["checkpoint"]["sha256"], "artifact": artifact,
             "evaluation": evaluation, "reference": reference, "reference_path": str(reference_path),
             "binding": train_binding, "seed": seed, "recovery": outcome,
             "state": state, "probe_steps": probe_steps, "initial_decision": decision,
             "probe_prereg": probe_prereg, "model_correction": model, "api_root": str(api_root.resolve()),
             "source_report": source_report, "source_report_path": str(source_report_path),
             "inventory_sha256": _files(paths)}
    write_new(path, value)
    return value


def verify_repair(path: Path, ledger: CorrectionLedger, *, expected_epoch: int, expected_applied: list,
                  expected_input_sha: str, previous: str | None, reference: dict, bands: dict,
                  expected_binding: dict, seed: int, backend: Any = E) -> dict:
    """원장·파일·동일한 고정 평가 규칙으로 receipt를 재검증한다."""
    value = read(path)
    after = C.validate_applied(value["applied_after"])
    if (value.get("schema") != "residual_repair_receipt_bundle_v1"
            or value["epoch"] != expected_epoch or value["applied_before"] != expected_applied
            or len(after) != expected_epoch or after[:-1] != expected_applied
            or value["input_checkpoint_sha256"] != expected_input_sha
            or value["previous_result_sha256"] != previous or value["reference"] != reference
            or value["binding"] != expected_binding or value["seed"] != seed):
        raise ValueError("교정 receipt의 epoch·입력·누적 교정·기준 결합 불일치")
    inventory = value["inventory_sha256"]
    required = [value["artifact"][key] for key in ("manifest_path", "readback_path", "records_path")]
    required += [value["artifact"]["checkpoint"]["path"], value["artifact"]["telemetry"]["path"],
                 value["reference_path"], *value["evaluation"]["report_paths"].values()]
    if not inventory or any(str(Path(p).resolve()) not in inventory for p in required):
        raise ValueError("receipt 필수 산출물 SHA 누락")
    for filename, digest in inventory.items():
        if sha(Path(filename)) != digest:
            raise ValueError("교정 receipt 파일 SHA 불일치")
    probe_path = path.parent / "probes.ledger.jsonl"
    if str(probe_path.resolve()) not in inventory:
        raise ValueError("검사 원장 SHA 누락")
    probe_ledger = C.M.RegistryLedger(probe_path, C.registry())
    probe_requests = probe_ledger.requests()
    prereg = {"epoch": expected_epoch, "checkpoint_sha256": expected_input_sha, "seed": seed,
              "applied_corrections": expected_applied, "prior_result_sha256": previous,
              "protocol_sha256_lf": expected_binding["protocol_sha256_lf"],
              "experiment_commit": expected_binding["source_commit_sha"],
              "registry_digest": C.M.registry_digest(C.registry())}
    if value["probe_prereg"] != prereg or len(probe_requests) != len(value["probe_steps"]):
        raise ValueError("검사 epoch 사전등록 또는 요청 수 불일치")
    replay = C.initial_state(value["initial_decision"], expected_applied)
    for row in value["probe_steps"]:
        rid, probe = row["request_id"], row["probe"]
        req = probe_requests.get(rid, {})
        events = [e["event"] for e in probe_ledger.events() if e.get("request_id") == rid]
        if (events != ["proposed", "approved", "consumed", "receipt"] or req.get("state") != "done"
                or req.get("prereg_digest") != C.L.digest(prereg) or req.get("probe") != probe
                or req.get("registry_digest") != probe_ledger.registry_digest
                or req.get("args_digest") != C.L.digest(C.registry()["probes"][probe]["args"])
                or {**req.get("receipt", {}), "probe": probe} != row
                or row.get("exit_code") != 0 or row.get("over_budget") is not False):
            raise ValueError("검사 승인·소비·receipt 재생 불일치")
        replay = C.advance(replay, probe, row["outcome"])
    if replay != value["state"]:
        raise ValueError("검사 receipt와 지역 확정 상태 불일치")
    request = ledger.requests().get(value["request_id"], {})
    events = [e["event"] for e in ledger.events() if e.get("request_id") == value["request_id"]]
    component = C.confirmed_component(value["state"], expected_applied)
    evidence = {"epoch": expected_epoch, "input_checkpoint_sha256": expected_input_sha,
                "applied_before": expected_applied, "previous_result_sha256": previous,
                "binding_digest": C.L.digest(expected_binding)}
    source_report_path = Path(value["source_report_path"])
    if (str(source_report_path.resolve()) not in inventory or read(source_report_path) != value["source_report"]
            or value["source_report"]["checkpoint"]["sha256"] != expected_input_sha):
        raise ValueError("교정 전 모델 입력 평가 결합 불일치")
    confirming = next((row for row in value["probe_steps"] if row["probe"] == (
        "P_reward" if component == "reward" else C.NEW_PROBE)), None)
    if confirming is None:
        raise ValueError("모델 교정 근거인 확정 검사 없음")
    expected_packet = C.correction_packet(value["state"], confirming, value["source_report"], expected_input_sha,
                                          sha(source_report_path), expected_applied)
    evidence["accepted_model"] = accepted_correction_evidence(value["model_correction"],
        path.parent / "model/correction.packet.json", Path(value["api_root"]), expected_packet)
    if (events != ["proposed", "approved", "consumed", "receipt"] or request.get("state") != "done"
            or request.get("registry_digest") != ledger.registry_digest
            or request.get("correction_id") != after[-1] or request.get("confirmed_mechanism") != component
            or component is None or C.CORRECTIONS[component] != after[-1]
            or request.get("loop_state_digest") != C.L.digest(value["state"])
            or request.get("receipts_digest") != C.L.digest(value["probe_steps"])
            or request.get("evidence") != evidence or request.get("recovery_mode") != "restart"):
        raise ValueError("승인·소비·성분 확정 원장 결합 불일치")
    receipt = request.get("receipt", {})
    seconds = receipt.get("gpu_s")
    if (receipt.get("exit_code") != 0 or receipt.get("over_budget") is not False
            or type(seconds) not in (int, float) or not 0 <= seconds <= CORRECTION_BUDGET_S
            or receipt.get("result_sha") != sha(path)):
        raise ValueError("유효한 완결 교정 receipt가 없다")
    arm = "FIRST" if expected_epoch == 1 else "BOTH"
    backend.validate_artifact(value["artifact"], expected_binding, arm, seed)
    if not E.V1._finished(value["evaluation"]["execution"]):
        raise ValueError("교정 평가 실행이 완결되지 않았다")
    for key, filename in value["evaluation"]["report_paths"].items():
        if read(Path(filename)) != value["evaluation"]["reports"][key]:
            raise ValueError("평가 파일과 receipt 내용 불일치")
    if read(Path(value["reference_path"])) != reference:
        raise ValueError("정상 기준 파일이 바뀌었다")
    validated = V.validated_reports(value["evaluation"], {arm: value["artifact"]})[arm]
    measured = recovery(validated, reference, bands)
    if measured != value["recovery"] or receipt.get("verdict") != measured["outcome"]:
        raise ValueError("고정 평가 재계산과 receipt 판정 불일치")
    return {"verdict": measured["outcome"], "correction_id": after[-1],
            "result_sha256": sha(path), "checkpoint_sha256": value["artifact"]["checkpoint"]["sha256"]}


def run_case(folder: Path, tag: str, seed: int, isaaclab: Path, protocol: dict, anchor: str,
             approver: str, api_root: Path, *, backend: Any = E, caller=None, original_faults=("F1", "R1")) -> dict:
    # 공개 execute는 복합 mask만 전달한다. 같은 고리의 CPU replay는 단일/NONE도 검증한다.
    faults = list(E.FAULTS.normalize_faults(original_faults))
    source_arm = "COMPOUND" if len(faults) == 2 else faults[0] if faults else "NONE"
    folder.mkdir(parents=True, exist_ok=False)
    result = {"schema": "residual_case_result_v1", "seed": seed, "outcome": "invalid", "epochs": [],
              "costs": {}, "unresolved_components": C.unresolved_components([])}
    correction_ledger = CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    active_correction = active_probe = None
    probe_ledger = None
    stage = "source_training"
    bands = read(ROOT / BANDS)["bands"]

    def finish(outcome: str, reason: str | None = None) -> dict:
        result.update(outcome=outcome, wording=OUTCOMES[outcome], reason=reason,
                      second_epoch_requests=len(result["epochs"]) - 1 if result["epochs"] else 0)
        result["demo_success"] = (outcome == "recovered" and len(result["epochs"]) == 2
                                  and result["epochs"][0].get("receipt", {}).get("verdict") == "not_recovered"
                                  and result["epochs"][1].get("receipt", {}).get("verdict") == "recovered")
        write_new(folder / "result.json", result)
        return result

    try:
        artifacts = {}
        for arm in dict.fromkeys(("NONE", source_arm)):
            verify(isaaclab, anchor)
            train_binding = binding(protocol, anchor, faults=[] if arm == "NONE" else faults)
            artifacts[arm] = backend.train(folder / "training" / arm.lower(), isaaclab, arm, seed,
                                           f"{tag}_s{seed}_{arm.lower()}", train_binding)
            backend.validate_artifact(artifacts[arm], train_binding, arm, seed)
            write_new(folder / f"artifact_{arm.lower()}.json", artifacts[arm])
            result["costs"][f"train_{arm}"] = V._duration(artifacts[arm])
        stage = "source_evaluation"
        verify(isaaclab, anchor)
        evaluation = backend.fixed_evaluate(folder / stage, isaaclab, artifacts, f"{tag}_s{seed}",
                                              binding(protocol, anchor, faults=faults))
        result["costs"][stage] = V._duration(evaluation)
        reports = V.validated_reports(evaluation, artifacts)
        gate = recovery(reports[source_arm], reports["NONE"], bands)
        result["source_gate"] = {**gate, "none_walks": V.R1.U.walks(reports["NONE"])}
        if not result["source_gate"]["none_walks"]:
            return finish("baseline_invalid")
        if gate["outcome"] == "recovered":
            return finish("source_failure_absent")
        if gate["outcome"] != "not_recovered":
            return finish("HOLD", "source_evaluation_unknown")
        stage = "reference"
        verify(isaaclab, anchor)
        reference_probe = backend.legacy_probe(folder / stage, isaaclab, {"NONE": artifacts["NONE"]},
            f"{tag}_s{seed}_ref", binding(protocol, anchor, faults=[]))
        result["costs"][stage] = V._duration(reference_probe)
        current, current_report = artifacts[source_arm], reports[source_arm]
        current_report_path = Path(evaluation["report_paths"][source_arm])
        reference_path = Path(evaluation["report_paths"]["NONE"])
        applied, prior, prior_path = [], None, None
        previous_input, previous_binding = None, None

        for epoch in (1, 2):
            # 이 검사보다 앞에서 epoch2 폴더·요청·모델 호출을 만들지 않는다.
            if epoch == 2:
                if prior_path is None or previous_input is None or previous_binding is None:
                    return finish("HOLD", "missing_first_receipt")
                prior = verify_repair(prior_path, correction_ledger, expected_epoch=1, expected_applied=[],
                    expected_input_sha=previous_input, previous=None, reference=reports["NONE"], bands=bands,
                    expected_binding=previous_binding, seed=seed, backend=backend)
                if prior["verdict"] != "not_recovered":
                    return finish("HOLD", "epoch2_requires_not_recovered")
            verify(isaaclab, anchor)
            epoch_folder = folder / f"epoch{epoch}"
            epoch_folder.mkdir(exist_ok=False)
            row = {"epoch": epoch, "applied_before": list(applied), "probe_steps": [], "model": {},
                   "confirmation_scope": "one_component_this_epoch", "unresolved_components": C.unresolved_components(applied)}
            result["epochs"].append(row)
            stage = f"epoch{epoch}_diagnosis"
            packet = diagnosis_packet(current, artifacts["NONE"], epoch, list(applied), prior)
            diagnosis = MC.decide(epoch_folder / "model", "diagnosis", packet, api_root, caller=caller)
            row["model"]["diagnosis"] = diagnosis
            review = diagnosis["review"]
            if review["status"] != "valid":
                return finish("HOLD", "diagnosis_" + review["status"])
            decision = review["decision"]
            if decision["action"] != "continue":
                return finish("HOLD", "diagnosis_hold")
            state = C.initial_state(decision, applied)
            probe_ledger = C.M.RegistryLedger(epoch_folder / "probes.ledger.jsonl", C.registry())
            prereg = {"epoch": epoch, "checkpoint_sha256": current["checkpoint"]["sha256"],
                      "seed": seed, "applied_corrections": list(applied),
                      "prior_result_sha256": prior["result_sha256"] if prior else None,
                      "protocol_sha256_lf": sha(PROTOCOL, True), "experiment_commit": anchor,
                      "registry_digest": C.M.registry_digest(C.registry())}
            while state["status"] == "open":
                verify(isaaclab, anchor)
                probe = state["next_probe"]
                stage = f"epoch{epoch}_probe{len(state['observed']) + 1}_{probe}"
                request = probe_ledger.propose(prereg, probe, PROBE_BUDGET_S)
                active_probe = request["request_id"]
                probe_ledger.approve(active_probe, approver)
                probe_ledger.consume(active_probe, prereg, probe)
                probe_stage = epoch_folder / f"probe{len(state['observed']) + 1}_{probe}"
                if probe == C.NEW_PROBE:
                    controls = {}
                    for label, artifact in (("control", artifacts["NONE"]), ("case", current)):
                        controls[label] = backend.inspect(probe_stage / label, isaaclab, artifact["arm"], seed,
                            f"{tag}_s{seed}_e{epoch}_opt_{label}", binding(protocol, anchor,
                                faults=[] if label == "control" else faults,
                                applied=[] if label == "control" else applied,
                                checkpoint=artifact["checkpoint"]["sha256"], updates=3))
                    measured = controls["case"]["measurement"]
                    outcome = measured["status"] if controls["control"]["measurement"]["status"] == "normal" else "unknown"
                    seconds = sum(V._duration(x) for x in controls.values())
                    write_new(probe_stage / "evidence.json", controls)
                    fields = ("status", "optimizer_steps", "minibatches")
                    measurement: dict = {label: {k: controls[label]["measurement"][k] for k in fields} for label in controls}
                    measurement.update(checkpoint_loaded=False, source_checkpoint_sha256=current["checkpoint"]["sha256"],
                                       execution_evidence_sha256=sha(probe_stage / "evidence.json"))
                else:
                    raw = backend.legacy_probe(probe_stage, isaaclab, {current["arm"]: current},
                        f"{tag}_s{seed}_e{epoch}_{probe.lower()}", binding(protocol, anchor,
                            applied=applied, faults=faults, checkpoint=current["checkpoint"]["sha256"]), probes=[probe])
                    item, ref = raw["report"]["probes"][probe], reference_probe["report"]["probes"][probe]
                    measurement = item["measurement"] or {}
                    outcome = C.L.classify(probe, measurement if item["exit"] == 0 else None,
                                           ref["measurement"] if ref["exit"] == 0 else None)
                    seconds = V._duration(raw)
                receipt = probe_ledger.receipt(active_probe, outcome, measurement, seconds, 0)
                receipt["probe"] = probe
                active_probe = None
                row["probe_steps"].append(receipt)
                result["costs"][stage] = seconds
                state = C.advance(state, probe, receipt["outcome"])
                write_new(epoch_folder / f"loop_{len(state['observed']):02d}.json", state)
            row["loop"] = state
            component = C.confirmed_component(state, applied)
            if component is None:
                return finish("HOLD", "unconfirmed_or_unregistered_component")
            row["confirmed_component"] = component
            confirming = next(x for x in row["probe_steps"]
                              if x["probe"] == ("P_reward" if component == "reward" else C.NEW_PROBE))
            correction_packet = C.correction_packet(state, confirming, current_report,
                current["checkpoint"]["sha256"], sha(current_report_path), applied)
            stage = f"epoch{epoch}_correction_model"
            model = MC.decide(epoch_folder / "model", "correction", correction_packet, api_root, caller=caller)
            row["model"]["correction"] = model
            review = model["review"]
            if review["status"] != "valid" or review["decision"]["decision"] != "propose" or not review["decision"]["agrees_with_rule"]:
                return finish("HOLD", "correction_not_accepted")
            cid = C.CORRECTIONS[component]
            next_applied = C.validate_applied([*applied, cid])
            verify(isaaclab, anchor)
            source_sha = current["checkpoint"]["sha256"]
            train_binding = binding(protocol, anchor, applied=next_applied, faults=faults, checkpoint=source_sha)
            previous_sha = prior["result_sha256"] if prior else None
            evidence = {"epoch": epoch, "input_checkpoint_sha256": source_sha, "applied_before": list(applied),
                        "previous_result_sha256": previous_sha, "binding_digest": C.L.digest(train_binding)}
            evidence["accepted_model"] = accepted_correction_evidence(model,
                epoch_folder / "model/correction.packet.json", api_root, correction_packet)
            entry = C.correction_registry()[component]
            request = correction_ledger.propose(C.L.digest(state), component, C.L.digest(row["probe_steps"]), cid,
                "restart", {"case": entry["protocol_case"], "arm": "corrected"}, CORRECTION_BUDGET_S, evidence=evidence)
            active_correction = request["request_id"]
            correction_ledger.approve(active_correction, approver)
            correction_ledger.consume(active_correction, C.L.digest(state))
            stage, arm = f"epoch{epoch}_training", "FIRST" if epoch == 1 else "BOTH"
            repaired = backend.train(epoch_folder / "training", isaaclab, arm, seed,
                                      f"{tag}_s{seed}_{arm.lower()}", train_binding)
            backend.validate_artifact(repaired, train_binding, arm, seed)
            write_new(epoch_folder / "artifact.json", repaired)
            result["costs"][stage] = train_seconds = V._duration(repaired)
            verify(isaaclab, anchor)
            stage = f"epoch{epoch}_evaluation"
            fixed = backend.fixed_evaluate(epoch_folder / "evaluation", isaaclab, {arm: repaired},
                                            f"{tag}_s{seed}_e{epoch}", train_binding)
            result["costs"][stage] = eval_seconds = V._duration(fixed)
            report = V.validated_reports(fixed, {arm: repaired})[arm]
            verdict = recovery(report, reports["NONE"], bands)
            bundle_path = epoch_folder / "repair_receipt.json"
            seal_repair(bundle_path, request_id=active_correction, epoch=epoch, previous=previous_sha,
                        applied_before=list(applied), applied_after=next_applied, source=current, artifact=repaired,
                        evaluation=fixed, reference=reports["NONE"], reference_path=reference_path,
                        train_binding=train_binding, seed=seed, outcome=verdict, state=state, probe_steps=row["probe_steps"],
                        decision=decision, probe_prereg=prereg, model=model, api_root=api_root,
                        source_report=current_report, source_report_path=current_report_path)
            receipt = correction_ledger.receipt(active_correction, tag, verdict["outcome"], sha(bundle_path),
                                                 train_seconds + eval_seconds, 0)
            active_correction = None
            row["receipt"] = receipt
            if receipt["verdict"] == "unknown":
                return finish("HOLD", "correction_receipt_unknown")
            checked = verify_repair(bundle_path, correction_ledger, expected_epoch=epoch,
                expected_applied=applied, expected_input_sha=source_sha, previous=previous_sha,
                reference=reports["NONE"], bands=bands, expected_binding=train_binding, seed=seed, backend=backend)
            row["verified_receipt"] = checked
            applied = next_applied
            result["unresolved_components"] = C.unresolved_components(applied)
            if checked["verdict"] == "recovered":
                return finish("recovered_after_one" if epoch == 1 else "recovered")
            if epoch == 2:
                return finish("not_recovered")
            previous_input, previous_binding, prior_path = source_sha, train_binding, bundle_path
            current, current_report, current_report_path = repaired, report, Path(fixed["report_paths"][arm])
        raise AssertionError("두 epoch 상한 이후 실행은 없다")
    except MC.BudgetStop:
        return finish("HOLD", "api_budget_or_pending")
    except Exception as error:
        if active_probe and probe_ledger and probe_ledger.requests()[active_probe]["state"] == "running":
            probe_ledger.receipt(active_probe, "unknown", {}, 0.0, 1, reason="execution_or_validation_failed")
        if active_correction and correction_ledger.requests()[active_correction]["state"] == "running":
            correction_ledger.receipt(active_correction, tag, "unknown", None, None, 1, reason="execution_or_validation_failed")
        result["failure"] = {"stage": stage, "type": type(error).__name__}
        finish("invalid")
        raise


def execute(tag: str, isaaclab: Path, approver: str, approve: bool, *, backend: Any = E, caller=None,
            seed_guard=assert_fresh_seeds) -> dict:
    if not TAG.fullmatch(tag) or not isinstance(approver, str) or not approver.strip() or not approve:
        raise ValueError("등록 tag·승인자·등록 행동 위임 플래그가 필요하다")
    protocol, anchor = verify(isaaclab)
    folder = PRIVATE / tag
    if folder.exists():
        raise FileExistsError("기존 tag를 다시 실행하지 않는다")
    paths = backend.preflight(isaaclab)
    fresh = seed_guard(SEEDS, isaaclab, repo=ROOT)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "seed_freshness.json", fresh)
    write_new(folder / "launch_paths.json", paths)
    write_new(folder / "begin.json", {"tag": tag, "seeds": list(SEEDS), "experiment_commit": anchor,
                                    "protocol_sha256_lf": sha(PROTOCOL, True), "approver": approver,
                                    "identity_attested": False})
    result = {"schema": "residual_demo_result_v1", "tag": tag, "seeds": list(SEEDS), "status": "invalid",
              "experiment_commit": anchor, "protocol_sha256_lf": sha(PROTOCOL, True), "cases": [],
              "design_coupling": protocol["design_coupling"]}
    api_root = folder / "api"
    try:
        for seed in SEEDS:
            fresh_case = seed_guard([seed], isaaclab, repo=ROOT)
            write_new(folder / f"fresh_seed_s{seed}.json", fresh_case)
            result["cases"].append(run_case(folder / f"s{seed}", tag, seed, isaaclab, protocol, anchor,
                approver, api_root, backend=backend, caller=caller))
            budget = MC.audit_budget(api_root)
            if budget["pending"] or budget["remaining"] == 0:
                break
        verify(isaaclab, anchor)
        result["api"] = MC.audit_budget(api_root)
        result["status"] = "complete" if len(result["cases"]) == 2 and result["api"]["pending"] == 0 else "HOLD"
        successes = sum(row["demo_success"] for row in result["cases"])
        result["demo_success"] = result["status"] == "complete" and successes == 2
        result["successes"] = successes
        result["wording"] = protocol["aggregate_wording"]["both" if result["demo_success"] else "mixed" if successes else "none"]
        write_new(folder / "result.json", result)
        return result
    except Exception as error:
        result["failure_type"] = type(error).__name__
        result["api"] = MC.audit_budget(api_root)
        write_new(folder / "result.json", result)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default="residual_v1_20261017")
    parser.add_argument("--approver", default="claude-main-user-delegated")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.isaaclab)
    elif args.command in ("check", "verify"):
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "ok", "anchor": anchor, "committed_gate": args.command == "verify",
                 "launch_paths": E.preflight(args.isaaclab)}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds(SEEDS, args.isaaclab, repo=ROOT)
    else:
        value = execute(args.tag, args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
