"""등록 F1의 Linux 고리. 모델 왕복마다 한 검사만 승인하고 진행한다."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import brev_recovery_contract_v1 as P
import preregistered_seed_guard as FRESH
import residual_demo_v2 as R
import selection_comparison_controller_v1 as C

POLICY = "model_adaptive"
APPROVER = "user_delegated_codex_main"


def valid_execution(value: dict) -> None:
    if (value.get("completed") is not True or value.get("exit_code") != 0
            or value.get("timed_out") is not False or value.get("cleanup_verified") is not True
            or value.get("gpu_before") != [] or value.get("gpu_after") != []
            or value.get("source_before") != value.get("source_after")
            or not C.V.number(value.get("wall_s")) or value["wall_s"] <= 0):
        raise ValueError("완결된 POSIX 실행 영수증이 아님")


def validate_training_semantics(artifact: dict, anchor: str) -> None:
    arm, seed = artifact["arm"], artifact["seed"]
    if arm not in {"NONE", "F1", "CORRECTED"} or seed not in {P.SEED, P.DRY_SEED}:
        raise ValueError("등록 밖 학습 arm 또는 seed")
    manifest = P.read(Path(artifact["manifest_path"]))
    if seed == P.DRY_SEED:
        if (arm != "NONE" or manifest != {"schema": "brev_recovery_dryrun_manifest_v1",
                "arm": "NONE", "seed": P.DRY_SEED, "iterations": 20}
                or artifact["readback"].get("remaining_faults") != []):
            raise ValueError("dry-run 학습 의미 불일치")
        return
    faults = [] if arm == "NONE" else ["F1"]
    corrections = [C.RC.REWARD] if arm == "CORRECTED" else []
    if (manifest.get("arm") != ("FIRST" if arm == "CORRECTED" else arm)
            or manifest.get("seed") != P.SEED or manifest.get("max_iterations") != 300
            or manifest.get("mode") != "train" or manifest.get("num_envs") != 4096
            or manifest.get("original_faults") != faults
            or manifest.get("applied_corrections") != corrections
            or manifest.get("source_commit_sha") != anchor
            or manifest.get("protocol_sha256_lf") != P.sha(P.PROTOCOL, True)):
        raise ValueError("학습 의미 또는 소스 계보 불일치")
    readback = P.read(Path(manifest["readback_path"]))
    if readback != artifact["readback"] or readback["remaining_faults"] != (["F1"] if arm == "F1" else []):
        raise ValueError("등록 교정 readback 결합")
    reward = readback["reward"]
    before, after_fault, after = (reward[key] for key in
                                 ("before_code_sha256", "after_fault_code_sha256", "after_code_sha256"))
    if ((before != after_fault) != (arm != "NONE")
            or (before != after) != (arm == "F1")
            or reward["fault_present_after"] is not (arm == "F1")):
        raise ValueError("보상 함수 복원 readback 불일치")


def verify_artifact(artifact: dict, anchor: str) -> None:
    valid_execution(artifact["execution"])
    for name in ("checkpoint", "telemetry"):
        item = artifact[name]
        if P.sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("정책 입력 아티팩트 SHA 불일치")
    if P.read(Path(artifact["telemetry"]["path"]))["summary"] != artifact["telemetry"]["summary"]:
        raise ValueError("텔레메트리 요약 결합")
    validate_training_semantics(artifact, anchor)


def classify(probe: str, case: dict, reference: dict) -> str:
    if probe != C.RC.NEW_PROBE:
        return C.L.classify(probe, case, reference)
    if reference.get("status") != "normal" or case.get("status") not in {"normal", "abnormal"}:
        return "unknown"
    return case["status"]


def stage_seal(folder: Path, stage: Path, value: dict) -> dict:
    valid_execution(value["execution"])
    result_path = stage / "backend_result.json"
    P.write_new(result_path, value)
    inventory = {path.relative_to(folder).as_posix(): P.sha(path)
                 for path in sorted(stage.rglob("*")) if path.is_file()}
    if not inventory:
        raise ValueError("실행 원본 파일이 없음")
    return {"result": deepcopy(value), "inventory": inventory, "result_digest": P.digest(value),
            "result_path": result_path.relative_to(folder).as_posix()}


def verify_seal(folder: Path, seal: dict) -> None:
    if P.digest(seal["result"]) != seal["result_digest"]:
        raise ValueError("실행 결과 봉인 불일치")
    valid_execution(seal["result"]["execution"])
    for relative, expected in seal["inventory"].items():
        path = (folder / relative).resolve()
        if folder.resolve() not in path.parents or P.sha(path) != expected:
            raise ValueError("실행 원본 봉인 불일치")
    if P.read(folder / seal["result_path"]) != seal["result"]:
        raise ValueError("실행 결과와 저장 원본 불일치")


def load_state(folder: Path) -> dict | None:
    files = sorted((folder / "states").glob("*.json"))
    state = None
    previous = None
    for index, path in enumerate(files):
        row = P.read(path)
        if (path.name != f"{index:04d}.json" or row["index"] != index
                or row["previous_sha256"] != previous):
            raise ValueError("상태 순서 또는 CAS 결합 불일치")
        if index > 0:
            if state is None:
                raise ValueError("이전 상태 없는 전이")
            intent = P.read(folder / "intents" / f"{index - 1:04d}.json")
            if intent["state_sha256"] != previous or intent["phase"] != state["phase"]:
                raise ValueError("실행 intent와 상태 전이 결합")
        state, previous = row, P.sha(path)
    if state is not None:
        for seal in state["seals"]:
            verify_seal(folder, seal)
        for artifact in state["artifacts"].values():
            verify_artifact(artifact, state["anchor"])
        if "source_reports" in state:
            source_seals = [seal for seal in state["seals"]
                            if seal["result_path"] == "source_evaluation/backend_result.json"]
            if len(source_seals) != 1 or state["source_reports"] != source_seals[0]["result"]["reports"]:
                raise ValueError("정상 참조 고정 평가가 봉인 원본과 다름")
            bands = P.read(P.ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json")["bands"]
            if state["source_gate"] != R.recovery(state["source_reports"]["F1"], state["source_reports"]["NONE"], bands):
                raise ValueError("초기 실패 판정 재계산 불일치")
        if state["phase"] == "done":
            verify_terminal(folder, state)
    return state


def save_state(folder: Path, state: dict) -> dict:
    files = sorted((folder / "states").glob("*.json"))
    index = len(files)
    value = {**deepcopy(state), "index": index,
             "previous_sha256": P.sha(files[-1]) if files else None}
    P.write_new(folder / "states" / f"{index:04d}.json", value)
    if value["phase"] == "done":
        P.write_new(folder / "result.json", {**value, "final_state_sha256":
                    P.sha(folder / "states" / f"{index:04d}.json")})
    return value


def packet(state: dict) -> dict:
    facts: dict = {"config:action_scale": 0.25}
    for arm, role in (("F1", "case"), ("NONE", "reference")):
        for name, row in state["artifacts"][arm]["telemetry"]["summary"].items():
            key = f"telemetry:{role}:{name}"
            if key in C.V.BASE_FACT_IDS and isinstance(row, dict):
                value = row.get("mean_last_20pct")
                if value is None or C.V.number(value):
                    facts[key] = value
    bindings = {role: {f"{kind}_sha256": state["artifacts"][arm][kind]["sha256"]
                       for kind in ("checkpoint", "telemetry")}
                for arm, role in (("F1", "case"), ("NONE", "reference"))}
    return C.V.S.build_packet(epoch=1, applied_corrections=[], prior_receipt=None,
                             facts=facts, budget_left=7, execution_budget_s=1800,
                             input_bindings=bindings)


def snapshot(folder: Path, state: dict) -> dict:
    return C.V.build_snapshot(packet(state), P.TAG, POLICY, C.V.file_digest(C.V.PROTOCOL),
                              folder / "probes.ledger.jsonl")


def verify_history(folder: Path, state: dict) -> None:
    from brev_recovery_backend_v1 import validate_probe_result

    ledger = C.S.M.RegistryLedger(folder / "probes.ledger.jsonl", C.RC.registry())
    requests = list(ledger.requests().values())
    if len(requests) != len(state["probes"]):
        raise ValueError("프로브 원장과 측정 개수 불일치")
    for request, step in zip(requests, state["probes"], strict=True):
        for key in ("reference", "case"):
            verify_seal(folder, step[key])
        values = {role: validate_probe_result(state["artifacts"][arm], step["probe"],
                                              step["run_names"][role], step[role]["result"])
                  for role, arm in (("case", "F1"), ("reference", "NONE"))}
        outcome = classify(step["probe"], values["case"], values["reference"])
        measured = (values["case"] if step["probe"] != C.RC.NEW_PROBE else
                    {"case": values["case"], "control": values["reference"]})
        if (request["state"] != "done" or request["probe"] != step["probe"]
                or request["receipt"]["outcome"] != outcome
                or request["receipt"]["measurement"] != measured):
            raise ValueError("프로브 원장의 판정 재계산 불일치")


def _seal_append(folder: Path, stage: Path, value: dict, state: dict) -> dict:
    seal = stage_seal(folder, stage, value)
    state["seals"].append(seal)
    return seal


def validate_report_set(evaluation: dict, artifacts: dict) -> dict:
    valid_execution(evaluation["execution"])
    if set(evaluation["reports"]) != set(artifacts):
        raise ValueError("고정 평가 arm 불일치")
    result = {}
    for arm, report in evaluation["reports"].items():
        expected = {"name": report.get("name"), "checkpoint_file": Path(artifacts[arm]["checkpoint"]["path"]).name,
                    "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"], "action_scale": 0.25,
                    "protocol_sha256_lf": C.FIXED_EVAL_PROTOCOL_SHA256}
        result[arm] = C.G.validate_fixed_report(report, expected)
    return result


def _reports(evaluation: dict, artifacts: dict, folder: Path) -> dict:
    result = validate_report_set(evaluation, artifacts)
    for arm, report in result.items():
        P.write_new(folder / f"checked_{arm.lower()}.json", report)
    return result


def verify_terminal(folder: Path, state: dict) -> None:
    value = P.read(folder / "result.json")
    if (value.pop("final_state_sha256") != P.sha(folder / "states" / f"{state['index']:04d}.json")
            or value != state):
        raise ValueError("완료 결과와 최종 상태 결합 불일치")
    verify_history(folder, state)
    probes = C.S.M.RegistryLedger(folder / "probes.ledger.jsonl", C.RC.registry()).requests()
    ledger = C.D1.CorrectionLedger(folder / "corrections.ledger.jsonl", C.RC.correction_registry())
    requests = ledger.requests()
    if any(row["state"] != "done" for row in [*probes.values(), *requests.values()]):
        raise ValueError("완료 뒤 미완결 요청")
    if state["demo_success"] != (state["outcome"] == "recovered" and not state["pending_ledgers"]):
        raise ValueError("완료 성공 플래그 불일치")
    if not requests:
        if state["outcome"] in {"recovered", "not_recovered"} or "repair" in state or "CORRECTED" in state["artifacts"]:
            raise ValueError("승인 교정 없는 회복 주장")
        return
    if len(requests) != 1 or [row["event"] for row in ledger.events()] != ["proposed", "approved", "consumed", "receipt"]:
        raise ValueError("등록 교정 횟수 또는 승인 생명주기")
    request = next(iter(requests.values()))
    path = Path(state["repair"]["path"])
    if folder.resolve() not in path.resolve().parents or P.sha(path) != state["repair"]["sha256"]:
        raise ValueError("교정 원본 파일 SHA 불일치")
    repair = P.read(path)
    seals = [seal for seal in state["seals"] if seal["result_path"] == "corrected_evaluation/backend_result.json"]
    if len(seals) != 1 or repair["evaluation"] != seals[0]["result"]:
        raise ValueError("교정 후 평가 봉인 불일치")
    corrected = state["artifacts"]["CORRECTED"]
    evaluated = validate_report_set(seals[0]["result"], {"CORRECTED": corrected})
    measured = R.recovery(evaluated["CORRECTED"], state["source_reports"]["NONE"],
                          P.read(P.ROOT / "evals/results/p0c_fixed_eval_20261009/summary.json")["bands"])
    snap = snapshot(folder, state)
    receipt = request["receipt"]
    expected_outcome = "HOLD" if measured["outcome"] == "unknown" else measured["outcome"]
    if (C.registered_correction(snap, []) != ("reward", C.RC.REWARD)
            or request["correction_id"] != C.RC.REWARD or request["recovery_mode"] != "restart"
            or request["loop_state_digest"] != P.digest(snap)
            or request["receipts_digest"] != P.digest(state["probes"])
            or request["confirmed_mechanism"] != "reward"
            or repair["correction_id"] != C.RC.REWARD or repair["recovery_mode"] != "restart"
            or repair["input_checkpoint_sha256"] != state["artifacts"]["F1"]["checkpoint"]["sha256"]
            or repair["artifact"] != corrected or repair["snapshot_sha256"] != P.digest(snap)
            or repair["recovery"] != measured or state["outcome"] != expected_outcome
            or receipt["result_sha"] != P.sha(path) or receipt["verdict"] != measured["outcome"]
            or receipt["exit_code"] != 0 or receipt["over_budget"] is not False
            or receipt["gpu_s"] != corrected["execution"]["wall_s"] + repair["evaluation"]["execution"]["wall_s"]):
        raise ValueError("교정 승인·계보·후속 평가 재계산 불일치")


def verify_model_audit(response: dict, envelope: dict, state: dict, dialogue: dict | None) -> None:
    audit = response["api_audit"]
    rows = audit["rows"]
    previous = state.get("api_reserved", 0)
    if (audit["pending"] != 0 or audit["reserved"] != audit["completed"]
            or audit["reserved"] != len(rows) or not 0 <= len(rows) <= 20
            or len(rows) - previous != len(response["safe_results"])
            or P.digest(rows[:previous]) != state.get("api_rows_digest", P.digest([]))):
        raise ValueError("모델 HTTP 소비 원장 순서 또는 상한")
    calls = dialogue["engine"]["calls"] if dialogue else []
    if len(calls) != len(response["safe_results"]):
        raise ValueError("모델 응답 소비 수")
    for row, call in zip(rows[previous:], calls, strict=True):
        request = row["request"]
        if (request["snapshot_sha256"] != envelope["snapshot_sha256"]
                or request["anchor"] != envelope["anchor"]
                or request["protocol_sha256_lf"] != envelope["protocol_sha256_lf"]
                or request["messages_digest"] != call["messages_digest"]
                or row["result"] != call["result"]):
            raise ValueError("실제 HTTP 원장과 모델 대화 결합")


def replay_budget_stop(snap: dict, rows: list) -> dict:
    inputs = C.dialogue_input(snap)
    cursor = iter(deepcopy(rows))

    def bounded_replay(messages, params):
        try:
            return next(cursor)
        except StopIteration:
            raise C.CALL.MC.BudgetStop("사전등록 API 상한 재생") from None

    engine = C.CALL.engine(snap, inputs["view"], inputs["messages"], "production", bounded_replay)
    if engine["stop_reason"] != "budget_exhausted" or len(engine["calls"]) != len(rows):
        raise ValueError("API 상한 판단의 재생 불일치")
    return {"engine": engine}


def _terminal(folder: Path, state: dict, outcome: str, reason: str | None = None) -> dict:
    state.update(phase="done", outcome=outcome, reason=reason)
    state["pending_ledgers"] = [key for name, registry in (
        ("probes", C.RC.registry()), ("corrections", C.RC.correction_registry()))
        for key, row in (C.S.M.RegistryLedger(folder / f"{name}.ledger.jsonl", registry).requests()
                         if name == "probes" else C.D1.CorrectionLedger(
                             folder / f"{name}.ledger.jsonl", registry).requests()).items()
        if row["state"] in {"pending", "approved", "running"}]
    if state["pending_ledgers"]:
        state.update(outcome="invalid", reason="pending_ledger")
    state["demo_success"] = state["outcome"] == "recovered" and not state["pending_ledgers"]
    return state


def validate_freshness(value: dict) -> None:
    keys = {"schema", "status", "seeds", "checked_at_utc", "checked_roots",
            "checked_file_count", "evidence", "scope"}
    if (set(value) != keys or value["schema"] != "preregistered_seed_freshness_v1"
            or value["status"] != "fresh" or value["seeds"] != [P.SEED, P.DRY_SEED]
            or value["evidence"] != [] or type(value["checked_file_count"]) is not int
            or value["checked_file_count"] < 0 or len(value["checked_roots"]) != 5
            or "<isaaclab>/logs" not in value["checked_roots"]):
        raise ValueError("seed guard 영수증 계약")


def failed_cost(completed: list[dict], error: BaseException) -> tuple[float | None, float]:
    known = sum(row["wall_s"] for row in completed)
    failure = getattr(error, "receipt", None)
    if isinstance(failure, dict) and C.V.number(failure.get("wall_s")) and failure["wall_s"] >= 0:
        return known + failure["wall_s"], known + failure["wall_s"]
    return None, known


def advance(folder: Path, backend, anchor: str, *, response: dict | None = None,
            freshness: dict | None = None) -> dict:
    """호출자는 run별 배타 잠금을 보유한다. 각 transition은 새 파일에만 기록한다."""
    protocol = P.verify(anchor)
    state = load_state(folder)
    if state is None:
        if response is not None or freshness is None or freshness.get("status") != "fresh":
            raise ValueError("새 실행의 seed 확인 근거 없음")
        validate_freshness(freshness)
        cloud_freshness = FRESH.assert_fresh_seeds([P.SEED, P.DRY_SEED], backend.lab)
        validate_freshness(cloud_freshness)
        P.write_new(folder / "freshness.local.json", freshness)
        P.write_new(folder / "freshness.cloud.json", cloud_freshness)
        state = {"schema": "brev_recovery_state_v1", "anchor": anchor,
                 "protocol_sha256_lf": P.sha(P.PROTOCOL, True), "tag": P.TAG,
                 "phase": "dry", "artifacts": {}, "seals": [], "probes": [],
                 "freshness": {"local": freshness, "cloud": cloud_freshness,
                               "local_sha256": P.sha(folder / "freshness.local.json"),
                               "cloud_sha256": P.sha(folder / "freshness.cloud.json")}, "source_gate": None}
        return save_state(folder, state)
    if state["anchor"] != anchor or state["protocol_sha256_lf"] != P.sha(P.PROTOCOL, True):
        raise ValueError("실행 중 소스 또는 사전등록 변경")
    if state["phase"] == "done":
        if response is not None:
            raise ValueError("종료 뒤 모델 응답")
        return state
    if state["phase"] == "needs_model" and response is None:
        return state
    verify_history(folder, state)
    index = state["index"]
    # 이전 transition 도중 중단되면 같은 실행을 다시 발사하지 않는다.
    P.write_new(folder / "intents" / f"{index:04d}.json",
                {"state_sha256": P.sha(folder / "states" / f"{index:04d}.json"),
                 "phase": state["phase"], "response_sha256": P.digest(response) if response else None})
    phase = state["phase"]
    if response is not None and phase != "needs_model":
        raise ValueError("등록되지 않은 단계의 모델 응답")
    if phase in {"dry", "NONE", "F1"}:
        dry = phase == "dry"
        arm, seed = ("NONE", P.DRY_SEED) if dry else (phase, P.SEED)
        stage = folder / "training" / phase.lower()
        artifact = backend.train(stage, arm, seed, f"brec_s{seed}_{phase.lower()}",
                                 iterations=20 if dry else 300)
        verify_artifact(artifact, anchor)
        _seal_append(folder, stage, artifact, state)
        if not dry:
            state["artifacts"][arm] = artifact
        state["phase"] = {"dry": "NONE", "NONE": "F1", "F1": "source_evaluation"}[phase]
    elif phase == "source_evaluation":
        stage = folder / "source_evaluation"
        evaluation = backend.evaluate(stage, state["artifacts"], "brec_source")
        _seal_append(folder, stage, evaluation, state)
        reports = _reports(evaluation, state["artifacts"], stage)
        state["source_reports"] = reports
        gate = R.recovery(reports["F1"], reports["NONE"], P.read(P.ROOT / protocol["evaluation"]["bands"])["bands"])
        state["source_gate"] = gate
        if not R.D1.V.R1.U.walks(reports["NONE"]):
            _terminal(folder, state, "baseline_invalid")
        elif gate["outcome"] == "recovered":
            _terminal(folder, state, "source_failure_absent")
        elif gate["outcome"] != "not_recovered":
            _terminal(folder, state, "HOLD", "source_evaluation_unknown")
        else:
            state["phase"] = "choose"
    elif phase == "choose":
        snap = snapshot(folder, state)
        if snap["view"]["status"] == "confirmed":
            component, correction = C.registered_correction(snap, [])
            if component != "reward" or correction != C.RC.REWARD:
                _terminal(folder, state, "HOLD", "correction_scope_insufficient")
            else:
                state["phase"] = "repair"
        elif snap["view"]["status"] != "open":
            _terminal(folder, state, "HOLD", "diagnosis_" + snap["view"]["status"])
        else:
            ordinal = len(state["probes"]) + 1
            if ordinal > protocol["selection"]["max_probes"]:
                raise ValueError("검사 상한 초과")
            envelope = {"schema": "brev_recovery_model_request_v1", "anchor": anchor,
                        "protocol_sha256_lf": state["protocol_sha256_lf"], "tag": P.TAG,
                        "ordinal": ordinal, "snapshot": snap, "snapshot_sha256": P.digest(snap),
                        "messages_digest": C.dialogue_input(snap)["messages_digest"],
                        "ledger_events": C.S.M.RegistryLedger(folder / "probes.ledger.jsonl", C.RC.registry()).events()}
            path = folder / "models" / f"{ordinal:02d}" / "request.json"
            P.write_new(path, envelope)
            state.update(phase="needs_model", model_request=str(path), model_request_sha256=P.sha(path))
    elif phase == "needs_model":
        if response is None:
            # 이미 봉인한 모델 입력을 재제시할 때 새 실행 intent를 남기지 않는다.
            raise ValueError("모델 응답 없음")
        envelope = P.read(Path(state["model_request"]))
        snap = snapshot(folder, state)
        if P.sha(Path(state["model_request"])) != state["model_request_sha256"]:
            raise ValueError("모델 요청 원본 변경")
        if (response.get("schema") != "brev_recovery_model_response_v1"
                or response.get("anchor") != anchor
                or response.get("protocol_sha256_lf") != state["protocol_sha256_lf"]
                or response.get("snapshot_sha256") != P.digest(snap)
                or response.get("messages_digest") != C.dialogue_input(snap)["messages_digest"]
                or response.get("api_audit", {}).get("pending") != 0):
            raise ValueError("모델 응답 실행·view·원장 결합")
        if envelope["snapshot"] != snap:
            raise ValueError("응답 전 snapshot이 변경됨")
        P.write_new(Path(state["model_request"]).with_name("accepted_response.json"), response)
        if response.get("stop_reason") == "budget_exhausted":
            stopped = replay_budget_stop(snap, response["safe_results"])
            verify_model_audit(response, envelope, state, stopped)
            if response["api_audit"]["reserved"] != 20 or stopped["engine"] != response["local_engine"]:
                raise ValueError("응답 없는 모델 대화")
            state.update(api_reserved=response["api_audit"]["reserved"],
                         api_rows_digest=P.digest(response["api_audit"]["rows"]))
            _terminal(folder, state, "HOLD", "api_budget_exhausted")
            return save_state(folder, state)
        dialogue = C.replay_dialogue(snap, "production", response["safe_results"])
        if response["local_engine"] != dialogue["engine"]:
            raise ValueError("WSL과 cloud의 모델 대화 재계산 불일치")
        verify_model_audit(response, envelope, state, dialogue)
        state.update(api_reserved=response["api_audit"]["reserved"],
                     api_rows_digest=P.digest(response["api_audit"]["rows"]))
        choice = C.choose(snap, POLICY, dialogue)
        if choice["action"] != "continue":
            _terminal(folder, state, "HOLD", choice["reason"])
        else:
            spec = C.request_spec(snap, choice)
            ledger = C.S.M.RegistryLedger(folder / "probes.ledger.jsonl", C.RC.registry())
            request = ledger.propose(snap["binding"], choice["probe"], spec["budget_gpu_s"])
            rid = request["request_id"]
            ledger.approve(rid, APPROVER)
            ledger.consume(rid, snap["binding"], choice["probe"])
            probe, ordinal = choice["probe"], len(state["probes"]) + 1
            roots = {role: folder / "probe_runs" / f"{ordinal:02d}" / role for role in ("reference", "case")}
            completed = []
            try:
                control = backend.probe(roots["reference"], state["artifacts"]["NONE"], probe, f"brec_p{ordinal}_ref")
                completed.append(control["execution"])
                case = backend.probe(roots["case"], state["artifacts"]["F1"], probe, f"brec_p{ordinal}_case")
                completed.append(case["execution"])
                reference_seal = _seal_append(folder, roots["reference"], control, state)
                case_seal = _seal_append(folder, roots["case"], case, state)
                outcome = classify(probe, case["measurement"], control["measurement"])
                measured = (case["measurement"] if probe != C.RC.NEW_PROBE else
                            {"case": case["measurement"], "control": control["measurement"]})
                wall = control["execution"]["wall_s"] + case["execution"]["wall_s"]
                ledger.receipt(rid, outcome, measured, wall, 0)
                state["probes"].append({"probe": probe, "selected_by": choice["selected_by"],
                                        "outcome": outcome, "case": case_seal, "reference": reference_seal,
                                        "run_names": {"reference": f"brec_p{ordinal}_ref", "case": f"brec_p{ordinal}_case"},
                                        "model_response_sha256": P.digest(response)})
                state["phase"] = "choose"
            except BaseException as error:
                total, known = failed_cost(completed, error)
                P.write_new(folder / "failed_probe_cost.json", {"parent_wall_s": total,
                            "known_parent_wall_s": known, "error_type": type(error).__name__})
                if total is None:
                    request = ledger.requests()[rid]
                    ledger._append({"event": "receipt", "request_id": rid, "outcome": "unknown",
                        "measurement": {}, "gpu_s": None, "exit_code": None, "over_budget": False,
                        "registry_digest": ledger.registry_digest, "args_digest": request["args_digest"],
                        "effect_key": request["effect_key"], "reason": "probe_execution_cost_unknown"})
                else:
                    ledger.receipt(rid, "unknown", {}, total, None, reason="probe_execution_failed")
                raise
    elif phase == "repair":
        snap = snapshot(folder, state)
        component, correction = C.registered_correction(snap, [])
        if component != "reward" or correction != C.RC.REWARD:
            raise ValueError("등록 보상 교정 밖 실행")
        registry = C.RC.correction_registry()
        ledger = C.D1.CorrectionLedger(folder / "corrections.ledger.jsonl", registry)
        state_digest = P.digest(snap)
        request = ledger.propose(state_digest, component, P.digest(state["probes"]), correction,
                                 "restart", {"case": registry[component]["protocol_case"], "arm": "corrected"},
                                 3600, evidence={"snapshot_sha256": state_digest,
                                                 "source_checkpoint_sha256": state["artifacts"]["F1"]["checkpoint"]["sha256"]})
        rid = request["request_id"]
        ledger.approve(rid, APPROVER)
        ledger.consume(rid, state_digest)
        completed = []
        try:
            stage = folder / "training" / "corrected"
            corrected = backend.train(stage, "CORRECTED", P.SEED, "brec_corrected", corrections=[correction],
                                      source_checkpoint_sha256=state["artifacts"]["F1"]["checkpoint"]["sha256"])
            completed.append(corrected["execution"])
            verify_artifact(corrected, anchor)
            manifest = P.read(Path(corrected["manifest_path"]))
            if manifest["source_checkpoint_sha256"] != state["artifacts"]["F1"]["checkpoint"]["sha256"]:
                raise ValueError("교정 restart의 원본 checkpoint 계보 불일치")
            _seal_append(folder, stage, corrected, state)
            state["artifacts"]["CORRECTED"] = corrected
            eval_stage = folder / "corrected_evaluation"
            evaluation = backend.evaluate(eval_stage, {"CORRECTED": corrected}, "brec_corrected")
            completed.append(evaluation["execution"])
            _seal_append(folder, eval_stage, evaluation, state)
            reports = _reports(evaluation, {"CORRECTED": corrected}, eval_stage)
            recovered = R.recovery(reports["CORRECTED"], state["source_reports"]["NONE"],
                                    P.read(P.ROOT / protocol["evaluation"]["bands"])["bands"])
            repair = {"correction_id": correction, "recovery_mode": "restart", "input_checkpoint_sha256":
                      state["artifacts"]["F1"]["checkpoint"]["sha256"], "artifact": corrected,
                      "evaluation": evaluation, "recovery": recovered, "snapshot_sha256": state_digest}
            repair_path = folder / "repair.json"
            P.write_new(repair_path, repair)
            wall = corrected["execution"]["wall_s"] + evaluation["execution"]["wall_s"]
            ledger.receipt(rid, P.TAG, recovered["outcome"], P.sha(repair_path), wall, 0)
            state["repair"] = {"path": str(repair_path), "sha256": P.sha(repair_path), "outcome": recovered["outcome"]}
            _terminal(folder, state, "HOLD" if recovered["outcome"] == "unknown" else recovered["outcome"],
                      "corrected_evaluation_unknown" if recovered["outcome"] == "unknown" else None)
        except BaseException as error:
            if ledger.requests()[rid]["state"] == "running":
                total, known = failed_cost(completed, error)
                P.write_new(folder / "failed_repair_cost.json", {"parent_wall_s": total,
                            "known_parent_wall_s": known, "error_type": type(error).__name__})
                ledger.receipt(rid, P.TAG, "unknown", None, total, None, reason="repair_execution_failed")
            raise
    else:
        raise ValueError("알 수 없는 상태")
    verify_history(folder, state)
    return save_state(folder, state)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--freshness", type=Path)
    args = parser.parse_args()
    import fcntl

    from brev_recovery_backend_v1 import Backend
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "controller.lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protocol = P.verify(args.anchor)
            backend = Backend(args.lab, args.python, {**protocol, "protocol_path": str(P.PROTOCOL),
                              "protocol_sha256_lf": P.protocol_sha_at(args.anchor)}, args.anchor)
            value = advance(args.run, backend, args.anchor,
                            response=P.read(args.response) if args.response else None,
                            freshness=P.read(args.freshness) if args.freshness else None)
            print(json.dumps({"phase": value["phase"], "index": value["index"],
                              "outcome": value.get("outcome"), "model_request": value.get("model_request")}))
            return 0
        except Exception as error:  # noqa: BLE001 - CLI 경계에는 종류만 남긴다.
            P.write_new(args.run / "failure.json", {"status": "invalid", "error_type": type(error).__name__})
            print(json.dumps({"status": "invalid", "error_type": type(error).__name__}), file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
