"""모델 선택을 검증하고 등록 규칙 전환을 구분하는 두 번째 진단 고리."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Any

import residual_demo_v1 as D1
import residual_selection_v2 as SELECT
import residual_model_call_v2 as MC

ROOT, C, E, V = D1.ROOT, D1.C, D1.E, D1.V
CorrectionLedger = D1.CorrectionLedger
assert_fresh_seeds = D1.assert_fresh_seeds
read, write_new, sha = D1.read, D1.write_new, D1.sha
recovery = D1.recovery
TASK, BANDS = D1.TASK, D1.BANDS
PROBE_BUDGET_S, CORRECTION_BUDGET_S = D1.PROBE_BUDGET_S, D1.CORRECTION_BUDGET_S
OUTCOMES = D1.OUTCOMES
PROTOCOL = ROOT / "bench/protocols/residual_demo_v2.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task31/runs"
SEEDS = (2120, 2121)
TAG = re.compile(r"residual_v2_[0-9]{8}[a-z0-9_]{0,8}")
NEW_FILES = {
    "evals/residual_selection_v2.py", "evals/residual_model_call_v2.py", "evals/residual_demo_v2.py",
    "bench/run_residual_demo_v2.ps1", "tests/test_residual_selection_v2.py",
    "tests/test_residual_model_call_v2.py", "tests/test_residual_demo_v2.py",
}
CODE = D1.CODE | NEW_FILES
EVIDENCE = set(D1.EVIDENCE) | {
    "evals/results/residual_failure_public_20261017/summary.json",
    "evals/results/residual_failure_public_20261017/model_contract_audit.json",
}


def fixed_protocol() -> dict:
    value = deepcopy(D1.fixed_protocol())
    value.update(schema="residual_demo_protocol_v2", seeds=list(SEEDS),
        design_coupling="설계자가 F1·R1 수리와 작업29 두 seed의 epoch2 모델 계약 위반을 본 뒤 만든 개발 측정이다. 새 seed는 이 결합을 없애지 않는다.",
        hypothesis="출력 계약과 현재 가설을 구분하는 검사 선택을 검증하고, 모델 무효일 때 등록 규칙의 다음 검사를 선택하면 두 번째 고리의 진행 여부를 확인할 수 있다.")
    value["selection"] = {
        "schema": SELECT.SCHEMA, "model_fields": ["action", "probe", "fact_ids"],
        "host_fields": ["candidate_groups", "fact_values", "expected_outcomes", "probe_args", "compatibility_state_input"],
        "format": "정확한 세 필드·등록 enum·고유 fact ID. 모델 수치·기대값·args·ranking 출력은 무효",
        "semantic": "남은 후보를 normal/abnormal로 나누는 미관측 검사, 또는 단일 후보의 등록 abnormal 확인 검사만 허용",
        "fallback": "최종 형식/의미 무효만 현재 후보·관측의 M._next로 전환; 규칙 선택도 같은 의미 검증 적용",
        "hold": "valid hold·완결된 인프라 오류 기록·예산·pending·unknown은 HOLD. 규칙으로 우회하지 않음",
        "invalid": "출처·packet·원장·receipt 결합 불일치는 invalid로 배치를 중단. 규칙으로 우회하지 않음",
        "correction": "규칙은 검사만 선택. 새 abnormal receipt·확정 상태·유효 모델 교정 제안·승인·일회 소비가 모두 필요",
        "provenance": "model/selection.json을 교정 inventory에 결합. 두 번째 epoch 전에 raw 응답·API receipt로 selection 재계산",
        "input_bindings": "case/reference checkpoint·telemetry SHA를 packet과 API 예약에 결합. 교정 receipt에 source/reference artifact를 봉인하고 epoch2 입력이 epoch1 출력인지 검증",
        "no_hidden_answer": "결함명·실제 remaining_faults·계측 readback을 모델 선택이나 규칙 순서 입력에 넣지 않음",
    }
    value["metrics"] = {
        "final_model_validity": "최종 valid diagnosis / valid 또는 invalid로 완료한 diagnosis; infra 제외 수 별도",
        "format_and_semantics": "형식 valid / 완료 diagnosis, 의미 valid / 형식 valid",
        "first_pass_validity": "대화 첫 응답 valid / 완료 diagnosis",
        "rule_transition": "registered_rule 선택 / 완료 diagnosis 및 invalid diagnosis 중 조건부 비율",
        "end_to_end_recovery": "두 고리·서로 다른 두 교정·최종 고정 평가 회복 / 계획한 새 seed 2개",
        "interpretation": "규칙으로 회복해도 모델 응답 유효로 재분류하지 않음. CPU 재생과 GPU 회복을 별도로 보고",
    }
    value["success_conditions"] = [*value["success_conditions"], "selection_origin_recomputed_from_model_receipts",
        "no_unapproved_or_unknown_correction", "model_validity_rule_transition_recovery_reported_separately"]
    value["cpu_replay"] = "NONE·단일 F1·단일 R1 epoch2 0, 양 교정 순서, 모델 형식/의미 무효 규칙 전환, 단일 후보 확인, HOLD·unknown·변조·예산 경계"
    return value


def dynamic(isaaclab: Path) -> dict:
    return {"source_sha256_lf": {p: sha(ROOT / p, True) for p in sorted(CODE)},
            "evidence_sha256_lf": {p: sha(ROOT / p, True) for p in sorted(EVIDENCE)},
            "runtime_source_sha256_lf": D1.dynamic(isaaclab)["runtime_source_sha256_lf"]}


def prepare(isaaclab: Path) -> dict:
    write_new(PROTOCOL, {**fixed_protocol(), **dynamic(isaaclab)})
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, anchor: str | None = None, *, committed: bool = True) -> tuple[dict, str]:
    value = read(PROTOCOL)
    if value != {**fixed_protocol(), **dynamic(isaaclab)}:
        raise ValueError("사전등록·소스·입력·런타임 결합 불일치")
    head = V.git("rev-parse", "HEAD")
    anchor = anchor or head
    if not isinstance(anchor, str) or not anchor:
        raise ValueError("실험 source commit을 확인하지 못했다")
    if committed:
        paths = sorted(CODE | EVIDENCE | {PROTOCOL.relative_to(ROOT).as_posix()})
        V.git("merge-base", "--is-ancestor", anchor, head)
        V.git("ls-files", "--error-unmatch", "--", *paths)
        if V.git("status", "--porcelain", "--", *paths):
            raise ValueError("결합 파일을 실행 전에 커밋해야 한다")
        for path in paths:
            if V.git("rev-parse", f"{anchor}:{path}") != V.git("rev-parse", f"{head}:{path}"):
                raise ValueError("실험 commit 뒤 결합 blob이 바뀌었다")
    return value, anchor


def binding(protocol: dict, anchor: str, **kwargs) -> dict:
    return {**D1.binding(protocol, anchor, **kwargs), "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256_lf": sha(PROTOCOL, True)}


def diagnosis_packet(current: dict, reference: dict, epoch: int, applied: list, prior: dict | None) -> dict:
    legacy = D1.diagnosis_packet(current, reference, epoch, applied, prior)
    return SELECT.build_packet(epoch=epoch, applied_corrections=applied, prior_receipt=prior,
        facts=legacy["facts"], budget_left=legacy["budget_left"], execution_budget_s=PROBE_BUDGET_S,
        input_bindings={"case": input_binding(current), "reference": input_binding(reference)})


def input_binding(artifact: dict) -> dict:
    result = {}
    for field in ("checkpoint", "telemetry"):
        actual = sha(Path(artifact[field]["path"]))
        if actual != artifact[field]["sha256"]:
            raise ValueError("진단 입력 checkpoint·telemetry 파일 SHA가 다르다")
        result[f"{field}_sha256"] = actual
    return result


def replay_selection(model_folder: Path, packet: dict, api_root: Path) -> dict:
    packet_path = model_folder / "diagnosis.packet.json"
    if read(packet_path) != packet or packet_path.read_bytes() != D1.MC._packet_bytes(packet):
        raise ValueError("검사 선택 입력이 현재 출처와 다르다")
    saved = read(model_folder / "selection.json")
    model = saved["model"]
    attempts = model["attempts"]
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise ValueError("검사 선택 모델 대화 횟수가 다르다")
    review = None
    for index, attempt in enumerate(attempts, 1):
        path = model_folder / f"diagnosis.dialogue{index}.json"
        record = read(path)
        review = MC._review_record(record, packet_path.read_bytes(), "diagnosis", api_root, record_path=path)
        expected = {"record": path.name, "record_sha256": sha(path), "review": review,
                    "request_receipts": record["request_receipts"]}
        if attempt != expected or read(model_folder / f"diagnosis.review{index}.json") != expected:
            raise ValueError("검사 선택 모델 review·API 결합이 다르다")
        if index < len(attempts) and review["status"] != "infra_error":
            raise ValueError("인프라 오류 외 추가 대화")
    if review is None or model["review"] != review or MC.audit_budget(api_root)["pending"]:
        raise ValueError("검사 선택 최종 모델 판정 또는 pending 원장이 다르다")
    expected = {"schema": "residual_selection_receipt_v2", "packet_sha256": sha(packet_path),
                "model": model, "selection": SELECT.select(review, packet)}
    if saved != expected:
        raise ValueError("모델 판정과 검사 선택 주체·전환 기록이 다르다")
    return saved["selection"]


def accepted_correction_evidence(model: dict, packet_path: Path, api_root: Path, expected: dict) -> dict:
    evidence = D1.accepted_correction_evidence(model, packet_path, api_root, expected)
    for index, attempt in enumerate(model["attempts"], 1):
        path = packet_path.parent / f"correction.dialogue{index}.json"
        review = MC._review_record(read(path), packet_path.read_bytes(), "correction", api_root, record_path=path)
        if review != attempt["review"]:
            raise ValueError("교정 모델의 요청 예약과 실제 호출 결합이 다르다")
    return evidence


def seal_repair(path: Path, *, request_id: str, epoch: int, previous: str | None,
                applied_before: list, applied_after: list, source: dict, artifact: dict,
                evaluation: dict, reference: dict, reference_path: Path,
                train_binding: dict, seed: int, outcome: dict, state: dict, probe_steps: list,
                decision: dict, probe_prereg: dict, model: dict, api_root: Path,
                source_report: dict, source_report_path: Path) -> dict:
    """v1 receipt 필드를 보존하고 진단 입력 artifact·telemetry 결합을 추가한다."""
    case_folder = path.parent.parent
    faults = train_binding["original_faults"]
    arm = "COMPOUND" if len(faults) == 2 else faults[0] if faults else "NONE"
    source_path = (case_folder / f"artifact_{arm.lower()}.json" if epoch == 1 else
                   case_folder / "epoch1/artifact.json")
    reference_artifact_path = case_folder / "artifact_none.json"
    reference_artifact = read(reference_artifact_path)
    if read(source_path) != source:
        raise ValueError("봉인할 진단 source artifact가 저장 원본과 다르다")
    paths = [source_path, reference_artifact_path, reference_path, source_report_path,
             path.parent / "probes.ledger.jsonl", *[Path(p) for p in evaluation["report_paths"].values()]]
    for item in (source, reference_artifact, artifact):
        paths.extend(Path(item[key]["path"]) for key in ("checkpoint", "telemetry"))
        paths.extend(Path(item[key]) for key in ("manifest_path", "readback_path", "records_path"))
    paths.extend(p for p in (path.parent / "model").rglob("*") if p.is_file())
    value = {"schema": "residual_repair_receipt_bundle_v1", "selection_contract": SELECT.SCHEMA,
        "request_id": request_id, "epoch": epoch, "previous_result_sha256": previous,
        "applied_before": applied_before, "applied_after": applied_after,
        "input_checkpoint_sha256": source["checkpoint"]["sha256"], "source": source,
        "reference_artifact": reference_artifact, "artifact": artifact,
        "evaluation": evaluation, "reference": reference, "reference_path": str(reference_path),
        "binding": train_binding, "seed": seed, "recovery": outcome, "state": state,
        "probe_steps": probe_steps, "initial_decision": decision, "probe_prereg": probe_prereg,
        "model_correction": model, "api_root": str(api_root.resolve()), "source_report": source_report,
        "source_report_path": str(source_report_path), "inventory_sha256": D1._files(paths)}
    write_new(path, value)
    return value


def verify_repair(path: Path, ledger: CorrectionLedger, **kwargs) -> dict:
    checked = D1.verify_repair(path, ledger, **kwargs)
    value = read(path)
    case_folder, epoch = path.parent.parent, value["epoch"]
    applied, prior = value["applied_before"], None
    if value.get("selection_contract") != SELECT.SCHEMA:
        raise ValueError("새 선택 계약의 진단 입력 결합이 없다")
    correction_packet = path.parent / "model/correction.packet.json"
    accepted_correction_evidence(value["model_correction"], correction_packet, Path(value["api_root"]),
                                 read(correction_packet))
    backend = kwargs.get("backend", E)
    external = kwargs["expected_binding"]
    faults = external["original_faults"]
    arm = "COMPOUND" if len(faults) == 2 else faults[0] if faults else "NONE"
    original = read(case_folder / f"artifact_{arm.lower()}.json")
    original_binding = {**external, "applied_corrections": [], "source_checkpoint_sha256": None}
    backend.validate_artifact(original, original_binding, arm, kwargs["seed"])
    reference_artifact = read(case_folder / "artifact_none.json")
    reference_binding = {**original_binding, "original_faults": []}
    backend.validate_artifact(reference_artifact, reference_binding, "NONE", kwargs["seed"])
    if (reference_artifact != value["reference_artifact"]
            or reference_artifact["checkpoint"]["sha256"] != kwargs["reference"]["checkpoint"]["sha256"]):
        raise ValueError("정상 기준 artifact·평가 결합 불일치")
    if epoch == 2:
        first_path = case_folder / "epoch1/repair_receipt.json"
        parent_binding = {**external, "applied_corrections": list(kwargs["expected_applied"]),
                          "source_checkpoint_sha256": original["checkpoint"]["sha256"]}
        prior = verify_repair(first_path, ledger, expected_epoch=1, expected_applied=[],
            expected_input_sha=original["checkpoint"]["sha256"], previous=None,
            reference=kwargs["reference"], bands=kwargs["bands"], expected_binding=parent_binding,
            seed=kwargs["seed"], backend=backend)
        if (prior["verdict"] != "not_recovered" or value["previous_result_sha256"] != prior["result_sha256"]
                or value["input_checkpoint_sha256"] != prior["checkpoint_sha256"]):
            raise ValueError("두 번째 교정의 앞선 비회복 receipt가 다르다")
        current = read(case_folder / "epoch1/artifact.json")
        backend.validate_artifact(current, parent_binding, "FIRST", kwargs["seed"])
        if current["checkpoint"]["sha256"] != prior["checkpoint_sha256"]:
            raise ValueError("두 번째 진단 입력은 첫 교정 출력 checkpoint여야 한다")
    else:
        current = original
    if current != value["source"] or current["checkpoint"]["sha256"] != value["input_checkpoint_sha256"]:
        raise ValueError("봉인한 진단 source artifact와 교정 입력 checkpoint가 다르다")
    packet = diagnosis_packet(current, reference_artifact, epoch, applied, prior)
    selection_path = path.parent / "model/selection.json"
    if str(selection_path.resolve()) not in value["inventory_sha256"]:
        raise ValueError("교정 inventory에 검사 선택 receipt가 없다")
    selection = replay_selection(path.parent / "model", packet, Path(value["api_root"]))
    if (selection["selected_by"] not in {"model", "registered_rule"}
            or selection["host_decision"] != value["initial_decision"]
            or not value["probe_steps"] or selection["probe"] != value["probe_steps"][0]["probe"]):
        raise ValueError("검사 선택 receipt와 실제 첫 검사·교정 초기 상태가 다르다")
    return checked


def decision_metrics(epochs: list) -> dict:
    rows = [row for row in epochs if "diagnosis" in row.get("model", {})]
    completed = [row for row in rows if row["model"]["diagnosis"]["review"]["status"] in {"valid", "invalid"}]
    reviews = [row["model"]["diagnosis"]["review"] for row in completed]
    count = len(completed)
    invalid = sum(r["status"] == "invalid" for r in reviews)
    rule = sum(row.get("selection", {}).get("selected_by") == "registered_rule" for row in completed)
    fmt = sum(r.get("format_status") == "valid" for r in reviews)
    def ratio(numerator: int, denominator: int) -> dict:
        return {"numerator": numerator, "denominator": denominator,
                "rate": numerator / denominator if denominator else None}
    return {"diagnosis_opportunities": len(rows), "completed_diagnoses": count,
        "infrastructure_or_call_errors": len(rows) - count,
        "model_validity": ratio(count - invalid, count),
        "format_validity": ratio(fmt, count),
        "semantic_validity_given_format": ratio(sum(r.get("semantic_status") == "valid" for r in reviews), fmt),
        "first_pass_validity": ratio(sum(r.get("first_status") == "valid" for r in reviews), count),
        "rule_transition": ratio(rule, count), "rule_transition_given_invalid": ratio(rule, invalid)}


def run_case(folder: Path, tag: str, seed: int, isaaclab: Path, protocol: dict, anchor: str,
             approver: str, api_root: Path, *, backend: Any = E, caller=None, original_faults=("F1", "R1")) -> dict:
    # 공개 execute는 복합 mask만 전달한다. 같은 고리의 CPU replay는 단일/NONE도 검증한다.
    faults = list(E.FAULTS.normalize_faults(original_faults))
    source_arm = "COMPOUND" if len(faults) == 2 else faults[0] if faults else "NONE"
    folder.mkdir(parents=True, exist_ok=False)
    result = {"schema": "residual_case_result_v2", "seed": seed, "outcome": "invalid", "epochs": [],
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
        result["decision_metrics"] = decision_metrics(result["epochs"])
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
            if MC.audit_budget(api_root)["pending"]:
                return finish("HOLD", "api_pending")
            selection = SELECT.select(review, packet)
            write_new(epoch_folder / "model/selection.json", {
                "schema": "residual_selection_receipt_v2",
                "packet_sha256": sha(epoch_folder / "model/diagnosis.packet.json"),
                "model": diagnosis, "selection": selection})
            selection = replay_selection(epoch_folder / "model", packet, api_root)
            row["selection"] = selection
            if selection["selected_by"] == "hold":
                return finish("HOLD", "selection_" + selection["reason"])
            decision = selection["host_decision"]
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
    paths = {**backend.preflight(isaaclab), "private_root": str(PRIVATE.resolve())}
    fresh = seed_guard(SEEDS, isaaclab, repo=ROOT)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "seed_freshness.json", fresh)
    write_new(folder / "launch_paths.json", paths)
    write_new(folder / "begin.json", {"tag": tag, "seeds": list(SEEDS), "experiment_commit": anchor,
                                    "protocol_sha256_lf": sha(PROTOCOL, True), "approver": approver,
                                    "identity_attested": False})
    result = {"schema": "residual_demo_result_v2", "tag": tag, "seeds": list(SEEDS), "status": "invalid",
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
        result["decision_metrics"] = decision_metrics([epoch for case in result["cases"] for epoch in case["epochs"]])
        result["end_to_end_recovery"] = {"numerator": successes, "denominator": len(SEEDS),
                                         "rate": successes / len(SEEDS)}
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
    parser.add_argument("--tag", default="residual_v2_20261017")
    parser.add_argument("--approver", default="claude-main-user-delegated")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.isaaclab)
    elif args.command in {"check", "verify"}:
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        result = {"status": "ok", "anchor": anchor, "committed_gate": args.command == "verify"}
    elif args.command == "fresh-seeds":
        result = assert_fresh_seeds(SEEDS, args.isaaclab, repo=ROOT)
    else:
        result = execute(args.tag, args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
