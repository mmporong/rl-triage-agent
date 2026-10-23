"""네 검사 선택 정책의 CPU 상태 전이·승인·교정 계보 계약.

이 판의 실행 검증은 합성 fixture만 다룬다. 실제 API, GPU, 학습을 시작하지 않으며 선택기 효과나 회복 효과를
측정하지 않는다. D4가 실제 실행을 연결할 때 쓰는 fail-closed backend adapter 경계도 정의한다. D2 선택 engine을
응답 iterator로 재생하고, arm별 probe/correction 원장과 raw 측정·고정 평가 파일을 다시 읽는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]
import residual_demo_v1 as D1  # noqa: E402
import residual_execution_v1 as BACKEND  # noqa: E402
import selection_comparison_call_v2 as CALL  # noqa: E402
import selection_comparison_gpu_v1 as GPU_CHECK  # noqa: E402

from rl_triage import grounded_recovery as G  # noqa: E402

V, S, RC, L = CALL.V, CALL.V.S, CALL.V.S.C1, CALL.V.L
POLICIES = V.POLICIES
PROTOCOL = ROOT / "bench/protocols/selection_comparison_controller_v1.json"
SOURCE_FILES = (
    "evals/selection_comparison_controller_v1.py",
    "tests/test_selection_comparison_controller_v1.py",
    "evals/selection_comparison_call_v2.py",
    "evals/selection_comparison_view_v1.py",
    "evals/residual_selection_v2.py",
    "evals/residual_contract_v1.py",
    "evals/residual_demo_v1.py",
    "evals/residual_execution_v1.py",
    "evals/selection_comparison_gpu_v1.py",
    "tests/test_selection_comparison_gpu_v1.py",
    "evals/export_ppo_ratio_repair_v3.py",
    "evals/ppo_ratio_repair_execution.py",
    "evals/ppo_ratio_repair_execution_v2.py",
    "evals/ppo_ratio_repair_execution_v3.py",
    "evals/r1_repair_execution.py",
    "bench/residual_training_v1.py",
    "bench/protocols/fixed_eval_v1.json",
    "bench/record_evidence_media.py",
    "src/rl_triage/correction_ledger.py",
    "src/rl_triage/mechanism_registry.py",
    "src/rl_triage/probe_loop.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
)
CORE_DIALOGUE_FIELDS = (
    "schema", "condition", "calls", "requests", "selection", "stop_reason", "feedback_used",
    "infra_retry_used", "costs", "used_for_decision", "approval_granted",
)
FIXTURE_EVIDENCE = "synthetic_cpu_fixture_v1"
ACTUAL_EVIDENCE = "actual_backend_v1"
GPU_CONTRACTS = {"residual_v1", "apps_bounded_v1"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def digest_file(path: Path) -> str:
    """실행 artifact와 evidence의 원본 byte SHA-256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_lf_file(path: Path) -> str:
    """source와 protocol 결합에만 쓰는 LF 정규화 SHA-256."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


FIXED_EVAL_PROTOCOL_SHA256 = digest_lf_file(BACKEND.FIXED_EVAL_PROTOCOL)


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def dialogue_input(snapshot: dict, condition: str = "production") -> dict:
    """D2와 같은 view, system prompt, canonical user body, params를 만드는 순수 입력 함수."""
    checked = V.checked(snapshot)
    view = CALL.query_view(checked, condition)
    extra = ("\nNon-executable observation query: candidates and eligible_probes are held at the state before "
             "the latest outcome. Use the exposed evidence to choose hold or the next probe. This query never executes."
             if condition != "production" else "")
    messages = [{"role": "system", "content": V.SYSTEM + extra},
                {"role": "user", "content": V.canonical(view).decode()}]
    policy = checked["inputs"]["policy"]
    needs_model = condition != "production" or (
        view["status"] == "open" and view["budget_left"] > 0
        and (policy == "model_adaptive" or (policy == "model_first_then_rule" and not view["observations"]))
    )
    return {"snapshot_digest": L.digest(checked), "view": view, "view_digest": L.digest(view),
            "messages": messages, "messages_digest": L.digest(messages),
            "params": CALL.F.request_params(CALL.MODEL), "needs_model": needs_model,
            "condition": condition}


def replay_dialogue(snapshot: dict, condition: str, safe_results: list[dict]) -> dict:
    """저장 가능한 provider 결과만 D2 engine의 읽기 전용 iterator로 재생한다."""
    inputs = dialogue_input(snapshot, condition)
    require(inputs["needs_model"], "규칙·닫힌·pending 상태에는 모델 대화를 만들지 않는다")
    require(isinstance(safe_results, list) and 1 <= len(safe_results) <= 3, "대화 응답 수 범위")
    rows = [CALL.safe_result(row) for row in safe_results]
    fixture = CALL.FixtureResponses(rows)
    record = CALL.engine(snapshot, inputs["view"], inputs["messages"], condition, fixture)
    require(len(fixture.sent) == record["requests"], "응답 iterator 소비 수 불일치")
    return {"schema": "selection_controller_dialogue_v1", "input_digest": L.digest(inputs),
            "safe_results": rows, "engine": record}


def verify_dialogue(snapshot: dict, condition: str, record: dict) -> dict:
    require(isinstance(record, dict) and set(record) == {"schema", "input_digest", "safe_results", "engine"}
            and record["schema"] == "selection_controller_dialogue_v1", "대화 record 형식")
    inputs = dialogue_input(snapshot, condition)
    require(record["input_digest"] == L.digest(inputs), "stale view 또는 입력 결합")
    expected = replay_dialogue(snapshot, condition, record["safe_results"])
    require(record["engine"] == expected["engine"], "선택·피드백·비용 재계산 불일치")
    require(set(record["engine"]) == set(CORE_DIALOGUE_FIELDS), "D2 engine 핵심 필드 범위")
    return deepcopy(record["engine"])


def _hold(snapshot: dict, policy: str, reason: str, model_status: str = "not_used") -> dict:
    checked = V.checked(snapshot)
    return {"schema": "selection_controller_choice_v1", "context_digest": L.digest(checked["binding"]),
            "view_digest": checked["view_digest"], "policy": policy, "selected_by": "hold",
            "action": "hold", "probe": None, "fact_ids": [], "model_status": model_status,
            "codes": [], "reason": reason, "used_for_decision": True, "approval_granted": False}


def choose(snapshot: dict, policy: str, dialogue: dict | None = None) -> dict:
    """invalid/infra/HOLD를 규칙으로 바꾸지 않는 controller 선택 함수."""
    checked = V.checked(snapshot)
    require(policy in POLICIES and checked["inputs"]["policy"] == policy, "snapshot 정책 결합")
    view = checked["view"]
    if view["status"] != "open" or view["budget_left"] <= 0:
        require(dialogue is None, "닫힌·pending·unknown 상태의 불필요한 대화")
        return _hold(checked, policy, "state_" + view["status"])
    uses_model = policy == "model_adaptive" or (policy == "model_first_then_rule" and not view["observations"])
    if uses_model:
        if dialogue is None:
            return _hold(checked, policy, "model_unavailable", "unknown")
        engine = verify_dialogue(checked, "production", dialogue)
        selected = engine["selection"]
        result = {"schema": "selection_controller_choice_v1", **{key: deepcopy(selected[key]) for key in (
            "context_digest", "view_digest", "policy", "selected_by", "action", "probe", "fact_ids",
            "model_status", "codes", "reason", "approval_granted")}, "used_for_decision": True}
        require(result["policy"] == policy, "모델 선택 정책 결합")
        return result
    require(dialogue is None, "규칙 단계에 모델 대화가 있다")
    rule = policy if policy in {"registry_next", "fixed_eligible"} else "registry_next"
    probe = V.rule_probe(view, rule)
    if probe is None:
        return _hold(checked, policy, "registered_policy_no_probe")
    return {**_hold(checked, policy, "registered_policy"), "selected_by": (
        "fixed_order" if rule == "fixed_eligible" else "registered_rule"),
        "action": "continue", "probe": probe}


def request_spec(snapshot: dict, choice: dict, *, condition: str = "production") -> dict:
    """production의 유효 continue만 host canonical args·예산을 가진 승인 후보로 만든다."""
    checked = V.checked(snapshot)
    view = checked["view"]
    require(condition == "production" and not checked["observation_masked_control"], "shadow 실행 거부")
    require(view["status"] == "open" and choice.get("action") == "continue"
            and choice.get("probe") in view["eligible_probes"]
            and choice.get("context_digest") == L.digest(checked["binding"])
            and choice.get("view_digest") == checked["view_digest"]
            and choice.get("approval_granted") is False, "선택·view 실행 후보 불일치")
    if choice.get("selected_by") == "model":
        require(choice.get("model_status") == "valid", "유효하지 않은 모델 선택 실행 거부")
    else:
        require(choice.get("selected_by") in {"registered_rule", "fixed_order"}
                and choice.get("model_status") == "not_used", "규칙 선택 주체 불일치")
    probe = choice["probe"]
    return {"prereg": deepcopy(checked["binding"]), "probe": probe,
            "args": deepcopy(RC.registry()["probes"][probe]["args"]),
            "checkpoint_sha256": checked["inputs"]["packet"]["input_bindings"]["case"]
            ["checkpoint_sha256"],
            "reference_checkpoint_sha256": checked["inputs"]["packet"]["input_bindings"]["reference"]
            ["checkpoint_sha256"],
            "budget_gpu_s": checked["inputs"]["packet"]["execution_budget_s"],
            "approval_granted": False}


def _frozen_request_spec(snapshot: dict, choice: dict) -> dict:
    """원장 쓰기 전 봉인한 snapshot을 현재 원장 상태로 다시 만들지 않고 검사한다."""
    require(isinstance(snapshot, dict) and snapshot.get("schema") == "selection_snapshot_v1",
            "봉인 snapshot 형식")
    view = snapshot.get("view", {})
    require(snapshot.get("view_digest") == L.digest(view) and view.get("status") == "open"
            and choice.get("action") == "continue" and choice.get("probe") in view.get("eligible_probes", [])
            and choice.get("context_digest") == L.digest(snapshot.get("binding"))
            and choice.get("view_digest") == snapshot.get("view_digest"), "봉인 선택·view 결합")
    if choice.get("selected_by") == "model":
        require(choice.get("model_status") == "valid", "봉인 모델 선택 상태")
    else:
        require(choice.get("selected_by") in {"registered_rule", "fixed_order"}
                and choice.get("model_status") == "not_used", "봉인 규칙 선택 주체")
    require(choice.get("approval_granted") is False, "봉인 선택은 미승인 후보여야 한다")
    probe = choice["probe"]
    return {"prereg": deepcopy(snapshot["binding"]), "probe": probe,
            "args": deepcopy(RC.registry()["probes"][probe]["args"]),
            "checkpoint_sha256": snapshot["inputs"]["packet"]["input_bindings"]["case"]
            ["checkpoint_sha256"],
            "reference_checkpoint_sha256": snapshot["inputs"]["packet"]["input_bindings"]["reference"]
            ["checkpoint_sha256"],
            "budget_gpu_s": snapshot["inputs"]["packet"]["execution_budget_s"],
            "approval_granted": False}


def _gpu_evidence(stage: Path, contract: str) -> dict:
    require(contract in GPU_CONTRACTS, "등록되지 않은 GPU receipt 계약")
    result = {}
    for name in ("gpu_before.json", "gpu_after.json"):
        path = stage / name
        value = read(path)
        if contract == "residual_v1":
            require(isinstance(value, dict) and value.get("admission_policy") == BACKEND.V1.GPU_PROFILE
                    and type(value.get("busy")) is bool and value["busy"] is False,
                    f"{name} GPU strict classifier receipt")
            require("bounded_resample" not in value and "samples" not in value,
                    f"{name} residual backend에 없는 bounded receipt")
            status = "strict_clear"
        else:
            status = GPU_CHECK.verify_gpu_receipt(value)
            require(status in {"initial_clear", "resampled_clear"}, f"{name} bounded GPU admission 거부")
        result[name] = digest_file(path)
        result[name + ".status"] = status
    return result


def adapt_actual_artifact(raw: dict, binding: dict, *, backend_arm: str, seed: int,
                          arm: str, policy: str, epoch: int, corrections: list[str],
                          source_checkpoint_sha256: str | None) -> dict:
    """동결 residual backend artifact를 controller 계보로 정규화한다."""
    corrections = RC.validate_applied(corrections)
    BACKEND.validate_artifact(raw, binding, backend_arm, seed)
    require(raw.get("source_checkpoint_sha256") == source_checkpoint_sha256,
            "backend artifact source checkpoint provenance")
    artifact = {
        "schema": "selection_controller_artifact_v1", "evidence_kind": ACTUAL_EVIDENCE,
        "actual_ready": True, "arm": arm, "policy": policy, "epoch": epoch,
        "corrections": corrections, "training_mode": "new_initialization",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "backend": {"artifact": deepcopy(raw), "binding": deepcopy(binding),
                    "arm": backend_arm, "seed": seed},
        "checkpoint": {"path": raw["checkpoint"]["path"], "sha256": raw["checkpoint"]["sha256"],
                       "identity": ""},
    }
    artifact["checkpoint"]["identity"] = checkpoint_identity(artifact)
    return artifact


def _actual_execution_files(stage: Path, execution: dict, gpu_contract: str) -> dict:
    require(stage.is_absolute() and stage.is_dir(), "actual probe stage")
    stored = read(stage / "execution.json")
    require(stored == execution and BACKEND.V1._finished(stored), "actual probe execution receipt")
    return {"execution.json": digest_file(stage / "execution.json"),
            **_gpu_evidence(stage, gpu_contract)}


def actual_legacy_probe_evidence(stage: Path, raw: dict, artifact: dict, probe: str,
                                 expected_run_name: str, *, expected_probes: list[str] | None = None,
                                 gpu_contract: str = "residual_v1") -> dict:
    """legacy probe의 jobs/report/execution/GPU 원본을 다시 읽는 actual adapter."""
    stage = Path(stage).resolve()
    arm, policy, epoch = artifact.get("arm"), artifact.get("policy"), artifact.get("epoch")
    if not isinstance(arm, str) or not isinstance(policy, str) or type(epoch) is not int:
        raise TypeError("actual artifact controller identity")
    validated = validate_artifact(
        artifact, arm=arm, policy=policy, epoch=epoch,
        corrections=artifact.get("corrections", []),
        source_checkpoint_sha256=artifact.get("source_checkpoint_sha256"), expected_kind=ACTUAL_EVIDENCE)
    expected_probes = [probe] if expected_probes is None else list(expected_probes)
    registered = sorted(L.PROBES)
    require(probe in L.PROBES and expected_probes in ([probe], registered)
            and raw.get("probes") == expected_probes
            and isinstance(expected_run_name, str) and bool(expected_run_name), "actual legacy probe 선택")
    report_path = stage / "report.json"
    jobs_path = stage / "jobs.json"
    report, jobs = read(report_path), read(jobs_path)
    require(raw.get("report_path") == str(report_path) and raw.get("report") == report
            and raw.get("execution") == read(stage / "execution.json"), "actual legacy raw return")
    backend = validated["backend"]
    backend_artifact = backend["artifact"]
    remaining = set(backend_artifact["remaining_faults"])
    expected_fault = "F1" if "F1" in remaining else "NONE"
    expected_job = {"name": expected_run_name,
                    "checkpoint": backend_artifact["checkpoint"]["path"], "probes": expected_probes,
                    "output": str(report_path)}
    job = jobs.get("jobs", [None])
    require(jobs.get("task") == BACKEND.TASK and jobs.get("fault") == expected_fault
            and jobs.get("seed") == backend["seed"] and isinstance(job, list) and len(job) == 1
            and isinstance(job[0], dict) and job[0].get("name") == expected_job["name"]
            and job[0].get("checkpoint") == expected_job["checkpoint"]
            and job[0].get("probes") == expected_probes and job[0].get("output") == str(report_path),
            "actual legacy jobs binding")
    observed = report.get("probes", {})
    require(report.get("name") == job[0].get("name") and report.get("task") == BACKEND.TASK
            and report.get("seed") == backend["seed"] and list(observed) == expected_probes
            and probe in observed and all(observed[item].get("exit") == 0 for item in expected_probes)
            and report.get("checkpoint", {}).get("sha256") == validated["checkpoint"]["sha256"],
            "actual legacy report binding")
    files = {"jobs.json": digest_file(jobs_path), "report.json": digest_file(report_path),
             **_actual_execution_files(stage, raw["execution"], gpu_contract)}
    return {"schema": "selection_actual_probe_evidence_v1", "backend_kind": "legacy",
            "stage": str(stage), "probe": probe, "artifact": deepcopy(artifact),
            "expected_run_name": expected_run_name, "expected_probes": expected_probes,
            "gpu_contract": gpu_contract,
            "raw": deepcopy(raw), "files": files, "measurement": deepcopy(observed[probe]["measurement"])}


def actual_optimizer_probe_evidence(stage: Path, raw: dict, binding: dict, *, backend_arm: str,
                                    seed: int, source_checkpoint_sha256: str,
                                    gpu_contract: str = "residual_v1") -> dict:
    """optimizer inspection의 manifest/records/readback/report와 GPU 원본을 재계산한다."""
    stage = Path(stage).resolve()
    paths = {name: stage / name for name in (
        "manifest.json", "records.jsonl", "readback.json", "inspection.json", "execution.json",
        "checkpoint.json", "checkpoint_execution.json")}
    require(all(path.is_file() for path in paths.values()), "actual optimizer 원본 파일")
    manifest = BACKEND.WRAPPER.validate_manifest(read(paths["manifest.json"]))
    run_name = raw.get("run_name")
    if not isinstance(run_name, str):
        raise TypeError("actual optimizer run_name")
    expected = BACKEND._manifest(binding, arm=backend_arm, seed=seed, run_name=run_name,
                                 mode="inspection", records=paths["records.jsonl"],
                                 readback=paths["readback.json"])
    require(manifest == expected and manifest.get("source_checkpoint_sha256") == source_checkpoint_sha256,
            "actual optimizer manifest binding")
    BACKEND._verify_repository_provenance(manifest)
    measured = BACKEND._inspection(paths["records.jsonl"], manifest)
    readback = BACKEND._validate_readback(read(paths["readback.json"]), manifest)
    remaining = list(BACKEND.FAULTS.remaining_faults(
        manifest["original_faults"], manifest["applied_corrections"]))
    summary = {key: measured[key] for key in ("status", "optimizer_steps", "minibatches")}
    configuration = {"seed": seed, "max_iterations": 3,
                     "num_learning_epochs": 5, "num_mini_batches": 4}
    run_dir = Path(str(raw.get("run_dir", "")))
    require(run_dir.is_absolute() and BACKEND.V1._verify_saved_configuration(run_dir, seed, 3) == configuration,
            "actual optimizer saved configuration")
    require(raw.get("manifest_path") == str(paths["manifest.json"])
            and raw.get("records_path") == str(paths["records.jsonl"])
            and raw.get("readback_path") == str(paths["readback.json"])
            and raw.get("report_path") == str(paths["inspection.json"])
            and raw.get("arm") == backend_arm and raw.get("seed") == seed
            and raw.get("source_checkpoint_sha256") == source_checkpoint_sha256
            and raw.get("checkpoint_loaded") is False
            and raw.get("original_faults") == manifest["original_faults"]
            and raw.get("applied_corrections") == manifest["applied_corrections"]
            and raw.get("remaining_faults") == remaining
            and raw.get("configuration") == configuration
            and raw.get("inspection") == summary and raw.get("measurement") == measured
            and raw.get("report") == measured["report"]
            and raw.get("readback") == readback and read(paths["inspection.json"]) == measured["report"]
            and raw.get("execution") == read(paths["execution.json"]), "actual optimizer raw 재계산")
    checkpoint = raw.get("checkpoint", {})
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    checkpoint_evidence = read(paths["checkpoint.json"])
    checkpoint_execution = read(paths["checkpoint_execution.json"])
    first, final = checkpoint_evidence.get("first", {}), checkpoint_evidence.get("final", {})
    first_path = Path(str(first.get("path", "")))
    require(raw.get("checkpoint_evidence") == checkpoint_evidence
            and raw.get("checkpoint_execution") == checkpoint_execution
            and BACKEND.V1._finished(checkpoint_execution)
            and checkpoint_evidence.get("device") == "cpu"
            and first.get("iteration") == 0 and final.get("iteration") == 2
            and first_path == run_dir / "model_0.pt"
            and checkpoint_path == run_dir / "model_2.pt"
            and first_path.is_file() and digest_file(first_path) == first.get("sha256")
            and checkpoint_path.is_file() and digest_file(checkpoint_path) == checkpoint.get("sha256")
            and final.get("path") == str(checkpoint_path) and final.get("sha256") == checkpoint.get("sha256"),
            "actual optimizer checkpoint evidence")
    files = {name: digest_file(path) for name, path in paths.items()}
    files.update(_actual_execution_files(stage, raw["execution"], gpu_contract))
    return {"schema": "selection_actual_probe_evidence_v1", "backend_kind": "optimizer",
            "stage": str(stage), "probe": RC.NEW_PROBE, "binding": deepcopy(binding),
            "backend_arm": backend_arm, "seed": seed, "source_checkpoint_sha256": source_checkpoint_sha256,
            "gpu_contract": gpu_contract,
            "raw": deepcopy(raw), "files": files, "measurement": {"status": measured["status"]}}


def validate_actual_probe_evidence(value: dict) -> dict:
    require(isinstance(value, dict) and value.get("schema") == "selection_actual_probe_evidence_v1",
            "actual probe evidence schema")
    kind = value.get("backend_kind")
    if kind == "legacy":
        expected = actual_legacy_probe_evidence(
            Path(value["stage"]), value["raw"], value["artifact"], value["probe"],
            value["expected_run_name"], expected_probes=value["expected_probes"],
            gpu_contract=value["gpu_contract"])
    elif kind == "optimizer":
        expected = actual_optimizer_probe_evidence(
            Path(value["stage"]), value["raw"], value["binding"], backend_arm=value["backend_arm"],
            seed=value["seed"], source_checkpoint_sha256=value["source_checkpoint_sha256"],
            gpu_contract=value["gpu_contract"])
    else:
        raise ValueError("actual probe backend kind")
    require(value == expected, "actual probe evidence 또는 원본 provenance 변경")
    return deepcopy(value["measurement"])


def _probe_measurement(value: dict, evidence_kind: str) -> dict:
    if evidence_kind == FIXTURE_EVIDENCE:
        require(value.get("schema") != "selection_actual_probe_evidence_v1", "actual evidence를 fixture로 사용")
        return deepcopy(value)
    require(evidence_kind == ACTUAL_EVIDENCE, "probe evidence kind")
    return validate_actual_probe_evidence(value)


def classify_raw(probe: str, case: dict, reference: dict) -> str:
    require(probe in RC.registry()["probes"] and isinstance(case, dict) and isinstance(reference, dict),
            "raw probe 측정 형식")
    if probe == RC.NEW_PROBE:
        require(case.get("status") in L.OUTCOMES and reference.get("status") in L.OUTCOMES,
                "optimizer raw status")
        return case["status"] if reference["status"] == "normal" else "unknown"
    return L.classify(probe, case, reference)


def _execution_plan(value: dict, spec: dict) -> dict:
    keys = {"stage", "run_name", "checkpoint_sha256", "seed", "args", "gpu_contract", "reference"}
    require(isinstance(value, dict) and set(value) == keys, "probe execution plan 필드")
    stage = Path(str(value["stage"]))
    sha = value["checkpoint_sha256"]
    require(stage.is_absolute() and isinstance(sha, str) and len(sha) == 64
            and all(char in "0123456789abcdef" for char in sha)
            and type(value["seed"]) is int and value["seed"] >= 0
            and sha == spec["checkpoint_sha256"] and value["args"] == spec["args"]
            and value["gpu_contract"] in GPU_CONTRACTS,
            "probe execution plan 값")
    BACKEND.V1._name(value["run_name"])
    reference = value["reference"]
    reference_keys = {"evidence_path", "evidence_sha256", "checkpoint_sha256", "stage",
                      "run_name", "seed", "gpu_contract"}
    require(isinstance(reference, dict) and set(reference) == reference_keys,
            "reference execution plan 필드")
    evidence_path, reference_stage = Path(str(reference["evidence_path"])), Path(str(reference["stage"]))
    reference_sha, reference_checkpoint = reference["evidence_sha256"], reference["checkpoint_sha256"]
    require(evidence_path.is_absolute() and evidence_path.is_file() and reference_stage.is_absolute()
            and isinstance(reference_sha, str) and digest_file(evidence_path) == reference_sha
            and isinstance(reference_checkpoint, str) and len(reference_checkpoint) == 64
            and all(char in "0123456789abcdef" for char in reference_checkpoint)
            and reference_checkpoint == spec["reference_checkpoint_sha256"]
            and type(reference["seed"]) is int and reference["seed"] >= 0
            and reference["gpu_contract"] in GPU_CONTRACTS, "reference execution plan 값")
    BACKEND.V1._name(reference["run_name"])
    return deepcopy(value)


def authorize_probe(root: Path, arm: str, epoch: int, snapshot: dict, choice: dict, approver: str,
                    *, evidence_kind: str, execution_plan: dict) -> dict:
    """외부 실행 전에 승인·소비를 끝내고 exact 실행 계획을 불변 파일로 봉인한다."""
    frozen = deepcopy(V.checked(snapshot))
    spec = request_spec(frozen, choice)
    require(isinstance(arm, str) and bool(arm) and epoch in (1, 2), "arm 또는 epoch")
    require(evidence_kind in {FIXTURE_EVIDENCE, ACTUAL_EVIDENCE}, "probe evidence kind")
    plan = _execution_plan(execution_plan, spec)
    prereg = spec["prereg"]
    ledger = S.M.RegistryLedger(Path(root) / arm / f"epoch{epoch}/probes.ledger.jsonl", RC.registry())
    request = ledger.propose(prereg, spec["probe"], spec["budget_gpu_s"])
    ledger.approve(request["request_id"], approver)
    ledger.consume(request["request_id"], prereg, spec["probe"])
    value = {"schema": "selection_probe_authorization_v1", "arm": arm, "epoch": epoch,
            "evidence_kind": evidence_kind,
            "snapshot_digest": L.digest(frozen), "snapshot": frozen, "choice": deepcopy(choice), "prereg": prereg,
            "execution_plan": plan, "request_id": request["request_id"]}
    path = Path(root) / arm / f"epoch{epoch}/authorization_{len(ledger.requests()):02d}.json"
    write_new(path, value)
    return {"path": str(path.resolve()), "sha256": digest_file(path),
            "request_id": request["request_id"], "snapshot": deepcopy(snapshot)}


def _authorization_reference(value: Path | dict, *, require_digest: bool) -> tuple[Path, str | None]:
    if isinstance(value, dict):
        require(set(value) >= {"path", "sha256", "request_id"}, "probe authorization reference")
        path, expected = Path(value["path"]), value["sha256"]
        require(isinstance(expected, str) and digest_file(path) == expected,
                "probe authorization immutable SHA")
        return path, expected
    require(not require_digest, "actual probe seal에는 authorization SHA reference가 필요하다")
    return Path(value), None


def _authorization(reference: Path | dict, snapshot: dict, state: str,
                   *, require_digest: bool = False) -> dict:
    path, _ = _authorization_reference(reference, require_digest=require_digest)
    value = read(path)
    require(value.get("schema") == "selection_probe_authorization_v1"
            and value.get("snapshot") == snapshot and value.get("snapshot_digest") == L.digest(snapshot),
            "probe authorization 입력 결합")
    require(Path(path).parent.name == f"epoch{value.get('epoch')}"
            and Path(path).parent.parent.name == value.get("arm"), "probe authorization arm/epoch 경로")
    choice = value["choice"]
    spec = _frozen_request_spec(snapshot, choice)
    _execution_plan(value["execution_plan"], spec)
    prereg = spec["prereg"]
    require(prereg == value["prereg"] and value.get("evidence_kind") in {FIXTURE_EVIDENCE, ACTUAL_EVIDENCE},
            "probe authorization prereg/evidence 불일치")
    ledger_path = Path(path).parent / "probes.ledger.jsonl"
    ledger = S.M.RegistryLedger(ledger_path, RC.registry())
    request = ledger.requests().get(value["request_id"], {})
    events = [row["event"] for row in ledger.events() if row.get("request_id") == value["request_id"]]
    expected_events = ["proposed", "approved", "consumed"] + (["receipt"] if state == "done" else [])
    require(state in {"running", "done"} and events == expected_events and request.get("state") == state
            and request.get("prereg_digest") == L.digest(prereg) and request.get("probe") == spec["probe"]
            and request.get("args_digest") == L.digest(spec["args"]),
            "probe authorization 승인·소비 상태")
    return {**deepcopy(value), "spec": spec, "ledger_path": str(ledger_path), "request": request}


def validate_probe_authorization(reference: Path | dict, snapshot: dict) -> dict:
    return _authorization(reference, snapshot, "running",
                          require_digest=isinstance(reference, dict))


def _probe_seal_path(authorization_path: Path) -> Path:
    suffix = Path(authorization_path).stem.removeprefix("authorization_")
    require(suffix.isdigit(), "probe authorization 파일명")
    return Path(authorization_path).with_name(f"probe_{suffix}.json")


def _validate_evidence_plan(evidence: dict, plan: dict) -> None:
    require(evidence.get("stage") == str(Path(plan["stage"]).resolve())
            and evidence.get("gpu_contract") == plan["gpu_contract"],
            "actual probe stage/GPU execution plan")
    if evidence.get("backend_kind") == "legacy":
        backend = evidence.get("artifact", {}).get("backend", {})
        require(evidence.get("expected_run_name") == plan["run_name"]
                and backend.get("seed") == plan["seed"]
                and evidence.get("artifact", {}).get("checkpoint", {}).get("sha256")
                == plan["checkpoint_sha256"], "legacy probe execution plan")
    elif evidence.get("backend_kind") == "optimizer":
        require(evidence.get("raw", {}).get("run_name") == plan["run_name"]
                and evidence.get("seed") == plan["seed"]
                and evidence.get("source_checkpoint_sha256") == plan["checkpoint_sha256"],
                "optimizer probe execution plan")
    else:
        raise ValueError("actual probe backend kind")


def _validate_reference_binding(reference_path: Path, evidence: dict, plan: dict,
                                evidence_kind: str) -> None:
    expected = plan["reference"]
    require(Path(reference_path).resolve() == Path(expected["evidence_path"]).resolve()
            and digest_file(reference_path) == expected["evidence_sha256"],
            "reference evidence file execution plan")
    if evidence_kind == ACTUAL_EVIDENCE:
        _validate_evidence_plan(evidence, {
            "stage": expected["stage"], "run_name": expected["run_name"],
            "checkpoint_sha256": expected["checkpoint_sha256"], "seed": expected["seed"],
            "gpu_contract": expected["gpu_contract"],
        })


def seal_probe(authorization_path: Path | dict, snapshot: dict, case_path: Path, reference_path: Path,
               gpu_s: float, exit_code: int = 0) -> dict:
    auth_path = _authorization_reference(authorization_path, require_digest=True)[0]
    auth = _authorization(authorization_path, snapshot, "running", require_digest=True)
    require(not isinstance(gpu_s, bool) and isinstance(gpu_s, (int, float)) and gpu_s > 0
            and exit_code == 0, "성공 probe 실측 비용·exit")
    case, reference = read(case_path), read(reference_path)
    _validate_reference_binding(reference_path, reference, auth["execution_plan"], auth["evidence_kind"])
    if auth["evidence_kind"] == ACTUAL_EVIDENCE:
        _validate_evidence_plan(case, auth["execution_plan"])
    case_measurement = _probe_measurement(case, auth["evidence_kind"])
    reference_measurement = _probe_measurement(reference, auth["evidence_kind"])
    outcome = classify_raw(auth["spec"]["probe"], case_measurement, reference_measurement)
    ledger = S.M.RegistryLedger(Path(auth["ledger_path"]), RC.registry())
    receipt = ledger.receipt(auth["request_id"], outcome, case_measurement, gpu_s, exit_code)
    value = {"schema": "selection_probe_seal_v1", "status": "completed",
             "authorization_path": str(auth_path.resolve()),
             "authorization_sha256": digest_file(auth_path),
             "arm": auth["arm"], "epoch": auth["epoch"], "evidence_kind": auth["evidence_kind"],
             "snapshot_digest": L.digest(snapshot), "snapshot": deepcopy(snapshot),
             "case_path": str(Path(case_path).resolve()), "case_sha256": digest_file(case_path),
             "reference_path": str(Path(reference_path).resolve()),
             "reference_sha256": digest_file(reference_path), "request_id": auth["request_id"],
             "receipt": receipt}
    path = _probe_seal_path(auth_path)
    write_new(path, value)
    return {"path": str(path.resolve()), "sha256": digest_file(path), "outcome": outcome,
            "request_id": auth["request_id"], "snapshot": deepcopy(snapshot)}


def fail_probe(authorization_path: Path | dict, snapshot: dict, *, gpu_s: float | None,
               exit_code: int | None, reason: str) -> dict:
    auth_path = _authorization_reference(authorization_path, require_digest=True)[0]
    auth = _authorization(authorization_path, snapshot, "running", require_digest=True)
    require((gpu_s is None or (not isinstance(gpu_s, bool) and isinstance(gpu_s, (int, float)) and gpu_s > 0))
            and (exit_code is None or type(exit_code) is int)
            and isinstance(reason, str) and bool(reason.strip()), "실패 probe 실측 비용·reason")
    ledger = S.M.RegistryLedger(Path(auth["ledger_path"]), RC.registry())
    if gpu_s is None:
        request = ledger.requests()[auth["request_id"]]
        receipt = ledger._append({  # noqa: SLF001 - null 비용 실패를 보존하는 local adapter
            "event": "receipt", "request_id": auth["request_id"], "outcome": "unknown",
            "measurement": {}, "gpu_s": None, "exit_code": exit_code, "over_budget": False,
            "registry_digest": ledger.registry_digest, "args_digest": request["args_digest"],
            "effect_key": request["effect_key"], "reason": reason,
        })
    else:
        receipt = ledger.receipt(auth["request_id"], "unknown", {}, gpu_s, exit_code, reason=reason)
    value = {"schema": "selection_probe_seal_v1", "status": "failed",
             "authorization_path": str(auth_path.resolve()),
             "authorization_sha256": digest_file(auth_path),
             "arm": auth["arm"], "epoch": auth["epoch"], "evidence_kind": auth["evidence_kind"],
             "snapshot_digest": L.digest(snapshot), "snapshot": deepcopy(snapshot),
             "case_path": None, "case_sha256": None, "reference_path": None, "reference_sha256": None,
             "request_id": auth["request_id"], "receipt": receipt}
    path = _probe_seal_path(auth_path)
    write_new(path, value)
    return {"path": str(path.resolve()), "sha256": digest_file(path), "outcome": "unknown",
            "request_id": auth["request_id"], "snapshot": deepcopy(snapshot)}


def record_probe(root: Path, arm: str, epoch: int, snapshot: dict, choice: dict,
                 case_path: Path, reference_path: Path, approver: str, gpu_s: float,
                 *, evidence_kind: str = FIXTURE_EVIDENCE) -> dict:
    """CPU fixture 편의 wrapper. actual은 authorize→validate→backend→seal 순서를 써야 한다."""
    require(evidence_kind == FIXTURE_EVIDENCE, "actual probe는 실행 전 authorization이 필요하다")
    frozen, spec = deepcopy(V.checked(snapshot)), request_spec(snapshot, choice)
    ledger = S.M.RegistryLedger(Path(root) / arm / f"epoch{epoch}/probes.ledger.jsonl", RC.registry())
    plan = {"stage": str(Path(case_path).parent.resolve()),
            "run_name": f"fixture_e{epoch}_{len(ledger.requests()) + 1}",
            "checkpoint_sha256": frozen["inputs"]["packet"]["input_bindings"]["case"]
            ["checkpoint_sha256"], "seed": 0,
            "args": spec["args"], "gpu_contract": "residual_v1",
            "reference": {
                "evidence_path": str(Path(reference_path).resolve()),
                "evidence_sha256": digest_file(reference_path),
                "checkpoint_sha256": frozen["inputs"]["packet"]["input_bindings"]["reference"]
                ["checkpoint_sha256"], "stage": str(Path(reference_path).parent.resolve()),
                "run_name": f"fixture_reference_e{epoch}", "seed": 0,
                "gpu_contract": "residual_v1",
            }}
    auth = authorize_probe(root, arm, epoch, snapshot, choice, approver,
                           evidence_kind=FIXTURE_EVIDENCE, execution_plan=plan)
    return seal_probe(auth, snapshot, case_path, reference_path, gpu_s, 0)


def verify_probe(path: Path, snapshot: dict) -> dict:
    value = read(path)
    require(value.get("schema") == "selection_probe_seal_v1"
            and value.get("snapshot") == snapshot and value.get("snapshot_digest") == L.digest(snapshot),
            "probe seal 입력 결합")
    authorization_path = Path(value["authorization_path"])
    require(digest_file(authorization_path) == value["authorization_sha256"], "probe authorization SHA")
    auth = _authorization(authorization_path, snapshot, "done")
    require(value.get("arm") == auth["arm"] and value.get("epoch") == auth["epoch"]
            and value.get("evidence_kind") == auth["evidence_kind"]
            and value.get("request_id") == auth["request_id"], "probe seal authorization 결합")
    request, receipt = auth["request"], value["receipt"]
    require(request.get("receipt") == receipt, "probe receipt 원장 불일치")
    if value.get("status") == "failed":
        require(all(value.get(key) is None for key in (
            "case_path", "case_sha256", "reference_path", "reference_sha256"))
            and receipt.get("outcome") == "unknown" and receipt.get("measurement") == {}
            and receipt.get("gpu_s") != 0 and receipt.get("reason"), "실패 probe unknown receipt")
        return {"probe": auth["spec"]["probe"], "outcome": "unknown", "measurement": {},
                "evidence_kind": auth["evidence_kind"], "request_id": auth["request_id"],
                "seal_sha256": digest_file(path)}
    require(value.get("status") == "completed", "probe seal status")
    case_path, reference_path = Path(value["case_path"]), Path(value["reference_path"])
    require(digest_file(case_path) == value["case_sha256"]
            and digest_file(reference_path) == value["reference_sha256"], "raw probe 파일 SHA")
    evidence_kind = value.get("evidence_kind")
    case_measurement = _probe_measurement(read(case_path), evidence_kind)
    reference_evidence = read(reference_path)
    _validate_reference_binding(reference_path, reference_evidence, auth["execution_plan"], evidence_kind)
    if evidence_kind == ACTUAL_EVIDENCE:
        _validate_evidence_plan(read(case_path), auth["execution_plan"])
    reference_measurement = _probe_measurement(reference_evidence, evidence_kind)
    outcome = classify_raw(auth["spec"]["probe"], case_measurement, reference_measurement)
    require(receipt.get("outcome") == outcome and receipt.get("measurement") == case_measurement
            and receipt.get("gpu_s") > 0 and receipt.get("exit_code") == 0
            and receipt.get("over_budget") is False, "probe raw 분류 receipt 불일치")
    return {"probe": auth["spec"]["probe"], "outcome": outcome, "measurement": case_measurement,
            "evidence_kind": evidence_kind, "request_id": value["request_id"],
            "seal_sha256": digest_file(path)}


def checkpoint_identity(artifact: dict) -> str:
    checkpoint = artifact.get("checkpoint", {})
    return L.digest({"arm": artifact.get("arm"), "policy": artifact.get("policy"),
                     "epoch": artifact.get("epoch"), "corrections": artifact.get("corrections"),
                     "training_mode": artifact.get("training_mode"),
                     "evidence_kind": artifact.get("evidence_kind"),
                     "source_checkpoint_sha256": artifact.get("source_checkpoint_sha256"),
                     "path": str(Path(checkpoint.get("path", "")).resolve()), "sha256": checkpoint.get("sha256")})


def validate_artifact(artifact: dict, *, arm: str, policy: str, epoch: int,
                      corrections: list[str], source_checkpoint_sha256: str | None,
                      expected_kind: str | None = None) -> dict:
    corrections = RC.validate_applied(corrections)
    require(isinstance(artifact, dict) and artifact.get("schema") == "selection_controller_artifact_v1"
            and artifact.get("arm") == arm and artifact.get("policy") == policy
            and artifact.get("epoch") == epoch and artifact.get("corrections") == corrections
            and artifact.get("training_mode") == "new_initialization"
            and artifact.get("source_checkpoint_sha256") == source_checkpoint_sha256, "artifact 계보·arm 결합")
    kind = artifact.get("evidence_kind")
    require(kind in {FIXTURE_EVIDENCE, ACTUAL_EVIDENCE}
            and (expected_kind is None or kind == expected_kind), "artifact evidence kind")
    if kind == FIXTURE_EVIDENCE:
        require(artifact.get("actual_ready") is False and "backend" not in artifact,
                "CPU fixture를 actual artifact로 사용할 수 없다")
    else:
        require(artifact.get("actual_ready") is True and isinstance(artifact.get("backend"), dict),
                "actual backend provenance")
        backend = artifact["backend"]
        BACKEND.validate_artifact(backend.get("artifact"), backend.get("binding"),
                                  backend.get("arm"), backend.get("seed"))
        require(backend["artifact"].get("source_checkpoint_sha256") == source_checkpoint_sha256,
                "actual backend source checkpoint provenance")
    checkpoint = artifact.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint 형식")
    require(set(checkpoint) == {"path", "sha256", "identity"}, "checkpoint 형식")
    path = Path(checkpoint["path"])
    require(path.is_file() and digest_file(path) == checkpoint["sha256"]
            and checkpoint["identity"] == checkpoint_identity(artifact), "checkpoint 파일·identity 불일치")
    return deepcopy(artifact)


def fixed_report(path: Path, artifact: dict) -> dict:
    raw = read(path)
    expected = {"name": raw.get("name"), "checkpoint_file": Path(artifact["checkpoint"]["path"]).name,
                "checkpoint_sha256": artifact["checkpoint"]["sha256"], "action_scale": 0.25,
                "protocol_sha256_lf": FIXED_EVAL_PROTOCOL_SHA256}
    return G.validate_fixed_report(raw, expected)


def registered_correction(snapshot: dict, applied: list[str]) -> tuple[str, str]:
    applied = RC.validate_applied(applied)
    view = V.checked(snapshot)["view"]
    require(view["status"] == "confirmed" and len(view["remaining_candidates"]) == 1, "교정 전 단일 확정 필요")
    component = view["remaining_candidates"][0]
    correction = RC.CORRECTIONS.get(component)
    confirming = "P_reward" if component == "reward" else RC.NEW_PROBE if component == RC.COUNT else None
    observed = {row["probe"]: row.get("outcome") for row in view["observations"]}
    if not isinstance(component, str) or not isinstance(correction, str) or confirming is None:
        raise ValueError("등록 교정 범위")
    require(correction not in applied and len(applied) < 2
            and observed.get(confirming) == "abnormal", "등록 교정 범위·중복·상한")
    return component, correction


def authorize_repair(ledger_path: Path, arm: str, snapshot: dict, applied: list[str],
                     probe_seals: list[dict], approver: str, budget_gpu_s: int) -> dict:
    component, correction = registered_correction(snapshot, applied)
    verified = [verify_probe(Path(row["path"]), row["snapshot"]) for row in probe_seals]
    current = V.checked(snapshot)
    require(all(row["snapshot"]["binding"] == current["binding"] for row in probe_seals),
            "다른 arm/epoch probe seal 혼합")
    require(any(row["outcome"] == "abnormal" and row["probe"] == (
        "P_reward" if component == "reward" else RC.NEW_PROBE) for row in verified), "확정 raw probe receipt 없음")
    evidence_kinds = {row["evidence_kind"] for row in verified}
    require(len(evidence_kinds) == 1, "fixture와 actual probe evidence 혼합")
    evidence_kind = evidence_kinds.pop()
    registry = RC.correction_registry()
    ledger = D1.CorrectionLedger(ledger_path, registry)
    entry = registry[component]
    evidence = {"arm": arm, "evidence_kind": evidence_kind,
                "probe_seals": [{"path": row["path"], "sha256": row["sha256"]}
                                  for row in probe_seals], "selection_source": "registered_host_rule"}
    request = ledger.propose(L.digest(V.checked(snapshot)), component, L.digest(verified), correction, "restart",
                             {"case": entry["protocol_case"], "arm": "corrected"}, budget_gpu_s, evidence=evidence)
    ledger.approve(request["request_id"], approver)
    ledger.consume(request["request_id"], L.digest(V.checked(snapshot)))
    return {"request_id": request["request_id"], "component": component, "correction_id": correction,
            "evidence_kind": evidence_kind, "evidence": evidence}


def seal_repair(path: Path, *, ledger_path: Path, authorization: dict, arm: str, policy: str, epoch: int,
                applied_before: list[str], input_artifact: dict, output_artifact: dict,
                output_report_path: Path, reference_artifact: dict, reference_report_path: Path,
                bands: dict, gpu_s: float, parent_path: Path | None = None) -> dict:
    before = RC.validate_applied(applied_before)
    require(epoch in (1, 2) and len(before) == epoch - 1, "repair epoch와 누적 교정")
    after = RC.validate_applied([*before, authorization["correction_id"]])
    evidence_kind = authorization.get("evidence_kind")
    require(evidence_kind == authorization.get("evidence", {}).get("evidence_kind"),
            "교정 authorization evidence kind")
    input_source = None if epoch == 1 else input_artifact.get("source_checkpoint_sha256")
    validate_artifact(input_artifact, arm=arm, policy=policy, epoch=epoch - 1, corrections=before,
                      source_checkpoint_sha256=input_source, expected_kind=evidence_kind)
    validate_artifact(output_artifact, arm=arm, policy=policy, epoch=epoch, corrections=after,
                      source_checkpoint_sha256=input_artifact["checkpoint"]["sha256"],
                      expected_kind=evidence_kind)
    validate_artifact(reference_artifact, arm="NONE", policy="shared_reference", epoch=0, corrections=[],
                      source_checkpoint_sha256=None, expected_kind=evidence_kind)
    parent = None
    if epoch == 1:
        require(parent_path is None, "첫 epoch 부모")
    else:
        if parent_path is None:
            raise ValueError("epoch2 부모 receipt")
        parent = verify_repair(parent_path, ledger_path=ledger_path, bands=bands)
        require(parent["outcome"] == "not_recovered" and parent["arm"] == arm and parent["policy"] == policy
                and parent["applied_after"] == before
                and parent["output_checkpoint_identity"] == input_artifact["checkpoint"]["identity"],
                "epoch2 부모 arm·비회복·출력 checkpoint 결합")
    report, reference = fixed_report(output_report_path, output_artifact), fixed_report(reference_report_path, reference_artifact)
    recovery = D1.recovery(report, reference, bands)
    value = {"schema": "selection_controller_repair_seal_v1", "arm": arm, "policy": policy, "epoch": epoch,
             "applied_before": before, "applied_after": after, "authorization": deepcopy(authorization),
             "input_artifact": deepcopy(input_artifact), "output_artifact": deepcopy(output_artifact),
             "input_checkpoint_identity": input_artifact["checkpoint"]["identity"],
             "output_checkpoint_identity": output_artifact["checkpoint"]["identity"],
             "output_report_path": str(Path(output_report_path).resolve()),
             "output_report_sha256": digest_file(output_report_path),
             "reference_artifact": deepcopy(reference_artifact),
             "reference_report_path": str(Path(reference_report_path).resolve()),
             "reference_report_sha256": digest_file(reference_report_path),
             "parent_path": str(Path(parent_path).resolve()) if parent_path else None,
             "parent_sha256": digest_file(parent_path) if parent_path else None,
             "recovery": recovery, "gpu_s": gpu_s,
             "training_semantics": "cumulative_corrections_new_initialization_not_source_warmstart"}
    write_new(path, value)
    ledger = D1.CorrectionLedger(ledger_path, RC.correction_registry())
    ledger.receipt(authorization["request_id"], f"controller_{evidence_kind}", recovery["outcome"],
                   digest_file(path), gpu_s, 0)
    return value


def verify_repair(path: Path, *, ledger_path: Path, bands: dict) -> dict:
    value = read(path)
    require(value.get("schema") == "selection_controller_repair_seal_v1"
            and value.get("training_semantics") == "cumulative_corrections_new_initialization_not_source_warmstart",
            "repair seal 형식")
    arm, policy, epoch = value["arm"], value["policy"], value["epoch"]
    before, after = RC.validate_applied(value["applied_before"]), RC.validate_applied(value["applied_after"])
    require(epoch in (1, 2) and len(before) == epoch - 1 and len(after) == epoch
            and after[:-1] == before and len(set(after)) == len(after), "repair 누적 교정 순서·상한")
    auth = value["authorization"]
    evidence_kind = auth.get("evidence_kind")
    require(evidence_kind == auth.get("evidence", {}).get("evidence_kind"),
            "교정 authorization evidence kind")
    input_source = None if epoch == 1 else value["input_artifact"].get("source_checkpoint_sha256")
    validate_artifact(value["input_artifact"], arm=arm, policy=policy, epoch=epoch - 1,
                      corrections=before, source_checkpoint_sha256=input_source, expected_kind=evidence_kind)
    validate_artifact(value["output_artifact"], arm=arm, policy=policy, epoch=epoch,
                      corrections=after,
                      source_checkpoint_sha256=value["input_artifact"]["checkpoint"]["sha256"],
                      expected_kind=evidence_kind)
    validate_artifact(value["reference_artifact"], arm="NONE", policy="shared_reference", epoch=0,
                      corrections=[], source_checkpoint_sha256=None, expected_kind=evidence_kind)
    require(value["input_checkpoint_identity"] == value["input_artifact"]["checkpoint"]["identity"]
            and value["output_checkpoint_identity"] == value["output_artifact"]["checkpoint"]["identity"],
            "repair checkpoint identity")
    if epoch == 1:
        require(value["parent_path"] is None and value["parent_sha256"] is None, "epoch1 부모")
    else:
        parent_path = Path(value["parent_path"])
        require(digest_file(parent_path) == value["parent_sha256"], "부모 result SHA")
        parent = verify_repair(parent_path, ledger_path=ledger_path, bands=bands)
        require(parent["outcome"] == "not_recovered" and parent["arm"] == arm and parent["policy"] == policy
                and parent["applied_after"] == before
                and parent["output_checkpoint_identity"] == value["input_checkpoint_identity"],
                "cross-arm 또는 다른 checkpoint 부모")
    output_path, reference_path = Path(value["output_report_path"]), Path(value["reference_report_path"])
    require(digest_file(output_path) == value["output_report_sha256"]
            and digest_file(reference_path) == value["reference_report_sha256"], "고정 평가 raw 파일 SHA")
    measured = D1.recovery(fixed_report(output_path, value["output_artifact"]),
                           fixed_report(reference_path, value["reference_artifact"]), bands)
    require(measured == value["recovery"], "자기신고 recovery와 raw 고정 평가 재계산 불일치")
    ledger = D1.CorrectionLedger(ledger_path, RC.correction_registry())
    request = ledger.requests().get(auth["request_id"], {})
    events = [row["event"] for row in ledger.events() if row.get("request_id") == auth["request_id"]]
    receipt = request.get("receipt", {})
    require(events == ["proposed", "approved", "consumed", "receipt"] and request.get("state") == "done"
            and request.get("correction_id") == after[-1] and request.get("evidence") == auth["evidence"]
            and receipt.get("result_sha") == digest_file(path) and receipt.get("verdict") == measured["outcome"]
            and receipt.get("exit_code") == 0 and receipt.get("over_budget") is False,
            "교정 승인·일회 소비·result SHA·판정 불일치")
    return {"arm": arm, "policy": policy, "epoch": epoch, "applied_after": after,
            "outcome": measured["outcome"], "result_sha256": digest_file(path),
            "output_checkpoint_identity": value["output_checkpoint_identity"]}


def next_epoch(parent_path: Path | None, *, ledger_path: Path, bands: dict,
               source_outcome: str | None = None) -> bool:
    """NONE/단일 회복에는 요청을 만들지 않고, 검증된 비회복 receipt에만 epoch2를 연다."""
    if parent_path is None:
        require(source_outcome in {"source_failure_absent", "recovered_after_one"}, "부모 없는 종료 상태")
        return False
    parent = verify_repair(parent_path, ledger_path=ledger_path, bands=bands)
    return parent["epoch"] == 1 and parent["outcome"] == "not_recovered" and len(parent["applied_after"]) == 1


def protocol_value() -> dict:
    return {"schema": "selection_comparison_controller_protocol_v1",
            "measurement_kind": "synthetic_cpu_controller_contract_only",
            "policies": list(POLICIES), "epochs_max": 2, "distinct_corrections_max": 2,
            "dialogue": {"engine": "selection_comparison_call_v2.engine", "conditions": list(CALL.CONDITIONS),
                         "invalid_or_infra": "HOLD_without_V.select_or_rule_probe",
                         "closed_pending_unknown": "API_0", "shadow_executable": False},
            "lineage": {"shared": "same source NONE and compound F1/R1 reference only",
                        "isolated": "arm ledgers, probe/correction/evaluation/checkpoint",
                        "epoch2": "verified epoch1 not_recovered receipt and exact output checkpoint identity",
                        "training": "cumulative corrections on new initialization; prior output is diagnosis input only",
                        "source_checkpoint": "epoch1 output binds compound source SHA; epoch2 output binds same-arm epoch1 output SHA"},
            "evidence_boundary": {
                "fixture": "synthetic_cpu_fixture_v1; actual_ready false",
                "actual": "residual_execution_v1 artifact validator plus raw probe provenance adapter",
                "probe_lifecycle": "authorize and consume exact execution plan before backend; seal raw evidence after backend",
                "execution_plan_fields": ["stage", "run_name", "checkpoint_sha256", "seed", "args",
                                          "gpu_contract", "reference"],
                "reference_plan_fields": ["evidence_path", "evidence_sha256", "checkpoint_sha256",
                                          "stage", "run_name", "seed", "gpu_contract"],
                "reference_binding": "snapshot reference checkpoint plus immutable shared evidence identity",
                "hash_contract": "artifact/evidence use raw bytes; source/protocol use LF-normalized bytes",
                "actual_authorization_reference": "path plus immutable SHA required at seal",
                "failed_probe": "unknown receipt with positive measured GPU seconds or null; zero is rejected",
                "shared_none_reference": "registered legacy six probes in exact order; upper controller counts its cost once",
                "gpu_contracts": {
                    "residual_v1": "residual backend strict profile and busy=false receipt",
                    "apps_bounded_v1": "selection_comparison_gpu_v1 processed receipt verifier; only initial_clear or resampled_clear",
                },
                "gpu_receipts_relabelled": False,
                "gpu_performance_claim": False,
                "probe_raw_required": ["jobs_or_manifest", "report_or_records", "readback", "execution",
                                       "gpu_before", "gpu_after"],
                "mixed_fixture_actual": "reject",
                "fixed_eval_protocol_sha256_lf": FIXED_EVAL_PROTOCOL_SHA256,
            },
            "correction_selection": "registered_host_rule_without_correction_model",
            "design_coupling": "설계자가 residual seed 2120·2121의 두 교정 회복 기록을 본 뒤 만든 CPU fixture 계약이다.",
            "observed_d2": {"status": "complete", "final_valid": "6/6", "api_requests": 7,
                            "errors": 1, "runtime_verified": False, "token_cost": None,
                            "selector_effect_measured": False, "gpu_recovery_compared": False},
            "claims": {"new_api_calls": 0, "new_gpu_runs": 0, "new_training_runs": 0,
                       "selector_effect": False, "recovery_effect": False, "generalization": False},
            "acceptance": ["same_public_view_four_policies", "invalid_model_never_rule_replaced",
                           "raw_probe_and_fixed_eval_recomputed", "arm_and_checkpoint_lineage_isolated",
                           "probe_authorized_before_backend_and_consumed_once",
                           "shared_none_exact_six_probes", "approval_once",
                           "none_and_single_fault_epoch2_zero", "compound_fixture_two_repairs"],
            "source_sha256_lf": {name: digest_lf_file(ROOT / name) for name in SOURCE_FILES}}


def prepare() -> dict:
    write_new(PROTOCOL, protocol_value())
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def check() -> dict:
    value = read(PROTOCOL)
    require(value == protocol_value(), "controller protocol 또는 source SHA 불일치")
    raw = json.dumps(value, ensure_ascii=False)
    require("C:\\Users" not in raw and "/home/" not in raw and "LIMMM" not in raw, "공개 protocol 사용자 경로")
    return {"status": "ok", "protocol_sha256_lf": digest_lf_file(PROTOCOL)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "선택 비교 controller").splitlines()[0])
    parser.add_argument("command", choices=("prepare", "check"))
    args = parser.parse_args(argv)
    print(json.dumps(prepare() if args.command == "prepare" else check(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
