"""등록 P_value를 CPU에서 실행한다. archive plan → propose → approve → execute 순서다."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
import residual_selection_v2 as S

C, M, L = S.C1, S.M, S.C1.L
PROTOCOL = ROOT / "bench/protocols/checkpoint_value_cpu_v1.json"
PUBLIC_PROVENANCE = ROOT / "evals/results/residual_success_public_20261017/provenance.json"
PUBLIC_DIGEST = "9178bed801456745e4dbeec7e05e300dfafc4c3550190c12d20c56a4e26de9a5"
ARCHIVE_SHA = "f856801a76f71e990955adf7c5856d10cc9d68395a63b699f59e7340d865e043"
ARCHIVE_PROTOCOL_SHA = "d2012679fc29be9a0cb687940a4577fc33212b655d4d89fe45db3489f9983ef4"
PLAN_SCHEMA = "checkpoint_value_cpu_plan_v1"
SCOPE = {"aggregation_unit": "conditional_epoch_contexts_on_recorded_history",
         "epoch2_conditioned_on_actual_epoch1": True, "trajectory_comparison_valid": False,
         "selector_performance_comparison_valid": False, "existing_receipt_replay": False,
         "new_followup_measurement": True,
         "first_checkpoint_anchor": "sealed_repair_receipt_checkpoint_evidence",
         "first_checkpoint_historical_integrity_verified": True}
FAILURES = {
    "torch_unavailable": ("torch_import", "CPU torch를 불러올 수 없음"),
    "cuda_initialized": ("device_check", "CUDA 초기화 상태에서 CPU 측정 거부"),
    "checkpoint_unreadable": ("checkpoint_hash", "체크포인트를 읽을 수 없음"),
    "checkpoint_sha_mismatch": ("checkpoint_hash", "승인 뒤 체크포인트 변경"),
    "weights_only_load_failed": ("checkpoint_load", "안전한 체크포인트 읽기 실패"),
    "model_state_dict_missing": ("checkpoint_load", "model_state_dict 없음"),
    "parameter_prefix_missing": ("parameter_check", "파라미터 prefix 없음"),
    "parameter_structure_invalid": ("parameter_check", "파라미터 구조 또는 shape 불일치"),
    "parameter_nonfinite": ("parameter_check", "비유한 체크포인트 파라미터"),
    "parameter_change_nonfinite": ("parameter_change", "비유한 파라미터 변화량"),
    "reference_critic_change_not_positive": ("reference_check", "정상 기준의 critic 변화량이 양수가 아님"),
    "unclassified_execution_error": ("current_phase", "등록되지 않은 CPU 실행 오류"),
}


class MeasurementFailure(ValueError):
    """절대경로나 원문 예외를 복사하지 않는 측정 실패 코드."""

    def __init__(self, code: str):
        self.code = code
        self.stage, message = FAILURES[code]
        super().__init__(message)


def measurement_require(ok: bool, code: str) -> None:
    if not ok:
        raise MeasurementFailure(code)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path, lf: bool = False) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True,
                            encoding="utf-8", errors="strict")
    return result.stdout.strip()


def sources() -> dict:
    paths = {"evals/checkpoint_value_cpu_v1.py", "tests/test_checkpoint_value_cpu_v1.py", "evals/probes.py",
             "evals/correction_model_call.py", "evals/r1_repair_contract.py", "evals/residual_contract_v1.py",
             "evals/residual_selection_v2.py", "src/rl_triage/__init__.py", "src/rl_triage/active_diagnosis.py",
             "src/rl_triage/correction_registry.py", "src/rl_triage/mechanism_registry.py", "src/rl_triage/probe_loop.py"}
    return {path: sha(ROOT / path, True) for path in sorted(paths)}


def protocol_value() -> dict:
    archived_protocol = ROOT / "bench/protocols/residual_demo_v2.json"
    require(sha(archived_protocol, True) == ARCHIVE_PROTOCOL_SHA, "원래 등록 프로토콜 SHA 불일치")
    frozen = read(archived_protocol)["source_sha256_lf"]
    for path in ("evals/probes.py", "src/rl_triage/probe_loop.py"):
        require(sha(ROOT / path, True) == frozen[path], "동결 측정·판정 정의 변경")
    return {"schema": "checkpoint_value_cpu_protocol_v1", "probe": "P_value", **SCOPE,
        "measurement": "동일 실행의 첫·마지막 actor/critic 파라미터 상대 L2 변화",
        "definition_source": "evals/probes.py", "classification_source": "src/rl_triage/probe_loop.py",
        "archived_public_provenance_digest": PUBLIC_DIGEST, "archived_result_sha256": ARCHIVE_SHA,
        "development_disclosure": "등록 F1·R1과 residual v2 결과를 본 뒤 만든 CPU 측정 경로다. 새 결함·새 seed 평가가 아니다.",
        "population": {"seeds": [2120, 2121], "epochs": [1, 2], "contexts": 4, "checkpoint_pairs": 6},
        "prediction": {"case_classification": "네 조건 모두 normal",
                       "equivalence": "수용한 실수·strided actor/critic tensor 입력에서 동결 P_value 정의와 같은 값"},
        "failure_policy": "출처·SHA·승인·소비·비유한 값 오류는 invalid로 종료하며 임계값·입력을 바꾸지 않는다",
        "input_validity": {"reference_critic_change": "> 0", "first_iteration": 0,
                           "last_iteration": 299, "checkpoint_loader": "torch.load(weights_only=True)",
                           "historical_first_sha_required": True},
        "failure_codes": {code: stage for code, (stage, _) in FAILURES.items()},
        "execution": {"device": "cpu", "app_launcher": False, "gpu_calls": 0, "model_calls": 0,
                      "cuda_visible_devices": "-1", "request_budget_gpu_s": 0},
        "interpretation": "critic 갱신 측정이며 critic 보정·최종 행동·회복·모델 기여를 판정하지 않는다",
        "source_sha256_lf": sources()}


def verify_sources(anchor: str | None = None) -> str:
    require(read(PROTOCOL) == protocol_value(), "CPU 프로토콜·소스 결합 불일치")
    head = git("rev-parse", "HEAD")
    anchor = anchor or head
    git("merge-base", "--is-ancestor", anchor, head)
    for path in [*sources(), PROTOCOL.relative_to(ROOT).as_posix()]:
        original = git("rev-parse", f"{anchor}:{path}")
        require(git("rev-parse", f"HEAD:{path}") == original
                and git("rev-parse", f":{path}") == original, "결합 Git blob 불일치: " + path)
        require(not git("status", "--porcelain", "--", path), "측정 전 소스 커밋 필요: " + path)
    return anchor


def descriptor(path: Path, expected: str | None = None) -> dict:
    path = path.resolve()
    actual = sha(path)
    require(expected is None or actual == expected, "체크포인트 SHA 불일치")
    return {"path": str(path), "sha256": actual}


def anchored_pair(artifact: dict, inventory: dict, bound_last_sha: str) -> dict:
    """공개 SHA로 봉인된 bundle의 checkpoint_evidence에서 두 끝점을 대조한다."""
    manifest = Path(artifact["manifest_path"]).resolve()
    require(inventory.get(str(manifest)) == sha(manifest), "보관 manifest inventory 불일치")
    evidence = artifact["checkpoint_evidence"]
    evidence_path = manifest.with_name("checkpoint.json")
    require(read(evidence_path) == evidence, "보관 checkpoint_evidence 불일치")
    first, final = evidence["first"], evidence["final"]
    require(type(first["iteration"]) is int and first["iteration"] == 0
            and type(final["iteration"]) is int and final["iteration"] == 299, "보관 iteration 불일치")
    run_dir = Path(artifact["run_dir"]).resolve()
    last_path = Path(artifact["checkpoint"]["path"]).resolve()
    first_path = last_path.with_name(L.PROBES["P_value"]["args"]["from"])
    require(last_path.parent == run_dir and last_path.name == "model_299.pt"
            and Path(first["path"]).resolve() == first_path
            and Path(final["path"]).resolve() == last_path, "보관 동일 실행 경로 불일치")
    require(final["sha256"] == artifact["checkpoint"]["sha256"] == bound_last_sha,
            "보관 final·packet SHA 불일치")
    return {"first": descriptor(first_path, first["sha256"]),
            "last": descriptor(last_path, final["sha256"]),
            "checkpoint_evidence": descriptor(evidence_path), "manifest": descriptor(manifest),
            "historical_evidence_digest": L.digest(evidence)}


def archive_plan(run: Path) -> dict:
    """SHA만 읽어 원본 입력을 고정한다. 텐서는 불러오지 않는다."""
    run = run.resolve()
    provenance = read(PUBLIC_PROVENANCE)
    require(L.digest(provenance) == PUBLIC_DIGEST, "공개 출처 digest 불일치")
    require(sha(run / "result.json") == ARCHIVE_SHA, "보관 실행 원본 SHA 불일치")
    pairs, contexts = {}, []
    for seed in (2120, 2121):
        for epoch in (1, 2):
            folder = run / f"s{seed}/epoch{epoch}"
            bundle_path = folder / "repair_receipt.json"
            require(sha(bundle_path) == provenance["source_roles_sha256"][f"s{seed}.epoch{epoch}.repair_receipt"],
                    "보관 receipt SHA 불일치")
            bundle = read(bundle_path)
            packet_path = folder / "model/diagnosis.packet.json"
            require(bundle["inventory_sha256"].get(str(packet_path.resolve())) == sha(packet_path),
                    "보관 packet inventory 불일치")
            packet = S.validate_packet(read(packet_path))
            require(bundle["seed"] == seed and bundle["epoch"] == packet["epoch"] == epoch,
                    "보관 seed·epoch 불일치")
            require(bundle["applied_before"] == packet["applied_corrections"], "보관 교정 이력 불일치")
            jobs = {}
            for role, artifact in (("case", bundle["source"]), ("reference", bundle["reference_artifact"])):
                key = f"s{seed}_e{epoch}" if role == "case" else f"s{seed}_reference"
                pair = anchored_pair(artifact, bundle["inventory_sha256"],
                                     packet["input_bindings"][role]["checkpoint_sha256"])
                require(key not in pairs or pairs[key] == pair, "공유 정상 기준 입력 불일치")
                pairs[key], jobs[role] = pair, key
            contexts.append({"id": f"s{seed}_e{epoch}", "seed": seed, "epoch": epoch,
                "applied_before": packet["applied_corrections"], "packet_sha256": sha(packet_path),
                "packet_digest": L.digest(packet), "jobs": jobs})
    require(len(pairs) == 6 and len(contexts) == 4, "CPU 분석 모집단 불일치")
    return {"schema": PLAN_SCHEMA, **SCOPE, "source_run": str(run), "source_result_sha256": ARCHIVE_SHA,
            "public_provenance_digest": PUBLIC_DIGEST, "pairs": pairs, "contexts": contexts}


def load_pair(pair: dict) -> dict:
    try:
        import torch
    except ImportError as error:
        raise MeasurementFailure("torch_unavailable") from error
    measurement_require(not torch.cuda.is_initialized(), "cuda_initialized")
    snapshots = {}
    for role in ("first", "last"):
        item = pair[role]
        try:
            snapshots[role] = Path(item["path"]).read_bytes()
        except OSError as error:
            raise MeasurementFailure("checkpoint_unreadable") from error
        actual = hashlib.sha256(snapshots[role]).hexdigest()
        measurement_require(actual == item["sha256"], "checkpoint_sha_mismatch")
    states = []
    for role in ("first", "last"):
        try:
            loaded = torch.load(io.BytesIO(snapshots[role]), map_location="cpu", weights_only=True)
        except Exception as error:
            raise MeasurementFailure("weights_only_load_failed") from error
        measurement_require(isinstance(loaded, dict) and isinstance(loaded.get("model_state_dict"), dict),
                            "model_state_dict_missing")
        states.append(loaded["model_state_dict"])
    first, last = states
    def change(prefix):
        keys = [key for key in first if key.startswith(prefix)]
        measurement_require(bool(keys), "parameter_prefix_missing")
        measurement_require(all(key in last and torch.is_tensor(first[key]) and torch.is_tensor(last[key])
            and first[key].layout == last[key].layout == torch.strided
            and not first[key].is_complex() and not last[key].is_complex()
            and first[key].shape == last[key].shape for key in keys), "parameter_structure_invalid")
        measurement_require(all(torch.isfinite(first[key]).all() and torch.isfinite(last[key]).all()
                                for key in keys), "parameter_nonfinite")
        delta = sum(float((last[key].float() - first[key].float()).norm() ** 2) for key in keys) ** 0.5
        norm = sum(float(first[key].float().norm() ** 2) for key in keys) ** 0.5
        value = delta / max(norm, 1e-12)
        measurement_require(math.isfinite(value), "parameter_change_nonfinite")
        return value
    result = {"critic_change": change("critic."), "actor_change": change("actor."),
              "from": Path(pair["first"]["path"]).name}
    measurement_require(not torch.cuda.is_initialized(), "cuda_initialized")
    return result


def prereg(plan: dict, context: dict, anchor: str) -> dict:
    return {"schema": "checkpoint_value_cpu_request_v1", **SCOPE, "context": context,
            "pairs": {role: plan["pairs"][key] for role, key in context["jobs"].items()},
            "plan_digest": L.digest(plan), "source_commit": anchor,
            "measurement_protocol_sha256_lf": sha(PROTOCOL, True), "execution_device": "cpu"}


def propose(plan_path: Path, output: Path) -> dict:
    anchor = verify_sources()
    plan = read(plan_path)
    require(plan == archive_plan(Path(plan["source_run"])), "CPU plan 또는 보관 입력 변경")
    output.mkdir(exist_ok=False)
    write_new(output / "plan.json", plan)
    ledger = M.RegistryLedger(output / "ledger.jsonl", C.registry())
    requests = []
    for context in plan["contexts"]:
        value = prereg(plan, context, anchor)
        request = ledger.propose(value, "P_value", 0)
        requests.append({"context": context["id"], "request_id": request["request_id"]})
    batch = {"schema": "checkpoint_value_cpu_batch_v1", "anchor": anchor,
             "plan_digest": L.digest(plan), "requests": requests}
    write_new(output / "batch.json", batch)
    return batch


def checked_batch(output: Path) -> tuple:
    batch, plan = read(output / "batch.json"), read(output / "plan.json")
    verify_sources(batch["anchor"])
    require(L.digest(plan) == batch["plan_digest"] and plan == archive_plan(Path(plan["source_run"])),
            "배치 plan·입력 변경")
    require([row["context"] for row in batch["requests"]] == [row["id"] for row in plan["contexts"]],
            "배치 요청 모집단 불일치")
    ledger = M.RegistryLedger(output / "ledger.jsonl", C.registry())
    requests = ledger.requests()
    require(len(requests) == len(batch["requests"]), "원장 요청 수 불일치")
    event_count = 0
    for context, row in zip(plan["contexts"], batch["requests"]):
        request = requests[row["request_id"]]
        require(request["prereg_digest"] == L.digest(prereg(plan, context, batch["anchor"]))
                and request["probe"] == "P_value"
                and request["budget_gpu_s"] == 0
                and request["args_digest"] == L.digest(C.registry()["probes"]["P_value"]["args"])
                and request["registry_digest"] == ledger.registry_digest, "원장 요청 결합 불일치")
        events = [event for event in ledger.events() if event.get("request_id") == row["request_id"]]
        order = {"pending": ["proposed"], "approved": ["proposed", "approved"],
                 "running": ["proposed", "approved", "consumed"],
                 "done": ["proposed", "approved", "consumed", "receipt"]}
        require([event["event"] for event in events] == order.get(request["state"]), "원장 승인·소비 순서 불일치")
        effect = ledger.effect_key(request["prereg_digest"], "P_value", request["args_digest"])
        for event in events:
            require(event.get("registry_digest") == ledger.registry_digest
                    and event.get("effect_key") == effect, "중간 원장 결합 불일치")
            if event["event"] != "approved":
                require(event.get("args_digest") == request["args_digest"], "중간 원장 인자 불일치")
            else:
                require(isinstance(event.get("approver"), str) and bool(event["approver"].strip()),
                        "중간 원장 승인 주체 누락")
        event_count += len(events)
    require(event_count == len(ledger.events()), "미집계 원장 이벤트")
    return batch, plan, ledger


def approve(output: Path, actor: str) -> dict:
    batch, _, ledger = checked_batch(output)
    require(bool(actor.strip()), "승인 주체 문자열 필요")
    require(all(row["state"] == "pending" for row in ledger.requests().values()), "대기 요청이 아님")
    for row in batch["requests"]:
        ledger.approve(row["request_id"], actor)
    return {"status": "approved", "requests": len(batch["requests"]), "identity_attested": False}


def execute(output: Path) -> dict:
    started = time.perf_counter()
    batch, plan, ledger = checked_batch(output)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU 실행은 CUDA_VISIBLE_DEVICES=-1 필요")
    require(not (output / "result.json").exists(), "기존 결과 재실행 거부")
    require(all(row["state"] == "approved" for row in ledger.requests().values()), "승인된 새 요청이 아님")
    cache, rows, active = {}, [], None
    context_id, pair_key, phase = None, None, "ledger_consume"
    result = {"schema": "checkpoint_value_cpu_result_v1", **SCOPE, "status": "invalid", "gpu_s": 0.0,
              "model_calls": 0, "device": "cpu", "new_training": False,
              "plan_digest": L.digest(plan), "source_commit": batch["anchor"], "contexts": rows,
              "interpretation": "봉인된 원래 체크포인트 쌍의 새 CPU 후속 측정. epoch 2는 실제 첫 교정 뒤 상태이며 대체 종단·선택 성능 비교가 아니다."}
    try:
        for context, row in zip(plan["contexts"], batch["requests"]):
            value = prereg(plan, context, batch["anchor"])
            active = row["request_id"]
            context_id, pair_key, phase = context["id"], None, "ledger_consume"
            ledger.consume(active, value, "P_value")
            measured = {}
            for role, key in context["jobs"].items():
                pair_key, phase = key, "checkpoint_measurement"
                if key not in cache:
                    cache[key] = load_pair(plan["pairs"][key])
                measured[role] = cache[key]
            pair_key, phase = context["jobs"]["reference"], "reference_check"
            measurement_require(measured["reference"]["critic_change"] > 0,
                                "reference_critic_change_not_positive")
            phase = "classification"
            outcome = L.classify("P_value", measured["case"], measured["reference"])
            phase = "receipt"
            receipt = ledger.receipt(active, outcome, measured, 0.0, 0)
            active = None
            rows.append({"context": context, "receipt": receipt})
        result["status"] = "complete"
        result["classification_counts"] = {outcome: sum(row["receipt"]["outcome"] == outcome for row in rows)
                                            for outcome in L.OUTCOMES}
    except Exception as error:
        result["error_type"] = type(error).__name__
        failure = {"failure_stage": error.stage if isinstance(error, MeasurementFailure) else phase,
                   "failure_code": error.code if isinstance(error, MeasurementFailure) else "unclassified_execution_error",
                   "failed_context_id": context_id, "failed_request_id": active, "failed_pair_key": pair_key}
        result["failure"] = failure
        if active and ledger.requests()[active]["state"] == "running":
            ledger.receipt(active, "unknown", {"failure": failure}, 0.0, 1, reason=failure["failure_code"])
        raise
    finally:
        result["execute_wall_s"] = time.perf_counter() - started
        result["execute_wall_scope"] = "execute_entry_including_validation_to_final_record"
        result["checkpoint_pairs_measured"] = len(cache)
        result["python_version"] = sys.version.split()[0]
        torch_module = sys.modules.get("torch")
        result["torch_version"] = str(torch_module.__version__) if torch_module else None
        result["cuda_initialized"] = torch_module.cuda.is_initialized() if torch_module else False
        write_new(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-protocol", "plan", "propose", "approve", "execute"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--actor")
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        write_new(PROTOCOL, protocol_value())
        result = {"status": "prepared_commit_required"}
    elif args.command == "plan":
        if not (args.source_run and args.output):
            parser.error("plan에는 --source-run과 --output이 필요합니다")
        result = archive_plan(args.source_run)
        write_new(args.output, result)
        result = {"status": "planned", "contexts": len(result["contexts"]), "pairs": len(result["pairs"])}
    elif args.command == "propose":
        if not (args.plan and args.output):
            parser.error("propose에는 --plan과 --output이 필요합니다")
        result = propose(args.plan, args.output)
    elif args.command == "approve":
        if not (args.output and args.actor):
            parser.error("approve에는 --output과 --actor가 필요합니다")
        result = approve(args.output, args.actor)
    else:
        if not args.output:
            parser.error("execute에는 --output이 필요합니다")
        result = execute(args.output)
    print(json.dumps({key: value for key, value in result.items() if key in ("status", "contexts", "pairs")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
