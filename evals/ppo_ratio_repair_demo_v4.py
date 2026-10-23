"""C3의 학습 뒤 외부 GPU 점유 차단을 보존하고 새 seed로 한 번만 실행하는 C4 판."""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import ppo_ratio_repair_demo_v3 as V3  # noqa: E402
from preregistered_seed_guard import assert_fresh_seeds  # noqa: E402

C = V3.C
E = V3.E
MODEL_CALL = V3.MODEL_CALL
PILOT = V3.PILOT
G = V3.G
REC = V3.REC
CorrectionLedger = V3.CorrectionLedger

PROTOCOL = ROOT / "bench/protocols/ppo_ratio_repair_demo_v4.json"
PRIVATE = V3.PRIVATE
SEED, RESTART_SEED = 2165, 3165
TASK = V3.TASK
FAULT = V3.FAULT
PROBE_BUDGET_S, CORRECTION_BUDGET_S = V3.PROBE_BUDGET_S, V3.CORRECTION_BUDGET_S
REQUEST_CAP = 4
TAG = re.compile(r"ppo_ratio_repair_s2165_[0-9]{8}[a-z0-9_]{0,16}")
NEW_FILES = {
    "evals/ppo_ratio_repair_demo_v4.py",
    "tests/test_ppo_ratio_repair_demo_v4.py",
    "bench/run_ppo_ratio_repair_demo_v4.ps1",
}
CODE = set(V3.CODE) | NEW_FILES | {
    "evals/ppo_ratio_repair_demo_v3.py",
    "bench/protocols/ppo_ratio_repair_demo_v3.json",
}
EVIDENCE = V3.EVIDENCE
SUCCESS_CONDITIONS = list(V3.SUCCESS_CONDITIONS)
OUTCOMES = dict(V3.OUTCOMES)
C3_RESULT = {
    "tag": "ppo_ratio_repair_s2164_20261023",
    "result_sha256": "58e166e51c3750d8c52f6d1602d3efef97b6a2b9f5ed2458751c497e16c85c61",
    "experiment_commit": "03c04a5c750dd8a0434a7d9dbfa550158e3da1eb",
    "protocol_sha256_lf": "194e0f551dd32f51124f938ed4f2df2cb7ad2768d87c04799790db560dee0864",
    "classification": "post_NONE_training_external_Kit_busy",
    "outcome": "invalid",
    "completed": ["NONE_training_300_updates"],
    "not_started": ["INVERSE_RATIO_training", "source_fixed_eval", "diagnosis", "probe_loop", "correction"],
    "seed_consumption": {"source": 2164, "source_consumed": True, "restart": 3164, "restart_reserved": True},
    "observed_cost": {
        "known_owned_child_wall_s_sum": 609.094,
        "components": {"training_parent_wall_s": 603.125, "checkpoint_parent_wall_s": 3.937,
                       "telemetry_extract_parent_wall_s": 2.032},
        "measurement": "소유 자식 실행 세 건의 부모 측정 벽시계 시간 합계이며 GPU 커널 시간이 아니다",
        "api_requests": 0, "fixed_evaluations": 0,
    },
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_protocol() -> dict:
    protocol = copy.deepcopy(V3.fixed_protocol())
    protocol.update(schema="ppo_ratio_repair_demo_protocol_v4", seed=SEED, restart_seed=RESTART_SEED)
    protocol["design_coupling"]["seen_before_design"].append(
        "본 실행 v3(seed 2164) NONE 300회 종료 뒤 GPU 사후 점검 invalid"
    )
    protocol["design_coupling"]["c3_observation"] = copy.deepcopy(C3_RESULT)
    protocol["recovery_rule"]["reference"] = (
        "원 실패 seed의 NONE 정책을 기준으로 쓴다. source NONE/INVERSE_RATIO는 2165, "
        "교정 새 학습은 3165이다"
    )
    protocol["c4_retry"] = {
        "previous_protocol": "bench/protocols/ppo_ratio_repair_demo_v3.json",
        "previous_result_sha256": C3_RESULT["result_sha256"],
        "previous_commit": C3_RESULT["experiment_commit"],
        "previous_protocol_sha256_lf": C3_RESULT["protocol_sha256_lf"],
        "classification": C3_RESULT["classification"],
        "completed_before_invalid": list(C3_RESULT["completed"]),
        "not_started": list(C3_RESULT["not_started"]),
        "previous_cost": copy.deepcopy(C3_RESULT["observed_cost"]),
        "seed_policy": (
            "v3 source 2164는 사용했고 restart 3164는 예약된 것으로 보존한다. "
            "2165·3165를 실행 직전 재검사하며 사용 흔적이면 자동 교체하지 않는다"
        ),
        "new_tag_and_seeds": "seed 2165, restart 3165; 이전 실행 폴더를 덮어쓰지 않는다",
        "start_gate": "FactoryOps 완료 ACK 뒤 standalone strict GPU clear를 확인하고 lease 안에서 실행한다",
        "api_ledger": {
            "v1": 1,
            "c2": 2,
            "c3": 0,
            "c4_cap": REQUEST_CAP,
            "execution_attempts_cap": 7,
            "prior_diagnostic_requests_separate": 3,
        },
        "dialogues": (
            "진단 1개와 교정 1개. 각 대화의 고정 semantic feedback 1회까지이며 "
            "별도 인프라 대화 재시도는 없다"
        ),
        "preservation": (
            "v3 결과·정상 300회 학습·seed 소비·부분 벽시계 시간·API 0·평가 0을 보존한다"
        ),
        "unchanged": (
            "모델 계약·검사 registry·확정 규칙·교정·허용 GPU profile·학습 예산·회복 판정은 v3와 같다"
        ),
        "interpretation": (
            "C3는 고정 평가 전에 외부 GPU 점유 검증으로 중단되어 의미 결과를 보지 않은 인프라 재시도다"
        ),
    }
    return protocol


def dynamic(isaaclab: Path) -> dict:
    runtime = PILOT.R1S.isaaclab_sources(isaaclab)
    return {
        "source_sha256_lf": {name: E.sha(ROOT / name, True) for name in sorted(CODE)},
        "evidence_sha256_lf": {name: E.sha(ROOT / name, True) for name in EVIDENCE},
        "runtime_source_sha256_lf": {row["path"]: row["sha256_lf"] for row in runtime.values()},
    }


def prepare(isaaclab: Path) -> dict:
    E.write_new(PROTOCOL, {**fixed_protocol(), **dynamic(isaaclab)})
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    ).stdout.strip()


def verify(isaaclab: Path, anchor: str | None = None, *, committed: bool = True) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    if protocol != {**fixed_protocol(), **dynamic(isaaclab)}:
        raise ValueError("사전등록 고정값·소스·입력·런타임 결합이 다르다")
    head = git("rev-parse", "HEAD")
    anchor = anchor or head
    if committed:
        paths = sorted(CODE | set(EVIDENCE) | {PROTOCOL.relative_to(ROOT).as_posix()})
        git("merge-base", "--is-ancestor", anchor, head)
        git("ls-files", "--error-unmatch", "--", *paths)
        if git("status", "--porcelain", "--", *paths):
            raise ValueError("실행 전에 결합 파일을 커밋해야 한다")
        for path in paths:
            if git("rev-parse", f"{anchor}:{path}") != git("rev-parse", f"{head}:{path}"):
                raise ValueError("실험 commit 뒤 결합 blob이 바뀌었다")
    return protocol, anchor


def binding(protocol: dict, anchor: str, checkpoint: str | None = None, updates: int = 300) -> dict:
    value = V3.binding(protocol, anchor, checkpoint, updates)
    value["protocol_path"] = PROTOCOL.relative_to(ROOT).as_posix()
    value["protocol_sha256_lf"] = E.sha(PROTOCOL, True)
    return value


def _duration(result: dict) -> float:
    return V3._duration(result)


def _inspection_receipt(probe: str, controls: dict, source: str, evidence_sha: str) -> tuple[str, dict]:
    return V3._inspection_receipt(probe, controls, source, evidence_sha)


def source_gate(reports: dict, bands: dict) -> dict:
    return V3.source_gate(reports, bands)


def diagnosis_packet(artifacts: dict) -> dict:
    return V3.diagnosis_packet(artifacts)


def model_decision(folder: Path, kind: str, packet: dict, caller=None) -> dict:
    return V3.model_decision(folder, kind, packet, caller)


def execute(
    tag: str,
    isaaclab: Path,
    approver: str,
    approve_registered_actions: bool,
    *,
    backend: Any = E,
    caller=None,
    seed_guard=assert_fresh_seeds,
) -> dict:
    if not TAG.fullmatch(tag) or not isinstance(approver, str) or not approver.strip() or not approve_registered_actions:
        raise ValueError("등록 tag·승인자·등록 행동 위임 플래그가 필요하다")
    protocol, anchor = verify(isaaclab)
    folder = PRIVATE / tag
    if folder.exists():
        raise FileExistsError("이미 시작한 tag이며 실행을 반복하지 않는다")
    launch_paths = backend.preflight(isaaclab)
    freshness = seed_guard([SEED, RESTART_SEED], isaaclab, repo=ROOT)
    folder.mkdir(parents=True, exist_ok=False)
    E.write_new(folder / "seed_freshness.json", freshness)
    E.write_new(folder / "launch_paths.json", launch_paths)
    E.write_new(
        folder / "begin.json",
        {
            "tag": tag,
            "experiment_commit": anchor,
            "protocol_sha256_lf": E.sha(PROTOCOL, True),
            "approver": approver,
            "identity_attested": False,
            "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임",
        },
    )
    result = {
        "schema": "ppo_ratio_repair_demo_result_v4",
        "tag": tag,
        "seed": SEED,
        "restart_seed": RESTART_SEED,
        "experiment_commit": anchor,
        "protocol_sha256_lf": E.sha(PROTOCOL, True),
        "conditions": {},
        "outcome": "invalid",
        "model": {},
        "probe_steps": [],
        "costs": {},
    }
    conditions = result["conditions"]
    conditions.update(committed_source_and_protocol=True, fresh_seeds=True, launch_paths_checked=True)
    probe_ledger = C.M.RegistryLedger(folder / "probes.ledger.jsonl", C.registry())
    correction_ledger = CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    active_probe = active_correction = None
    stage = "source_training"
    try:
        artifacts = {}
        for arm in ("NONE", FAULT):
            verify(isaaclab, anchor)
            artifacts[arm] = backend.train(
                folder / "training" / arm.lower(), isaaclab, arm, SEED, f"{tag}_{arm.lower()}", binding(protocol, anchor)
            )
            E.write_new(folder / f"artifact_{arm.lower()}.json", artifacts[arm])
            result["costs"][f"train_{arm}"] = _duration(artifacts[arm])
        source = artifacts[FAULT]["checkpoint"]["sha256"]
        stage = "source_evaluation"
        evaluation = backend.fixed_evaluate(
            folder / "source_evaluation", isaaclab, artifacts, tag, binding(protocol, anchor)
        )
        result["costs"][stage] = _duration(evaluation)
        reports = PILOT.validated_reports(evaluation, artifacts)
        bands = read(ROOT / PILOT.P1.BANDS_EVIDENCE)["bands"]
        gate = source_gate(reports, bands)
        E.write_new(folder / "source_gate.json", gate)
        result["source_gate"] = gate
        if gate["outcome"] != "failure_present":
            result["outcome"] = gate["outcome"]
            return finish(folder, result)
        conditions["source_failure_present"] = True
        stage = "reference"
        reference = backend.legacy_probe(
            folder / "reference", isaaclab, {"NONE": artifacts["NONE"]}, tag + "_ref", binding(protocol, anchor)
        )
        result["costs"][stage] = _duration(reference)
        stage = "diagnosis"
        diagnosis = model_decision(folder / "model", stage, diagnosis_packet(artifacts), caller)
        result["model"][stage] = diagnosis
        review = diagnosis["review"]
        if review["status"] != "valid":
            result["outcome"] = "model_invalid" if review["status"] == "invalid" else "model_infra_failed"
            return finish(folder, result)
        decision = review["decision"]
        if decision["action"] != "continue":
            result["outcome"] = "diagnosis_hold"
            return finish(folder, result)
        conditions["accepted_fresh_diagnosis"] = True
        state = C.initial_state(decision)
        prereg = {
            "checkpoint_sha256": source,
            "seed": SEED,
            "protocol_sha256_lf": E.sha(PROTOCOL, True),
            "experiment_commit": anchor,
            "registry_digest": C.M.registry_digest(C.registry()),
        }
        while state["status"] == "open":
            verify(isaaclab, anchor)
            probe = state["next_probe"]
            stage = f"probe_{len(state['observed']) + 1}_{probe}"
            request = probe_ledger.propose(prereg, probe, PROBE_BUDGET_S)
            active_probe = request["request_id"]
            probe_ledger.approve(active_probe, approver)
            probe_ledger.consume(active_probe, prereg, probe)
            probe_stage = folder / stage
            if probe in (C.COUNT_PROBE, C.GRADIENT_PROBE):
                controls = {}
                for arm in ("NONE", FAULT):
                    controls[arm] = backend.inspect(
                        probe_stage / arm.lower(), isaaclab, arm, SEED,
                        f"{tag}_{probe.lower()}_{arm.lower()}",
                        binding(protocol, anchor, artifacts[arm]["checkpoint"]["sha256"], 3),
                    )
                E.write_new(probe_stage / "evidence.json", controls)
                outcome, measurement = _inspection_receipt(
                    probe, controls, source, E.sha(probe_stage / "evidence.json")
                )
                seconds = sum(_duration(row) for row in controls.values())
            else:
                raw = backend.legacy_probe(
                    probe_stage, isaaclab, {FAULT: artifacts[FAULT]}, tag + "_" + probe.lower(),
                    binding(protocol, anchor), probes=[probe],
                )
                item = raw["report"]["probes"][probe]
                measurement = item["measurement"] or {}
                ref = reference["report"]["probes"][probe]
                outcome = C.L.classify(
                    probe, measurement if item["exit"] == 0 else None,
                    ref["measurement"] if ref["exit"] == 0 else None,
                )
                seconds = _duration(raw)
            receipt = probe_ledger.receipt(active_probe, outcome, measurement, seconds, 0)
            receipt["probe"] = probe
            active_probe = None
            result["probe_steps"].append(receipt)
            result["costs"][stage] = seconds
            state = C.advance(state, probe, receipt["outcome"])
            E.write_new(folder / f"loop_{len(state['observed']):02d}.json", state)
        result["loop"] = state
        conditions["first_probe_from_model"] = result["probe_steps"][0]["probe"] == decision["probe"]
        conditions["approved_probe_receipts"] = all(r["state"] == "done" for r in probe_ledger.requests().values())
        if state["status"] != "confirmed":
            result["outcome"] = "loop_not_confirmed"
            return finish(folder, result)
        if not C.correction_allowed(state):
            result["outcome"] = "correction_scope_hold"
            return finish(folder, result)
        conditions["optimizer_surrogate_confirmed"] = True
        stage = "correction_model"
        confirming = next(row for row in result["probe_steps"] if row["probe"] == C.GRADIENT_PROBE)
        packet = C.correction_packet(
            state, confirming, reports[FAULT], source, E.sha(Path(evaluation["report_paths"][FAULT]))
        )
        correction = model_decision(folder / "model", "correction", packet, caller)
        result["model"]["correction"] = correction
        reviewed = correction["review"]
        if reviewed["status"] != "valid":
            result["outcome"] = "model_invalid" if reviewed["status"] == "invalid" else "model_infra_failed"
            return finish(folder, result)
        proposed = reviewed["decision"]
        if proposed["decision"] != "propose" or not proposed["agrees_with_rule"]:
            result["outcome"] = "correction_not_proposed"
            return finish(folder, result)
        conditions["accepted_fresh_correction"] = True
        verify(isaaclab, anchor)
        request = correction_ledger.propose(
            C.L.digest(state), "optimizer", C.L.digest(result["probe_steps"]), C.CORRECTION, "restart",
            {"case": "optimizer_surrogate_ratio", "arm": "corrected"}, CORRECTION_BUDGET_S,
            evidence={
                "packet_digest": C.L.digest(packet),
                "model_review_digest": C.L.digest(correction),
                "protocol_sha256_lf": E.sha(PROTOCOL, True),
                "checkpoint_sha256": source,
            },
        )
        active_correction = request["request_id"]
        correction_ledger.approve(active_correction, approver)
        correction_ledger.consume(active_correction, C.L.digest(state))
        stage = "correction_training"
        corrected = backend.train(
            folder / stage, isaaclab, "CORRECTED", RESTART_SEED, tag + "_corrected",
            binding(protocol, anchor, source),
        )
        E.write_new(folder / "artifact_corrected.json", corrected)
        result["costs"][stage] = _duration(corrected)
        observed = corrected["inspection"]
        conditions["restored_surrogate_gradient"] = (
            observed.get("count_status") == "normal"
            and observed.get("gradient_status") == "normal"
            and observed.get("optimizer_steps") == [20, 20, 20]
            and observed.get("minibatches") == [20, 20, 20]
        )
        if not conditions["restored_surrogate_gradient"]:
            raise ValueError("교정 뒤 surrogate gradient 계산이 복원되지 않았다")
        stage = "correction_evaluation"
        repaired_eval = backend.fixed_evaluate(
            folder / stage, isaaclab, {"CORRECTED": corrected}, tag + "_fixed", binding(protocol, anchor, source)
        )
        result["costs"][stage] = _duration(repaired_eval)
        repaired = PILOT.validated_reports(repaired_eval, {"CORRECTED": corrected})["CORRECTED"]
        if repaired["versions"] != reports["NONE"]["versions"]:
            raise ValueError("교정 평가의 런타임 버전이 정상 기준과 다르다")
        verdict = REC.verdict(REC.compare(G._behavior(repaired), G._behavior(reports["NONE"])), bands)
        outcome = "recovered" if verdict["label"] == "healthy" else "not_recovered"
        E.write_new(
            folder / "recovery.json",
            {
                "outcome": outcome,
                "verdict": verdict,
                "source_metrics": reports[FAULT]["metrics"],
                "reference_metrics": reports["NONE"]["metrics"],
                "corrected_metrics": repaired["metrics"],
                "checkpoint_sha256": corrected["checkpoint"]["sha256"],
            },
        )
        receipt = correction_ledger.receipt(
            active_correction, tag, outcome, E.sha(folder / "recovery.json"),
            result["costs"]["correction_training"] + result["costs"]["correction_evaluation"], 0,
        )
        active_correction = None
        conditions["approved_correction_consumed_once"] = True
        conditions["corrected_fixed_eval_healthy"] = outcome == "recovered"
        result["outcome"] = "correction_unknown" if receipt["verdict"] == "unknown" else outcome
        result["recovery"] = read(folder / "recovery.json")
        verify(isaaclab, anchor)
        return finish(folder, result)
    except Exception as error:
        if active_probe and probe_ledger.requests()[active_probe]["state"] == "running":
            probe_ledger.receipt(active_probe, "unknown", {}, 0.0, 1, reason="execution_or_validation_failed")
        if active_correction and correction_ledger.requests()[active_correction]["state"] == "running":
            correction_ledger.receipt(
                active_correction, tag, "unknown", None, None, 1, reason="execution_or_validation_failed"
            )
        result["outcome"] = (
            "correction_unknown" if active_correction else "model_infra_failed" if isinstance(error, TimeoutError) else "invalid"
        )
        trace = error.__traceback__
        while trace and trace.tb_next:
            trace = trace.tb_next
        site = None
        if trace:
            file = Path(trace.tb_frame.f_code.co_filename).resolve()
            site = {
                "file": file.relative_to(ROOT).as_posix() if file.is_relative_to(ROOT) else file.name,
                "line": trace.tb_lineno,
            }
        result["failure"] = {"stage": stage, "type": type(error).__name__, "site": site}
        finish(folder, result)
        raise


def finish(folder: Path, result: dict) -> dict:
    records = [read(path) for path in sorted((folder / "model").glob("*.feedback[12].json"))]
    calls = [call for record in records for call in record["calls"]]
    requests = sum(record["requests"] for record in records)
    if requests > REQUEST_CAP:
        raise ValueError("모델 요청 상한을 넘었다")
    cost_ok = all(
        type(call.get("input_tokens")) is int
        and type(call.get("output_tokens")) is int
        and isinstance(call.get("wall_s"), (int, float))
        for call in calls
    )
    pending = [
        path.name
        for path in (folder / "model").glob("*.reserved.json")
        if not path.with_name(path.name.replace(".reserved.json", ".json")).is_file()
    ]
    transport_errors = sorted(path.name for path in (folder / "model").glob("*.transport_error.json"))
    result["api_requests"] = requests if not pending and not transport_errors else None
    result["api_requests_in_saved_records"] = requests
    result["pending_model_records"] = pending
    result["transport_errors"] = transport_errors
    result["conditions"]["costs_recorded"] = cost_ok and bool(result["costs"]) and not pending and not transport_errors
    result["demo_success"] = result["outcome"] == "recovered" and all(
        result["conditions"].get(key) is True for key in SUCCESS_CONDITIONS
    )
    result["wording"] = OUTCOMES[result["outcome"]]
    result["design_coupling"] = fixed_protocol()["design_coupling"]
    E.write_new(folder / "result.json", result)
    return result


def gpu_gate(backend: Any = E) -> dict:
    receipt = backend.query_gpu()
    if receipt.get("busy") is not False:
        raise RuntimeError("standalone strict GPU clear를 확인하지 못했다")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "PPO ratio C4 실행").splitlines()[0])
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "gpu-gate", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    parser.add_argument("--approver")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.isaaclab)
    elif args.command in ("check", "verify"):
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {
            "status": "ok",
            "committed_gate": args.command == "verify",
            "anchor": anchor,
            "launch_paths": E.preflight(args.isaaclab),
            "runtime_gpu_gate": "required_before_execute",
        }
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
    elif args.command == "gpu-gate":
        value = gpu_gate()
    else:
        value = execute(args.tag or "", args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
