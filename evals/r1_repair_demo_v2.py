"""R1 진단·승인 검사·epoch 복원·새 학습·고정 평가를 연결하는 새 시연 판."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))
import r1_repair_contract as C
import r1_repair_execution_v2 as E
import r1_repair_demo as V1
import r1_repair_model_call as MODEL_CALL
import run_upstream_r1_smoke as R1
from preregistered_seed_guard import assert_fresh_seeds
from rl_triage import grounded_recovery as G
from rl_triage import recovery as REC
from rl_triage import tracking_alarm as ALARM
from rl_triage.correction_ledger import CorrectionLedger

PROTOCOL = ROOT / "bench/protocols/r1_repair_demo_v2.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task21/runs"
SEED, RESTART_SEED = 2071, 3071
TASK = E.TASK
PROBE_BUDGET_S, CORRECTION_BUDGET_S = 1200, 3600
TAG = re.compile(r"r1_repair_s2071_[0-9]{8}[a-z0-9_]{0,16}")
NEW_FILES = (
    "evals/r1_repair_demo_v2.py", "evals/r1_repair_execution_v2.py",
    "bench/run_r1_repair_demo_v2.ps1", "tests/test_r1_repair_execution_v2.py",
    "tests/test_r1_repair_demo_v2.py",
)
CODE = set(V1.CODE) | set(NEW_FILES)
EVIDENCE = (*V1.EVIDENCE, "bench/protocols/r1_repair_demo_v1.json")

OUTCOMES = {
    "invalid": "출처·실행·산출물 결합을 검증하지 못해 종단 결과를 판정하지 않는다.",
    "baseline_invalid": "같은 seed 정상 정책이 걷기 조건을 통과하지 못해 진단을 시작하지 않았다.",
    "source_failure_absent": "원 결함 정책이 정상 기준 대비 고정 평가를 통과해 진단할 실패가 없었다.",
    "model_infra_failed": "등록 호출 수 안에서 모델 응답을 확보하지 못해 멈췄다.",
    "model_invalid": "피드백 한 번 뒤에도 모델 출력 계약이 맞지 않아 멈췄다.",
    "diagnosis_hold": "모델이 첫 검사를 제안하지 않고 보류했다.",
    "loop_not_confirmed": "등록된 검사 관측으로 원인을 확정하지 못해 교정하지 않았다.",
    "correction_scope_hold": "확정된 가설이 epoch 복원 교정의 범위와 달라 교정하지 않았다.",
    "correction_not_proposed": "교정 제안이 보류되거나 등록 회복 방식과 달라 교정하지 않았다.",
    "correction_unknown": "승인된 교정 실행의 완결 또는 고정 평가를 검증하지 못했다.",
    "not_recovered": "epoch 복원 후 새로 학습했지만 등록한 정상 기준 대비 행동 허용 범위를 통과하지 못했다.",
    "recovered": "새 seed R1에서 승인 검사로 갱신 횟수 결함을 확인하고 epoch 복원 후 새로 학습한 정책이 고정 평가를 통과했다.",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_protocol() -> dict:
    value = deepcopy(V1.fixed_protocol())
    value.update(schema="r1_repair_demo_protocol_v2", seed=SEED, restart_seed=RESTART_SEED)
    value["recovery_rule"]["reference"] = (
        "v4 reference_for(restart)와 같이 원 실패 seed의 NONE 정책을 사용한다. "
        "source NONE/R1은 2071, 교정 새 학습은 3071이다. 교정과 기준의 seed는 같지 않다")
    value["flow"].insert(0, "launch_paths_preflight")
    value["success_conditions"].append("launch_paths_checked")
    value["launch_contract"] = {
        "preflight": "Isaac root·python.bat·train.py·로그 폴더·등록 실행 파일을 CPU에서 확인",
        "owned_leaf": "모든 owned 실행 leaf를 호출 전에 배타 생성; 기존 leaf나 부모 부재는 거부",
        "backend": "v1 backend에 동일 준비 runner를 주입; 학습·계측·checkpoint·추출·평가·probe 모두 적용",
    }
    value["infrastructure_retry"] = {
        "previous_commit": "b9e4d5e943819209f10ec438afb81ba0bcf0d2d7",
        "previous_tag": "r1_repair_s2070_20261016",
        "previous_protocol": "bench/protocols/r1_repair_demo_v1.json",
        "previous_protocol_sha256_lf": "bc727461894d3e271f68374b90147b43d12c82562f4b8412676faee7f0553ddf",
        "previous_result_sha256": "143e438b73be462ed7a4e4be0581a78df62acc67200e53bf39f6f95e1491b570",
        "classification": "prelaunch_infrastructure_invalid",
        "cause": "owned 실행 leaf가 없는 상태에서 launch.json 생성이 거부됐다. 학습 프로세스 시작 전 종료다",
        "previous_training_started": False, "previous_api_requests": 0,
        "seed_policy": "2070·3070은 학습 없이 등록 시도 기록에 남아 기존 seed guard가 차단한다. 예외 없이 2071·3071을 새로 등록하고 실행 직전 재검사한다",
        "preservation": "v1 소스·프로토콜·invalid 산출물은 보존하며 이 판은 새 tag로 한 번만 실행한다",
        "interpretation": "학습·검사·고정 평가 결과를 본 뒤의 재시도가 아니다. v1 invalid를 R1 관측 결과로 세지 않는다",
        "unchanged": "모델·피드백·HTTP 상한·검사 registry·확정 규칙·교정·학습 예산·회복 판정은 v1과 동일",
    }
    return value


def dynamic(isaaclab: Path) -> dict:
    runtime = R1.isaaclab_sources(isaaclab)
    return {"source_sha256_lf": {name: E.sha(ROOT / name, True) for name in sorted(CODE)},
            "evidence_sha256_lf": {name: E.sha(ROOT / name, True) for name in EVIDENCE},
            "runtime_source_sha256_lf": {row["path"]: row["sha256_lf"] for row in runtime.values()}}


def prepare(isaaclab: Path) -> dict:
    E.write_new(PROTOCOL, {**fixed_protocol(), **dynamic(isaaclab)})
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout.strip()


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
    return {"source_commit_sha": anchor, "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256_lf": E.sha(PROTOCOL, True), "source_sha256_lf": protocol["source_sha256_lf"],
            "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"], "task": TASK,
            "num_envs": 4096, "max_iterations": updates, "configured_learning_epochs": 5,
            "configured_mini_batches": 4, "runtime_updates_to_observe": 3,
            "source_checkpoint_sha256": checkpoint}


def validated_reports(evaluation: dict, artifacts: dict) -> dict:
    reports = {}
    for arm, artifact in artifacts.items():
        report = evaluation["reports"][arm]
        reports[arm] = G.validate_fixed_report(report, {
            "name": report["name"], "checkpoint_file": "model_299.pt",
            "checkpoint_sha256": artifact["checkpoint"]["sha256"], "action_scale": 0.25,
            "protocol_sha256_lf": E.sha(E.FIXED_EVAL_PROTOCOL, True)})
    if len({C.L.digest(r["versions"]) for r in reports.values()}) != 1:
        raise ValueError("고정 평가 런타임 버전이 서로 다르다")
    return reports


def source_gate(reports: dict, bands: dict) -> dict:
    healthy = REC.verdict(REC.compare(G._behavior(reports["R1"]), G._behavior(reports["NONE"])), bands)
    walks = R1.U.walks(reports["NONE"])
    outcome = "baseline_invalid" if not walks else "failure_present" if healthy["label"] == "unhealthy" else "source_failure_absent"
    return {"outcome": outcome, "none_walks": walks, "fault_verdict": healthy}


def diagnosis_packet(artifacts: dict) -> dict:
    facts = {}
    for arm, label in (("R1", "case"), ("NONE", "reference")):
        telemetry = read(Path(artifacts[arm]["telemetry"]["path"]))
        for name, row in sorted(telemetry["summary"].items()):
            if isinstance(row, dict) and "mean_last_20pct" in row:
                facts[f"telemetry:{label}:{name}"] = row["mean_last_20pct"]
    return {"schema": "r1_repair_diagnosis_packet_v1", "facts": facts, "observed": [],
            "budget_left": C.MAX_PROBES, "execution_budget_s": PROBE_BUDGET_S}


def wsl_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", value):
        raise ValueError("모델 호출에는 Windows 저장소 경로가 필요하다")
    return f"/mnt/{value[0].lower()}/{value[3:]}"


def stop_model_worker(pid_file: Path, packet_path: Path) -> str:
    """해당 호출의 PID·생성 시점·인자가 일치할 때만 WSL worker를 닫는다."""
    if not pid_file.is_file():
        return "unverified_no_pid"
    cleanup = '''import hashlib,json,os,signal,sys,time
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text()); packet=Path(sys.argv[2]); pid=record['pid']
assert type(pid) is int and pid>1
assert hashlib.sha256(packet.read_bytes()).hexdigest()==record['packet_sha256']
proc=Path('/proc')/str(pid)
def same():
    if not (proc/'stat').exists(): return False
    if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=record['start_ticks']: raise RuntimeError('pid identity changed')
    args=(proc/'cmdline').read_bytes().split(bytes([0]))
    if not any(x.endswith(b'/r1_repair_model_call.py') for x in args) or str(packet).encode() not in args: raise RuntimeError('worker args differ')
    return True
if same():
    os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        if not proc.exists(): break
        time.sleep(0.1)
    if same(): os.kill(pid,signal.SIGKILL)
    for _ in range(30):
        if not proc.exists(): break
        time.sleep(0.1)
    if proc.exists(): raise RuntimeError('worker remains')
'''
    command = '"$HOME/.venvs/rl-triage/bin/python" -c ' + shlex.join([cleanup, wsl_path(pid_file), wsl_path(packet_path)])
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], check=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
    return "verified"


def model_decision(folder: Path, kind: str, packet: dict, caller=None) -> dict:
    """기록 이름은 고정. 인프라 오류만 최대 두 대화이며 대화당 최대 두 요청이다."""
    folder.mkdir(parents=True, exist_ok=True)
    packet_path = folder / f"{kind}.packet.json"
    E.write_new(packet_path, packet)
    frozen_data = packet_path.read_bytes()
    results = []
    for attempt in (1, 2):
        if packet_path.read_bytes() != frozen_data:
            raise ValueError("호출 사이에 모델 packet이 바뀌었다")
        output = folder / f"{kind}.feedback{attempt}.json"
        if caller is None:
            pid_file = output.with_suffix(".pid.json")
            argv = [wsl_path(ROOT / "evals/r1_repair_model_call.py"), "--kind", kind,
                    "--packet", wsl_path(packet_path), "--output", wsl_path(output), "--pid-file", wsl_path(pid_file)]
            command = '. "$HOME/.config/nvidia/env" && exec "$HOME/.venvs/rl-triage/bin/python" -B ' + shlex.join(argv)
            # API 요청별 timeout=300, SDK 재시도0. 쉘도 Python으로 교체되어 남는 worker를 만들지 않는다.
            try:
                process = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
            except subprocess.TimeoutExpired:
                receipt = {"type": "dialogue_timeout", "worker_cleanup": "unverified"}
                try:
                    receipt["worker_cleanup"] = stop_model_worker(pid_file, packet_path)
                except Exception as exc:
                    receipt["cleanup_error_type"] = type(exc).__name__
                E.write_new(folder / f"{kind}.transport_error.json", receipt)
                raise TimeoutError("모델 대화 시간 상한 초과") from None
            if not output.is_file():
                raise RuntimeError(f"모델 호출 기록 없음(exit={process.returncode})")
        else:
            reserved = MODEL_CALL.reservation_for(kind, frozen_data)
            E.write_new(output.with_suffix(".reserved.json"), reserved)
            mocked = caller(kind, frozen_data)
            mocked["reservation_sha256"] = C.L.digest(reserved)
            E.write_new(output, mocked)
        raw = read(output)
        if packet_path.read_bytes() != frozen_data:
            raise ValueError("모델 호출 중 packet이 바뀌었다")
        verdict = MODEL_CALL.review(raw, frozen_data, kind, read(output.with_suffix(".reserved.json")))
        row = {"record": output.name, "record_sha256": E.sha(output), "review": verdict,
               "calls": [{k: c.get(k) for k in ("provider_model", "input_tokens", "output_tokens", "total_tokens", "wall_s")}
                         for c in raw["calls"]]}
        results.append(row)
        E.write_new(folder / f"{kind}.review{attempt}.json", row)
        if verdict["status"] != "infra_error":
            return {"review": verdict, "attempts": results}
    return {"review": verdict, "attempts": results}


def _duration(result: dict) -> float:
    receipt = result["execution"]
    for key in ("wall_s", "parent_wall_s", "elapsed_s"):
        if isinstance(receipt.get(key), (int, float)):
            return float(receipt[key])
    raise ValueError("실행 receipt에 벽시계 시간이 없다")


def execute(tag: str, isaaclab: Path, approver: str, approve_registered_actions: bool,
            *, backend=E, caller=None, seed_guard=assert_fresh_seeds) -> dict:
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
    E.write_new(folder / "begin.json", {"tag": tag, "experiment_commit": anchor,
        "protocol_sha256_lf": E.sha(PROTOCOL, True), "approver": approver, "identity_attested": False,
        "delegation": "승인된 등록 검사와 교정을 순서대로 실행하는 운영자 위임"})
    result = {"schema": "r1_repair_demo_result_v2", "tag": tag, "seed": SEED, "restart_seed": RESTART_SEED,
              "experiment_commit": anchor, "protocol_sha256_lf": E.sha(PROTOCOL, True), "conditions": {},
              "outcome": "invalid", "model": {}, "probe_steps": [], "costs": {}}
    conditions = result["conditions"]
    conditions.update(committed_source_and_protocol=True, fresh_seeds=True, launch_paths_checked=True)
    probe_ledger = C.M.RegistryLedger(folder / "probes.ledger.jsonl", C.registry())
    correction_ledger = CorrectionLedger(folder / "correction.ledger.jsonl", C.correction_registry())
    active_probe = active_correction = None
    stage = "source_training"
    try:
        artifacts = {}
        for arm in ("NONE", "R1"):
            verify(isaaclab, anchor)
            artifacts[arm] = backend.train(folder / "training" / arm.lower(), isaaclab, arm, SEED,
                                          f"{tag}_{arm.lower()}", binding(protocol, anchor))
            E.write_new(folder / f"artifact_{arm.lower()}.json", artifacts[arm])
            result["costs"][f"train_{arm}"] = _duration(artifacts[arm])
        source = artifacts["R1"]["checkpoint"]["sha256"]
        metric = ALARM.METRIC
        series = {arm: read(Path(value["telemetry"]["path"]))["series"][metric] for arm, value in artifacts.items()}
        E.write_new(folder / "alarm.json", {arm: ALARM.r1(row, [series["NONE"]]) for arm, row in series.items()})
        stage = "source_evaluation"
        evaluation = backend.fixed_evaluate(folder / "source_evaluation", isaaclab, artifacts, tag, binding(protocol, anchor))
        result["costs"][stage] = _duration(evaluation)
        reports = validated_reports(evaluation, artifacts)
        bands = read(ROOT / EVIDENCE[4])["bands"]
        gate = source_gate(reports, bands)
        E.write_new(folder / "source_gate.json", gate)
        result["source_gate"] = gate
        if gate["outcome"] != "failure_present":
            result["outcome"] = gate["outcome"]
            return finish(folder, result)
        conditions["source_failure_present"] = True
        stage = "reference"
        reference = backend.legacy_probe(folder / "reference", isaaclab, {"NONE": artifacts["NONE"]},
                                         tag + "_ref", binding(protocol, anchor))
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
        prereg = {"checkpoint_sha256": source, "seed": SEED, "protocol_sha256_lf": E.sha(PROTOCOL, True),
                  "experiment_commit": anchor, "registry_digest": C.M.registry_digest(C.registry())}
        while state["status"] == "open":
            verify(isaaclab, anchor)
            probe = state["next_probe"]
            stage = f"probe_{len(state['observed']) + 1}_{probe}"
            request = probe_ledger.propose(prereg, probe, PROBE_BUDGET_S)
            active_probe = request["request_id"]
            probe_ledger.approve(active_probe, approver)
            probe_ledger.consume(active_probe, prereg, probe)
            probe_stage = folder / stage
            if probe == C.NEW_PROBE:
                controls = {}
                for arm in ("NONE", "R1"):
                    controls[arm] = backend.inspect(probe_stage / arm.lower(), isaaclab, arm, SEED,
                        f"{tag}_inspect_{arm.lower()}", binding(protocol, anchor, artifacts[arm]["checkpoint"]["sha256"], 3))
                measured = controls["R1"]["measurement"]
                outcome = measured["status"] if controls["NONE"]["measurement"]["status"] == "normal" else "unknown"
                seconds = sum(_duration(row) for row in controls.values())
                E.write_new(probe_stage / "evidence.json", controls)
                fields = ("status", "optimizer_steps", "minibatches")
                measurement = {"case": {key: measured[key] for key in fields},
                               "control": {key: controls["NONE"]["measurement"][key] for key in fields},
                               "source_checkpoint_sha256": source, "checkpoint_loaded": False,
                               "execution_evidence_sha256": E.sha(probe_stage / "evidence.json")}
            else:
                raw = backend.legacy_probe(probe_stage, isaaclab, {"R1": artifacts["R1"]},
                                          tag + "_" + probe.lower(), binding(protocol, anchor), probes=[probe])
                item = raw["report"]["probes"][probe]
                measurement = item["measurement"] or {}
                ref = reference["report"]["probes"][probe]
                outcome = C.L.classify(probe, measurement if item["exit"] == 0 else None,
                                      ref["measurement"] if ref["exit"] == 0 else None)
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
        conditions["optimizer_update_count_confirmed"] = True
        stage = "correction_model"
        confirming = next(row for row in result["probe_steps"] if row["probe"] == C.NEW_PROBE)
        packet = C.correction_packet(state, confirming, reports["R1"], source,
                                     E.sha(Path(evaluation["report_paths"]["R1"])))
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
        request = correction_ledger.propose(C.L.digest(state), "optimizer", C.L.digest(result["probe_steps"]),
            C.CORRECTION, "restart", {"case": "optimizer_epoch_count", "arm": "corrected"}, CORRECTION_BUDGET_S,
            evidence={"packet_digest": C.L.digest(packet), "model_review_digest": C.L.digest(correction),
                      "protocol_sha256_lf": E.sha(PROTOCOL, True), "checkpoint_sha256": source})
        active_correction = request["request_id"]
        correction_ledger.approve(active_correction, approver)
        correction_ledger.consume(active_correction, C.L.digest(state))
        stage = "correction_training"
        corrected = backend.train(folder / stage, isaaclab, "CORRECTED", RESTART_SEED, tag + "_corrected",
                                  binding(protocol, anchor, source))
        E.write_new(folder / "artifact_corrected.json", corrected)
        result["costs"][stage] = _duration(corrected)
        conditions["restored_update_count"] = corrected["inspection"] == {
            "status": "normal", "optimizer_steps": [20, 20, 20], "minibatches": [20, 20, 20]}
        if not conditions["restored_update_count"]:
            raise ValueError("교정 뒤 실제 갱신 횟수가 복원되지 않았다")
        stage = "correction_evaluation"
        repaired_eval = backend.fixed_evaluate(folder / stage, isaaclab, {"CORRECTED": corrected},
                                               tag + "_fixed", binding(protocol, anchor, source))
        result["costs"][stage] = _duration(repaired_eval)
        repaired = validated_reports(repaired_eval, {"CORRECTED": corrected})["CORRECTED"]
        if repaired["versions"] != reports["NONE"]["versions"]:
            raise ValueError("교정 평가의 런타임 버전이 정상 기준과 다르다")
        verdict = REC.verdict(REC.compare(G._behavior(repaired), G._behavior(reports["NONE"])), bands)
        outcome = "recovered" if verdict["label"] == "healthy" else "not_recovered"
        E.write_new(folder / "recovery.json", {"outcome": outcome, "verdict": verdict,
            "source_metrics": reports["R1"]["metrics"], "reference_metrics": reports["NONE"]["metrics"],
            "corrected_metrics": repaired["metrics"], "checkpoint_sha256": corrected["checkpoint"]["sha256"]})
        receipt = correction_ledger.receipt(active_correction, tag, outcome, E.sha(folder / "recovery.json"),
            result["costs"]["correction_training"] + result["costs"]["correction_evaluation"], 0)
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
            correction_ledger.receipt(active_correction, tag, "unknown", None, None, 1, reason="execution_or_validation_failed")
        result["outcome"] = ("correction_unknown" if active_correction else
                             "model_infra_failed" if isinstance(error, TimeoutError) else "invalid")
        # 오류 문자열은 private 기록에도 자격증명을 포함하지 않도록 예외 종류만 둔다.
        trace = error.__traceback__
        while trace and trace.tb_next:
            trace = trace.tb_next
        site = None
        if trace:
            file = Path(trace.tb_frame.f_code.co_filename).resolve()
            site = {"file": file.relative_to(ROOT).as_posix() if file.is_relative_to(ROOT) else file.name,
                    "line": trace.tb_lineno}
        result["failure"] = {"stage": stage, "type": type(error).__name__, "site": site}
        finish(folder, result)
        raise


def finish(folder: Path, result: dict) -> dict:
    # 검토에서 거절된 기록도 이미 사용한 API 요청이므로 원시 파일에서 집계한다.
    records = [read(path) for path in sorted((folder / "model").glob("*.feedback[12].json"))]
    calls = [call for record in records for call in record["calls"]]
    requests = sum(record["requests"] for record in records)
    if requests > 8:
        raise ValueError("모델 요청 상한을 넘었다")
    cost_ok = all(type(call.get("input_tokens")) is int and type(call.get("output_tokens")) is int
                  and isinstance(call.get("wall_s"), (int, float)) for call in calls)
    pending = [path.name for path in (folder / "model").glob("*.reserved.json")
               if not path.with_name(path.name.replace(".reserved.json", ".json")).is_file()]
    transport_errors = sorted(path.name for path in (folder / "model").glob("*.transport_error.json"))
    result["api_requests"] = requests if not pending and not transport_errors else None
    result["api_requests_in_saved_records"] = requests
    result["pending_model_records"] = pending
    result["transport_errors"] = transport_errors
    result["conditions"]["costs_recorded"] = cost_ok and bool(result["costs"]) and not pending and not transport_errors
    result["demo_success"] = result["outcome"] == "recovered" and all(
        result["conditions"].get(key) is True for key in fixed_protocol()["success_conditions"])
    result["wording"] = OUTCOMES[result["outcome"]]
    result["design_coupling"] = fixed_protocol()["design_coupling"]
    E.write_new(folder / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    parser.add_argument("--approver")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.isaaclab)
    elif args.command in ("check", "verify"):
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "ok", "committed_gate": args.command == "verify", "anchor": anchor, "launch_paths": E.preflight(args.isaaclab)}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
    else:
        value = execute(args.tag or "", args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
