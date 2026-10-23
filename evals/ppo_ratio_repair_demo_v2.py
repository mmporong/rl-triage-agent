"""PPO ratio 계산 오류의 진단·승인 검사·ratio 식 복원·새 학습·고정 평가를 연결하는 종단 실행 판.

R1 종단 판 v2(r1_repair_demo_v2.py)와 같은 흐름이다. 바뀌는 것은 결함 arm(INVERSE_RATIO), 검사 등록(optimizer
surrogate 하위형과 P_ppo_gradient), 교정(ratio 식 복원)과 seed다. 사전등록의 예측·설계 결합은 파일럿 v2 결과로 정했다.
파일럿에서 P_torque가 하류 증상으로 abnormal이 나와 검사 등록은 계약 v2(P_torque의 surrogate 예상 null)를 쓴다.
v1의 모델 엔드포인트 404 기록은 보존한다. 사용자 선택에 따라 이 판부터 3.5 Super VL과 허용 반환 ID를 등록한다.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import ppo_ratio_pilot_v2 as PILOT  # noqa: E402
import ppo_ratio_repair_contract_v2 as C  # noqa: E402
import ppo_ratio_repair_execution_v2 as E  # noqa: E402
import ppo_ratio_repair_model_call_v2 as MODEL_CALL  # noqa: E402
from preregistered_seed_guard import assert_fresh_seeds  # noqa: E402

from rl_triage import grounded_recovery as G  # noqa: E402
from rl_triage import recovery as REC  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/ppo_ratio_repair_demo_v2.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_repair/runs"
SEED, RESTART_SEED = 2163, 3163
TASK = E.TASK
FAULT = "INVERSE_RATIO"
PROBE_BUDGET_S, CORRECTION_BUDGET_S = 1200, 3600
REQUEST_CAP = 8
TAG = re.compile(r"ppo_ratio_repair_s2163_[0-9]{8}[a-z0-9_]{0,16}")
NEW_FILES = (
    "evals/ppo_ratio_repair_demo_v2.py", "evals/ppo_ratio_repair_model_call_v2.py",
    "evals/ppo_ratio_repair_contract_v2.py", "bench/run_ppo_ratio_repair_demo_v2.ps1",
    "tests/test_ppo_ratio_repair_demo_v2.py", "tests/test_ppo_ratio_repair_contract_v2.py",
    "tests/test_ppo_ratio_repair_model_call_v2.py",
)
CODE = PILOT.CODE | set(NEW_FILES) | {
    "bench/protocols/ppo_ratio_pilot_v2.json", "evals/feedback_model_call.py", "evals/correction_model_call.py",
    "src/rl_triage/correction_ledger.py", "bench/reference/nvidia_skills/isaaclab-debugging-rl-training/SKILL.md",
    "evals/ppo_ratio_repair_demo_v1.py", "evals/ppo_ratio_repair_model_call.py",
    "bench/protocols/ppo_ratio_repair_demo_v1.json",
}
EVIDENCE = PILOT.EVIDENCE
# 파일럿 v2 결과(비공개)의 결합값. 사전등록 전에 본 결과이며 설계 결합으로 기록한다.
PILOT_RESULT = {"tag": "ppo_ratio_pilot_s2162_20261022",
                "result_sha256": "1ab5cab9d7b99c6ab4ecaa274ed73028a4b2af19fb83e0bc4c99dd2d81df5401",
                "experiment_commit": "e9445abd77138306a5d1c3fc6b2a5ecee9e06dc5"}
PILOT_OBSERVATIONS = {
    "source_gate": "failure_present: NONE 걷기, INVERSE_RATIO 낙상 0·생존 20초이나 선속도·yaw 추종 실패",
    "lin_vel_rmse_mps": {"NONE": 0.1455, "INVERSE_RATIO": 2.5907},
    "yaw_rate_rmse_radps": {"NONE": 0.2401, "INVERSE_RATIO": 1.6282},
    "outcomes": dict(C.PILOT_OUTCOMES),
    "P_torque_low_speed_saturation": {"case": 0.7455, "reference": 0.00067},
    "P_noise_logged_std_mean": {"case": 1.5936, "reference": 0.3660},
    "contract_v1_replay": "첫 검사 8개 중 7개 optimizer_surrogate 확정, P_torque 첫 검사는 actuator 오확정(2검사)",
    "contract_v2_replay": "첫 검사 8개 모두 optimizer_surrogate 확정, P_torque 첫 검사는 3검사",
}
SUCCESS_CONDITIONS = [
    "committed_source_and_protocol", "fresh_seeds", "launch_paths_checked", "source_failure_present",
    "accepted_fresh_diagnosis", "first_probe_from_model", "approved_probe_receipts",
    "optimizer_surrogate_confirmed", "accepted_fresh_correction", "approved_correction_consumed_once",
    "restored_surrogate_gradient", "corrected_fixed_eval_healthy", "costs_recorded",
]
OUTCOMES = {
    "invalid": "출처·실행·산출물 결합을 검증하지 못해 종단 결과를 판정하지 않는다.",
    "baseline_invalid": "같은 seed 정상 정책이 걷기 조건을 통과하지 못해 진단을 시작하지 않았다.",
    "source_failure_absent": "ratio 오류 정책이 정상 기준 대비 고정 평가를 통과해 진단할 실패가 없었다.",
    "model_infra_failed": "등록 호출 수 안에서 모델 응답을 확보하지 못해 멈췄다.",
    "model_invalid": "피드백 한 번 뒤에도 모델 출력 계약이 맞지 않아 멈췄다.",
    "diagnosis_hold": "모델이 첫 검사를 제안하지 않고 보류했다.",
    "loop_not_confirmed": "등록된 검사 관측으로 원인을 확정하지 못해 교정하지 않았다.",
    "correction_scope_hold": "확정된 가설이 ratio 식 복원 교정의 범위와 달라 교정하지 않았다.",
    "correction_not_proposed": "교정 제안이 보류되거나 등록 회복 방식과 달라 교정하지 않았다.",
    "correction_unknown": "승인된 교정 실행의 완결 또는 고정 평가를 검증하지 못했다.",
    "not_recovered": "ratio 식을 복원해 새로 학습했지만 등록한 정상 기준 대비 행동 허용 범위를 통과하지 못했다.",
    "recovered": ("새 seed의 ratio 계산 오류에서 승인 검사로 surrogate gradient 불일치를 확인하고, ratio 식을 복원해 "
                  "새로 학습한 정책이 고정 평가를 통과했다."),
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_protocol() -> dict:
    return {
        "schema": "ppo_ratio_repair_demo_protocol_v2", "seed": SEED, "restart_seed": RESTART_SEED,
        "task": TASK, "num_envs": 4096, "training_updates": 300, "fault": "inverse_ratio",
        "design_coupling": {
            "seen_before_design": ["PPO 계산 검사 v5(seed 2143~2146, 3 update)",
                                   "R1 종단 판 v2(seed 2071·3071)", "설계 파일럿 v2(seed 2162, 모델 호출 없음)",
                                   "본 실행 v1(seed 2161) source 평가: failure_present, 첫 진단 요청 404"],
            "pilot": dict(PILOT_RESULT),
            "pilot_observations": PILOT_OBSERVATIONS,
            "registry_change": ("파일럿에서 ratio 오류 정책의 큰 행동 분산이 저속 토크 포화를 만들어 P_torque가 abnormal이었다. "
                                "계약 v2는 P_torque의 optimizer_surrogate 예상만 null로 바꿨고 나머지 예상·임계값·고리 "
                                "규칙은 v1과 같다. 실제 actuator 관측표는 v2에서도 actuator로 확정된다"),
            "statement": "결함 유형과 검사를 이미 알고 설계한 새 seed의 개발 시연이다",
        },
        "question": ("갱신 횟수가 정상인 PPO ratio 계산 오류를 진단 고리가 surrogate gradient 검사로 확정하고, "
                     "ratio 식 복원과 새 학습으로 행동 회복까지 연결할 수 있는가"),
        "flow": ["launch_paths_preflight", "fresh_seed", "NONE_training", "INVERSE_RATIO_training",
                 "source_fixed_eval", "NONE_legacy_reference", "diagnosis_feedback", "approved_probe_loop",
                 "correction_feedback", "approved_ratio_restore_restart", "corrected_fixed_eval", "summary"],
        "loop_registry": C.registry(), "correction_registry": C.correction_registry(),
        "loop_rule": {"hypotheses": "six_categories_all_expanded_to_eight_subtypes", "max_probes": C.MAX_PROBES,
                      "first_probe": "model_choice_without_rule_replacement",
                      "later_probe": "mechanism_registry._next", "update": "mechanism_registry.update",
                      "confirmation": "mechanism_registry._status", "unknown": "stop_without_correction",
                      "legacy_classification": "probe_loop.classify_unchanged",
                      "scope_gate": "only_optimizer_surrogate_confirmed_by_P_ppo_gradient_abnormal"},
        "inspection": {"updates": 3, "epochs": 5, "mini_batches": 4, "expected_steps": 20,
                       "implementation": "ppo_gradient_invariant_v2.inspect_records",
                       "count_probe": "P_optimizer_updates는 optimizer_step_count·minibatch_count 판정을 읽는다",
                       "gradient_probe": "P_ppo_gradient는 ppo_surrogate_gradient 판정을 읽는다",
                       "relation": ("승인 뒤 같은 source/config/fault/seed로 별도 초기화한 3 update. 원 정책 체크포인트에서 "
                                    "읽은 관측이 아니며 checkpoint SHA는 계보 결합값이다"),
                       "control": "같은 seed NONE 검사도 실행하고 해당 판정이 normal이어야 한다",
                       "training_observer": "300회 학습의 첫 3회 기록은 완결 확인에만 쓴다"},
        "recovery_rule": {"required_subtype": C.SURROGATE, "mode": "restart", "correction": C.CORRECTION,
                          "updates": 300, "source_policy_loaded": False,
                          "reference": ("원 실패 seed의 NONE 정책을 기준으로 쓴다. source NONE/INVERSE_RATIO는 2163, "
                                        "교정 새 학습은 3163이다"),
                          "basis": ("같은 설정의 NONE 학습 근거로 새 초기화 경로를 시험한다. 오류 정책 이어 학습과 비교한 "
                                    "근거는 없다")},
        "model": {"id": MODEL_CALL.MODEL, "accepted_provider_models": sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
                  "path": "single_feedback", "feedback_retries": 1, "infra_dialogue_attempts": 2,
                  "total_http_request_cap": REQUEST_CAP, "nominal_requests": 2, "dialogue_wall_timeout_s": 900,
                  "invalid_diagnosis": "stop_no_rule_replacement", "invalid_correction": "stop_no_rule_replacement"},
        "infrastructure_retry": {
            "previous_commit": "2c7d441d197a12030a3334e0c74e4eebe37ffe7e",
            "previous_tag": "ppo_ratio_repair_s2161_20261022",
            "previous_protocol": "bench/protocols/ppo_ratio_repair_demo_v1.json",
            "previous_protocol_sha256_lf": "6c6074dfecbfd46655f6ab57c968170495e97347aad90e658c35213dee711abd",
            "previous_result_sha256": "4da4b78b2af825c569421862c641abcc6710fdf5325bf5ef40345a5692ebbe1c",
            "classification": "model_endpoint_not_found_before_diagnosis",
            "cause": "기존 3 Super 요청이 NotFoundError 404로 끝나 진단 결정을 확보하지 못했다",
            "previous_training_started": True, "previous_api_requests": 1,
            "previous_source_gate": "failure_present", "previous_probe_count": 0,
            "diagnostic_requests": 3,
            "diagnostic_evidence": "인계 기록에서 기존 3 Super 404, 3 Ultra와 3.5 Super VL 응답을 확인했다",
            "seed_policy": "2160·2161·2162·3161을 피하고 2163·3163을 새로 등록한다. 실행 직전 기록을 재검사한다",
            "preservation": "v1 소스·프로토콜·결과·학습과 API 소비는 보존하며 새 tag로 한 번만 실행한다",
            "interpretation": "v1 source 고정 평가를 이미 본 개발 시연이다. 모델 결정·검사·교정 결과를 본 뒤의 의미 재시도는 아니다",
            "model_change": "2026-10-23 사용자 선택으로 새 실행을 3.5 Super VL로 통일하며 이전 모델 결과와 합산하지 않는다",
            "unchanged": "고정 피드백·요청 상한·검사 registry·확정 규칙·교정·학습 예산·회복 판정은 v1과 같다",
        },
        "approval": {"mode": "operator_delegation_explicit_cli_flag", "identity_attested": False,
                     "probe": "registry digest + source checkpoint + protocol + canonical args",
                     "correction": "confirmed state + receipt + accepted model proposal + registry + restart"},
        "gpu": {"lease": "Use-GpuLease.ps1", "profile": E.GPU_PROFILE, "base_profile": E.BASE_PROFILE,
                "added_display_process_names": list(E.EXTRA_DISPLAY_PROCESS_NAMES), "sequential": True,
                "probe_execution_budget_s": PROBE_BUDGET_S, "correction_execution_budget_s": CORRECTION_BUDGET_S,
                "measurement_unit": "소유 프로세스 parent wall seconds; GPU kernel time이 아니다"},
        "failure_policy": {"gpu_attempts": 1, "overwrite": False, "semantic_retry": False,
                           "process_error": "보존 후 중단, 승인 소비 뒤 자동 재실행 없음",
                           "seed": "실행 직전 로그·저장소 재검사; 사용 흔적이면 중단하고 seed 자동 교체 없음"},
        "prediction": {"NONE": "walks", FAULT: "unhealthy", "loop": "optimizer_surrogate confirmed",
                       "P_torque": "abnormal 가능. 계약 v2에서 surrogate를 탈락시키지 않으며 고리를 혼자 닫지 않는다",
                       "corrected": "recovered",
                       "model_decisions": "순위·첫 검사·계약 통과 여부는 예측하지 않고 기록한다",
                       "competing": "모델 보류·무효, 다른 검사 abnormal·unknown, 정상 기준 실패, 교정 후 미회복은 등록 결과 그대로 남긴다"},
        "success_conditions": list(SUCCESS_CONDITIONS),
        "outcome_wording": dict(OUTCOMES),
        "interpretation": ("새 seed 단일 ratio 계산 오류의 등록 종단 경로 확인. gradient 검사 정상은 계산 계약만 뜻하며 "
                           "회복은 별도 고정 평가로만 판정한다"),
    }


def dynamic(isaaclab: Path) -> dict:
    runtime = PILOT.R1S.isaaclab_sources(isaaclab)
    return {"source_sha256_lf": {name: E.sha(ROOT / name, True) for name in sorted(CODE)},
            "evidence_sha256_lf": {name: E.sha(ROOT / name, True) for name in EVIDENCE},
            "runtime_source_sha256_lf": {row["path"]: row["sha256_lf"] for row in runtime.values()}}


def prepare(isaaclab: Path) -> dict:
    if not all(PILOT_RESULT.values()):
        raise ValueError("파일럿 결과 결합값을 정하기 전에는 사전등록을 만들지 않는다")
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


def source_gate(reports: dict, bands: dict) -> dict:
    healthy = REC.verdict(REC.compare(G._behavior(reports[FAULT]), G._behavior(reports["NONE"])), bands)
    walks = PILOT.R1S.U.walks(reports["NONE"])
    outcome = ("baseline_invalid" if not walks else "failure_present" if healthy["label"] == "unhealthy"
               else "source_failure_absent")
    return {"outcome": outcome, "none_walks": walks, "fault_verdict": healthy}


def diagnosis_packet(artifacts: dict) -> dict:
    facts = {}
    for arm, label in ((FAULT, "case"), ("NONE", "reference")):
        telemetry = read(Path(artifacts[arm]["telemetry"]["path"]))
        for name, row in sorted(telemetry["summary"].items()):
            if isinstance(row, dict) and "mean_last_20pct" in row:
                facts[f"telemetry:{label}:{name}"] = row["mean_last_20pct"]
    return {"schema": C.PACKET_SCHEMA, "facts": facts, "observed": [],
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
    if not any(x.endswith(b'/ppo_ratio_repair_model_call_v2.py') for x in args) or str(packet).encode() not in args: raise RuntimeError('worker args differ')
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
            argv = [wsl_path(ROOT / "evals/ppo_ratio_repair_model_call_v2.py"), "--kind", kind,
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


def _inspection_receipt(probe: str, controls: dict, source: str, evidence_sha: str) -> tuple[str, dict]:
    case, control = controls[FAULT]["measurement"], controls["NONE"]["measurement"]
    outcome = PILOT.inspection_outcome(probe, case, control)
    fields = ("count_status", "gradient_status", "first_mismatch_update", "optimizer_steps", "minibatches")
    return outcome, {"case": {key: case.get(key) for key in fields},
                     "control": {key: control.get(key) for key in fields},
                     "source_checkpoint_sha256": source, "checkpoint_loaded": False,
                     "execution_evidence_sha256": evidence_sha}


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
    result = {"schema": "ppo_ratio_repair_demo_result_v2", "tag": tag, "seed": SEED, "restart_seed": RESTART_SEED,
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
        for arm in ("NONE", FAULT):
            verify(isaaclab, anchor)
            artifacts[arm] = backend.train(folder / "training" / arm.lower(), isaaclab, arm, SEED,
                                          f"{tag}_{arm.lower()}", binding(protocol, anchor))
            E.write_new(folder / f"artifact_{arm.lower()}.json", artifacts[arm])
            result["costs"][f"train_{arm}"] = _duration(artifacts[arm])
        source = artifacts[FAULT]["checkpoint"]["sha256"]
        stage = "source_evaluation"
        evaluation = backend.fixed_evaluate(folder / "source_evaluation", isaaclab, artifacts, tag, binding(protocol, anchor))
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
            if probe in (C.COUNT_PROBE, C.GRADIENT_PROBE):
                controls = {}
                for arm in ("NONE", FAULT):
                    controls[arm] = backend.inspect(probe_stage / arm.lower(), isaaclab, arm, SEED,
                        f"{tag}_{probe.lower()}_{arm.lower()}",
                        binding(protocol, anchor, artifacts[arm]["checkpoint"]["sha256"], 3))
                E.write_new(probe_stage / "evidence.json", controls)
                outcome, measurement = _inspection_receipt(probe, controls, source, E.sha(probe_stage / "evidence.json"))
                seconds = sum(_duration(row) for row in controls.values())
            else:
                raw = backend.legacy_probe(probe_stage, isaaclab, {FAULT: artifacts[FAULT]},
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
        conditions["optimizer_surrogate_confirmed"] = True
        stage = "correction_model"
        confirming = next(row for row in result["probe_steps"] if row["probe"] == C.GRADIENT_PROBE)
        packet = C.correction_packet(state, confirming, reports[FAULT], source,
                                     E.sha(Path(evaluation["report_paths"][FAULT])))
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
            C.CORRECTION, "restart", {"case": "optimizer_surrogate_ratio", "arm": "corrected"}, CORRECTION_BUDGET_S,
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
        observed = corrected["inspection"]
        conditions["restored_surrogate_gradient"] = (
            observed.get("count_status") == "normal" and observed.get("gradient_status") == "normal"
            and observed.get("optimizer_steps") == [20, 20, 20] and observed.get("minibatches") == [20, 20, 20])
        if not conditions["restored_surrogate_gradient"]:
            raise ValueError("교정 뒤 surrogate gradient 계산이 복원되지 않았다")
        stage = "correction_evaluation"
        repaired_eval = backend.fixed_evaluate(folder / stage, isaaclab, {"CORRECTED": corrected},
                                               tag + "_fixed", binding(protocol, anchor, source))
        result["costs"][stage] = _duration(repaired_eval)
        repaired = PILOT.validated_reports(repaired_eval, {"CORRECTED": corrected})["CORRECTED"]
        if repaired["versions"] != reports["NONE"]["versions"]:
            raise ValueError("교정 평가의 런타임 버전이 정상 기준과 다르다")
        verdict = REC.verdict(REC.compare(G._behavior(repaired), G._behavior(reports["NONE"])), bands)
        outcome = "recovered" if verdict["label"] == "healthy" else "not_recovered"
        E.write_new(folder / "recovery.json", {"outcome": outcome, "verdict": verdict,
            "source_metrics": reports[FAULT]["metrics"], "reference_metrics": reports["NONE"]["metrics"],
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
    if requests > REQUEST_CAP:
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
        result["conditions"].get(key) is True for key in SUCCESS_CONDITIONS)
    result["wording"] = OUTCOMES[result["outcome"]]
    result["design_coupling"] = fixed_protocol()["design_coupling"]
    E.write_new(folder / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "PPO ratio 종단 실행").splitlines()[0])
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
        value = {"status": "ok", "committed_gate": args.command == "verify", "anchor": anchor,
                 "launch_paths": E.preflight(args.isaaclab), "gpu": E.query_gpu()}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED, RESTART_SEED], args.isaaclab, repo=ROOT)
    else:
        value = execute(args.tag or "", args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
