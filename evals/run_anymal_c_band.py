"""두 번째 로봇 task 단계 1: ANYmal-C flat 정상 seed 2개를 더 학습해 단계 0 정상 실행과 함께 P0-C 방식 밴드를 만들고,
비 기준 probe(P_value·P_torque)의 seed 간 기준과 추종 경고 R1의 오경보를 확인한다.

단계 0(evals/run_anymal_c_gate.py, 결과 evals/results/anymal_c_gate_20261014, pass)과 같은 학습 진입·고정 평가·probe 구동기를
쓴다. 정상 실행 셋(seed 2043·2044·2045)의 순서쌍 6개로 밴드를 정하고, 같은 순서쌍으로 비 기준 probe를 동결 classify에 넣고,
실행마다 다른 두 실행을 참조로 R1을 돌린다. 기준 하나라도 어긋나면 단계 2(F1 종단 시연)로 가지 않는다. Go2 수치와 비교하지 않는다.

  python evals/run_anymal_c_band.py prepare
  python evals/run_anymal_c_band.py verify
  python evals/run_anymal_c_band.py execute --tag anymal_c_band_<날짜>     # GPU: seed 2044·2045 차례로 학습·평가·probe, 끝나면 요약
  python evals/run_anymal_c_band.py retry --tag <같은 tag> --seed 2044      # 그 seed가 인프라 원인 invalid였을 때 한 번
  python evals/run_anymal_c_band.py summarize --tag <같은 tag>              # 두 seed가 끝났을 때(execute가 자동으로 부른다)
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from rl_triage import grounded_recovery as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import recovery as RC  # noqa: E402
from rl_triage import tracking_alarm as TA  # noqa: E402
import run_anymal_c_gate as G  # noqa: E402

U, FE = G.U, G.FE
PROTOCOL = ROOT / "bench/protocols/anymal_c_band_v1.json"
STAGE0_PROTOCOL = ROOT / "bench/protocols/anymal_c_gate_v1.json"
STAGE0_TAG = "anymal_c_gate_20261014"
STAGE0_SEED = 2043
STAGE0_RUN = "anymal_c_gate_none_s2043"
STAGE0_FILES = {"result": f"evals/results/{STAGE0_TAG}/result.json",
                "report": f"evals/results/{STAGE0_TAG}/runs/{STAGE0_RUN}.json",
                "probe_report": f"evals/results/{STAGE0_TAG}/probes/{STAGE0_RUN}.json",
                "telemetry": f"bench/runs_anymal/{STAGE0_RUN}.telemetry.json"}
PRIVATE = ROOT / "bench/private/anymal_c_band"
RESULTS = ROOT / "evals/results"
SEEDS = (2044, 2045)
ALL_SEEDS = (STAGE0_SEED, *SEEDS)
TAG_RE = r"anymal_c_band_[0-9]{8}[a-z0-9_]{0,10}"
CODE = tuple(dict.fromkeys(("evals/run_anymal_c_band.py", *G.CODE, "src/rl_triage/tracking_alarm.py",
                            "bench/protocols/anymal_c_gate_v1.json")))
FIXED = {
    "schema": "second_task_band_protocol_v1",
    "question": "ANYmal-C flat 정상 실행 셋(seed 2043·2044·2045)으로 P0-C 방식 밴드를 만들 수 있고, 동결 비 기준 probe와 추종 경고 R1이 "
                "정상 실행끼리 오경보를 내지 않는가",
    "scope": "두 번째 로봇 task 단계 1. 정상 실행 셋의 변동 기록이며 진단 성능·일반화·Go2와의 수치 비교가 아니다",
    "stage": {"index": 1, "previous": "단계 0 pass(evals/results/anymal_c_gate_20261014/result.json, 5ea5557). 그 정상 실행(seed 2043)을 셋 중 "
                                      "하나로 쓴다",
              "next": "pass일 때만 단계 2(보상 좌표 결함 F1 종단 시연)를 따로 사전등록한다. 단계 2의 판정 밴드·R1 참조·probe 기준은 이 단계 "
                      "결과를 쓴다"},
    "task": G.TASK, "seeds": list(SEEDS), "stage0_seed": STAGE0_SEED, "num_envs": G.NUM_ENVS, "max_iterations": G.MAX_ITERATIONS,
    "action_scale": G.ACTION_SCALE,
    "procedure": "seed마다 단계 0과 같다: bench/train_with_fault.py(TRIAGE_FAULT=NONE) 학습 → 텔레메트리 추출 → evals/fixed_eval_task.py "
                 "고정 평가(fixed_eval_v1_anymal_c, 평가 환경 push_robot·base_com 끔) → evals/probes_task.py probe 6종(task 등록 학습 설정). "
                 "유효 조건도 단계 0과 같다(seed만 다름). seed 2044 다음 2045 순서",
    "bands": {"method": "P0-C와 같다: 정상 실행 셋의 순서쌍 6개(실행 i 대 다른 seed 실행 j)를 recovery.compare로 비교하고 "
                        "recovery.derive_bands(여유 비 0.10, 낙상 0.05)로 정한다",
              "use": "단계 2 판정에 쓴다. 이 단계의 통과 기준이 아니라 기록이다(정상끼리는 정의상 밴드 안이다)"},
    "ratio_probes": {"method": "P_value·P_torque 측정을 순서쌍 6개마다 probe_loop.classify(probe, 실행 i 측정, 실행 j 측정)에 넣는다(12개). "
                                "동결 임계값: P_value는 기준의 10% 미만이면 abnormal, P_torque는 3 × max(기준, 0.01) 초과면 abnormal",
                     "use": "단계 2에서 같은 seed 정상 실행을 기준으로 쓰기 전에, 정상 실행끼리 abnormal이 나오지 않는지 본다"},
    "tracking_alarm": {"method": "rl_triage.tracking_alarm.r1(실행 i의 Metrics/base_velocity/error_vel_xy, 다른 두 정상 실행의 같은 시계열). "
                                 "규칙(20회 평균, 60회 이후, 2배, 10회 연속)은 동결 상수 그대로",
                       "use": "단계 2의 경고 단계 참조로 쓰기 전에 정상 실행끼리 울리지 않는지 본다"},
    "criteria": {
        "new_runs_walk": "seed 2044·2045 정상 정책이 고정 평가에서 걷는다(정지 대비 오차 비 < 0.5, 낙상 ≤ 0.05)",
        "absolute_probes_normal": "seed 2044·2045의 절대 기준 probe 4종이 normal",
        "ratio_probes_cross_seed_normal": "비 기준 probe 순서쌍 12개가 모두 normal",
        "no_tracking_false_alarm": "정상 실행 셋 모두에서 R1이 울리지 않는다",
    },
    "outcomes": {"pass": "두 seed가 유효하고 기준 네 개를 모두 채움 → 단계 2를 사전등록한다",
                 "fail": "유효하지만 기준 하나 이상이 어긋남 → 멈춘다. 단계 2로 가지 않는다",
                 "invalid": "어느 seed든 유효 조건을 채우지 못한 채 끝남(재시도 뒤 포함). 판정하지 않는다"},
    "stop_rule": "fail이면 이 task로 단계 2를 하지 않는다. 임계값·R1 규칙·밴드 여유·seed를 바꿔 다시 시도하지 않는다. 다음 결정은 사용자가 한다",
    "validity": "seed마다 단계 0의 validity_checks 8개(파이프라인·종료 코드·체크포인트 둘·[hidden_fault] NONE·params·[eval_env]·고정 평가 "
                "보고서·probe 보고서 결합). 단계 0 입력은 결과 pass·프로토콜 SHA·보고서·텔레메트리 SHA가 결합값과 같아야 쓴다",
    "execution": {
        "watchdog_s": G.WATCHDOG_S, "gpu_admission_profile": U.GPU_PROFILE,
        "seed_unused": "seed마다 실행 직전에 단계 0과 같은 범위(저장소 bench/runs_*·bench/private·evals/results·evals/loops, Isaac Lab 로그 "
                       "전체)에서 흔적을 찾는다. 시도 1은 흔적이 없어야 하고 시도 2는 그 seed 시도 1 실행 이름만 허용한다",
        "retry_rule": "seed마다 시도 1이 인프라 원인 invalid면 retry로 새 실행 이름(_a2)으로 한 번 더 한다. 그 밖 원인의 invalid나 시도 2의 "
                      "invalid는 그 seed의 최종이다",
        "summary_rule": "두 seed가 모두 최종이 되면 요약을 한 번 쓴다(공개 result.json). 다시 쓰지 않는다",
    },
    "prediction": {
        "outcome": "pass",
        "basis": {"ratio_probes": "단계 0 정상 실행의 P_value critic 변화 1.778, P_torque 저속 포화 2.1e-6이다. seed 간 critic 변화 비는 1 근처, "
                                  "포화는 바닥값 0.01의 3배(0.03)보다 훨씬 작을 것으로 본다",
                  "tracking_alarm": "정상 실행끼리 추종 오차가 2배 차이 나는 구간이 10회 이어지지 않을 것으로 본다",
                  "walk": "단계 0 정상 실행이 걸었다(정지 대비 0.155, 낙상 0.013)"},
        "risk": "seed에 따라 300회에서 덜 걷는 실행이 나오면 new_runs_walk가 어긋나고, 학습 초반 오차 차이로 R1이 울릴 수 있다",
    },
    "no_tuning_rule": "결과를 본 뒤 기준·임계값·R1 상수·밴드 여유·seed·평가 환경을 바꾸지 않는다",
    "wording": {
        "pass": "ANYmal-C flat 정상 실행 셋(새 seed 셋)이 모두 걸었고, 동결 비 기준 probe와 추종 경고 R1이 정상 실행끼리 오경보를 내지 않았다. "
                "이 셋으로 P0-C 방식 밴드를 정했다. 단계 2로 간다. 정상 실행 변동 기록이며 진단 성능·일반화가 아니다.",
        "fail": "ANYmal-C flat 단계 1의 사전등록 기준 일부를 채우지 못해 단계 2로 가지 않는다. 기준·임계값을 바꿔 다시 시도하지 않는다.",
        "invalid": "단계 1 실행이 사전등록한 유효 조건을 채우지 못해 판정하지 않는다.",
    },
}
DYNAMIC = {"source_files", "isaaclab_sources", "eval_protocol_sha256_lf", "stage0_inputs"}


# --- 결합 ---
def stage0_inputs() -> dict:
    """단계 0 결과(pass)와 그 보고서·텔레메트리. 결과의 프로토콜 SHA·체크포인트·텔레메트리 SHA가 파일과 맞아야 쓴다."""
    result = U.read(ROOT / STAGE0_FILES["result"])
    if result.get("outcome") != "pass" or result.get("protocol_sha256_lf") != U.sha(STAGE0_PROTOCOL, True):
        raise ValueError("단계 0 결과가 pass가 아니거나 단계 0 프로토콜과 맞지 않는다")
    if result.get("seed") != STAGE0_SEED or result.get("run_name") != STAGE0_RUN:
        raise ValueError("단계 0 결과의 seed·실행 이름이 등록값과 다르다")
    if U.sha(ROOT / STAGE0_FILES["telemetry"]) != result.get("telemetry_sha256"):
        raise ValueError("단계 0 텔레메트리가 결과에 기록된 것과 다르다")
    report, probe = U.read(ROOT / STAGE0_FILES["report"]), U.read(ROOT / STAGE0_FILES["probe_report"])
    if (report.get("checkpoint") or {}).get("sha256") != result["checkpoint_sha256"] or \
            (probe.get("checkpoint") or {}).get("sha256") != result["checkpoint_sha256"]:
        raise ValueError("단계 0 보고서의 체크포인트가 결과와 다르다")
    return {key: {"path": path, "sha256_lf": U.sha(ROOT / path, True)} for key, path in STAGE0_FILES.items()}


def dynamic_fields(isaaclab: Path) -> dict:
    return {"source_files": {name: U.sha(ROOT / name, True) for name in CODE},
            "isaaclab_sources": G.isaaclab_sources(isaaclab), "eval_protocol_sha256_lf": U.sha(G.EVAL_PROTOCOL, True),
            "stage0_inputs": stage0_inputs()}


def check_protocol(protocol: dict, dynamic: dict) -> None:
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("프로토콜 필드가 다르다")
    if any(protocol[k] != v for k, v in FIXED.items()):
        raise ValueError("고정 계약이 바뀌었다")
    if any(protocol[k] != v for k, v in dynamic.items()):
        raise ValueError("결합한 코드·Isaac Lab 소스·평가 프로토콜·단계 0 입력이 바뀌었다")


def prepare(isaaclab: Path) -> dict:
    U.write_new(PROTOCOL, {**deepcopy(FIXED), **dynamic_fields(isaaclab)})
    return {"status": "prepared_commit_before_execute", "path": PROTOCOL.relative_to(ROOT).as_posix()}


def verify(isaaclab: Path, verify_committed=None) -> tuple[dict, str]:
    protocol = U.read(PROTOCOL)
    check_protocol(protocol, dynamic_fields(isaaclab))
    if verify_committed is None:
        import run_runtime_readback_smoke as P
        verify_committed = P.verify_committed
    paths = set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix(), *STAGE0_FILES.values()}
    try:
        commit = verify_committed(ROOT, paths)
    except subprocess.CalledProcessError as error:
        raise ValueError("결합 코드·프로토콜·단계 0 결과가 커밋되지 않았다") from error
    return protocol, commit


# --- seed별 실행(단계 0과 같은 절차, seed만 다름) ---
def run_name(seed: int, attempt: int) -> str:
    return f"anymal_c_band_none_s{seed}" + ("" if attempt == 1 else f"_a{attempt}")


def params_expected(params: dict | None, seed: int) -> bool:
    if not isinstance(params, dict):
        return False
    expected = {"agent.yaml:seed": seed, "env.yaml:seed": seed, "env.yaml:scene.num_envs": G.NUM_ENVS,
                "agent.yaml:max_iterations": G.MAX_ITERATIONS, "env.yaml:actions.joint_pos.scale": G.ACTION_SCALE}
    return all(params.get(key) == value for key, value in expected.items())


def probe_bound(report: dict | None, evidence: dict, seed: int) -> bool:
    return (isinstance(report, dict) and report.get("name") == evidence["run_name"] and report.get("task") == G.TASK
            and report.get("seed") == seed and (report.get("checkpoint") or {}).get("sha256") == evidence.get("checkpoint_sha256")
            and set(report.get("probes") or {}) == set(L.PROBES))


def training_checks(evidence: dict, seed: int) -> dict[str, bool]:
    return {"training_exit_zero": evidence.get("train_exit_code") == 0,
            "final_checkpoint_present": bool(evidence.get("checkpoint_sha256")) and evidence.get("first_checkpoint") is True,
            "training_readback": G.readback_ok(evidence.get("readback")),
            "params_expected": params_expected(evidence.get("params"), seed)}


def judge_seed(protocol: dict, evidence: dict, seed: int) -> dict:
    """seed 하나의 유효 조건(단계 0 validity_checks와 같은 이름·순서·원인)."""
    stage0 = U.read(STAGE0_PROTOCOL)
    checks = {"pipeline_completed": evidence.get("failure") is None, **training_checks(evidence, seed),
              "eval_env_readback": G.eval_env_ok(evidence.get("eval_env")),
              "fixed_report_valid": G.report_valid(evidence.get("report"), evidence, protocol),
              "probe_report_bound": probe_bound(evidence.get("probe_report"), evidence, seed)}
    checks = {name: checks[name] for name in stage0["validity_checks"]}
    failed = [name for name, ok in checks.items() if not ok]
    cause = None
    if failed:
        cause = stage0["invalid_causes"]["by_check"][failed[0]]
        stage = (evidence.get("failure") or {}).get("stage")
        failed_training = [name for name in training_checks(evidence, seed) if name in failed]
        if failed[0] == "pipeline_completed" and stage not in (None, "training") and failed_training:
            cause = stage0["invalid_causes"]["by_check"][failed_training[0]]
    return {"checks": checks, "failed_checks": failed, "valid": not failed, "invalid_cause": cause}


def seed_evidence(isaaclab: Path, seed: int, allowed: set[str]) -> dict:
    """단계 0 seed_evidence와 같은 범위(저장소 산출물 폴더와 Isaac Lab 로그 전체)에서 그 seed 흔적을 찾는다."""
    logs = Path(isaaclab) / "logs/rsl_rl"
    roots = {"bench/runs_anymal": G.RUNS_DIR, "bench/runs_p0c": ROOT / "bench/runs_p0c", "bench/runs_d3": ROOT / "bench/runs_d3",
             "bench/runs_upstream": ROOT / "bench/runs_upstream", "bench/private": ROOT / "bench/private",
             "evals/results": RESULTS, "evals/loops": ROOT / "evals/loops"}
    hits = sorted(p.name for root in roots.values() if root.exists() for p in root.iterdir() if f"_s{seed}" in p.name)
    if logs.exists():
        for experiment in (e for e in logs.iterdir() if e.is_dir()):
            hits += [f"{experiment.name}/{run.name}" for run in experiment.iterdir() if f"_s{seed}" in run.name]
            for agent in experiment.glob("*/params/agent.yaml"):
                match = re.search(r"^seed: (\d+)\s*$", agent.read_text(encoding="utf-8", errors="replace"), re.M)
                if match and int(match[1]) == seed:
                    hits.append(f"{experiment.name}/{agent.parent.parent.name}: seed {seed}")
    own = sorted({h for h in hits if any(name in h for name in allowed)})
    return {"seed": seed, "checked_roots": sorted(roots) + ["isaaclab_logs/*"], "hits": sorted(set(hits) - set(own)),
            "allowed_hits": own}


def train_command(isaaclab: Path, name: str, seed: int) -> list[str]:
    return [G.python_bat(isaaclab), str(ROOT / "bench" / "train_with_fault.py"), "--task", G.TASK, "--num_envs", str(G.NUM_ENVS),
            "--max_iterations", str(G.MAX_ITERATIONS), "--seed", str(seed), "--headless", "--run_name", name]


def _check_tag(tag) -> None:
    if not isinstance(tag, str) or not re.fullmatch(TAG_RE, tag):
        raise ValueError("tag는 anymal_c_band_<YYYYMMDD>[접미]")


def _consumed(kind: str) -> Path:
    return PRIVATE / "consumed" / f"{U.sha(PROTOCOL, True)[:16]}_{kind}.json"


def seed_record_path(tag: str, seed: int, attempt: int) -> Path:
    return RESULTS / tag / "seeds" / f"s{seed}_a{attempt}.json"


def seed_final(tag: str, seed: int) -> dict | None:
    """그 seed의 최종 기록. 유효, 인프라 밖 원인 invalid, 시도 2 invalid가 최종이다. 인프라 원인 시도 1 invalid면 None(재시도 대기)."""
    second, first = seed_record_path(tag, seed, 2), seed_record_path(tag, seed, 1)
    if second.is_file():
        return U.read(second)
    if not first.is_file():
        return None
    record = U.read(first)
    return None if (not record["valid"] and record["invalid_cause"] == "infrastructure") else record


def _reverify(ctx: dict) -> None:
    protocol, commit = verify(ctx["isaaclab"])
    if protocol != ctx["protocol"]:
        raise ValueError("실행 중 프로토콜이나 결합 소스가 바뀌었다")
    paths = [*CODE, PROTOCOL.relative_to(ROOT).as_posix()]
    if commit != ctx["commit"] and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", ctx["commit"], commit, "--", *paths],
                                                  capture_output=True).returncode != 0:
        raise ValueError("시작 커밋과 현재 HEAD 사이에 결합 경로가 바뀌었다")
    if commit not in ctx["commits_seen"]:
        ctx["commits_seen"].append(commit)


def _run_seed(ctx: dict) -> None:
    """단계 0 _run과 같은 순서(학습·추출·params·평가 전 재확인·고정 평가·probe·사후 확인). seed와 출력 경로만 다르다."""
    isaaclab, evidence, private, seed = ctx["isaaclab"], ctx["evidence"], ctx["private"], ctx["seed"]
    ctx["stage"] = "training"
    os.environ["TRIAGE_FAULT"] = "NONE"
    try:
        execution = U._owned(private / "train", train_command(isaaclab, evidence["run_name"], seed), isaaclab,
                             G.WATCHDOG_S["training"])
    finally:
        os.environ.pop("TRIAGE_FAULT", None)
    ctx["executions"]["training"] = execution
    evidence.update(train_wall_s=round(execution["wall_s"], 1), train_exit_code=execution["exit_code"],
                    readback=G._lines(private / "train/stdout.log", "[hidden_fault] "))
    run_dirs = G.find_run_dir(isaaclab, evidence["run_name"])
    run_dir = run_dirs[0] if len(run_dirs) == 1 else None
    ctx["run_dir"] = run_dir
    checkpoint = run_dir / G.FINAL_CHECKPOINT if run_dir else None
    evidence["checkpoint_sha256"] = U.sha(checkpoint) if checkpoint and checkpoint.is_file() else None
    evidence["first_checkpoint"] = bool(run_dir and (run_dir / G.FIRST_CHECKPOINT).is_file())
    if not U.finished(execution):
        raise RuntimeError("학습 프로세스가 시간 안에 스스로 끝나지 않았거나 정리되지 않았다")
    if run_dir is None:
        return
    ctx["stage"] = "telemetry_extract"
    telemetry = G.RUNS_DIR / f"{evidence['run_name']}.telemetry.json"
    if not telemetry.exists():
        G.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        extract = U._owned(private / "extract", [G.python_bat(isaaclab), str(ROOT / "bench" / "extract_tb.py"), str(run_dir),
                                                 str(telemetry)], isaaclab, G.WATCHDOG_S["telemetry_extract"])
        ctx["executions"]["telemetry_extract"] = extract
        if not U.finished(extract) or not telemetry.is_file():
            raise RuntimeError("텔레메트리 추출이 끝나지 않았거나 파일이 없다")
    evidence["telemetry"], evidence["telemetry_sha256"] = U.read(telemetry), U.sha(telemetry)
    evidence["telemetry_path"] = telemetry.relative_to(ROOT).as_posix()
    ctx["stage"] = "params"
    evidence["params"] = U.load_params(run_dir)
    if not all(training_checks(evidence, seed).values()):
        return
    ctx["stage"] = "pre_evaluation"
    _reverify(ctx)
    U._wait_for_gpu(private)
    ctx["stage"] = "fixed_evaluation"
    report_path = ctx["out"] / "runs" / f"{evidence['run_name']}.json"
    jobs = private / "eval" / "jobs.json"
    U.write_new(jobs, {"action_scale": G.ACTION_SCALE, "jobs": [{"name": evidence["run_name"],
                                                                 "checkpoint": str(run_dir / G.FINAL_CHECKPOINT),
                                                                 "output": str(report_path)}]})
    execution = U._owned(private / "eval", [G.python_bat(isaaclab), str(ROOT / "evals" / "fixed_eval_task.py"),
                                            "--protocol", str(G.EVAL_PROTOCOL), "--jobs", str(jobs), "--headless"],
                         isaaclab, G.WATCHDOG_S["fixed_evaluation"])
    ctx["executions"]["fixed_evaluation"] = execution
    evidence["eval_wall_s"] = round(execution["wall_s"], 1)
    log = private / "eval/stdout.log"
    evidence["eval_env"] = FE.parse(log.read_text(encoding="utf-8", errors="replace") if log.is_file() else "")
    if not U.finished(execution) or execution["exit_code"] != 0 or not report_path.is_file():
        raise RuntimeError("고정 평가가 끝나지 않았거나 보고서가 없다")
    evidence["report"] = U.read(report_path)
    ctx["stage"] = "probes"
    probe_path = ctx["out"] / "probes" / f"{evidence['run_name']}.json"
    probe_jobs = private / "probes" / "jobs.json"
    U.write_new(probe_jobs, {"task": G.TASK, "fault": "NONE", "seed": seed, "jobs": [
        {"name": evidence["run_name"], "checkpoint": str(run_dir / G.FINAL_CHECKPOINT), "probes": list(G.PROBE_ORDER),
         "output": str(probe_path)}]})
    execution = U._owned(private / "probes", [G.python_bat(isaaclab), str(ROOT / "evals" / "probes_task.py"),
                                              "--jobs", str(probe_jobs), "--headless"], isaaclab, G.WATCHDOG_S["probes"])
    ctx["executions"]["probes"] = execution
    evidence["probe_wall_s"] = round(execution["wall_s"], 1)
    if not U.finished(execution) or execution["exit_code"] != 0 or not probe_path.is_file():
        raise RuntimeError("probe 프로세스가 끝나지 않았거나 보고서가 없다")
    evidence["probe_report"] = U.read(probe_path)
    ctx["stage"] = "postflight"
    _reverify(ctx)
    if U.sha(run_dir / G.FINAL_CHECKPOINT) != evidence["checkpoint_sha256"]:
        raise ValueError("평가·probe 중 체크포인트가 바뀌었다")
    after = U._admission()
    U.write_new(private / "gpu_after.json", after)
    owned = {pid: created for execution in ctx["executions"].values()
             for pid, created in execution["owned_process_identities"].items()}
    if any(str(row["pid"]) in owned and U._process_created(int(row["pid"])) == owned[str(row["pid"])]
           for row in after["processes"]):
        raise ValueError("소유 프로세스가 GPU에 남아 있다")


def run_seed(tag: str, seed: int, isaaclab: Path, attempt: int) -> dict:
    protocol, commit = verify(isaaclab)
    name = run_name(seed, attempt)
    if attempt == 2:
        first = seed_record_path(tag, seed, 1)
        if not first.is_file() or U.read(first)["valid"] or U.read(first)["invalid_cause"] != "infrastructure":
            raise ValueError("시도 2는 그 seed 시도 1이 인프라 원인 invalid일 때만 허용된다")
    if (seed_record_path(tag, seed, attempt).exists() or _consumed(f"s{seed}_attempt{attempt}").exists()
            or (G.RUNS_DIR / f"{name}.telemetry.json").exists() or G.find_run_dir(isaaclab, name)):
        raise FileExistsError(f"seed {seed} 시도 {attempt}의 산출물이 이미 있다")
    evidence = seed_evidence(isaaclab, seed, {run_name(seed, 1)} if attempt == 2 else set())
    if evidence["hits"]:
        raise ValueError(f"seed {seed}가 이미 쓰였다: {evidence['hits'][:5]}")
    gpu = U._admission()
    if gpu["busy"]:
        return {"status": "gpu_busy_no_launch", "busy_reasons": gpu["busy_reasons"], "training_launched": False}
    ctx = {"isaaclab": isaaclab, "protocol": protocol, "commit": commit, "seed": seed, "out": RESULTS / tag,
           "private": PRIVATE / tag / f"s{seed}_a{attempt}", "run_dir": None, "stage": "training", "failure": None,
           "executions": {}, "evidence": {"run_name": name}, "commits_seen": [commit]}
    U.write_new(_consumed(f"s{seed}_attempt{attempt}"), {"tag": tag, "seed": seed, "attempt": attempt, "run_name": name})
    U.write_new(ctx["private"] / "gpu_before_training.json", gpu)
    try:
        _run_seed(ctx)
    except Exception as error:
        ctx["failure"] = {"stage": ctx["stage"], "error_type": type(error).__name__}
        U.write_new(ctx["private"] / "failure.json", {**ctx["failure"], "detail": str(error)})
    ev = ctx["evidence"]
    try:
        verdict = judge_seed(protocol, {**ev, "failure": ctx["failure"]}, seed)
    except Exception as error:
        verdict = {"checks": None, "failed_checks": ["pipeline_completed"], "valid": False, "invalid_cause": "runner"}
        U.write_new(ctx["private"] / "judge_error.json", {"error_type": type(error).__name__, "detail": str(error)})
    record = {"schema": "second_task_band_seed_v1", "tag": tag, "seed": seed, "attempt": attempt, "run_name": name,
              "source_commit_sha": commit, "commits_seen": ctx["commits_seen"], "protocol_sha256_lf": U.sha(PROTOCOL, True),
              "seed_evidence": evidence, **verdict,
              "failure": ctx["failure"], "checkpoint_sha256": ev.get("checkpoint_sha256"),
              "telemetry_sha256": ev.get("telemetry_sha256"), "telemetry": ev.get("telemetry_path"),
              "report": (ctx["out"] / "runs" / f"{name}.json").relative_to(ROOT).as_posix() if ev.get("report") else None,
              "probe_report": (ctx["out"] / "probes" / f"{name}.json").relative_to(ROOT).as_posix() if ev.get("probe_report") else None,
              "eval_env": ev.get("eval_env"), "parent_wall_s": {"train": ev.get("train_wall_s"), "fixed_eval": ev.get("eval_wall_s"),
                                                                "probes": ev.get("probe_wall_s")}}
    U.write_new(ctx["private"] / "execution.json", {"run_dir": str(ctx["run_dir"]) if ctx["run_dir"] else None,
                                                    "train_command": train_command(isaaclab, name, seed), "failure": ctx["failure"]})
    U.write_new(seed_record_path(tag, seed, attempt), record)
    return {"seed": seed, "attempt": attempt, "valid": record["valid"], "invalid_cause": record["invalid_cause"]}


# --- 요약(GPU 없이 시험한다) ---
def normal_runs(tag: str) -> dict[int, dict]:
    """정상 실행 셋: 단계 0(2043)과 이 단계 두 seed의 고정 평가 보고서·probe 보고서·텔레메트리."""
    runs = {STAGE0_SEED: {"run_name": STAGE0_RUN, "report": U.read(ROOT / STAGE0_FILES["report"]),
                          "probe_report": U.read(ROOT / STAGE0_FILES["probe_report"]),
                          "telemetry": U.read(ROOT / STAGE0_FILES["telemetry"])}}
    for seed in SEEDS:
        record = seed_final(tag, seed)
        if record is None or not record["valid"]:
            raise ValueError(f"seed {seed}의 유효한 최종 기록이 없다")
        runs[seed] = {"run_name": record["run_name"], "report": U.read(ROOT / record["report"]),
                      "probe_report": U.read(ROOT / record["probe_report"]), "telemetry": U.read(ROOT / record["telemetry"])}
    return runs


def analyze(runs: dict[int, dict]) -> dict:
    behavior = {seed: C._behavior(run["report"]) for seed, run in runs.items()}
    pairs = [(i, j) for i, j in itertools.permutations(sorted(runs), 2)]
    comparisons = [{"pair": f"s{i} vs s{j}", **RC.compare(behavior[i], behavior[j])} for i, j in pairs]
    bands = RC.derive_bands([{k: v for k, v in c.items() if k != "pair"} for c in comparisons])
    probes = {seed: run["probe_report"]["probes"] for seed, run in runs.items()}
    ratio = [{"pair": f"s{i} vs s{j}", "probe": p,
              "outcome": L.classify(p, probes[i][p].get("measurement"), probes[j][p].get("measurement"))}
             for i, j in pairs for p in G.RATIO_PROBES]
    series = {seed: run["telemetry"]["series"][TA.METRIC] for seed, run in runs.items()}
    alarms = {seed: TA.r1(series[seed], [series[other] for other in sorted(runs) if other != seed]) for seed in sorted(runs)}
    walks = {seed: U.walks(runs[seed]["report"]) for seed in SEEDS}
    absolute = {seed: {p: L.classify(p, probes[seed][p].get("measurement"), None) for p in G.ABSOLUTE_PROBES} for seed in SEEDS}
    criteria = {"new_runs_walk": all(walks.values()),
                "absolute_probes_normal": all(v == "normal" for row in absolute.values() for v in row.values()),
                "ratio_probes_cross_seed_normal": all(row["outcome"] == "normal" for row in ratio),
                "no_tracking_false_alarm": not any(a["warned"] for a in alarms.values())}
    return {"criteria": criteria, "bands": bands, "comparisons": comparisons, "ratio_probes": ratio,
            "ratio_measurements": {seed: {p: probes[seed][p].get("measurement") for p in G.RATIO_PROBES} for seed in sorted(runs)},
            "tracking_alarm": {str(seed): a for seed, a in alarms.items()}, "walks": {str(s): w for s, w in walks.items()},
            "absolute_probes": {str(s): a for s, a in absolute.items()},
            "standstill_ratio": {str(seed): U.standstill_ratio(run["report"]) for seed, run in runs.items()},
            "behavior": {str(seed): b for seed, b in behavior.items()}}


def summarize(tag: str, isaaclab: Path) -> dict:
    _check_tag(tag)
    protocol, commit = verify(isaaclab)
    if (RESULTS / tag / "result.json").exists():
        raise FileExistsError("요약은 한 번만 쓴다")
    finals = {seed: seed_final(tag, seed) for seed in SEEDS}
    if any(record is None for record in finals.values()):
        raise ValueError(f"아직 최종이 아닌 seed가 있다: {[s for s, r in finals.items() if r is None]}")
    result = {"schema": "second_task_band_result_v1", "tag": tag, "task": G.TASK, "seeds": list(ALL_SEEDS),
              "protocol_sha256_lf": U.sha(PROTOCOL, True), "source_commit_sha": commit,
              "seed_records": {str(s): {k: r[k] for k in ("attempt", "run_name", "valid", "invalid_cause", "failed_checks",
                                                          "parent_wall_s")} for s, r in finals.items()},
              "stage0": {"tag": STAGE0_TAG, "run_name": STAGE0_RUN}, "scope": FIXED["scope"], "model_calls": 0}
    if not all(record["valid"] for record in finals.values()):
        result.update(outcome="invalid", failed_criteria=[], analysis=None, wording=FIXED["wording"]["invalid"])
    else:
        analysis = analyze(normal_runs(tag))
        failed = [name for name, ok in analysis["criteria"].items() if not ok]
        outcome = "fail" if failed else "pass"
        result.update(outcome=outcome, failed_criteria=failed, analysis=analysis, wording=FIXED["wording"][outcome])
    prediction = FIXED["prediction"]["outcome"]
    result.update(prediction=prediction, prediction_match=None if result["outcome"] == "invalid" else result["outcome"] == prediction)
    U.write_new(RESULTS / tag / "result.json", result)
    return {"outcome": result["outcome"], "failed_criteria": result["failed_criteria"], "result": f"evals/results/{tag}/result.json"}


def execute(tag: str, isaaclab: Path) -> dict:
    _check_tag(tag)
    out = []
    for seed in SEEDS:
        if seed_record_path(tag, seed, 1).exists():
            out.append({"seed": seed, "status": "already_run"})
            continue
        step = run_seed(tag, seed, isaaclab, 1)
        out.append(step)
        if step.get("status") == "gpu_busy_no_launch":
            return {"status": "gpu_busy_no_launch", "seeds": out}
    pending = [seed for seed in SEEDS if seed_final(tag, seed) is None]
    if pending:
        return {"status": "retry_needed", "seeds": out, "retry": [f"retry --tag {tag} --seed {s}" for s in pending]}
    return {"seeds": out, **summarize(tag, isaaclab)}


def retry(tag: str, seed: int, isaaclab: Path) -> dict:
    _check_tag(tag)
    if seed not in SEEDS:
        raise ValueError(f"seed는 {SEEDS} 중 하나")
    step = run_seed(tag, seed, isaaclab, 2)
    if step.get("status") == "gpu_busy_no_launch" or any(seed_final(tag, s) is None for s in SEEDS):
        return step
    return {"seed_step": step, **summarize(tag, isaaclab)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "verify", "execute", "retry", "summarize"))
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    args = parser.parse_args(argv)
    if args.command in ("execute", "retry", "summarize") and not args.tag:
        parser.error(f"{args.command}에는 --tag가 필요하다")
    if args.command == "retry" and args.seed is None:
        parser.error("retry에는 --seed가 필요하다")
    import record_evidence_media as M
    try:
        out = {"prepare": lambda: prepare(args.isaaclab),
               "verify": lambda: {"status": "pass", "source_commit_sha": verify(args.isaaclab)[1]},
               "execute": lambda: execute(args.tag, args.isaaclab),
               "retry": lambda: retry(args.tag, args.seed, args.isaaclab),
               "summarize": lambda: summarize(args.tag, args.isaaclab)}[args.command]()
    except (ValueError, KeyError, FileExistsError, FileNotFoundError, M.PreflightError,
            subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 3 if out.get("status") == "gpu_busy_no_launch" else 0


if __name__ == "__main__":
    sys.exit(main())
