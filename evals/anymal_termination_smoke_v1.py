"""ANYmal-C F6의 새 seed 학습 실패를 확인하는 별도 smoke. 모델 호출은 없다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import e2e_demo_anymal_c as E
import preregistered_seed_guard as SEEDS
import run_recovery_reset_v2 as ANCHOR
import run_training_correction_v4 as V4
import record_evidence_media as GPU

G, B, U, OWNED = E.G, E.B, E.U, E.P
read, write_new, sha = E.read, E.write_new, E.sha
PROTOCOL = ROOT / "bench/protocols/anymal_termination_smoke_v1.json"
CPU_PROTOCOL = ROOT / "bench/protocols/anymal_termination_semantics_v1.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task27"
TAG = "anymal_f6_smoke_s2100_20261017"
SEED = 2100
ARMS = ("NONE", "F6")
BAND_COMMIT = "b25db74337bf7df3ea0fd13b252c708f078f9aad"
CODE = set(E.CODE) | {
    "evals/anymal_termination_smoke_v1.py", "bench/run_anymal_termination_smoke_v1.ps1",
    "tests/test_anymal_termination_smoke_v1.py", "bench/anymal_termination_semantics_v1.py",
    "tests/test_anymal_termination_semantics_v1.py", "evals/preregistered_seed_guard.py",
    "evals/run_recovery_reset_v2.py", "evals/run_training_correction_v4.py",
}
FIXED = {
    "schema": "anymal_termination_smoke_protocol_v1",
    "question": "ANYmal-C의 등록 F6가 새 seed 학습 정책의 고정 평가 실패로 이어지는가",
    "design_disclosure": "Go2 F6와 ANYmal-C F1 결과, F2 적용 불가 CPU 결과를 본 뒤 선택한 개발 smoke다",
    "seed": SEED, "tag": TAG, "task": G.TASK, "arms": list(ARMS),
    "num_envs": 4096, "learning_iterations": 300, "num_steps_per_env": 24,
    "action_scale": 0.5, "checkpoint": "model_299.pt",
    "fault": "동결 F6: time_out 본문을 max_episode_length의 5%로 바꾼다. 설정값은 유지한다",
    "runtime_injection_gate": {"tail_iterations": 60, "f6_episode_steps_max": 50.0,
        "rounding_tolerance": 1e-6, "f6_time_out_observed": True, "none_episode_median_gt": 50.0,
        "on_mismatch": "invalid", "scope": "학습 로그에서 조기 종료와 맞는 관측을 확인한다. 보행 실패 판정은 고정 평가가 맡는다"},
    "band_commit": BAND_COMMIT, "band_result": E.STAGE1_RESULT,
    "evaluation": "기존 ANYmal-C 고정 평가 26조건·조건당40env·1000step·eval seed2026. 결함을 평가 환경에 주입하지 않는다",
    "prediction": "failure_present",
    "decision": "e2e_demo_v4.source_gate 그대로: NONE가 걷고 F6가 같은 seed NONE 대비 unhealthy일 때 failure_present",
    "wording": {
        "failure_present": "ANYmal-C 새 seed에서 등록 F6 학습 정책의 행동 실패를 확인했다. 별도 종단 시연 사전등록을 준비한다.",
        "source_failure_absent": "새 seed에서 F6의 등록 고정 평가 실패를 확인하지 못했다. 종단 시연으로 진행하지 않는다.",
        "baseline_invalid": "정상 대조의 걷기 조건을 충족하지 못해 F6 시연으로 진행하지 않는다.",
        "invalid": "실행·출처·평가 계약이 완결되지 않아 smoke를 판정하지 않는다.",
    },
    "followup": {
        "gate": "유효한 failure_present 결과를 재검증한 뒤에만 별도 시연 protocol을 작성·커밋한다",
        "fresh_seed": 2101, "mechanism": "termination", "correction": "restore_configured_timeout",
        "recovery_mode": "resume", "decision_path": "single_feedback", "decision_model": E.MODEL,
        "budget": "새 seed 정상·F6 학습 300회, 등록 교정 후 이어 학습 300회와 정상 이어 학습 대조",
        "prior_evidence": "Go2 timeout_resume_dev_20261013은 no_recovery였다. termination 등록 규칙은 resume이며 ANYmal 회복 성공을 보장하는 근거가 아니다",
        "prior_result": "evals/results/timeout_resume_dev_20261013/result.json",
        "automatic_execution": False,
    },
    "semantic_retries": 0, "model_calls": 0,
    "infrastructure_retry": "같은 tag 재실행 금지. invalid 기록과 seed 소비를 보존하고 원인 확인 후 별도 판을 사전등록한다",
    "no_tuning": "실패가 없으면 결함 크기·평가 밴드·probe를 바꾸거나 다른 seed로 찾지 않는다",
    "gpu_admission_profile": U.GPU_PROFILE,
    "watchdogs_s": {"cpu": 180, "train": 3600, "extract": 300, "evaluate": 600},
    "time_scope": "부모 실행부터 자연 종료 및 소유 프로세스 정리까지의 벽시계. GPU 커널 시간 아님",
}


def band_inputs() -> dict:
    ANCHOR._git(ROOT, "merge-base", "--is-ancestor", BAND_COMMIT, "HEAD")
    bound = E.stage1_inputs()
    path = ROOT / E.STAGE1_RESULT
    original = ANCHOR._git(ROOT, "show", f"{BAND_COMMIT}:{E.STAGE1_RESULT}")
    if ANCHOR._lf_bytes(path.read_bytes()) != ANCHOR._lf_bytes(original):
        raise ValueError("등록된 ANYmal 밴드 공개 커밋과 결과가 다르다")
    return {"binding": bound, "bands": read(path)["analysis"]["bands"]}


def dynamic(isaaclab: Path) -> dict:
    maps = V4.source_maps(isaaclab)
    maps["source_files"] = {name: sha(ROOT / name, True) for name in sorted(CODE)}
    return {**maps, "anymal_sdk": G.isaaclab_sources(isaaclab), **band_inputs(),
            "cpu_protocol_sha256_lf": sha(CPU_PROTOCOL, True),
            "fixed_eval_sha256_lf": sha(G.EVAL_PROTOCOL, True)}


def bound_files(protocol: dict) -> tuple[str, ...]:
    stage = protocol["binding"]
    paths = set(CODE) | {PROTOCOL.relative_to(ROOT).as_posix(), CPU_PROTOCOL.relative_to(ROOT).as_posix(),
                        G.EVAL_PROTOCOL.relative_to(ROOT).as_posix(), B.PROTOCOL.relative_to(ROOT).as_posix()}
    paths |= {stage["result"]["path"], stage["stage0_result"]["path"]}
    paths |= {row["path"] for group in ("seed_records", "reference_telemetry") for row in stage[group].values()}
    return tuple(sorted(paths))


def cpu_summary(report: dict) -> dict:
    # 의미 검사 결과 계약은 같은 판의 별도 CPU 실행기에서 검사한다.
    import anymal_termination_semantics_v1 as SEM
    return SEM.smoke_binding(report)


def prepare(isaaclab: Path, report_path: Path) -> dict:
    document = {**deepcopy(FIXED), **dynamic(isaaclab), "cpu_semantics": cpu_summary(read(report_path))}
    ANCHOR._reject_public_absolute_paths(document)
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execute", "sha256_lf": sha(PROTOCOL, True)}


def check(isaaclab: Path, committed: bool = False) -> tuple[dict, str | None]:
    protocol = read(PROTOCOL)
    current = dynamic(isaaclab)
    if (set(protocol) != set(FIXED) | set(current) | {"cpu_semantics"}
            or any(protocol.get(k) != v for k, v in {**FIXED, **current}.items())):
        raise ValueError("사전등록 또는 결합 소스가 바뀌었다")
    if protocol["cpu_semantics"].get("applicable") is not True:
        raise ValueError("CPU 의미 게이트 미통과")
    if E.V3.S._git_head(isaaclab) != V4.FIXED["isaaclab_commit"] or not E.V3.S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout이 등록값과 다르다")
    return protocol, ANCHOR.verify_bound_at_head(ROOT, bound_files(protocol)) if committed else None


def folder_for(tag: str) -> Path:
    if tag != TAG:
        raise ValueError("등록 tag가 아니다")
    folder = PRIVATE / "runs" / tag
    if folder.is_symlink() or folder.resolve().parent != (PRIVATE / "runs").resolve():
        raise ValueError("결과 경로가 등록 root 밖이다")
    return folder


def consume_path() -> Path:
    return PRIVATE / "consumed" / f"{sha(PROTOCOL, True)}.json"


def name_for(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("등록 arm이 아니다")
    return f"anymal_f6_smoke_{arm.lower()}_s{SEED}"


def finished(execution: dict) -> bool:
    return U.finished(execution) and execution.get("exit_code") == 0


def cpu_command(isaaclab: Path, stage: Path) -> list[str]:
    return [G.python_bat(isaaclab), "-B", str(ROOT / "bench/anymal_termination_semantics_v1.py"), "run",
            "--isaaclab", str(isaaclab), "--output", str(stage / "report.json")]


def validate_cpu_execution(stage: Path, isaaclab: Path) -> None:
    if (not finished(read(stage / "execution.json"))
            or read(stage / "launch.json") != {"args": cpu_command(isaaclab, stage), "cwd": str(isaaclab)}):
        raise ValueError("CPU 의미 검사 실행/명령 결합이 다르다")


def admit(stage: Path) -> None:
    row = GPU.query_gpu(U.GPU_PROFILE)
    if row.get("admission_policy") != U.GPU_PROFILE or row.get("busy") is not False:
        raise RuntimeError("GPU 사용 중이거나 입장 검사 실패")
    write_new(stage / "gpu_before.json", row)


def run(stage: Path, argv: list[str], isaaclab: Path, timeout: int) -> dict:
    execution = U._owned(stage, argv, isaaclab, timeout)
    if not finished(execution):
        raise RuntimeError(f"소유 프로세스 실행 미완결: {stage.name}")
    return execution


def training_info(stage: Path, arm: str, isaaclab: Path) -> dict:
    execution = read(stage / "execution.json")
    argv = B.train_command(isaaclab, name_for(arm), SEED)
    if not finished(execution) or read(stage / "launch.json") != {"args": argv, "cwd": str(isaaclab)}:
        raise ValueError("학습 실행/명령 결합이 다르다")
    if G._lines(stage / "stdout.log", "[hidden_fault] ") != [{"fault": arm}]:
        raise ValueError("결함 주입 확인 줄이 다르다")
    found = G.find_run_dir(isaaclab, name_for(arm))
    if len(found) != 1 or not B.params_expected(U.load_params(found[0]), SEED):
        raise ValueError("학습 폴더/설정이 등록값과 다르다")
    env_cfg = E.V3.load_config(found[0] / "params/env.yaml")
    agent = E.V3.load_config(found[0] / "params/agent.yaml")
    if (env_cfg.get("episode_length_s") != 20.0 or agent.get("num_steps_per_env") != 24
            or env_cfg.get("terminations", {}).get("time_out") != {
                "func": "isaaclab.envs.mdp.terminations:time_out", "params": {}, "time_out": True}):
        raise ValueError("ANYmal 종료 설정/학습 rollout 길이가 다르다")
    files = {"checkpoint": found[0] / "model_299.pt", "first_checkpoint": found[0] / "model_0.pt",
             "environment": found[0] / "params/env.yaml", "agent": found[0] / "params/agent.yaml"}
    return {key: {"path": str(path), "sha256": sha(path)} for key, path in files.items()}


def train_arm(folder: Path, arm: str, isaaclab: Path, protocol: dict) -> dict:
    stage = folder / arm / "train"
    if G.find_run_dir(isaaclab, name_for(arm)):
        raise FileExistsError("학습 실행 이름이 이미 있다")
    stage.mkdir(parents=True, exist_ok=False)
    admit(stage)
    previous = os.environ.get("TRIAGE_FAULT")
    os.environ["TRIAGE_FAULT"] = arm
    try:
        run(stage, B.train_command(isaaclab, name_for(arm), SEED), isaaclab, protocol["watchdogs_s"]["train"])
    finally:
        if previous is None: os.environ.pop("TRIAGE_FAULT", None)
        else: os.environ["TRIAGE_FAULT"] = previous
    artifacts = training_info(stage, arm, isaaclab)
    output = folder / arm / "telemetry.json"
    run(folder / arm / "extract", [G.python_bat(isaaclab), str(ROOT / "bench/extract_tb.py"),
        str(Path(artifacts["checkpoint"]["path"]).parent), str(output)], isaaclab, protocol["watchdogs_s"]["extract"])
    telemetry = read(output)
    if len(telemetry.get("series", {}).get("Train/mean_episode_length", [])) != 300:
        raise ValueError("학습 telemetry가 300회를 포함하지 않는다")
    write_new(folder / arm / "artifacts.json", artifacts)
    return artifacts


def eval_command(isaaclab: Path, stage: Path) -> list[str]:
    return [G.python_bat(isaaclab), str(ROOT / "evals/fixed_eval_task.py"), "--protocol", str(G.EVAL_PROTOCOL),
            "--jobs", str(stage / "jobs.json"), "--headless"]


def evaluate(folder: Path, artifacts: dict, isaaclab: Path, protocol: dict) -> None:
    stage = folder / "evaluate"
    stage.mkdir(exist_ok=False)
    admit(stage)
    write_new(stage / "jobs.json", {"action_scale": 0.5, "jobs": [
        {"name": name_for(arm), "checkpoint": artifacts[arm]["checkpoint"]["path"],
         "output": str(stage / f"{arm}.json")} for arm in ARMS]})
    run(stage, eval_command(isaaclab, stage), isaaclab, protocol["watchdogs_s"]["evaluate"])


def injection_signature(telemetry: dict[str, dict]) -> dict:
    tail = FIXED["runtime_injection_gate"]["tail_iterations"]
    length, timeout = {}, {}
    for arm in ARMS:
        series = telemetry[arm].get("series", {})
        for key in ("Train/mean_episode_length", "Episode_Termination/time_out"):
            values = series.get(key)
            if (not isinstance(values, list) or len(values) != 300
                    or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError("종료 주입 관측 telemetry가 유효하지 않다")
        length[arm] = series["Train/mean_episode_length"][-tail:]
        timeout[arm] = series["Episode_Termination/time_out"][-tail:]
    checks = {"f6_episodes_at_most_50_steps": max(length["F6"]) <= 50.0 + 1e-6,
              "f6_time_out_observed": max(timeout["F6"]) > 0,
              "none_episode_median_above_50": statistics.median(length["NONE"]) > 50.0}
    if not all(checks.values()):
        raise ValueError("F6 조기 종료의 학습 로그 signature가 성립하지 않는다")
    return {"checks": checks, "tail_iterations": tail,
            "episode_steps_median": {arm: statistics.median(length[arm]) for arm in ARMS},
            "f6_episode_steps_max": max(length["F6"]), "f6_time_out_max": max(timeout["F6"])}


def replay(folder: Path, isaaclab: Path, protocol: dict) -> dict:
    request = read(folder / "request.json")
    anchor = ANCHOR.verify_source_anchor(request["source_anchor"], ROOT, bound_files(protocol))
    if request != {"tag": TAG, "seed": SEED, "protocol_sha256_lf": sha(PROTOCOL, True),
                   "source_anchor": anchor, "isaaclab": str(isaaclab)}:
        raise ValueError("요청 결합이 다르다")
    if read(consume_path()) != {"request_sha256": sha(folder / "request.json")}:
        raise ValueError("소비 기록 결합이 다르다")
    freshness = read(folder / "seed_freshness.json")
    if (freshness.get("schema") != "preregistered_seed_freshness_v1" or freshness.get("status") != "fresh"
            or freshness.get("seeds") != [SEED] or freshness.get("evidence") != []):
        raise ValueError("seed 미사용 기록이 다르다")
    validate_cpu_execution(folder / "cpu", isaaclab)
    if cpu_summary(read(folder / "cpu/report.json")) != protocol["cpu_semantics"]:
        raise ValueError("CPU 의미 재검증이 사전등록과 다르다")
    artifacts, costs = {}, {}
    for arm in ARMS:
        stage = folder / arm / "train"
        artifacts[arm] = training_info(stage, arm, isaaclab)
        if read(folder / arm / "artifacts.json") != artifacts[arm]:
            raise ValueError("학습 산출물 SHA가 바뀌었다")
        train = read(stage / "execution.json")
        extract = read(folder / arm / "extract/execution.json")
        if not finished(extract) or len(read(folder / arm / "telemetry.json").get("series", {}).get("Train/mean_episode_length", [])) != 300:
            raise ValueError("telemetry 실행/길이가 다르다")
        costs[arm] = {"training_wall_s": train["wall_s"], "extract_wall_s": extract["wall_s"]}
        admission = read(stage / "gpu_before.json")
        if admission.get("admission_policy") != U.GPU_PROFILE or admission.get("busy") is not False:
            raise ValueError("학습 GPU 입장 기록이 다르다")
        extract_argv = [G.python_bat(isaaclab), str(ROOT / "bench/extract_tb.py"),
                        str(Path(artifacts[arm]["checkpoint"]["path"]).parent), str(folder / arm / "telemetry.json")]
        if read(folder / arm / "extract/launch.json") != {"args": extract_argv, "cwd": str(isaaclab)}:
            raise ValueError("telemetry 명령 결합이 다르다")
    for kind in ("environment", "agent"):
        configs = {arm: E.V3.load_config(Path(artifacts[arm][kind]["path"])) for arm in ARMS}
        if kind == "agent":
            for arm in ARMS:
                if configs[arm].pop("run_name", None) != name_for(arm):
                    raise ValueError("학습 run_name이 다르다")
        if configs["NONE"] != configs["F6"]:
            raise ValueError("정상과 F6 사이에 등록 밖 학습 설정 차이가 있다")
    signature = injection_signature({arm: read(folder / arm / "telemetry.json") for arm in ARMS})
    stage = folder / "evaluate"
    expected_jobs = {"action_scale": 0.5, "jobs": [{"name": name_for(arm),
        "checkpoint": artifacts[arm]["checkpoint"]["path"], "output": str(stage / f"{arm}.json")} for arm in ARMS]}
    execution = read(stage / "execution.json")
    admission = read(stage / "gpu_before.json")
    if admission.get("admission_policy") != U.GPU_PROFILE or admission.get("busy") is not False:
        raise ValueError("평가 GPU 입장 기록이 다르다")
    if (not finished(execution) or read(stage / "jobs.json") != expected_jobs
            or read(stage / "launch.json") != {"args": eval_command(isaaclab, stage), "cwd": str(isaaclab)}
            or not G.eval_env_ok(E.FE.parse((stage / "stdout.log").read_text(encoding="utf-8", errors="replace")))):
        raise ValueError("고정 평가 실행/환경 결합이 다르다")
    reports = {arm: E.validate_report(read(stage / f"{arm}.json"), {
        "name": name_for(arm), "checkpoint_file": "model_299.pt",
        "checkpoint_sha256": artifacts[arm]["checkpoint"]["sha256"], "action_scale": 0.5,
        "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"]}) for arm in ARMS}
    versions = reports["NONE"]["versions"]
    if (versions != reports["F6"]["versions"] or versions.get("rsl-rl-lib") != V4.FIXED["rsl_rl_version"]
            or versions.get("torch") != E.fixed("f1")["torch_version"]):
        raise ValueError("고정 평가 runtime이 다르다")
    gate = E.E4.source_gate(reports["NONE"], reports["F6"], protocol["bands"])
    for row in [*costs.values(), {"eval": execution["wall_s"]}]:
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("비용 기록이 유효하지 않다")
    return {"schema": "anymal_termination_smoke_result_v1", "status": "complete", "seed": SEED,
            "gate": gate, "outcome": gate["outcome"], "prediction_match": gate["outcome"] == protocol["prediction"],
            "wording": protocol["wording"][gate["outcome"]], "model_calls": 0,
            "may_preregister_demo": gate["outcome"] == "failure_present", "demo_executed": False,
            "source_commit_sha": anchor["commit"], "protocol_sha256_lf": sha(PROTOCOL, True),
            "runtime_injection_signature": signature,
            "costs": {"training": costs, "fixed_eval_wall_s": execution["wall_s"]},
            "metrics": {arm: reports[arm]["metrics"] for arm in ARMS},
            "evidence_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
                if p.is_file() and p.name not in {"result.json", "failure.json"}}}


def execute(isaaclab: Path, tag: str) -> dict:
    protocol, commit = check(isaaclab, True)
    if commit is None:
        raise ValueError("실행에는 커밋된 source anchor가 필요하다")
    folder = folder_for(tag)
    if folder.exists() or consume_path().exists():
        raise FileExistsError("등록 smoke가 이미 소비됐다")
    fresh = SEEDS.assert_fresh_seeds([SEED], isaaclab, repo=ROOT)
    anchor = ANCHOR.make_source_anchor(commit, ROOT, bound_files(protocol))
    write_new(folder / "seed_freshness.json", fresh)
    write_new(folder / "request.json", {"tag": tag, "seed": SEED, "protocol_sha256_lf": sha(PROTOCOL, True),
                                        "source_anchor": anchor, "isaaclab": str(isaaclab)})
    write_new(consume_path(), {"request_sha256": sha(folder / "request.json")})
    try:
        cpu = folder / "cpu"
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            run(cpu, cpu_command(isaaclab, cpu), isaaclab, protocol["watchdogs_s"]["cpu"])
        finally:
            if previous is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = previous
        if cpu_summary(read(cpu / "report.json")) != protocol["cpu_semantics"]:
            raise ValueError("CPU 의미 재검증이 사전등록과 다르다")
        artifacts = {}
        for arm in ARMS:
            check(isaaclab, True)
            ANCHOR.verify_source_anchor(anchor, ROOT, bound_files(protocol))
            artifacts[arm] = train_arm(folder, arm, isaaclab, protocol)
        injection_signature({arm: read(folder / arm / "telemetry.json") for arm in ARMS})
        check(isaaclab, True)
        evaluate(folder, artifacts, isaaclab, protocol)
        check(isaaclab, True)
        result = replay(folder, isaaclab, protocol)
        write_new(folder / "result.json", result)
        return result
    except Exception as error:
        write_new(folder / "failure.json", {"type": type(error).__name__, "detail": str(error)[:1000]})
        write_new(folder / "result.json", {"schema": "anymal_termination_smoke_result_v1", "status": "invalid",
            "outcome": "invalid", "seed": SEED, "model_calls": 0, "may_preregister_demo": False,
            "wording": protocol["wording"]["invalid"]})
        raise


def verify_result(isaaclab: Path, tag: str) -> dict:
    protocol, _ = check(isaaclab, True)
    folder = folder_for(tag)
    expected = replay(folder, isaaclab, protocol)
    if read(folder / "result.json") != expected:
        raise ValueError("저장 결과가 재계산과 다르다")
    return {"status": "valid", "outcome": expected["outcome"], "may_preregister_demo": expected["may_preregister_demo"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "freshness", "execute", "verify-result"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args(argv)
    isaaclab = args.isaaclab.resolve()
    try:
        if args.command == "prepare":
            if args.cpu_report is None: parser.error("prepare는 --cpu-report가 필요하다")
            result = prepare(isaaclab, args.cpu_report)
        elif args.command in {"check", "verify"}:
            _, commit = check(isaaclab, args.command == "verify")
            result = {"status": "valid", "commit": commit}
        elif args.command == "freshness": result = SEEDS.assert_fresh_seeds([SEED], isaaclab, repo=ROOT)
        elif args.command == "execute": result = execute(isaaclab, args.tag)
        else: result = verify_result(isaaclab, args.tag)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"거부: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
