"""PPO ratio 계산 오류 종단 실행의 사전등록 전 설계 파일럿(모델 호출 없음).

seed 하나에서 NONE·INVERSE_RATIO 정책을 300회 학습하고 고정 평가, 동결 legacy 검사 여섯 개, 3 update 검사를 모두 실행한다.
본 실행의 사전등록을 정하기 위한 설계 근거이며 성공·회복 판정에 쓰지 않는다. 결과는 본 실행의 설계 결합으로 기록한다.

  python evals/ppo_ratio_pilot_v1.py prepare            # 사전 고정값·소스 SHA를 프로토콜로 쓴다(커밋 필요)
  python evals/ppo_ratio_pilot_v1.py verify             # 커밋·소스·런타임 결합과 실행 경로를 확인한다
  python evals/ppo_ratio_pilot_v1.py execute --tag ppo_ratio_pilot_s2160_<YYYYMMDD>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
import ppo_ratio_repair_contract as C  # noqa: E402
import ppo_ratio_repair_execution as E  # noqa: E402
from preregistered_seed_guard import assert_fresh_seeds  # noqa: E402
from rl_triage import grounded_recovery as G  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage import recovery as REC  # noqa: E402
import run_upstream_r1_smoke as R1S  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/ppo_ratio_pilot_v1.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/ppo_ratio_pilot/runs"
SEED = 2160
TASK = E.TASK
TAG = re.compile(r"ppo_ratio_pilot_s2160_[0-9]{8}[a-z0-9_]{0,16}")
ARMS = ("NONE", "INVERSE_RATIO")
NEW_FILES = (
    "evals/ppo_ratio_pilot_v1.py", "evals/ppo_ratio_repair_contract.py", "evals/ppo_ratio_repair_execution.py",
    "bench/ppo_ratio_repair_training.py", "bench/run_ppo_ratio_pilot_v1.ps1",
    "tests/test_ppo_ratio_repair_contract.py", "tests/test_ppo_ratio_repair_execution.py",
    "tests/test_ppo_ratio_pilot_v1.py",
)
CODE = set(R1S.CODE) | set(NEW_FILES) | {
    "evals/r1_repair_execution.py", "evals/r1_repair_execution_v2.py", "bench/r1_repair_training.py",
    "evals/optimizer_observability.py", "evals/ppo_gradient_invariant_v1.py", "evals/ppo_gradient_invariant_v2.py",
    "bench/ppo_ratio_mutation_v1.py", "evals/probes.py", "evals/probes_task.py", "bench/hidden_faults.py",
    "evals/r1_repair_contract.py", "evals/preregistered_seed_guard.py", "src/rl_triage/triage_tools.py",
    "src/rl_triage/mechanism_registry.py", "src/rl_triage/active_diagnosis.py",
    "src/rl_triage/correction_registry.py",
}
EVIDENCE = (
    "evals/results/ppo_gradient_v5_20261017/result.json",
    "bench/protocols/ppo_gradient_check_v5.json",
    "evals/results/p0c_fixed_eval_20261009/summary.json",
)
BANDS_EVIDENCE = "evals/results/p0c_fixed_eval_20261009/summary.json"
INSPECTION_PROBES = (C.COUNT_PROBE, C.GRADIENT_PROBE)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_protocol() -> dict:
    return {
        "schema": "ppo_ratio_pilot_protocol_v1", "seed": SEED, "task": TASK, "num_envs": 4096,
        "training_updates": 300, "fault": "inverse_ratio",
        "purpose": ("본 종단 실행 사전등록 전에 inverse_ratio 정책의 행동 실패 여부와 등록 검사 여덟 개의 관측을 "
                    "확인하는 설계 파일럿이다. 결과는 본 실행의 설계 결합으로 기록하며 성공·회복 판정에 쓰지 않는다"),
        "design_coupling": ("PPO 계산 검사 v5(seed 2143~2146, 3 update)를 본 뒤 설계했다. inverse_ratio로 300회 "
                            "학습한 정책은 아직 관측하지 않았다"),
        "flow": ["launch_paths_preflight", "fresh_seed", "NONE_training", "INVERSE_RATIO_training", "fixed_eval",
                 "NONE_legacy_all", "INVERSE_RATIO_legacy_each", "NONE_inspection", "INVERSE_RATIO_inspection",
                 "loop_replay", "summary"],
        "loop_registry": C.registry(),
        "readout": {
            "source_gate": "기존 Go2 고정 평가 밴드로 INVERSE_RATIO 대 NONE을 비교한다. NONE 걷기 실패면 baseline_invalid",
            "legacy": "동결 probe_loop.classify와 같은 seed NONE 측정을 기준값으로 쓴다",
            "inspection": ("같은 seed로 새로 시작한 3 update. P_optimizer_updates는 횟수 두 판정, P_ppo_gradient는 "
                           "gradient 판정을 읽는다. 같은 seed NONE 검사가 normal이 아니면 unknown"),
            "training_observer": "300회 학습 첫 3 update 기록은 완결 확인에만 쓴다",
            "loop_replay": "관측표로 첫 검사 여덟 가지 각각의 등록 고리 끝을 계산한다(실행 없음)",
            "continue_policy": "행동 실패가 없어도 등록 단계를 모두 실행해 설계 근거를 남긴다",
        },
        "api_requests": 0,
        "gpu": {"lease": "Use-GpuLease.ps1", "profile": E.V1.GPU_PROFILE, "sequential": True,
                "measurement_unit": "소유 프로세스 parent wall seconds; GPU kernel time이 아니다"},
        "failure_policy": {"gpu_attempts": 1, "overwrite": False, "semantic_retry": False,
                           "process_error": "보존 후 중단, 같은 tag 재실행 없음",
                           "seed": "실행 직전 로그·저장소 재검사; 사용 흔적이면 중단하고 seed 자동 교체 없음"},
    }


def dynamic(isaaclab: Path) -> dict:
    runtime = R1S.isaaclab_sources(isaaclab)
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
        raise ValueError("사전 고정값·소스·입력·런타임 결합이 다르다")
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
    if len({L.digest(r["versions"]) for r in reports.values()}) != 1:
        raise ValueError("고정 평가 런타임 버전이 서로 다르다")
    return reports


def source_gate(reports: dict, bands: dict) -> dict:
    healthy = REC.verdict(REC.compare(G._behavior(reports["INVERSE_RATIO"]), G._behavior(reports["NONE"])), bands)
    walks = R1S.U.walks(reports["NONE"])
    outcome = ("baseline_invalid" if not walks else "failure_present" if healthy["label"] == "unhealthy"
               else "source_failure_absent")
    return {"outcome": outcome, "none_walks": walks, "fault_verdict": healthy}


def inspection_outcome(probe: str, case: dict, control: dict) -> str:
    """같은 seed NONE 검사가 normal일 때만 결함 검사 판정을 쓴다."""
    key = "count_status" if probe == C.COUNT_PROBE else "gradient_status"
    return case[key] if control[key] == "normal" else "unknown"


def _duration(result: dict) -> float:
    receipt = result["execution"]
    for key in ("wall_s", "parent_wall_s", "elapsed_s"):
        if isinstance(receipt.get(key), (int, float)):
            return float(receipt[key])
    raise ValueError("실행 receipt에 벽시계 시간이 없다")


def execute(tag: str, isaaclab: Path, *, backend=E, seed_guard=assert_fresh_seeds) -> dict:
    if not TAG.fullmatch(tag):
        raise ValueError("등록 tag 형식이 다르다")
    protocol, anchor = verify(isaaclab)
    folder = PRIVATE / tag
    if folder.exists():
        raise FileExistsError("이미 시작한 tag이며 실행을 반복하지 않는다")
    launch_paths = backend.preflight(isaaclab)
    freshness = seed_guard([SEED], isaaclab, repo=ROOT)
    folder.mkdir(parents=True, exist_ok=False)
    E.write_new(folder / "seed_freshness.json", freshness)
    E.write_new(folder / "launch_paths.json", launch_paths)
    E.write_new(folder / "begin.json", {"tag": tag, "experiment_commit": anchor,
                                        "protocol_sha256_lf": E.sha(PROTOCOL, True)})
    result = {"schema": "ppo_ratio_pilot_result_v1", "tag": tag, "seed": SEED, "experiment_commit": anchor,
              "protocol_sha256_lf": E.sha(PROTOCOL, True), "api_requests": 0, "costs": {}}
    stage = "training"
    try:
        artifacts = {}
        for arm in ARMS:
            verify(isaaclab, anchor)
            stage = f"train_{arm}"
            artifacts[arm] = backend.train(folder / "training" / arm.lower(), isaaclab, arm, SEED,
                                          f"{tag}_{arm.lower()}", binding(protocol, anchor))
            E.write_new(folder / f"artifact_{arm.lower()}.json", artifacts[arm])
            result["costs"][stage] = _duration(artifacts[arm])
        result["training_observer"] = {arm: artifacts[arm]["inspection"] for arm in ARMS}
        result["readback"] = {arm: artifacts[arm]["readback"] for arm in ARMS}
        stage = "fixed_eval"
        evaluation = backend.fixed_evaluate(folder / "fixed_eval", isaaclab, artifacts, tag, binding(protocol, anchor))
        result["costs"][stage] = _duration(evaluation)
        reports = validated_reports(evaluation, artifacts)
        result["source_gate"] = source_gate(reports, read(ROOT / BANDS_EVIDENCE)["bands"])
        result["metrics"] = {arm: reports[arm]["metrics"] for arm in ARMS}
        stage = "legacy_reference"
        reference = backend.legacy_probe(folder / "legacy_none", isaaclab, {"NONE": artifacts["NONE"]},
                                         tag + "_ref", binding(protocol, anchor))
        result["costs"][stage] = _duration(reference)
        legacy = {}
        for probe in sorted(L.PROBES):
            stage = f"legacy_{probe}"
            raw = backend.legacy_probe(folder / f"legacy_{probe.lower()}", isaaclab,
                                       {"INVERSE_RATIO": artifacts["INVERSE_RATIO"]},
                                       tag + "_" + probe.lower(), binding(protocol, anchor), probes=[probe])
            item, ref = raw["report"]["probes"][probe], reference["report"]["probes"][probe]
            legacy[probe] = {"outcome": L.classify(probe, item["measurement"] if item["exit"] == 0 else None,
                                                   ref["measurement"] if ref["exit"] == 0 else None),
                             "measurement": item["measurement"], "reference": ref["measurement"]}
            result["costs"][stage] = _duration(raw)
        result["legacy"] = legacy
        controls = {}
        for arm in ("NONE", "INVERSE_RATIO"):
            stage = f"inspect_{arm}"
            controls[arm] = backend.inspect(folder / "inspection" / arm.lower(), isaaclab, arm, SEED,
                                            f"{tag}_inspect_{arm.lower()}",
                                            binding(protocol, anchor, artifacts[arm]["checkpoint"]["sha256"], 3))
            result["costs"][stage] = _duration(controls[arm])
        result["inspection"] = {probe: {"outcome": inspection_outcome(probe, controls["INVERSE_RATIO"]["measurement"],
                                                                      controls["NONE"]["measurement"]),
                                        "case": controls["INVERSE_RATIO"]["measurement"],
                                        "control": controls["NONE"]["measurement"]}
                                for probe in INSPECTION_PROBES}
        outcomes = {**{p: row["outcome"] for p, row in legacy.items()},
                    **{p: row["outcome"] for p, row in result["inspection"].items()}}
        result["outcomes"] = outcomes
        result["loop_replay"] = {first: C.replay_loop(first, outcomes) for first in sorted(C.registry()["probes"])}
        result["status"] = "completed"
        E.write_new(folder / "result.json", result)
        verify(isaaclab, anchor)
        return result
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {"stage": stage, "type": type(error).__name__}
        E.write_new(folder / "result.json", result)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.isaaclab)
    elif args.command in ("check", "verify"):
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "ok", "committed_gate": args.command == "verify", "anchor": anchor,
                 "launch_paths": E.preflight(args.isaaclab)}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED], args.isaaclab, repo=ROOT)
    else:
        value = execute(args.tag or "", args.isaaclab)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
