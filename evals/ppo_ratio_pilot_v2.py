"""PPO ratio 설계 파일럿 v2: v1의 학습 시작 전 인프라 중단 뒤 새 seed로 같은 측정을 한 번 실행한다.

v1(seed 2160)은 첫 학습 단계의 GPU 사전 점검에서 화면 앱 두 개(claude.exe·msedge.exe)를 허용 목록 밖으로 거부해
학습 프로세스를 시작하기 전에 끝났다. v2는 그 두 이름과 편집기 이름 두 개(cursor.exe·antigravity.exe)를 허용 목록에
더한 backend v2를 쓰고, 측정·판정·고리 재생은 v1과 같다. 편집기 이름은 첫 시도 전 읽기 전용 GPU 점검에서 cursor.exe가
거부돼 더했다. v1 기록이 seed 2160을 남겨 seed guard가 막으므로 새 seed 2162를 쓴다. 모델 호출은 없다.

  python evals/ppo_ratio_pilot_v2.py prepare
  python evals/ppo_ratio_pilot_v2.py verify
  python evals/ppo_ratio_pilot_v2.py execute --tag ppo_ratio_pilot_s2162_<YYYYMMDD>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
import ppo_ratio_pilot_v1 as P1  # noqa: E402
import ppo_ratio_repair_execution_v2 as E  # noqa: E402
from preregistered_seed_guard import assert_fresh_seeds  # noqa: E402

C, L, R1S = P1.C, P1.L, P1.R1S
PROTOCOL = ROOT / "bench/protocols/ppo_ratio_pilot_v2.json"
PRIVATE = P1.PRIVATE
SEED = 2162
TASK = P1.TASK
TAG = re.compile(r"ppo_ratio_pilot_s2162_[0-9]{8}[a-z0-9_]{0,16}")
ARMS, INSPECTION_PROBES = P1.ARMS, P1.INSPECTION_PROBES
NEW_FILES = ("evals/ppo_ratio_pilot_v2.py", "evals/ppo_ratio_repair_execution_v2.py",
             "bench/run_ppo_ratio_pilot_v2.ps1", "tests/test_ppo_ratio_pilot_v2.py",
             "tests/test_ppo_ratio_repair_execution_v2.py")
CODE = P1.CODE | set(NEW_FILES) | {"bench/protocols/ppo_ratio_pilot_v1.json"}
EVIDENCE = P1.EVIDENCE
read, git = P1.read, P1.git
validated_reports, source_gate, inspection_outcome, _duration = (
    P1.validated_reports, P1.source_gate, P1.inspection_outcome, P1._duration)


def fixed_protocol() -> dict:
    value = P1.fixed_protocol()
    value.update(schema="ppo_ratio_pilot_protocol_v2", seed=SEED)
    value["gpu"] = {**value["gpu"], "profile": E.GPU_PROFILE, "base_profile": E.BASE_PROFILE,
                    "added_display_process_names": list(E.EXTRA_DISPLAY_PROCESS_NAMES),
                    "unchanged": "계산 작업 차단, 메모리 4096 MiB 상한, 프로세스·엔진 종류 기준은 기존 프로필과 같다",
                    "pre_attempt_amendment": {
                        "previous_commit": "5d04228aea4cada4c3750d265d609c2996a1a168",
                        "previous_profile": "desktop_shared_recording_claude_v1",
                        "cause": ("첫 시도 전 같은 프로필로 읽기 전용 GPU 점검을 했을 때 cursor.exe(C+G, 계산 작업 패턴 "
                                  "아님, 활성 엔진 없음)를 허용 목록 밖으로 거부했다"),
                        "change": ("cursor.exe와 기존 base_com_reset_readback_v2 프로필이 허용한 antigravity.exe를 더하고 "
                                   "프로필 이름을 바꿨다"),
                        "seed_2162_used_before": False}}
    value["infrastructure_retry"] = {
        "previous_commit": "bb27b73708bdeba9ad9a7437f9d42ec9f25ff90a",
        "previous_tag": "ppo_ratio_pilot_s2160_20261022",
        "previous_protocol": "bench/protocols/ppo_ratio_pilot_v1.json",
        "previous_protocol_sha256_lf": "000b89266ca86868fdde23f34ff24b2a653a4bc9af40aa23e15440dbe46a0ff3",
        "previous_result_sha256": "56692a6fece77257bbe8b060d23c030885ad63ac6aacefe314cea69ceb269353",
        "classification": "prelaunch_infrastructure_invalid",
        "cause": ("첫 학습 단계 GPU 사전 점검이 허용 목록 밖 화면 앱 claude.exe 두 개와 msedge.exe 하나(C+G, 계산 작업 "
                  "아님, 메모리 2534 MiB)를 거부했다. 학습 프로세스 시작 전 종료다"),
        "previous_training_started": False, "previous_api_requests": 0,
        "seed_policy": "2160은 학습 없이 시도 기록에 남아 seed guard가 차단한다. 새 seed 2162를 쓰고 실행 직전 재검사한다",
        "preservation": "v1 소스·프로토콜·실패 기록은 보존하며 이 판은 새 tag로 한 번만 실행한다",
        "unchanged": "학습·검사·고정 평가·판정·고리 재생은 v1과 같다",
    }
    return value


def dynamic(isaaclab: Path) -> dict:
    runtime = R1S.isaaclab_sources(isaaclab)
    return {"source_sha256_lf": {name: E.sha(ROOT / name, True) for name in sorted(CODE)},
            "evidence_sha256_lf": {name: E.sha(ROOT / name, True) for name in EVIDENCE},
            "runtime_source_sha256_lf": {row["path"]: row["sha256_lf"] for row in runtime.values()}}


def prepare(isaaclab: Path) -> dict:
    E.write_new(PROTOCOL, {**fixed_protocol(), **dynamic(isaaclab)})
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


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
    return {**P1.binding(protocol, anchor, checkpoint, updates),
            "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256_lf": E.sha(PROTOCOL, True)}


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
    result = {"schema": "ppo_ratio_pilot_result_v2", "tag": tag, "seed": SEED, "experiment_commit": anchor,
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
        result["source_gate"] = source_gate(reports, read(ROOT / P1.BANDS_EVIDENCE)["bands"])
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
        for arm in ARMS:
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
                 "launch_paths": E.preflight(args.isaaclab), "gpu": E.query_gpu()}
    elif args.command == "fresh-seeds":
        value = assert_fresh_seeds([SEED], args.isaaclab, repo=ROOT)
    else:
        value = execute(args.tag or "", args.isaaclab)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
