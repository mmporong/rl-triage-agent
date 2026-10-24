"""Stop 지원 Brev의 등록 F1 고리에 대한 실행 전 계약과 공통 검증."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "bench/protocols/brev_recovery_v4.json"
TAG = "brev_recovery_v4_s2196_20261024"
SEED = 2196
DRY_SEED = 2197


def sha(path: Path, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout.strip()


def protocol_sha_at(anchor: str) -> str:
    raw = subprocess.run(["git", "-C", str(ROOT), "show",
                          f"{anchor}:{PROTOCOL.relative_to(ROOT).as_posix()}"],
                         check=True, capture_output=True).stdout
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def verify(anchor: str) -> dict:
    if re.fullmatch(r"[a-f0-9]{40}", anchor) is None:
        raise ValueError("실험 커밋 형식")
    protocol = read(PROTOCOL)
    if protocol["schema"] != "brev_recovery_protocol_v4" or protocol["seed"] != SEED:
        raise ValueError("실행 계약 불일치")
    paths = sorted({*protocol["source_sha256_lf"], PROTOCOL.relative_to(ROOT).as_posix()})
    git("merge-base", "--is-ancestor", anchor, "HEAD")
    git("ls-files", "--error-unmatch", "--", *paths)
    if git("status", "--porcelain", "--", *paths) or git("diff", anchor, "HEAD", "--", *paths):
        raise ValueError("결합 소스의 미커밋 변경 또는 실험 뒤 변경")
    committed = git("show", f"{anchor}:{PROTOCOL.relative_to(ROOT).as_posix()}")
    if json.loads(committed) != protocol or sha(PROTOCOL, True) != protocol_sha_at(anchor):
        raise ValueError("사전등록 커밋 불일치")
    for relative, expected in protocol["source_sha256_lf"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts or "\\" in relative:
            raise ValueError("소스 경로 계약")
        if sha(ROOT / relative, True) != expected:
            raise ValueError("결합 소스 해시 불일치")
    return protocol


def prepare() -> dict:
    pilot = read(ROOT / "bench/protocols/brev_compute_pilot_v1.json")
    previous = read(ROOT / "bench/protocols/selection_comparison_demo_v2.json")
    sources = set(previous["source_sha256_lf"])
    sources.update(path.relative_to(ROOT).as_posix() for directory in ("bench", "evals", "tests")
                   for path in (ROOT / directory).glob("*brev_recovery*v[1234].py"))
    sources.update({"bench/protocols/fixed_eval_v1.json",
                    "evals/preregistered_seed_guard.py",
                    "bench/brev_compute_probe_v1.py", "bench/brev_compute_parent_v1.py",
                    "bench/protocols/selection_comparison_view_v1.json",
                    "bench/protocols/selection_comparison_call_v2.json",
                    "evals/results/p0c_fixed_eval_20261009/summary.json"})
    runtime = previous["runtime_source_sha256_lf"]
    package_paths = {
        "rsl_rl/algorithms/ppo.py": "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
        "rsl_rl/storage/rollout_storage.py": "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/storage/rollout_storage.py",
        "rsl_rl/runners/on_policy_runner.py": "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
    }
    protocol = {
        "schema": "brev_recovery_protocol_v4", "registered_before_measurement": True,
        "seed": SEED, "dry_seed": DRY_SEED, "tag": TAG,
        "supersedes": {"protocol": "bench/protocols/brev_recovery_v3.json", "anchor": "d63d00f725575e80d17231d3c7e3f0f62603c831", "outcome": "dry_training_completed_outer_receipt_missing", "reason_ko": "v3 dry 학습은 20회·14.726192884초로 완료됐으나 같은 프로세스의 Isaac 앱 종료 뒤 바깥 worker 완료 기록이 작성되지 않아 invalid였다. 종료 코드0·GPU정리와 원본 체크포인트를 보존한다. 본 측정·모델 요청은0이다. v4는 dry 학습만 별도 자식으로 실행해 반환 뒤 영수증을 완결한다. 학습·검사·회복 판정 조건은 유지한다."},
        "development_provenance_ko": "F1·R1의 기존 진단·교정 결과와 두 Brev 처리량 기록을 본 뒤 만든 단일 등록 F1의 클라우드 이식 확인이다. 처음 보는 결함 대응을 평가하지 않는다.",
        "hypothesis_ko": "기존 등록 검사·수리·고정 평가 기준을 유지한 Linux 고리에서 F1을 확정하고 보상 함수를 복원한 재학습이 회복 기준을 통과한다.",
        "environment": pilot["environment"],
        "runtime_source_sha256_lf": {key: runtime[key] for key in (
            "scripts/reinforcement_learning/rsl_rl/train.py",
            "source/isaaclab/isaaclab/envs/mdp/rewards.py")},
        "python_package_source_sha256_lf": {key: runtime[value] for key, value in package_paths.items()},
        "source_sha256_lf": {path: sha(ROOT / path, True) for path in sorted(sources)},
        "training": {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "num_envs": 4096,
                     "iterations": 300, "dry_iterations": 20, "initialization": "restart",
                     "arms": ["NONE", "F1", "CORRECTED"],
                     "seed_contract_ko": "본 세 arm은 동일 seed의 짝 조건이며 독립 사례 세 건으로 세지 않는다. 로컬 기록·클라우드 로그의 사용 여부를 시작 전에 재검사한다."},
        "selection": {"policy": "model_adaptive", "model": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                      "max_probes": 7, "max_corrections": 1, "api_cap": 20,
                      "transport": "WSL의 NVIDIA API; 키를 클라우드로 전송하지 않음",
                      "contract": "기존 공통뷰·3필드 선택·고정 피드백1회·인프라재시도1회; invalid/infra/unknown/HOLD를 규칙으로 전환하지 않음",
                      "repair": "새 abnormal 영수증으로 단일 reward 후보가 확정된 경우에만 호스트가 등록 보상 교정과 restart300을 제안하고 위임 운영자 승인을 봉인"},
        "evaluation": {"protocol": "bench/protocols/fixed_eval_v1.json", "conditions": 26,
                       "bands": "evals/results/p0c_fixed_eval_20261009/summary.json",
                       "reference": "같은 seed의 새 NONE 고정 평가"},
        "resources": {"instance_type": "l40s-48gb.1x", "stoppable_required": True,
                      "compute_quote_usd_per_h": 1.74, "max_owned_instances": 1,
                      "total_project_pilot_budget_usd": 50, "host_watchdog_s": 14400,
                      "finish": "결과·체크포인트 회수 뒤 exact ID Stop 및 STOPPED 확인; 저장소 과금은 남을 수 있음",
                      "stage_timeouts_s": {"train": 1800, "probe": 900, "evaluate": 900}},
        "success_conditions": ["fresh_seed", "dry_run_receipt_valid", "NONE_walks",
                               "F1_source_not_recovered", "reward_confirmed_by_new_abnormal_receipt",
                               "single_approved_consumed_correction", "restored_reward_readback",
                               "corrected_fixed_evaluation_recovered", "all_receipts_verified",
                               "no_pending_API_or_ledger_requests", "artifacts_retrieved_and_VM_stopped"],
        "outcome_wording": {"recovered": "등록 F1 한 사례의 클라우드 고리에서 검사 확정·교정 재학습·고정 평가 회복을 확인했다.",
                            "not_recovered": "등록 교정 뒤 고정 평가의 회복 기준을 충족하지 못했다.",
                            "HOLD": "필요한 판단 또는 실행 근거가 확정되지 않아 다음 실행을 요청하지 않았다.",
                            "source_failure_absent": "F1 원본에서 행동 실패가 확인되지 않아 교정 고리를 진행하지 않았다.",
                            "baseline_invalid": "정상 대조가 걷기 기준을 충족하지 않아 고리를 진행하지 않았다.",
                            "invalid": "소스·실행·영수증 결합이 완료되지 않아 고리 결과를 확정하지 않았다."},
        "retry_ko": "완료·실패 기록을 덮어쓰지 않는다. 학습 시작 전 인프라 실패만 새 태그 또는 새 판으로 재시도한다. 학습 뒤 결과가 실패하면 그 결과로 보존하며 의미 재시도를 하지 않는다.",
    }
    write_new(PROTOCOL, protocol)
    return {"status": "prepared_commit_required", "source_files": len(sources)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--anchor")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.command == "prepare" else
                     {"verified": bool(verify(args.anchor))}, ensure_ascii=False))
