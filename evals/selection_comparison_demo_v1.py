"""같은 공개 정보에서 네 검사 선택 정책을 비교하는 새 seed 개발 측정."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import residual_demo_v2 as RESIDUAL  # noqa: E402
import selection_comparison_controller_v1 as CONTROL  # noqa: E402
import selection_comparison_execution_v1 as EXECUTION  # noqa: E402

PROTOCOL = ROOT / "bench/protocols/selection_comparison_demo_v1.json"
PRIVATE = ROOT / "bench/private/claude_reward_track_20261013/selection_comparison/runs"
SEED = 2170
TAG = re.compile(r"selection_comparison_s2170_[0-9]{8}[a-z0-9_]{0,12}")
POLICIES = tuple(CONTROL.POLICIES)
NEW_FILES = {
    "evals/selection_comparison_demo_v1.py",
    "evals/selection_comparison_flow_v1.py",
    "evals/selection_comparison_conditional_call_v1.py",
    "evals/selection_comparison_execution_v1.py",
    "evals/selection_comparison_gpu_v1.py",
    "evals/selection_comparison_worker_v1.py",
    "tests/test_selection_comparison_demo_v1.py",
    "tests/test_selection_comparison_flow_v1.py",
    "tests/test_selection_comparison_conditional_call_v1.py",
    "tests/test_selection_comparison_execution_v1.py",
    "tests/test_selection_comparison_gpu_v1.py",
    "tests/test_selection_comparison_worker_v1.py",
    "bench/run_selection_comparison_demo_v1.ps1",
}
CODE = (set(RESIDUAL.CODE) | set(CONTROL.SOURCE_FILES) | set(CONTROL.CALL.SOURCE_FILES)
        | NEW_FILES | {"bench/protocols/selection_comparison_controller_v1.json",
                       "bench/protocols/selection_comparison_call_v2.json",
                       "bench/record_evidence_media.py",
                       "bench/ppo_ratio_mutation_v1.py",
                       "bench/ppo_ratio_repair_training.py",
                       "bench/upstream_faults.py",
                       "evals/export_ppo_ratio_repair_v2.py",
                       "evals/ppo_gradient_invariant_v1.py",
                       "evals/ppo_gradient_invariant_v2.py",
                       "evals/ppo_ratio_pilot_v1.py",
                       "evals/ppo_ratio_pilot_v2.py",
                       "evals/ppo_ratio_repair_contract.py",
                       "evals/ppo_ratio_repair_contract_v2.py",
                       "evals/ppo_ratio_repair_demo_v2.py",
                       "evals/ppo_ratio_repair_demo_v3.py",
                       "evals/ppo_ratio_repair_demo_v4.py",
                       "evals/ppo_ratio_repair_model_call_v2.py",
                       "src/rl_triage/grounded_evidence.py",
                       "src/rl_triage/runtime_observers.py",
                       "src/rl_triage/single_run.py"})
EVIDENCE = set(RESIDUAL.EVIDENCE) | {
    "evals/results/selection_runtime_v2_20261023/summary.json",
    "evals/results/selection_runtime_v2_20261023/provenance.json",
}
read, write_new, sha = RESIDUAL.read, RESIDUAL.write_new, RESIDUAL.sha


def api_module():
    import selection_comparison_conditional_call_v1
    return selection_comparison_conditional_call_v1


def fixed_protocol() -> dict:
    return {
        "schema": "selection_comparison_demo_protocol_v1", "seed": SEED, "task": RESIDUAL.TASK,
        "measurement_kind": "one_paired_source_development_comparison",
        "design_coupling": (
            "설계자가 residual 2120·2121의 두 교정 회복과 D2의 여섯 모델 대화를 본 뒤 만든 비교다. "
            "최종 실행 계약을 고정하기 전에 C4 seed 2165의 모델 계약 위반도 확인했다. "
            "새 seed 한 개의 결과이며 네 정책을 독립 사례 네 개로 세지 않는다."
        ),
        "hypothesis": "같은 공개 정보와 검사·교정 표에서 선택 정책에 따른 검사 순서와 회복 경로를 기록할 수 있다.",
        "prediction": {
            "NONE": "walks", "COMPOUND": "not_recovered", "first_repair": "not_recovered",
            "second_repair": "recovered", "model_policy_advantage": "no_directional_prediction",
            "counterexample": "실패 부재·첫 교정 회복·HOLD를 보존하며 seed나 결함 강도를 바꾸지 않는다",
        },
        "policies": list(POLICIES), "policy_order": list(POLICIES),
        "shared": ["source_NONE", "source_COMPOUND", "source_fixed_evaluation", "NONE_legacy_six", "NONE_optimizer_three_updates"],
        "isolated": ["probe_ledger", "correction_ledger", "probe_execution", "repair_training", "repair_fixed_evaluation", "checkpoint_lineage"],
        "training": {
            "iterations": 300, "num_envs": 4096, "original_faults": ["F1", "R1"],
            "same_case_seed_all_arms": True, "checkpoint_loaded": False,
            "mode": "cumulative_corrections_new_initialization",
            "source_checkpoint": "진단 입력과 계보만 결합하며 가중치 또는 optimizer 상태는 읽어 학습하지 않는다",
        },
        "selection": {
            "shared_view": "selection_comparison_view_v1",
            "registry_next": "등록 후보와 관측에 대한 registry 순서",
            "fixed_eligible": "고정 목록에서 현재 유효한 첫 검사",
            "model_first_then_rule": "각 epoch 첫 선택만 모델. 유효한 선택 이후 등록 순서. 무효 모델 응답은 HOLD",
            "model_adaptive": "각 open 상태에서 같은 공개 view를 모델에 제공",
            "invalid_or_infra": "HOLD_without_rule_fallback",
            "host_owned": ["arguments", "expected_outcomes", "correction_id", "approval", "recovery_verdict"],
            "correction_model_calls": 0,
            "scope": "검사 선택 정책 비교이며 제품의 교정 모델 판단 경로 전체 비교가 아니다",
        },
        "api": {
            "model": CONTROL.CALL.MODEL,
            "accepted_provider_models": sorted(CONTROL.CALL.ACCEPTED_PROVIDER_MODELS),
            "logical_slots": 20, "http_cap": 60, "http_per_slot_cap": 3,
            "semantic_feedback_max": 1, "infrastructure_retry_max": 1,
            "slot_plan": api_module().plan_value(),
            "pending": "전체 실행 invalid; 소비 미확정을 0으로 대체하지 않는다",
            "unneeded": "미발생 epoch와 닫힌 상태의 slot도 사유를 남겨 닫고 요청하지 않는다",
        },
        "exposure": {
            "policy": "model_adaptive", "per_epoch": ["exposure_full", "exposure_masked"],
            "when": "해당 epoch의 첫 유효 검사 관측 뒤",
            "used_for_decision": False, "probe_execution": False,
            "interpretation": "동일 상태의 보조 선택 차이이며 인과 효과나 회복 효과로 해석하지 않는다",
        },
        "budgets": {
            "full_training_runs_max": 10, "fixed_evaluation_stages_max": 9,
            "case_probe_runs_max": 56, "shared_reference_stages": 2,
            "epochs_per_policy": 2, "distinct_corrections_per_policy": 2,
            "probe_parent_wall_budget_s": 1200, "correction_parent_wall_budget_s": 3600,
            "owned_parent_wall_s_max": int(EXECUTION.OWNED_PARENT_WALL_CAP_S),
            "overall_controller_elapsed_s_max": int(EXECUTION.OVERALL_ELAPSED_CAP_S),
            "clock_scope": "controller의 예산 계측 시작부터 계산하며 lease 대기와 실행 전 검증은 포함하지 않는다",
            "meaning": "실행 상한이며 예상 소요 시간이나 GPU 커널 시간 측정값이 아니다",
            "watchdog": "요청 watchdog과 남은 owned·전체 경과 예산 중 최솟값",
        },
        "gpu": {
            "lease": "Use-GpuLease", "contract": "apps_bounded_v1",
            "source": "ppo_ratio_repair_execution_v3.gpu_admission",
            "receipt": "처리된 classifier receipt를 내용·SHA로 재검증하며 원본을 재표기하지 않는다",
            "foreign_workload": "거부 후 중단; 다른 작업을 종료하지 않는다",
        },
        "approval": "사용자 위임 운영자 승인. 검사 실행 전에 propose/approve/consume을 봉인하고 일회 실행 후 receipt를 기록한다",
        "epoch2": "같은 정책의 첫 교정·원장·고정 평가를 재검증한 not_recovered receipt가 있을 때만 진입",
        "evaluation": {
            "protocol": "bench/protocols/fixed_eval_v1.json", "conditions": 26,
            "bands": RESIDUAL.BANDS, "reference": "공유 source NONE 한 개로 고정",
        },
        "metrics": {
            "policy": ["outcome", "epochs", "probe_order", "probe_count", "confirmed_components", "corrections"],
            "model": ["first_response_validity", "final_contract_validity", "HOLD", "infra_errors", "requests", "tokens", "wall_s"],
            "cost": "공유 source/reference와 정책별 소유 프로세스 벽시계 시간을 분리; 참조 비용 중복 청구 없음",
            "unknown_cost": "known subtotal과 unknown을 분리하며 전체 unknown은 null",
            "interpretation": "한 개발 사례의 기술 비교. 등록 결함 밖 대응·우위·절감·통계적 유의성은 주장하지 않는다",
        },
        "success_conditions": ["fresh_source_seed", "source_failure_present", "four_policy_terminal_records",
                               "all_twenty_slots_closed", "no_pending_API_or_ledger_requests",
                               "all_executions_verified", "same_reference_and_isolated_lineage"],
        "outcome_wording": {
            "complete": "한 복합 개발 사례에서 네 선택 정책의 검사·교정·후속 평가 경로를 기록했다.",
            "source_failure_absent": "복합 정책의 실패가 확인되지 않아 선택 비교를 실행하지 않았다.",
            "baseline_invalid": "정상 대조의 걷기 기준이 충족되지 않아 비교를 멈췄다.",
            "HOLD": "필요한 판단 근거가 확정되지 않은 정책은 다음 검사를 요청하지 않았다.",
            "invalid": "출처·실행·원장 결합 검증이 완료되지 않아 비교 판정을 멈췄다.",
        },
    }


def dynamic(isaaclab: Path) -> dict:
    return {
        "source_sha256_lf": {name: sha(ROOT / name, True) for name in sorted(CODE)},
        "evidence_sha256_lf": {name: sha(ROOT / name, True) for name in sorted(EVIDENCE)},
        "runtime_source_sha256_lf": RESIDUAL.D1.dynamic(isaaclab)["runtime_source_sha256_lf"],
    }


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="strict").stdout.strip()


def verify_repository(protocol: dict, anchor: str, expected_protocol_sha: str) -> None:
    """Windows controller와 WSL worker가 같은 commit/index/worktree 봉인을 확인한다."""
    CONTROL.require(sha(PROTOCOL, True) == expected_protocol_sha, "실행 protocol SHA 불일치")
    head = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", anchor, head)
    expected = {**protocol["source_sha256_lf"], **protocol["evidence_sha256_lf"],
                PROTOCOL.relative_to(ROOT).as_posix(): expected_protocol_sha}
    CONTROL.require(set(protocol["source_sha256_lf"]) == CODE
                    and set(protocol["evidence_sha256_lf"]) == EVIDENCE, "source closure 범위 불일치")
    for name, digest in expected.items():
        CONTROL.require(sha(ROOT / name, True) == digest, "worktree source SHA 불일치")
        for version in (f"{anchor}:{name}", f"HEAD:{name}", f":{name}"):
            raw = subprocess.run(["git", "show", version], cwd=ROOT, capture_output=True, check=True).stdout
            CONTROL.require(CONTROL.CALL.raw_sha(raw.replace(b"\r\n", b"\n")) == digest,
                            "commit/index source SHA 불일치")


def verify(isaaclab: Path, anchor: str | None = None, *, committed: bool = True) -> tuple[dict, str]:
    protocol = read(PROTOCOL)
    CONTROL.require(protocol == {**fixed_protocol(), **dynamic(isaaclab)}, "고정 계약·소스·runtime 불일치")
    CONTROL.check()
    anchor = anchor or git("rev-parse", "HEAD")
    if committed:
        verify_repository(protocol, anchor, sha(PROTOCOL, True))
    return protocol, anchor


def binding(protocol: dict, anchor: str, *, applied=None, faults=None, checkpoint=None, updates=300) -> dict:
    value = RESIDUAL.binding(protocol, anchor, applied=applied, faults=faults,
                             checkpoint=checkpoint, updates=updates)
    return {**value, "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
            "protocol_sha256_lf": sha(PROTOCOL, True)}


def prepare(isaaclab: Path) -> dict:
    CONTROL.check()
    value = {**fixed_protocol(), **dynamic(isaaclab)}
    encoded = json.dumps(value, ensure_ascii=False)
    CONTROL.require(not any(text in encoded for text in ("C:\\Users", "/home/", "LIMMM", "bench/private")),
                    "공개 protocol 사용자·비공개 경로")
    write_new(PROTOCOL, value)
    return {"status": "prepared_commit_required", "protocol": PROTOCOL.relative_to(ROOT).as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "gpu-gate", "execute"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default="selection_comparison_s2170_20261023")
    parser.add_argument("--approver", default="codex-main-user-delegated")
    parser.add_argument("--approve-registered-actions", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.isaaclab)
    elif args.command in {"check", "verify"}:
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        result = {"status": "verified", "anchor": anchor, "protocol_sha256_lf": sha(PROTOCOL, True)}
    elif args.command == "fresh-seeds":
        result = RESIDUAL.assert_fresh_seeds([SEED], args.isaaclab, repo=ROOT)
    elif args.command == "gpu-gate":
        import ppo_ratio_repair_demo_v4
        result = ppo_ratio_repair_demo_v4.gpu_gate()
    else:
        import selection_comparison_flow_v1
        result = selection_comparison_flow_v1.execute(
            args.tag, args.isaaclab, args.approver, args.approve_registered_actions)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
