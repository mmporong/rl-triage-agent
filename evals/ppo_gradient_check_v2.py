"""launch 기록 소유권을 고정한 PPO 계산 검사 v2를 사전등록·실행한다."""
from __future__ import annotations

import argparse
import ast
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]
import ppo_gradient_invariant_v1 as G
import optimizer_observability as O
import preregistered_seed_guard as SEED
import run_optimizer_observability_batch as OLD
import ppo_ratio_mutation_v1 as MUT
import ppo_gradient_check_v1 as V1

PROTOCOL = ROOT / "bench/protocols/ppo_gradient_check_v2.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task34"
SEEDS = (2133, 2134, 2135)
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
TAG = re.compile(r"ppo_gradient_v2_[0-9]{8}[a-z0-9_]{0,8}\Z")
LEASE_OWNER = "rl-triage-ppo-gradient-v2"
NEW_FILES = {
    "evals/ppo_gradient_check_v2.py", "bench/ppo_gradient_training_v2.py",
    "bench/run_ppo_gradient_check_v2.ps1", "tests/test_ppo_gradient_check_v2.py",
}
CODE = V1.CODE | NEW_FILES | {"bench/protocols/ppo_gradient_check_v1.json"}
RUNTIME = (
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/storage/rollout_storage.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl_lib-2.3.3.dist-info/METADATA",
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
)
read = OLD.INSPECT.read_json
write_new = OLD.write_new


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                           text=True, encoding="utf-8", errors="strict").stdout.strip()


def specs() -> list[dict]:
    rows = []
    for index, seed in enumerate(SEEDS):
        for observed in ((False, True) if index % 2 == 0 else (True, False)):
            rows.append({"id": f"none_s{seed}_{'on' if observed else 'off'}", "arm": "NONE",
                         "seed": seed, "observed": observed})
    rows += [{"id": f"{arm.lower()}_s{SEEDS[0]}_on", "arm": arm,
              "seed": SEEDS[0], "observed": True} for arm in ("INVERSE_RATIO", "RESTORED")]
    return rows


def fixed_protocol() -> dict:
    return {"schema": "ppo_gradient_check_protocol_v2", "status": "preregistered_before_gpu_measurement",
        "measurement_kind": "development", "task": TASK, "seeds": list(SEEDS), "runs": specs(),
        "runtime_versions": {"torch": "2.7.0+cu128", "rsl_rl": "2.3.3"},
        "supersedes": {"protocol": "bench/protocols/ppo_gradient_check_v1.json",
            "experiment_commit": "c8e9fb51f7db04dfd6c3c7a59f20a29686331f78",
            "tag": "ppo_gradient_v1_20261017", "outcome": "invalid",
            "result_sha256_lf": "bec3ea4b5af979d1e38f131c0839536101ccb03a0ae47c3e158a0e0e9e9055b9",
            "stage": "before_owned_worker_creation", "training_executions": 0, "api_requests": 0,
            "reason": "controller가 먼저 만든 launch.json을 run_owned가 배타 생성하면서 충돌했다. 기존 invalid·소비 기록은 보존한다.",
            "seed_disposition": {"2130": "행정 소비 기록이 있어 재사용하지 않음; 학습 미시작",
                                 "2131": "소비 기록과 학습 실행 없음", "2132": "소비 기록과 학습 실행 없음"},
            "retry_basis": "학습·계산 검사 결과가 나오기 전 실행 계층 오류다. 결과에 따른 판정 변경 없이 새 cohort 2133·2134·2135를 사용한다."},
        "launch_contract": {"folder_owner": "controller creates each arm folder once",
            "intent_owner": "controller: launch_intent.json",
            "reserved_owner": "run_owned: launch.json, stdout.log, stderr.log",
            "execution_owner": "controller: execution.json after run_owned returns",
            "validation": "reserved 파일이 없을 때만 run_owned 호출; intent와 생성된 launch의 args/cwd 일치 검증; training 진입에서도 같은 계약 확인",
            "regression": "실제 Windows Job run_owned로 CPU child를 실행한다. launcher·WindowsJob·Popen은 대체하지 않는다."},
        "design_coupling": "R1 횟수 검사와 이전 수리 결과를 본 설계자가 만든 검사다. inverse ratio 오류는 CPU fixture로 확인한 인공 계산 오류이며 외부 버그 재현이 아니다.",
        "question": "갱신 횟수가 정상인 PPO 계산 오류를 clipped surrogate gradient 불변식으로 가르는가",
        "budget": {"num_envs": 4096, "updates": 3, "epochs": 5, "minibatches": 4,
                   "max_gpu_processes": 8, "watchdog_s": 300, "api_cap": 0},
        "invariant": {"schema": G.GRADIENT_SCHEMA, "atol": G.ATOL, "rtol": G.RTOL,
                      "scale": G.SCALE, "clip_param": G.CLIP_PARAM},
        "formula": "r=exp(new-old); N*dL/dnew=-A*r if (A>=0 and r<=1+eps) or (A<0 and r>=1-eps), else 0",
        "scope": "float32 feedforward single-GPU PPO, no RND/symmetry/per-minibatch advantage normalization; missing/nonfinite evidence is unknown. 반환 log-prob 이후의 정책 손실 미분만 검사하며 advantage 입력·정책 Jacobian 전체의 정확성을 증명하지 않는다.",
        "mutation": "PPO.update ratio 식 하나의 new-old를 old-new로 교환. 설치 파일은 보존하고 인스턴스 메서드만 바꾼다. 판정기는 arm/readback이 아닌 관측 gradient로 검사한다.",
        "prediction": {"normal": "세 seed 모두 count와 gradient normal",
                       "inverse_ratio": "count normal, 첫 3 update 안 gradient abnormal",
                       "restored": "변이 메서드 설치 후 원래 PPO.update로 복원하고 학습 전 identity를 확인한 새 초기화 대조에서 count와 gradient normal"},
        "noninterference": "같은 seed OFF/ON의 policy/optimizer tensor·CPU/CUDA torch RNG·learning rate·loss 완전 일치. 복원 arm도 같은 seed NONE와 일치.",
        "cost": {"metric": "paired_sum_update_wall_on_divided_by_off", "upper_bound_each_seed": 1.5,
                 "timing": "모든 arm의 update 전후 같은 CUDA synchronization; 시작·rollout·최종 상태 저장은 제외. owned process wall은 별도 기록",
                 "design_threshold": "실행 전 정한 허용 오버헤드이며 측정값에 맞춰 보정하지 않는다"},
        "sequence": "정상 OFF/ON 3쌍을 먼저 완료하고 ON 검사 모두 normal·비간섭·비용 관문 통과 때만 오류·복원 측정을 실행한다",
        "lease": "공용 Use-GpuLease 소유자·관리 child PID(venv launcher 포함)·wrapper 부모 PID·현재 호출 프로세스의 조상 계층·활성 heartbeat·배타 lock을 확인한다. 각 arm 전과 학습 진입에서 같은 lease를 다시 확인한다",
        "stop": "출처/실행/완료 오류는 invalid와 배치 중단. 정상 쌍 비간섭 또는 비용 관문 실패는 기록하고 오류 측정을 열지 않는다. unknown을 normal로 바꾸지 않는다",
        "freshness": "실행 시작과 각 정상 seed 쌍 시작 전에 현재 저장소와 Isaac 로그 재검사. seed는 쌍 단위 한 번 소비하며 같은 태그 재사용 금지",
        "success": "정상3/3 normal, 오류 count normal+gradient abnormal, 복원 normal, 세 비간섭·비용 쌍 통과, 복원상태=정상. 모든 조건을 만족할 때만 confirmed",
        "wording": {
            "confirmed": "개발 측정에서 갱신 횟수가 같은 inverse-ratio 오류를 계산 불변식으로 구분했고 등록한 비간섭·비용 기준을 통과했다.",
            "observer_interference": "계측 OFF/ON의 학습 상태가 달라 오류 측정을 진행하지 않았다.",
            "observer_cost_exceeded": "계측 비용이 사전등록 상한을 넘어 오류 측정을 진행하지 않았다.",
            "normal_baseline_unconfirmed": "정상 세 seed의 검사 결과가 모두 normal이 아니어서 오류 측정을 진행하지 않았다.",
            "not_confirmed": "개발 측정이 새 계산 검사 등록 조건을 충족하지 못했다.",
            "invalid": "실행 또는 출처 결합이 완결되지 않아 계산 검사 성능을 판정하지 않았다."},
        "claim_scope": "계산 일치 검사다. 행동 실패·회복·종단 교정은 측정하지 않는다. 모델 호출과 자동 교정 등록은 없다.",
        "references": ["https://arxiv.org/abs/1707.06347",
            "https://github.com/leggedrobotics/rsl_rl/blob/v2.3.3/rsl_rl/algorithms/ppo.py",
            "https://github.com/leggedrobotics/rsl_rl/blob/v2.3.3/rsl_rl/storage/rollout_storage.py"],
        "references_checked": "2026-10-17"}


def dynamic(isaaclab: Path) -> dict:
    return {"source_sha256_lf": {p: sha(ROOT / p) for p in sorted(CODE)},
            "runtime_source_sha256_lf": {p: sha(isaaclab / p) for p in RUNTIME},
            "lease_helper_sha256_lf": sha(Path.home() / "CodexSignals/Use-GpuLease.ps1"),
            "mutation_readback": runtime_mutation(isaaclab)}


def runtime_mutation(isaaclab: Path) -> dict:
    tree = ast.parse((isaaclab / RUNTIME[0]).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPO")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update")
    _, evidence = MUT.mutation_tree(ast.unparse(method))
    return evidence


def validate_lease(metadata: dict, ancestors: list[dict], child_pid: int, now: datetime) -> dict:
    if (metadata.get("owner") != LEASE_OWNER or metadata.get("state") != "running"
            or metadata.get("childPid") != child_pid or metadata.get("completedUtc") is not None):
        raise ValueError("활성 공용 GPU lease 소유자·child가 다르다")
    by_pid = {row["pid"]: row for row in ancestors}
    wrapper = metadata.get("wrapperPid")
    if (child_pid not in by_pid or wrapper not in by_pid
            or by_pid[child_pid]["parent"] != wrapper
            or by_pid[wrapper]["name"].lower() not in {"powershell.exe", "pwsh.exe"}):
        raise ValueError("현재 프로세스의 부모 계층과 lease PID가 다르다")
    age = (now - datetime.fromisoformat(metadata["heartbeatUtc"].replace("Z", "+00:00"))).total_seconds()
    if not 0 <= age <= 60 or not re.fullmatch(r"[a-f0-9-]{36}", metadata.get("leaseId", "")):
        raise ValueError("GPU lease heartbeat 또는 ID가 유효하지 않다")
    return {key: metadata[key] for key in ("owner", "leaseId", "wrapperPid", "childPid", "acquiredUtc", "heartbeatUtc", "state")}


def lease_context(child_pid: int | None = None) -> dict:
    if os.name != "nt":
        raise RuntimeError("등록한 GPU lease 실행 환경은 Windows다")
    signals = Path.home() / "CodexSignals"
    metadata = {}
    for attempt in range(20):
        metadata = read(signals / "gpu-lease.json")
        if (type(metadata.get("childPid")) is int and metadata.get("state") == "running"
                and (child_pid is None or metadata["childPid"] == child_pid)):
            break
        if attempt < 19:
            time.sleep(0.1)
    script = (f"$nodeId={os.getpid()}; $chain=@(); for($i=0; $i -lt 16 -and $nodeId -gt 0; $i++) {{ "
        "$p=Get-CimInstance Win32_Process -Filter ('ProcessId='+$nodeId); if($null -eq $p){break}; "
        "$chain += @{pid=[int]$p.ProcessId; parent=[int]$p.ParentProcessId; name=$p.Name}; "
        "$nodeId=[int]$p.ParentProcessId }; ConvertTo-Json -InputObject @($chain) -Compress")
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="strict", timeout=30)
    ancestors = json.loads(proc.stdout)
    if not ancestors or ancestors[0]["pid"] != os.getpid():
        raise ValueError("현재 호출 프로세스의 조상 증거가 없다")
    context = validate_lease(metadata, ancestors, child_pid or metadata["childPid"], datetime.now(timezone.utc))
    context["callerPid"] = os.getpid()
    assert_exclusive_lock(signals / "gpu-lease.lock")
    return context


def assert_exclusive_lock(path: Path) -> None:
    """Win32 sharing violation만 활성 배타 lock의 증거로 인정한다."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error == 32:
            return
        raise OSError(error, "공용 GPU lease lock 확인 실패")
    kernel.CloseHandle(handle)
    raise ValueError("공용 GPU lease 배타 lock이 유지되지 않는다")


def verify(isaaclab: Path, anchor: str | None = None, *, committed=True) -> tuple[dict, str]:
    value = read(PROTOCOL)
    if value != {**fixed_protocol(), **dynamic(isaaclab)}:
        raise ValueError("사전등록·소스·런타임 결합이 다르다")
    head = git("rev-parse", "HEAD")
    anchor = anchor or head
    if committed:
        paths = sorted(CODE | {PROTOCOL.relative_to(ROOT).as_posix()})
        git("merge-base", "--is-ancestor", anchor, head)
        git("ls-files", "--error-unmatch", "--", *paths)
        if git("status", "--porcelain", "--", *paths):
            raise ValueError("결합 파일을 먼저 커밋해야 한다")
        for path in paths:
            if git("rev-parse", f"{anchor}:{path}") != git("rev-parse", f"{head}:{path}"):
                raise ValueError("실험 커밋 뒤 결합 blob이 바뀌었다")
    return value, anchor


def manifest(spec: dict, protocol: dict, anchor: str) -> dict:
    return G.validate_manifest({"schema": O.MANIFEST_SCHEMA, "arm": spec["arm"], "seed": spec["seed"],
        "task": TASK, "num_envs": 4096, "max_iterations": 3,
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "expected_optimizer_steps_per_update": 20, "expected_minibatches_per_update": 20,
        "runtime_updates_to_observe": 3, "source_commit_sha": anchor,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256_lf": sha(PROTOCOL),
        "source_sha256_lf": protocol["source_sha256_lf"], "gradient_invariant": protocol["invariant"],
        "inspection_primitives": ["optimizer_step_count", "minibatch_count", G.PRIMITIVE],
        "observer_enabled": spec["observed"], "run_id": spec["id"]})


def training_args(isaaclab: Path, folder: Path, spec: dict, tag: str) -> list[str]:
    return [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/ppo_gradient_training_v2.py"),
        "--manifest", str(folder / "manifest.json"), "--output", str(folder), "--",
        "--task", TASK, "--num_envs", "4096", "--max_iterations", "3", "--seed", str(spec["seed"]),
        "--headless", "--run_name", f"{tag}_{spec['id']}"]


def execute_owned(folder: Path, args: list[str], cwd: Path, watchdog_s: float = 300) -> dict:
    """controller의 의도와 owned launcher의 실행 기록을 서로 다른 파일에 둔다."""
    if not folder.is_dir():
        raise FileNotFoundError("controller가 arm 폴더를 먼저 생성해야 한다")
    reserved = ("launch_intent.json", "launch.json", "stdout.log", "stderr.log", "execution.json")
    if any((folder / name).exists() for name in reserved):
        raise FileExistsError("기존 launch·실행 기록이 있는 arm은 재사용하지 않는다")
    expected = {"args": args, "cwd": str(cwd.resolve())}
    write_new(folder / "launch_intent.json", expected)
    execution = OLD.OWNED.run_owned(folder, args, cwd.resolve(), watchdog_s)
    write_new(folder / "execution.json", execution)
    if read(folder / "launch.json") != expected:
        raise ValueError("run_owned launch와 controller 의도가 다르다")
    OLD.verify_execution(execution)
    return execution


def inspect_run(folder: Path, spec: dict, protocol: dict, anchor: str) -> dict:
    fixed = read(folder / "manifest.json")
    if fixed != manifest(spec, protocol, anchor):
        raise ValueError("arm manifest가 등록 입력과 다르다")
    result = read(folder / "training_evidence.json")
    if (result.get("schema") != "ppo_gradient_training_evidence_v1"
            or result["manifest_sha256"] != O.manifest_fingerprint(fixed) or type(result["updates"]) is not int
            or result["updates"] != 3 or result.get("torch_version") != protocol["runtime_versions"]["torch"]
            or result.get("device_type") != "cuda"
            or result["runtime_configuration"] != {"epochs": 5, "minibatches": 4, "clip_param": 0.2}
            or result["method_cleanup"] is not True or result["arm"] != spec["arm"]
            or result["observed"] is not spec["observed"]):
        raise ValueError("학습 완료·설정·cleanup 결합 불일치")
    expected_readback = {**protocol["mutation_readback"], "arm": spec["arm"],
        "active_mutation": spec["arm"] == "INVERSE_RATIO", "restored_original": spec["arm"] == "RESTORED",
        "restoration_before_training": spec["arm"] == "RESTORED",
        "installed_ast_sha256": protocol["mutation_readback"][
            "mutated_ast_sha256" if spec["arm"] == "INVERSE_RATIO" else "original_ast_sha256"]}
    if result["readback"] != expected_readback:
        raise ValueError("PPO 변이·복원 readback이 등록한 AST와 다르다")
    validate_learning_evidence(result)
    if len(result["update_wall_s"]) != 3 or any(type(x) not in (int, float) or not 0 < x < 300
                                               for x in result["update_wall_s"]):
        raise ValueError("update 시간 기록 불완전")
    rows = OLD.INSPECT.read_jsonl(folder / "records.jsonl")
    if spec["observed"]:
        if [row["update_index"] for row in rows] != [0, 1, 2] or any(row["error"] for row in rows):
            raise ValueError("계측 기록이 완결되지 않았다")
        inspection = G.inspect_records(rows, fixed)
    else:
        if rows:
            raise ValueError("OFF arm에 계측 기록이 있다")
        inspection = None
    return {"id": spec["id"], "arm": spec["arm"], "seed": spec["seed"], "observed": spec["observed"],
            "state": result["state"], "losses": result["losses"], "update_wall_s": result["update_wall_s"],
            "readback": result["readback"], "inspection": inspection,
            "source_sha256_lf": {name: sha(folder / name) for name in
                                  ("manifest.json", "records.jsonl", "training_evidence.json")}}


def validate_learning_evidence(result: dict) -> None:
    state = result["state"]
    keys = {"policy_sha256", "optimizer_sha256", "cpu_rng_sha256", "cuda_rng_sha256", "learning_rate"}
    if not isinstance(state, dict) or set(state) != keys:
        raise ValueError("학습 상태 결합 필드 누락")
    for key in keys - {"learning_rate"}:
        if not isinstance(state[key], str) or not re.fullmatch(r"[a-f0-9]{64}", state[key]):
            raise ValueError("학습 상태 SHA가 유효하지 않다")
    rate = state["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("학습률이 유효하지 않다")
    losses = result["losses"]
    if not isinstance(losses, list) or len(losses) != 3:
        raise ValueError("세 update loss가 필요하다")
    for loss in losses:
        if (not isinstance(loss, dict) or set(loss) != {"value_function", "surrogate", "entropy"}
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in loss.values())):
            raise ValueError("loss 반환값이 등록 범위와 다르다")


def pairs(rows: list[dict]) -> list[dict]:
    indexed = {row["id"]: row for row in rows}
    result = []
    for seed in SEEDS:
        off, on = (indexed[f"none_s{seed}_{mode}"] for mode in ("off", "on"))
        ratio = sum(on["update_wall_s"]) / sum(off["update_wall_s"])
        result.append({"seed": seed, "state_exact": off["state"] == on["state"],
            "losses_exact": off["losses"] == on["losses"], "update_wall_ratio": ratio,
            "cost_within_bound": ratio <= 1.5})
    return result


def summarize(rows: list[dict]) -> dict:
    if len(rows) not in (6, 8) or [
            {key: row[key] for key in ("id", "arm", "seed", "observed")} for row in rows] != specs()[:len(rows)]:
        raise ValueError("등록한 실행 목록·순서와 다르다")
    paired = pairs(rows)
    interference = any(not (pair["state_exact"] and pair["losses_exact"]) for pair in paired)
    costly = any(not pair["cost_within_bound"] for pair in paired)
    normal = [row for row in rows if row["arm"] == "NONE" and row["observed"]]
    normal_ok = all(row["inspection"]["status"] == "normal" for row in normal)
    checks = {"noninterference_3_pairs": not interference, "cost_3_pairs": not costly,
              "normal_3_seeds": normal_ok}
    outcome = ("observer_interference" if interference else "observer_cost_exceeded" if costly
               else "normal_baseline_unconfirmed" if not normal_ok else "not_confirmed")
    if len(rows) == 8:
        indexed = {row["id"]: row for row in rows}
        bad, repaired = (indexed[f"{arm}_s{SEEDS[0]}_on"] for arm in ("inverse_ratio", "restored"))
        checks.update(count_unchanged_in_fault=all(bad["inspection"]["primitives"][key]["status"] == "normal"
                for key in ("optimizer_step_count", "minibatch_count")),
            fault_gradient_abnormal=bad["inspection"]["primitives"][G.PRIMITIVE]["status"] == "abnormal",
            restored_normal=repaired["inspection"]["status"] == "normal",
            restored_state_exact=repaired["state"] == normal[0]["state"] and repaired["losses"] == normal[0]["losses"])
        if all(checks.values()):
            outcome = "confirmed"
    return {"schema": "ppo_gradient_check_summary_v2", "status": "complete", "outcome": outcome,
            "checks": checks, "pairs": paired, "runs": rows, "completed_runs": len(rows),
            "planned_runs": 8, "wording": fixed_protocol()["wording"][outcome],
            "behavior_recovery_measured": False, "api_requests": 0}


def run(tag: str, isaaclab: Path) -> dict:
    if not TAG.fullmatch(tag):
        raise ValueError("미등록 tag 형식")
    isaaclab = isaaclab.resolve()
    protocol, anchor = verify(isaaclab)
    lease = lease_context()
    initial_freshness = SEED.assert_fresh_seeds(SEEDS, isaaclab)
    for seed in SEEDS:
        if (PRIVATE / "consumed" / f"s{seed}.json").exists():
            raise FileExistsError("이 protocol의 seed는 이미 소비됐다")
    target = PRIVATE / "runs" / tag
    target.mkdir(parents=True, exist_ok=False)
    write_new(target / "plan.json", {"experiment_commit": anchor, "protocol_sha256_lf": sha(PROTOCOL),
                                    "runs": specs(), "runtime_root": str(isaaclab.resolve()), "lease": lease,
                                    "freshness": initial_freshness})
    rows, started = [], set()
    result: dict[str, Any] = {"status": "invalid", "outcome": "invalid", "runs": rows}
    try:
        for spec in specs():
            pair_freshness = None
            if len(rows) == 6:
                gate = summarize(rows)
                if not all(gate["checks"].values()):
                    result = gate
                    break
            if spec["seed"] not in started:
                pair_freshness = SEED.assert_fresh_seeds([spec["seed"]], isaaclab)
                write_new(PRIVATE / "consumed" / f"s{spec['seed']}.json",
                          {"seed": spec["seed"], "tag": tag, "protocol_sha256_lf": sha(PROTOCOL)})
                started.add(spec["seed"])
            verify(isaaclab, anchor)
            current_lease = lease_context()
            if current_lease["leaseId"] != lease["leaseId"]:
                raise ValueError("배치 도중 GPU lease가 바뀌었다")
            folder = target / spec["id"]
            folder.mkdir()
            if pair_freshness is not None:
                write_new(folder / "freshness.json", pair_freshness)
            fixed = manifest(spec, protocol, anchor)
            write_new(folder / "manifest.json", fixed)
            write_new(folder / "lease_before.json", current_lease)
            before = OLD.validate_gpu_snapshot(OLD.GPU.query_gpu(OLD.GPU.EXPECTED_GPU_ADMISSION["policy"]))
            write_new(folder / "gpu_before.json", before)
            if before["busy"]:
                raise RuntimeError("GPU 입장 검사 busy")
            args = training_args(isaaclab, folder, spec, tag)
            execution = execute_owned(folder, args, isaaclab, 300)
            after = OLD.validate_gpu_snapshot(OLD.GPU.query_gpu(OLD.GPU.EXPECTED_GPU_ADMISSION["policy"]))
            write_new(folder / "gpu_after.json", after)
            if any(str(item["pid"]) in execution["owned_process_identities"] for item in after["processes"]):
                raise RuntimeError("소유 GPU 프로세스가 종료 뒤 남았다")
            if after["busy"]:
                raise RuntimeError("GPU 종료 후 busy")
            verify(isaaclab, anchor)
            row = inspect_run(folder, spec, protocol, anchor)
            row["owned_process_wall_s"] = execution["wall_s"]
            write_new(folder / "inspection.json", row)
            rows.append(row)
        else:
            result = summarize(rows)
    except Exception as exc:
        result.update(error={"type": type(exc).__name__, "message": str(exc)}, runs=rows)
        raise
    finally:
        result.update(experiment_commit=anchor, protocol_sha256_lf=sha(PROTOCOL), tag=tag)
        result["artifact_sha256_lf"] = {path.relative_to(target).as_posix(): sha(path)
            for path in sorted(target.rglob("*")) if path.is_file()}
        write_new(target / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "plan", "run"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default="ppo_gradient_v2_20261017")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        write_new(PROTOCOL, {**fixed_protocol(), **dynamic(args.isaaclab)})
        value = {"status": "prepared_commit_required"}
    elif args.command in {"check", "verify"}:
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "valid", "anchor": anchor, "committed_gate": args.command == "verify"}
    elif args.command == "fresh-seeds":
        value = SEED.assert_fresh_seeds(SEEDS, args.isaaclab)
    elif args.command == "plan":
        value = fixed_protocol()
    else:
        value = run(args.tag, args.isaaclab)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
