"""동일 비용 지표·상한 아래 묶음 계측을 측정하는 PPO 계산 검사 v5."""
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
import ppo_gradient_invariant_v2 as G
import optimizer_observability as O
import preregistered_seed_guard as SEED
import run_optimizer_observability_batch as OLD
import ppo_ratio_mutation_v1 as MUT
import ppo_gradient_check_v4 as V4
import isaac_training_guard_v1 as GUARD
from lease_timestamps import parse_lease_timestamp

PROTOCOL = ROOT / "bench/protocols/ppo_gradient_check_v5.json"
PRIVATE = ROOT / "bench/private/codex_support_20261015/task37"
SEEDS = (2144, 2145, 2146)
TASK = "Isaac-Velocity-Flat-Unitree-Go2-v0"
TAG = re.compile(r"ppo_gradient_v5_[0-9]{8}[a-z0-9_]{0,8}\Z")
BASE_TAG = re.compile(r"ppo_gradient_v5_[0-9]{8}[a-z0-9_]{0,4}\Z")
LEASE_OWNER = "rl-triage-ppo-gradient-v5"
NEW_FILES = {
    "evals/ppo_gradient_check_v5.py", "bench/ppo_gradient_training_v5.py",
    "bench/run_ppo_gradient_check_v5.ps1", "tests/test_ppo_gradient_check_v5.py",
    "evals/ppo_gradient_invariant_v2.py", "tests/test_ppo_gradient_invariant_v2.py",
    "bench/profile_ppo_observer_v1.py", "tests/test_ppo_observer_profile_v1.py",
}
CODE = V4.CODE | NEW_FILES | {"bench/protocols/ppo_gradient_check_v4.json"}
RUNTIME = (
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/algorithms/ppo.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/storage/rollout_storage.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/modules/actor_critic.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl/runners/on_policy_runner.py",
    "_isaac_sim/kit/python/Lib/site-packages/rsl_rl_lib-2.3.3.dist-info/METADATA",
    "scripts/reinforcement_learning/rsl_rl/train.py",
    "_isaac_sim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py",
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


def dry_spec() -> dict:
    return {"id": "dry_s2143_on", "arm": "NONE", "seed": 2143, "observed": True, "updates": 1}


def fixed_protocol() -> dict:
    value = V4.fixed_protocol()
    value.update(schema="ppo_gradient_check_protocol_v5", seeds=list(SEEDS), runs=specs())
    value["supersedes"] = {
        "protocol": "bench/protocols/ppo_gradient_check_v4.json",
        "experiment_commit": "b40d9925e1c0996140134c5f5d621d582877b54c",
        "tag": "ppo_gradient_v4_20261017", "outcome": "observer_cost_exceeded", "status": "complete",
        "result_sha256_lf": "0989938d5583afd5d2bef3b3b3863cc9cca6b59652b102528265980a591828a0",
        "completed_runs": 6, "planned_runs": 8,
        "api_requests": 0,
        "update_wall_ratios": {"2140": 1.5643724637519756, "2141": 1.5218924266757203, "2142": 1.6318819312298947},
        "reason": "v4는 실측 결과다. 정상 3쌍 state·loss 일치와 normal을 확인했지만 비용비가 모두 1.5를 넘어 오류·복원 arm을 실행하지 않았다.",
        "seed_disposition": {"2139": "선행 점검 소비, 재사용하지 않음", "2140": "본배치 소비, 재사용하지 않음",
            "2141": "본배치 소비, 재사용하지 않음", "2142": "본배치 소비, 재사용하지 않음",
            "2137": "미사용 확인. 과거 판 코호트와 구분하기 위해 재사용하지 않음",
            "2138": "미사용 확인. 과거 판 코호트와 구분하기 위해 재사용하지 않음"},
        "development_coupling": "v4 결과를 본 뒤 계측 구현을 바꿨고 지표·상한은 유지했다. invalid 재시도가 아닌 새 seed 측정이다.",
        "previous_history": value["supersedes"],
    }
    value["observer_implementation"] = {
        "module": "evals/ppo_gradient_invariant_v2.py",
        "capture": "backward hook에서 log-prob·gradient·advantage·old-log-prob를 detach·clone한다. 파라미터별 norm은 원래 optimizer.step 직전에 계산한다.",
        "aggregate": "update 끝에서 동일 shape·dtype·device batch를 묶어 기존 수식·허용치로 계산한다. scalar 증거를 device별로 묶어 한 번 CPU로 읽는다.",
        "timing_boundary": "snapshot·계산·CPU 읽기·증거 직렬화 모두 기존 timed update 내부다. 비용을 측정 구간 밖으로 옮기지 않는다.",
        "noninterference": "추가 backward·autograd.grad·RNG 소비·학습 텐서 변경 없음. state·loss 완전 일치를 요구한다.",
        "evidence": "기존 기록 schema·필드·unknown 조건·검사 evaluator·mutation·판정 문구를 유지한다. 메모리는 한 update의 관측 batch로 제한한다.",
        "cpu_development": "이미 본 합성 seed 17의 실제 RSL CPU learn으로 검증한다. 각 판 warmup 1회 뒤 v4/v5 순서를 번갈아 6쌍, 3 update 합·중앙값·비율을 보고한다. cProfile은 별도 실행한다. CPU 수치는 GPU 판정 근거가 아니다.",
    }
    value["dry_run"]["spec"] = dry_spec()
    return value


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
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("GPU lease 검증 기준 시각에 시간대가 없다")
    heartbeat = parse_lease_timestamp(metadata["heartbeatUtc"])
    acquired = parse_lease_timestamp(metadata["acquiredUtc"])
    if acquired > heartbeat:
        raise ValueError("GPU lease heartbeat가 취득 시각보다 이르다")
    age = (now - heartbeat).total_seconds()
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
        "task": TASK, "num_envs": 4096, "max_iterations": spec.get("updates", 3),
        "configured_learning_epochs": 5, "configured_mini_batches": 4,
        "expected_optimizer_steps_per_update": 20, "expected_minibatches_per_update": 20,
        "runtime_updates_to_observe": spec.get("updates", 3), "source_commit_sha": anchor,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256_lf": sha(PROTOCOL),
        "source_sha256_lf": protocol["source_sha256_lf"], "gradient_invariant": protocol["invariant"],
        "inspection_primitives": ["optimizer_step_count", "minibatch_count", G.PRIMITIVE],
        "observer_enabled": spec["observed"], "run_id": spec["id"]})


def training_args(isaaclab: Path, folder: Path, spec: dict, tag: str) -> list[str]:
    return [str(isaaclab / "_isaac_sim/python.bat"), "-B", str(ROOT / "bench/ppo_gradient_training_v5.py"),
        "--manifest", str(folder / "manifest.json"), "--output", str(folder), "--",
        "--task", TASK, "--num_envs", "4096", "--max_iterations", str(spec.get("updates", 3)), "--seed", str(spec["seed"]),
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
    GUARD.validate_completion(folder, O.manifest_fingerprint(fixed))
    result = read(folder / "training_evidence.json")
    updates = spec.get("updates", 3)
    if (result.get("schema") != "ppo_gradient_training_evidence_v1"
            or result["manifest_sha256"] != O.manifest_fingerprint(fixed) or type(result["updates"]) is not int
            or result["updates"] != updates or result.get("torch_version") != protocol["runtime_versions"]["torch"]
            or result.get("device_type") != "cuda"
            or result["runtime_configuration"] != {"epochs": 5, "minibatches": 4, "clip_param": 0.2}
            or result.get("learning_completed") is not True or result.get("recorded_before_app_close") is not True
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
    validate_learning_evidence(result, updates)
    if len(result["update_wall_s"]) != updates or any(type(x) not in (int, float) or not 0 < x < 300
                                               for x in result["update_wall_s"]):
        raise ValueError("update 시간 기록 불완전")
    rows = OLD.INSPECT.read_jsonl(folder / "records.jsonl")
    if spec["observed"]:
        if [row["update_index"] for row in rows] != list(range(updates)) or any(row["error"] for row in rows):
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
                                  ("manifest.json", "records.jsonl", "training_evidence.json", "close_ready.json")}}


def validate_learning_evidence(result: dict, updates: int = 3) -> None:
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
    if not isinstance(losses, list) or len(losses) != updates:
        raise ValueError("등록 update 수만큼 loss가 필요하다")
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
    return {"schema": "ppo_gradient_check_summary_v5", "status": "complete", "outcome": outcome,
            "checks": checks, "pairs": paired, "runs": rows, "completed_runs": len(rows),
            "planned_runs": 8, "wording": fixed_protocol()["wording"][outcome],
            "behavior_recovery_measured": False, "api_requests": 0}


def run(tag: str, isaaclab: Path, *, dry_run: bool = False) -> dict:
    base = tag[:-4] if dry_run and tag.endswith("_dry") else tag
    if not BASE_TAG.fullmatch(base) or (dry_run and tag != base + "_dry"):
        raise ValueError("미등록 tag 형식")
    isaaclab = isaaclab.resolve()
    protocol, anchor = verify(isaaclab)
    lease = lease_context()
    dry_binding = None if dry_run else verify_dry_run(tag + "_dry", isaaclab)
    active_specs = [dry_spec()] if dry_run else specs()
    seeds = [dry_spec()["seed"]] if dry_run else list(SEEDS)
    initial_freshness = SEED.assert_fresh_seeds(seeds, isaaclab)
    for seed in seeds:
        if (PRIVATE / "consumed" / f"s{seed}.json").exists():
            raise FileExistsError("이 protocol의 seed는 이미 소비됐다")
    target = PRIVATE / ("dry_runs" if dry_run else "runs") / tag
    target.mkdir(parents=True, exist_ok=False)
    write_new(target / "plan.json", {"experiment_commit": anchor, "protocol_sha256_lf": sha(PROTOCOL),
                                    "runs": active_specs, "dry_run": dry_run, "dry_run_binding": dry_binding, "runtime_root": str(isaaclab.resolve()), "lease": lease,
                                    "freshness": initial_freshness})
    rows, started = [], set()
    result: dict[str, Any] = {"status": "invalid", "outcome": "invalid", "runs": rows}
    try:
        for spec in active_specs:
            pair_freshness = None
            if not dry_run and len(rows) == 6:
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
            try:
                GUARD.validate_completion(folder, O.manifest_fingerprint(fixed))
            except Exception as exc:
                GUARD.record_failure(folder, exc, "parent_completion_check")
                raise
            row = inspect_run(folder, spec, protocol, anchor)
            row["owned_process_wall_s"] = execution["wall_s"]
            write_new(folder / "inspection.json", row)
            rows.append(row)
        else:
            result = summarize_dry(rows) if dry_run else summarize(rows)
    except Exception as exc:
        result.update(error={"type": type(exc).__name__, "message": str(exc)}, runs=rows)
        raise
    finally:
        result.update(experiment_commit=anchor, protocol_sha256_lf=sha(PROTOCOL), tag=tag, dry_run=dry_run)
        result["artifact_sha256_lf"] = {path.relative_to(target).as_posix(): sha(path)
            for path in sorted(target.rglob("*")) if path.is_file()}
        write_new(target / "result.json", result)
    return result



def summarize_dry(rows: list[dict]) -> dict:
    if (len(rows) != 1 or rows[0]["id"] != dry_spec()["id"]
            or rows[0]["inspection"]["status"] != "normal" or len(rows[0]["losses"]) != 1):
        raise ValueError("실제 Isaac 한 update 점검이 통과하지 못했다")
    return {"schema": "ppo_gradient_dry_run_v5", "status": "complete", "outcome": "dry_run_passed",
            "runs": rows, "completed_runs": 1, "api_requests": 0, "measurement": False}


def verify_dry_run(tag: str, isaaclab: Path) -> dict:
    if not TAG.fullmatch(tag):
        raise ValueError("미등록 dry tag")
    target = PRIVATE / "dry_runs" / tag
    result = read(target / "result.json")
    protocol, anchor = verify(isaaclab, result["experiment_commit"])
    if (result.get("dry_run") is not True or result.get("tag") != tag
            or result.get("protocol_sha256_lf") != sha(PROTOCOL)):
        raise ValueError("dry-run 출처 결합이 다르다")
    hashes = {p.relative_to(target).as_posix(): sha(p) for p in sorted(target.rglob("*"))
              if p.is_file() and p != target / "result.json"}
    if hashes != result.get("artifact_sha256_lf"):
        raise ValueError("dry-run 원본 산출물이 바뀌었다")
    folder = target / dry_spec()["id"]
    execution = read(folder / "execution.json")
    OLD.verify_execution(execution)
    row = inspect_run(folder, dry_spec(), protocol, anchor)
    row["owned_process_wall_s"] = execution["wall_s"]
    expected = summarize_dry([row])
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("dry-run 요약과 실행 원본이 다르다")
    return {"tag": tag, "result_sha256_lf": sha(target / "result.json"), "experiment_commit": anchor}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "verify", "fresh-seeds", "plan", "run", "dry-run", "run-all"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag", default="ppo_gradient_v5_20261017")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        write_new(PROTOCOL, {**fixed_protocol(), **dynamic(args.isaaclab)})
        value = {"status": "prepared_commit_required"}
    elif args.command in {"check", "verify"}:
        _, anchor = verify(args.isaaclab, committed=args.command == "verify")
        value = {"status": "valid", "anchor": anchor, "committed_gate": args.command == "verify"}
    elif args.command == "fresh-seeds":
        value = SEED.assert_fresh_seeds([dry_spec()["seed"], *SEEDS], args.isaaclab)
    elif args.command == "plan":
        value = fixed_protocol()
    elif args.command == "dry-run":
        value = run(args.tag + "_dry", args.isaaclab, dry_run=True)
    elif args.command == "run-all":
        run(args.tag + "_dry", args.isaaclab, dry_run=True)
        value = run(args.tag, args.isaaclab)
    else:
        value = run(args.tag, args.isaaclab)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
