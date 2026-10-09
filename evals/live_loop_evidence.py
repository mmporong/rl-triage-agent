"""사전등록한 사례의 실제 Isaac 승인 고리와 비용을 기록한다.

prepare는 기존 입력·코드 해시와 선택 규칙만 새 파일에 쓴다(GPU 없음).
run은 커밋된 사전등록·동일 소스를 요구한다. 기존 모델 순위를 재생하며
승인 주체는 사용자에게 작업을 위임받은 Codex 운영자로 명시한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import audit_experiment as A  # noqa: E402
import loop  # noqa: E402
from loop_compare import load_rankings  # noqa: E402
from run_fixed_eval import find_run, last_checkpoint  # noqa: E402

L = loop.L
ACTOR = "codex_operator_user_authorized_test"
CODE = ("evals/live_loop_evidence.py", "evals/resume_live_loop_evidence.py", "evals/metered_probes.py", "evals/loop.py", "evals/run_probes.py",
        "evals/probes.py", "bench/hidden_faults.py", "bench/external_faults.py", *A._code_sources())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_file(relative):
    return {"path": relative, "sha256_lf": sha(ROOT / relative)}


def prepare(out):
    seed = 2027
    p0_rank = "evals/results/p0c_holdout_20261009/seed2027.jsonl"
    t1_rank = "evals/results/t1_agent_20261009/seed2027.jsonl"
    rank_maps = {"p0": load_rankings(f"agent={ROOT / Path(p0_rank).parent}")["agent"],
                 "t1": load_rankings(f"agent={ROOT / Path(t1_rank).parent}")["agent"]}
    cases = []
    for name in [*(f"h{i:02d}_s{seed}" for i in range(1, 7)), f"baseline_p0c_s{seed}", f"e01_s{seed}"]:
        t1 = name.startswith("e")
        ref = f"ref_e01_s{seed}" if t1 else f"baseline_p0c_s{seed}"
        probes_tag = "t1_probes_20261009" if t1 else "p1a_probes_holdout_20261009"
        fixed_tag = "t1_fixed_eval_20261009" if t1 else "p0c_fixed_eval_20261009"
        paths = {"prior_probes": f"evals/results/{probes_tag}/probes/{name}.json",
                 "probe_reference": f"evals/results/{probes_tag}/probes/{ref}.json",
                 "before": f"evals/results/{fixed_tag}/runs/{name}.json",
                 "reference": f"evals/results/{fixed_tag}/runs/{ref}.json",
                 "telemetry": f"bench/runs_{'t1' if t1 else 'p0c'}/{name}.telemetry.json"}
        ranking = list(L.MECHANISMS) if name.startswith("baseline") else rank_maps["t1" if t1 else "p0"][name]
        if not name.startswith("baseline"):
            paths["ranking_source"] = t1_rank if t1 else p0_rank
        checkpoint_sha = read(ROOT / paths["prior_probes"])["checkpoint"]["sha256"]
        run_dir = find_run(Path.home() / "IsaacLab/logs/rsl_rl/unitree_go2_flat", name)
        actual_sha = hashlib.sha256(last_checkpoint(run_dir).read_bytes()).hexdigest()
        if actual_sha != checkpoint_sha or read(ROOT / paths["before"])["checkpoint"]["sha256"] != checkpoint_sha:
            raise ValueError(f"{name}: 원본 체크포인트와 기존 자료 SHA가 다르다")
        cases.append({"case": name, "seed": seed, "ranking": ranking, "checkpoint_sha256": checkpoint_sha,
                      "ranking_kind": "canonical_category_order" if name.startswith("baseline") else "cached_agent",
                      "sources": {k: frozen_file(p) for k, p in paths.items()}})
    protocol = read(ROOT / "bench/protocols/fixed_eval_v1.json")
    grid = A.command_grid(protocol["command_grid"])
    gates = {"min_survival_fraction": 0.9, "max_fall_rate": 0.05,
             "max_lin_vel_rmse_mps": 0.5 * math.sqrt(sum(x*x + y*y for x, y, _ in grid) / len(grid)),
             "max_yaw_rate_rmse_radps": 0.5 * math.sqrt(sum(z*z for _, _, z in grid) / len(grid))}
    for case in cases:
        A.O.register_contract(read(ROOT / case["sources"]["reference"]["path"]), [], gates)
    result = {"schema": "live_loop_prereg_v1", "prepared_at": now(), "cases": cases,
              "selection_rule": "seed2027_all_six_p0c_faults_plus_none_plus_t1_e01; no outcome selection",
              "top_k": 3, "budget_probes": 4, "reject_rule": "first proposed probe in every case",
              "interrupt_case": "h04_s2027", "interrupt_rule": "first execution after output and execution sidecar, before receipt",
              "actor": ACTOR, "approval_identity_attested": False,
              "diagnosis_scope": "replay_cached_rankings_no_fresh_model_call",
              "repair_scope": "no_repair; audit_finalize_abstain", "absolute_gates": gates,
              "relative_tolerance": 1e-3, "absolute_tolerance": 1e-8,
              "scalar_fields": {p: [v[0]] for p, v in L.THRESHOLDS.items()},
              "limits": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "gpu_wall_s": 480,
                         "cpu_s": 900, "human_review_s": 0, "execution_wall_s": 900, "simulator_steps": 1228800},
              "end_to_end_wall_limit_s": 900, "code_sha256_lf": {p: sha(ROOT / p) for p in dict.fromkeys(CODE)}}
    result["scalar_fields"]["P_reward"].append("track_ang_vel_z_exp_rel_error")
    A._write_new(out, result)
    return result


def gpu_snapshot():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    util, used, total = (int(v.strip()) for v in gpu.split(","))
    cmd = "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'probes.py|fixed_eval.py|smoke_external.py|rsl_rl[\\/]train.py' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True).strip()
    pids = json.loads(raw) if raw else []
    pids = pids if isinstance(pids, list) else [pids]
    try:
        with urlopen("http://127.0.0.1:11434/api/ps", timeout=2) as fh:
            models = len(json.load(fh)["models"])
    except URLError:
        models = 0
    snapshot = {"at": now(), "utilization_pct": util, "memory_mib": used, "memory_total_mib": total,
                "other_isaac_pids": pids, "ollama_models": models}
    if pids or models or util > 10 or used > 2500:
        raise RuntimeError(f"GPU 대기 필요: {snapshot}")
    return snapshot


def verify_prereg(path, prereg):
    relative = path.resolve().relative_to(ROOT).as_posix()
    subprocess.run(["git", "ls-files", "--error-unmatch", relative, *prereg["code_sha256_lf"]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", relative, *prereg["code_sha256_lf"]], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for p, expected in prereg["code_sha256_lf"].items():
        if sha(ROOT / p) != expected:
            raise ValueError(f"동결 코드 변경: {p}")
    for case in prereg["cases"]:
        for source in case["sources"].values():
            if sha(ROOT / source["path"]) != source["sha256_lf"]:
                raise ValueError(f"사전등록 입력 변경: {source['path']}")
    verify_amendment(prereg)


def verify_amendment(prereg):
    amendment = prereg.get("amendment")
    if not amendment:
        return
    if sha(ROOT / amendment["previous_protocol"]) != amendment["previous_sha256_lf"]:
        raise ValueError("이전 사전등록 파일이 달라졌다")
    for source in amendment["prior_artifacts"].values():
        if sha(ROOT / source["path"]) != source["sha256_lf"]:
            raise ValueError(f"최초 실패 증거 변경: {source['path']}")
    prefix = amendment["ledger_prefix"]
    data = (ROOT / prefix["path"]).read_bytes().replace(b"\r\n", b"\n")[:prefix["bytes_lf"]]
    if hashlib.sha256(data).hexdigest() != prefix["sha256_lf"] or len(data.splitlines()) != prefix["line_count"]:
        raise ValueError("최초 원장의 사전등록 prefix가 달라졌다")


def worker(case, rid, tag, interrupt):
    def execute(c, p):
        result = loop.isaac_execute(c, p, tag, metered=True)
        if interrupt and result[2] == 0:
            raise SystemExit(75)
        return result
    probe = loop._load(case)[2].requests()[rid]["probe"]
    try:
        print(loop.run(case, rid, execute, preflight=lambda c, p: loop.isaac_preflight(c, p, tag)), flush=True)
    finally:
        meta_path = loop._execution_path(case, probe, tag)
        if meta_path.exists():
            meta = read(meta_path)
            A._write_new(ROOT / "evals/results" / tag / "workers" / f"{case}__{probe}.json",
                         {"case": case, "probe": probe, "cpu_s": time.process_time(),
                          "cpu_s_in_execution_metadata": meta["parent_cpu_s"],
                          "cpu_scope": "worker_python_process_lifetime_through_loop_run"})
    return 0


def zero_cost(**changes):
    return {**dict.fromkeys(A.C.COST_KEYS, 0), **changes}


def measured_action(events, case, stage, action):
    started, wall0, cpu0 = now(), time.perf_counter(), time.process_time()
    result = action()
    events.append({"attempt_id": f"{case}:{stage}:{len(events)}", "stage": stage, "status": "completed",
                   "started_at": started, "finished_at": now(),
                   "costs": zero_cost(cpu_s=time.process_time() - cpu0, execution_wall_s=time.perf_counter() - wall0)})
    return result


def measurement_agreement(probe, live, prior, reference, prereg):
    fields = prereg["scalar_fields"][probe]
    compared = {key: {"live": live.get(key), "prior": prior.get(key),
                      "within_tolerance": (isinstance(live.get(key), (int, float)) and isinstance(prior.get(key), (int, float))
                                           and math.isfinite(live[key]) and math.isfinite(prior[key])
                                           and math.isclose(live[key], prior[key], rel_tol=prereg["relative_tolerance"],
                                                            abs_tol=prereg["absolute_tolerance"]))} for key in fields}
    return {"probe": probe, "live_outcome": L.classify(probe, live, reference),
            "prior_outcome": L.classify(probe, prior, reference), "scalar_comparison": compared,
            "scalar_agreement": all(c["within_tolerance"] for c in compared.values()),
            "all_live_measurements": live, "all_prior_measurements": prior}


def run_case(path, tag, case_name):
    coordinator_start, coordinator_wall0 = now(), time.perf_counter()
    prereg = read(path)
    verify_prereg(path, prereg)
    spec = next(c for c in prereg["cases"] if c["case"] == case_name)
    case = spec["case"]
    folder = ROOT / "evals/results" / tag / "cases" / case
    if folder.exists() or loop._dir(case).exists():
        raise ValueError("기존 사례·고리 결과는 덮어쓰지 않는다")
    snapshot = gpu_snapshot()
    started_at = A._timestamp(coordinator_start).replace(microsecond=0).isoformat()
    folder.mkdir(parents=True)
    A._write_new(folder / "gpu_before.json", snapshot)
    events = []
    sources = {k: ROOT / v["path"] for k, v in spec["sources"].items()}
    prior = read(sources["prior_probes"])
    ref = {p: row["measurement"] for p, row in read(sources["probe_reference"])["probes"].items()}
    info = measured_action(events, case, "diagnosis", lambda: {"ranking": spec["ranking"], "ranking_kind": spec["ranking_kind"],
                          "telemetry": read(sources["telemetry"]), "source_hashes": spec["sources"]})
    for name, obj in (("information", info), ("gates", prereg["absolute_gates"]), ("limits", prereg["limits"])):
        A._write_new(folder / f"{name}.json", obj)
    measured_action(events, case, "control", lambda: loop.start(case, info["ranking"], prereg["top_k"],
                    prereg["budget_probes"], spec["checkpoint_sha256"], ref))
    args = SimpleNamespace(loop_dir=loop._dir(case), reference=sources["reference"], normals=[],
                           gates=folder / "gates.json", limits=folder / "limits.json", information=folder / "information.json",
                           method="cached_agent_live_loop", wall_budget=prereg["end_to_end_wall_limit_s"],
                           protocol=ROOT / "bench/protocols/fixed_eval_v1.json")
    contract = measured_action(events, case, "control", lambda: A.register(args))
    A._write_new(folder / "contract.json", contract)
    _, state, led = loop._load(case)
    rejected = state["pending"]
    measured_action(events, case, "control", lambda: loop.reject(case, rejected, ACTOR, "사전등록된 첫 제안 거절 시험"))
    events.append({"attempt_id": f"{case}:{rejected}", "status": "rejected",
                   "costs": {"cpu_s": 0, "human_review_s": 0, "simulator_steps": 0}})
    runs, agreements, recovery = [], [], None
    while (state := loop._load(case)[1])["pending"] is not None:
        rid = state["pending"]
        _, _, led = loop._load(case)
        probe = led.requests()[rid]["probe"]
        measured_action(events, case, "control", lambda: led.approve(rid, ACTOR))
        interrupt = case == prereg["interrupt_case"] and not runs
        cmd = [sys.executable, str(ROOT / "evals/live_loop_evidence.py"), "execute", "--case", case,
               "--request-id", rid, "--tag", tag]
        if interrupt:
            cmd.append("--interrupt-after-output")
        process_started = now()
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code not in ((0, 75) if interrupt else (0,)):
            raise RuntimeError(f"{case}/{probe}: 실행 종료 코드 {code}; 산출물·고아 기록을 보존했다")
        interrupted = code == 75
        meta_path = loop._execution_path(case, probe, tag)
        meta = read(meta_path)
        if interrupted:
            before_calls = len(list((ROOT / "evals/results" / tag / "execution").glob("*.json")))
            before_sha = sha(loop._probe_path(case, probe, tag))
            measured_action(events, case, "recovery", lambda: loop.recover(case, lambda c, p, pr: loop.file_artifact(c, p, pr, tag)))
            after_calls = len(list((ROOT / "evals/results" / tag / "execution").glob("*.json")))
            recovery = {"request_id": rid, "orphans_after": led.orphans(), "execution_records_before": before_calls,
                        "execution_records_after": after_calls, "artifact_unchanged": before_sha == sha(loop._probe_path(case, probe, tag)),
                        "extra_gpu_wall_s": 0, "extra_simulator_steps": 0,
                        "basis": "recover_has_no_executor_and_no_new_execution_record"}
        worker_cost = read(ROOT / "evals/results" / tag / "workers" / f"{case}__{probe}.json")
        cpu = None if meta["cpu_s"] is None else (meta["cpu_s"] + worker_cost["cpu_s"] - meta["parent_cpu_s"])
        events.append({"attempt_id": f"{case}:{rid}", "status": "completed" if meta["exit_code"] == 0 else "failed",
                       "costs": {"cpu_s": cpu, "human_review_s": 0, "simulator_steps": meta["simulator_steps"]}})
        report = read(loop._probe_path(case, probe, tag))
        if report["checkpoint"]["sha256"] != spec["checkpoint_sha256"] or report["seed"] != spec["seed"]:
            raise ValueError("실시간 probe의 체크포인트·seed가 다르다")
        live = report["probes"][probe]["measurement"] or {}
        agreement = measurement_agreement(probe, live, prior["probes"][probe]["measurement"], ref.get(probe), prereg)
        agreement["receipt_outcome"] = led.requests()[rid]["receipt"]["outcome"]
        agreement["receipt_agreement"] = agreement["receipt_outcome"] == agreement["prior_outcome"]
        agreements.append(agreement)
        runs.append({"request_id": rid, "probe": probe, "worker_exit": code, "process_started_at": process_started,
                     "receipt_observed_at": now(), "process_to_receipt_s": (datetime.now(timezone.utc) - A._timestamp(process_started)).total_seconds(),
                     "execution": meta_path.relative_to(ROOT).as_posix(), "interrupted_before_receipt": interrupted})
        print(json.dumps({"case": case, "probe": probe, "outcome": agreement["live_outcome"],
                          "prior_outcome": agreement["prior_outcome"], "scalar_agreement": agreement["scalar_agreement"],
                          "gpu_wall_s": meta["gpu_wall_s"]}), flush=True)
    # 동기 하위 실행의 대기 시간은 probe receipt에서 한 번만 센다.
    # 운영자 프로세스의 나머지 CPU·벽시계 구간을 별도 비용으로 보존한다.
    trace = A.audit_loop(*A._loop(loop._dir(case))[:4])
    receipt_wall = sum(e["costs"]["execution_wall_s"] or 0 for e in trace["cost_events"])
    action_cpu = sum(e["costs"]["cpu_s"] for e in events if e.get("stage") in ("diagnosis", "control", "recovery"))
    action_wall = sum(e["costs"]["execution_wall_s"] for e in events if e.get("stage") in ("diagnosis", "control", "recovery"))
    events.append({"attempt_id": f"{case}:coordinator_overhead", "stage": "control", "status": "completed",
                   "started_at": coordinator_start, "finished_at": now(),
                   "costs": zero_cost(cpu_s=max(0, time.process_time() - action_cpu),
                                      execution_wall_s=max(0, time.perf_counter() - coordinator_wall0 - receipt_wall - action_wall))})
    A._write_new(folder / "events.json", events)
    A._write_new(folder / "timeline.json", {"started_at": started_at, "finished_at": now()})
    audit_args = SimpleNamespace(loop_dir=loop._dir(case), contract=folder / "contract.json", reference=sources["reference"],
                                before=sources["before"], after=None, abstain=True, events=folder / "events.json", timeline=folder / "timeline.json")
    audit = A.finalize(audit_args)
    A._write_new(folder / "audit.json", audit)
    result = {"schema": "live_loop_case_v1", "case": case, "prereg_sha256_lf": sha(path),
              "actor": ACTOR, "scope": prereg["diagnosis_scope"], "runs": runs, "recovery": recovery,
              "agreements": agreements, "raw_classification_agreement": all(a["live_outcome"] == a["prior_outcome"] for a in agreements),
              "classification_agreement": all(a["receipt_agreement"] for a in agreements),
              "scalar_agreement": all(a["scalar_agreement"] for a in agreements), "audit_finalize_error": None,
              "loop_status": audit["loop"]["diagnosis_status"], "observed": audit["loop"]["observed"],
              "costs": audit["costs"]["totals"], "end_to_end_wall": audit["end_to_end_wall"],
              "checkpoint_sha256_verified_against_prereg": True, "repair_executed": False,
              "none_wrong_confirmation": case.startswith("baseline") and audit["loop"]["diagnosis_status"] == "confirmed"}
    A._write_new(folder / "result.json", result)
    print(json.dumps({"case": case, "status": result["loop_status"], "costs": result["costs"],
                      "classification_agreement": result["classification_agreement"], "scalar_agreement": result["scalar_agreement"]}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--prereg", type=Path, required=True)
    execute = sub.add_parser("execute")
    for command in (run, execute):
        command.add_argument("--case", required=True)
        command.add_argument("--tag", required=True)
    execute.add_argument("--request-id", required=True)
    execute.add_argument("--interrupt-after-output", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args.out)
    else:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.tag):
            raise SystemExit("안전한 결과 tag가 필요하다")
        if args.cmd == "run":
            run_case(args.prereg, args.tag, args.case)
        else:
            return worker(args.case, args.request_id, args.tag, args.interrupt_after_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
