"""새 Nemotron 선택 → 승인 원장 → 고정 Isaac 측정 → 새 판단의 통합 기록.

prepare로 성공/실패 문구·사례·소스 해시를 먼저 쓰고 커밋한다.
run은 Windows에서 수행하며 모델 호출만 기존 WSL 환경으로 보낸다.
모델 생성 셸/코드/args는 실행하지 않는다. 과거 loop 결과는 수정하지 않는다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from rl_triage import active_diagnosis as D, probe_loop as L  # noqa: E402
import loop  # noqa: E402
from run_fixed_eval import find_run, last_checkpoint  # noqa: E402

ACTOR = "codex_operator_user_authorized_test"
MODEL = "nvidia/nemotron-3-super-120b-a12b"
CODE = ("src/rl_triage/active_diagnosis.py", "src/rl_triage/probe_loop.py",
        "evals/run_active_diagnosis.py", "evals/active_model_call.py", "evals/loop.py",
        "evals/run_probes.py", "evals/probes.py", "evals/metered_probes.py",
        "evals/run_fixed_eval.py", "evals/audit_active_diagnosis.py", "evals/build_active_replay.py",
        "bench/hidden_faults.py", "bench/external_faults.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source(path):
    return {"path": path, "sha256_lf": sha(ROOT / path)}


def prepare(path):
    old = read(ROOT / "bench/protocols/live_loop_v2.json")
    cases = []
    for spec in old["cases"]:
        case = spec["case"]
        t1 = case.startswith("e")
        reference = "ref_e01_s2027" if t1 else "baseline_p0c_s2027"
        tele_root = "bench/runs_t1" if t1 else "bench/runs_p0c"
        inputs = {"telemetry": source(f"{tele_root}/{case}.telemetry.json"),
                  "reference_telemetry": source(f"{tele_root}/{reference}.telemetry.json"),
                  "probe_reference": spec["sources"]["probe_reference"]}
        checkpoint = last_checkpoint(find_run(Path.home() / "IsaacLab/logs/rsl_rl/unitree_go2_flat", case))
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != spec["checkpoint_sha256"]:
            raise ValueError(f"{case}: checkpoint가 기존 동결 입력과 다르다")
        cases.append({"case": case, "checkpoint_sha256": actual, "sources": inputs})
    protocol = {"schema": "active_diagnosis_protocol_v1", "prepared_at": now(),
                "scope": "seen_seed2027_integration_not_accuracy_or_generalization",
                "cases": cases, "model": MODEL, "max_probes": 2, "max_infra_attempts": 2,
                "probe_execution_budget_s": 180, "per_case_isaac_wall_limit_s": 360,
                "budget_scope": "launcher_wall_including_startup_not_device_busy",
                "probe_arguments": {p: v["args"] for p, v in L.PROBES.items()},
                "thresholds": L.THRESHOLDS, "code_sha256_lf": {p: sha(ROOT / p) for p in CODE},
                "approver": ACTOR, "human_review_s": None,
                "model_input": "same_last20pct_summary_and_public_catalog_for_all_controllers; no_fault_keys",
                "baselines": ["fixed_order", "cheapest_discrimination"],
                "baseline_scope": "fresh_independent_metered_executions; same_input_all6_candidates_2probe_180s_each; no_model_ranking_for_rules",
                "ablation": "each_post_stage_pair_differs_only_in_latest_receipt_and_its_metric_facts; alternate_order_by_registered_case_index_plus_stage",
                "spotlight_rule": "first_case_in_registered_order_with_verified_top1_change_and_receipt_claim",
                "success_gate": ["fresh_initial_and_post_model_calls", "model_selected_allowed_probe",
                                 "approval_consumed_once", "checkpoint_and_report_hash_match",
                                 "raw_model_leader_changed", "latest_receipt_metric_cited_exactly",
                                 "previous_leader_rejected_by_actual_normal_or_abnormal_observation",
                                 "post_model_leader_matches_frozen_survivors", "masked_control_retains_prior_leader",
                                 "all_cost_fields_observed_or_explicit_null"],
                "wording": {
                    "change_supported": "이 기록에서는 AI가 고른 검사 결과를 인용해 우선 가설을 바꿨다. 기존 사례의 통합 증거이며 정확도 향상 근거는 아니다.",
                    "narrowing_only": "검사로 가설을 좁혔지만 모델의 우선 가설 변경은 입증하지 못했다.",
                    "unknown": "검사 또는 모델 응답이 불명이라 원인을 확정하지 않고 보류했다.",
                    "baseline_equal_or_better": "단순 규칙보다 나은 진단 또는 비용을 입증하지 못했다.",
                    "no_superiority_claim": "같은 조건의 새 보류 세트 비교 전에는 AI 우월성이나 GPU 절감을 주장하지 않는다."}}
    write_new(path, protocol)
    return protocol


def verify_protocol(path, protocol):
    cases = [*(f"h{i:02d}_s2027" for i in range(1, 7)), "baseline_p0c_s2027", "e01_s2027"]
    if (protocol.get("schema") != "active_diagnosis_protocol_v1" or protocol.get("model") != MODEL
            or protocol.get("max_probes") != 2 or protocol.get("max_infra_attempts") != 2
            or protocol.get("probe_execution_budget_s") != 180 or protocol.get("per_case_isaac_wall_limit_s") != 360
            or protocol.get("approver") != ACTOR or [c.get("case") for c in protocol.get("cases", [])] != cases
            or set(protocol.get("code_sha256_lf", {})) != set(CODE)
            or protocol.get("probe_arguments") != {p: v["args"] for p, v in L.PROBES.items()}
            or protocol.get("thresholds") != {p: list(v) for p, v in L.THRESHOLDS.items()}):
        raise ValueError("동결 프로토콜의 사례·모델·예산·소스·검사 계약 불일치")
    source_paths = []
    for spec in protocol["cases"]:
        case, t1 = spec["case"], spec["case"].startswith("e")
        root = "bench/runs_t1" if t1 else "bench/runs_p0c"
        reference = "ref_e01_s2027" if t1 else "baseline_p0c_s2027"
        tag = "t1_probes_20261009" if t1 else "p1a_probes_holdout_20261009"
        expected = {"telemetry": f"{root}/{case}.telemetry.json",
                    "reference_telemetry": f"{root}/{reference}.telemetry.json",
                    "probe_reference": f"evals/results/{tag}/probes/{reference}.json"}
        if set(spec.get("sources", {})) != set(expected):
            raise ValueError("동결 입력 source 종류 불일치")
        for key, item in spec["sources"].items():
            if item.get("path") != expected[key] or (ROOT / item["path"]).is_symlink():
                raise ValueError("허용한 공개 입력 경로가 아니다")
            if (ROOT / item["path"]).resolve().relative_to(ROOT.resolve()).as_posix() != expected[key]:
                raise ValueError("공개 입력 경로 이탈")
            source_paths.append(item["path"])
    paths = [path.resolve().relative_to(ROOT).as_posix(), *protocol["code_sha256_lf"], *source_paths]
    subprocess.run(["git", "ls-files", "--error-unmatch", *paths], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *paths], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    for relative, expected in protocol["code_sha256_lf"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"사전등록 이후 소스 변경: {relative}")
    for spec in protocol["cases"]:
        for item in spec["sources"].values():
            if sha(ROOT / item["path"]) != item["sha256_lf"]:
                raise ValueError(f"사전등록 입력 변경: {item['path']}")


def input_packet(spec, protocol):
    inputs = spec["sources"]
    case = read(ROOT / inputs["telemetry"]["path"])["summary"]
    ref = read(ROOT / inputs["reference_telemetry"]["path"])["summary"]
    facts = {}
    for prefix, summary in (("case", case), ("reference", ref)):
        for tag, values in sorted(summary.items()):
            if "mean_last_20pct" in values:
                facts[f"telemetry:{prefix}:{tag}"] = values["mean_last_20pct"]
    return {"schema": "active_decision_packet_v1", "phase": "initial", "facts": facts,
            "task": "Unitree Go2 flat RSL-RL PPO; run may be normal, outside supported faults, or failed",
            "mechanisms": list(L.MECHANISMS), "probe_catalog": L.PROBES,
            "thresholds": L.THRESHOLDS, "budget_left": protocol["max_probes"],
            "execution_budget_s": protocol["probe_execution_budget_s"], "observed": []}


def wsl_path(path):
    p = Path(path).resolve().as_posix()
    if not re.fullmatch(r"[A-Za-z]:/.*", p):
        raise ValueError("native Windows 경로가 필요하다")
    return f"/mnt/{p[0].lower()}/{p[3:]}"


def stop_model_worker(output, packet_path):
    pid_path = output.with_suffix(".pid")
    if not pid_path.exists():
        return
    pid = pid_path.read_text().strip()
    if not pid.isdigit():
        raise RuntimeError("모델 worker PID 기록 오류")
    # Kill only this worker after verifying its exact packet argument, avoiding PID reuse.
    script = "import os,sys,signal,time; from pathlib import Path; p=int(sys.argv[1]); f=Path('/proc')/str(p)/'cmdline'; a=f.read_bytes().split(b'\\0') if f.exists() else []; ok=sys.argv[2].encode() in a and any(x.endswith(b'active_model_call.py') for x in a); sys.exit(4) if a and not ok else None; os.kill(p,signal.SIGTERM) if ok else None; time.sleep(0.5); a=f.read_bytes().split(b'\\0') if f.exists() else []; sys.exit(5) if sys.argv[2].encode() in a else None"
    command = shlex.join(["python3", "-c", script, pid, wsl_path(packet_path)])
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command],
                   check=True, capture_output=True, timeout=15)
    pid_path.unlink(missing_ok=True)


def nullable_sum(values):
    return sum(v for v in values if v is not None) if all(v is not None for v in values) else None


def fresh_decision(folder, stage, packet, protocol):
    packet_path = folder / "model" / f"{stage}.packet.json"
    write_new(packet_path, packet)
    records = []
    for attempt in range(1, protocol["max_infra_attempts"] + 1):
        output = folder / "model" / f"{stage}.attempt{attempt}.json"
        argv = [wsl_path(ROOT / "evals/active_model_call.py"), "--packet", wsl_path(packet_path),
                "--output", wsl_path(output), "--model", protocol["model"]]
        command = '. "$HOME/.config/nvidia/env"; exec "$HOME/.venvs/rl-triage/bin/python" ' + shlex.join(argv)
        attempt_start, attempt_wall = now(), time.perf_counter()
        failed = None
        try:
            proc = subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", command], cwd=ROOT,
                                  capture_output=True, timeout=330)
        except subprocess.TimeoutExpired:
            try:
                stop_model_worker(output, packet_path)
                cleanup_failed = False
            except (OSError, RuntimeError, subprocess.SubprocessError):
                cleanup_failed = True
            proc = None
            failed = "ModelWorkerTimeout"
        else:
            cleanup_failed = False
        if not output.exists():
            write_new(output, {"schema": "active_model_call_v1", "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                               "messages_sha256": L.digest(D.decision_messages(packet)), "started_at": attempt_start,
                               "finished_at": now(), "request": {"model": protocol["model"], "temperature": 0,
                                                                         "max_tokens": 16384, "stream": False,
                                                                         "extra_body": {"reasoning_effort": "none"}},
                               "client_cache_used": False, "saved_response_cached": False,
                               "response_id": None, "provider_model": None, "finish_reason": None,
                               "reasoning_content": None, "input_tokens": None, "output_tokens": None,
                               "total_tokens": None, "cpu_s": None, "execution_wall_s": time.perf_counter() - attempt_wall,
                               "wall_scope": "parent_worker_attempt; provider_request_unobserved", "exit_code": 2,
                               "error_type": failed or "ModelWorkerMissingArtifact", "retryable": bool(failed) and not cleanup_failed,
                               "worker_cleanup_failed": cleanup_failed,
                               **({"infra_error": failed} if failed and not cleanup_failed else {})})
        record = read(output)
        if record["packet_sha256"] != hashlib.sha256(packet_path.read_bytes()).hexdigest():
            raise ValueError("모델 입력 해시 불일치")
        records.append(record)
        if record.get("infra_error"):
            if attempt < protocol["max_infra_attempts"]:
                time.sleep(2)
            continue
        if record.get("error_type"):
            raise RuntimeError(f"모델 요청 오류: {record['error_type']}")
        if proc is None or proc.returncode != 0:
            raise RuntimeError("모델 worker 비정상 종료")
        text = record.get("content") or ""
        if text.startswith("```json") and text.rstrip().endswith("```"):
            text = text[7:text.rfind("```")].strip()
        decision = D.validate_decision(json.loads(text), packet)
        write_new(folder / "model" / f"{stage}.decision.json", decision)
        return decision, records
    raise RuntimeError("모델 infrastructure 오류 한도 도달")


def approved_probe(folder, tag, spec, protocol, packet, probe, stage, ledger, binding):
    case = spec["case"]
    snapshot = gpu_snapshot()
    write_new(folder / f"gpu_preflight_{stage}.json", snapshot)
    checkpoint = last_checkpoint(find_run(Path.home() / "IsaacLab/logs/rsl_rl/unitree_go2_flat", case))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != spec["checkpoint_sha256"]:
        raise ValueError("승인 전 checkpoint SHA 불일치")
    loop.isaac_preflight(case, probe, tag)
    reference_report = read(ROOT / spec["sources"]["probe_reference"]["path"])
    reference = reference_report["probes"].get(probe, {}).get("measurement")
    envelope = {"case": case, "checkpoint_sha256": spec["checkpoint_sha256"],
                "protocol_sha256_lf": binding["protocol_sha256_lf"], "stage": stage,
                "reference_source": spec["sources"]["probe_reference"],
                "prior_receipt_digests": [r["receipt_digest"] for r in packet["observed"]],
                "binding": binding, "budget_probes": protocol["max_probes"],
                "probe": probe, "args": L.PROBES[probe]["args"], "thresholds": L.THRESHOLDS}
    request = ledger.propose(envelope, probe, protocol["probe_execution_budget_s"])
    rid = request["request_id"]
    write_new(folder / "approvals" / f"{rid}.json", envelope)
    ledger.approve(rid, ACTOR)
    ledger.consume(rid, envelope, probe)
    execution_start = time.perf_counter()
    failure = None
    measurement, wall_s, exit_code, elapsed = None, None, None, None
    try:
        measurement, wall_s, exit_code = loop.isaac_execute(case, probe, tag, metered=True)
        artifact = loop.file_artifact(case, probe, envelope, tag)
        if artifact is None:
            failure = "native_report_missing"
        execution_path = ROOT / "evals/results" / tag / "execution" / f"{case}__{probe}.json"
        elapsed = read(execution_path)["gpu_wall_s"] if execution_path.exists() else round(wall_s, 1)
    except (Exception, SystemExit) as error:
        failure = type(error).__name__
    if wall_s is None:
        wall_s = time.perf_counter() - execution_start
    if elapsed is None:
        elapsed = round(wall_s, 1)
    outcome = L.classify(probe, measurement, reference) if exit_code == 0 and failure is None else "unknown"
    # Existing execution metadata records launcher wall at 0.1s resolution.
    # Bind the receipt to that same value, rather than a second precision convention.
    receipt = ledger.receipt(rid, outcome, measurement or {}, elapsed, exit_code, reason=failure)
    report_path = ROOT / "evals/results" / tag / "probes" / f"{case}__{probe}.json"
    return {**receipt, "receipt_digest": L.digest(receipt), "probe": probe,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "report_sha256_lf": sha(report_path) if report_path.exists() else None}


def stage_pair(folder, index, before, packet, protocol):
    latest = packet["observed"][-1]
    prefix = f"receipt:{latest['request_id']}:"
    masked = {**packet, "observed": packet["observed"][:-1],
              "facts": {k: v for k, v in packet["facts"].items() if not k.startswith(prefix)}}
    stage = len(packet["observed"])
    order = ("observed", "masked") if (index + stage) % 2 == 0 else ("masked", "observed")
    decisions = {}
    for kind in order:
        name = f"after_{stage}" if kind == "observed" else f"without_receipt_{stage}"
        chosen_packet = packet if kind == "observed" else masked
        decisions[kind], _ = fresh_decision(folder, name, chosen_packet, protocol)
    return decisions["observed"], decisions["masked"], D.contrast(
        before, decisions["observed"], decisions["masked"], packet["observed"])


def gpu_snapshot():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                      "--format=csv,noheader,nounits"], text=True).strip()
    util, used, total = [int(v.strip()) for v in output.split(",")]
    command = "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|kit)' -and $_.CommandLine -match 'Isaac|probes.py|fixed_eval.py|smoke_external.py|rsl_rl[\\/]train.py' } | Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command], text=True).strip()
    pids = json.loads(raw) if raw else []
    pids = pids if isinstance(pids, list) else [pids]
    try:
        with urlopen("http://127.0.0.1:11434/api/ps", timeout=2) as response:
            models = len(json.load(response)["models"])
    except URLError:
        models = 0
    snap = {"at": now(), "gpu_utilization_pct": util, "memory_mib": used,
            "memory_total_mib": total, "other_isaac_pids": pids, "ollama_models": models}
    # WDDM desktop utilization alone is not an Isaac job. A loaded model or unexplained
    # large allocation still blocks the new run; another session is never terminated.
    if pids or models or used > 2500:
        raise RuntimeError(f"GPU 대기: {snap}")
    return snap


def run_case(protocol_path, tag, case, controller="model"):
    protocol = read(protocol_path)
    verify_protocol(protocol_path, protocol)
    spec = next(c for c in protocol["cases"] if c["case"] == case)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag 형식 오류")
    folder = ROOT / "evals/results" / tag / "cases" / case
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "protocol.json", protocol)
    start, clock = now(), time.perf_counter()
    decisions, observed, contrasts, shadows = [], [], [], []
    prereg = {"case": case, "checkpoint_sha256": spec["checkpoint_sha256"],
              "protocol_sha256_lf": sha(protocol_path), "hypotheses": list(L.MECHANISMS),
              "budget_probes": protocol["max_probes"], "thresholds": L.THRESHOLDS}
    write_new(folder / "prereg.json", prereg)
    ledger = L.Ledger(folder / "ledger.jsonl")
    packet = input_packet(spec, protocol)
    write_new(folder / "initial_packet.json", packet)
    result = {"schema": "active_diagnosis_case_v1", "case": case, "tag": tag,
              "scope": protocol["scope"], "started_at": start, "protocol_sha256_lf": sha(protocol_path),
              "checkpoint_sha256": spec["checkpoint_sha256"], "human_review_s": None,
              "approval_actor": ACTOR, "approval_identity_attested": False,
              "controller": controller, "gpu_device_busy_s": None}
    try:
        if controller == "model":
            decision, _ = fresh_decision(folder, "initial", packet, protocol)
            decisions.append(decision)
        else:
            decision = None
        while packet["budget_left"] > 0:
            state = D.evidence_state(observed, packet["budget_left"])
            if controller == "model":
                if decision is None:
                    raise ValueError("모델 결정이 없다")
                if decision["action"] != "continue":
                    result["stop_reason"] = "model_" + decision["action"]
                    break
                probe = decision["probe"]
                name = "initial" if not observed else f"after_{len(observed)}"
                call_paths = sorted((folder / "model").glob(f"{name}.attempt*.json"))
                binding = {"controller": controller, "packet_sha256_lf": sha(folder / "model" / f"{name}.packet.json"),
                           "model_call_sha256_lf": sha(call_paths[-1]),
                           "decision_sha256_lf": sha(folder / "model" / f"{name}.decision.json")}
            elif controller == "fixed_order":
                probe = next((p for p in L.FIXED_ORDER if p not in [r["probe"] for r in observed]), None)
                binding = {"controller": controller, "initial_packet_sha256_lf": sha(folder / "initial_packet.json")}
            elif controller == "cheapest_discrimination":
                probe = L.next_probe(state["hypotheses"], {r["probe"]: r["outcome"] for r in observed})
                binding = {"controller": controller, "initial_packet_sha256_lf": sha(folder / "initial_packet.json")}
            else:
                raise ValueError("모르는 controller")
            if probe is None:
                break
            binding["protocol_sha256_lf"] = sha(protocol_path)
            receipt = approved_probe(folder, tag, spec, protocol, packet, probe,
                                     str(len(observed) + 1), ledger, binding)
            observed.append(receipt)
            packet = {**packet, "phase": "after_receipt", "observed": observed,
                      "facts": {**packet["facts"], **D.receipt_facts(receipt)},
                      "budget_left": protocol["max_probes"] - len(observed),
                      "previous_decision": decision}
            if receipt["outcome"] == "unknown":
                result["stop_reason"] = "unknown_receipt_metrics_not_used_for_decision"
                break
            if controller == "model":
                index = next(i for i, s in enumerate(protocol["cases"]) if s["case"] == case)
                decision, shadow, contrast = stage_pair(folder, index, decision, packet, protocol)
                decisions.append(decision)
                contrasts.append(contrast)
                shadows.append(shadow)
            if L.status(D.evidence_state(observed)["hypotheses"],
                        {r["probe"]: r["outcome"] for r in observed}, packet["budget_left"]) == "confirmed":
                break
        if controller == "model" and decision and decision["action"] == "hold":
            result["stop_reason"] = "model_hold"
        elif not result.get("stop_reason"):
            result["stop_reason"] = "model_confirm" if controller == "model" and decision and decision["action"] == "confirm" else "probe_budget_or_frozen_stop"
        result["status"] = "hold" if result["stop_reason"] in ("model_hold", "unknown_receipt_metrics_not_used_for_decision") else "completed"
    except (ValueError, RuntimeError, subprocess.SubprocessError, L.LedgerError) as error:
        result.update(status="hold", error_type=type(error).__name__, error=str(error),
                      orphan_requests=ledger.orphans())
    result.update(finished_at=now(), end_to_end_wall_s=time.perf_counter() - clock,
                  decisions=decisions, observed=observed, contrasts=contrasts,
                  without_receipt=shadows,
                  budget_left=protocol["max_probes"] - len(observed),
                  model_action=decisions[-1]["action"] if decisions else None,
                  evidence_state=D.evidence_state(observed, protocol["max_probes"] - len(observed)))
    # Read every attempt from disk, including rejected responses and infra retries.
    attempts = [read(p) for p in sorted((folder / "model").glob("*.attempt*.json"))]
    result["costs"] = {"model_calls": len(attempts),
                       "input_tokens": sum(c["input_tokens"] for c in attempts) if all(c["input_tokens"] is not None for c in attempts) else None,
                       "output_tokens": sum(c["output_tokens"] for c in attempts) if all(c["output_tokens"] is not None for c in attempts) else None,
                       "model_wall_s": sum(c["execution_wall_s"] for c in attempts),
                       "model_cpu_s": nullable_sum([c["cpu_s"] for c in attempts]),
                       "isaac_wall_s": sum(r["gpu_s"] for r in observed), "human_review_s": None,
                       "gpu_device_busy_s": None}
    for field in ("cpu_s", "simulator_steps"):
        source_folder = "execution" if field == "cpu_s" else "resources"
        paths = [ROOT / "evals/results" / tag / source_folder / f"{case}__{r['probe']}.json" for r in observed]
        values = [read(p).get(field) if p.exists() else None for p in paths]
        result["costs"][f"isaac_{field}"] = sum(v for v in values if v is not None) if values and all(v is not None for v in values) else (0 if not observed else None)
    if ledger.orphans():
        result["costs"].update(isaac_wall_s=None, isaac_cpu_s=None, isaac_simulator_steps=None)
    write_new(folder / "result.json", result)
    print(json.dumps({"case": case, "status": result["status"], "probes": len(observed),
                      "contrasts": result.get("contrasts"), "costs": result["costs"],
                      "result": str(folder / "result.json")}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--protocol", type=Path, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--case", required=True)
    r.add_argument("--controller", choices=["model", "fixed_order", "cheapest_discrimination"], default="model")
    batch = sub.add_parser("run-all")
    batch.add_argument("--protocol", type=Path, required=True)
    batch.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = prepare(args.out)
        print(json.dumps({"path": str(args.out), "cases": len(protocol["cases"])}))
    elif args.command == "run-all":
        protocol = read(args.protocol)
        for index, spec in enumerate(protocol["cases"]):
            controllers = [("model", ""), ("fixed_order", "_fixed"), ("cheapest_discrimination", "_cheap")]
            # All policies have the same frozen budget. Alternating execution order
            # reduces machine/load order effects without claiming randomized inference.
            offset = index % len(controllers)
            controllers = controllers[offset:] + controllers[:offset]
            for controller, suffix in controllers:
                run_case(args.protocol, args.tag + suffix, spec["case"], controller)
    else:
        run_case(args.protocol, args.tag, args.case, args.controller)


if __name__ == "__main__":
    main()
