"""P1 사전등록·승인 원장·고정 행동 평가·비용을 연결하는 offline 감사 CLI.

register는 probe 소비 전에 계약을 저장한다. finalize는 기존 결과만 읽는다.
GPU 실행, 모델 호출, 승인 부여, 재시도 또는 기존 결과 변경을 수행하지 않는다.
사용 계약: docs/P1-EVIDENCE-AUDIT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))
from fixed_eval import command_grid  # noqa: E402
from rl_triage import behavior_oracle as O  # noqa: E402
from rl_triage import experiment_cost as C  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402


def _finite(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("JSON에 NaN 또는 무한대가 있다")
    if isinstance(obj, dict):
        for value in obj.values():
            _finite(value)
    elif isinstance(obj, list):
        for value in obj:
            _finite(value)
    return obj


def _json(data: bytes):
    return _finite(json.loads(data.decode("utf-8-sig")))


def _read(path: Path):
    data = path.read_bytes()
    return _json(data), hashlib.sha256(data).hexdigest()


def _write_new(path: Path, obj: dict):
    content = json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def _code_sources():
    names = ("evals/audit_experiment.py", "evals/fixed_eval.py", "src/rl_triage/behavior_oracle.py",
             "src/rl_triage/experiment_cost.py", "src/rl_triage/recovery.py", "src/rl_triage/probe_loop.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}


def _cost_events(probe_events: list[dict], external: list[dict]) -> list[dict]:
    """동일 시도의 미기록 지표만 보충한다. receipt 실측값을 바꾸거나 시도를 두 번 세지 않는다."""
    C.summarize_costs(external, {})
    merged = {ev["attempt_id"]: {**ev, "costs": dict(ev["costs"])} for ev in probe_events}
    for ev in external:
        rid = ev["attempt_id"]
        if rid not in merged:
            merged[rid] = ev
            continue
        original = merged[rid]
        if original["status"] != ev["status"]:
            raise ValueError("비용 보충 자료가 원장의 실행 상태와 다르다")
        for key in ("stage", "started_at", "finished_at"):
            if key in ev and ev[key] != original.get(key):
                raise ValueError("비용 보충 자료가 원장의 단계·실행 시각을 바꾸려 한다")
        for key, value in ev["costs"].items():
            known = original["costs"].get(key)
            if known is not None and value != known:
                raise ValueError("비용 보충 자료가 원장의 실측값을 바꾸려 한다")
            if value is not None:
                original["costs"][key] = value
    return list(merged.values())


def _registry():
    return {"probes": L.PROBES, "thresholds": {p: list(v) for p, v in L.THRESHOLDS.items()}}


def _timestamp(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("시각은 timezone을 포함한 문자열이어야 한다")
    at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if at.utcoffset() is None:
        raise ValueError("시각에는 timezone이 필요하다")
    return at


def _wall_time(path, contract, events, costs, after, after_sha, abstained):
    limit = contract["end_to_end_wall_limit_s"]
    if path is None:
        return {"elapsed_s": None, "limit_s": limit, "budget_status": "unknown", "timeline_sha256": None,
                "coverage_complete": False, "missing": ["timeline"]}
    timeline, sha = _read(path)
    start, finish = _timestamp(timeline["started_at"]), _timestamp(timeline["finished_at"])
    registered = _timestamp(contract["registered_at"])
    if not start <= registered <= finish or finish > datetime.now(timezone.utc):
        raise ValueError("전체 작업 시간에 계약 등록이 포함되지 않거나 종료 시각이 미래다")
    if any(not start <= _timestamp(ev["at"]) <= finish for ev in events):
        raise ValueError("전체 작업 시간 밖의 원장 이벤트가 있다")
    missing, timed = [], {}
    for attempt in costs:
        rid, stage = attempt["attempt_id"], attempt.get("stage")
        if not isinstance(stage, str) or not stage:
            missing.append(f"{rid}:stage")
        if attempt.get("started_at") is None or attempt.get("finished_at") is None:
            missing.append(f"{rid}:timestamps")
            continue
        a, b = _timestamp(attempt["started_at"]), _timestamp(attempt["finished_at"])
        if not start <= a <= b <= finish:
            raise ValueError("전체 작업 시간 밖의 비용 시도 또는 역순 시각이 있다")
        # 기존 원장의 초 단위 시각은 양 끝에서 반올림 오차가 있을 수 있다.
        for key in ("gpu_wall_s", "execution_wall_s"):
            value = attempt["costs"].get(key)
            if value is not None and value > (b - a).total_seconds() + 2:
                raise ValueError("시도 비용의 실행 시간이 시작·종료 구간보다 길다")
        timed.setdefault(stage, []).append((attempt, a, b))
    if "diagnosis" not in timed:
        missing.append("diagnosis_receipt")
    if not abstained:
        if after is None:
            missing.append("fixed_eval_result")
        else:
            matching_eval = [(ev, a, b) for ev, a, b in timed.get("fixed_eval", [])
                             if ev.get("artifact_sha256") == after_sha and ev["status"] == "completed"]
            matching_repair = [(ev, a, b) for ev, a, b in timed.get("repair", [])
                               if ev.get("artifact_sha256") == after["checkpoint"]["sha256"]
                               and ev["status"] == "completed"]
            if not matching_eval:
                missing.append("fixed_eval_receipt_matching_artifact")
            if not matching_repair:
                missing.append("repair_receipt_matching_checkpoint")
            if after.get("started_at") is None or after.get("finished_at") is None:
                missing.append("fixed_eval_timestamps")
            else:
                a, b = _timestamp(after["started_at"]), _timestamp(after["finished_at"])
                if not start <= a <= b <= finish:
                    raise ValueError("최종 평가의 실행 시각이 전체 작업 구간 밖이다")
                if matching_eval and not any(x <= a + timedelta(seconds=1) and b <= y + timedelta(seconds=1)
                                             for _, x, y in matching_eval):
                    raise ValueError("고정 평가 시각이 산출물 receipt의 실행 구간과 다르다")
                if matching_repair and not any(y <= a + timedelta(seconds=1) for _, _, y in matching_repair):
                    raise ValueError("수리 receipt가 고정 평가보다 늦다")
    elapsed = (finish - start).total_seconds()
    status = "exceeded" if limit is not None and elapsed > limit else "unknown" if limit is None or missing else "within"
    return {"elapsed_s": elapsed, "limit_s": limit, "budget_status": status, "timeline_sha256": sha,
            "coverage_complete": not missing, "missing": missing}


def _loop(folder: Path):
    prereg, prereg_sha = _read(folder / "prereg.json")
    reference, reference_sha = _read(folder / "reference.json")
    state, state_sha = _read(folder / "state.json")
    ledger = (folder / "ledger.jsonl").read_bytes()
    events = [_json(line) for line in ledger.splitlines() if line.strip()]
    if any(not isinstance(obj, dict) for obj in (prereg, reference, state, *events)):
        raise ValueError("고리 사전등록·기준·상태·원장 행은 객체여야 한다")
    hyps = prereg["hypotheses"]
    if not isinstance(hyps, list) or not hyps or any(h not in L.MECHANISMS for h in hyps) or len(set(hyps)) != len(hyps):
        raise ValueError("고리의 초기 가설 목록이 유효하지 않다")
    if type(prereg["budget_probes"]) is not int or prereg["budget_probes"] <= 0:
        raise ValueError("고리의 probe 예산은 양의 정수여야 한다")
    if re.fullmatch(r"[0-9a-f]{64}", prereg["checkpoint_sha256"]) is None:
        raise ValueError("고리의 체크포인트 SHA256이 유효하지 않다")
    if prereg["thresholds"] != _registry()["thresholds"]:
        raise ValueError("현재 판정 임계값과 고리의 사전등록 임계값이 다르다")
    expected = {p: {h: L.PROBES[p]["expect"].get(h) for h in hyps} for p in L.PROBES}
    if prereg["expectations"] != expected:
        raise ValueError("사전등록한 가설별 예상과 probe 레지스트리가 다르다")
    sources = {"prereg_sha256": prereg_sha, "probe_reference_sha256": reference_sha,
               "state_sha256": state_sha, "ledger_sha256": hashlib.sha256(ledger).hexdigest()}
    return prereg, reference, state, events, sources


def audit_loop(prereg: dict, reference: dict, state: dict, events: list[dict]) -> dict:
    """승인 순서와 digest를 검사하고 receipt에서 관측·가설·실측 비용을 다시 만든다."""
    requests, observed, costs, skipped, history = {}, {}, [], [], []
    hypotheses = list(prereg["hypotheses"])
    first_at = last_at = None
    overruns = []
    for ev in events:
        at = _timestamp(ev["at"])
        if last_at is not None and at < last_at:
            raise ValueError("원장의 시각은 timezone을 포함하고 시간순이어야 한다")
        first_at = first_at or at
        last_at = at
        rid, kind = ev["request_id"], ev["event"]
        if not isinstance(rid, str) or not rid:
            raise ValueError("원장의 request_id가 비었다")
        if kind == "proposed":
            probe = ev["probe"]
            if rid in requests or probe not in L.PROBES:
                raise ValueError("중복 요청 또는 모르는 probe")
            if any(req["state"] in ("pending", "approved", "running") for req in requests.values()):
                raise ValueError("동시에 열린 probe 요청은 하나만 허용한다")
            consumed = sum(req["state"] == "done" for req in requests.values())
            if L.status(hypotheses, observed, prereg["budget_probes"] - consumed) != "open":
                raise ValueError("종료한 가설 고리 또는 소진된 예산에 새 probe를 제안했다")
            expected_probe = L.next_probe(hypotheses, {**{p: "unknown" for p in skipped}, **observed})
            if probe != expected_probe:
                raise ValueError("현재 가설을 가르는 사전등록 선택 규칙과 다른 probe다")
            if ev["prereg_digest"] != L.digest(prereg) or ev["args_digest"] != L.digest(L.PROBES[probe]["args"]):
                raise ValueError("승인 요청의 사전등록 또는 canonical args digest가 다르다")
            C.summarize_costs([], {"gpu_wall_s": ev["budget_gpu_s"]})
            requests[rid] = {**ev, "state": "pending"}
            continue
        req = requests.get(rid)
        required = {"approved": "pending", "rejected": "pending", "consumed": "approved", "receipt": "running"}
        if kind not in required or req is None or req["state"] != required[kind]:
            raise ValueError(f"{rid}: 승인·소비·결과 순서 위반({kind})")
        if kind in ("approved", "rejected") and (not isinstance(ev.get("approver"), str) or not ev["approver"].strip()):
            raise ValueError("승인·거절 기록의 담당자가 비었다")
        if kind == "approved":
            req["state"] = "approved"
        elif kind == "rejected":
            if not isinstance(ev.get("reason"), str) or not ev["reason"].strip():
                raise ValueError("거절 이유가 비었다")
            req["state"] = "rejected"
            skipped.append(req["probe"])
            history.append({"request_id": rid, "probe": req["probe"], "rejected": ev["reason"]})
            costs.append({"attempt_id": f"{prereg['case']}:{rid}", "status": "rejected",
                          "stage": "probe", "started_at": req["at"], "finished_at": ev["at"],
                          "costs": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                    "gpu_wall_s": 0, "execution_wall_s": 0}})
        elif kind == "consumed":
            req["state"] = "running"
            req["consumed_at"] = ev["at"]
        else:
            probe = req["probe"]
            code = ev["exit_code"]
            if code is not None and type(code) is not int:
                raise ValueError("종료 코드는 정수 또는 null이어야 한다")
            if not isinstance(ev["measurement"], dict):
                raise ValueError("probe 측정값은 객체여야 한다")
            gpu_wall = ev.get("gpu_s")
            over = gpu_wall is not None and gpu_wall > req["budget_gpu_s"]
            measured = L.classify(probe, ev["measurement"], reference.get(probe)) if code == 0 and not over else "unknown"
            if measured != ev["outcome"]:
                raise ValueError("receipt의 관측 판정이 사전등록한 측정·임계값과 다르다")
            observed[probe] = measured
            hypotheses, dropped = L.update(hypotheses, probe, measured)
            history.append({"request_id": rid, "probe": probe, "outcome": measured, "dropped": dropped})
            cost = {"attempt_id": f"{prereg['case']}:{rid}",
                    "status": "completed" if code == 0 else "unknown" if code is None else "failed",
                    "stage": "probe", "started_at": req["consumed_at"], "finished_at": ev["at"],
                    "costs": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
                              "gpu_wall_s": gpu_wall, "execution_wall_s": gpu_wall}}
            C.summarize_costs([cost], {})
            if over:
                overruns.append(rid)
            costs.append(cost)
            req["state"] = "done"
    incomplete = [rid for rid, req in requests.items() if req["state"] == "running"]
    for rid in incomplete:
        costs.append({"attempt_id": f"{prereg['case']}:{rid}", "status": "unknown",
                      "stage": "probe", "started_at": requests[rid]["consumed_at"], "finished_at": None,
                      "costs": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}})
    if state["observed"] != observed or state["hypotheses"] != hypotheses or state["budget"] != prereg["budget_probes"]:
        raise ValueError("state.json과 원장에서 복원한 관측·가설·예산이 다르다")
    if state.get("skipped", []) != skipped:
        raise ValueError("state.json과 원장의 거절 이력이 다르다")
    if state.get("history") != history:
        raise ValueError("state.json과 원장의 갱신 이력이 다르다")
    count = sum(req["state"] in ("running", "done") for req in requests.values())
    active = [rid for rid, req in requests.items() if req["state"] in ("pending", "approved", "running")]
    if state.get("pending") != (active[0] if active else None):
        raise ValueError("state.json의 대기 요청과 원장의 활성 요청이 다르다")
    status = L.status(hypotheses, observed, prereg["budget_probes"] - len(observed))
    if status == "open" and L.next_probe(hypotheses, {**{p: "unknown" for p in skipped}, **observed}) is None:
        status = "unidentifiable"
    if state.get("status") != status:
        raise ValueError("state.json의 종료 상태와 원장의 복원 상태가 다르다")
    return {"case": prereg["case"], "diagnosis_status": status, "hypotheses": hypotheses,
            "observed": observed, "incomplete_requests": incomplete,
            "pending_requests": [rid for rid, req in requests.items() if req["state"] in ("pending", "approved")],
            "probe_attempts": count, "probe_budget_exceeded": count > prereg["budget_probes"],
            "request_budget_overruns": overruns, "cost_events": costs,
            "observed_loop_span_s": (last_at - first_at).total_seconds() if first_at else None,
            "approval_identity_attested": False}


def register(args) -> dict:
    prereg, ref_probe, state, events, sources = _loop(args.loop_dir)
    trace = audit_loop(prereg, ref_probe, state, events)
    if trace["probe_attempts"]:
        raise ValueError("probe를 소비하기 전에 감사 계약을 등록해야 한다")
    reference, reference_sha = _read(args.reference)
    normals = [_read(p)[0] for p in args.normals]
    gates, gates_sha = _read(args.gates)
    limits, limits_sha = _read(args.limits)
    C.summarize_costs([], limits)
    if not limits:
        raise ValueError("비용 한도를 하나 이상 사전등록해야 한다")
    info = args.information.read_bytes()
    if not info:
        raise ValueError("공통 입력 스냅샷이 비었다")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.method) is None:
        raise ValueError("method에는 영문·숫자·_-만 쓴다")
    if args.wall_budget is not None:
        C.summarize_costs([], {"execution_wall_s": args.wall_budget})
    oracle = O.register_contract(reference, normals, gates)
    protocol_bytes = args.protocol.read_bytes()
    protocol = _json(protocol_bytes)
    protocol_sha = hashlib.sha256(protocol_bytes.replace(b"\r\n", b"\n")).hexdigest()
    evaluation = oracle["evaluation"]
    grid = sorted(command_grid(protocol["command_grid"]))
    expected = {"protocol": protocol["name"], "protocol_sha256_lf": protocol_sha,
                "task": protocol["task"], "eval_seed": protocol["eval_seed"],
                "num_envs": len(grid) * protocol["envs_per_condition"], "horizon_steps": protocol["horizon_steps"],
                "step_dt": protocol["step_dt"], "commands": [list(c) for c in grid]}
    if any(evaluation[key] != value for key, value in expected.items()):
        raise ValueError("기준 평가 자료가 지정한 프로토콜 파일의 해시·조건과 다르다")
    if protocol["horizon_steps"] * protocol["step_dt"] != protocol["episode_length_s"]:
        raise ValueError("프로토콜의 스텝 수·제어 주기·평가 시간이 다르다")
    comparison = {"oracle_sha256": oracle["contract_sha256"], "limits": limits,
                  "end_to_end_wall_limit_s": args.wall_budget,
                  "cost_keys": list(C.COST_KEYS), "probe_registry": _registry()}
    payload = {"schema": "p1_experiment_audit_v1", "registered_at": datetime.now(timezone.utc).isoformat(),
               "method": args.method, "information_sha256": hashlib.sha256(info).hexdigest(),
               "end_to_end_wall_limit_s": args.wall_budget,
               "information_bytes": len(info), "comparison_contract_sha256": L.digest(comparison),
               "oracle": oracle, "limits": limits, "case": prereg["case"],
               "loop_prereg_digest": L.digest(prereg), "probe_reference_digest": L.digest(ref_probe),
               "probe_registry_digest": L.digest(_registry()), "reference_file_sha256": reference_sha,
               "protocol_file_sha256_lf": protocol_sha,
               "gates_file_sha256": gates_sha, "limits_file_sha256": limits_sha,
               "registration_sources": sources, "code_sha256_lf": _code_sources()}
    return {**payload, "audit_contract_sha256": L.digest(payload)}


def finalize(args) -> dict:
    contract, contract_file_sha = _read(args.contract)
    payload = {k: v for k, v in contract.items() if k != "audit_contract_sha256"}
    if contract["schema"] != "p1_experiment_audit_v1" or L.digest(payload) != contract["audit_contract_sha256"]:
        raise ValueError("감사 계약의 버전 또는 해시가 다르다")
    if contract["code_sha256_lf"] != _code_sources():
        raise ValueError("계약 등록 뒤 감사·판정 코드가 달라졌다")
    prereg, probe_ref, state, events, sources = _loop(args.loop_dir)
    if (contract["loop_prereg_digest"] != L.digest(prereg)
            or contract["probe_reference_digest"] != L.digest(probe_ref)
            or contract["probe_registry_digest"] != L.digest(_registry())):
        raise ValueError("등록 뒤 사전등록·기준 probe·레지스트리가 달라졌다")
    trace = audit_loop(prereg, probe_ref, state, events)
    registered_second = _timestamp(contract["registered_at"]).replace(microsecond=0)
    if any(_timestamp(ev["at"]) < registered_second for ev in events if ev["event"] == "consumed"):
        raise ValueError("계약 등록 이전에 소비된 probe가 있다")
    reference, ref_sha = _read(args.reference)
    before, before_sha = _read(args.before)
    if before["checkpoint"]["sha256"] != prereg["checkpoint_sha256"] or before["name"] != prereg["case"]:
        raise ValueError("고리의 케이스·체크포인트와 개입 전 고정 평가가 다르다")
    if ref_sha != contract["reference_file_sha256"]:
        raise ValueError("사전등록한 기준 평가 파일과 다르다")
    after, after_sha = _read(args.after) if args.after else (None, None)
    behavior = O.recovery_outcome(before, after, reference, contract["oracle"], intervened=not args.abstain)
    external, external_sha = _read(args.events) if args.events else ([], None)
    if not isinstance(external, list):
        raise ValueError("추가 비용 자료는 시도 목록이어야 한다")
    cost_events = _cost_events(trace.pop("cost_events"), external)
    costs = C.summarize_costs(cost_events, contract["limits"])
    result = {"schema": "p1_experiment_audit_result_v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "method": contract["method"], "information_sha256": contract["information_sha256"],
            "contract_sha256": contract["comparison_contract_sha256"], "limits": contract["limits"],
            "loop": trace, "behavior": behavior, "cost_events": cost_events, "costs": costs,
            "end_to_end_wall": _wall_time(args.timeline, contract, events, cost_events, after, after_sha, args.abstain),
            "cost_scope": "probe_receipts_and_supplied_attempts", "repair_execution_verified": False,
            "probe_execution_checkpoint_verified": False, "code_sha256_lf": contract["code_sha256_lf"],
            "sources": {**sources, "audit_contract_file_sha256": contract_file_sha,
                        "reference_sha256": ref_sha, "before_sha256": before_sha,
                        "after_sha256": after_sha, "external_cost_sha256": external_sha},
            "environment": {"model_calls": 0, "network": "not used", "gpu": "not used"}}
    return {**result, "result_sha256": L.digest(result)}


def compare(args) -> dict:
    reports = [_read(path) for path in args.reports]
    runs, denominators = [], []
    for report, sha in reports:
        if report["schema"] != "p1_experiment_audit_result_v1":
            raise ValueError("비교 입력은 finalize의 감사 결과여야 한다")
        if L.digest({k: v for k, v in report.items() if k != "result_sha256"}) != report["result_sha256"]:
            raise ValueError("감사 결과의 내용 해시가 다르다")
        if report["costs"] != C.summarize_costs(report["cost_events"], report["limits"]):
            raise ValueError("감사 결과의 비용 집계와 시도 기록이 다르다")
        runs.append({"method": report["method"], "information_sha256": report["information_sha256"],
                     "contract_sha256": report["contract_sha256"], "limits": report["limits"],
                     "events": report["cost_events"]})
        b = report["behavior"]
        faulty, healthy = b["before"]["label"] == "unhealthy", b["before"]["label"] == "healthy"
        denominators.append({"method": report["method"], "F_failed": int(faulty), "H_healthy": int(healthy),
                             "N_all": 1, "behavior_recovered_F": int(faulty and b["outcome"] == "recovered"),
                             "validated_recovery_F": None,
                             "recovery_validation_status": "repair_execution_not_verified",
                             "unnecessary_H": None, "unnecessary_N": None,
                             "behavior_unnecessary_H": int(healthy and b["outcome"] == "unnecessary_intervention"),
                             "behavior_unnecessary_N": int(healthy and b["outcome"] == "unnecessary_intervention"),
                             "behavior_regression_H": int(healthy and b["outcome"] == "regression"),
                             "abstained_N": int(b["outcome"] == "abstained"),
                             "undetermined_N": int(b["outcome"] == "undetermined"), "source_sha256": sha})
    result = C.compare_runs(runs)
    evidence = []
    if reports and any(r["sources"]["before_sha256"] != reports[0][0]["sources"]["before_sha256"]
                       or r["loop"]["case"] != reports[0][0]["loop"]["case"] for r, _ in reports[1:]):
        evidence.append("before_evaluation_mismatch")
    for report, _ in reports:
        loop = report["loop"]
        if (loop["incomplete_requests"] or loop["pending_requests"] or loop["diagnosis_status"] == "open"
                or loop["probe_budget_exceeded"] or loop["request_budget_overruns"]):
            evidence.append(f"{report['method']}: incomplete_or_over_budget_probe_trace")
        if any(value is None for value in report["costs"]["totals"].values()):
            evidence.append(f"{report['method']}: incomplete_cost_vector")
        if report["end_to_end_wall"]["budget_status"] != "within":
            evidence.append(f"{report['method']}: end_to_end_wall_{report['end_to_end_wall']['budget_status']}")
        if (not report["behavior"]["before"].get("metrics_verified")
                or (report["behavior"]["intervened"] and not report["behavior"]["after"].get("metrics_verified"))):
            evidence.append(f"{report['method']}: behavior_statistics_unverified")
    if evidence:
        result["comparable"] = False
        result["reasons"].extend(evidence)
    return {"schema": "p1_experiment_comparison_v1", **result, "denominators": denominators,
            "comparison_scope": "declared_information_and_recorded_costs",
            "input_delivery_verified": False, "causal_confirmation": False}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--loop-dir", type=Path, required=True)
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--protocol", type=Path, default=ROOT / "bench/protocols/fixed_eval_v1.json")
    r.add_argument("--normals", type=Path, nargs="*", default=[])
    r.add_argument("--gates", type=Path, required=True)
    r.add_argument("--limits", type=Path, required=True)
    r.add_argument("--information", type=Path, required=True)
    r.add_argument("--method", required=True)
    r.add_argument("--wall-budget", type=float, help="진단 시작부터 최종 평가까지 전체 벽시계 시간 한도(초)")
    f = sub.add_parser("finalize")
    f.add_argument("--loop-dir", type=Path, required=True)
    f.add_argument("--contract", type=Path, required=True)
    f.add_argument("--reference", type=Path, required=True)
    f.add_argument("--before", type=Path, required=True)
    action = f.add_mutually_exclusive_group()
    action.add_argument("--after", type=Path)
    action.add_argument("--abstain", action="store_true")
    f.add_argument("--events", type=Path, help="진단·평가·실패·사람 검토를 포함한 추가 비용 JSON 목록")
    f.add_argument("--timeline", type=Path, help="전체 과정의 started_at·finished_at JSON(timezone 포함)")
    c = sub.add_parser("compare")
    c.add_argument("--reports", type=Path, nargs="+", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--events", type=Path, required=True)
    b.add_argument("--limits", type=Path, required=True)
    b.add_argument("--next-cost", type=Path, required=True)
    for parser in (r, f, c, b):
        parser.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError("출력 파일이 이미 있다. 결과는 덮어쓰지 않는다")
        if args.cmd == "register":
            result = register(args)
        elif args.cmd == "finalize":
            result = finalize(args)
        elif args.cmd == "compare":
            result = compare(args)
        else:
            result = C.can_afford(_read(args.events)[0], _read(args.limits)[0], _read(args.next_cost)[0])
        _write_new(args.out, result)
        print(json.dumps({"command": args.cmd, "saved": True}, ensure_ascii=False))
        if args.cmd == "budget" and not result["allowed"]:
            return 3
        if args.cmd == "compare" and not result["comparable"]:
            return 3
    except (O.OracleError, C.CostError, ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print(f"실험 감사 입력 오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
