"""E2 진단 확정 뒤 교정 제안·승인·실행 CLI(사양 SPEC-correction-step 구성 3). 원장은 evals/loops/<케이스>/correction_ledger.jsonl.

  python evals/correct.py propose <케이스> --source-report <fixed_eval_v1 report JSON> [--budget-gpu-s 7200]
  python evals/correct.py approve <케이스> <request_id> --approver <이름>
  python evals/correct.py reject  <케이스> <request_id> --approver <이름> --reason <이유>
  python evals/correct.py run     <케이스> <request_id> --tag <결과 tag> [--isaaclab <경로>]   # GPU: 메인 오케스트레이터만
  python evals/correct.py status  <케이스>

- propose는 evals/loops/<케이스> 상태가 confirmed일 때만 한다. 등록 밖 메커니즘·none_supported 등은 HOLD를 기록하고 요청을 만들지 않는다.
- run은 승인을 한 번 소비하고 주입된 실행 함수를 부른 뒤 receipt를 남긴다. 모델·에이전트는 이 CLI를 부르지 않는다.
- 실행 함수 계약: execute(request: dict) -> {"exit_code": int, "gpu_s": float, "verdict": "recovered"|"not_recovered"|"unknown",
  "result_sha": str|None, "tag": str}. request에는 case, request_id, correction_id, recovery_mode, protocol_case_arm, budget_gpu_s, tag가 있다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage import correction_registry as R  # noqa: E402
from rl_triage import probe_loop as L  # noqa: E402
from rl_triage.correction_ledger import CorrectionLedger, LedgerError  # noqa: E402

LOOPS = ROOT / "evals" / "loops"
DEFAULT_BUDGET_GPU_S = 7200


def _dir(case: str, loops: Path) -> Path:
    if not case.replace("_", "").isalnum():
        raise SystemExit(f"케이스 이름이 이상하다: {case!r}")
    return loops / case


def _ledger(case: str, loops: Path) -> CorrectionLedger:
    return CorrectionLedger(_dir(case, loops) / "correction_ledger.jsonl")


def _loop_state(case: str, loops: Path) -> tuple[dict, dict, L.Ledger]:
    d = _dir(case, loops)
    try:
        prereg = json.loads((d / "prereg.json").read_text(encoding="utf-8"))
        state = json.loads((d / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise LedgerError(f"{case}: loop 상태 파일을 읽을 수 없다: {e}") from e
    return prereg, state, L.Ledger(d / "ledger.jsonl")


def loop_state_digest(prereg: dict, state: dict) -> str:
    return L.digest({"prereg": prereg, "state": state})


def propose(case: str, source_report: dict, budget_gpu_s: int = DEFAULT_BUDGET_GPU_S, loops: Path = LOOPS,
            source_report_sha: str | None = None) -> dict:
    """확정된 loop에서 교정 하나·회복 방식 하나를 제안한다. 지원 밖이면 {"decision": "HOLD"}만 반환(요청 0)."""
    prereg, state, loop_led = _loop_state(case, loops)
    led = _ledger(case, loops)
    lsd = loop_state_digest(prereg, state)
    status = state.get("status")
    if status in ("none_supported", "unidentifiable"):
        led.hold(lsd, "", f"loop status {status}")
        return {"decision": R.HOLD, "reason": f"loop status {status}"}
    if status != "confirmed" or len(state.get("hypotheses", [])) != 1:
        raise LedgerError(f"{case}: loop 상태가 confirmed가 아니다({status!r})")
    mech = state["hypotheses"][0]
    if mech not in R.REGISTRY:
        led.hold(lsd, mech, "unregistered_mechanism")
        return {"decision": R.HOLD, "reason": "unregistered_mechanism", "mechanism": mech}
    probe = L.confirming_probe(mech)
    reqs = loop_led.requests()
    confirm = [r for r in reqs.values() if r["probe"] == probe and r["state"] == "done"
               and r["receipt"]["outcome"] == "abnormal"]
    if not confirm:
        raise LedgerError(f"{case}: 확정 probe({probe}) abnormal receipt가 원장에 없다")
    rule = R.recommend_recovery_mode(mech, source_report)
    receipts = [e for e in loop_led.events() if e["event"] == "receipt"]
    evidence = {"confirming_probe": probe, "confirming_request_id": confirm[-1]["request_id"],
                "standstill_ratio": rule["standstill_ratio"], "rule": rule["rule"], "rule_mode": rule["mode"],
                "source_report_sha256": source_report_sha}
    ev = led.propose(lsd, mech, L.digest(receipts), rule["correction_id"], rule["mode"],
                     R.protocol_case_arm(mech), budget_gpu_s, evidence)
    return {"decision": "proposed", **ev}


def run(case: str, rid: str, execute, loops: Path = LOOPS, tag: str | None = None) -> dict:
    led = _ledger(case, loops)
    if led.orphans():
        raise LedgerError(f"결과 없이 끝난 요청 {led.orphans()}이 있다. 먼저 recover로 닫는다")
    req = led.requests().get(rid)
    if req is None:
        raise LedgerError(f"{rid}: 없는 요청")
    prereg, state, _ = _loop_state(case, loops)
    led.consume(rid, loop_state_digest(prereg, state))  # 승인 없음·거절·재소비·상태 변경이면 여기서 멈춘다
    request = {"case": case, "request_id": rid, "correction_id": req["correction_id"],
               "recovery_mode": req["recovery_mode"], "protocol_case_arm": req["protocol_case_arm"],
               "budget_gpu_s": req["budget_gpu_s"], "tag": tag or f"correct_{rid}"}
    t0 = time.perf_counter()
    res = execute(request)  # 예외는 그대로 전파한다. 요청은 running(고아)으로 남아 recover로 닫는다.
    gpu_s = res.get("gpu_s", time.perf_counter() - t0)
    return led.receipt(rid, res.get("tag", request["tag"]), res["verdict"], res.get("result_sha"), gpu_s,
                       res.get("exit_code"))


def isaac_execute(isaaclab: Path):
    """evals/run_training_correction.py execute를 부르는 실행 함수. Isaac·GPU를 쓰므로 메인 오케스트레이터만 부른다.

    v3 실행기는 case(case_01..03)의 세 arm을 모두 학습한다. verdict 판정은 v4 실행기가 정해지면 이 함수에서 채운다.
    """
    def _execute(request: dict) -> dict:
        sys.path.insert(0, str(ROOT / "evals"))
        import run_training_correction as T  # noqa: E402  (GPU 모듈은 호출 시점에만 불러온다)
        t0 = time.perf_counter()
        out = T.execute(request["tag"], request["protocol_case_arm"]["case"], isaaclab)
        sha = hashlib.sha256(json.dumps(out, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        bad = out.get("status") in ("invalid", "gpu_busy_no_launch")
        return {"exit_code": 1 if bad else 0, "gpu_s": round(time.perf_counter() - t0, 1), "verdict": "unknown",
                "result_sha": sha, "tag": request["tag"]}
    return _execute


def status(case: str, loops: Path = LOOPS) -> dict:
    led = _ledger(case, loops)
    return {"requests": led.requests(), "holds": [e for e in led.events() if e["event"] == "held"],
            "orphans": led.orphans()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E2 교정 제안·승인·실행")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose")
    p.add_argument("case")
    p.add_argument("--source-report", type=Path, required=True, help="결함 실행의 fixed_eval_v1 report JSON")
    p.add_argument("--budget-gpu-s", type=int, default=DEFAULT_BUDGET_GPU_S)
    for name in ("approve", "reject"):
        q = sub.add_parser(name)
        q.add_argument("case")
        q.add_argument("request_id")
        q.add_argument("--approver", required=True)
        if name == "reject":
            q.add_argument("--reason", required=True)
    r = sub.add_parser("run")
    r.add_argument("case")
    r.add_argument("request_id")
    r.add_argument("--tag")
    r.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    s = sub.add_parser("status")
    s.add_argument("case")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "propose":
            raw = args.source_report.read_bytes()
            sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
            out = propose(args.case, json.loads(raw), args.budget_gpu_s, source_report_sha=sha)
        elif args.cmd == "approve":
            out = _ledger(args.case, LOOPS).approve(args.request_id, args.approver)
        elif args.cmd == "reject":
            out = _ledger(args.case, LOOPS).reject(args.request_id, args.approver, args.reason)
        elif args.cmd == "run":
            out = run(args.case, args.request_id, isaac_execute(args.isaaclab), tag=args.tag)
        else:
            out = status(args.case)
        print(json.dumps(out, ensure_ascii=False, indent=1))
    except (LedgerError, ValueError) as e:
        print(f"거부: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
