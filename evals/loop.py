"""P1-A 사람 승인 고리 CLI(docs/P1-A-LOOP.md 3절). 원장·사전등록·상태는 evals/loops/<케이스>/에 남는다.

  python evals/loop.py start <케이스> --ranking reward physics optimizer --top-k 3 --budget 4 \
      --checkpoint-sha <sha256> --reference <NONE probe 결과 JSON>
  python evals/loop.py approve <케이스> <request_id> --approver <이름>
  python evals/loop.py reject  <케이스> <request_id> --approver <이름> --reason <이유>
  python evals/loop.py run     <케이스> <request_id> [--tag <probe 결과 tag>]   # 승인된 probe 하나를 Isaac에서 잰다
  python evals/loop.py status  <케이스>

- start는 상위 가설과 각 가설의 probe 예상을 사전등록하고 첫 probe를 제안한다(승인 대기).
- run은 승인을 한 번 소비하고 probe를 재서 판정(probe_loop.classify)·갱신한 뒤, 결론이 안 났으면 다음 probe를 제안한다.
- 에이전트·모델은 이 CLI를 부르지 않는다. 승인·거절은 사람이 한다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage import probe_loop as L  # noqa: E402

LOOPS = ROOT / "evals" / "loops"


def _dir(case: str) -> Path:
    if not case.replace("_", "").isalnum():
        raise SystemExit(f"케이스 이름이 이상하다: {case!r}")
    return LOOPS / case


def _load(case: str) -> tuple[dict, dict, L.Ledger]:
    d = _dir(case)
    return (json.loads((d / "prereg.json").read_text(encoding="utf-8")),
            json.loads((d / "state.json").read_text(encoding="utf-8")), L.Ledger(d / "ledger.jsonl"))


def _save_state(case: str, state: dict) -> None:
    (_dir(case) / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _propose_next(case: str, prereg: dict, state: dict, led: L.Ledger) -> str:
    left = state["budget"] - len(state["observed"])
    st = L.status(state["hypotheses"], state["observed"], left)
    # 사람이 거절한 probe는 다시 제안하지 않는다(관측으로 세지 않으므로 예산도 쓰지 않는다).
    done = {**{p: "unknown" for p in state.get("skipped", [])}, **state["observed"]}
    probe = L.next_probe(state["hypotheses"], done) if st == "open" else None
    if probe is None:
        state["status"] = "unidentifiable" if st == "open" else st
        _save_state(case, state)
        return f"종료: {state['status']} 남은 가설={state['hypotheses']}"
    rid = led.propose(prereg, probe, budget_gpu_s=L.PROBES[probe]["cost_gpu_s"] * 2)["request_id"]
    state["pending"] = rid
    _save_state(case, state)
    expect = {h: L.PROBES[probe]["expect"].get(h) for h in state["hypotheses"]}
    return f"제안 {rid}: {probe} ({L.PROBES[probe]['measures']}) 가설별 예상={expect} — 사람 승인 필요"


def start(case: str, ranking: list[str], top_k: int, budget: int, checkpoint_sha: str, reference: dict) -> str:
    d = _dir(case)
    if d.exists():
        raise SystemExit(f"evals/loops/{case}가 이미 있다. 고리 기록은 덮어쓰지 않는다")
    bad = [m for m in ranking if m not in L.MECHANISMS]
    if bad or not ranking:
        raise SystemExit(f"모르는 범주 {bad}")
    hyps = ranking[:top_k]
    prereg = {"case": case, "hypotheses": hyps, "checkpoint_sha256": checkpoint_sha,
              "expectations": {p: {h: L.PROBES[p]["expect"].get(h) for h in hyps} for p in L.PROBES},
              "thresholds": {p: list(v) for p, v in L.THRESHOLDS.items()}, "budget_probes": budget}
    d.mkdir(parents=True)
    (d / "prereg.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (d / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    state = {"hypotheses": hyps, "observed": {}, "skipped": [], "budget": budget, "status": "open", "pending": None,
             "history": []}
    return _propose_next(case, prereg, state, L.Ledger(d / "ledger.jsonl"))


def isaac_execute(case: str, probe: str, tag: str) -> tuple[dict | None, float, int | None]:
    """evals/run_probes.py로 probe 하나를 잰다. (측정값, GPU 초, 종료 코드)."""
    import time
    t0 = time.time()
    out = ROOT / "evals" / "results" / tag / "probes" / f"{case}.json"
    proc = subprocess.run([sys.executable, str(ROOT / "evals" / "run_probes.py"), "--tag", tag, "--runs", case,
                           "--probes", probe], cwd=ROOT)
    if proc.returncode != 0 or not out.exists():
        return None, time.time() - t0, proc.returncode
    rec = json.loads(out.read_text(encoding="utf-8"))["probes"].get(probe, {})
    return rec.get("measurement"), time.time() - t0, rec.get("exit")


def run(case: str, rid: str, execute) -> str:
    prereg, state, led = _load(case)
    req = led.requests().get(rid)
    if req is None:
        raise SystemExit(f"{rid}: 없는 요청")
    probe = req["probe"]
    led.consume(rid, prereg, probe)  # 승인 없거나 이미 소비했으면 여기서 멈춘다
    measurement, gpu_s, exit_code = execute(case, probe)
    reference = json.loads((_dir(case) / "reference.json").read_text(encoding="utf-8"))
    outcome = L.classify(probe, measurement, reference.get(probe)) if exit_code == 0 else "unknown"
    led.receipt(rid, outcome, measurement or {}, gpu_s=round(gpu_s, 1), exit_code=exit_code)
    state["observed"][probe] = outcome
    state["hypotheses"], dropped = L.update(state["hypotheses"], probe, outcome)
    state["history"].append({"request_id": rid, "probe": probe, "outcome": outcome, "dropped": dropped})
    state["pending"] = None
    msg = f"{probe}: {outcome} 기각={dropped} 남은 가설={state['hypotheses']}\n"
    return msg + _propose_next(case, prereg, state, led)


def reject(case: str, rid: str, approver: str, reason: str) -> str:
    prereg, state, led = _load(case)
    probe = led.requests()[rid]["probe"] if rid in led.requests() else None
    led.reject(rid, approver, reason)
    state["skipped"].append(probe)
    state["history"].append({"request_id": rid, "probe": probe, "rejected": reason})
    state["pending"] = None
    return f"거절 {rid}({probe}): {reason}\n" + _propose_next(case, prereg, state, led)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1-A 사람 승인 probe 고리")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("case")
    s.add_argument("--ranking", nargs="+", required=True)
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--budget", type=int, default=4)
    s.add_argument("--checkpoint-sha", required=True)
    s.add_argument("--reference", type=Path, required=True, help="같은 seed NONE 실행의 probe 결과 JSON")
    for name in ("approve", "reject"):
        p = sub.add_parser(name)
        p.add_argument("case")
        p.add_argument("request_id")
        p.add_argument("--approver", required=True)
        if name == "reject":
            p.add_argument("--reason", required=True)
    r = sub.add_parser("run")
    r.add_argument("case")
    r.add_argument("request_id")
    r.add_argument("--tag", default="p1a_loop_probes")
    st = sub.add_parser("status")
    st.add_argument("case")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "start":
            ref = json.loads(args.reference.read_text(encoding="utf-8"))
            ref = {p: v.get("measurement") for p, v in ref.get("probes", {}).items()}
            print(start(args.case, args.ranking, args.top_k, args.budget, args.checkpoint_sha, ref))
        elif args.cmd == "approve":
            _, _, led = _load(args.case)
            led.approve(args.request_id, args.approver)
            print(f"승인 {args.request_id}")
        elif args.cmd == "reject":
            print(reject(args.case, args.request_id, args.approver, args.reason))
        elif args.cmd == "run":
            print(run(args.case, args.request_id, lambda c, p: isaac_execute(c, p, args.tag)))
        else:
            _, state, _ = _load(args.case)
            print(json.dumps(state, ensure_ascii=False, indent=1))
    except L.LedgerError as e:
        print(f"원장 규칙 위반: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
