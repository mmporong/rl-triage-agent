"""Isaac 재평가 브리지: 에이전트가 사전등록한 실험을 사람이 승인하면 실제로 재학습해 가설을 판정한다.

에이전트 쪽(샌드박스)은 REST로 요청만 할 수 있다.
  GET  /runs/<case_id>?seed=42   이 케이스 텔레메트리 요약(읽기 전용)
  POST /eval                      {"case_id", "seed"} → 사전등록을 근거로 승인 대기 요청 생성(202)
  GET  /eval/<request_id>         상태 조회
승인·거절은 브리지 호스트의 사람만 CLI로 한다(에이전트 샌드박스에는 이 명령과 큐 경로가 없다).
  python -m rl_triage.eval_bridge list | approve <id> | reject <id> --reason ...
설정 변경 API(PUT /config/**)는 존재하지 않으며, OpenShell 경계(site_boundary.yaml)에서도 금지된다.

실행: python -m rl_triage.eval_bridge serve --port 8750
"""
from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "bench" / "private" / "eval_queue"
RUNS = ROOT / "bench" / "runs"

# 회복 판정: 무해 변경만 적용한 실행(benign_all)의 기준 대비 비율이 들어가는 범위를 여유 있게 잡는다.
RECOVERY_BAND = {
    "Train/mean_episode_length": (0.9, 1.1),
    "Train/mean_reward": (0.5, 1.8),
    "Policy/mean_noise_std": (0.5, 1.5),
}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _last20(tele: dict, tag: str):
    s = tele["summary"].get(tag)
    return None if s is None else s["mean_last_20pct"]


def recovery_verdict(run_tele: dict, ref_tele: dict) -> dict:
    checks, ok = {}, True
    for tag, (lo, hi) in RECOVERY_BAND.items():
        a, b = _last20(run_tele, tag), _last20(ref_tele, tag)
        ratio = None if a is None or not b else a / b
        passed = ratio is not None and lo <= ratio <= hi
        checks[tag] = {"run": a, "reference": b, "ratio": None if ratio is None else round(ratio, 3),
                       "band": [lo, hi], "pass": passed}
        ok &= passed
    return {"recovered": ok, "checks": checks}


def _workspace(seed: int) -> Path:
    return ROOT / "workspace" / f"seed{seed}"


def submit(case_id: str, seed: int) -> dict:
    prereg = _workspace(seed) / "preregistrations" / f"{case_id}.json"
    if not prereg.exists():
        return {"status": 400, "error": "사전등록 없음: 먼저 write_preregistration으로 실험을 등록해야 한다"}
    doc = json.loads(prereg.read_text(encoding="utf-8"))
    rid = uuid.uuid4().hex[:10]
    req = {"request_id": rid, "case_id": case_id, "seed": seed, "state": "pending_approval",
           "created_at": _now(), "preregistration": doc}
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / f"{rid}.json").write_text(json.dumps(req, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"status": 202, **req}


def _save(req):
    (QUEUE / f"{req['request_id']}.json").write_text(json.dumps(req, ensure_ascii=False, indent=1), encoding="utf-8")


def load(rid: str) -> dict:
    return json.loads((QUEUE / f"{rid}.json").read_text(encoding="utf-8"))


def approve(rid: str, max_iterations: int = 100) -> dict:
    """사람이 승인하면 의심 변경 하나만 되돌린 설정으로 Isaac에서 재학습하고 회복 여부를 판정한다."""
    req = load(rid)
    if req["state"] != "pending_approval":
        return req
    case = json.loads((_workspace(req["seed"]) / "cases" / req["case_id"] / "case.json").read_text(encoding="utf-8"))
    idx = int(req["preregistration"]["suspected_change_id"].removeprefix("ch")) - 1
    reverted = case["overrides"][idx]
    kept = [o for i, o in enumerate(case["overrides"]) if i != idx]
    run_id = f"{req['case_id']}_revert_{req['preregistration']['suspected_change_id']}"
    req.update(state="running", approved_at=_now(), reverted_override=reverted, kept_overrides=kept)
    _save(req)
    ps = ("& '{script}' -CaseId '{cid}' -Seed {seed} -MaxIterations {it} -Overrides @({ov})".format(
        script=ROOT / "bench" / "run_case.ps1", cid=run_id, seed=req["seed"], it=max_iterations,
        ov=",".join("'" + o.replace("'", "''") + "'" for o in kept)))
    proc = subprocess.run(["pwsh", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    tele_path = RUNS / f"{run_id}_s{req['seed']}.telemetry.json"
    if proc.returncode != 0 or not tele_path.exists():
        req.update(state="failed", finished_at=_now(), error=(proc.stdout + proc.stderr)[-1500:])
    else:
        run_tele = json.loads(tele_path.read_text(encoding="utf-8"))
        ref_tele = json.loads((RUNS / f"baseline_s{req['seed']}.telemetry.json").read_text(encoding="utf-8"))
        req.update(state="done", finished_at=_now(), telemetry=str(tele_path.name),
                   verdict=recovery_verdict(run_tele, ref_tele))
    _save(req)
    return req


def reject(rid: str, reason: str) -> dict:
    req = load(rid)
    req.update(state="rejected", rejected_at=_now(), reason=reason)
    _save(req)
    return req


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = u.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "runs":
            seed = int(parse_qs(u.query).get("seed", ["42"])[0])
            p = _workspace(seed) / "cases" / parts[1] / "telemetry.json"
            if not p.exists():
                return self._send(404, {"error": "없는 케이스"})
            return self._send(200, json.loads(p.read_text(encoding="utf-8"))["summary"])
        if len(parts) == 2 and parts[0] == "eval":
            try:
                return self._send(200, load(parts[1]))
            except FileNotFoundError:
                return self._send(404, {"error": "없는 요청"})
        self._send(404, {"error": "없는 경로"})

    def do_POST(self):
        if urlparse(self.path).path != "/eval":
            return self._send(404, {"error": "없는 경로"})
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        res = submit(body.get("case_id", ""), int(body.get("seed", 42)))
        self._send(res.pop("status"), res)

    def do_PUT(self):
        self._send(405, {"error": "설정 변경 API는 없다"})

    def log_message(self, fmt, *args):
        print(f"[bridge] {self.address_string()} {fmt % args}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8750)
    sub.add_parser("list")
    a = sub.add_parser("approve"); a.add_argument("request_id"); a.add_argument("--max-iterations", type=int, default=100)
    r = sub.add_parser("reject"); r.add_argument("request_id"); r.add_argument("--reason", required=True)
    p = sub.add_parser("submit"); p.add_argument("case_id"); p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd == "serve":
        print(f"eval bridge on {args.host}:{args.port}")
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    elif args.cmd == "list":
        for f in sorted(QUEUE.glob("*.json")) if QUEUE.exists() else []:
            r_ = json.loads(f.read_text(encoding="utf-8"))
            print(r_["request_id"], r_["state"], r_["case_id"], f"s{r_['seed']}",
                  r_["preregistration"]["suspected_change_id"], r_["preregistration"]["single_experimental_variable"][:60])
    elif args.cmd == "approve":
        print(json.dumps(approve(args.request_id, args.max_iterations), ensure_ascii=False, indent=1, default=str))
    elif args.cmd == "reject":
        print(json.dumps(reject(args.request_id, args.reason), ensure_ascii=False, indent=1))
    elif args.cmd == "submit":
        print(json.dumps(submit(args.case_id, args.seed), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
