"""결함 주입 벤치마크 평가: 에이전트(NAT 도구 호출) vs 대조군(단일 프롬프트) vs 무작위.

사용: uv run python evals/run_eval.py --seed 42 --mode agent|control|both
NVIDIA_API_KEY 환경변수가 필요하다(OpenShell 샌드박스에서는 provider가 주입).
정답표(v1: 공개 bench/answer_key.json)는 채점 단계에서만 읽고 에이전트 입력에는 넣지 않는다.
같은 checkout에 정답표가 있으므로 새 채점 평가는 정답 파일이 없는 실행 위치에서 돌린다(docs/IMPLEMENTATION-ORDER.md 2절 A).
저장된 결과의 재채점은 evals/replay.py가 같은 채점 규칙(rl_triage.scoring)으로 한다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.leakcheck import check_blind_workspace  # noqa: E402
from rl_triage.scoring import score_blind, score_changes  # noqa: E402

MODEL = os.environ.get("TRIAGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")

CONTROL_PROMPT = """You are given a failed Isaac Lab locomotion RL training run and a healthy reference run.
Several config changes were made; at most one is the root cause of the failure.
Rank ALL change_ids from most to least likely root cause.
Answer ONLY with JSON: {{"ranking": ["chX", ...], "reason": "..."}}

CHANGES:
{changes}

TELEMETRY (last-20% means vs reference):
{overview}
"""


BLIND_CONTROL_PROMPT = """A failed Isaac Lab locomotion RL training run (Unitree Go2, flat, RSL-RL PPO) deviates from a healthy
reference run. Nobody knows what was changed. Rank these failure mechanisms from most to least likely:
reward, actuator, exploration, optimizer, physics, termination.
Answer ONLY with JSON: {{"mechanism_ranking": ["...", ...], "reason": "..."}}

TELEMETRY (last-20% means vs reference):
{overview}
"""


def run_agent_blind(ws: Path, case_id: str) -> dict:
    diag = ws / "diagnoses" / f"{case_id}.json"
    if diag.exists():
        diag.unlink()
    trace = ROOT / "evals" / "results" / "traces" / f"blind_{ws.name}_{case_id}_{int(time.time())}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TRIAGE_WORKSPACE": str(ws), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "TRIAGE_TRACE": str(trace)}
    t0 = time.time()
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "nat", "run", "--config_file", str(ROOT / "configs" / "triage_blind.yml"),
         "--input", f"Diagnose the failed training case_id={case_id}. Rank the failure mechanisms."],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT, timeout=900)
    out = {"elapsed_s": round(time.time() - t0, 1), "exit_code": proc.returncode, "trace": trace.name,
           "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-1500:], "ranking": None}
    if diag.exists():
        out["ranking"] = json.loads(diag.read_text(encoding="utf-8"))["mechanism_ranking"]
    out["suspected"] = out["ranking"][0] if out["ranking"] else None
    return out


def run_control_blind(ws: Path, case_id: str) -> dict:
    from openai import OpenAI
    sys.path.insert(0, str(ROOT / "src"))
    from rl_triage import triage_tools as T
    T.WORKSPACE = ws
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
    prompt = BLIND_CONTROL_PROMPT.format(overview=json.dumps(T.telemetry_overview(case_id)["rows"], ensure_ascii=False))
    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                              temperature=0.2, max_tokens=16384)
        text = resp.choices[0].message.content or "empty response"
    except Exception as e:
        return {"elapsed_s": round(time.time() - t0, 1), "suspected": None, "ranking": None,
                "raw_tail": f"{type(e).__name__}: {e}"[-1500:]}
    ranking = None
    try:
        ranking = json.loads(text[text.index("{"): text.rindex("}") + 1])["mechanism_ranking"]
    except (ValueError, KeyError):
        pass
    return {"elapsed_s": round(time.time() - t0, 1), "suspected": ranking[0] if ranking else None,
            "ranking": ranking, "raw_tail": text[-1500:]}


def score(ws: Path, case_id: str, suspected: str | None, key: dict) -> dict:
    case = json.loads((ws / "cases" / case_id / "case.json").read_text(encoding="utf-8"))
    return score_changes(case_id, suspected, case["overrides"], key)


def run_agent(ws: Path, case_id: str) -> dict:
    trace = ROOT / "evals" / "results" / "traces" / f"{ws.name}_{case_id}_{int(time.time())}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TRIAGE_WORKSPACE": str(ws), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "TRIAGE_TRACE": str(trace)}
    prereg = ws / "preregistrations" / f"{case_id}.json"
    if prereg.exists():
        prereg.unlink()
    t0 = time.time()
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "nat", "run", "--config_file", str(ROOT / "configs" / "triage_workflow.yml"),
         "--input", f"Triage failed training case_id={case_id}. Find the root-cause change and register the next experiment."],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT, timeout=900)
    out = {"elapsed_s": round(time.time() - t0, 1), "exit_code": proc.returncode, "trace": trace.name,
           "stdout_tail": proc.stdout[-3000:], "stderr_tail": proc.stderr[-1500:]}
    if prereg.exists():
        doc = json.loads(prereg.read_text(encoding="utf-8"))
        out.update(suspected=doc["suspected_change_id"], ranking=doc["ranking"], prereg=doc)
    else:
        out.update(suspected=None, ranking=None)
    return out


def run_control(ws: Path, case_id: str) -> dict:
    from openai import OpenAI
    sys.path.insert(0, str(ROOT / "src"))
    from rl_triage import triage_tools as T
    T.WORKSPACE = ws
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
    prompt = CONTROL_PROMPT.format(changes=json.dumps(T.list_changes(case_id), ensure_ascii=False, default=str),
                                   overview=json.dumps(T.telemetry_overview(case_id)["rows"], ensure_ascii=False))
    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                              temperature=0.2, max_tokens=16384)
        text = resp.choices[0].message.content or ""
    except Exception as e:  # API 오류는 인프라 오류로 기록하고 평가를 계속한다
        return {"elapsed_s": round(time.time() - t0, 1), "suspected": None, "ranking": None,
                "raw_tail": f"{type(e).__name__}: {e}"[-1500:]}
    if not text:
        text = "empty response"
    ranking = None
    try:
        ranking = json.loads(text[text.index("{"): text.rindex("}") + 1])["ranking"]
    except (ValueError, KeyError):
        pass
    return {"elapsed_s": round(time.time() - t0, 1), "suspected": ranking[0] if ranking else None,
            "ranking": ranking, "raw_tail": text[-1500:]}


INFRA_MARKERS = ("empty response", "rate limit", "429", "502", "503", "504", "timed out", "Connection", "overloaded")


def is_infra_error(res: dict) -> bool:
    text = (res.get("stderr_tail") or "") + (res.get("raw_tail") or "")
    return any(m in text for m in INFRA_MARKERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["agent", "control", "both"], default="both")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--retry-infra", action="store_true",
                    help="기존 결과에서 인프라 오류(LLM 빈 응답 등으로 결론 없이 종료)만 다시 실행해 attempt를 늘려 기록한다")
    ap.add_argument("--bench", choices=["v1", "v2"], default="v1")
    ap.add_argument("--task", choices=["changes", "blind"], default="changes",
                    help="changes: 설정 변경 중 원인 찾기, blind: 변경 목록 없이 메커니즘 진단(과제 B)")
    ap.add_argument("--tag", default="", help="결과 파일 하위 폴더(evals/results/<tag>/)")
    ap.add_argument("--max-attempts", type=int, default=4, help="인프라 오류일 때만 대기 후 재시도하는 최대 횟수")
    args = ap.parse_args()
    ws_root = "workspace_blind" if args.task == "blind" else ("workspace" if args.bench == "v1" else "workspace_v2")
    ws = ROOT / ws_root / f"seed{args.seed}"
    # v1 정답표는 평가 종료 후 공개했다(비공개 사본과 SHA256 동일, bench/reference/manifest.json w0_check).
    # v2는 채점에 쓰지 않아 공개본이 없다.
    key_path = ROOT / "bench" / ("answer_key.json" if args.bench == "v1" else "private/answer_key_v2.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if args.task == "blind":  # workspace_blind는 bench/build_workspace.py가 bench/cases.json으로 만든다
        cases = json.loads((ROOT / "bench" / "cases.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "bench" / "reference" / "manifest.json").read_text(encoding="utf-8"))
        ref_sha = manifest["seeds"].get(str(args.seed), {}).get("sha256")  # 공개 기준 params가 없는 seed는 문자열 검사만
        problems = check_blind_workspace(ws, cases, key, ref_sha)
        if problems:
            raise SystemExit("blind 작업공간 유출 검사 실패:\n" + "\n".join(problems))
    case_ids = args.cases or sorted(p.name for p in (ws / "cases").iterdir())
    modes = ["agent", "control"] if args.mode == "both" else [args.mode]
    results_path = ROOT / "evals" / "results" / args.tag / f"seed{args.seed}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [(cid, mode, 1) for cid in case_ids for mode in modes]
    if args.retry_infra:
        # 모델 판단이 아니라 인프라 오류로 결론 없이 끝난 실행만 다시 돌린다. 원래 기록은 그대로 둔다.
        rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        latest = {}
        for r in rows:
            latest[(r["case_id"], r["mode"])] = r
        todo = [(c, m, r.get("attempt", 1) + 1) for (c, m), r in sorted(latest.items())
                if m in modes and r.get("suspected") is None and is_infra_error(r)]
        print(f"infra retry: {len(todo)} runs")
    for cid, mode, attempt in todo:
        # 인프라 오류(서버 과부하·빈 응답 등)만 대기 후 재시도한다. 모든 시도를 기록한다.
        while True:
            if args.task == "blind":
                res = run_agent_blind(ws, cid) if mode == "agent" else run_control_blind(ws, cid)
                infra = res.get("suspected") is None and is_infra_error(res)
                res.update(score_blind(cid, res.get("ranking"), key), case_id=cid, seed=args.seed, mode=mode,
                           model=MODEL, bench=args.bench, task="blind", attempt=attempt, infra_error=infra)
            else:
                res = run_agent(ws, cid) if mode == "agent" else run_control(ws, cid)
                infra = res.get("suspected") is None and is_infra_error(res)
                res.update(score(ws, cid, res.get("suspected"), key), case_id=cid, seed=args.seed, mode=mode,
                           model=MODEL, bench=args.bench, attempt=attempt, infra_error=infra)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"{cid} {mode:7s} attempt={attempt} suspected={res['suspected']} truth={res['truth']} "
                  f"correct={res['correct']} infra={infra} ({res['elapsed_s']}s)", flush=True)
            if not infra or attempt >= args.max_attempts:
                break
            time.sleep(30 * 2 ** (attempt - 1))
            attempt += 1


if __name__ == "__main__":
    main()
