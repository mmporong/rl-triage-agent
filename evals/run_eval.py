"""결함 주입 벤치마크 평가: 에이전트(NAT 도구 호출) vs 대조군(단일 프롬프트) vs 무작위.

사용: uv run python evals/run_eval.py --seed 42 --mode agent|control|control_full|both [...]
      --bench p0c --task blind: 숨은 결함 holdout(docs/P0-C-HOLDOUT.md). TRIAGE_ANALYSIS_SANDBOX=required와
      평가 전 canary(분석 코드가 정답표·/proc 환경을 못 읽음)를 통과해야 시작한다.
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
    out.update(trace_sandbox_counts(trace))
    return out


BLIND_CONTROL_FULL_PROMPT = """A failed Isaac Lab locomotion RL training run (Unitree Go2, flat, RSL-RL PPO) deviates from a healthy
reference run. Nobody knows what was changed. Rank these failure mechanisms from most to least likely:
reward, actuator, exploration, optimizer, physics, termination.
Answer ONLY with JSON: {{"mechanism_ranking": ["...", ...], "reason": "..."}}

FULL TELEMETRY OF THE FAILED RUN (every logged scalar, one value per training iteration):
{run}

FULL TELEMETRY OF THE HEALTHY REFERENCE RUN:
{ref}
"""


def _full_series(path: Path) -> str:
    series = json.loads(path.read_text(encoding="utf-8"))["series"]
    return json.dumps({tag: [float(f"{v:.4g}") for v in vals] for tag, vals in sorted(series.items())}, separators=(",", ":"))


def run_control_blind(ws: Path, case_id: str, full: bool = False) -> dict:
    """단일 프롬프트 대조군. full=True면 요약 대신 실패·기준 실행의 전체 시계열을 준다(에이전트와 같은 정보)."""
    from openai import OpenAI
    sys.path.insert(0, str(ROOT / "src"))
    from rl_triage import triage_tools as T
    T.WORKSPACE = ws
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])
    if full:
        prompt = BLIND_CONTROL_FULL_PROMPT.format(run=_full_series(ws / "cases" / case_id / "telemetry.json"),
                                                  ref=_full_series(ws / "reference" / "telemetry.json"))
    else:
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


AGENT_PYTHON = ("uv", "run", "--no-sync", "python")  # run_agent_blind의 `uv run --no-sync nat`과 같은 환경


def analysis_canary(ws: Path, case_id: str, answer_path: Path, runner=AGENT_PYTHON) -> dict:
    """에이전트와 같은 실행 경로(같은 runner·환경변수)의 하위 프로세스에서 run_analysis로 정답 파일과
    /proc 환경변수를 열어 본다(P0-C 수용 조건). 둘 다 거부되고 sandbox가 landlock이어야 통과다."""
    targets = [str(answer_path.resolve()), "/proc/self/environ"]
    code = ("for p in %r:\n    try:\n        open(p).read(1)\n        print('READ', p)\n"
            "    except PermissionError:\n        print('DENIED', p)\n" % targets)
    snippet = ("import json, sys\nfrom rl_triage import triage_tools as T\n"
               "print(json.dumps(T.run_analysis(sys.argv[1], sys.argv[2])))\n")
    env = {**os.environ, "TRIAGE_WORKSPACE": str(ws), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([*runner, "-c", snippet, code, case_id], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, cwd=ROOT, timeout=300)
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"runner": " ".join(runner), "denied": False, "error": (proc.stdout + proc.stderr)[-500:]}
    denied = all(f"DENIED {t}" in out["stdout"] for t in targets)
    return {"runner": " ".join(runner), "sandbox_mode": os.environ.get("TRIAGE_ANALYSIS_SANDBOX", "auto"),
            "sandbox": out["sandbox"], "targets": ["<answer key>", "/proc/self/environ"],
            "denied": denied, "exit_code": out["exit_code"], "stderr_tail": out["stderr"][-300:]}


def trace_sandbox_counts(trace: Path) -> dict:
    """에이전트 trace에서 run_analysis 호출 수와 Landlock이 걸린 호출 수."""
    calls = sandboxed = 0
    if trace.exists():
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("tool") == "run_analysis":
                calls += 1
                sandboxed += '"sandbox": "landlock"' in rec.get("result_head", "")
    return {"analysis_calls": calls, "analysis_sandboxed": sandboxed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", nargs="+", choices=["agent", "control", "control_full", "both"], default=["both"],
                    help="both=agent+control. control_full은 요약 대신 전체 시계열을 받는 단일 프롬프트(과제 B)")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--retry-infra", action="store_true",
                    help="기존 결과에서 인프라 오류(LLM 빈 응답 등으로 결론 없이 종료)만 다시 실행해 attempt를 늘려 기록한다")
    ap.add_argument("--bench", choices=["v1", "v2", "p0c"], default="v1")
    ap.add_argument("--task", choices=["changes", "blind"], default="changes",
                    help="changes: 설정 변경 중 원인 찾기, blind: 변경 목록 없이 메커니즘 진단(과제 B)")
    ap.add_argument("--tag", default="", help="결과 파일 하위 폴더(evals/results/<tag>/)")
    ap.add_argument("--max-attempts", type=int, default=4, help="인프라 오류일 때만 대기 후 재시도하는 최대 횟수")
    args = ap.parse_args()
    if args.bench == "p0c" and args.task != "blind":
        raise SystemExit("p0c는 과제 B(--task blind)만 있다")
    ws_root = {"v1": "workspace_blind" if args.task == "blind" else "workspace", "v2": "workspace_v2",
               "p0c": "workspace_p0c"}[args.bench]
    ws = ROOT / ws_root / f"seed{args.seed}"
    # v1 정답표는 평가 종료 후 공개했다(비공개 사본과 SHA256 동일, bench/reference/manifest.json w0_check).
    # v2는 채점에 쓰지 않아 공개본이 없다. p0c는 평가가 끝날 때까지 bench/private에만 있다.
    key_path = ROOT / "bench" / {"v1": "answer_key.json", "v2": "private/answer_key_v2.json",
                                 "p0c": "private/answer_key_p0c.json"}[args.bench]
    key = json.loads(key_path.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "bench" / "reference" / "manifest.json").read_text(encoding="utf-8"))
    if args.bench == "p0c":  # workspace_p0c는 bench/build_workspace_p0c.py가 만든다
        cases = [{"case_id": p.name, "overrides": []} for p in sorted((ws / "cases").iterdir())]
        ref_sha = manifest.get("p0c_seeds", {}).get(str(args.seed), {}).get("sha256")
        if not ref_sha:
            raise SystemExit(f"manifest에 p0c seed {args.seed}가 없다")
    elif args.task == "blind":  # workspace_blind는 bench/build_workspace.py가 bench/cases.json으로 만든다
        cases = json.loads((ROOT / "bench" / "cases.json").read_text(encoding="utf-8"))
        ref_sha = manifest["seeds"].get(str(args.seed), {}).get("sha256")  # 공개 기준 params가 없는 seed는 문자열 검사만
    if args.task == "blind":
        problems = check_blind_workspace(ws, cases, key, ref_sha)
        if problems:
            raise SystemExit("blind 작업공간 유출 검사 실패:\n" + "\n".join(problems))
    case_ids = args.cases or sorted(p.name for p in (ws / "cases").iterdir())
    modes = []
    for m in args.mode:
        modes += ["agent", "control"] if m == "both" else [m]
    modes = list(dict.fromkeys(modes))
    if "control_full" in modes and args.task != "blind":
        raise SystemExit("control_full은 과제 B(--task blind)만 있다")
    results_path = ROOT / "evals" / "results" / args.tag / f"seed{args.seed}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    sandbox = os.environ.get("TRIAGE_ANALYSIS_SANDBOX", "auto")
    if args.bench == "p0c":
        # 같은 실행 위치에 정답 파일이 있으므로, 분석 코드가 그것을 못 읽는다는 것을 평가 전에 확인한다.
        if sandbox != "required":
            raise SystemExit("p0c 평가는 TRIAGE_ANALYSIS_SANDBOX=required에서만 돌린다")
        canary = analysis_canary(ws, case_ids[0], key_path)
        (results_path.parent / f"canary_seed{args.seed}.json").write_text(json.dumps(canary, indent=1), encoding="utf-8")
        if not canary["denied"] or canary.get("sandbox") != "landlock":
            raise SystemExit(f"canary 실패: 분석 코드가 정답 파일 또는 /proc 환경을 읽을 수 있다 ({canary})")
        print(f"canary: denied ({canary['sandbox_mode']})", flush=True)
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
                res = (run_agent_blind(ws, cid) if mode == "agent"
                       else run_control_blind(ws, cid, full=(mode == "control_full")))
                infra = res.get("suspected") is None and is_infra_error(res)
                res.update(score_blind(cid, res.get("ranking"), key), case_id=cid, seed=args.seed, mode=mode,
                           model=MODEL, bench=args.bench, task="blind", attempt=attempt, infra_error=infra,
                           analysis_sandbox=sandbox)
            else:
                res = run_agent(ws, cid) if mode == "agent" else run_control(ws, cid)
                infra = res.get("suspected") is None and is_infra_error(res)
                res.update(score(ws, cid, res.get("suspected"), key), case_id=cid, seed=args.seed, mode=mode,
                           model=MODEL, bench=args.bench, attempt=attempt, infra_error=infra)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if args.bench == "p0c" and res.get("analysis_calls", 0) != res.get("analysis_sandboxed", 0):
                raise SystemExit(f"{cid} {mode}: 샌드박스 없이 실행된 run_analysis 호출이 있다. 평가를 멈춘다")
            print(f"{cid} {mode:7s} attempt={attempt} suspected={res['suspected']} truth={res['truth']} "
                  f"correct={res['correct']} infra={infra} ({res['elapsed_s']}s)", flush=True)
            if not infra or attempt >= args.max_attempts:
                break
            time.sleep(30 * 2 ** (attempt - 1))
            attempt += 1


if __name__ == "__main__":
    main()
