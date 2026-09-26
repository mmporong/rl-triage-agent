"""결함 주입 벤치마크 평가: 에이전트(NAT 도구 호출) vs 대조군(단일 프롬프트) vs 무작위.

사용: uv run python evals/run_eval.py --seed 42 --mode agent|control|both
NVIDIA_API_KEY 환경변수가 필요하다(OpenShell 샌드박스에서는 provider가 주입).
정답표(bench/private/answer_key.json)는 채점 단계에서만 읽고 에이전트 입력에는 넣지 않는다.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def score(ws: Path, case_id: str, suspected: str | None, key: dict) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from rl_triage import triage_tools as T
    T.WORKSPACE = ws
    changes = {c["change_id"]: f'{c["key"]}={json.dumps(c["new"]) if not isinstance(c["new"], str) else c["new"]}'
               for c in T.list_changes(case_id)}
    harmful = key["cases"][case_id]["harmful_override"].split("=", 1)[0]
    truth = next(cid for cid, ov in changes.items() if ov.split("=", 1)[0] == harmful)
    return {"truth": truth, "suspected": suspected, "correct": suspected == truth,
            "category": key["cases"][case_id]["category"]}


def run_agent(ws: Path, case_id: str) -> dict:
    env = {**os.environ, "TRIAGE_WORKSPACE": str(ws), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    prereg = ws / "preregistrations" / f"{case_id}.json"
    if prereg.exists():
        prereg.unlink()
    t0 = time.time()
    proc = subprocess.run(
        ["uv", "run", "--no-sync", "nat", "run", "--config_file", str(ROOT / "configs" / "triage_workflow.yml"),
         "--input", f"Triage failed training case_id={case_id}. Find the root-cause change and register the next experiment."],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT, timeout=900)
    out = {"elapsed_s": round(time.time() - t0, 1), "exit_code": proc.returncode,
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
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}],
                                          temperature=0.2, max_tokens=4096)
    text = resp.choices[0].message.content or ""
    ranking = None
    try:
        ranking = json.loads(text[text.index("{"): text.rindex("}") + 1])["ranking"]
    except (ValueError, KeyError):
        pass
    return {"elapsed_s": round(time.time() - t0, 1), "suspected": ranking[0] if ranking else None,
            "ranking": ranking, "raw_tail": text[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["agent", "control", "both"], default="both")
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    ws = ROOT / "workspace" / f"seed{args.seed}"
    key = json.loads((ROOT / "bench" / "private" / "answer_key.json").read_text(encoding="utf-8"))
    case_ids = args.cases or sorted(p.name for p in (ws / "cases").iterdir())
    modes = ["agent", "control"] if args.mode == "both" else [args.mode]
    results_path = ROOT / "evals" / "results" / f"seed{args.seed}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        for mode in modes:
            res = run_agent(ws, cid) if mode == "agent" else run_control(ws, cid)
            res.update(score(ws, cid, res.get("suspected"), key), case_id=cid, seed=args.seed, mode=mode, model=MODEL)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            print(f"{cid} {mode:7s} suspected={res['suspected']} truth={res['truth']} correct={res['correct']} "
                  f"({res['elapsed_s']}s)")


if __name__ == "__main__":
    main()
