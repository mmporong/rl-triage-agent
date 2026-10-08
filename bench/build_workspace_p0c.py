"""P0-C holdout 실행을 과제 B(blind) 작업공간으로 정리한다. 정답표는 복사하지 않는다(docs/P0-C-HOLDOUT.md).

사용:
  python bench/build_workspace_p0c.py register <seed> --run-params <Isaac Lab rsl_rl/unitree_go2_flat 폴더>
      # 기준 실행(baseline_p0c_s<seed>)의 params를 bench/reference/params/p0c_seed<seed>/에 복사하고 manifest에 기록
  python bench/build_workspace_p0c.py build <seed> [--out <작업공간 루트>]
      # bench/runs_p0c 텔레메트리 + 공개 기준 params → workspace_p0c/seed<seed>, 유출 검사 통과 필수

같은 seed의 TRIAGE_FAULT=NONE 실행이 reference다. 케이스 텔레메트리 이름은 h01~h06이라 결함 정보가 없다.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rl_triage.leakcheck import check_blind_workspace, sha256  # noqa: E402

RUNS = ROOT / "bench" / "runs_p0c"
MANIFEST = ROOT / "bench" / "reference" / "manifest.json"
KEY_PATH = ROOT / "bench" / "private" / "answer_key_p0c.json"
CASE_IDS = [f"h{i:02d}" for i in range(1, 7)]
USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/")


def _show(p: Path) -> str:
    p = p.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else f"<external>/{p.name}"


def register(seed: int, log_root: Path) -> None:
    tele = RUNS / f"baseline_p0c_s{seed}.telemetry.json"
    run_dir_name = json.loads(tele.read_text(encoding="utf-8"))["run_dir_name"]
    src = log_root / run_dir_name / "params"
    dst = ROOT / "bench" / "reference" / "params" / f"p0c_seed{seed}"
    if dst.exists():
        raise SystemExit(f"{_show(dst)}가 이미 있다. 덮어쓰지 않는다")
    for name in ("env.yaml", "agent.yaml"):
        if USER_PATH.search((src / name).read_text(encoding="utf-8", errors="replace")):
            raise SystemExit(f"{name}에 사용자 경로가 있어 공개 사본을 만들지 않는다")
    dst.mkdir(parents=True)
    for name in ("env.yaml", "agent.yaml"):
        shutil.copyfile(src / name, dst / name)  # 바이트 그대로(.gitattributes -text)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest.setdefault("p0c_seeds", {})[str(seed)] = {
        "source_run_dir_name": run_dir_name, "telemetry": _show(tele),
        "train": "4096 env x 300 iterations, TRIAGE_FAULT=NONE via bench/train_with_fault.py",
        "sha256": {n: sha256(dst / n) for n in ("env.yaml", "agent.yaml")}}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"registered {_show(dst)} from {run_dir_name}")


def build(seed: int, out: Path | None) -> None:
    ref = json.loads(MANIFEST.read_text(encoding="utf-8")).get("p0c_seeds", {}).get(str(seed))
    if ref is None:
        raise SystemExit(f"manifest에 p0c seed {seed}가 없다. 먼저 register한다")
    params = ROOT / "bench" / "reference" / "params" / f"p0c_seed{seed}"
    for name, want in ref["sha256"].items():
        if sha256(params / name) != want:
            raise SystemExit(f"{_show(params / name)} SHA256이 manifest와 다르다")
    ref_tele = ROOT / ref["telemetry"]
    if json.loads(ref_tele.read_text(encoding="utf-8"))["run_dir_name"] != ref["source_run_dir_name"]:
        raise SystemExit("기준 텔레메트리와 params의 실행이 다르다")
    ws = (out.resolve() if out else ROOT / "workspace_p0c") / f"seed{seed}"
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "reference" / "params").mkdir(parents=True)
    shutil.copy(ref_tele, ws / "reference" / "telemetry.json")
    for name in ref["sha256"]:
        shutil.copy(params / name, ws / "reference" / "params" / name)
    for cid in CASE_IDS:
        tele = RUNS / f"{cid}_s{seed}.telemetry.json"
        if not tele.exists():
            shutil.rmtree(ws)
            raise SystemExit(f"{_show(tele)}가 없다")
        (ws / "cases" / cid).mkdir(parents=True)
        shutil.copy(tele, ws / "cases" / cid / "telemetry.json")
    key = json.loads(KEY_PATH.read_text(encoding="utf-8")) if KEY_PATH.exists() else None
    problems = check_blind_workspace(ws, [{"case_id": c, "overrides": []} for c in CASE_IDS], key, ref["sha256"])
    if problems:
        shutil.rmtree(ws)
        raise SystemExit("blind 작업공간 유출 검사 실패(작업공간 삭제):\n" + "\n".join(problems))
    print(f"{_show(ws)}: cases={len(CASE_IDS)} leak_check=pass key_markers={'yes' if key else 'no'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("seed", type=int)
    r.add_argument("--run-params", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("seed", type=int)
    b.add_argument("--out", type=Path)
    args = ap.parse_args()
    register(args.seed, args.run_params) if args.cmd == "register" else build(args.seed, args.out)


if __name__ == "__main__":
    main()
