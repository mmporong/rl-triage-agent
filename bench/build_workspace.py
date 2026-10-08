"""벤치마크 실행 결과를 에이전트 작업공간(workspace/)으로 정리한다. 정답표는 복사하지 않는다.

케이스 ID는 '<case>_s<seed>'이고, 같은 seed의 정상 기준 실행을 reference로 쓴다.
기준 params는 공개 사본 bench/reference/params/seed<N>/에서 가져오고 manifest의 SHA256·실행 이름과 대조한다.
과제 B(--blind) 작업공간은 만든 뒤 정답 유출 검사(rl_triage.leakcheck)를 통과해야 한다.

사용: python bench/build_workspace.py <seed> [--blind] [--out <작업공간 루트>]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from rl_triage.leakcheck import check_blind_workspace, sha256  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int, nargs="?", default=42)
ap.add_argument("--blind", action="store_true", help="과제 B: case.json(설정 변경 목록)을 넣지 않는다")
ap.add_argument("--out", type=Path, help="작업공간 루트(기본: workspace/ 또는 workspace_blind/)")
args = ap.parse_args()
seed, blind = args.seed, args.blind

runs = root / "bench" / "runs"
cases_list = json.loads((root / "bench" / "cases.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "bench" / "reference" / "manifest.json").read_text(encoding="utf-8"))
ref_tele = runs / f"baseline_s{seed}.telemetry.json"
if not ref_tele.exists():
    raise SystemExit(f"기준 실행 없음: {ref_tele.relative_to(root)}")
ref = manifest["seeds"].get(str(seed))
if ref is None:
    raise SystemExit(f"공개 기준 params 없음: seed {seed}. bench/reference/params/seed{seed}/와 manifest.json에 먼저 추가한다")
params_dir = root / "bench" / "reference" / "params" / f"seed{seed}"
for name, want in ref["sha256"].items():
    if sha256(params_dir / name) != want:
        raise SystemExit(f"{(params_dir / name).relative_to(root)} SHA256이 manifest와 다르다")
run_name = json.loads(ref_tele.read_text(encoding="utf-8"))["run_dir_name"]
if run_name != ref["source_run_dir_name"]:
    raise SystemExit(f"기준 텔레메트리({run_name})와 params({ref['source_run_dir_name']})의 실행이 다르다")

ws_seed = (args.out.resolve() if args.out else root / ("workspace_blind" if blind else "workspace")) / f"seed{seed}"
if ws_seed.exists():
    shutil.rmtree(ws_seed)
(ws_seed / "reference" / "params").mkdir(parents=True)
shutil.copy(ref_tele, ws_seed / "reference" / "telemetry.json")
for name in ref["sha256"]:
    shutil.copy(params_dir / name, ws_seed / "reference" / "params" / name)
n = 0
for case in cases_list:
    cid = case["case_id"]
    tele = runs / f"{cid}_s{seed}.telemetry.json"
    if not tele.exists():
        continue
    d = ws_seed / "cases" / cid
    d.mkdir(parents=True)
    shutil.copy(tele, d / "telemetry.json")
    if not blind:
        (d / "case.json").write_text(json.dumps({"case_id": cid, "overrides": case["overrides"]}, ensure_ascii=False), encoding="utf-8")
    n += 1
if blind:
    key = json.loads((root / "bench" / "answer_key.json").read_text(encoding="utf-8"))
    problems = check_blind_workspace(ws_seed, cases_list, key, ref["sha256"])
    if problems:
        shutil.rmtree(ws_seed)
        raise SystemExit("blind 작업공간 유출 검사 실패(작업공간 삭제):\n" + "\n".join(problems))
shown = ws_seed.relative_to(root).as_posix() if ws_seed.is_relative_to(root) else ws_seed.as_posix()
print(f"{shown}: cases={n} blind={blind} leak_check={'pass' if blind else 'n/a'}")
