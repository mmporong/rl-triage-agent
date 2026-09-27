"""벤치마크 실행 결과를 에이전트 작업공간(workspace/)으로 정리한다. 정답표는 복사하지 않는다.

케이스 ID는 '<case>_s<seed>'이고, 같은 seed의 정상 기준 실행을 reference로 쓴다.
params(yaml)는 기준 실행 폴더의 params/에서 가져온다.
"""
import json, shutil, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs, ws = root / "bench" / "runs", root / "workspace"
cases = {c["case_id"]: c for c in json.loads((root / "bench" / "cases.json").read_text(encoding="utf-8"))}
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
blind = "--blind" in sys.argv  # 과제 B: case.json(설정 변경 목록)을 넣지 않는다

ref_tele = runs / f"baseline_s{seed}.telemetry.json"
meta = root / "bench" / "private" / f"baseline_s{seed}.meta.json"
if not ref_tele.exists():
    raise SystemExit(f"기준 실행 없음: {ref_tele}")
ws_seed = (root / "workspace_blind" if blind else ws) / f"seed{seed}"
if ws_seed.exists():
    shutil.rmtree(ws_seed)
(ws_seed / "reference" / "params").mkdir(parents=True)
shutil.copy(ref_tele, ws_seed / "reference" / "telemetry.json")
run_dir = Path(json.loads(meta.read_text(encoding="utf-8-sig"))["run_dir"]) if meta.exists() else None
if run_dir is None:  # 수동 추출한 기준 실행: 가장 최근 Go2 flat 기준 폴더
    cands = sorted((Path.home() / "IsaacLab/logs/rsl_rl/unitree_go2_flat").glob(f"*baseline_s{seed}"))
    run_dir = cands[-1]
for name in ("env.yaml", "agent.yaml"):
    shutil.copy(run_dir / "params" / name, ws_seed / "reference" / "params" / name)
n = 0
for cid, case in cases.items():
    tele = runs / f"{cid}_s{seed}.telemetry.json"
    if not tele.exists():
        continue
    d = ws_seed / "cases" / cid
    d.mkdir(parents=True)
    shutil.copy(tele, d / "telemetry.json")
    if not blind:
        (d / "case.json").write_text(json.dumps({"case_id": cid, "overrides": case["overrides"]}, ensure_ascii=False), encoding="utf-8")
    n += 1
print(f"{ws_seed.relative_to(root)}: cases={n}")
