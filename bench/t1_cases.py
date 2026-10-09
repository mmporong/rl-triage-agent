"""T1 외부 원인 결함 사례와 비밀 대응을 만든다(docs/T1-EXTERNAL-FAULTS.md).

사용: python bench/t1_cases.py --faults X1 X2 X3   # smoke·보정을 통과한 결함만 넣어 bench/private/answer_key_t1.json 생성
      python bench/t1_cases.py --print-plan        # 학습할 holdout 실행 목록(실행 이름·seed)만 출력

- 사례 ID e01~는 결함에 하나씩 대응하고 모든 holdout seed에 같은 대응을 쓴다(P0-C와 같은 방식).
- 짝 설계 결함(X1·X3)의 기준 실행은 ref_<사례>_s<seed>(X?_REF로 학습), X2의 기준은 순정 baseline_p0c_s<seed>
  (P0-C holdout에서 같은 seed·예산으로 학습한 실행)다.
- 대응은 정답이라 bench/private에만 쓰고 평가가 끝난 뒤 공개한다. 이미 있으면 덮어쓰지 않는다.
"""
import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_SEEDS = (2026, 2027, 2028)
DEV_SEED = 7
CANDIDATES = ("X1", "X2", "X3")
KEY_PATH = ROOT / "bench" / "private" / "answer_key_t1.json"


def _faults():
    spec = importlib.util.spec_from_file_location("external_faults_meta", ROOT / "bench" / "external_faults.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAULTS


def make_key(secret_seed: int, faults: list[str], stock_reference: tuple[str, ...] = ()) -> dict:
    """stock_reference: smoke에서 짝 기준이 깨진(X1 Bug 2) 결함. 순정 baseline_p0c를 기준으로 쓴다."""
    meta = _faults()
    bad = [f for f in faults if f not in CANDIDATES]
    if bad or not faults or len(set(faults)) != len(faults):
        raise SystemExit(f"후보는 {CANDIDATES} 중에서 겹치지 않게 고른다: {faults}")
    order = sorted(faults)
    random.Random(secret_seed).shuffle(order)
    cases = {}
    for i, f in enumerate(order, start=1):
        cid = f"e{i:02d}"
        paired = meta[f]["pair"] is not None and f not in stock_reference
        cases[cid] = {"fault": f, "category": meta[f]["category"], "note": meta[f]["note"],
                      "source": meta[f]["source"], "reference_fault": f"{f}_REF" if paired else "NONE",
                      "reference": f"ref_{cid}" if paired else "baseline_p0c"}
    return {"bench": "t1", "secret_seed": secret_seed, "holdout_seeds": list(HOLDOUT_SEEDS), "cases": cases}


def plan(key: dict) -> list[dict]:
    """학습할 실행. baseline_p0c는 P0-C에서 이미 학습했으므로 넣지 않는다. 결함 이름은 담지 않는다."""
    runs = []
    for s in HOLDOUT_SEEDS:
        for cid, c in sorted(key["cases"].items()):
            if c["reference"] != "baseline_p0c":
                runs.append({"case_id": c["reference"], "seed": s})
            runs.append({"case_id": cid, "seed": s})
    return runs


def fault_of_run(case_id: str, key: dict) -> str:
    """holdout 실행 이름의 사례 부분 → TRIAGE_FAULT."""
    if case_id == "baseline_p0c":
        return "NONE"
    for cid, c in key["cases"].items():
        if case_id == cid:
            return c["fault"]
        if case_id == c["reference"]:
            return c["reference_fault"]
    raise SystemExit(f"{case_id}: T1 정답표에 없는 사례")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", nargs="+", help="smoke·보정을 통과한 결함(X1 X2 X3 중)")
    ap.add_argument("--stock-reference", nargs="*", default=[], help="짝 기준 대신 순정 기준을 쓸 결함(smoke 판정)")
    ap.add_argument("--print-plan", action="store_true")
    args = ap.parse_args()
    if args.print_plan:
        print(json.dumps(plan(json.loads(KEY_PATH.read_text(encoding="utf-8"))), indent=1))
        return 0
    if not args.faults:
        raise SystemExit("--faults가 필요하다")
    if KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH.relative_to(ROOT).as_posix()}가 이미 있다. 덮어쓰지 않는다")
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = make_key(random.SystemRandom().randrange(1 << 30), args.faults, tuple(args.stock_reference))
    KEY_PATH.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cases={len(key['cases'])} seeds={key['holdout_seeds']} (answer key: bench/private/answer_key_t1.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
