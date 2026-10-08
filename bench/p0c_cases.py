"""P0-C holdout 케이스와 숨은 결함의 비밀 대응을 만든다(docs/P0-C-HOLDOUT.md).

사용: python bench/p0c_cases.py            # 비밀 seed를 새로 뽑아 bench/private/answer_key_p0c.json을 만든다
      python bench/p0c_cases.py --print-plan # 학습할 실행 목록(케이스·seed)만 출력, 대응은 출력하지 않는다

케이스 ID h01~h06은 결함 F1~F6에 하나씩 대응하고 모든 holdout seed에 같은 대응을 쓴다(v1과 같은 방식).
대응·결함 범주는 정답이라 bench/private에만 쓴다. 이미 있으면 덮어쓰지 않는다.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from hidden_faults import FAULTS  # noqa: E402

HOLDOUT_SEEDS = (2026, 2027, 2028)
KEY_PATH = ROOT / "bench" / "private" / "answer_key_p0c.json"


def make_key(secret_seed: int) -> dict:
    faults = sorted(f for f in FAULTS if f != "NONE")
    random.Random(secret_seed).shuffle(faults)
    cases = {f"h{i:02d}": {"fault": f, "category": FAULTS[f]["category"], "note": FAULTS[f]["note"]}
             for i, f in enumerate(faults, start=1)}
    return {"bench": "p0c", "secret_seed": secret_seed, "holdout_seeds": list(HOLDOUT_SEEDS), "cases": cases}


def plan() -> list[dict]:
    runs = [{"case_id": "baseline_p0c", "seed": s} for s in HOLDOUT_SEEDS]
    runs += [{"case_id": f"h{i:02d}", "seed": s} for s in HOLDOUT_SEEDS for i in range(1, len(FAULTS))]
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-plan", action="store_true")
    args = ap.parse_args()
    if args.print_plan:
        print(json.dumps(plan(), indent=1))
        return 0
    if KEY_PATH.exists():
        raise SystemExit(f"{KEY_PATH.relative_to(ROOT).as_posix()}가 이미 있다. 덮어쓰지 않는다")
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = make_key(random.SystemRandom().randrange(1 << 30))
    KEY_PATH.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cases={len(key['cases'])} seeds={key['holdout_seeds']} (answer key: bench/private/answer_key_p0c.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
