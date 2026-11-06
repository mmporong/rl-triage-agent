"""제한 패치 후보의 실제 CPU 검사 묶음을 실행하고 원문을 봉인한다."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import ppo_patch_feedback_v1 as FEEDBACK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if type(jobs) is not dict or set(jobs) != {"sources"} or not jobs["sources"]:
        raise ValueError("실제 CPU 입력이 비었다")
    args.out.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, source in jobs["sources"].items():
        if not isinstance(name, str) or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("CPU job 이름이 잘못됐다")
        value = FEEDBACK.validate_on_cpu(source)
        with (args.out / f"{name}.json").open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        records[name] = {"source_sha256": value["source_sha256"], "all_passed": value["all_passed"]}
        print(json.dumps({"job": name, "all_passed": value["all_passed"]}), flush=True)
    with (args.out / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"status": "complete", "records": records, "SDK_app": 0, "GPU": 0}, stream, indent=2)


if __name__ == "__main__":
    main()
