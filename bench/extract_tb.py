"""Isaac Lab RSL-RL 실행 폴더의 TensorBoard 이벤트를 에이전트용 텔레메트리 JSON으로 변환한다.

Isaac Sim 번들 파이썬에서 실행한다: isaaclab.bat -p bench/extract_tb.py <run_dir> <out_json>
"""
import glob
import json
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(values):
    n = len(values)
    if n == 0:
        return None
    q = max(1, n // 5)
    first, last = values[:q], values[-q:]
    return {
        "n": n,
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean_first_20pct": sum(first) / len(first),
        "mean_last_20pct": sum(last) / len(last),
    }


def main(run_dir, out_path):
    events = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not events:
        raise SystemExit(f"TensorBoard 이벤트 없음: {run_dir}")
    ea = EventAccumulator(events[0], size_guidance={"scalars": 0})
    ea.Reload()
    series, summary = {}, {}
    for tag in ea.Tags()["scalars"]:
        vals = [e.value for e in ea.Scalars(tag)]
        # 시간축 중복 태그(/time)는 같은 값이라 제외한다.
        if tag.endswith("/time"):
            continue
        series[tag] = [round(v, 6) for v in vals]
        summary[tag] = summarize(vals)
    out = {"run_dir_name": os.path.basename(os.path.normpath(run_dir)), "summary": summary, "series": series}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {out_path} tags={len(series)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
