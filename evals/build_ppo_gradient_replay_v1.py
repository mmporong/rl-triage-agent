"""공개 PPO 계산 오류 검사 결과를 별도 발표 재생 화면으로 결합한다.

공개 exporter의 검증과 Git 커밋 상태가 모두 확인된 경우에만 수치를 표시한다.
GPU·모델·네트워크 호출과 새 측정은 수행하지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_replay_page as B
import export_ppo_gradient_v5 as E


PUBLIC = Path("evals/results/ppo_gradient_v5_20261017")
OUTPUT = Path("docs/replay/ppo-gradient-data.js")
TOP_FILES = {
    "result.json",
    "dry_run_result.json",
    "v4_comparison.json",
    "cpu_profile.json",
    "provenance.json",
}
RATIOS = {
    2144: 0.9551199859002757,
    2145: 1.1546251037745405,
    2146: 1.1819551288255512,
}
V4_RATIOS = {
    2140: 1.5643724637519756,
    2141: 1.5218924266757203,
    2142: 1.6318819312298947,
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_files() -> set[str]:
    artifacts = (*E.ARTIFACTS, "execution.json")
    files = set(TOP_FILES)
    files.update(f"runs/{spec['id']}/{name}" for spec in E.Q.specs() for name in artifacts)
    files.update(f"dry_run/{E.Q.dry_spec()['id']}/{name}" for name in artifacts)
    if not TOP_FILES.issubset(files):
        raise ValueError("공개 결과의 필수 파일 목록이 다르다")
    if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in files):
        raise ValueError("공개 결과 목록에 허용되지 않은 경로가 있다")
    return files


def _primitive(run: dict, name: str) -> dict:
    inspection = run.get("inspection") or {}
    return inspection.get("primitives", {}).get(name, {})


def validate_payloads(result: dict, comparison: dict, profile: dict) -> dict:
    if (result.get("schema") != "ppo_gradient_check_summary_v5"
            or result.get("status") != "complete" or result.get("outcome") != "confirmed"
            or result.get("completed_runs") != 8 or result.get("planned_runs") != 8
            or result.get("api_requests") != 0 or result.get("behavior_recovery_measured") is not False
            or result.get("checks") != {
                "noninterference_3_pairs": True,
                "cost_3_pairs": True,
                "normal_3_seeds": True,
                "count_unchanged_in_fault": True,
                "fault_gradient_abnormal": True,
                "restored_normal": True,
                "restored_state_exact": True,
            }):
        raise ValueError("v5 집계 계약이 다르다")
    pairs = {row["seed"]: row for row in result.get("pairs", [])}
    if set(pairs) != set(RATIOS):
        raise ValueError("v5 비용 seed가 다르다")
    for seed, expected in RATIOS.items():
        row = pairs[seed]
        if (row.get("update_wall_ratio") != expected or row.get("cost_within_bound") is not True
                or row.get("state_exact") is not True or row.get("losses_exact") is not True):
            raise ValueError("v5 비용 또는 비간섭 결과가 다르다")
    runs = {row["id"]: row for row in result.get("runs", [])}
    if set(runs) != {
        "none_s2144_off", "none_s2144_on", "none_s2145_off", "none_s2145_on",
        "none_s2146_off", "none_s2146_on", "inverse_ratio_s2144_on", "restored_s2144_on",
    }:
        raise ValueError("v5 실행 목록이 다르다")
    fault, restored = runs["inverse_ratio_s2144_on"], runs["restored_s2144_on"]
    for name in ("optimizer_step_count", "minibatch_count"):
        if _primitive(fault, name).get("status") != "normal":
            raise ValueError("계산 오류에서 횟수 검사가 normal이 아니다")
    gradient = _primitive(fault, "ppo_surrogate_gradient")
    if (gradient.get("status") != "abnormal" or gradient.get("first_mismatch_update") != 0
            or gradient.get("batches") != 60):
        raise ValueError("계산 오류 gradient 판정이 다르다")
    if any(_primitive(restored, name).get("status") != "normal" for name in
           ("optimizer_step_count", "minibatch_count", "ppo_surrogate_gradient")):
        raise ValueError("복원 실행의 검사 판정이 다르다")
    if (comparison.get("metric") != "paired_sum_update_wall_on_divided_by_off"
            or comparison.get("max_ratio") != 1.5
            or comparison.get("v4", {}).get("outcome") != "observer_cost_exceeded"
            or comparison.get("v5", {}).get("outcome") != "confirmed"):
        raise ValueError("v4·v5 비교 계약이 다르다")
    v4_pairs = {row["seed"]: row["update_wall_ratio"] for row in comparison["v4"].get("pairs", [])}
    v5_pairs = {row["seed"]: row["update_wall_ratio"] for row in comparison["v5"].get("pairs", [])}
    if v4_pairs != V4_RATIOS or v5_pairs != RATIOS:
        raise ValueError("v4·v5 비용비가 다르다")
    if profile.get("device") != "cpu" or profile.get("updates") != 3:
        raise ValueError("CPU profile 범위가 다르다")
    profile_runs = {row["label"]: row for row in profile.get("runs", [])}
    if set(profile_runs) != {"OFF", "V4_ON", "V5_ON"}:
        raise ValueError("CPU profile 실행 목록이 다르다")
    item_counts = {label: row.get("counts", {}).get("item") for label, row in profile_runs.items()}
    if item_counts != {"OFF": 726, "V4_ON": 2211, "V5_ON": 726}:
        raise ValueError("CPU profile item 호출 수가 다르다")
    return {"pairs": pairs, "fault": fault, "restored": restored, "profile": profile_runs}


def build(root: Path = ROOT) -> dict:
    sources = B.Sources(root)
    relative_files = _public_files()
    source_ids = {}
    for name in sorted(relative_files):
        relative = (PUBLIC / name).as_posix()
        source_ids[name] = sources.file(relative, kind="public_ppo_gradient_v5")["id"]
    verified = None
    if all(sources.ready(source_id) for source_id in source_ids.values()):
        try:
            E.verify_public(root / PUBLIC)
            verified = validate_payloads(
                _read(root / PUBLIC / "result.json"),
                _read(root / PUBLIC / "v4_comparison.json"),
                _read(root / PUBLIC / "cpu_profile.json"),
            )
        except (KeyError, TypeError, ValueError):
            for source_id in source_ids.values():
                sources.rows[source_id].update(status="mismatch", reason="public_replay_failed")

    def cell(source: str, value: Any, digits: int | None = None) -> dict:
        return {"source": source_ids[source], "value": value if verified is not None else None, "digits": digits}

    result_source = "result.json"
    compare_source = "v4_comparison.json"
    profile_source = "cpu_profile.json"
    fault = verified["fault"] if verified else {}
    restored = verified["restored"] if verified else {}
    scenes = [{
        "id": "ppo-gradient-invariant",
        "number": 1,
        "title": "갱신 횟수는 정상인데 PPO 계산이 틀린 경우",
        "lead": "5 epoch × 4 minibatch의 횟수 검사는 통과하지만, clipped-surrogate gradient 불변식은 첫 update부터 어긋난다.",
        "scope": "inverse-ratio 인공 계산 오류로 확인한 개발 측정이다. 외부 버그 재현이나 행동 회복 측정이 아니며 모델 판단은 쓰지 않았다.",
        "blocks": [{
            "type": "table", "wide": True, "title": "같은 seed 2144 · 3 update 검사",
            "columns": ["조건", "optimizer step", "minibatch", "gradient 불변식", "첫 어긋남", "확인 batch"],
            "rows": [
                ["inverse-ratio 오류",
                 cell(result_source, _primitive(fault, "optimizer_step_count").get("status")),
                 cell(result_source, _primitive(fault, "minibatch_count").get("status")),
                 cell(result_source, _primitive(fault, "ppo_surrogate_gradient").get("status")),
                 cell(result_source, _primitive(fault, "ppo_surrogate_gradient").get("first_mismatch_update"), 0),
                 cell(result_source, _primitive(fault, "ppo_surrogate_gradient").get("batches"), 0)],
                ["ratio 계산 복원",
                 cell(result_source, _primitive(restored, "optimizer_step_count").get("status")),
                 cell(result_source, _primitive(restored, "minibatch_count").get("status")),
                 cell(result_source, _primitive(restored, "ppo_surrogate_gradient").get("status")),
                 cell(result_source, "없음"), cell(result_source, 60, 0)],
            ],
            "status_sources": [source_ids["provenance.json"]],
        }],
        "notes": ["검사는 PPO clipped-surrogate 미분식과 실제 gradient를 atol 1e-6, rtol 1e-5로 비교한다.",
                  "복원 실행에서는 횟수 두 항과 gradient 불변식이 모두 normal이었다."],
    }, {
        "id": "ppo-gradient-observer-cost",
        "number": 2,
        "title": "같은 비용 상한에서 계측 읽기를 갱신당 한 번으로 묶음",
        "lead": "v4 결과를 본 뒤 계측 구현만 바꾸고, ON/OFF 비용비 지표와 상한 1.5, 불변식 식·허용치는 유지했다.",
        "scope": "비용비는 세 seed의 개발 측정이며 CPU profile은 호출 수 확인 자료다. GPU 시간 절감이나 다른 조건의 성능을 뜻하지 않는다.",
        "blocks": [{
            "type": "table", "title": "정상 정책 ON/OFF 갱신 시간 비",
            "columns": ["observer", "seed", "ON/OFF 비용비", "상한 통과"],
            "rows": [
                *[["v4", str(seed), cell(compare_source, ratio, 3), cell(compare_source, "아니요")]
                  for seed, ratio in V4_RATIOS.items()],
                *[["v5", str(seed), cell(result_source, ratio, 3), cell(result_source, "예")]
                  for seed, ratio in RATIOS.items()],
            ],
        }, {
            "type": "table", "title": "Isaac Python CPU profile · 3 update",
            "columns": ["조건", "Tensor .item() 호출", "자료 범위"],
            "rows": [
                ["계측 OFF", cell(profile_source, 726, 0), "CPU 호출 수"],
                ["v4 ON", cell(profile_source, 2211, 0), "CPU 호출 수"],
                ["v5 ON", cell(profile_source, 726, 0), "CPU 호출 수"],
            ],
        }],
        "notes": ["세 v5 seed에서 정책·optimizer·RNG 상태와 손실은 대응하는 OFF 실행과 일치했다.",
                  "CPU profile 수치는 GPU 비용비의 대체 측정이 아니다."],
    }]
    data = {
        "schema": B.SCHEMA,
        "scope": "공개 결과에 결합한 PPO 계산 오류 검사 재생 화면",
        "status_labels": {"ready": "확인", "waiting": "대기", "mismatch": "불일치"},
        "sources": sources.rows,
        "scenes": scenes,
    }
    B.check_public(data)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = build()
    content = B.render(data)
    output = ROOT / OUTPUT
    if args.check:
        current = output.is_file() and output.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if current else "stale", "scenes": len(data["scenes"]),
                          "source_status": dict(Counter(row["status"] for row in data["sources"].values()))},
                         ensure_ascii=False))
        return 0 if current else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "scenes": len(data["scenes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
