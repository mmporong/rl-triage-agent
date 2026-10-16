"""공개 U1 확장 교정·optimizer 관측 결과를 두 재생 장면으로 결합한다.

기존 화면과 결과를 바꾸지 않는다. 입력 목록과 SHA를 등록 파일에서 읽고,
커밋 상태·파일 지문·장면 수치의 교차 검증이 맞을 때만 숫자를 표시한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import build_replay_page as B

PROTOCOL = Path("bench/protocols/replay_additional_checks_v1.json")
OUTPUT = Path("docs/replay/additional-checks-data.js")
U1 = "evals/results/u1_scope_extension_s2051_20261015/"
OPT = "evals/results/optimizer_observability_gpu_20261015/"
FAMILY = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "lin_vel_z_l2", "ang_vel_xy_l2"]
FIXED = {
    "schema": "additional_checks_replay_contract_v1",
    "scenes": ["u1-expanded-scope", "optimizer-update-observation"],
    "validation": {"require_committed_unmodified_inputs": True, "hide_numbers_when_invalid": True,
                   "recompute_optimizer_counts": True, "cross_check_u1_scope_and_evaluations": True,
                   "viewport": [1920, 1080], "overflow_px_max": 0},
    "scope": "public_result_replay_no_new_measurement", "model_calls": 0, "gpu_calls": 0,
}


def validate_protocol(protocol):
    if {key: value for key, value in protocol.items() if key != "inputs_sha256"} != FIXED:
        raise ValueError("replay fixed contract differs")
    expected = {U1 + name for name in ("summary.json", "scope_evidence.json", "source_failure_evidence.json",
                                      "correction_readback.json", "evaluations/none_family_control.json",
                                      "evaluations/u1_family_restart.json")}
    expected.add(OPT + "summary.json")
    ids = ["none_s2042", "r1_s2042"] + [f"{arm}_s{seed}" for seed in (2026, 2027, 2028)
                                        for arm in ("none", "h01", "h02", "h03", "h04", "h05", "h06")]
    expected.update(OPT + "runs/" + name + ".json" for name in ids)
    hashes = protocol.get("inputs_sha256", {})
    if (set(hashes) != expected or any(not isinstance(value, str) or len(value) != 64
                                     or any(c not in "0123456789abcdef" for c in value)
                                     for value in hashes.values())):
        raise ValueError("replay input inventory differs")


def validate_u1(rows: dict) -> dict:
    summary, scope = rows[U1 + "summary.json"], rows[U1 + "scope_evidence.json"]
    before = rows[U1 + "source_failure_evidence.json"]
    after = rows[U1 + "evaluations/u1_family_restart.json"]
    control = rows[U1 + "evaluations/none_family_control.json"]
    readback = rows[U1 + "correction_readback.json"]["arms"]
    if (summary["status"] != "complete" or summary["outcome"] != "recovered"
            or summary["source_training_seed"] != 2051
            or summary["corrected_u1_restart_training_seed"] != 3051
            or scope["scope_verdict"] != "scope_insufficient"
            or scope["source_registered_terms"] != FAMILY
            or scope["measured_probe_terms"] != FAMILY[:2]
            or scope["api_calls"] != 0 or scope["model_loop"] != "inactive"
            or before["gate"]["outcome"] != "failure_present"
            or not before["gate"]["none_walks"]
            or before["gate"]["u1_label"] != "unhealthy"):
        raise ValueError("U1 source/scope/summary contract differs")
    for arm, result, seed in (("none_family_control", control, 2051), ("u1_family_restart", after, 3051)):
        if (result["arm"] != arm or result["training"]["seed"] != seed or result["label"] != "healthy"
                or not all(result["verdict"]["checks"].values())
                or len(result["verdict"]["checks"]) != 4
                or result["fixed_evaluation"]["condition_count"] != 26
                or len(result["fixed_evaluation"]["conditions"]) != 26
                or readback[arm]["clean_function_count"] != 4
                or not readback[arm]["environment_binding_ok"]):
            raise ValueError("U1 evaluation/readback contract differs")
    if readback["u1_family_restart"]["restored_functions"] != FAMILY:
        raise ValueError("restored family differs")
    if readback["none_family_control"]["changed_by_fault_count"] != 0:
        raise ValueError("NONE correction control differs")
    return {"before": before["arms"]["U1"]["metrics"],
            "after": after["fixed_evaluation"]["metrics"],
            "none": control["fixed_evaluation"]["metrics"]}


def validate_optimizer(rows: dict) -> dict:
    summary = rows[OPT + "summary.json"]
    expected_ids = ["none_s2042", "r1_s2042"] + [f"{arm}_s{seed}" for seed in (2026, 2027, 2028)
                                                   for arm in ("none", "h01", "h02", "h03", "h04", "h05", "h06")]
    runs = [rows[OPT + "runs/" + run_id + ".json"] for run_id in expected_ids]
    classifications = Counter()
    for run_id, run in zip(expected_ids, runs):
        expected = "abnormal" if run_id == "r1_s2042" else "normal"
        count = 4 if expected == "abnormal" else 20
        if run["run_id"] != run_id or len(run["updates"]) != 3 or run["status"] != expected:
            raise ValueError("optimizer run identity/status differs")
        for index, update in enumerate(run["updates"]):
            if (update["update_index"] != index or update["declared_learning_epochs"] != 5
                    or update["declared_mini_batches"] != 4
                    or update["expected_optimizer_steps"] != 20 or update["expected_minibatches"] != 20
                    or update["observed_optimizer_steps"] != count or update["observed_minibatches"] != count):
                raise ValueError("optimizer update count differs")
        classifications[run["status"]] += 1
    if (summary["run_count"] != 23 or summary["status"] != "complete"
            or summary["classification_counts"] != {"abnormal": 1, "normal": 22, "unknown": 0}
            or classifications != {"abnormal": 1, "normal": 22}
            or set(summary["controls"]["run_ids"]) != set(expected_ids) - {"r1_s2042"}):
        raise ValueError("optimizer aggregate differs")
    return {"r1_steps": runs[1]["updates"][0]["observed_optimizer_steps"],
            "none_steps": runs[0]["updates"][0]["observed_optimizer_steps"], "specificity": len(runs) - 2}


def build(root: Path = ROOT) -> dict:
    protocol = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    validate_protocol(protocol)
    sources = B.Sources(root)
    loaded = {}
    for relative, expected in protocol["inputs_sha256"].items():
        if not relative.startswith((U1, OPT)) or ".." in Path(relative).parts or "\\" in relative:
            raise ValueError("input path is outside the registered public bundles")
        row = sources.file(relative, expected_sha256=expected)
        if row["status"] == "ready":
            loaded[relative] = json.loads((root / relative).read_text(encoding="utf-8"))

    def verified(prefix, validator):
        ids = [name for name in protocol["inputs_sha256"] if name.startswith(prefix)]
        if any(name not in loaded for name in ids):
            return None
        try:
            return validator(loaded)
        except (KeyError, TypeError, ValueError):
            for name in ids:
                sources.rows[name].update(status="mismatch", reason="cross_validation_failed")
            return None

    u1 = verified(U1, validate_u1)
    optimizer = verified(OPT, validate_optimizer)

    def cell(source, value, digits=None):
        return {"source": source, "value": value, "digits": digits}

    scope_source = U1 + "scope_evidence.json"
    summary_source = U1 + "summary.json"
    u1_rows = []
    for name, key, source in (("결함 정책 · seed 2051", "before", U1 + "source_failure_evidence.json"),
                              ("네 항 복원 후 재학습 · seed 3051", "after", U1 + "evaluations/u1_family_restart.json"),
                              ("정상 정책 대조 · seed 2051", "none", U1 + "evaluations/none_family_control.json")):
        metrics = u1[key] if u1 else {}
        u1_rows.append([name, cell(source, metrics.get("lin_vel_rmse_mps"), 3),
                        cell(source, metrics.get("yaw_rate_rmse_radps"), 3),
                        cell(source, metrics.get("mean_survival_s"), 1)])
    scenes = [{
        "id": "u1-expanded-scope", "number": 1, "title": "교정 범위를 넓힌 뒤 고정 평가 통과",
        "lead": "범위 부족 확인 → 등록된 보상 함수 네 항 복원 → 처음부터 300회 재학습",
        "scope": "기존 U1 결과를 본 뒤 만든 알려진 함수군 교정의 개발 사례다. API 0회, 모델 판단 고리는 비활성이다.",
        "blocks": [
            {"type": "table", "title": "검사와 교정의 범위", "columns": ["단계", "기록"], "rows": [
                ["기존 교정 범위", cell(scope_source, "scope_insufficient" if u1 else None)],
                ["probe가 측정한 항", cell(scope_source, 2 if u1 else None)],
                ["소스·실행 확인으로 결합한 복원 항", cell(U1 + "correction_readback.json", 4 if u1 else None)],
                ["고정 평가 결과", cell(summary_source, "recovered / NONE healthy" if u1 else None)]]},
            {"type": "table", "title": "동일한 26조건 평가", "columns": ["정책", "이동 오차(m/s)", "회전 오차(rad/s)", "생존(s)"], "rows": u1_rows,
             "status_sources": [name for name in protocol["inputs_sha256"] if name.startswith(U1)]}],
        "notes": ["probe 두 항의 측정과 소스에서 확인한 네 항의 범위를 구분한다.",
                  "동일 seed에서 좁은 교정과 넓은 교정을 비교한 실험은 아니다. 이 사례로 일반화·GPU 절감을 주장하지 않는다."]}, {
        "id": "optimizer-update-observation", "number": 2, "title": "PPO 갱신 횟수를 세어 구현 결함을 포착",
        "lead": "설정은 5 epoch × 4 minibatch = 20회. R1에서는 실제 갱신이 4회였다.",
        "scope": "R1 결과를 본 뒤 만든 검사의 개발 측정이다. 새 검사의 normal은 PPO 갱신 횟수 조건에서 음성이라는 뜻이다.",
        "blocks": [
            {"type": "table", "title": "같은 seed 2042에서 3 update 관측", "columns": ["조건", "설정 횟수/update", "실제 횟수/update", "검사 결과"], "rows": [
                ["정상 NONE", cell(OPT + "runs/none_s2042.json", 20 if optimizer else None), cell(OPT + "runs/none_s2042.json", optimizer["none_steps"] if optimizer else None), cell(OPT + "runs/none_s2042.json", "normal" if optimizer else None)],
                ["epoch 무시 R1", cell(OPT + "runs/r1_s2042.json", 20 if optimizer else None), cell(OPT + "runs/r1_s2042.json", optimizer["r1_steps"] if optimizer else None), cell(OPT + "runs/r1_s2042.json", "abnormal" if optimizer else None)]]},
            {"type": "table", "title": "추가 특이성 확인 · seed 2026~2028", "columns": ["검사 범위", "기록"], "rows": [
                ["NONE + h01~h06, 각 3 seed", cell(OPT + "summary.json", optimizer["specificity"] if optimizer else None)],
                ["epoch/update 검사에서 normal", cell(OPT + "summary.json", "21 / 21" if optimizer else None)],
                ["전체 실행", cell(OPT + "summary.json", "23개: R1 1 abnormal, 대조 22 normal" if optimizer else None)]]}],
        "notes": ["h01~h06의 다른 결함까지 정상으로 판정한 것은 아니다. KL·clip 비율은 측정되지 않았다.",
                  "R1 한 seed의 개발 측정이며 진단 정확도·일반화의 근거가 아니다."]}]
    data = {"schema": B.SCHEMA, "scope": "공개 결과에 결합한 검사·교정 재생 화면",
            "status_labels": {"ready": "확인", "waiting": "대기", "mismatch": "불일치"},
            "sources": sources.rows, "scenes": scenes}
    B.check_public(data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = build()
    content = B.render(data)
    output = ROOT / OUTPUT
    if args.check:
        current = output.is_file() and output.read_text(encoding="utf-8") == content
        print(json.dumps({"status": "current" if current else "stale", "scenes": len(data["scenes"])}))
        return 0 if current else 1
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError("기존 재생 자료를 덮어쓰지 않는다")
    else:
        output.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "written", "sources": dict(Counter(row["status"] for row in data["sources"].values()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
