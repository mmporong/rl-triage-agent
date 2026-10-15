"""공개 학습 prefix로 최종 행동 건강성을 보수적으로 개발 측정한다.

이 도구는 텔레메트리 ``series[:N]``만 특징으로 사용한다. ``summary``와 고정
평가 수치는 라벨 연결에만 쓰며 특징 계산 경로에 전달하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PROTOCOL = ROOT / "bench/protocols/early_recovery_decider_v1.json"
DEFAULT_PROTOCOL = ROOT / "bench/protocols/early_recovery_decider_v2.json"
DEFAULT_OUTPUT = ROOT / "bench/private/codex_support_20261015/task3"
SEED_RE = re.compile(r"_s(\d+)")
MASK_RE = re.compile(r"^(d3v2_[a-z0-9]+_s\d+)_m([0-9a-f]{2})$")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않는다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _input_inventory(protocol: dict[str, Any]) -> dict[str, Any]:
    if protocol["schema"] == "early_recovery_decider_protocol_v2":
        return protocol["inventory"]
    if protocol["schema"] != "early_recovery_decider_protocol_v1":
        raise ValueError("지원하지 않는 조기 판정 프로토콜")
    # v1 파일을 보존하면서 당시 입력을 명시한 v2 목록으로 재생한다.
    frozen = read_json(DEFAULT_PROTOCOL)
    if sha256(LEGACY_PROTOCOL) != frozen["revision"]["source_protocol"]["sha256_lf"]:
        raise ValueError("v1 원본 프로토콜 SHA 불일치")
    if protocol["inventory"] != read_json(LEGACY_PROTOCOL)["inventory"]:
        raise ValueError("v1 입력 정의 변경은 새 프로토콜이 필요하다")
    return frozen["inventory"]


def _bound_files(entries: list[dict[str, Any]]) -> list[Path]:
    paths = []
    seen = set()
    for entry in entries:
        relative = entry["path"]
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts):
            raise ValueError("등록 입력은 저장소 내부 상대경로여야 한다")
        path = ROOT / relative
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT.resolve()) or path.is_symlink() or not path.is_file():
            raise ValueError(f"등록 입력 누락 또는 허용되지 않는 경로: {relative}")
        if resolved in seen:
            raise ValueError(f"등록 입력 중복: {relative}")
        seen.add(resolved)
        expected = entry["sha256_lf"]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(path) != expected:
            raise ValueError(f"등록 입력 SHA 불일치: {relative}")
        paths.append(path)
    return paths


def inventory(protocol: dict[str, Any]) -> list[Path]:
    registered = _input_inventory(protocol)
    paths = _bound_files(registered["telemetry_files"])
    expected = registered["expected_file_count"]
    if len(paths) != expected:
        raise ValueError(f"텔레메트리 파일 수 {len(paths)} != 사전등록 {expected}")
    for path, entry in zip(paths, registered["telemetry_files"]):
        if (path.parent.relative_to(ROOT).as_posix() not in registered["telemetry_roots"]
                or path.name != entry["run"] + ".telemetry.json"):
            raise ValueError("등록 telemetry 경로·실행 이름 불일치")
    return paths


def report_index(protocol: dict[str, Any]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    entries = _input_inventory(protocol)["fixed_reports"]
    for path, entry in zip(_bound_files(entries), entries):
        record = read_json(path)
        name = entry["run"]
        if not isinstance(record, dict) or record.get("name") != name or name in found:
            raise ValueError(f"등록 고정 평가 실행 이름 불일치 또는 중복: {entry['path']}")
        found[name].append(path)
    return found


def _pick_report(name: str, reports: dict[str, list[Path]]) -> tuple[Path, dict[str, Any]] | None:
    candidates = reports.get(name, [])
    if not candidates:
        return None
    material = [(path, read_json(path)) for path in candidates]
    hashes = {row.get("checkpoint", {}).get("sha256") for _, row in material}
    hashes.discard(None)
    if len(hashes) > 1:
        raise ValueError(f"{name}: 서로 다른 checkpoint를 가리키는 고정 평가 보고서")
    contracts = {tuple(row.get(key) for key in CONTRACT_FIELDS) for _, row in material}
    if len(contracts) != 1:
        raise ValueError(f"{name}: 서로 다른 평가 계약")
    return material[0]


CONTRACT_FIELDS = ("protocol", "protocol_sha256_lf", "task", "eval_seed", "num_envs",
                   "horizon_steps", "step_dt", "action_scale")


def validate_label_binding(name: str, telemetry: dict, report: dict, reference: dict) -> None:
    if not telemetry.get("run_dir_name", "").endswith("_" + name) or report.get("name") != name:
        raise ValueError(f"{name}: telemetry 실행 이름 결합 실패")
    if report.get("task") != _task(name):
        raise ValueError(f"{name}: task 불일치")
    if any(report.get(k) is None or report[k] != reference.get(k) for k in CONTRACT_FIELDS):
        raise ValueError(f"{name}: 기준 평가와 계약 불일치")
    anymal = "Anymal" in report["task"]
    if (report["protocol"] != ("fixed_eval_v1_anymal_c" if anymal else "fixed_eval_v1") or report["num_envs"] != 1040
            or report["horizon_steps"] != 1000 or report["step_dt"] != 0.02
            or report["action_scale"] != (0.5 if anymal else 0.25) or len(report.get("by_condition", [])) != 26):
        raise ValueError(f"{name}: 등록 고정 평가 형식 불일치")
    for row in (report, reference):
        values = row.get("metrics", {})
        for key in ("fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps"):
            value = values.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}: 평가 수치 누락/비정상")
    ref = reference["metrics"]
    standstill = reference.get("standstill", {}).get("lin_vel_rmse_mps", 0)
    if (standstill <= 0 or ref["fall_rate"] > 0.05 or ref["lin_vel_rmse_mps"] / standstill >= 0.5
            or min(ref["mean_survival_s"], ref["lin_vel_rmse_mps"], ref["yaw_rate_rmse_radps"]) <= 0):
        raise ValueError(f"{name}: 정상 기준 정책의 걷기/낙상 조건 실패")


def _seed(name: str) -> int:
    match = SEED_RE.search(name)
    if not match:
        raise ValueError(f"seed를 읽을 수 없다: {name}")
    return int(match.group(1))


def _task(name: str) -> str:
    return "Isaac-Velocity-Flat-Anymal-C-v0" if name.startswith("anymal_c_") else "Isaac-Velocity-Flat-Unitree-Go2-v0"


def _reference_name(name: str) -> str | None:
    seed = _seed(name)
    if name.startswith("baseline_p0c_"):
        return name
    if re.match(r"^(h\d\d|u1)_s", name):
        return f"baseline_p0c_s{seed}"
    if name.startswith("d3_harmless_"):
        return f"baseline_p0c_s{seed}"
    mask = MASK_RE.match(name)
    if mask:
        return f"{mask.group(1)}_m00"
    if name == f"upstream_r1_none_s{seed}":
        return name
    if name == f"upstream_r1_s{seed}":
        return f"upstream_r1_none_s{seed}"
    if name.startswith("upstream_u1_"):
        return f"baseline_p0c_s{seed}"
    if name in {f"anymal_c_gate_none_s{seed}", f"anymal_c_band_none_s{seed}", f"anymal_c_none_s{seed}"}:
        return name
    if name == f"anymal_c_h02_s{seed}":
        return f"anymal_c_none_s{seed}"
    return None


def _behavior_label(name: str, report: dict[str, Any], reference: dict[str, Any], bands: dict[str, float]) -> str:
    if name == reference.get("name") or _reference_name(name) == name:
        return "healthy"
    metrics = report["metrics"]
    ref = reference["metrics"]
    survival_ratio = metrics["mean_survival_s"] / ref["mean_survival_s"]
    fall_frac = metrics["fall_rate"]
    err_xy_ratio = metrics["lin_vel_rmse_mps"] / ref["lin_vel_rmse_mps"]
    err_yaw_ratio = metrics["yaw_rate_rmse_radps"] / ref["yaw_rate_rmse_radps"]
    healthy = (
        survival_ratio >= bands["survival_ratio_min"]
        and fall_frac <= bands["fall_frac_max"]
        and err_xy_ratio <= bands["err_xy_ratio_max"]
        and err_yaw_ratio <= bands["err_yaw_ratio_max"]
    )
    return "healthy" if healthy else "unhealthy"


def build_catalog(protocol: dict[str, Any]) -> dict[str, Any]:
    reports = report_index(protocol)
    bands = _input_inventory(protocol)["band_summaries"]
    p0c_path, anymal_path = _bound_files([bands["go2"], bands["anymal"]])
    p0c = read_json(p0c_path)["bands"]
    anymal = read_json(anymal_path)["analysis"]["bands"]
    telemetry_paths = inventory(protocol)
    included = []
    excluded = []
    for telemetry_path in telemetry_paths:
        name = telemetry_path.name.removesuffix(".telemetry.json")
        picked = _pick_report(name, reports)
        ref_name = _reference_name(name)
        ref_picked = _pick_report(ref_name, reports) if ref_name else None
        if not picked or not ref_picked:
            excluded.append({"run": name, "reason": "missing_fixed_eval_join"})
            continue
        report_path, report = picked
        ref_path, reference = ref_picked
        telemetry = read_json(telemetry_path)
        validate_label_binding(name, telemetry, report, reference)
        checkpoint = report.get("checkpoint", {}).get("sha256")
        if not isinstance(checkpoint, str) or len(checkpoint) != 64:
            excluded.append({"run": name, "reason": "missing_checkpoint_sha256"})
            continue
        task = _task(name)
        label = _behavior_label(name, report, reference, anymal if "Anymal" in task else p0c)
        included.append({
            "run": name,
            "seed": _seed(name),
            "task": task,
            "telemetry_path": telemetry_path.relative_to(ROOT).as_posix(),
            "telemetry_sha256_lf": sha256(telemetry_path),
            "fixed_report_path": report_path.relative_to(ROOT).as_posix(),
            "fixed_report_sha256_lf": sha256(report_path),
            "reference_report_path": ref_path.relative_to(ROOT).as_posix(),
            "reference_report_sha256_lf": sha256(ref_path),
            "run_dir_name": telemetry["run_dir_name"],
            "fixed_contract": {k: report[k] for k in CONTRACT_FIELDS},
            "binding_scope": "공개 run 이름·run_dir suffix·평가 계약 결합; telemetry 자체에 checkpoint SHA 없음",
            "checkpoint_sha256": checkpoint,
            "behavior_label": label,
            "recovery_label": None,
        })

    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in included:
        by_checkpoint[row["checkpoint_sha256"]].append(row)
    deduped = []
    duplicates = []
    for checkpoint, rows in sorted(by_checkpoint.items()):
        labels = {row["behavior_label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"동일 checkpoint 라벨 충돌: {checkpoint}")
        rows.sort(key=lambda row: row["run"])
        deduped.append(rows[0])
        for row in rows[1:]:
            duplicates.append({"run": row["run"], "canonical_run": rows[0]["run"], "checkpoint_sha256": checkpoint})

    for row in deduped:
        if row["task"].endswith("Anymal-C-v0"):
            row["split"] = "transfer_audit"
        elif row["seed"] in protocol["leakage_controls"]["development_lineages"]["Isaac-Velocity-Flat-Unitree-Go2-v0"]:
            row["split"] = "development"
        else:
            row["split"] = "evaluation"
    return {
        "inventory_count": len(telemetry_paths),
        "joined_before_dedup": len(included),
        "included_after_dedup": len(deduped),
        "excluded_missing_label": excluded,
        "duplicates": duplicates,
        "runs": sorted(deduped, key=lambda row: row["telemetry_path"]),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def extract_features(telemetry: dict[str, Any], protocol: dict[str, Any], n: int) -> list[float] | None:
    """summary를 읽지 않고 series prefix만 특징 벡터로 바꾼다."""
    series = telemetry.get("series")
    if not isinstance(series, dict):
        return None
    width = protocol["features"]["tail_window"]
    vector = []
    for key in protocol["features"]["series"]:
        values = series.get(key)
        if not isinstance(values, list) or len(values) < n:
            return None
        prefix = values[:n]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in prefix):
            return None
        head = [float(value) for value in prefix[:width]]
        tail = [float(value) for value in prefix[-width:]]
        vector.extend((_mean(tail), _mean(tail) - _mean(head)))
    return vector


def _scale(columns: list[list[float]]) -> tuple[list[float], list[float]]:
    centers, scales = [], []
    for values in columns:
        center = statistics.median(values)
        mad = statistics.median(abs(value - center) for value in values) * 1.4826
        if mad == 0:
            mad = statistics.pstdev(values)
        centers.append(center)
        scales.append(mad or 1.0)
    return centers, scales


def fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if {row["behavior_label"] for row in rows} != {"healthy", "unhealthy"}:
        raise ValueError("두 행동 라벨이 모두 있어야 한다")
    columns = [list(values) for values in zip(*(row["features"] for row in rows))]
    center, scale = _scale(columns)
    standardized = [[(value - center[i]) / scale[i] for i, value in enumerate(row["features"])] for row in rows]
    centroids = {}
    for label in ("healthy", "unhealthy"):
        selected = [vector for vector, row in zip(standardized, rows) if row["behavior_label"] == label]
        centroids[label] = [_mean(list(values)) for values in zip(*selected)]
    return {"center": center, "scale": scale, "centroids": centroids}


def score(model: dict[str, Any], features: list[float]) -> float:
    vector = [(value - model["center"][i]) / model["scale"][i] for i, value in enumerate(features)]
    distance = {label: sum((value - centroid[i]) ** 2 for i, value in enumerate(vector))
                for label, centroid in model["centroids"].items()}
    return distance["unhealthy"] - distance["healthy"]


def decide(value: float, margin: float) -> str:
    if abs(value) < margin:
        return "abstain"
    return "healthy" if value > 0 else "unhealthy"


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((row["behavior_label"], "abstain" if row["prediction"].startswith("abstain") else row["prediction"]) for row in rows)
    healthy_total = sum(row["behavior_label"] == "healthy" for row in rows)
    unhealthy_total = sum(row["behavior_label"] == "unhealthy" for row in rows)
    return {
        "n": len(rows),
        "confusion": {f"{actual}_as_{predicted}": counts[(actual, predicted)]
                      for actual in ("healthy", "unhealthy") for predicted in ("healthy", "unhealthy", "abstain")},
        "false_healthy": counts[("unhealthy", "healthy")],
        "healthy_coverage": counts[("healthy", "healthy")] / healthy_total if healthy_total else None,
        "unhealthy_detection": counts[("unhealthy", "unhealthy")] / unhealthy_total if unhealthy_total else None,
        "abstention_rate": sum(row["prediction"].startswith("abstain") for row in rows) / len(rows) if rows else None,
        "counterfactual_early_behavior_iterations": sum(max(0, 300 - row["n"]) for row in rows if row["prediction"] in ("healthy", "unhealthy")),
    }


def _load_feature_rows(catalog: dict[str, Any], protocol: dict[str, Any], n: int) -> list[dict[str, Any]]:
    rows = []
    for item in catalog["runs"]:
        values = extract_features(read_json(ROOT / item["telemetry_path"]), protocol, n)
        rows.append({**item, "n": n, "features": values})
    return rows


def select(protocol: dict[str, Any], catalog: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    cached = {}
    for n in protocol["features"]["candidate_prefix_iterations"]:
        all_rows = _load_feature_rows(catalog, protocol, n)
        dev = [row for row in all_rows if row["split"] == "development" and row["features"] is not None]
        model = fit(dev)
        for margin in protocol["classifier"]["candidate_margins"]:
            predicted = [{**row, "score": score(model, row["features"]),
                          "prediction": decide(score(model, row["features"]), margin)} for row in dev]
            result = metrics(predicted)
            candidates.append({"prefix_n": n, "margin": margin, **result})
            cached[(n, margin)] = (model, all_rows)
    safe = [row for row in candidates if row["false_healthy"] == 0]
    pool = safe or candidates
    pool.sort(key=lambda row: (
        row["false_healthy"],
        -row["confusion"]["healthy_as_healthy"],
        -(row["confusion"]["healthy_as_healthy"] + row["confusion"]["unhealthy_as_unhealthy"]),
        row["prefix_n"],
        -row["margin"],
    ))
    chosen = pool[0]
    model, all_rows = cached[(chosen["prefix_n"], chosen["margin"])]
    return {"chosen": chosen, "safe_candidate_found": bool(safe),
            "all_candidates": candidates, "model": model}, all_rows


def evaluate(protocol: dict[str, Any], catalog: dict[str, Any], protocol_path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if protocol != read_json(protocol_path):
        raise ValueError("계산 프로토콜과 기록할 원본 파일이 다름")
    selection, rows = select(protocol, catalog)
    margin = selection["chosen"]["margin"]
    model = selection["model"]
    predictions = []
    for row in rows:
        if row["features"] is None:
            predictions.append({**row, "score": None, "prediction": "abstain_missing_input"})
        else:
            value = score(model, row["features"])
            predictions.append({**row, "score": value, "prediction": decide(value, margin)})
    split_metrics = {split: metrics([row for row in predictions if row["split"] == split])
                     for split in ("development", "evaluation", "transfer_audit")}
    gate = protocol["acceptance"]["reliable_behavior_gate"]
    evaluation = split_metrics["evaluation"]
    reliable_behavior = bool(
        selection["safe_candidate_found"]
        and evaluation["n"]
        and evaluation["false_healthy"] == 0
        and (evaluation["healthy_coverage"] or 0) >= 0.5
        and (evaluation["unhealthy_detection"] or 0) >= 0.5
    )
    recovery_counts = Counter(row["recovery_label"] for row in catalog["runs"] if row["recovery_label"])
    report = {
        "schema": "early_recovery_decider_report_v1",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256_lf": sha256(protocol_path),
        "implementation_sha256_lf": sha256(Path(__file__)),
        "measurement_class": "development_measurement_after_results_were_seen",
        "catalog": {key: value for key, value in catalog.items() if key != "runs"},
        "label_counts_after_dedup": dict(Counter(row["behavior_label"] for row in catalog["runs"])),
        "split_counts": dict(Counter(row["split"] for row in catalog["runs"])),
        "selection": selection,
        "metrics": split_metrics,
        "acceptance_expression": gate,
        "reliable_behavior_decider": reliable_behavior,
        "recovery_label_counts": dict(recovery_counts),
        "reliable_recovery_decider": False,
        "recovery_limitation": "공개 72개 텔레메트리에는 교정 lineage와 recovered/not_recovered receipt가 함께 연결된 표본이 없다.",
        "false_recovery_cost": "측정 불가: 행동 healthy를 recovered로 바꾸지 않았다.",
        "counterfactual_scope": "300-N은 오프라인 가상 조기 행동 판정 반복 수이며 GPU 시간·실측 절감량이 아니다.",
        "wording": protocol["acceptance"]["behavior_only_wording"] if reliable_behavior
                   else protocol["acceptance"]["unreliable_wording"],
    }
    return report, predictions


def write_outputs(output: Path, report: dict[str, Any], catalog: dict[str, Any], predictions: list[dict[str, Any]]) -> None:
    write_json(output / "catalog.json", catalog)
    write_json(output / "report.json", report)
    csv_path = output / "predictions.csv"
    if csv_path.exists():
        raise FileExistsError(f"기존 결과를 덮어쓰지 않는다: {csv_path}")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["run", "seed", "task", "split", "behavior_label", "recovery_label", "n", "score", "prediction"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    catalog = build_catalog(protocol)
    if args.catalog_only:
        print(json.dumps({key: value for key, value in catalog.items() if key != "runs"}, ensure_ascii=False, indent=2))
        return 0
    report, predictions = evaluate(protocol, catalog, args.protocol)
    write_outputs(args.output, report, catalog, predictions)
    print(json.dumps({"output": str(args.output), "reliable_behavior_decider": report["reliable_behavior_decider"],
                      "reliable_recovery_decider": report["reliable_recovery_decider"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
