"""전향 source 8조건의 등록 적격 판정을 경로 없는 공개 근거로 내보낸다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]

import run_prospective_reasoning_confirmation_v1 as R  # noqa: E402

SUMMARY_SCHEMA = "prospective_source_qualification_public_summary_v1"
CONDITION_SCHEMA = "prospective_source_qualification_public_condition_v1"
PROVENANCE_SCHEMA = "prospective_source_qualification_public_provenance_v1"
MANIFEST_SCHEMA = "prospective_source_qualification_public_manifest_v1"
METRICS = ("fall_rate", "mean_survival_s", "lin_vel_rmse_mps", "yaw_rate_rmse_radps")


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON object여야 한다")
    return cast(dict[str, Any], value)


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(cast(float, value)))
            and float(cast(float, value)) >= 0, f"{label}은 유한 비음수여야 한다")
    return float(cast(float, value))


def _checked_ref(value: object, label: str) -> Path:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{label} ref 형식")
    ref = cast(dict[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve()
    require(path.is_file() and ref["sha256"] == _sha(path), f"{label} ref/SHA 불일치")
    return path


def _owned_time(value: object, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} execution 형식")
    execution = cast(dict[str, Any], value)
    require(execution.get("exit_code") == 0 and execution.get("natural_exit") is True
            and execution.get("cleanup_verified") is True
            and execution.get("timed_out") is False and execution.get("error_type") is None,
            f"{label} owned execution 불완결")
    scope = execution.get("wall_scope")
    require(isinstance(scope, str) and bool(scope), f"{label} wall scope 누락")
    return {"wall_s": _finite(execution.get("wall_s"), f"{label} wall"),
            "scope": scope, "gpu_kernel_time": False}


def _metrics(report: dict[str, Any], checkpoint_sha256: str) -> dict[str, float]:
    require(report.get("protocol") == "fixed_eval_v1"
            and isinstance(report.get("by_condition"), list)
            and len(report["by_condition"]) == 26
            and report.get("checkpoint", {}).get("sha256") == checkpoint_sha256,
            "fixed26 report protocol/조건/checkpoint 불일치")
    raw = report.get("metrics")
    require(isinstance(raw, dict) and set(raw) == set(METRICS), "fixed26 metrics 계약 불일치")
    return {name: _finite(cast(dict[str, Any], raw)[name], f"fixed26 {name}")
            for name in METRICS}


def _condition(row: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    checkpoint = row.get("checkpoint")
    require(isinstance(checkpoint, dict) and set(checkpoint) >= {"sha256"},
            "source checkpoint 형식")
    checkpoint = cast(dict[str, Any], checkpoint)
    checkpoint_sha = checkpoint["sha256"]
    require(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64,
            "source checkpoint SHA 형식")
    fixed_report = row.get("fixed_report")
    require(isinstance(fixed_report, dict), "source fixed report 형식")
    fixed_report = cast(dict[str, Any], fixed_report)
    result = _read(Path(row["result"]))
    require(result.get("schema") == R.TRAIN.RESULT_SCHEMA and result.get("status") == "complete"
            and result.get("seed") == row["seed"] and result.get("category") == row["category"]
            and result.get("checkpoint", {}).get("sha256") == checkpoint_sha
            and result.get("report") == fixed_report
            and result.get("claims") == {"recovery": False, "diagnosis": False,
                                           "model_advantage": False},
            "source result identity/claims 불일치")
    training_path = _checked_ref(result.get("training"), "training result")
    training = _read(training_path)
    require(training.get("checkpoint", {}).get("sha256") == checkpoint_sha,
            "training/checkpoint 불일치")
    return {
        "schema": CONDITION_SCHEMA, "case_id": row["case_id"], "seed": row["seed"],
        "category": row["category"], "checkpoint_sha256": checkpoint_sha,
        "fixed_conditions": 26, "registered_metrics": _metrics(fixed_report, checkpoint_sha),
        "gate": deepcopy(gate),
        "timing": {"training_owned_job": _owned_time(training.get("execution"), "training"),
                   "fixed_eval_owned_job": _owned_time(result.get("fixed_execution"), "fixed")},
        "claims": {"diagnosis": False, "repair": False, "recovery": False,
                   "model_advantage": False, "visual_benefit": False},
    }


def _expected_qualification(rows: list[dict[str, Any]],
                            gate_fn: Callable[[dict[str, Any], dict[str, Any],
                                               dict[str, Any]], dict[str, Any]],
                            bands: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    references = {row["seed"]: row for row in rows if row["category"] == "NONE"}
    require(set(references) == {row["seed"] for row in rows}, "same-seed NONE reference 불완전")
    gates = [{"case_id": row["case_id"], "category": row["category"],
              "gate": deepcopy(gate_fn(row, references[row["seed"]], bands))}
             for row in rows]
    qualified = (sum(item["gate"].get("reason") == "healthy_gate" for item in gates) == 2
                 and sum(item["gate"].get("run") is True for item in gates) == 6
                 and all((item["category"] == "NONE")
                         == (item["gate"].get("reason") == "healthy_gate") for item in gates))
    value = {"schema": "prospective_reasoning_source_qualification_v1",
             "status": "qualified" if qualified else "source_not_qualified",
             "gates": gates, "api_requests": 0, "replacement_seed": False,
             "comparison_outcome": None if qualified else "indeterminate"}
    return value, gates


def _safe(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).lower()
    forbidden = ("bench/private", "bench\\private", "c:\\users\\", "/home/", "/mnt/",
                 "reasoning_content", '"reasoning"', "api_key", "authorization")
    require(not any(token in encoded for token in forbidden),
            "공개 projection에 비공개 경로·추론·자격 정보가 있다")


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def export(design_path: Path, anchor: str, source_batch: Path, qualification_path: Path,
           out: Path, *, verify_design_fn: Callable[[Path, str], dict[str, Any]] = R.verify_design,
           source_rows_fn: Callable[[Path, dict[str, Any]], list[dict[str, Any]]] = R._source_rows,
           gate_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]],
                             dict[str, Any]] = R.BASE.RUN._gate) -> dict[str, Any]:
    """등록된 source 8조건과 qualification을 재검증해 fresh 공개 폴더로 내보낸다."""
    design_path, source_batch, qualification_path, out = map(
        Path.resolve, (Path(design_path), Path(source_batch), Path(qualification_path), Path(out)))
    require(not out.exists(), "공개 output은 fresh여야 한다")
    design = verify_design_fn(design_path, anchor)
    batch = _read(source_batch)
    protocol_ref = batch.get("protocol")
    source_protocol = ROOT / design["source_protocol"]["path"]
    require(batch.get("schema") == R.SOURCE.BATCH_SCHEMA and batch.get("status") == "complete"
            and batch.get("anchor") == anchor and batch.get("model_requests") == 0
            and batch.get("repairs") == 0 and batch.get("automatic_retry") is False
            and isinstance(protocol_ref, dict)
            and Path(protocol_ref["path"]).resolve() == source_protocol.resolve()
            and protocol_ref.get("sha256") == design["source_protocol"]["sha256"],
            "source batch registration/anchor/권한 불일치")
    rows = source_rows_fn(source_batch, design)
    expected_identity = [(f"c{i:02d}", seed, category)
                         for i, (seed, category) in enumerate(
                             ((seed, category) for seed in design["seeds"]
                              for category in design["categories"]), 1)]
    require([(row.get("case_id"), row.get("seed"), row.get("category")) for row in rows]
            == expected_identity, "source 8조건 identity/category 순서 불일치")
    bands_path = R.BASE.RUN.BANDS
    require(_sha(bands_path) == R.BASE.RUN.BANDS_SHA256, "fixed26 bands SHA 불일치")
    bands_value = _read(bands_path)
    bands = bands_value.get("bands")
    require(isinstance(bands, dict), "fixed26 bands 형식")
    expected, gates = _expected_qualification(rows, gate_fn, cast(dict[str, Any], bands))
    qualification = _read(qualification_path)
    require(qualification == expected, "source qualification gate/summary 재계산 불일치")
    conditions = [_condition(row, gate["gate"]) for row, gate in zip(rows, gates, strict=True)]
    summary = {
        "schema": SUMMARY_SCHEMA, "status": qualification["status"],
        "comparison_outcome": qualification["comparison_outcome"],
        "conditions": 8, "source_seeds": len(design["seeds"]),
        "categories": deepcopy(design["categories"]),
        "healthy_reference_gates": sum(item["gate"]["reason"] == "healthy_gate"
                                       for item in gates),
        "runnable_fault_gates": sum(item["gate"].get("run") is True for item in gates),
        "api_requests": 0, "replacement_seed": False,
        "batch_wall_observation": {"wall_s": _finite(batch.get("wall_s"), "batch wall"),
                                   "scope": "source_batch_parent_wall_not_gpu_kernel_time"},
        "claims": {"source_qualification_only": True, "diagnosis": False,
                   "repair": False, "recovery": False, "model_advantage": False,
                   "llm_superiority": False, "new_fault_family_generalization": False,
                   "visual_benefit": False},
    }
    provenance = {
        "schema": PROVENANCE_SCHEMA, "stage_a_anchor": anchor,
        "design_protocol": {"path": design_path.relative_to(ROOT).as_posix(),
                            "sha256": _sha(design_path)},
        "source_protocol": deepcopy(design["source_protocol"]),
        "source_batch_sha256": _sha(source_batch),
        "qualification_sha256": _sha(qualification_path),
        "fixed26_bands": {"path": bands_path.relative_to(ROOT).as_posix(),
                          "sha256": _sha(bands_path)},
        "source_sha256_lf": deepcopy(design["source_sha256_lf"]),
        "private_paths_published": False, "reasoning_body_published": False,
    }
    artifacts: dict[str, object] = {"summary.json": summary,
                                   "source-provenance.json": provenance}
    for condition in conditions:
        artifacts[f"conditions/{condition['case_id']}.json"] = condition
    for value in artifacts.values():
        _safe(value)
    out.mkdir(parents=True, exist_ok=False)
    for relative, value in artifacts.items():
        _write_new(out / relative, value)
    files = {name: _sha(out / name) for name in sorted(artifacts)}
    manifest = {"schema": MANIFEST_SCHEMA, "status": "complete",
                "artifact_count": len(files), "files": files}
    _safe(manifest)
    _write_new(out / "artifact-manifest.json", manifest)
    return deepcopy(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.design, args.anchor, args.source_batch, args.qualification, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export"]
