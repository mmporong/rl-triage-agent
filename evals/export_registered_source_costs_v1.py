"""등록된 공통 원천의 측정 비용을 방식별 비용과 분리해 한 번만 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rl_triage.gpu_resource_sampler import integrate_device_energy

SCHEMA = "registered_source_costs_v1"
EXPECTED_SOURCE_COUNT = 20
EXPECTED_PUBLIC_FILE_COUNT = 63
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_TEXT = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\Users\\|/home/|/mnt/|bench[\\/]private)", re.IGNORECASE
)


class SourceCostError(ValueError):
    """원천 비용 증거가 등록된 결합과 다를 때 발생한다."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceCostError(f"{label}을 읽을 수 없다: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceCostError(f"{label}은 객체여야 한다")
    return value


def _ref(value: object, label: str) -> tuple[Path, str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise SourceCostError(f"{label} ref 필드가 다르다")
    path_value, digest = value["path"], value["sha256"]
    if not isinstance(path_value, str) or not isinstance(digest, str) or not SHA256.fullmatch(digest):
        raise SourceCostError(f"{label} ref 값이 유효하지 않다")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != digest:
        raise SourceCostError(f"{label} ref SHA가 다르다")
    return path, digest, _json(path, label)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceCostError(f"{label}은 수여야 한다")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise SourceCostError(f"{label}은 유한한 음수 아닌 수여야 한다")
    return number


def _assert_public(value: object) -> None:
    if isinstance(value, str) and PRIVATE_TEXT.search(value):
        raise SourceCostError("공개 결과에 로컬 또는 비공개 경로가 포함됐다")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_public(key)
            _assert_public(child)
    elif isinstance(value, list):
        for child in value:
            _assert_public(child)


def _verify_source_export(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path, "source export manifest")
    if (manifest.get("schema") != "lineage_source_collection_public_v2_manifest"
            or manifest.get("status") != "complete"):
        raise SourceCostError("source export manifest 상태가 다르다")
    files, public_files = manifest.get("files"), manifest.get("public_files")
    if not isinstance(files, dict) or not isinstance(public_files, list):
        raise SourceCostError("source export manifest 파일 목록이 없다")
    if len(public_files) != EXPECTED_PUBLIC_FILE_COUNT or set(public_files) != set(files) | {"manifest.json"}:
        raise SourceCostError("source export 공개 파일 수 또는 목록이 다르다")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != set(public_files):
        raise SourceCostError("source export 실제 파일 목록이 manifest와 다르다")
    for relative, binding in files.items():
        if not isinstance(binding, dict) or set(binding) != {"sha256", "size_bytes"}:
            raise SourceCostError("source export 파일 결합 필드가 다르다")
        path = root / relative
        if (_sha(path) != binding["sha256"] or path.stat().st_size != binding["size_bytes"]):
            raise SourceCostError(f"source export 파일 결합이 다르다: {relative}")
    provenance = _json(root / "provenance.json", "source export provenance")
    summary = _json(root / "summary.json", "source export summary")
    if (provenance.get("schema") != "lineage_source_collection_public_v2_provenance"
            or summary.get("schema") != "lineage_source_collection_public_v2_summary"
            or summary.get("status") != "complete"):
        raise SourceCostError("source export 공개 원문 상태가 다르다")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or counts.get("training_sdk_completed") != EXPECTED_SOURCE_COUNT \
            or counts.get("fixed26_sdk_completed") != EXPECTED_SOURCE_COUNT:
        raise SourceCostError("source export 완료 수가 다르다")
    return {
        "manifest_sha256": _sha(manifest_path),
        "provenance_sha256": _sha(root / "provenance.json"),
        "summary_sha256": _sha(root / "summary.json"),
        "protocol_sha256": provenance.get("protocol_sha256"),
        "source_result_sha256": provenance.get("source_result_sha256"),
        "source_index_sha256": provenance.get("source_index_sha256"),
        "file_count": len(public_files),
    }


def _verify_extract_receipt(ref: object, seen: set[tuple[str, str]]) -> tuple[dict[str, str], str]:
    path, digest, body = _ref(ref, "telemetry extract receipt")
    identity = (str(path), digest)
    if identity in seen:
        raise SourceCostError("telemetry extract receipt가 중복됐다")
    seen.add(identity)
    required = {"cpu_only", "event_sha256_after", "event_sha256_before", "extractor_sha256_lf",
                "returncode", "schema", "telemetry_sha256", "timeout_s"}
    if (set(body) != required or body.get("schema") != "lineage_source_inputs_extract_receipt_v2"
            or body.get("cpu_only") is not True or body.get("returncode") != 0
            or body.get("event_sha256_before") != body.get("event_sha256_after")):
        raise SourceCostError("telemetry extract receipt가 CPU 성공 계약과 다르다")
    # 이 receipt에는 wall 시간이 없다. 부재를 0으로 바꾸지 않는다.
    return {"sha256": digest}, str(path)


def _verify_sources(source_root: Path, source_index: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = source_root / "result.json"
    result = _json(result_path, "source result")
    if (result.get("schema") != "lineage_source_collection_result_v2" or result.get("status") != "complete"
            or result.get("source_jobs_completed") != EXPECTED_SOURCE_COUNT
            or result.get("fixed26_completed") != EXPECTED_SOURCE_COUNT or result.get("model_api") != 0):
        raise SourceCostError("source result 완료 결합이 다르다")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SOURCE_COUNT:
        raise SourceCostError("source result record 수가 다르다")

    index = _json(source_index, "source index")
    if (index.get("schema") != "lineage_source_inputs_v2" or index.get("source_count") != EXPECTED_SOURCE_COUNT):
        raise SourceCostError("source index 수가 다르다")
    metadata_ref = index.get("metadata_ref")
    if not isinstance(metadata_ref, dict) or set(metadata_ref) != {"file", "sha256"}:
        raise SourceCostError("source metadata ref가 다르다")
    metadata_path = source_index.parent / metadata_ref["file"]
    if not metadata_path.is_file() or _sha(metadata_path) != metadata_ref["sha256"]:
        raise SourceCostError("source metadata SHA가 다르다")
    metadata = _json(metadata_path, "source metadata")
    source_result_path, source_result_sha, _ = _ref(metadata.get("source_result"), "metadata source result")
    if source_result_path != result_path.resolve() or source_result_sha != _sha(result_path):
        raise SourceCostError("metadata source result 결합이 다르다")
    protocol_path, protocol_sha, protocol = _ref(metadata.get("protocol"), "source protocol")
    if protocol.get("schema") != "lineage_source_collection_protocol_v2":
        raise SourceCostError("source protocol schema가 다르다")
    jobs = protocol.get("training_jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_SOURCE_COUNT:
        raise SourceCostError("source protocol job 수가 다르다")
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
            raise SourceCostError("source protocol job ID가 다르다")
        jobs_by_id[job["job_id"]] = job
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("job_id"), str):
            raise SourceCostError("source result job ID가 다르다")
        records_by_id[record["job_id"]] = record
    if len(jobs_by_id) != EXPECTED_SOURCE_COUNT or set(jobs_by_id) != set(records_by_id):
        raise SourceCostError("source protocol과 result job 결합이 다르다")
    sources = metadata.get("sources")
    public_sources = index.get("sources")
    if not isinstance(sources, dict) or not isinstance(public_sources, dict) \
            or len(sources) != EXPECTED_SOURCE_COUNT or len(public_sources) != EXPECTED_SOURCE_COUNT:
        raise SourceCostError("source metadata 수가 다르다")

    roles: dict[str, dict[str, Any]] = {
        "case": {"jobs": 0, "training_owned_wall_s": 0.0, "fixed_owned_wall_s": 0.0,
                 "cpu_telemetry_extractions": 0, "cpu_telemetry_wall_s": None},
        "reference": {"jobs": 0, "training_owned_wall_s": 0.0, "fixed_owned_wall_s": 0.0,
                      "cpu_telemetry_extractions": 0, "cpu_telemetry_wall_s": None},
    }
    receipt_seen: set[tuple[str, str]] = set()
    private_stages: list[dict[str, Any]] = []
    private_receipts: list[dict[str, Any]] = []
    for job_id in sorted(jobs_by_id):
        job, record = jobs_by_id[job_id], records_by_id[job_id]
        role = job.get("role")
        if role not in roles or record.get("role") != role:
            raise SourceCostError(f"job role 결합이 다르다: {job_id}")
        source = sources.get(job_id)
        if not isinstance(source, dict) or source.get("manifest", {}).get("job_id") != job_id:
            raise SourceCostError(f"metadata job 결합이 다르다: {job_id}")
        if source["manifest"].get("protocol_sha256_lf") != protocol_sha:
            raise SourceCostError(f"metadata protocol SHA가 다르다: {job_id}")
        public_id = source.get("public_id")
        if not isinstance(public_id, str) or public_id not in public_sources:
            raise SourceCostError(f"public source 결합이 다르다: {job_id}")
        material = source.get("material")
        if not isinstance(material, dict) or not isinstance(material.get("stage"), str):
            raise SourceCostError(f"source stage가 없다: {job_id}")
        stage = Path(material["stage"]).resolve()
        if stage.parent != source_root.resolve() or stage.name != job_id:
            raise SourceCostError(f"source stage 경계가 다르다: {job_id}")
        owned_path, fixed_path, fixed_owned_path = (
            stage / "owned-result.json", stage / "fixed-result.json", stage / "fixed" / "owned-result.json"
        )
        owned = _json(owned_path, f"{job_id} training owned result")
        fixed = _json(fixed_path, f"{job_id} fixed result")
        fixed_owned = _json(fixed_owned_path, f"{job_id} fixed owned result")
        if fixed.get("execution") != fixed_owned:
            raise SourceCostError(f"fixed owned result 결합이 다르다: {job_id}")
        training_wall = _number(owned.get("wall_s"), f"{job_id} training wall")
        fixed_wall = _number(fixed_owned.get("wall_s"), f"{job_id} fixed wall")
        if record.get("owned_parent_wall_s") != owned.get("wall_s") \
                or record.get("fixed26", {}).get("execution") != fixed_owned:
            raise SourceCostError(f"source result wall 결합이 다르다: {job_id}")
        roles[role]["jobs"] += 1
        roles[role]["training_owned_wall_s"] += training_wall
        roles[role]["fixed_owned_wall_s"] += fixed_wall
        roles[role]["cpu_telemetry_extractions"] += 1
        receipt_public, receipt_path = _verify_extract_receipt(material.get("receipt"), receipt_seen)
        private_receipts.append({"job_id": job_id, "path": receipt_path, **receipt_public})
        private_stages.append({
            "job_id": job_id, "role": role, "stage": str(stage),
            "training_owned_result": {"path": str(owned_path), "sha256": _sha(owned_path)},
            "fixed_result": {"path": str(fixed_path), "sha256": _sha(fixed_path)},
            "fixed_owned_result": {"path": str(fixed_owned_path), "sha256": _sha(fixed_owned_path)},
        })
    if roles["case"]["jobs"] != 10 or roles["reference"]["jobs"] != 10:
        raise SourceCostError("case/reference 역할 수가 다르다")
    for value in roles.values():
        value["simulator_owned_wall_s"] = (
            float(value["training_owned_wall_s"]) + float(value["fixed_owned_wall_s"])
        )
        value["gpu_simulator_owned_wall_s"] = value["simulator_owned_wall_s"]
    parent_wall = _number(result.get("wall_s"), "source parent wall")
    public = {
        "source_result_sha256": _sha(result_path),
        "_source_result_path": str(result_path.resolve()),
        "source_protocol_sha256": protocol_sha,
        "source_index_sha256": _sha(source_index),
        "source_metadata_sha256": _sha(metadata_path),
        "parent_span_s": parent_wall,
        "roles": roles,
        "simulator_owned_wall_s": sum(float(value["simulator_owned_wall_s"]) for value in roles.values()),
        "telemetry_extraction": {
            "jobs": len(receipt_seen), "cpu_only_receipts": len(receipt_seen),
            "cpu_wall_s": None, "wall_measurement": "not_recorded_in_receipts",
        },
    }
    private = {
        "source_root": str(source_root.resolve()), "source_result": str(result_path.resolve()),
        "source_protocol": str(protocol_path), "source_index": str(source_index.resolve()),
        "source_metadata": str(metadata_path.resolve()), "stages": private_stages,
        "telemetry_receipts": private_receipts,
    }
    return public, private


def _load_sample_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    unavailable = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            batch = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceCostError(f"sampler JSONL {line_number}이 유효하지 않다") from exc
        if not isinstance(batch, dict) or batch.get("status") not in {"sampled", "unavailable"}:
            raise SourceCostError("sampler batch 상태가 다르다")
        if batch["status"] == "unavailable":
            unavailable += 1
            continue
        rows = batch.get("samples")
        if not isinstance(rows, list):
            raise SourceCostError("sampler samples가 목록이 아니다")
        for row in rows:
            if not isinstance(row, dict):
                raise SourceCostError("sampler sample이 객체가 아니다")
            scope = row.get("measurement_scope")
            if scope != {"power": "device_wide", "process_vram_attribution": "not_measured",
                         "board_power_process_attribution": "not_permitted"}:
                raise SourceCostError("sampler가 허용되지 않은 장치 귀속을 주장한다")
            samples.append(row)
    return samples, unavailable


def _verify_metering(path: Path, source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metering = _json(path, "source metering result")
    if (metering.get("schema") != "measured_source_result_v1" or metering.get("status") != "complete"
            or metering.get("source_jobs_completed") != EXPECTED_SOURCE_COUNT
            or metering.get("fixed26_completed") != EXPECTED_SOURCE_COUNT or metering.get("model_api") != 0
            or metering.get("provider_gpu_seconds") is not None or metering.get("actual_billing_usd") is not None):
        raise SourceCostError("source metering result 결합이 다르다")
    source_result_path, source_result_sha, _ = _ref(metering.get("source_result"), "metered source result")
    if str(source_result_path) != source["_source_result_path"] \
            or source_result_sha != source["source_result_sha256"] \
            or metering.get("source_protocol_sha256") != source["source_protocol_sha256"]:
        raise SourceCostError("metering source SHA 결합이 다르다")
    start_outer, stop_outer = metering.get("device_sampling_start"), metering.get("device_sampling_stop")
    if not isinstance(start_outer, dict) or start_outer.get("scope") != "device_wide" \
            or not isinstance(stop_outer, dict) or stop_outer.get("scope") != "device_wide":
        raise SourceCostError("metering sampler scope가 다르다")
    start_path, start_sha, start = _ref(start_outer.get("receipt_ref"), "sampler start")
    stop_path, stop_sha, stop = _ref(stop_outer.get("receipt_ref"), "sampler stop")
    if (start.get("schema") != "periodic_device_sampling_start_v1" or start.get("scope") != "device_wide"
            or stop.get("schema") != "periodic_device_sampling_stop_v1" or stop.get("scope") != "device_wide"
            or stop.get("lease_matches") is not True or stop.get("errors") != []):
        raise SourceCostError("sampler receipt 상태가 다르다")
    stop_start_path, stop_start_sha, _ = _ref(stop.get("start_ref"), "sampler stop start ref")
    if stop_start_path != start_path or stop_start_sha != start_sha:
        raise SourceCostError("sampler start receipt 결합이 다르다")
    samples_path, samples_sha, _ = _ref_jsonl(stop.get("samples_ref"), "sampler samples")
    samples, unavailable = _load_sample_rows(samples_path)
    if len(samples) != stop.get("sample_count"):
        raise SourceCostError("sampler sample 수가 다르다")
    recomputed = integrate_device_energy(
        samples, window_start=stop.get("started_utc"), window_end=stop.get("ended_utc"),
        max_gap_s=_number(start.get("max_gap_s"), "sampler max gap"),
    )
    if recomputed != stop.get("energy"):
        raise SourceCostError("sampler energy 재적분 결과가 다르다")
    devices: list[dict[str, Any]] = []
    summary = stop.get("device_summary")
    if not isinstance(summary, dict) or set(summary) != set(recomputed["devices"]):
        raise SourceCostError("sampler device summary 결합이 다르다")
    for index, uuid in enumerate(sorted(recomputed["devices"]), 1):
        device = recomputed["devices"][uuid]
        measured = summary[uuid]
        if not isinstance(measured, dict) or measured.get("scope") != "whole_device_including_other_loads":
            raise SourceCostError("device summary scope가 다르다")
        device_rows = [row for row in samples if row.get("device_uuid") == uuid]
        expected_metrics = {
            "gpu_util_percent_sample_mean": _metric(
                [row.get("utilization", {}).get("gpu_percent") for row in device_rows], mean=True
            ),
            "memory_used_mib_peak": _metric([row.get("memory_used_mib") for row in device_rows]),
            "power_draw_w_peak": _metric([row.get("power_draw_w") for row in device_rows]),
        }
        if measured.get("metrics") != expected_metrics:
            raise SourceCostError("device summary metric 재계산 결과가 다르다")
        devices.append({
            "logical_id": f"device-{index:02d}", "device_uuid_sha256": hashlib.sha256(uuid.encode()).hexdigest(),
            "scope": "whole_device_including_other_loads", "sample_count": device["sample_count"],
            "window_s": device["window_s"], "covered_s": device["covered_s"],
            "coverage_fraction": device["coverage_fraction"], "energy_wh": device["energy_wh"],
            "gpu_util_percent_sample_mean": measured.get("metrics", {}).get("gpu_util_percent_sample_mean"),
            "memory_used_mib_peak": measured.get("metrics", {}).get("memory_used_mib_peak"),
            "power_draw_w_peak": measured.get("metrics", {}).get("power_draw_w_peak"),
            "process_attribution": None,
        })
    public = {
        "scope": "device_wide_including_other_loads_not_attributable_to_source_jobs",
        "sample_count": len(samples), "unavailable_batches": unavailable, "devices": devices,
        "process_gpu_seconds": stop.get("process_gpu_seconds"), "provider_gpu_seconds": None,
    }
    private = {
        "metering_result": str(path.resolve()), "source_result": str(source_result_path),
        "sampler_start": {"path": str(start_path), "sha256": start_sha},
        "sampler_stop": {"path": str(stop_path), "sha256": stop_sha},
        "sampler_samples": {"path": str(samples_path), "sha256": samples_sha},
    }
    return public, private


def _ref_jsonl(value: object, label: str) -> tuple[Path, str, None]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise SourceCostError(f"{label} ref 필드가 다르다")
    if not isinstance(value["path"], str):
        raise SourceCostError(f"{label} ref 경로가 다르다")
    path, digest = Path(value["path"]).resolve(), value["sha256"]
    if not isinstance(digest, str) or not SHA256.fullmatch(digest) or not path.is_file() or _sha(path) != digest:
        raise SourceCostError(f"{label} ref SHA가 다르다")
    return path, digest, None


def _metric(values: list[object], *, mean: bool = False) -> dict[str, float | int | None]:
    valid = [float(value) for value in values if isinstance(value, (int, float))
             and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0]
    return {
        "value": ((sum(valid) / len(valid) if mean else max(valid)) if valid else None),
        "valid_samples": len(valid), "missing_or_invalid_samples": len(values) - len(valid),
    }


def export(*, source_root: Path, source_export: Path, metering_result: Path,
           source_index: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"새 출력 폴더가 필요하다: {out}")
    export_binding = _verify_source_export(source_export.resolve())
    source, source_private = _verify_sources(source_root.resolve(), source_index.resolve())
    if (export_binding["protocol_sha256"] != source["source_protocol_sha256"]
            or export_binding["source_result_sha256"] != source["source_result_sha256"]
            or export_binding["source_index_sha256"] != source["source_index_sha256"]):
        raise SourceCostError("공개 source export와 원문 SHA 결합이 다르다")
    metering, metering_private = _verify_metering(metering_result.resolve(), source)
    public = {
        "schema": SCHEMA, "status": "complete", "scope": "shared_common_source_cost",
        "allocation": {
            "charged_once": True, "per_method": False,
            "rule": "shared source acquisition is not added to each comparison method",
        },
        "bindings": {
            "source_protocol_sha256": source["source_protocol_sha256"],
            "source_result_sha256": source["source_result_sha256"],
            "source_index_sha256": source["source_index_sha256"],
            "source_metadata_sha256": source["source_metadata_sha256"],
            "source_export_manifest_sha256": export_binding["manifest_sha256"],
            "source_export_file_count": export_binding["file_count"],
            "metering_result_sha256": _sha(metering_result),
        },
        "counts": {"training_jobs": 20, "fixed_evaluations": 20, "telemetry_extractions": 20,
                   "model_api_calls": 0},
        "wall_clock": {
            "source_parent_span_s": source["parent_span_s"],
            "simulator_owned_wall_s": source["simulator_owned_wall_s"],
            "by_role": source["roles"],
            "aggregation_rule": "parent span and owned simulator sum are separate views and are not added",
        },
        "telemetry_extraction": source["telemetry_extraction"],
        "gpu_measurement": metering,
        "provider_cost": {"provider_gpu_seconds": None, "provider_price_usd": None,
                          "actual_billing_usd": None, "reason": "not_recorded"},
        "exclusions": [{
            "item": "prior_development_and_smoke_sdk_jobs", "count": 7, "included": False,
            "reason": "outside_this_registered_source_scope",
            "classification": "includes_failures; exact validity breakdown not counted in this scope",
        }],
    }
    _assert_public(public)
    refs = {
        "schema": "registered_source_cost_refs_v1", "public_result": "summary.json",
        "source_export": str(source_export.resolve()), "source": source_private, "metering": metering_private,
    }
    out.mkdir(parents=True)
    _write_new(out / "summary.json", public)
    _write_new(out / "refs.json", refs)
    _write_new(out / "manifest.json", {
        "schema": "registered_source_cost_manifest_v1", "status": "complete",
        "files": {name: {"sha256": _sha(out / name), "size_bytes": (out / name).stat().st_size}
                  for name in ("summary.json", "refs.json")},
    })
    return public


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-export", type=Path, required=True)
    parser.add_argument("--metering-result", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = export(source_root=args.source_root, source_export=args.source_export,
                    metering_result=args.metering_result, source_index=args.source_index, out=args.out)
    print(json.dumps({"status": result["status"], "output": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
