from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from evals import export_registered_source_costs_v1 as E
from rl_triage.gpu_resource_sampler import integrate_device_energy


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def execution(wall: float) -> dict:
    return {
        "cleanup_verified": True, "error_type": None, "exit_code": 0,
        "job_assigned_before_release": True, "natural_exit": True,
        "owned_process_identities": {}, "timed_out": False, "wall_s": wall,
        "wall_scope": "parent_launch_to_exit_and_owned_job_empty_verification",
    }


@pytest.fixture
def evidence(tmp_path: Path) -> dict[str, Path]:
    protocol_path = tmp_path / "protocol.json"
    jobs = []
    for index in range(20):
        role = "case" if index < 10 else "reference"
        jobs.append({"job_id": f"job-{index:02d}", "role": role, "seed": 1000 + index})
    write(protocol_path, {"schema": "lineage_source_collection_protocol_v2", "training_jobs": jobs})
    protocol_sha = digest(protocol_path)

    source_root = tmp_path / "source"
    records = []
    metadata_sources = {}
    public_sources = {}
    for index, job in enumerate(jobs):
        stage = source_root / job["job_id"]
        train = execution(10.0 + index)
        fixed = execution(1.0 + index / 10)
        write(stage / "owned-result.json", train)
        write(stage / "fixed" / "owned-result.json", fixed)
        write(stage / "fixed-result.json", {"execution": fixed, "report": {}})
        receipt = write(tmp_path / "inputs" / "sources" / job["job_id"] / "extract.receipt.json", {
            "cpu_only": True, "event_sha256_after": "a" * 64, "event_sha256_before": "a" * 64,
            "extractor_sha256_lf": "b" * 64, "returncode": 0,
            "schema": "lineage_source_inputs_extract_receipt_v2",
            "telemetry_sha256": f"{index:064x}", "timeout_s": 120,
        })
        public_id = f"source-{index + 1:02d}"
        metadata_sources[job["job_id"]] = {
            "manifest": {"job_id": job["job_id"], "protocol_sha256_lf": protocol_sha},
            "material": {"stage": str(stage.resolve()), "receipt": ref(receipt)},
            "public_id": public_id,
        }
        public_sources[public_id] = {"checkpoint_sha256": f"{index + 1:064x}"}
        records.append({"job_id": job["job_id"], "role": job["role"],
                        "owned_parent_wall_s": train["wall_s"], "fixed26": {"execution": fixed}})
    source_result = write(source_root / "result.json", {
        "schema": "lineage_source_collection_result_v2", "status": "complete", "model_api": 0,
        "source_jobs_completed": 20, "fixed26_completed": 20, "wall_s": 400.0, "records": records,
    })
    metadata = write(tmp_path / "inputs" / "metadata.json", {
        "schema": "lineage_source_inputs_metadata_v2", "protocol": ref(protocol_path),
        "source_result": ref(source_result), "sources": metadata_sources,
    })
    source_index = write(tmp_path / "inputs" / "collected.json", {
        "schema": "lineage_source_inputs_v2", "source_count": 20,
        "metadata_ref": {"file": "metadata.json", "sha256": digest(metadata)}, "sources": public_sources,
    })

    export_root = tmp_path / "public-source"
    summary = write(export_root / "summary.json", {
        "schema": "lineage_source_collection_public_v2_summary", "status": "complete",
        "counts": {"training_sdk_completed": 20, "fixed26_sdk_completed": 20},
    })
    provenance = write(export_root / "provenance.json", {
        "schema": "lineage_source_collection_public_v2_provenance",
        "protocol_sha256": protocol_sha, "source_result_sha256": digest(source_result),
        "source_index_sha256": digest(source_index),
    })
    exported = ["summary.json", "provenance.json"]
    for index in range(20):
        for name in ("fixed26.json", "training-curves.json", "provenance.json"):
            relative = f"sources/source-{index + 1:02d}/{name}"
            write(export_root / relative, {"source": index + 1, "kind": name})
            exported.append(relative)
    bindings = {name: {"sha256": digest(export_root / name), "size_bytes": (export_root / name).stat().st_size}
                for name in exported}
    write(export_root / "manifest.json", {
        "schema": "lineage_source_collection_public_v2_manifest", "status": "complete",
        "files": bindings, "public_files": [*exported, "manifest.json"],
    })
    assert len(exported) + 1 == 63 and digest(summary) and digest(provenance)

    sample_root = tmp_path / "meter" / "samples" / "collection-0001"
    started = datetime(2026, 11, 4, tzinfo=timezone.utc)  # noqa: UP017 - Python 3.10 회귀
    ended = started + timedelta(seconds=2)
    rows = []
    for second, power in ((0, 100.0), (2, 200.0)):
        rows.append({
            "sample_timestamp": (started + timedelta(seconds=second)).isoformat(),
            "device_uuid": "GPU-fixture", "power_draw_w": power, "memory_used_mib": 1000.0,
            "utilization": {"gpu_percent": 50.0, "memory_percent": 5.0},
            "measurement_scope": {"power": "device_wide", "process_vram_attribution": "not_measured",
                                  "board_power_process_attribution": "not_permitted"},
        })
    samples_path = sample_root / "samples.jsonl"
    samples_path.parent.mkdir(parents=True)
    samples_path.write_text("".join(json.dumps({"status": "sampled", "samples": [row]}) + "\n" for row in rows),
                            encoding="utf-8")
    start = write(sample_root / "start.json", {
        "schema": "periodic_device_sampling_start_v1", "scope": "device_wide", "max_gap_s": 6.0,
    })
    energy = integrate_device_energy(rows, window_start=started.isoformat(), window_end=ended.isoformat(), max_gap_s=6)
    metrics = {
        "gpu_util_percent_sample_mean": {"value": 50.0, "valid_samples": 2, "missing_or_invalid_samples": 0},
        "memory_used_mib_peak": {"value": 1000.0, "valid_samples": 2, "missing_or_invalid_samples": 0},
        "power_draw_w_peak": {"value": 200.0, "valid_samples": 2, "missing_or_invalid_samples": 0},
    }
    stop = write(sample_root / "stop.json", {
        "schema": "periodic_device_sampling_stop_v1", "scope": "device_wide",
        "started_utc": started.isoformat(), "ended_utc": ended.isoformat(), "sample_count": 2,
        "errors": [], "lease_matches": True, "start_ref": ref(start), "samples_ref": ref(samples_path),
        "energy": energy, "device_summary": {"GPU-fixture": {
            "scope": "whole_device_including_other_loads", "metrics": metrics}},
        "process_gpu_seconds": None, "provider_gpu_seconds": None,
    })
    metering = write(tmp_path / "meter" / "result.json", {
        "schema": "measured_source_result_v1", "status": "complete", "source_jobs_completed": 20,
        "fixed26_completed": 20, "model_api": 0, "provider_gpu_seconds": None,
        "actual_billing_usd": None, "source_protocol_sha256": protocol_sha,
        "source_result": ref(source_result),
        "device_sampling_start": {"scope": "device_wide", "receipt_ref": ref(start)},
        "device_sampling_stop": {"scope": "device_wide", "receipt_ref": ref(stop)},
    })
    return {"source_root": source_root, "source_export": export_root, "source_index": source_index,
            "metering": metering, "metadata": metadata, "source_result": source_result,
            "samples": samples_path, "stop": stop}


def run_export(evidence: dict[str, Path], out: Path) -> dict:
    return E.export(source_root=evidence["source_root"], source_export=evidence["source_export"],
                    metering_result=evidence["metering"], source_index=evidence["source_index"], out=out)


def rebind_metadata_source_result(evidence: dict[str, Path]) -> None:
    metadata = json.loads(evidence["metadata"].read_text(encoding="utf-8"))
    metadata["source_result"] = ref(evidence["source_result"])
    write(evidence["metadata"], metadata)
    index = json.loads(evidence["source_index"].read_text(encoding="utf-8"))
    index["metadata_ref"]["sha256"] = digest(evidence["metadata"])
    write(evidence["source_index"], index)


def test_export_counts_shared_cost_once_and_preserves_unknown_costs(evidence: dict[str, Path], tmp_path: Path) -> None:
    result = run_export(evidence, tmp_path / "out")
    assert result["scope"] == "shared_common_source_cost"
    assert result["allocation"] == {
        "charged_once": True, "per_method": False,
        "rule": "shared source acquisition is not added to each comparison method",
    }
    assert result["counts"] == {"training_jobs": 20, "fixed_evaluations": 20,
                                 "telemetry_extractions": 20, "model_api_calls": 0}
    assert result["wall_clock"]["source_parent_span_s"] == 400.0
    assert result["wall_clock"]["simulator_owned_wall_s"] > 0
    assert result["telemetry_extraction"]["cpu_wall_s"] is None
    assert result["provider_cost"] == {"provider_gpu_seconds": None, "provider_price_usd": None,
                                        "actual_billing_usd": None, "reason": "not_recorded"}
    assert result["gpu_measurement"]["devices"][0]["process_attribution"] is None
    assert result["exclusions"] == [{
        "item": "prior_development_and_smoke_sdk_jobs", "count": 7, "included": False,
        "reason": "outside_this_registered_source_scope",
        "classification": "includes_failures; exact validity breakdown not counted in this scope",
    }]
    public_text = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in public_text and "bench/private" not in public_text
    refs = json.loads((tmp_path / "out" / "refs.json").read_text(encoding="utf-8"))
    assert refs["source_export"] == str(evidence["source_export"].resolve())


def test_count_mismatch_is_rejected(evidence: dict[str, Path]) -> None:
    result = json.loads(evidence["source_result"].read_text(encoding="utf-8"))
    result["source_jobs_completed"] = 19
    write(evidence["source_result"], result)
    rebind_metadata_source_result(evidence)
    with pytest.raises(E.SourceCostError, match="완료 결합"):
        E._verify_sources(evidence["source_root"], evidence["source_index"])


def test_source_export_sha_mismatch_is_rejected(evidence: dict[str, Path], tmp_path: Path) -> None:
    path = evidence["source_export"] / "sources" / "source-01" / "fixed26.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(E.SourceCostError, match="파일 결합"):
        run_export(evidence, tmp_path / "out")


def test_role_binding_mismatch_is_rejected(evidence: dict[str, Path]) -> None:
    result = json.loads(evidence["source_result"].read_text(encoding="utf-8"))
    result["records"][0]["role"] = "reference"
    write(evidence["source_result"], result)
    rebind_metadata_source_result(evidence)
    with pytest.raises(E.SourceCostError, match="role 결합"):
        E._verify_sources(evidence["source_root"], evidence["source_index"])


def test_duplicate_extract_receipt_is_rejected(evidence: dict[str, Path]) -> None:
    metadata = json.loads(evidence["metadata"].read_text(encoding="utf-8"))
    keys = sorted(metadata["sources"])
    metadata["sources"][keys[1]]["material"]["receipt"] = metadata["sources"][keys[0]]["material"]["receipt"]
    write(evidence["metadata"], metadata)
    index = json.loads(evidence["source_index"].read_text(encoding="utf-8"))
    index["metadata_ref"]["sha256"] = digest(evidence["metadata"])
    write(evidence["source_index"], index)
    with pytest.raises(E.SourceCostError, match="중복"):
        E._verify_sources(evidence["source_root"], evidence["source_index"])


def test_device_sample_cannot_claim_process_attribution(evidence: dict[str, Path]) -> None:
    batch = json.loads(evidence["samples"].read_text(encoding="utf-8").splitlines()[0])
    batch["samples"][0]["measurement_scope"]["board_power_process_attribution"] = "source_job"
    other = evidence["samples"].read_text(encoding="utf-8").splitlines()[1]
    evidence["samples"].write_text(json.dumps(batch) + "\n" + other + "\n", encoding="utf-8")
    stop = json.loads(evidence["stop"].read_text(encoding="utf-8"))
    stop["samples_ref"] = ref(evidence["samples"])
    write(evidence["stop"], stop)
    metering = json.loads(evidence["metering"].read_text(encoding="utf-8"))
    metering["device_sampling_stop"]["receipt_ref"] = ref(evidence["stop"])
    write(evidence["metering"], metering)
    source, _ = E._verify_sources(evidence["source_root"], evidence["source_index"])
    with pytest.raises(E.SourceCostError, match="허용되지 않은 장치 귀속"):
        E._verify_metering(evidence["metering"], source)
