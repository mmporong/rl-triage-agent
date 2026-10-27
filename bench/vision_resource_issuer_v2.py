"""새 촬영 승인 기한으로 기존 source-compatible GPU 자원 증거를 발급한다."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_vision_brev_batch_v1 as B
import vision_remote_gpu_v1 as G  # pyright: ignore[reportMissingImports]
import vision_resource_authorization_v1 as AUTH

UTC = timezone.utc  # noqa: UP017 - Python 3.10
ISSUANCE_SCHEMA = "vision_resource_issuance_v2"


def issue(authorization_path: Path, start_path: Path, provider: Path, gpu: Path,
          processes: Path, output: Path, *, now: datetime,
          repo: Path = AUTH.ROOT) -> dict[str, Any]:
    authorization = AUTH.validate(authorization_path, repo=repo, now=now)
    source = authorization["_source"]
    start = B.read(start_path)
    resource_id = authorization["resource_id"]
    required = {"schema", "resource_id", "resource_name", "before_inventory_path",
                "before_inventory_sha256", "command", "request_before_utc",
                "response_after_utc", "exit_code", "stdout_path", "stdout_sha256",
                "stderr_path", "stderr_sha256"}
    if (set(start) != required or start["schema"] != "v04_start_observation_v1"
            or start["resource_id"] != resource_id
            or start["resource_name"] != authorization["resource_name"]
            or start["exit_code"] != 0
            or start["command"] != ["brev", "start", start["resource_name"], "--detached"]):
        raise ValueError("소유 자원의 실제 시작 기록이 새 승인과 다르다")
    for anchor in ("before_inventory", "stdout", "stderr"):
        G._regular(Path(start[f"{anchor}_path"]), start[f"{anchor}_sha256"], anchor)
    previous = B.read(Path(start["before_inventory_path"]))
    rows = previous.get("workspaces")
    owned = ([row for row in rows if isinstance(row, dict) and row.get("id") == resource_id]
             if isinstance(rows, list) else [])
    if (len(owned) != 1 or owned[0].get("name") != authorization["resource_name"]
            or owned[0].get("status") != "STOPPED"):
        raise ValueError("시작 전 정확한 소유 ID·이름·STOPPED 원문이 다르다")
    started, responded = (B._stamp(start[key]) for key in ("request_before_utc", "response_after_utc"))
    current = now.astimezone(UTC)
    registered = B._stamp(authorization["registered_at_utc"])
    deadline = B._stamp(authorization["authorization_deadline_utc"])
    if not registered <= started <= responded <= current:
        raise ValueError("새 승인·시작 요청·응답·입장 시각 순서가 다르다")
    if (deadline - current).total_seconds() < AUTH.MINIMUM_REMAINING_S:
        raise ValueError("새 자원 승인의 첫 action 유효 시간이 부족하다")
    capture = {
        "resource_id": resource_id,
        "gpu_uuid": gpu.read_text(encoding="utf-8").strip(),
        "provider_inventory_path": str(provider.resolve()),
        "provider_inventory_sha256": B.sha256(provider),
        "nvidia_gpu_csv_path": str(gpu.resolve()),
        "nvidia_gpu_csv_sha256": B.sha256(gpu),
        "nvidia_process_csv_path": str(processes.resolve()),
        "nvidia_process_csv_sha256": B.sha256(processes),
    }
    G.verify_capture_proof(capture)
    row = G.verify_provider_inventory(provider.resolve(), B.sha256(provider), resource_id)
    if row["name"] != authorization["resource_name"]:
        raise ValueError("현재 provider 원문의 소유 이름이 다르다")
    if output.exists():
        raise FileExistsError("입장 증거 출력은 새 폴더여야 한다")
    source_sha = authorization["source_protocol"]["sha256_lf"]
    auth_sha = AUTH._sha(authorization_path, lf=True)
    common = {
        "status": "granted", "resource_id": resource_id, "gpu_uuid": capture["gpu_uuid"],
        "lease_id": f"v04a-{auth_sha[:18]}", "issued_at_utc": current.isoformat(),
        "expires_at_utc": deadline.isoformat(), "compute_started_at_utc": started.isoformat(),
        "protocol_sha256": source_sha,
        **{key: capture[key] for key in (
            "provider_inventory_sha256", "nvidia_gpu_csv_sha256", "nvidia_process_csv_sha256")},
    }
    raw_path, budget_path = output / "raw-gpu.json", output / "budget.json"
    B.write_new(raw_path, {"schema": B.RAW_GPU_RECEIPT_SCHEMA, **common})
    B.write_new(budget_path, {
        "schema": B.BUDGET_ADMISSION_SCHEMA, **common,
        **{key: source["resources"][key] for key in (
            "whole_batch_budget_usd", "compute_quote_usd_per_h", "max_owned_instances", "cost_scope")},
    })
    proof = {
        "schema": B.RESOURCE_SCHEMA, "resource_kind": "remote_gpu", **common, **capture,
        "raw_gpu_receipt_path": str(raw_path.resolve()), "raw_gpu_receipt_sha256": B.sha256(raw_path),
        "budget_admission_path": str(budget_path.resolve()), "budget_admission_sha256": B.sha256(budget_path),
    }
    verifier = object.__new__(B.SourceBatch)
    verifier.protocol, verifier.protocol_sha, verifier.now = source, source_sha, lambda: now
    verifier._verify_resource(proof)
    B.write_new(output / "resource.json", proof)
    B.write_new(output / "issuance.json", {
        "schema": ISSUANCE_SCHEMA,
        "authorization_path": str(Path(authorization_path).resolve()),
        "authorization_sha256_lf": auth_sha,
        "execution_protocol_sha256_lf": authorization["execution_protocol"]["sha256_lf"],
        "historical_source_protocol_sha256_lf": source_sha,
        "historical_source_deadline_utc": authorization["historical_source_deadline_utc"],
        "authorization_deadline_utc": authorization["authorization_deadline_utc"],
        "resource_id": resource_id,
        "resource_proof_sha256": B.sha256(output / "resource.json"),
        "start_observation_sha256": B.sha256(start_path),
        "scope": authorization["scope"],
        "source_experiment_extended": False,
        "gpu_query_performed_by_issuer": False,
        "vm_mutation_performed_by_issuer": False,
    })
    return proof


def verify_issuance(authorization_path: Path, issuance_path: Path, resource_path: Path,
                    *, now: datetime, require_current: bool = True,
                    repo: Path = AUTH.ROOT) -> dict[str, Any]:
    authorization = AUTH.validate(
        authorization_path, repo=repo, now=now, require_current=require_current
    )
    issuance = B.read(issuance_path)
    expected = {
        "schema": ISSUANCE_SCHEMA,
        "authorization_path": str(Path(authorization_path).resolve()),
        "authorization_sha256_lf": AUTH._sha(authorization_path, lf=True),
        "execution_protocol_sha256_lf": authorization["execution_protocol"]["sha256_lf"],
        "historical_source_protocol_sha256_lf": authorization["source_protocol"]["sha256_lf"],
        "historical_source_deadline_utc": authorization["historical_source_deadline_utc"],
        "authorization_deadline_utc": authorization["authorization_deadline_utc"],
        "resource_id": authorization["resource_id"],
        "resource_proof_sha256": B.sha256(resource_path),
        "start_observation_sha256": issuance.get("start_observation_sha256"),
        "scope": authorization["scope"],
        "source_experiment_extended": False,
        "gpu_query_performed_by_issuer": False,
        "vm_mutation_performed_by_issuer": False,
    }
    if (
        issuance != expected
        or not isinstance(issuance["start_observation_sha256"], str)
        or AUTH.HEX64.fullmatch(issuance["start_observation_sha256"]) is None
    ):
        raise ValueError("resource issuance가 새 승인·자원 원문과 다르다")
    resource = B.read(resource_path)
    if (resource.get("protocol_sha256") != authorization["source_protocol"]["sha256_lf"]
            or B._stamp(resource.get("expires_at_utc"))
            != B._stamp(authorization["authorization_deadline_utc"])):
        raise ValueError("resource proof가 새 승인 기한과 과거 source 계보를 함께 보존하지 않는다")
    return issuance


def verify_job_binding(authorization_path: Path, issuance_path: Path, resource_path: Path,
                       job_path: Path, *, now: datetime,
                       repo: Path = AUTH.ROOT) -> dict[str, Any]:
    """실행 직전에 job이 승인된 protocol·checkpoint·resource만 쓰는지 확인한다."""
    authorization = AUTH.validate(authorization_path, repo=repo, now=now)
    verify_issuance(
        authorization_path, issuance_path, resource_path, now=now, repo=repo
    )
    job = B.read(job_path)
    protocol = job.get("protocol")
    resource = job.get("resource_proof")
    checkpoint = job.get("checkpoint")
    expected_protocol = {
        "path": str(authorization["_execution_path"].resolve()),
        "sha256_lf": authorization["execution_protocol"]["sha256_lf"],
    }
    expected_resource = {
        "path": str(Path(resource_path).resolve()),
        "sha256": B.sha256(resource_path),
    }
    catalog = {row["logical_id"]: row for row in authorization["checkpoint_catalog"]}
    if (protocol != expected_protocol or resource != expected_resource
            or not isinstance(checkpoint, dict)):
        raise ValueError("job이 새 자원 승인 protocol·resource에 결합되지 않았다")
    logical_id = checkpoint.get("logical_id")
    expected_checkpoint = catalog.get(logical_id)
    if (expected_checkpoint is None
            or checkpoint.get("sha256") != expected_checkpoint["sha256"]
            or not isinstance(checkpoint.get("path"), str)
            or not Path(checkpoint["path"]).as_posix().endswith(
                expected_checkpoint["isaaclab_relative"]
            )):
        raise ValueError("job checkpoint가 새 승인 catalog와 다르다")
    return {"authorization": authorization, "issuance": B.read(issuance_path), "job": job}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authorization", "start", "provider", "gpu", "processes", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    issue(args.authorization, args.start, args.provider, args.gpu, args.processes, args.output,
          now=datetime.now(UTC))
    print((args.output / "resource.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
