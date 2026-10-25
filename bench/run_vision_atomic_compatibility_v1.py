"""기존 seed 2208 checkpoint로 atomic smoke와 full record만 실행한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD
import run_vision_brev_batch_v1 as SOURCE
import run_vision_brev_preflight_v1 as PREFLIGHT
import vision_brev_render_backend_v5 as BACKEND  # pyright: ignore[reportMissingImports]
import vision_metadata_lifecycle_v1 as METADATA

SCHEMA = "vision_atomic_compatibility_protocol_v1"
CUMULATIVE_SCHEMA = "vision_atomic_compatibility_protocol_v2"
MIXED_LIFECYCLE_SCHEMA = "vision_atomic_compatibility_protocol_v3"
REGISTRATION_SCHEMA = "vision_atomic_checkpoint_registration_v1"
APPROVAL_SCHEMA = "vision_atomic_compatibility_approval_v1"
INTENT_SCHEMA = "vision_atomic_compatibility_intent_v1"
RAW_SCHEMA = "vision_atomic_compatibility_raw_v1"
RECEIPT_SCHEMA = "vision_atomic_compatibility_receipt_v1"
FAILURE_SCHEMA = "vision_atomic_compatibility_failure_v1"
INVALID_SCHEMA = "vision_atomic_compatibility_invalid_v1"
STAGE_ORDER = ("atomic_smoke", "atomic_record")
MODES = {
    "atomic_smoke": RECORD.RENDER_SYNC_SMOKE_MODE,
    "atomic_record": RECORD.RECORD_MODE,
}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
SLUG = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


def _now() -> datetime:
    return datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10


def _sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}는 JSON object여야 한다")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("timestamp가 문자열이 아니다")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp에 timezone이 없다")
    return parsed.astimezone(timezone.utc)  # noqa: UP017 - Python 3.10


def _reference(repo: Path, path: Path, *, lf: bool = False) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_nlink != 1:
        raise ValueError("참조 원문이 단일 regular file이 아니다")
    return {"path": resolved.relative_to(repo).as_posix(), "sha256_lf" if lf else "sha256": _sha(resolved, lf=lf)}


def _regular(repo: Path, reference: Any, anchor: str, *, lf: bool = False) -> Path:
    key = "sha256_lf" if lf else "sha256"
    if not isinstance(reference, Mapping) or set(reference) != {"path", key}:
        raise ValueError(f"{anchor} 참조 필드가 다르다")
    relative, expected = reference["path"], reference[key]
    if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
            or not isinstance(expected, str) or HEX64.fullmatch(expected) is None):
        raise ValueError(f"{anchor} 참조가 잘못됐다")
    path = (repo / relative).resolve(strict=True)
    path.relative_to(repo)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or _sha(path, lf=lf) != expected:
        raise ValueError(f"{anchor} 원문 SHA가 다르다")
    return path


def _finite_decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError("누적 비용 숫자 형식이 다르다")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("누적 비용 숫자를 읽을 수 없다") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("누적 비용이 음수이거나 유한하지 않다")
    return parsed


def _validate_prior_stops(
    resources: Mapping[str, Any], source: Mapping[str, Any], repo: Path,
) -> tuple[datetime, set[str]]:
    references = resources.get("prior_stop_receipts")
    if not isinstance(references, list) or not references:
        raise ValueError("누적 비용에 STOPPED 원문 목록이 없다")
    total = Decimal(0)
    previous_end: datetime | None = None
    seen_runs: set[str] = set()
    seen_leases: set[str] = set()
    seen_hashes: set[str] = set()
    for reference in references:
        path = _regular(repo, reference, "prior stop receipt")
        stop = _read(path)
        identity, cost = stop.get("execution_identity"), stop.get("cost_estimate")
        if (stop.get("schema") != "v04_owned_vm_stop_receipt_v1" or stop.get("status") != "STOPPED"
                or not isinstance(stop.get("owned_instance"), Mapping)
                or stop["owned_instance"].get("id") != source["resources"].get("owned_instance_id")
                or not isinstance(identity, Mapping) or not isinstance(cost, Mapping)):
            raise ValueError("누적 비용 STOPPED 인스턴스 또는 계약이 다르다")
        run, lease = identity.get("run_id"), identity.get("lease_id")
        if (not isinstance(run, str) or SLUG.fullmatch(run) is None or run == source["protocol_id"]
                or not isinstance(lease, str) or not lease.strip()
                or run in seen_runs or lease in seen_leases or reference["sha256"] in seen_hashes):
            raise ValueError("누적 비용 lifecycle identity가 비었거나 중복됐다")
        for field in ("protocol_sha256", "resource_proof_sha256"):
            if not isinstance(identity.get(field), str) or HEX64.fullmatch(identity[field]) is None:
                raise ValueError("누적 비용 lifecycle SHA가 다르다")
        if (not isinstance(cost.get("resource_proof"), Mapping)
                or cost["resource_proof"].get("sha256") != identity["resource_proof_sha256"]
                or cost.get("estimate_basis") != "compute_start_to_stopped_observation_wall_clock"
                or cost.get("storage_charges_included") is not False
                or cost.get("provider_billing_equivalence_claimed") is not False):
            raise ValueError("누적 비용 근거 범위가 다르다")
        start = _stamp(cost.get("compute_started_at_utc"))
        end = _stamp(cost.get("stopped_observed_at_utc"))
        if end < start or (previous_end is not None and start < previous_end):
            raise ValueError("누적 비용 lifecycle 시각이 겹치거나 순서가 다르다")
        elapsed = _finite_decimal(cost.get("estimated_wall_clock_seconds"))
        rate = _finite_decimal(cost.get("hourly_compute_rate_usd"))
        amount = _finite_decimal(cost.get("estimated_compute_cost_usd"))
        if (rate <= 0 or abs(elapsed - Decimal(str((end - start).total_seconds()))) > Decimal("0.000001")
                or amount != (elapsed * rate / Decimal(3600)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)):
            raise ValueError("누적 비용 시각·요율·금액 계산이 다르다")
        total += amount
        previous_end = end
        seen_runs.add(run)
        seen_leases.add(lease)
        seen_hashes.add(reference["sha256"])
    if total != _finite_decimal(resources["prior_lifecycle_compute_usd"]):
        raise ValueError("누적 비용 합계가 STOPPED 기록과 다르다")
    assert previous_end is not None
    return previous_end, seen_leases


def _native_stop_fingerprint(receipt: Mapping[str, Any], owner_name: str, *, nested: bool) -> str:
    """원문 보관 위치가 달라도 같은 native Stop 요청을 구별한다."""
    if (receipt.get("command") != ["brev", "stop", owner_name]
            or type(receipt.get("exit_code")) is not int or receipt["exit_code"] != 0):
        raise ValueError("누적 비용 native Stop command 또는 exit code가 다르다")
    body = {key: receipt.get(key) for key in ("command", "request_before_utc", "response_after_utc", "exit_code")}
    if _stamp(body["response_after_utc"]) < _stamp(body["request_before_utc"]):
        raise ValueError("누적 비용 native Stop 시각이 역전됐다")
    for stream in ("stdout", "stderr"):
        value = receipt.get(stream) if nested else None
        digest = value.get("sha256") if isinstance(value, Mapping) else receipt.get(f"{stream}_sha256")
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise ValueError("누적 비용 native Stop stream SHA가 다르다")
        body[f"{stream}_sha256"] = digest
    return _digest(body)


def _validate_mixed_prior_stops(
    resources: Mapping[str, Any], source: Mapping[str, Any], repo: Path,
) -> tuple[datetime, set[str]]:
    """기존 STOPPED 기록과 proof 발급 전 종료 기록을 별도 계약으로 누적한다."""
    references = resources.get("prior_stop_receipts")
    contract = resources.get("metadata_lifecycle_contract")
    if (not isinstance(references, list) or not references or not isinstance(contract, Mapping)
            or set(contract) != {"owner", "allowed_inspection_script_sha256", "runs"}):
        raise ValueError("누적 비용 metadata 계약 필드가 다르다")
    owner, allowed, registrations = contract["owner"], contract["allowed_inspection_script_sha256"], contract["runs"]
    if (not isinstance(owner, Mapping) or set(owner) != {"id", "name", "instance_type"}
            or owner["id"] != source["resources"].get("owned_instance_id")
            or not all(isinstance(value, str) and value for value in owner.values())
            or not isinstance(allowed, list) or not allowed or len(allowed) != len(set(allowed))
            or any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in allowed)
            or not isinstance(registrations, Mapping) or not registrations):
        raise ValueError("누적 비용 metadata owner 또는 등록 집합이 다르다")
    for run, binding in registrations.items():
        if (not isinstance(run, str) or SLUG.fullmatch(run) is None or run == source["protocol_id"]
                or not isinstance(binding, Mapping)
                or set(binding) != {"envelope_sha256", "start_receipt_sha256", "stop_receipt_sha256"}
                or any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in binding.values())):
            raise ValueError("누적 비용 metadata run 결합이 다르다")
    previous_end: datetime | None = None
    total, normal_total = Decimal(0), Decimal(0)
    normal_refs: list[Mapping[str, Any]] = []
    seen_runs: set[str] = set()
    seen_raw: set[str] = set()
    seen_starts: set[str] = set()
    seen_stops: set[str] = set()
    seen_native_stops: set[str] = set()
    seen_metadata: set[str] = set()
    for reference in references:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise TypeError("누적 비용 lifecycle ref가 다르다")
        lexical = repo / reference["path"]
        METADATA._under_repo(repo, lexical, "prior lifecycle")
        path = _regular(repo, reference, "prior lifecycle")
        raw = _read(path)
        if raw.get("schema") == "vision_metadata_lifecycle_v1":
            validated = METADATA.validate_metadata_lifecycle(
                path, repo=repo, expected_owner=owner,
                expected_hourly_rate_usd=_finite_decimal(resources["compute_quote_usd_per_h"]),
                allowed_inspection_script_sha256=set(allowed),
            )
            run = validated.run_id
            start, end = _stamp(validated.started_at_utc), _stamp(validated.stopped_at_utc)
            amount = validated.cost_usd
            evidence = raw["evidence"]
            start_sha, stop_sha = evidence["start_receipt"]["sha256"], evidence["stop_receipt"]["sha256"]
            if (registrations.get(run) != {"envelope_sha256": reference["sha256"],
                    "start_receipt_sha256": start_sha, "stop_receipt_sha256": stop_sha}
                    or start_sha in seen_starts or stop_sha in seen_stops):
                raise ValueError("누적 비용 metadata 원문이 미등록이거나 중복됐다")
            seen_starts.add(start_sha)
            seen_stops.add(stop_sha)
            seen_metadata.add(run)
            native_stop = _read(METADATA._reference(repo, evidence["stop_receipt"], "metadata stop"))
            stop_fingerprint = _native_stop_fingerprint(native_stop, owner["name"], nested=False)
        elif raw.get("schema") == "v04_owned_vm_stop_receipt_v1":
            identity, cost = raw.get("execution_identity"), raw.get("cost_estimate")
            if (not isinstance(identity, Mapping) or not isinstance(cost, Mapping)
                    or not isinstance(raw.get("owned_instance"), Mapping)
                    or raw["owned_instance"].get("name") != owner["name"]
                    or not isinstance(raw.get("stop_call"), Mapping)):
                raise ValueError("누적 비용 normal lifecycle owner 또는 native Stop이 다르다")
            run = identity.get("run_id")
            start, end = _stamp(cost.get("compute_started_at_utc")), _stamp(cost.get("stopped_observed_at_utc"))
            if _finite_decimal(cost.get("hourly_compute_rate_usd")) != _finite_decimal(resources["compute_quote_usd_per_h"]):
                raise ValueError("누적 비용 normal lifecycle 요율이 등록값과 다르다")
            amount = _finite_decimal(cost.get("estimated_compute_cost_usd"))
            normal_total += amount
            normal_refs.append(reference)
            stop_fingerprint = _native_stop_fingerprint(raw["stop_call"], owner["name"], nested=True)
            if _stamp(raw["stop_call"]["response_after_utc"]) > end:
                raise ValueError("누적 비용 normal STOPPED 관측이 Stop 응답보다 빠르다")
        else:
            raise ValueError("누적 비용 lifecycle schema가 다르다")
        if (not isinstance(run, str) or SLUG.fullmatch(run) is None or run == source["protocol_id"]
                or run in seen_runs or reference["sha256"] in seen_raw or stop_fingerprint in seen_native_stops
                or end < start or (previous_end is not None and start < previous_end)):
            raise ValueError("누적 비용 lifecycle이 중복되거나 시각이 겹친다")
        seen_runs.add(run)
        seen_raw.add(reference["sha256"])
        seen_native_stops.add(stop_fingerprint)
        total += amount
        previous_end = end
    if seen_metadata != set(registrations) or total != _finite_decimal(resources["prior_lifecycle_compute_usd"]):
        raise ValueError("누적 비용 metadata 등록 또는 합계가 다르다")
    prior_leases: set[str] = set()
    if normal_refs:
        normal_resources = {**resources, "prior_stop_receipts": normal_refs, "prior_lifecycle_compute_usd": str(normal_total)}
        _, prior_leases = _validate_prior_stops(normal_resources, source, repo)
    assert previous_end is not None
    return previous_end, prior_leases


def _validate_lifecycle_history(
    protocol: Mapping[str, Any], source: Mapping[str, Any], repo: Path,
) -> tuple[datetime, set[str]]:
    if protocol["schema"] == MIXED_LIFECYCLE_SCHEMA:
        return _validate_mixed_prior_stops(protocol["resources"], source, repo)
    return _validate_prior_stops(protocol["resources"], source, repo)


def validate_protocol(value: Mapping[str, Any], repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {
        "schema", "protocol_id", "registered_before_measurement", "source_anchor",
        "source_protocol", "historical_checkpoint", "stages", "resources", "expected",
        "controller_sources_lf",
    }
    if set(value) != required or value.get("schema") not in (SCHEMA, CUMULATIVE_SCHEMA, MIXED_LIFECYCLE_SCHEMA) or value.get("registered_before_measurement") is not True:
        raise ValueError("compatibility protocol 필드 또는 등록 상태가 다르다")
    protocol = dict(value)
    if not isinstance(protocol.get("protocol_id"), str) or SLUG.fullmatch(protocol["protocol_id"]) is None:
        raise ValueError("compatibility protocol_id가 잘못됐다")
    if not isinstance(protocol.get("source_anchor"), str) or HEX40.fullmatch(protocol["source_anchor"]) is None:
        raise ValueError("source anchor가 잘못됐다")
    source_path = _regular(repo, protocol["source_protocol"], "source protocol", lf=True)
    source = SOURCE.validate_protocol(_read(source_path))
    if source["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("compatibility/source protocol_id가 다르다")

    historical = protocol.get("historical_checkpoint")
    if not isinstance(historical, Mapping) or set(historical) != {
        "seed", "raw", "receipt", "checkpoint_sha256", "prior_protocol_sha256_lf"
    }:
        raise ValueError("historical checkpoint 계약 필드가 다르다")
    if historical["seed"] != 2208 or not isinstance(historical["checkpoint_sha256"], str) \
            or HEX64.fullmatch(historical["checkpoint_sha256"]) is None \
            or not isinstance(historical["prior_protocol_sha256_lf"], str) \
            or HEX64.fullmatch(historical["prior_protocol_sha256_lf"]) is None:
        raise ValueError("historical checkpoint identity가 다르다")
    raw_path = _regular(repo, historical["raw"], "historical raw")
    receipt_path = _regular(repo, historical["receipt"], "historical receipt")
    raw, receipt = _read(raw_path), _read(receipt_path)
    checkpoint = raw.get("artifact", {}).get("checkpoint") if isinstance(raw.get("artifact"), Mapping) else None
    execution = raw.get("artifact", {}).get("execution") if isinstance(raw.get("artifact"), Mapping) else None
    snapshot = execution.get("source_before") if isinstance(execution, Mapping) else None
    prior_protocol = snapshot.get("protocol") if isinstance(snapshot, Mapping) else None
    if (raw.get("stage") != "train_none20" or raw.get("job_id") != "preflight-train-s2208"
            or raw.get("request", {}).get("seed") != 2208 or not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != historical["checkpoint_sha256"]
            or not isinstance(prior_protocol, Mapping)
            or prior_protocol.get("sha256_lf") != historical["prior_protocol_sha256_lf"]
            or receipt.get("raw_sha256") != historical["raw"]["sha256"]
            or receipt.get("job_id") != raw.get("job_id") or receipt.get("stage") != raw.get("stage")):
        raise ValueError("historical raw/receipt/checkpoint 결합이 다르다")

    training_plan = source.get("training_plan")
    if not isinstance(training_plan, list) or len(training_plan) != 6:
        raise ValueError("source training plan은 보존된 6행이어야 한다")
    stages = protocol.get("stages")
    if not isinstance(stages, list) or [row.get("stage") for row in stages if isinstance(row, Mapping)] != list(STAGE_ORDER):
        raise ValueError("compatibility stage 순서가 다르다")
    quote = float(source["resources"]["compute_quote_usd_per_h"])
    seen: set[str] = set()
    source_job_ids = {
        row.get("job_id")
        for row in source.get("source_action_order", [])
        if isinstance(row, Mapping)
    }
    for row in stages:
        if not isinstance(row, Mapping) or set(row) != {
            "stage", "job_id", "checkpoint_logical_id", "mode", "max_wall_s", "max_compute_usd"
        }:
            raise ValueError("compatibility stage 필드가 다르다")
        if (row["mode"] != MODES[row["stage"]] or row["max_wall_s"] != 900
                or not isinstance(row["job_id"], str) or SLUG.fullmatch(row["job_id"]) is None
                or row["job_id"] in seen or row["job_id"] in source_job_ids
                or row["job_id"] == "preflight-smoke-s2208"
                or not isinstance(row["checkpoint_logical_id"], str)
                or SLUG.fullmatch(row["checkpoint_logical_id"]) is None
                or isinstance(row["max_compute_usd"], bool)
                or not isinstance(row["max_compute_usd"], (int, float))
                or abs(float(row["max_compute_usd"]) - quote * 900 / 3600.0) > 1e-9):
            raise ValueError("compatibility stage 값이 고정 계약과 다르다")
        seen.add(row["job_id"])

    resources = protocol.get("resources")
    cumulative = protocol["schema"] in (CUMULATIVE_SCHEMA, MIXED_LIFECYCLE_SCHEMA)
    prior_field = "prior_stop_receipts" if cumulative else "prior_stop_receipt"
    resource_fields = {
        "prior_lifecycle_compute_usd", "fresh_lifecycle_budget_usd", "compute_quote_usd_per_h",
        "total_program_budget_usd", prior_field, "deadline_utc", "fresh_admission_max_age_s",
    }
    if protocol["schema"] == MIXED_LIFECYCLE_SCHEMA:
        resource_fields.add("metadata_lifecycle_contract")
    if not isinstance(resources, Mapping) or set(resources) != resource_fields:
        raise ValueError("compatibility resource 계약 필드가 다르다")
    numbers = (resources["prior_lifecycle_compute_usd"], resources["fresh_lifecycle_budget_usd"],
               resources["total_program_budget_usd"])
    if cumulative:
        _validate_lifecycle_history(protocol, source, repo)
    else:
        stop_path = _regular(repo, resources["prior_stop_receipt"], "prior stop receipt")
        stop = _read(stop_path)
        stopped_cost = stop.get("cost_estimate")
        if (resources["prior_lifecycle_compute_usd"] != 6.610104
                or stop.get("schema") != "v04_owned_vm_stop_receipt_v1" or stop.get("status") != "STOPPED"
                or not isinstance(stopped_cost, Mapping)
                or float(stopped_cost.get("estimated_compute_cost_usd", -1))
                != float(resources["prior_lifecycle_compute_usd"])):
            raise ValueError("compatibility 비용·deadline 계약이 다르다")
    if (any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
                   or float(item) < 0 for item in numbers)
            or resources["total_program_budget_usd"] != 20.0
            or abs(float(resources["prior_lifecycle_compute_usd"])
                   + float(resources["fresh_lifecycle_budget_usd"])
                   - float(resources["total_program_budget_usd"])) > 1e-9
            or float(source["resources"]["whole_batch_budget_usd"])
            != float(resources["fresh_lifecycle_budget_usd"])
            or float(resources["compute_quote_usd_per_h"]) != quote
            or resources["deadline_utc"] != source["resources"]["deadline_utc"]
            or resources["fresh_admission_max_age_s"] != 30
            or sum(float(row["max_compute_usd"]) for row in stages)
            > float(resources["fresh_lifecycle_budget_usd"]) + 1e-9):
        raise ValueError("compatibility 비용·deadline 계약이 다르다")
    _stamp(resources["deadline_utc"])
    if protocol.get("expected") != {
        "new_training_actions": 0, "api_calls": 0, "smoke_captures": 4,
        "record_frames": 625, "stage_order": list(STAGE_ORDER),
    }:
        raise ValueError("compatibility 기대 실행량이 다르다")

    closure = protocol.get("controller_sources_lf")
    required_sources = {
        "bench/run_vision_atomic_compatibility_v1.py",
        "tests/test_run_vision_atomic_compatibility_v1.py",
        *BACKEND.SOURCE_FILES,
    }
    if protocol["schema"] == MIXED_LIFECYCLE_SCHEMA:
        required_sources.update({"bench/vision_metadata_lifecycle_v1.py", "tests/test_vision_metadata_lifecycle_v1.py",
                                 "tests/test_vision_mixed_lifecycle_v1.py"})
    if not isinstance(closure, Mapping) or not required_sources <= set(closure):
        raise ValueError("compatibility controller/backend source 결합이 빠졌다")
    source_map = source.get("source_sha256_lf")
    if not isinstance(source_map, Mapping) or not BACKEND.SOURCE_FILES <= set(source_map):
        raise ValueError("referenced source protocol에 atomic backend source 결합이 빠졌다")
    for relative, expected in closure.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(expected, str) or HEX64.fullmatch(expected) is None):
            raise ValueError("controller source 결합 형식이 잘못됐다")
        path = repo / relative
        if not path.is_file() or path.is_symlink() or _sha(path, lf=True) != expected:
            raise ValueError(f"controller source LF SHA가 다르다: {relative}")
        anchored = SOURCE._git(repo, "show", f"{protocol['source_anchor']}:{relative}", check=False)
        anchored_lf = anchored.stdout.encode().replace(b"\r\n", b"\n")
        if anchored.returncode != 0 or hashlib.sha256(anchored_lf).hexdigest() != expected:
            raise ValueError(f"source anchor의 controller source blob이 다르다: {relative}")
    for relative in BACKEND.SOURCE_FILES:
        if source_map.get(relative) != closure.get(relative):
            raise ValueError(f"source protocol/controller closure SHA가 다르다: {relative}")
    return protocol, source


class AtomicCompatibilityController:
    def __init__(
        self, repo: Path, protocol_path: Path, protocol_sha256_lf: str, resource_path: Path,
        resource_sha256: str, isaaclab: Path, python: Path, *,
        backend_factory: Callable[..., Any] = BACKEND.AtomicVisionBrevBackend.from_runtime,
        now: Callable[[], datetime] = _now,
    ):
        self.repo = Path(repo).resolve(strict=True)
        self.protocol_path = Path(protocol_path).resolve(strict=True)
        if self.protocol_path.is_symlink() or _sha(self.protocol_path, lf=True) != protocol_sha256_lf:
            raise ValueError("compatibility protocol 실제 LF SHA가 다르다")
        self.protocol_sha = protocol_sha256_lf
        self.protocol, self.source = validate_protocol(_read(self.protocol_path), self.repo)
        relative = self.protocol_path.relative_to(self.repo).as_posix()
        SOURCE._git(self.repo, "ls-files", "--error-unmatch", "--", relative)
        if SOURCE._git(self.repo, "status", "--porcelain", "--", relative).stdout.strip():
            raise ValueError("compatibility protocol의 index 또는 working tree가 clean하지 않다")
        committed = SOURCE._git(self.repo, "show", f"HEAD:{relative}").stdout.encode().replace(b"\r\n", b"\n")
        if hashlib.sha256(committed).hexdigest() != self.protocol_sha:
            raise ValueError("현재 HEAD의 compatibility protocol blob SHA가 다르다")
        self.source_path = _regular(self.repo, self.protocol["source_protocol"], "source protocol", lf=True)
        self.resource_path = Path(resource_path).resolve(strict=True)
        if self.resource_path.is_symlink() or _sha(self.resource_path) != resource_sha256:
            raise ValueError("resource proof 원문 SHA가 다르다")
        self.resource_sha = resource_sha256
        self.now, self.isaaclab, self.python = now, Path(isaaclab), Path(python)
        self.backend_factory = backend_factory
        self.runtime = self.repo / ".triage-runtime/vision-brev" / self.protocol["protocol_id"]
        self._batches: dict[tuple[str, str], Any] = {}
        self.root = self.runtime / "atomic-compatibility-v1"
        self.registration_path = self.root / "checkpoint.registration.json"
        self.approval_path = self.root / "approval.json"

    def _verify_inputs(self) -> None:
        if _sha(self.protocol_path, lf=True) != self.protocol_sha:
            raise ValueError("실행 중 compatibility protocol이 바뀌었다")
        validate_protocol(_read(self.protocol_path), self.repo)
        if self.resource_path.is_symlink() or not self.resource_path.is_file() \
                or _sha(self.resource_path) != self.resource_sha:
            raise ValueError("실행 중 resource proof 원문이 바뀌었다")

    def _batch(self, path: Path, sha256: str, *, at: datetime | None = None) -> Any:
        resolved = Path(path).resolve(strict=True)
        key = (str(resolved), sha256)
        cached = self._batches.get(key)
        if cached is not None:
            return cached
        clock = self.now if at is None else (lambda: at)
        batch = SOURCE.SourceBatch(
            self.repo, self.source_path, self.protocol["source_protocol"]["sha256_lf"],
            self.protocol["source_anchor"], resolved, sha256, self.isaaclab, self.python,
            backend_factory=self.backend_factory, now=clock,
        )
        self._batches[key] = batch
        return batch

    def _current_batch(self) -> Any:
        return self._batch(self.resource_path, self.resource_sha)

    def _stage_batch(self, intent: Mapping[str, Any]) -> Any:
        reference = intent.get("resource_proof")
        if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
            raise ValueError("stage intent에 resource proof 결합이 없다")
        path, sha256 = reference["path"], reference["sha256"]
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(sha256, str) or HEX64.fullmatch(sha256) is None):
            raise ValueError("stage resource proof 참조 형식이 잘못됐다")
        resource = _read(Path(path))
        issued = _stamp(resource.get("issued_at_utc"))
        return self._batch(Path(path), sha256, at=issued)

    def _stage(self, index: int) -> Path:
        return self.root / "stages" / f"{index:02d}_{STAGE_ORDER[index]}"

    def _state(self, index: int) -> str:
        folder = self._stage(index)
        for name, state in (("failure", "failed"), ("invalid", "invalid"), ("receipt", "complete"), ("intent", "pending")):
            if (folder / f"{name}.json").exists():
                return state
        return "absent"

    def _registration(self) -> dict[str, Any]:
        body = _read(self.registration_path)
        if set(body) != {"schema", "protocol_sha256_lf", "checkpoint", "historical_raw", "historical_receipt",
                         "lifecycle_binding", "registered_at_utc"} \
                or body.get("schema") != REGISTRATION_SCHEMA or body.get("protocol_sha256_lf") != self.protocol_sha:
            raise ValueError("checkpoint registration 필드가 다르다")
        checkpoint = body.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"path", "sha256", "seed"} \
                or checkpoint.get("sha256") != self.protocol["historical_checkpoint"]["checkpoint_sha256"] \
                or checkpoint.get("seed") != 2208:
            raise ValueError("등록 checkpoint identity가 다르다")
        path = Path(str(checkpoint.get("path", "")))
        if not path.is_absolute() or not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1 \
                or _sha(path) != checkpoint["sha256"]:
            raise ValueError("등록 checkpoint 원문 SHA가 다르다")
        if body["historical_raw"] != self.protocol["historical_checkpoint"]["raw"] \
                or body["historical_receipt"] != self.protocol["historical_checkpoint"]["receipt"]:
            raise ValueError("registration의 historical 증거 결합이 다르다")
        binding = body.get("lifecycle_binding")
        expected_binding_keys = {
            "resource_id", "gpu_uuid", "compute_started_at_utc", "source_protocol_sha256_lf",
            "compute_quote_usd_per_h", "cost_scope", "max_owned_instances",
        }
        if (not isinstance(binding, Mapping) or set(binding) != expected_binding_keys
                or not isinstance(binding.get("resource_id"), str)
                or not isinstance(binding.get("gpu_uuid"), str)
                or binding.get("source_protocol_sha256_lf") != self.protocol["source_protocol"]["sha256_lf"]
                or binding.get("compute_quote_usd_per_h") != self.protocol["resources"]["compute_quote_usd_per_h"]
                or binding.get("cost_scope") != self.source["resources"]["cost_scope"]
                or binding.get("max_owned_instances") != 1):
            raise ValueError("registration lifecycle binding이 다르다")
        _stamp(binding["compute_started_at_utc"])
        return body

    def _lifecycle_binding(self, batch: Any) -> dict[str, Any]:
        return {
            "resource_id": batch.resource["resource_id"],
            "gpu_uuid": batch.resource["gpu_uuid"],
            "compute_started_at_utc": batch.resource["compute_started_at_utc"],
            "source_protocol_sha256_lf": self.protocol["source_protocol"]["sha256_lf"],
            "compute_quote_usd_per_h": self.protocol["resources"]["compute_quote_usd_per_h"],
            "cost_scope": self.source["resources"]["cost_scope"],
            "max_owned_instances": 1,
        }

    def _verify_lifecycle_binding(self, batch: Any) -> None:
        if self._registration()["lifecycle_binding"] != self._lifecycle_binding(batch):
            raise ValueError("resource proof가 등록 lifecycle과 다르다")

    def _approval(self) -> dict[str, Any]:
        body = _read(self.approval_path)
        if (set(body) != {"schema", "protocol_sha256_lf", "registration_sha256", "decision", "actor", "approved_stages", "approved_at_utc"}
                or body.get("schema") != APPROVAL_SCHEMA or body.get("protocol_sha256_lf") != self.protocol_sha
                or body.get("registration_sha256") != _sha(self.registration_path)
                or body.get("decision") != "approve" or body.get("approved_stages") != list(STAGE_ORDER)
                or not isinstance(body.get("actor"), str) or not body["actor"].strip()):
            raise ValueError("compatibility approval 결합이 다르다")
        _stamp(body["approved_at_utc"])
        return body

    def register_checkpoint(self, checkpoint: Path) -> dict[str, Any]:
        self._verify_inputs()
        if self.registration_path.exists():
            body = self._registration()
            return {"status": "checkpoint_registered", "cached": True, "registration": body}
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError("registration 없는 compatibility runtime은 재사용하지 않는다")
        batch = self._current_batch()
        backend = batch._backend()
        with backend.action_admission():
            self._fresh_resource(batch, self.now().astimezone(timezone.utc))  # noqa: UP017
            batch._verify_resource(_read(self.resource_path))
            path = Path(checkpoint).absolute()
            if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
                raise ValueError("remote checkpoint가 단일 regular file로 존재하지 않는다")
            expected = self.protocol["historical_checkpoint"]["checkpoint_sha256"]
            if _sha(path) != expected:
                raise ValueError("remote checkpoint SHA가 historical raw와 다르다")
            body = {
                "schema": REGISTRATION_SCHEMA, "protocol_sha256_lf": self.protocol_sha,
                "checkpoint": {"path": str(path), "sha256": expected, "seed": 2208},
                "historical_raw": self.protocol["historical_checkpoint"]["raw"],
                "historical_receipt": self.protocol["historical_checkpoint"]["receipt"],
                "lifecycle_binding": self._lifecycle_binding(batch),
                "registered_at_utc": self.now().astimezone(timezone.utc).isoformat(),  # noqa: UP017
            }
            _write_new(self.registration_path, body)
        return {"status": "checkpoint_registered", "cached": False, "registration": body}

    def approve(self, actor: str) -> dict[str, Any]:
        self._verify_inputs()
        self._registration()
        if not actor.strip():
            raise ValueError("승인 actor가 비었다")
        if self.approval_path.exists():
            body = self._approval()
            if body["actor"] != actor:
                raise ValueError("기존 승인 actor와 다르다")
            return {"status": "approved", "cached": True, "approval": body}
        body = {
            "schema": APPROVAL_SCHEMA, "protocol_sha256_lf": self.protocol_sha,
            "registration_sha256": _sha(self.registration_path), "decision": "approve",
            "actor": actor, "approved_stages": list(STAGE_ORDER),
            "approved_at_utc": self.now().astimezone(timezone.utc).isoformat(),  # noqa: UP017
        }
        _write_new(self.approval_path, body)
        return {"status": "approved", "cached": False, "approval": body}

    def _fresh_resource(self, batch: Any, current: datetime) -> None:
        resource = batch._verify_resource(_read(batch.resource_path))
        if self.protocol["schema"] in (CUMULATIVE_SCHEMA, MIXED_LIFECYCLE_SCHEMA):
            previous_end, prior_leases = _validate_lifecycle_history(self.protocol, self.source, self.repo)
            if _stamp(resource["compute_started_at_utc"]) < previous_end:
                raise ValueError("새 lifecycle 시작이 이전 STOPPED 시각보다 빠르다")
            if resource["lease_id"] in prior_leases:
                raise ValueError("새 lifecycle lease가 이전 STOPPED lease와 같다")
        issued = _stamp(resource["issued_at_utc"])
        age = (current - issued).total_seconds()
        if not 0 <= age <= self.protocol["resources"]["fresh_admission_max_age_s"]:
            raise ValueError("compatibility resource admission이 30초 범위를 벗어났다")

    def _lifecycle_cost(self, batch: Any, current: datetime) -> float:
        started = _stamp(batch.resource["compute_started_at_utc"])
        seconds = (current - started).total_seconds()
        if seconds < 0:
            raise ValueError("compute 시작 전 lifecycle 비용을 계산할 수 없다")
        return seconds * float(self.protocol["resources"]["compute_quote_usd_per_h"]) / 3600.0

    def _smoke_job(self, batch: Any, checkpoint: Mapping[str, Any], *, create: bool, current: datetime) -> dict[str, Any]:
        builder = object.__new__(PREFLIGHT.PreflightLedger)
        builder.repo, builder.runtime = self.repo, batch.runtime
        builder.source, builder.anchor = self.source, self.protocol["source_anchor"]
        builder.resource, builder.resource_sha = batch.resource, batch.resource_sha
        builder.deadline = _stamp(self.protocol["resources"]["deadline_utc"])
        builder.expires = _stamp(batch.resource["expires_at_utc"])
        builder.folder = self.root
        row = self.protocol["stages"][0]
        builder.smoke = {"job_id": row["job_id"], "checkpoint_logical_id": row["checkpoint_logical_id"]}
        return PREFLIGHT.PreflightLedger.smoke_job(builder, checkpoint, create=create, now=current)

    def _guard_record_inputs(self, batch: Any, *, create: bool) -> None:
        inputs = batch.runtime / "record-inputs" / STAGE_ORDER[1]
        if create:
            if inputs.exists() or inputs.is_symlink():
                raise ValueError("intent 없는 record input orphan은 보완하지 않는다")
            return
        if not inputs.is_dir() or inputs.is_symlink():
            raise ValueError("완료 record input 폴더가 없다")
        for name in ("protocol.json", "admission.json", "resource_parent.json"):
            path = inputs / name
            if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
                raise ValueError(f"완료 record input 원문이 없다: {name}")

    def _job(self, batch: Any, index: int, checkpoint: Mapping[str, Any], *, create: bool, current: datetime) -> dict[str, Any]:
        row = self.protocol["stages"][index]
        if index == 0:
            job = self._smoke_job(batch, checkpoint, create=create, current=current)
        else:
            self._guard_record_inputs(batch, create=create)
            action = {
                "action_id": row["stage"], "job_id": row["job_id"],
                "checkpoint_logical_id": row["checkpoint_logical_id"], "source_job_id": "preflight-train-s2208",
            }
            artifact = {"checkpoint": dict(checkpoint)}
            job, _ = batch._record_job(action, artifact)
        if job["mode"] != row["mode"]:
            raise ValueError("생성 job mode가 stage 계약과 다르다")
        return job

    def _finish(self, batch: Any, index: int, job: Mapping[str, Any], capture: Mapping[str, Any], references: Mapping[str, Any], *, recovered: bool) -> dict[str, Any]:
        folder = self._stage(index)
        intent = _read(folder / "intent.json")
        raw = {
            "schema": RAW_SCHEMA, "stage": STAGE_ORDER[index], "job_id": job["job_id"],
            "job_sha256": _digest(dict(job)), "capture": dict(capture), "references": dict(references),
        }
        raw_path = folder / "raw.json"
        if raw_path.exists():
            if _read(raw_path) != raw:
                raise ValueError("기존 compatibility raw가 검증 결과와 다르다")
        else:
            _write_new(raw_path, raw)
        attestation_ref = references.get("attestation")
        attestation_path = _regular(self.repo, attestation_ref, "capture attestation")
        finished = _stamp(_read(attestation_path)["attested_at_utc"])
        started = _stamp(intent["started_at_utc"])
        elapsed = (finished - started).total_seconds()
        row = self.protocol["stages"][index]
        if not 0 <= elapsed <= row["max_wall_s"] or finished > _stamp(self.protocol["resources"]["deadline_utc"]):
            raise ValueError("compatibility stage 완료 시각이 상한을 벗어났다")
        cost_after = self._lifecycle_cost(batch, finished)
        if cost_after > float(self.protocol["resources"]["fresh_lifecycle_budget_usd"]) + 1e-9:
            raise ValueError("compatibility fresh lifecycle 비용 상한을 넘었다")
        recovery_path = folder / "cache_recovery.json"
        if recovered and not recovery_path.is_file():
            raise ValueError("cache recovery 경로 증거가 없다")
        if not recovered and recovery_path.exists():
            raise ValueError("새 실행 stage에 cache recovery 증거가 있다")
        receipt = {
            "schema": RECEIPT_SCHEMA, "stage": STAGE_ORDER[index], "job_id": job["job_id"],
            "intent_sha256": _sha(folder / "intent.json"), "raw_sha256": _sha(raw_path),
            "started_at_utc": intent["started_at_utc"], "finished_at_utc": finished.isoformat(),
            "elapsed_wall_s": elapsed, "cost_before_usd": intent["cost_before_usd"],
            "cost_after_usd": cost_after, "recovered_from_cache": recovered,
            "cache_recovery_sha256": _sha(recovery_path) if recovered else None,
        }
        _write_new(folder / "receipt.json", receipt)
        self._verify_stage(index)
        return {"status": "advanced" if index == 0 else "complete", "stage": STAGE_ORDER[index],
                "completed_stages": index + 1, "remaining_stages": 1 - index,
                "new_training_actions": 0, "api_calls": 0, "recovered_from_cache": recovered}

    def _verify_stage(self, index: int) -> dict[str, Any]:
        folder = self._stage(index)
        intent, raw, receipt = (_read(folder / f"{name}.json") for name in ("intent", "raw", "receipt"))
        batch = self._stage_batch(intent)
        self._verify_lifecycle_binding(batch)
        row = self.protocol["stages"][index]
        registration, approval = self._registration(), self._approval()
        precheck = folder / "job.precheck.json"
        if not precheck.is_file() or precheck.is_symlink() or precheck.stat().st_nlink != 1:
            raise ValueError("저장 job precheck 원문이 없다")
        stored_job = _read(precheck)
        if intent.get("job_sha256") != _digest(stored_job):
            raise ValueError("저장 job precheck SHA가 intent와 다르다")
        RECORD.validate_job(precheck, now=_stamp(intent.get("started_at_utc")))
        checkpoint = registration["checkpoint"]
        job = self._job(batch, index, checkpoint, create=False, current=_stamp(intent["started_at_utc"]))
        if stored_job != job:
            raise ValueError("저장 job precheck 의미가 재구성 job과 다르다")
        expected_intent = {
            "schema": INTENT_SCHEMA, "stage": row["stage"], "ordinal": index, "job_id": row["job_id"],
            "protocol_sha256_lf": self.protocol_sha, "source_protocol_sha256_lf": self.protocol["source_protocol"]["sha256_lf"],
            "resource_sha256": batch.resource_sha,
            "resource_proof": {"path": str(batch.resource_path), "sha256": batch.resource_sha},
            "registration_sha256": _sha(self.registration_path),
            "approval_sha256": _sha(self.approval_path), "job_sha256": _digest(job),
            "started_at_utc": intent.get("started_at_utc"), "cost_before_usd": intent.get("cost_before_usd"),
            "reserved_max_wall_s": row["max_wall_s"], "reserved_max_compute_usd": row["max_compute_usd"],
        }
        if intent != expected_intent or approval["approved_stages"] != list(STAGE_ORDER):
            raise ValueError("compatibility intent/approval 결합이 다르다")
        references = raw.get("references")
        if (raw.get("schema") != RAW_SCHEMA or raw.get("stage") != row["stage"]
                or raw.get("job_id") != row["job_id"] or raw.get("job_sha256") != _digest(job)
                or not isinstance(references, Mapping)):
            raise ValueError("compatibility raw identity가 다르다")
        backend = batch._backend()
        if backend.capture_references(job) != references:
            raise ValueError("backend cache reference가 compatibility raw와 다르다")
        capture = backend.smoke(job) if index == 0 else backend.record(job)
        if capture != raw.get("capture"):
            raise ValueError("backend cache capture가 compatibility raw와 다르다")
        receipt_keys = {
            "schema", "stage", "job_id", "intent_sha256", "raw_sha256", "started_at_utc",
            "finished_at_utc", "elapsed_wall_s", "cost_before_usd", "cost_after_usd",
            "recovered_from_cache", "cache_recovery_sha256",
        }
        if (set(receipt) != receipt_keys or receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("stage") != row["stage"]
                or receipt.get("job_id") != row["job_id"]
                or receipt.get("intent_sha256") != _sha(folder / "intent.json")
                or receipt.get("raw_sha256") != _sha(folder / "raw.json")):
            raise ValueError("compatibility receipt SHA 결합이 다르다")
        numeric_fields = ("elapsed_wall_s", "cost_before_usd", "cost_after_usd")
        if any(
            isinstance(receipt.get(field), bool)
            or not isinstance(receipt.get(field), (int, float))
            or not math.isfinite(float(receipt[field]))
            for field in numeric_fields
        ):
            raise ValueError("compatibility receipt 수치가 잘못됐다")
        started, finished = _stamp(receipt["started_at_utc"]), _stamp(receipt["finished_at_utc"])
        attestation = _regular(self.repo, references.get("attestation"), "capture attestation")
        attested = _stamp(_read(attestation)["attested_at_utc"])
        elapsed = (finished - started).total_seconds()
        expected_before = self._lifecycle_cost(batch, started)
        expected_after = self._lifecycle_cost(batch, finished)
        prior_after = None if index == 0 else _read(self._stage(index - 1) / "receipt.json")["cost_after_usd"]
        action_cost = elapsed * float(self.protocol["resources"]["compute_quote_usd_per_h"]) / 3600.0
        if (receipt["started_at_utc"] != intent["started_at_utc"] or finished != attested
                or receipt["elapsed_wall_s"] != elapsed or not 0 <= elapsed <= row["max_wall_s"]
                or receipt["cost_before_usd"] != expected_before
                or receipt["cost_after_usd"] != expected_after
                or (prior_after is not None and expected_before + 1e-9 < float(prior_after))
                or action_cost > float(row["max_compute_usd"]) + 1e-9
                or expected_after > float(self.protocol["resources"]["fresh_lifecycle_budget_usd"]) + 1e-9
                or finished > _stamp(self.protocol["resources"]["deadline_utc"])):
            raise ValueError("compatibility receipt 시간 결합이 다르다")
        recovery_path = folder / "cache_recovery.json"
        if receipt["recovered_from_cache"] is True:
            if not recovery_path.is_file() or recovery_path.is_symlink():
                raise ValueError("cache recovery receipt 원문이 없다")
            recovery = _read(recovery_path)
            expected_recovery = {
                "schema": "vision_atomic_compatibility_cache_recovery_v1", "stage": row["stage"],
                "intent_sha256": _sha(folder / "intent.json"), "job_sha256": _digest(job),
                "execution_sha256": _sha(batch.runtime / "record" / str(job["job_id"]) / "execution.json"),
                "result_sha256": _sha(batch.runtime / "record" / str(job["job_id"]) / "output/result.json"),
            }
            if recovery != expected_recovery or receipt["cache_recovery_sha256"] != _sha(recovery_path):
                raise ValueError("cache recovery receipt 결합이 다르다")
        elif receipt["recovered_from_cache"] is not False \
                or receipt["cache_recovery_sha256"] is not None or recovery_path.exists():
            raise ValueError("새 실행/cache recovery 표지가 다르다")
        return raw

    def _settle(self) -> dict[str, Any] | None:
        states = [self._state(index) for index in range(2)]
        for index, state in enumerate(states):
            if state in {"failed", "invalid"}:
                return {"status": "HOLD", "reason": f"compatibility_stage_{state}", "stage": STAGE_ORDER[index]}
            if state == "complete":
                self._verify_stage(index)
            elif state == "pending":
                folder = self._stage(index)
                intent = _read(folder / "intent.json")
                batch = self._stage_batch(intent)
                registration = self._registration()
                job = self._job(batch, index, registration["checkpoint"], create=False, current=_stamp(intent["started_at_utc"]))
                record_stage = batch.runtime / "record" / str(job["job_id"])
                if not (record_stage / "execution.json").is_file() or not (record_stage / "output/result.json").is_file():
                    return {"status": "HOLD", "reason": "compatibility_stage_pending", "stage": STAGE_ORDER[index]}
                try:
                    backend = batch._backend()
                    references = backend.capture_references(job)
                    capture = backend.smoke(job) if index == 0 else backend.record(job)
                    recovery = {
                        "schema": "vision_atomic_compatibility_cache_recovery_v1", "stage": STAGE_ORDER[index],
                        "intent_sha256": _sha(folder / "intent.json"), "job_sha256": _digest(job),
                        "execution_sha256": _sha(record_stage / "execution.json"),
                        "result_sha256": _sha(record_stage / "output/result.json"),
                    }
                    recovery_path = folder / "cache_recovery.json"
                    if recovery_path.exists():
                        if recovery_path.is_symlink() or _read(recovery_path) != recovery:
                            raise ValueError("기존 cache recovery 증거가 다르다")
                    else:
                        _write_new(recovery_path, recovery)
                    return self._finish(batch, index, job, capture, references, recovered=True)
                except (OSError, TypeError, ValueError) as error:
                    _write_new(folder / "invalid.json", {"schema": INVALID_SCHEMA, "stage": STAGE_ORDER[index], "reason": "cached_evidence_invalid", "message": str(error)})
                    return {"status": "HOLD", "reason": "compatibility_stage_invalid", "stage": STAGE_ORDER[index]}
        if states == ["complete", "complete"]:
            return {"status": "complete", "completed_stages": 2, "remaining_stages": 0,
                    "new_training_actions": 0, "api_calls": 0, "cached": True}
        if states[0] != "complete" and states[1] != "absent":
            raise ValueError("smoke 완료 전 record stage가 존재한다")
        return None

    def advance_one(self) -> dict[str, Any]:
        self._verify_inputs()
        self._registration()
        self._approval()
        settled = self._settle()
        if settled is not None:
            return settled
        index = 0 if self._state(0) == "absent" else 1
        row = self.protocol["stages"][index]
        batch = self._current_batch()
        self._verify_lifecycle_binding(batch)
        batch.verify()
        if index == 1:
            self._guard_record_inputs(batch, create=True)
        backend = batch._backend()
        with backend.action_admission():
            current = self.now().astimezone(timezone.utc)  # noqa: UP017
            self._fresh_resource(batch, current)
            deadline = _stamp(self.protocol["resources"]["deadline_utc"])
            if current + timedelta(seconds=row["max_wall_s"]) > deadline:
                return {"status": "HOLD", "reason": "compatibility_window_insufficient", "stage": row["stage"]}
            cost = self._lifecycle_cost(batch, current)
            if index > 0:
                prior_after = _read(self._stage(index - 1) / "receipt.json")["cost_after_usd"]
                if isinstance(prior_after, bool) or not isinstance(prior_after, (int, float)) \
                        or cost + 1e-9 < float(prior_after):
                    raise ValueError("resource lifecycle 비용 시계가 이전 stage보다 되돌아갔다")
            remaining = sum(float(candidate["max_compute_usd"]) for candidate in self.protocol["stages"][index:])
            if cost + remaining > float(self.protocol["resources"]["fresh_lifecycle_budget_usd"]) + 1e-9:
                return {"status": "HOLD", "reason": "compatibility_budget_insufficient", "stage": row["stage"]}
            checkpoint = self._registration()["checkpoint"]
            job = self._job(batch, index, checkpoint, create=True, current=current)
            folder = self._stage(index)
            folder.mkdir(parents=True, exist_ok=False)
            precheck = folder / "job.precheck.json"
            _write_new(precheck, job)
            RECORD.validate_job(precheck, now=current)
            intent = {
                "schema": INTENT_SCHEMA, "stage": row["stage"], "ordinal": index, "job_id": row["job_id"],
                "protocol_sha256_lf": self.protocol_sha,
                "source_protocol_sha256_lf": self.protocol["source_protocol"]["sha256_lf"],
                "resource_sha256": batch.resource_sha,
                "resource_proof": {"path": str(batch.resource_path), "sha256": batch.resource_sha},
                "registration_sha256": _sha(self.registration_path),
                "approval_sha256": _sha(self.approval_path), "job_sha256": _digest(job),
                "started_at_utc": current.isoformat(), "cost_before_usd": cost,
                "reserved_max_wall_s": row["max_wall_s"], "reserved_max_compute_usd": row["max_compute_usd"],
            }
            _write_new(folder / "intent.json", intent)
            try:
                capture = backend.smoke(job) if index == 0 else backend.record(job)
                if capture.get("status") == "HOLD":
                    return {"status": "HOLD", "reason": "compatibility_stage_pending", "stage": row["stage"]}
                references = backend.capture_references(job)
                return self._finish(batch, index, job, capture, references, recovered=False)
            except BaseException as error:  # noqa: BLE001 - terminal 증거를 남기고 원인을 보존한다.
                _write_new(folder / "failure.json", {"schema": FAILURE_SCHEMA, "stage": row["stage"], "reason": type(error).__name__, "message": str(error)})
                return {"status": "HOLD", "reason": "compatibility_stage_failed", "stage": row["stage"], "detail": type(error).__name__}

    def verify(self) -> dict[str, Any]:
        self._verify_inputs()
        registration = self._registration() if self.registration_path.exists() else None
        approval = self._approval() if self.approval_path.exists() else None
        states = [self._state(index) for index in range(2)]
        for index, state in enumerate(states):
            if state == "complete":
                self._verify_stage(index)
        status = "complete" if states == ["complete", "complete"] else "HOLD"
        return {"status": status, "protocol_id": self.protocol["protocol_id"], "stages": states,
                "checkpoint_registered": registration is not None, "approved": approval is not None,
                "new_training_actions": 0, "api_calls": 0,
                "prior_lifecycle_compute_usd": self.protocol["resources"]["prior_lifecycle_compute_usd"]}


def _controller(args: argparse.Namespace, backend_factory: Callable[..., Any] | None) -> AtomicCompatibilityController:
    return AtomicCompatibilityController(
        args.repo, args.protocol, args.protocol_sha256_lf, args.resource_proof, args.resource_sha256,
        args.isaaclab, args.python, backend_factory=backend_factory or BACKEND.AtomicVisionBrevBackend.from_runtime,
    )


def main(argv: list[str] | None = None, *, backend_factory: Callable[..., Any] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register_checkpoint", "approve", "advance_one", "verify"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--resource-proof", required=True, type=Path)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--actor")
    args = parser.parse_args(argv)
    controller = _controller(args, backend_factory)
    if args.command == "register_checkpoint":
        if args.checkpoint is None:
            parser.error("register_checkpoint에는 --checkpoint가 필요하다")
        result = controller.register_checkpoint(args.checkpoint)
    elif args.command == "approve":
        if args.actor is None:
            parser.error("approve에는 --actor가 필요하다")
        result = controller.approve(args.actor)
    elif args.command == "advance_one":
        result = controller.advance_one()
    else:
        result = controller.verify()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
