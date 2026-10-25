"""V04 source 원장 뒤에 attested render-smoke와 readiness 생성을 잇는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import record_vision_evidence_v1 as RECORD
import run_vision_brev_batch_v1 as SOURCE
import vision_brev_render_backend_v2 as RENDER  # pyright: ignore[reportMissingImports]

SCHEMA = "vision_brev_render_protocol_v2"
METHOD = "context_then_capture_source_seal_v1"
GATE_POLICY = "none_walks_f1_unhealthy_v1"
READINESS_SCOPE = "registered_renderer_combination_only"
SMOKE_INTENT_SCHEMA = "vision_brev_render_action_intent_v2"
SMOKE_RAW_SCHEMA = "vision_brev_render_action_raw_v2"
SMOKE_RECEIPT_SCHEMA = "vision_brev_render_action_receipt_v2"
MATERIALS_SCHEMA = "vision_brev_render_materials_v2"
PROTOCOL_KEYS = {
    "schema", "protocol_id", "registered_before_measurement", "method", "source_gate_policy",
    "readiness_scope", "source_protocol", "smoke_plan", "smoke_limits",
}
REQUIRED_CAPTURE_SOURCES = frozenset({
    "bench/run_vision_brev_render_v2.py",
    "bench/run_vision_brev_batch_v1.py",
    "bench/record_vision_evidence_v1.py",
    "bench/vision_play_worker_v1.py",
    "evals/brev_recovery_backend_v4.py",
    "evals/vision_brev_backend_v1.py",
    "evals/vision_brev_render_backend_v2.py",
    "evals/vision_remote_gpu_v1.py",
    "bench/vision_render_context_v1.py",
    "bench/vision_render_attestation_v1.py",
    "evals/vision_selection_sources_v1.py",
    "evals/vision_render_readiness_v2.py",
})


def _source_reference(value: Mapping[str, Any]) -> dict[str, str]:
    if (set(value) != {"path", "sha256_lf"} or not isinstance(value.get("path"), str)
            or Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts
            or not isinstance(value.get("sha256_lf"), str)
            or SOURCE.HEX64.fullmatch(value["sha256_lf"]) is None):
        raise ValueError("source protocol reference가 잘못됐다")
    return dict(value)


def _smoke_plan(source_protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for record in source_protocol["source_action_order"]:
        if record["kind"] != "record":
            continue
        smoke_id = f"smoke-{record['job_id']}"
        rows.append({
            "action_id": smoke_id, "job_id": smoke_id, "record_job_id": record["job_id"],
            "source_job_id": record["source_job_id"],
            "checkpoint_logical_id": record["checkpoint_logical_id"],
        })
    if len(rows) != 4 or len({row["action_id"] for row in rows}) != 4:
        raise ValueError("smoke plan은 네 record에서 정확히 한 행씩 파생돼야 한다")
    return rows


def build_protocol(source_protocol: Mapping[str, Any], source_reference: Mapping[str, Any]) -> dict[str, Any]:
    """검증된 source protocol에서 별도 render protocol을 결정적으로 만든다."""
    source = SOURCE.validate_protocol(source_protocol)
    reference = _source_reference(source_reference)
    quote = float(source["resources"]["compute_quote_usd_per_h"])
    protocol = {
        "schema": SCHEMA, "protocol_id": source["protocol_id"],
        "registered_before_measurement": True, "method": METHOD,
        "source_gate_policy": GATE_POLICY, "readiness_scope": READINESS_SCOPE,
        "source_protocol": reference, "smoke_plan": _smoke_plan(source),
        "smoke_limits": {"max_wall_s": 900, "max_compute_usd": quote * 900 / 3600.0},
    }
    return validate_protocol(protocol, source, reference)


def validate_protocol(value: Mapping[str, Any], source_protocol: Mapping[str, Any],
                      source_reference: Mapping[str, Any]) -> dict[str, Any]:
    source = SOURCE.validate_protocol(source_protocol)
    reference = _source_reference(source_reference)
    if set(value) != PROTOCOL_KEYS:
        raise ValueError("render protocol 필드가 다르다")
    protocol = dict(value)
    limits = protocol.get("smoke_limits")
    quote = float(source["resources"]["compute_quote_usd_per_h"])
    if (protocol.get("schema") != SCHEMA or protocol.get("protocol_id") != source["protocol_id"]
            or protocol.get("registered_before_measurement") is not True
            or protocol.get("method") != METHOD or protocol.get("source_gate_policy") != GATE_POLICY
            or protocol.get("readiness_scope") != READINESS_SCOPE
            or protocol.get("source_protocol") != reference
            or protocol.get("smoke_plan") != _smoke_plan(source)
            or not isinstance(limits, dict) or set(limits) != {"max_wall_s", "max_compute_usd"}
            or limits["max_wall_s"] != 900
            or isinstance(limits["max_compute_usd"], bool)
            or not isinstance(limits["max_compute_usd"], (int, float))
            or not math.isfinite(float(limits["max_compute_usd"]))
            or abs(float(limits["max_compute_usd"]) - quote * 900 / 3600.0) > 1e-9):
        raise ValueError("render protocol 값이 source 파생 계약과 다르다")
    source_limits = source["resources"]["action_limits"]
    source_reserve = sum(float(source_limits[row["kind"]]["max_compute_usd"])
                         for row in source["source_action_order"])
    total = source_reserve + len(protocol["smoke_plan"]) * float(limits["max_compute_usd"])
    if total > float(source["resources"]["whole_batch_budget_usd"]) + 1e-9:
        raise ValueError("source와 smoke 최대 예약 합이 whole batch 상한을 넘는다")
    return protocol


class RenderBatch(SOURCE.SourceBatch):
    """동결 source action을 보존하고 검증된 smoke 네 건만 뒤에 추가한다."""

    def __init__(
        self, repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
        render_protocol_path: Path, render_protocol_sha256_lf: str, source_anchor: str,
        resource_path: Path, resource_sha256: str, lab: Path, python: Path, *,
        backend_factory: Callable[..., Any] = RENDER.AttestedVisionBrevBackend.from_runtime,
        seed_guard: Callable[..., dict[str, Any]] = SOURCE.SEEDS.assert_fresh_seeds,
        now: Callable[[], datetime] = lambda: datetime.now(SOURCE.UTC_TZ),
    ):
        self.render_protocol_path = Path(render_protocol_path).resolve()
        if (not self.render_protocol_path.is_file() or self.render_protocol_path.is_symlink()
                or SOURCE.HEX64.fullmatch(str(render_protocol_sha256_lf)) is None
                or SOURCE.sha256(self.render_protocol_path, lf=True) != render_protocol_sha256_lf):
            raise ValueError("render protocol 실제 LF SHA가 다르다")
        self.render_protocol_sha = render_protocol_sha256_lf
        raw_source = SOURCE.read(Path(source_protocol_path))
        source_reference = {
            "path": Path(source_protocol_path).resolve().relative_to(Path(repo).resolve()).as_posix(),
            "sha256_lf": source_protocol_sha256_lf,
        }
        self.render_protocol = validate_protocol(
            SOURCE.read(self.render_protocol_path), raw_source, source_reference,
        )
        super().__init__(
            repo, source_protocol_path, source_protocol_sha256_lf, source_anchor,
            resource_path, resource_sha256, lab, python, backend_factory=backend_factory,
            seed_guard=seed_guard, now=now,
        )

    def _verify_bound_inputs(self) -> None:
        super()._verify_bound_inputs()
        if SOURCE.sha256(self.render_protocol_path, lf=True) != self.render_protocol_sha:
            raise ValueError("실행 중 render protocol LF SHA가 바뀌었다")

    def _verify_source(self) -> None:
        super()._verify_source()
        if not hasattr(self, "render_protocol_path"):
            return
        relative = self.render_protocol_path.relative_to(self.repo).as_posix()
        blob = SOURCE._git(self.repo, "show", f"{self.anchor}:{relative}").stdout.encode().replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != self.render_protocol_sha:
            raise ValueError("source anchor의 render protocol blob SHA가 다르다")
        source_map = self.protocol["source_sha256_lf"]
        recording_map = self.protocol["recording"]["source_files_lf"]
        if not REQUIRED_CAPTURE_SOURCES <= set(source_map) or not REQUIRED_CAPTURE_SOURCES <= set(recording_map):
            raise ValueError("render controller/backend/context/attester/helper source 결합이 빠졌다")
        names = sorted({relative, self.protocol_path.relative_to(self.repo).as_posix(),
                        *source_map, *recording_map})
        SOURCE._git(self.repo, "ls-files", "--error-unmatch", "--", *names)
        if SOURCE._git(self.repo, "status", "--porcelain", "--", *names).stdout.strip():
            raise ValueError("render 결합 파일의 index 또는 working tree가 clean 상태가 아니다")
        head = SOURCE._git(self.repo, "rev-parse", "HEAD").stdout.strip()
        for name in names:
            anchor_blob = SOURCE._git(self.repo, "rev-parse", f"{self.anchor}:{name}").stdout.strip()
            if not anchor_blob == SOURCE._git(self.repo, "rev-parse", f"{head}:{name}").stdout.strip() \
                    == SOURCE._git(self.repo, "rev-parse", f":{name}").stdout.strip():
                raise ValueError("render 결합 blob이 source anchor 뒤 변경됐다")

    def verify(self) -> dict[str, Any]:
        result = super().verify()
        result.update({"render_protocol_sha256_lf": self.render_protocol_sha,
                       "smoke_actions": len(self.render_protocol["smoke_plan"])})
        return result

    def _source_gate(self) -> dict[str, Any]:
        evaluations = [self._raw(index) for index in range(4, 6)]
        for raw in evaluations:
            reports, gate = raw["evaluation"]["reports"], raw["source_gate"]
            if not SOURCE.RECOVERY.V.R1.U.walks(reports["NONE"]):
                return {"status": "HOLD", "reason": "source_none_not_walking"}
            outcome = gate.get("outcome")
            if outcome == "recovered":
                return {"status": "source_failure_absent", "reason": "f1_recovered"}
            if outcome != "not_recovered":
                return {"status": "HOLD", "reason": "source_evaluation_unknown"}
        return {"status": "source_failure_confirmed"}

    def _smoke_folder(self, index: int) -> Path:
        return self.runtime / "render-actions" / f"{index:02d}"

    def _smoke_job(self, row: Mapping[str, str], *, create: bool) -> tuple[dict[str, Any], dict[str, str]]:
        record = next(action for action in self.protocol["source_action_order"]
                      if action["kind"] == "record" and action["job_id"] == row["record_job_id"])
        train_index = next(index for index, action in enumerate(self.protocol["source_action_order"])
                           if action["kind"] == "train" and action["job_id"] == row["source_job_id"])
        artifact = self._raw(train_index)["artifact"]
        checkpoint = artifact["checkpoint"]
        inputs = self.runtime / "render-inputs" / row["job_id"]
        protocol_path, admission_path = inputs / "protocol.json", inputs / "admission.json"
        if create:
            if inputs.exists():
                raise FileExistsError("intent 없는 기존 smoke input은 재사용하지 않는다")
            inputs.mkdir(parents=True, exist_ok=False)
        elif not inputs.is_dir() or inputs.is_symlink():
            raise ValueError("완료 smoke input 폴더가 없다")
        protocol_body = {
            "schema": RECORD.PROTOCOL_SCHEMA, "method": RECORD.METHOD,
            "mode": RECORD.RENDER_SYNC_SMOKE_MODE, "profile": RECORD.fixed_profile(),
            "source_commit": self.anchor,
            "source_files_lf": self.protocol["recording"]["source_files_lf"],
            "checkpoint_sha256": checkpoint["sha256"],
            "resource": {"resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"]},
        }
        if create:
            SOURCE.write_new(protocol_path, protocol_body)
        elif (not protocol_path.is_file() or protocol_path.is_symlink() or protocol_path.stat().st_nlink != 1
              or SOURCE.read(protocol_path) != protocol_body):
            raise ValueError("기존 smoke protocol이 다르다")
        output = self.runtime / "record" / row["job_id"] / "output"
        job = {
            "schema": RECORD.JOB_SCHEMA, "method": RECORD.METHOD,
            "mode": RECORD.RENDER_SYNC_SMOKE_MODE, "job_id": row["job_id"],
            "profile": RECORD.fixed_profile(),
            "source": {"repo_root": str(self.repo), "commit": self.anchor,
                       "files_lf": self.protocol["recording"]["source_files_lf"]},
            "protocol": {"path": str(protocol_path), "sha256": SOURCE.sha256(protocol_path)},
            "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256"],
                           "logical_id": record["checkpoint_logical_id"]},
            "resource": {"resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"]},
            "remote_gpu_admission": {"resource_parent_sha256": self.resource_sha,
                                     "live_lease_proof_sha256": self.resource["raw_gpu_receipt_sha256"]},
            "output_directory": str(output),
        }
        existing = None
        if not create:
            if (not admission_path.is_file() or admission_path.is_symlink()
                    or admission_path.stat().st_nlink != 1):
                raise ValueError("기존 smoke admission이 없다")
            existing = SOURCE.read(admission_path)
            if not isinstance(existing, dict):
                raise ValueError("기존 smoke admission이 object가 아니다")
        issued_at = existing["issued_at_utc"] if existing else self.now().astimezone(SOURCE.UTC_TZ).isoformat()
        issued = SOURCE._stamp(issued_at)
        expires = SOURCE._stamp(self.resource["expires_at_utc"])
        deadline = SOURCE._stamp(self.protocol["resources"]["deadline_utc"])
        if issued + timedelta(seconds=900) > min(expires, deadline):
            raise ValueError("smoke admission 900초가 lease/deadline 범위를 벗어난다")
        admission = {
            "schema": RECORD.ADMISSION_SCHEMA, "status": "granted", "job_id": row["job_id"],
            "resource_kind": "remote_gpu", "lease_id": self.resource["lease_id"],
            "request_binding": RECORD.request_binding(job), "issued_at_utc": issued_at,
            "expires_at_utc": self.resource["expires_at_utc"],
            "resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"],
            "resource_parent_sha256": self.resource_sha,
            "live_lease_proof_sha256": self.resource["raw_gpu_receipt_sha256"],
        }
        if not create:
            if existing != admission:
                raise ValueError("기존 smoke admission이 다르다")
        else:
            SOURCE.write_new(admission_path, admission)
        job["remote_gpu_admission"].update({"path": str(admission_path),
                                             "sha256": SOURCE.sha256(admission_path)})
        return job, {"path": str(self.resource_path), "sha256": self.resource_sha}

    def _smoke_raw(self, index: int) -> dict[str, Any]:
        row = self.render_protocol["smoke_plan"][index]
        folder = self._smoke_folder(index)
        paths = {name: folder / f"{name}.json" for name in ("intent", "raw", "receipt")}
        if any(not path.is_file() or path.is_symlink() for path in paths.values()):
            raise ValueError("완료 smoke action 파일이 없다")
        intent, raw, receipt = (SOURCE.read(paths[name]) for name in ("intent", "raw", "receipt"))
        job, _ = self._smoke_job(row, create=False)
        binding = SOURCE.digest(job)
        limits = self.render_protocol["smoke_limits"]
        expected_intent = {
            "schema": SMOKE_INTENT_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
            "ordinal": index, "source_protocol_sha256_lf": self.protocol_sha,
            "render_protocol_sha256_lf": self.render_protocol_sha,
            "resource_sha256": self.resource_sha, "job_binding_sha256": binding,
            "cost_before_usd": intent.get("cost_before_usd"), "started_at_utc": intent.get("started_at_utc"),
            "reserved_max_wall_s": limits["max_wall_s"],
            "reserved_max_compute_usd": limits["max_compute_usd"],
        }
        if intent != expected_intent:
            raise ValueError("smoke intent 결합이 다르다")
        cached_smoke, cached_references = self._cached_smoke(job)
        if (set(raw) != {"schema", "action_id", "job_id", "job_binding_sha256", "smoke", "references"}
                or raw.get("schema") != SMOKE_RAW_SCHEMA or raw.get("action_id") != row["action_id"]
                or raw.get("job_id") != row["job_id"] or raw.get("job_binding_sha256") != binding
                or cached_smoke != raw.get("smoke") or cached_references != raw.get("references")):
            raise ValueError("smoke raw 원문 또는 backend cache가 다르다")
        required_receipt = {
            "schema", "action_id", "job_id", "source_protocol_sha256_lf", "render_protocol_sha256_lf",
            "resource_sha256", "job_binding_sha256", "intent_sha256", "raw_sha256",
            "started_at_utc", "finished_at_utc", "elapsed_wall_s", "cost_before_usd", "cost_after_usd",
        }
        if (set(receipt) != required_receipt or receipt.get("schema") != SMOKE_RECEIPT_SCHEMA
                or receipt.get("action_id") != row["action_id"] or receipt.get("job_id") != row["job_id"]
                or receipt.get("source_protocol_sha256_lf") != self.protocol_sha
                or receipt.get("render_protocol_sha256_lf") != self.render_protocol_sha
                or receipt.get("resource_sha256") != self.resource_sha
                or receipt.get("job_binding_sha256") != binding
                or receipt.get("intent_sha256") != SOURCE.sha256(paths["intent"])
                or receipt.get("raw_sha256") != SOURCE.sha256(paths["raw"])):
            raise ValueError("smoke receipt SHA 결합이 다르다")
        started, finished = SOURCE._stamp(receipt["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
        elapsed = (finished - started).total_seconds()
        if (receipt["started_at_utc"] != intent["started_at_utc"]
                or receipt["cost_before_usd"] != intent["cost_before_usd"]
                or receipt["elapsed_wall_s"] != elapsed or not 0 <= elapsed <= limits["max_wall_s"]
                or receipt["cost_before_usd"] != self._lifecycle_cost(started)
                or receipt["cost_after_usd"] != self._lifecycle_cost(finished)
                or elapsed * float(self.protocol["resources"]["compute_quote_usd_per_h"]) / 3600.0
                > float(limits["max_compute_usd"]) + 1e-9):
            raise ValueError("smoke receipt 경과 시간·비용이 다르다")
        return raw

    def _cached_smoke(self, job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        references = self._backend().capture_references(job)
        if not isinstance(references, dict):
            raise TypeError("backend capture reference가 object가 아니다")
        values: dict[str, Any] = {}
        for name in ("result", "execution"):
            reference = references.get(name)
            if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
                    or not isinstance(reference.get("path"), str)
                    or not isinstance(reference.get("sha256"), str)
                    or SOURCE.HEX64.fullmatch(reference["sha256"]) is None):
                raise ValueError(f"cached smoke {name} reference가 잘못됐다")
            path = (self.repo / reference["path"]).resolve(strict=True)
            path.relative_to(self.repo)
            if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 \
                    or SOURCE.sha256(path) != reference["sha256"]:
                raise ValueError(f"cached smoke {name} 원문·SHA가 다르다")
            values[name] = SOURCE.read(path)
        payload = {
            "status": "render_sync_smoke_complete", "model_use_ready": False,
            "result": values["result"], "execution": values["execution"],
        }
        return payload, references

    def _cache_files_present(self, job: Mapping[str, Any]) -> bool:
        stage = self.runtime / "record" / str(job["job_id"])
        required = (stage / "job.json", stage / "execution.json", stage / "output/result.json")
        return all(path.is_file() and not path.is_symlink() for path in required)

    def _recover_pending_smoke(self, index: int, row: Mapping[str, str]) -> dict[str, Any] | None:
        folder = self._smoke_folder(index)
        intent_path, raw_path = folder / "intent.json", folder / "raw.json"
        intent = SOURCE.read(intent_path)
        inputs = self.runtime / "render-inputs" / row["job_id"]
        if not (inputs / "protocol.json").is_file() or not (inputs / "admission.json").is_file():
            raise ValueError("smoke intent에 결합된 protocol/admission 원문이 없다")
        job, _ = self._smoke_job(row, create=False)
        binding = SOURCE.digest(job)
        limits = self.render_protocol["smoke_limits"]
        expected = {
            "schema": SMOKE_INTENT_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
            "ordinal": index, "source_protocol_sha256_lf": self.protocol_sha,
            "render_protocol_sha256_lf": self.render_protocol_sha,
            "resource_sha256": self.resource_sha, "job_binding_sha256": binding,
            "cost_before_usd": intent.get("cost_before_usd"), "started_at_utc": intent.get("started_at_utc"),
            "reserved_max_wall_s": limits["max_wall_s"],
            "reserved_max_compute_usd": limits["max_compute_usd"],
        }
        if intent != expected:
            raise ValueError("pending smoke intent 결합이 다르다")
        current = self.now().astimezone(SOURCE.UTC_TZ)
        started = SOURCE._stamp(intent["started_at_utc"])
        expires = SOURCE._stamp(self.resource["expires_at_utc"])
        deadline = SOURCE._stamp(self.protocol["resources"]["deadline_utc"])
        cost_after = self._lifecycle_cost(current)
        if ((current - started).total_seconds() < 0
                or (current - started).total_seconds() > limits["max_wall_s"]
                or current > min(expires, deadline)
                or cost_after > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9):
            return None
        if not self._cache_files_present(job):
            return None
        smoke, references = self._cached_smoke(job)
        expected_raw = {
            "schema": SMOKE_RAW_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
            "job_binding_sha256": binding, "smoke": smoke, "references": references,
        }
        if raw_path.exists():
            if SOURCE.read(raw_path) != expected_raw:
                raise ValueError("pending smoke raw가 검증된 backend cache와 다르다")
        else:
            SOURCE.write_new(raw_path, expected_raw)
        receipt = {
            "schema": SMOKE_RECEIPT_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
            "source_protocol_sha256_lf": self.protocol_sha,
            "render_protocol_sha256_lf": self.render_protocol_sha,
            "resource_sha256": self.resource_sha, "job_binding_sha256": binding,
            "intent_sha256": SOURCE.sha256(intent_path), "raw_sha256": SOURCE.sha256(raw_path),
            "started_at_utc": intent["started_at_utc"], "finished_at_utc": current.isoformat(),
            "elapsed_wall_s": (current - started).total_seconds(),
            "cost_before_usd": intent["cost_before_usd"], "cost_after_usd": cost_after,
        }
        SOURCE.write_new(folder / "receipt.json", receipt)
        self._smoke_raw(index)
        return {"status": "advanced", "action_id": row["action_id"],
                "completed_smokes": index + 1, "remaining_smokes": 3 - index,
                "recovered_from_cached_capture": True}

    def _advance_smoke(self) -> dict[str, Any]:
        for index, row in enumerate(self.render_protocol["smoke_plan"]):
            folder = self._smoke_folder(index)
            intent_path, receipt_path = folder / "intent.json", folder / "receipt.json"
            if receipt_path.exists():
                self._smoke_raw(index)
                continue
            if intent_path.exists():
                recovered = self._recover_pending_smoke(index, row)
                if recovered is not None:
                    return recovered
                return {"status": "HOLD", "reason": "render_action_pending",
                        "action_id": row["action_id"], "completed_smokes": index}
            current = self.now().astimezone(SOURCE.UTC_TZ)
            expires = SOURCE._stamp(self.resource["expires_at_utc"])
            deadline = SOURCE._stamp(self.protocol["resources"]["deadline_utc"])
            limits = self.render_protocol["smoke_limits"]
            cost = self._lifecycle_cost(current)
            if current + timedelta(seconds=limits["max_wall_s"]) > min(expires, deadline):
                return {"status": "HOLD", "reason": "render_authorization_expired",
                        "completed_smokes": index}
            self._verify_resource(SOURCE.read(self.resource_path))
            if cost + float(limits["max_compute_usd"]) \
                    > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9:
                return {"status": "HOLD", "reason": "whole_batch_budget_reached",
                        "completed_smokes": index}
            try:
                job, _ = self._smoke_job(row, create=True)
            except FileExistsError:
                return {"status": "HOLD", "reason": "render_input_orphaned",
                        "action_id": row["action_id"], "completed_smokes": index}
            binding = SOURCE.digest(job)
            folder.mkdir(parents=True, exist_ok=False)
            intent = {
                "schema": SMOKE_INTENT_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
                "ordinal": index, "source_protocol_sha256_lf": self.protocol_sha,
                "render_protocol_sha256_lf": self.render_protocol_sha,
                "resource_sha256": self.resource_sha, "job_binding_sha256": binding,
                "cost_before_usd": cost, "started_at_utc": current.isoformat(),
                "reserved_max_wall_s": limits["max_wall_s"],
                "reserved_max_compute_usd": limits["max_compute_usd"],
            }
            SOURCE.write_new(intent_path, intent)
            smoke = self._backend().smoke(job)
            if smoke.get("status") == "HOLD":
                return {"status": "HOLD", "reason": smoke.get("reason", "render_backend_hold"),
                        "action_id": row["action_id"], "completed_smokes": index}
            references = self._backend().capture_references(job)
            cached_smoke, cached_references = self._cached_smoke(job)
            if smoke != cached_smoke or references != cached_references:
                raise ValueError("새 smoke 결과가 attested cache와 다르다")
            raw = {"schema": SMOKE_RAW_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
                   "job_binding_sha256": binding, "smoke": smoke, "references": references}
            raw_path = folder / "raw.json"
            SOURCE.write_new(raw_path, raw)
            finished = self.now().astimezone(SOURCE.UTC_TZ)
            elapsed = (finished - current).total_seconds()
            cost_after = self._lifecycle_cost(finished)
            if (elapsed < 0 or elapsed > limits["max_wall_s"] or finished > min(expires, deadline)
                    or cost_after > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9):
                SOURCE.write_new(folder / "invalid.json", {
                    "schema": "vision_brev_render_action_invalid_v2", "action_id": row["action_id"],
                    "intent_sha256": SOURCE.sha256(intent_path), "raw_sha256": SOURCE.sha256(raw_path),
                    "finished_at_utc": finished.isoformat(), "cost_after_usd": cost_after,
                })
                return {"status": "HOLD", "reason": "render_action_limit_exceeded",
                        "action_id": row["action_id"], "completed_smokes": index}
            receipt = {
                "schema": SMOKE_RECEIPT_SCHEMA, "action_id": row["action_id"], "job_id": row["job_id"],
                "source_protocol_sha256_lf": self.protocol_sha,
                "render_protocol_sha256_lf": self.render_protocol_sha,
                "resource_sha256": self.resource_sha, "job_binding_sha256": binding,
                "intent_sha256": SOURCE.sha256(intent_path), "raw_sha256": SOURCE.sha256(raw_path),
                "started_at_utc": current.isoformat(), "finished_at_utc": finished.isoformat(),
                "elapsed_wall_s": elapsed, "cost_before_usd": cost, "cost_after_usd": cost_after,
            }
            SOURCE.write_new(receipt_path, receipt)
            self._smoke_raw(index)
            result = {"status": "advanced", "action_id": row["action_id"],
                      "completed_smokes": index + 1, "remaining_smokes": 3 - index}
            if index == 3:
                result["materials"] = self._export_materials()
            return result
        materials = self._export_materials()
        return {"status": "needs_readiness_finalize", "model_use_ready": False,
                "source_data_ready": False, "completed_smokes": 4,
                "lifecycle_cost_status": "open",
                "lifecycle_cost_observed_usd": self._lifecycle_cost(self.now().astimezone(SOURCE.UTC_TZ)),
                "materials": materials}

    def _repo_reference(self, path: Path, anchor: str) -> dict[str, str]:
        candidate = Path(path)
        current = candidate
        while current != self.repo:
            if current.is_symlink() or current.parent == current:
                raise ValueError(f"{anchor} 경로에 symlink가 있거나 저장소를 벗어났다")
            current = current.parent
        actual = candidate.resolve(strict=True)
        actual.relative_to(self.repo)
        if not actual.is_file() or actual.stat().st_nlink != 1:
            raise ValueError(f"{anchor} 원문이 regular file이 아니다")
        return {"path": actual.relative_to(self.repo).as_posix(), "sha256": SOURCE.sha256(actual)}

    def _export_materials(self) -> dict[str, str]:
        import vision_render_readiness_v2 as FINALIZE  # pyright: ignore[reportMissingImports]

        pairs = []
        for index, row in enumerate(self.render_protocol["smoke_plan"]):
            train_index = next(
                position for position, action in enumerate(self.protocol["source_action_order"])
                if action["kind"] == "train" and action["job_id"] == row["source_job_id"]
            )
            artifact = self._raw(train_index)["artifact"]
            record_job_path = self.runtime / "record" / row["record_job_id"] / "job.json"
            if not record_job_path.is_file() or record_job_path.is_symlink():
                raise ValueError("완료 record stage에 기존 job 원문이 없다")
            record_job = SOURCE.read(record_job_path)
            checkpoint = record_job.get("checkpoint")
            if (record_job.get("job_id") != row["record_job_id"]
                    or record_job.get("mode") != RECORD.RECORD_MODE
                    or not isinstance(checkpoint, dict)
                    or checkpoint.get("sha256") != artifact["checkpoint"]["sha256"]
                    or checkpoint.get("logical_id") != row["checkpoint_logical_id"]):
                raise ValueError("기존 record job이 등록 source/checkpoint와 다르다")
            smoke_job, _ = self._smoke_job(row, create=False)
            record_refs = self._backend().capture_references(record_job)
            smoke_refs = self._backend().capture_references(smoke_job)
            material = FINALIZE.build_material(
                self.repo, self.protocol_path, self.protocol_sha, self.resource_path,
                self.runtime, row["record_job_id"], record_refs, smoke_refs,
            )
            FINALIZE.verify_material(self.repo, self.protocol_path, self.protocol_sha, material)
            folder = self._smoke_folder(index)
            pairs.append({
                **row, "output_path": f".triage-runtime/vision-brev/{self.protocol['protocol_id']}"
                                      f"/readiness/{row['record_job_id']}.json",
                "material": material,
                "smoke_journal": {
                    name: self._repo_reference(folder / f"{name}.json", f"smoke {name}")
                    for name in ("intent", "raw", "receipt")
                },
            })
        body = {
            "schema": MATERIALS_SCHEMA,
            "source_protocol": {"path": self.protocol_path.relative_to(self.repo).as_posix(),
                                "sha256_lf": self.protocol_sha},
            "render_protocol": {"path": self.render_protocol_path.relative_to(self.repo).as_posix(),
                                "sha256_lf": self.render_protocol_sha},
            "pairs": pairs,
        }
        path = self.runtime / "materials.json"
        if path.exists():
            if SOURCE.read(path) != body:
                raise ValueError("기존 readiness materials가 현재 완료 원장과 다르다")
        else:
            SOURCE.write_new(path, body)
        return self._repo_reference(path, "readiness materials")

    def _advance_admitted(self) -> dict[str, Any]:
        first_six_complete = all(self._action_folder(index, action).joinpath("receipt.json").is_file()
                                 for index, action in enumerate(self.protocol["source_action_order"][:6]))
        if first_six_complete:
            gate = self._source_gate()
            if gate["status"] != "source_failure_confirmed":
                return {**gate, "model_use_ready": False, "source_data_ready": False}
        result = super()._advance_admitted()
        if result.get("status") != "needs_render_sync_smoke":
            return result
        return self._advance_smoke()

def _portable_reference(repo: Path, value: Any, anchor: str) -> tuple[Path, dict[str, str]]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value.get("path"), str) or Path(value["path"]).is_absolute()
            or ".." in Path(value["path"]).parts or not isinstance(value.get("sha256"), str)
            or SOURCE.HEX64.fullmatch(value["sha256"]) is None):
        raise ValueError(f"{anchor} portable reference가 잘못됐다")
    candidate = repo / value["path"]
    current = candidate
    while current != repo:
        if current.is_symlink() or current.parent == current:
            raise ValueError(f"{anchor} 경로에 symlink가 있거나 저장소를 벗어났다")
        current = current.parent
    path = candidate.resolve(strict=True)
    path.relative_to(repo)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 \
            or SOURCE.sha256(path) != value["sha256"]:
        raise ValueError(f"{anchor} 원문·SHA가 다르다")
    return path, dict(value)


def _path_reference(repo: Path, path: Path, anchor: str) -> dict[str, str]:
    candidate = Path(path)
    current = candidate
    while current != repo:
        if current.is_symlink() or current.parent == current:
            raise ValueError(f"{anchor} 경로에 symlink가 있거나 저장소를 벗어났다")
        current = current.parent
    actual = candidate.resolve(strict=True)
    actual.relative_to(repo)
    if not actual.is_file() or actual.stat().st_nlink != 1:
        raise ValueError(f"{anchor} 원문이 regular file이 아니다")
    return {"path": actual.relative_to(repo).as_posix(), "sha256": SOURCE.sha256(actual)}


def _read_lf_bound_json(path: Path, expected_sha256: str, anchor: str) -> dict[str, Any]:
    candidate = Path(path)
    if (not candidate.is_file() or candidate.is_symlink() or candidate.stat().st_nlink != 1
            or (SOURCE.sha256(candidate) != expected_sha256
                and SOURCE.sha256(candidate, lf=True) != expected_sha256)):
        raise ValueError(f"{anchor} 원문 또는 LF SHA가 다르다")
    return SOURCE.read(candidate)


def _verify_offline_binding(repo: Path, source_path: Path, source_sha: str,
                            render_path: Path, render_sha: str, anchor: str,
                            source: Mapping[str, Any]) -> None:
    if SOURCE._git(repo, "merge-base", "--is-ancestor", anchor, "HEAD", check=False).returncode:
        raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
    required = REQUIRED_CAPTURE_SOURCES
    source_map, recording_map = source["source_sha256_lf"], source["recording"]["source_files_lf"]
    if not required <= set(source_map) or not required <= set(recording_map):
        raise ValueError("offline export source 결합이 빠졌다")
    explicit = {
        source_path.relative_to(repo).as_posix(): source_sha,
        render_path.relative_to(repo).as_posix(): render_sha,
    }
    maps = (source_map, recording_map, explicit)
    for mapping in maps:
        for name, expected in mapping.items():
            path = repo / name
            if (not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1
                    or SOURCE.sha256(path, lf=True) != expected):
                raise ValueError("offline export source LF SHA가 다르다")
    names = sorted(set().union(*(set(mapping) for mapping in maps)))
    SOURCE._git(repo, "ls-files", "--error-unmatch", "--", *names)
    if SOURCE._git(repo, "status", "--porcelain", "--", *names).stdout.strip():
        raise ValueError("offline export 결합 파일이 clean 상태가 아니다")
    head = SOURCE._git(repo, "rev-parse", "HEAD").stdout.strip()
    for name in names:
        anchor_blob = SOURCE._git(repo, "rev-parse", f"{anchor}:{name}").stdout.strip()
        if not anchor_blob == SOURCE._git(repo, "rev-parse", f"{head}:{name}").stdout.strip() \
                == SOURCE._git(repo, "rev-parse", f":{name}").stdout.strip():
            raise ValueError("offline export 결합 blob이 source anchor 뒤 변경됐다")


def _verify_source_journals(repo: Path, runtime: Path, source: Mapping[str, Any], source_sha: str,
                            resource: Mapping[str, Any], resource_sha: str) -> bool:
    quote = float(source["resources"]["compute_quote_usd_per_h"])
    compute_started = SOURCE._stamp(resource["compute_started_at_utc"])
    deadline = SOURCE._stamp(source["resources"]["deadline_utc"])
    lease_expiry = SOURCE._stamp(resource["expires_at_utc"])
    bands_path = repo / source["evaluation"]["bands_path"]
    bands = _read_lf_bound_json(
        bands_path, source["evaluation"]["bands_sha256"], "source evaluation bands",
    )["bands"]
    gates = []
    for index, action in enumerate(source["source_action_order"]):
        folder = runtime / "actions" / f"{index:02d}_{action['action_id']}"
        paths = {name: folder / f"{name}.json" for name in ("intent", "raw", "receipt")}
        if not paths["receipt"].exists():
            return False
        if any(not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1 for path in paths.values()):
            raise ValueError("완료 source journal 원문이 없거나 regular file이 아니다")
        intent, raw, receipt = (SOURCE.read(paths[name]) for name in ("intent", "raw", "receipt"))
        limit = source["resources"]["action_limits"][action["kind"]]
        intent_keys = {
            "schema", "action_id", "kind", "ordinal", "protocol_sha256_lf", "resource_sha256",
            "cost_before_usd", "started_at_utc", "reserved_max_wall_s",
            "reserved_max_compute_usd", "cost_scope",
        }
        receipt_keys = {
            "schema", "action_id", "intent_sha256", "raw_sha256", "cost_after_usd",
            "finished_at_utc",
        }
        if (set(intent) != intent_keys or set(receipt) != receipt_keys
                or intent.get("schema") != SOURCE.INTENT_SCHEMA
                or intent.get("action_id") != action["action_id"]
                or intent.get("kind") != action["kind"] or intent.get("ordinal") != index
                or intent.get("protocol_sha256_lf") != source_sha
                or intent.get("resource_sha256") != resource_sha
                or intent.get("reserved_max_wall_s") != limit["max_wall_s"]
                or intent.get("reserved_max_compute_usd") != limit["max_compute_usd"]
                or intent.get("cost_scope") != source["resources"]["cost_scope"]
                or receipt.get("schema") != SOURCE.RECEIPT_SCHEMA
                or receipt.get("action_id") != action["action_id"]
                or receipt.get("intent_sha256") != SOURCE.sha256(paths["intent"])
                or receipt.get("raw_sha256") != SOURCE.sha256(paths["raw"])
                or raw.get("kind") != action["kind"] or raw.get("action_id") != action["action_id"]
                or raw.get("job_id") != action.get("job_id", action.get("action_id"))):
            raise ValueError("source journal 등록 identity/SHA가 다르다")
        started = SOURCE._stamp(intent["started_at_utc"])
        finished = SOURCE._stamp(receipt["finished_at_utc"])
        elapsed = (finished - started).total_seconds()
        payload = raw.get({"train": "artifact", "evaluate": "evaluation", "record": "record"}[action["kind"]])
        execution = payload.get("execution") if isinstance(payload, dict) else None
        wall_s = execution.get("wall_s") if isinstance(execution, dict) else None
        if (isinstance(wall_s, bool) or not isinstance(wall_s, (int, float))
                or not math.isfinite(wall_s) or wall_s < 0):
            raise ValueError("source action execution wall_s가 잘못됐다")
        cost_before = (started - compute_started).total_seconds() * quote / 3600.0
        cost_after = (finished - compute_started).total_seconds() * quote / 3600.0
        action_cost = float(wall_s) * quote / 3600.0
        if (intent["cost_before_usd"] != cost_before or receipt["cost_after_usd"] != cost_after
                or elapsed < 0 or elapsed > float(limit["max_wall_s"])
                or action_cost > float(limit["max_compute_usd"]) + 1e-9
                or cost_after > float(source["resources"]["whole_batch_budget_usd"]) + 1e-9
                or started < compute_started or finished > min(deadline, lease_expiry)):
            raise ValueError("source action 완료 비용·deadline 결합이 다르다")
        if action["kind"] == "evaluate":
            evaluation, gate = raw.get("evaluation"), raw.get("source_gate")
            if not isinstance(evaluation, dict) or not isinstance(gate, dict) \
                    or set(evaluation.get("reports", {})) != {"NONE", "F1"} \
                    or not isinstance(evaluation.get("execution"), dict) \
                    or evaluation["execution"].get("completed") is not True:
                raise ValueError("source evaluation/gate 원문이 없다")
            measured = SOURCE.RECOVERY.recovery(
                evaluation["reports"]["F1"], evaluation["reports"]["NONE"], bands,
            )
            if gate != measured:
                raise ValueError("source failure gate 재계산이 raw와 다르다")
            if not SOURCE.RECOVERY.V.R1.U.walks(evaluation["reports"]["NONE"]):
                raise ValueError("source NONE 정책이 등록 walks 조건을 충족하지 않는다")
            if gate.get("outcome") != "not_recovered":
                raise ValueError("source F1 실패가 not_recovered로 고정되지 않았다")
            gates.append(gate)
    if len(gates) != 2:
        raise ValueError("source evaluation gate가 두 seed가 아니다")
    return True


def _verify_archived_record_raw(
    repo: Path, runtime: Path, source: Mapping[str, Any], record_job_id: str,
    references: Mapping[str, Any], resource: Mapping[str, Any], resource_sha: str,
    archived_posix_path: Callable[[Path, str, str], Path],
) -> None:
    action_index, action = next(
        (index, candidate) for index, candidate in enumerate(source["source_action_order"])
        if candidate["kind"] == "record" and candidate["job_id"] == record_job_id
    )
    raw_path = runtime / "actions" / f"{action_index:02d}_{action['action_id']}" / "raw.json"
    raw = SOURCE.read(raw_path)
    job_path, _ = _portable_reference(repo, references.get("job"), "record job")
    result_path, _ = _portable_reference(repo, references.get("result"), "record result")
    execution_path, _ = _portable_reference(repo, references.get("execution"), "record execution")
    admission_path, admission_ref = _portable_reference(
        repo, references.get("admission"), "record admission",
    )
    job, result, execution = (SOURCE.read(path) for path in (job_path, result_path, execution_path))
    expected_record = {
        "status": "capture_complete_gpu_smoke_pending", "model_use_ready": False,
        "result": result, "execution": execution,
    }
    if raw.get("record") != expected_record:
        raise ValueError("source record raw가 보관 capture 원문과 다르다")
    parent = raw.get("resource_parent")
    if (not isinstance(parent, dict) or set(parent) != {"path", "sha256"}
            or not isinstance(parent.get("path"), str)):
        raise ValueError("source record resource parent reference가 잘못됐다")
    job_source = job.get("source")
    if not isinstance(job_source, dict) or not isinstance(job_source.get("repo_root"), str):
        raise TypeError("record job source root가 없다")
    parent_path = _archived_original_path(
        repo, job_source["repo_root"], parent["path"], archived_posix_path,
    )
    if SOURCE.sha256(parent_path) != parent["sha256"]:
        raise ValueError("source record resource parent 원문·SHA가 다르다")
    admission = SOURCE.read(admission_path)
    expected_parent = {
        "schema": "vision_record_resource_parent_v1", "job_id": record_job_id,
        "resource_proof_sha256": resource_sha, "resource_id": resource["resource_id"],
        "gpu_uuid": resource["gpu_uuid"],
        "raw_gpu_receipt_sha256": resource["raw_gpu_receipt_sha256"],
        "budget_admission_sha256": resource["budget_admission_sha256"],
        "compute_started_at_utc": resource["compute_started_at_utc"],
        "admission_sha256": admission_ref["sha256"],
    }
    if SOURCE.read(parent_path) != expected_parent \
            or admission.get("resource_parent_sha256") != resource_sha:
        raise ValueError("source record resource parent 결합이 다르다")


def _archived_original_path(
    repo: Path, original_root: str, original_path: str,
    posix_resolver: Callable[[Path, str, str], Path],
) -> Path:
    root_posix, path_posix = PurePosixPath(original_root), PurePosixPath(original_path)
    if root_posix.is_absolute() or path_posix.is_absolute():
        return posix_resolver(repo, original_root, original_path)
    root_windows, path_windows = PureWindowsPath(original_root), PureWindowsPath(original_path)
    if (not root_windows.is_absolute() or not path_windows.is_absolute()
            or ".." in root_windows.parts or ".." in path_windows.parts
            or str(root_windows) != original_root or str(path_windows) != original_path):
        raise ValueError("보관 원문의 경로가 정규형이 아니다")
    try:
        relative = path_windows.relative_to(root_windows)
    except ValueError as error:
        raise ValueError("보관 원문이 원본 저장소 밖이다") from error
    candidate = repo.joinpath(*relative.parts).resolve(strict=True)
    candidate.relative_to(repo)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_nlink != 1:
        raise ValueError("보관 원문이 regular file이 아니다")
    return candidate


def export_materials(
    repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
    render_protocol_path: Path, render_protocol_sha256_lf: str, source_anchor: str,
    resource_path: Path, resource_sha256: str, *, resource_locations_path: Path | None = None,
) -> dict[str, Any]:
    """완료 14개 원문만 읽어 portable materials를 CPU에서 만든다."""
    repository = Path(repo).resolve(strict=True)
    source_path, render_path = Path(source_protocol_path).resolve(strict=True), Path(render_protocol_path).resolve(strict=True)
    if SOURCE.sha256(source_path, lf=True) != source_protocol_sha256_lf:
        raise ValueError("export source protocol LF SHA가 다르다")
    if SOURCE.sha256(render_path, lf=True) != render_protocol_sha256_lf:
        raise ValueError("export render protocol LF SHA가 다르다")
    source = SOURCE.validate_protocol(SOURCE.read(source_path))
    source_reference = {"path": source_path.relative_to(repository).as_posix(),
                        "sha256_lf": source_protocol_sha256_lf}
    render = validate_protocol(SOURCE.read(render_path), source, source_reference)
    _verify_offline_binding(
        repository, source_path, source_protocol_sha256_lf,
        render_path, render_protocol_sha256_lf, source_anchor, source,
    )
    proof_path = Path(resource_path).resolve(strict=True)
    if (proof_path.is_symlink() or proof_path.stat().st_nlink != 1
            or SOURCE.sha256(proof_path) != resource_sha256):
        raise ValueError("export resource proof 원본 SHA가 다르다")
    runtime = repository / ".triage-runtime/vision-brev" / source["protocol_id"]
    resource = SOURCE.read(proof_path)
    if not _verify_source_journals(
        repository, runtime, source, source_protocol_sha256_lf, resource, resource_sha256,
    ):
        return {"status": "HOLD", "reason": "source_or_render_journal_incomplete",
                "model_use_ready": False, "source_data_ready": False}
    import vision_render_readiness_v2 as FINALIZE  # pyright: ignore[reportMissingImports]

    resource_locations = None
    if resource_locations_path is not None:
        locations_path = Path(resource_locations_path).resolve(strict=True)
        locations_path.relative_to(repository)
        resource_locations = SOURCE.read(locations_path)
    pairs = []
    for index, row in enumerate(render["smoke_plan"]):
        folder = runtime / "render-actions" / f"{index:02d}"
        if not (folder / "receipt.json").exists():
            return {"status": "HOLD", "reason": "source_or_render_journal_incomplete",
                    "model_use_ready": False, "source_data_ready": False}
        record_refs = FINALIZE.archive_capture_references(repository, runtime, row["record_job_id"])
        smoke_refs = FINALIZE.archive_capture_references(repository, runtime, row["job_id"])
        _verify_archived_record_raw(
            repository, runtime, source, row["record_job_id"], record_refs, resource,
            resource_sha256, FINALIZE._archived_path,
        )
        material = FINALIZE.build_material(
            repository, source_path, source_protocol_sha256_lf, proof_path, runtime,
            row["record_job_id"], record_refs, smoke_refs, resource_locations=resource_locations,
        )
        FINALIZE.verify_material(repository, source_path, source_protocol_sha256_lf, material)
        pairs.append({
            **row,
            "output_path": f".triage-runtime/vision-brev/{source['protocol_id']}"
                           f"/readiness/{row['record_job_id']}.json",
            "material": material,
            "smoke_journal": {
                name: _path_reference(repository, folder / f"{name}.json", f"smoke {name}")
                for name in ("intent", "raw", "receipt")
            },
        })
    body = {
        "schema": MATERIALS_SCHEMA, "source_protocol": source_reference,
        "render_protocol": {"path": render_path.relative_to(repository).as_posix(),
                            "sha256_lf": render_protocol_sha256_lf},
        "pairs": pairs,
    }
    for index, row in enumerate(pairs):
        _verify_finalize_row(
            repository, row, render["smoke_plan"][index], index,
            source_protocol_sha256_lf, render_protocol_sha256_lf,
            render["smoke_limits"], source["resources"], source["protocol_id"],
        )
    path = runtime / "materials.json"
    if path.exists():
        if SOURCE.read(path) != body:
            raise ValueError("기존 readiness materials가 완료 원장과 다르다")
    else:
        SOURCE.write_new(path, body)
    reference = _path_reference(repository, path, "readiness materials")
    return {"status": "needs_readiness_finalize", "model_use_ready": False,
            "source_data_ready": False, "completed_smokes": 4, "materials": reference}


def _verify_finalize_row(repo: Path, row: Any, expected: Mapping[str, str], ordinal: int,
                         source_sha: str, render_sha: str, limits: Mapping[str, Any],
                         resources: Mapping[str, Any], protocol_id: str) -> tuple[Path, dict[str, Any]]:
    quote = float(resources["compute_quote_usd_per_h"])
    keys = {"action_id", "job_id", "record_job_id", "source_job_id", "checkpoint_logical_id",
            "output_path", "material", "smoke_journal"}
    if not isinstance(row, dict) or set(row) != keys \
            or any(row.get(key) != expected[key] for key in expected):
        raise ValueError("readiness pair가 파생 smoke plan과 다르다")
    expected_output = (f".triage-runtime/vision-brev/{protocol_id}/readiness/"
                       f"{expected['record_job_id']}.json")
    if row.get("output_path") != expected_output:
        raise ValueError("readiness output 경로가 protocol/record 파생값과 다르다")
    output_candidate = repo / expected_output
    current = output_candidate
    while current != repo:
        if current.is_symlink() or current.parent == current:
            raise ValueError("readiness output 경로에 symlink가 있거나 저장소를 벗어났다")
        current = current.parent
    output = output_candidate.resolve()
    output.relative_to(repo)
    if output.suffix != ".json":
        raise ValueError("readiness output은 저장소 내부 JSON이어야 한다")
    journal = row["smoke_journal"]
    if not isinstance(journal, dict) or set(journal) != {"intent", "raw", "receipt"}:
        raise ValueError("smoke journal reference 형식이 다르다")
    paths = {name: _portable_reference(repo, journal[name], f"smoke {name}")[0]
             for name in ("intent", "raw", "receipt")}
    intent, raw, receipt = (SOURCE.read(paths[name]) for name in ("intent", "raw", "receipt"))
    if (set(intent) != {"schema", "action_id", "job_id", "ordinal", "source_protocol_sha256_lf",
                       "render_protocol_sha256_lf", "resource_sha256", "job_binding_sha256",
                       "cost_before_usd", "started_at_utc", "reserved_max_wall_s",
                       "reserved_max_compute_usd"}
            or intent.get("schema") != SMOKE_INTENT_SCHEMA or intent.get("action_id") != expected["action_id"]
            or intent.get("job_id") != expected["job_id"] or intent.get("ordinal") != ordinal
            or intent.get("source_protocol_sha256_lf") != source_sha
            or intent.get("render_protocol_sha256_lf") != render_sha
            or intent.get("reserved_max_wall_s") != limits["max_wall_s"]
            or intent.get("reserved_max_compute_usd") != limits["max_compute_usd"]):
        raise ValueError("보관 smoke intent가 protocol과 다르다")
    if (set(raw) != {"schema", "action_id", "job_id", "job_binding_sha256", "smoke", "references"}
            or raw.get("schema") != SMOKE_RAW_SCHEMA or raw.get("action_id") != expected["action_id"]
            or raw.get("job_id") != expected["job_id"]
            or raw.get("job_binding_sha256") != intent.get("job_binding_sha256")
            or not isinstance(raw.get("smoke"), dict)
            or raw["smoke"].get("status") != "render_sync_smoke_complete"
            or raw["smoke"].get("model_use_ready") is not False):
        raise ValueError("보관 smoke raw가 protocol과 다르다")
    receipt_keys = {
        "schema", "action_id", "job_id", "source_protocol_sha256_lf", "render_protocol_sha256_lf",
        "resource_sha256", "job_binding_sha256", "intent_sha256", "raw_sha256", "started_at_utc",
        "finished_at_utc", "elapsed_wall_s", "cost_before_usd", "cost_after_usd",
    }
    if (set(receipt) != receipt_keys or receipt.get("schema") != SMOKE_RECEIPT_SCHEMA
            or receipt.get("action_id") != expected["action_id"] or receipt.get("job_id") != expected["job_id"]
            or receipt.get("source_protocol_sha256_lf") != source_sha
            or receipt.get("render_protocol_sha256_lf") != render_sha
            or receipt.get("resource_sha256") != intent.get("resource_sha256")
            or receipt.get("job_binding_sha256") != intent.get("job_binding_sha256")
            or receipt.get("intent_sha256") != SOURCE.sha256(paths["intent"])
            or receipt.get("raw_sha256") != SOURCE.sha256(paths["raw"])
            or receipt.get("started_at_utc") != intent.get("started_at_utc")
            or receipt.get("cost_before_usd") != intent.get("cost_before_usd")):
        raise ValueError("보관 smoke receipt 결합이 다르다")
    started, finished = SOURCE._stamp(receipt["started_at_utc"]), SOURCE._stamp(receipt["finished_at_utc"])
    elapsed = (finished - started).total_seconds()
    before, after = receipt["cost_before_usd"], receipt["cost_after_usd"]
    if (isinstance(before, bool) or not isinstance(before, (int, float)) or not math.isfinite(float(before))
            or isinstance(after, bool) or not isinstance(after, (int, float)) or not math.isfinite(float(after))
            or receipt.get("elapsed_wall_s") != elapsed or not 0 <= elapsed <= limits["max_wall_s"]
            or abs((float(after) - float(before)) - elapsed * quote / 3600.0) > 1e-9):
        raise ValueError("보관 smoke 시간·비용 원장이 다르다")
    material = row["material"]
    if not isinstance(material, dict) or not isinstance(material.get("record"), dict) \
            or not isinstance(material.get("smoke"), dict) or raw.get("references") != material["smoke"]:
        raise ValueError("smoke raw reference와 readiness material이 다르다")
    resource_path, resource_reference = _portable_reference(
        repo, material.get("resource_parent"), "resource parent",
    )
    resource = SOURCE.read(resource_path)
    required_resource = {
        "schema", "status", "resource_kind", "lease_id", "protocol_sha256", "issued_at_utc",
        "expires_at_utc", "compute_started_at_utc", "resource_id", "gpu_uuid",
        "raw_gpu_receipt_path", "raw_gpu_receipt_sha256", "budget_admission_path",
        "budget_admission_sha256", "provider_inventory_path", "provider_inventory_sha256",
        "nvidia_gpu_csv_path", "nvidia_gpu_csv_sha256", "nvidia_process_csv_path",
        "nvidia_process_csv_sha256",
    }
    if (set(resource) != required_resource or resource.get("schema") != SOURCE.RESOURCE_SCHEMA
            or resource.get("status") != "granted" or resource.get("resource_kind") != "remote_gpu"
            or resource.get("protocol_sha256") != source_sha
            or resource.get("resource_id") != resources["owned_instance_id"]
            or intent.get("resource_sha256") != resource_reference["sha256"]):
        raise ValueError("smoke journal resource proof 결합이 다르다")
    compute_started = SOURCE._stamp(resource["compute_started_at_utc"])
    issued = SOURCE._stamp(resource["issued_at_utc"])
    expires = SOURCE._stamp(resource["expires_at_utc"])
    deadline = SOURCE._stamp(resources["deadline_utc"])
    absolute_before = (started - compute_started).total_seconds() * quote / 3600.0
    absolute_after = (finished - compute_started).total_seconds() * quote / 3600.0
    if (not compute_started <= issued <= started <= finished <= min(expires, deadline)
            or abs(float(before) - absolute_before) > 1e-9
            or abs(float(after) - absolute_after) > 1e-9
            or float(before) < 0 or float(after) < float(before)
            or float(after) > float(resources["whole_batch_budget_usd"]) + 1e-9):
        raise ValueError("smoke journal lifecycle 비용·lease·deadline 결합이 다르다")
    record_job_path, _ = _portable_reference(repo, material["record"].get("job"), "record job")
    smoke_job_path, _ = _portable_reference(repo, material["smoke"].get("job"), "smoke job")
    record_job, smoke_job = SOURCE.read(record_job_path), SOURCE.read(smoke_job_path)
    if (record_job.get("job_id") != expected["record_job_id"]
            or smoke_job.get("job_id") != expected["job_id"]
            or record_job.get("mode") != RECORD.RECORD_MODE
            or smoke_job.get("mode") != RECORD.RENDER_SYNC_SMOKE_MODE
            or record_job.get("checkpoint") != smoke_job.get("checkpoint")
            or record_job.get("checkpoint", {}).get("logical_id") != expected["checkpoint_logical_id"]):
        raise ValueError("record와 smoke checkpoint/job 연결이 다르다")
    return output, material


def finalize_materials(repo: Path, source_protocol_path: Path, source_protocol_sha256_lf: str,
                       render_protocol_path: Path, render_protocol_sha256_lf: str,
                       materials_path: Path) -> dict[str, Any]:
    """보관된 자료만 읽어 네 readiness pair를 CPU에서 검증·봉인한다."""
    repository = Path(repo).resolve(strict=True)
    source_path = Path(source_protocol_path).resolve(strict=True)
    if SOURCE.sha256(source_path, lf=True) != source_protocol_sha256_lf:
        raise ValueError("finalize source protocol LF SHA가 다르다")
    source_protocol = SOURCE.validate_protocol(SOURCE.read(source_path))
    render_path = Path(render_protocol_path).resolve(strict=True)
    render_path.relative_to(repository)
    if SOURCE.sha256(render_path, lf=True) != render_protocol_sha256_lf:
        raise ValueError("finalize render protocol LF SHA가 다르다")
    source_reference = {"path": source_path.relative_to(repository).as_posix(),
                        "sha256_lf": source_protocol_sha256_lf}
    render_protocol = validate_protocol(SOURCE.read(render_path), source_protocol, source_reference)
    materials_candidate = Path(materials_path)
    if not materials_candidate.is_absolute():
        materials_candidate = repository / materials_candidate
    current = materials_candidate
    while current != repository:
        if current.is_symlink() or current.parent == current:
            raise ValueError("readiness materials manifest 경로가 저장소를 벗어났거나 symlink다")
        current = current.parent
    materials_file = materials_candidate.resolve(strict=True)
    materials_file.relative_to(repository)
    if materials_file.is_symlink() or not materials_file.is_file() or materials_file.stat().st_nlink != 1:
        raise ValueError("readiness materials manifest 원본이 regular file이 아니다")
    body = SOURCE.read(materials_file)
    if set(body) != {"schema", "source_protocol", "render_protocol", "pairs"} \
            or body.get("schema") != MATERIALS_SCHEMA:
        raise ValueError("readiness materials manifest 형식이 다르다")
    expected_render = {"path": render_path.relative_to(repository).as_posix(),
                       "sha256_lf": render_protocol_sha256_lf}
    if body.get("source_protocol") != source_reference or body.get("render_protocol") != expected_render \
            or not isinstance(body.get("pairs"), list) \
            or len(body["pairs"]) != 4:
        raise ValueError("readiness materials protocol/pair 결합이 다르다")
    verified = [
        _verify_finalize_row(
            repository, row, render_protocol["smoke_plan"][index], index,
            source_protocol_sha256_lf, render_protocol_sha256_lf,
            render_protocol["smoke_limits"], source_protocol["resources"], source_protocol["protocol_id"],
        )
        for index, row in enumerate(body["pairs"])
    ]
    outputs = [output for output, _ in verified]
    material_digests = [SOURCE.digest(material) for _, material in verified]
    record_digests = [SOURCE.digest(material["record"]) for _, material in verified]
    smoke_digests = [SOURCE.digest(material["smoke"]) for _, material in verified]
    if (len(set(outputs)) != 4 or len(set(material_digests)) != 4
            or len(set(record_digests)) != 4 or len(set(smoke_digests)) != 4):
        raise ValueError("readiness output 또는 record/smoke material이 중복됐다")
    import vision_render_readiness_v2 as FINALIZE  # pyright: ignore[reportMissingImports]
    receipts = [
        FINALIZE.finalize_pair(repository, source_path, source_protocol_sha256_lf, material, output)
        for output, material in verified
    ]
    return {"status": "source_data_ready", "model_use_ready": True, "pairs": receipts}


def _all_capture_receipts(repo: Path, source_protocol_path: Path) -> bool:
    source = SOURCE.validate_protocol(SOURCE.read(source_protocol_path))
    runtime = repo / ".triage-runtime/vision-brev" / source["protocol_id"]
    source_receipts = [
        runtime / "actions" / f"{index:02d}_{action['action_id']}" / "receipt.json"
        for index, action in enumerate(source["source_action_order"])
    ]
    smoke_receipts = [runtime / "render-actions" / f"{index:02d}" / "receipt.json" for index in range(4)]
    return all(path.is_file() and not path.is_symlink() for path in [*source_receipts, *smoke_receipts])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "advance", "export", "finalize"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--render-protocol", required=True, type=Path)
    parser.add_argument("--render-protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor")
    parser.add_argument("--resource-proof", type=Path)
    parser.add_argument("--resource-sha256")
    parser.add_argument("--isaaclab", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--resource-locations", type=Path)
    args = parser.parse_args(argv)
    if args.command == "finalize":
        if args.materials is None:
            parser.error("finalize에는 --materials가 필요하다")
        result = finalize_materials(
            ROOT, args.protocol, args.protocol_sha256_lf,
            args.render_protocol, args.render_protocol_sha256_lf, args.materials,
        )
    elif args.command == "export" or (args.command == "advance" and _all_capture_receipts(ROOT, args.protocol)):
        if any(value is None for value in (args.source_anchor, args.resource_proof, args.resource_sha256)):
            parser.error("export에는 source anchor와 resource proof 인자가 필요하다")
        result = export_materials(
            ROOT, args.protocol, args.protocol_sha256_lf,
            args.render_protocol, args.render_protocol_sha256_lf, args.source_anchor,
            args.resource_proof, args.resource_sha256,
            resource_locations_path=args.resource_locations,
        )
    else:
        if any(value is None for value in (args.source_anchor, args.resource_proof,
                                            args.resource_sha256, args.isaaclab, args.python)):
            parser.error("verify/advance에는 source/resource/Isaac 경로 인자가 필요하다")
        batch = RenderBatch(
            ROOT, args.protocol, args.protocol_sha256_lf,
            args.render_protocol, args.render_protocol_sha256_lf, args.source_anchor,
            args.resource_proof, args.resource_sha256, args.isaaclab, args.python,
        )
        result = batch.verify() if args.command == "verify" else batch.advance()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
