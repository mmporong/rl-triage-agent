"""사전등록된 V04 Brev source 배치를 한 action씩 재개 가능하게 실행한다.

VM 생성·SSH·API 호출은 이 모듈의 범위가 아니다. 이미 준비된 Brev Linux
환경에서 고정 source protocol과 자원 증거를 검증한 뒤 학습 네 건, 평가 두
건, 같은-Play 기록 네 건을 순서대로 ``VisionBrevBackend``에 전달한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import preregistered_seed_guard as SEEDS  # pyright: ignore[reportMissingImports]
import record_vision_evidence_v1 as RECORD
import residual_demo_v1 as RECOVERY  # pyright: ignore[reportMissingImports]
import vision_brev_backend_v1 as VISION  # pyright: ignore[reportMissingImports]
import vision_remote_gpu_v1 as REMOTE  # pyright: ignore[reportMissingImports]

SCHEMA = "vision_brev_source_protocol_v1"
MANIFEST_SCHEMA = "vision_brev_source_manifest_v1"
RESOURCE_SCHEMA = "brev_resource_proof_v1"
RAW_GPU_RECEIPT_SCHEMA = "brev_raw_gpu_receipt_v1"
BUDGET_ADMISSION_SCHEMA = "brev_budget_admission_v1"
INTENT_SCHEMA = "vision_brev_source_action_intent_v1"
RECEIPT_SCHEMA = "vision_brev_source_action_receipt_v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
PROTOCOL_KEYS = {
    "schema", "protocol_id", "registered_before_measurement", "record_profile_sha256",
    "training_plan", "source_action_order", "evaluation", "recording", "resources",
    "source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf",
    "environment", "fixed_eval_protocol",
}
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10
ACTION_AUTHORIZATION_S = 1800


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def read(path: Path) -> dict[str, Any]:
    def constant(value: str) -> None:
        raise ValueError(f"비유한 JSON 값 {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=constant)
    if not isinstance(value, dict):
        raise TypeError("JSON object가 필요하다")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True,
        encoding="utf-8", errors="strict",
    )


def _stamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("UTC timestamp가 문자열이 아니다")
    match = re.fullmatch(r"(.+\.)(\d{7,})([+-]\d\d:\d\d|Z)", value)
    if match:
        value = match.group(1) + match.group(2)[:6] + match.group(3)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162 - Python 3.10
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp에 timezone이 없다")
    return parsed.astimezone(UTC_TZ)


def _finite_positive(value: Any, anchor: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{anchor}는 양의 유한 수여야 한다")
    return float(value)


def _hash_map(value: Any, anchor: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{anchor}가 비었다")
    result: dict[str, str] = {}
    for relative, value_hash in value.items():
        path = Path(relative) if isinstance(relative, str) else Path("/")
        if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                or not isinstance(value_hash, str) or HEX64.fullmatch(value_hash) is None):
            raise ValueError(f"{anchor} 형식이 잘못됐다")
        result[relative] = value_hash
    return result


def _validate_actions(actions: Any, plan: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(actions, list) or len(actions) != 10:
        raise ValueError("source_action_order는 4 train+2 eval+4 record여야 한다")
    result: list[dict[str, Any]] = []
    plan_by_id = {row["job_id"]: row for row in plan}
    kinds = []
    ids: set[str] = set()
    train_ids: list[str] = []
    eval_seeds: list[int] = []
    record_sources: list[str] = []
    for raw in actions:
        if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str):
            raise TypeError("source action 형식이 잘못됐다")
        kind = raw["kind"]
        required = {
            "train": {"action_id", "kind", "job_id"},
            "evaluate": {"action_id", "kind", "seed", "none_job_id", "f1_job_id", "run_name"},
            "record": {"action_id", "kind", "job_id", "source_job_id", "checkpoint_logical_id"},
        }.get(kind)
        if required is None or set(raw) != required:
            raise ValueError("source action 필드가 kind 계약과 다르다")
        action_id = raw["action_id"]
        if not isinstance(action_id, str) or SLUG.fullmatch(action_id) is None or action_id in ids:
            raise ValueError("action_id가 잘못됐거나 중복됐다")
        ids.add(action_id)
        kinds.append(kind)
        row = dict(raw)
        if kind == "train":
            job_id = row["job_id"]
            if job_id not in plan_by_id or plan_by_id[job_id]["arm"] not in {"NONE", "F1"}:
                raise ValueError("source phase train은 등록 NONE/F1만 허용한다")
            train_ids.append(job_id)
        elif kind == "evaluate":
            seed = row["seed"]
            none, fault = plan_by_id.get(row["none_job_id"]), plan_by_id.get(row["f1_job_id"])
            if (type(seed) is not int or not none or not fault or none["seed"] != seed or fault["seed"] != seed
                    or none["arm"] != "NONE" or fault["arm"] != "F1"
                    or not isinstance(row["run_name"], str) or SLUG.fullmatch(row["run_name"]) is None):
                raise ValueError("evaluate가 같은 seed NONE/F1과 결합되지 않았다")
            eval_seeds.append(seed)
        else:
            source = plan_by_id.get(row["source_job_id"])
            if (not source or source["arm"] not in {"NONE", "F1"} or row["source_job_id"] not in train_ids
                    or not isinstance(row["job_id"], str) or SLUG.fullmatch(row["job_id"]) is None
                    or not isinstance(row["checkpoint_logical_id"], str)
                    or SLUG.fullmatch(row["checkpoint_logical_id"]) is None):
                raise ValueError("record가 등록 source checkpoint와 결합되지 않았다")
            record_sources.append(row["source_job_id"])
        result.append(row)
    if kinds != ["train"] * 4 + ["evaluate"] * 2 + ["record"] * 4:
        raise ValueError("source action 종류 순서가 고정 4→2→4와 다르다")
    expected_train = {row["job_id"] for row in plan if row["arm"] in {"NONE", "F1"}}
    if set(train_ids) != expected_train or len(train_ids) != 4:
        raise ValueError("네 source 학습이 정확히 한 번씩 등록되지 않았다")
    seeds = sorted({row["seed"] for row in plan})
    if sorted(eval_seeds) != seeds or set(record_sources) != expected_train or len(record_sources) != 4:
        raise ValueError("평가 seed 또는 record source 집합이 다르다")
    return result


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != PROTOCOL_KEYS or value.get("schema") != SCHEMA:
        raise ValueError("source protocol schema/필드가 다르다")
    protocol = dict(value)
    protocol_id = protocol["protocol_id"]
    if not isinstance(protocol_id, str) or SLUG.fullmatch(protocol_id) is None:
        raise ValueError("protocol_id가 잘못됐다")
    if protocol["registered_before_measurement"] is not True:
        raise ValueError("source protocol은 실행 전에 등록돼야 한다")
    if protocol["record_profile_sha256"] != RECORD.digest(RECORD.fixed_profile()):
        raise ValueError("record profile SHA가 현재 고정 profile과 다르다")
    plan = VISION._validate_plan(protocol["training_plan"])
    protocol["training_plan"] = plan
    protocol["source_action_order"] = _validate_actions(protocol["source_action_order"], plan)
    evaluation = protocol["evaluation"]
    if (not isinstance(evaluation, dict) or set(evaluation) != {"bands_path", "bands_sha256"}
            or not isinstance(evaluation["bands_path"], str)
            or not isinstance(evaluation["bands_sha256"], str)
            or HEX64.fullmatch(evaluation["bands_sha256"]) is None):
        raise ValueError("evaluation 결합이 잘못됐다")
    recording = protocol["recording"]
    if not isinstance(recording, dict) or set(recording) != {"source_files_lf"}:
        raise ValueError("recording 필드가 다르다")
    recording["source_files_lf"] = _hash_map(recording["source_files_lf"], "recording.source_files_lf")
    resources = protocol["resources"]
    if (not isinstance(resources, dict)
            or set(resources) != {"deadline_utc", "whole_batch_budget_usd", "compute_quote_usd_per_h",
                                  "owned_instance_id", "max_owned_instances", "action_limits",
                                  "reservation_method", "cost_scope"}):
        raise ValueError("resource 계약 필드가 다르다")
    _stamp(resources["deadline_utc"])
    _finite_positive(resources["whole_batch_budget_usd"], "whole_batch_budget_usd")
    _finite_positive(resources["compute_quote_usd_per_h"], "compute_quote_usd_per_h")
    if (not isinstance(resources["owned_instance_id"], str)
            or SLUG.fullmatch(resources["owned_instance_id"]) is None
            or resources["max_owned_instances"] != 1):
        raise ValueError("Brev owned instance 계약이 잘못됐다")
    if (resources["reservation_method"] != "owned_instance_elapsed_plus_next_max_v1"
            or resources["cost_scope"]
            != "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1"):
        raise ValueError("Brev 비용 예약 범위·계산 방식이 고정 계약과 다르다")
    limits = resources["action_limits"]
    expected_wall = {"train": 1800, "evaluate": 900, "record": 900}
    if not isinstance(limits, dict) or set(limits) != set(expected_wall):
        raise ValueError("action별 비용 상한이 없다")
    quote = float(resources["compute_quote_usd_per_h"])
    for kind, wall in expected_wall.items():
        limit = limits[kind]
        if (not isinstance(limit, dict) or set(limit) != {"max_wall_s", "max_compute_usd"}
                or limit["max_wall_s"] != wall
                or abs(_finite_positive(limit["max_compute_usd"], f"{kind}.max_compute_usd")
                       - quote * wall / 3600.0) > 1e-9):
            raise ValueError("action별 시간·compute 비용 상한이 quote와 다르다")
    reserved = sum(float(limits[action["kind"]]["max_compute_usd"])
                   for action in protocol["source_action_order"])
    if reserved > float(resources["whole_batch_budget_usd"]) + 1e-9:
        raise ValueError("전체 action 최대 compute 예약이 batch 상한을 넘는다")
    for key in ("source_sha256_lf", "runtime_source_sha256_lf", "python_package_source_sha256_lf"):
        protocol[key] = _hash_map(protocol[key], key)
    if not isinstance(protocol["environment"], dict) or not isinstance(protocol["fixed_eval_protocol"], str):
        raise TypeError("environment/fixed_eval_protocol이 잘못됐다")
    return protocol


class SourceBatch:
    def __init__(
        self, repo: Path, protocol_path: Path, protocol_sha256_lf: str, source_anchor: str,
        resource_path: Path, resource_sha256: str, lab: Path, python: Path, *,
        backend_factory: Callable[..., Any] = VISION.VisionBrevBackend.from_runtime,
        seed_guard: Callable[..., dict[str, Any]] = SEEDS.assert_fresh_seeds,
        now: Callable[[], datetime] = lambda: datetime.now(UTC_TZ),
    ):
        self.repo, self.lab, self.python = Path(repo).resolve(), Path(lab).resolve(), Path(python).absolute()
        self.protocol_path, self.resource_path = Path(protocol_path).resolve(), Path(resource_path).resolve()
        if (not self.protocol_path.is_file() or self.protocol_path.is_symlink()
                or not isinstance(protocol_sha256_lf, str) or HEX64.fullmatch(protocol_sha256_lf) is None
                or sha256(self.protocol_path, lf=True) != protocol_sha256_lf):
            raise ValueError("source protocol 실제 LF SHA가 다르다")
        if not isinstance(source_anchor, str) or HEX40.fullmatch(source_anchor) is None:
            raise ValueError("source anchor 형식이 잘못됐다")
        self.protocol = validate_protocol(read(self.protocol_path))
        self.protocol_sha, self.anchor = protocol_sha256_lf, source_anchor
        self.backend_factory, self.seed_guard, self.now = backend_factory, seed_guard, now
        self._verify_source()
        if (not self.resource_path.is_file() or self.resource_path.is_symlink()
                or not isinstance(resource_sha256, str) or HEX64.fullmatch(resource_sha256) is None
                or sha256(self.resource_path) != resource_sha256):
            raise ValueError("resource proof 원본 SHA가 다르다")
        self.resource_sha = resource_sha256
        self.resource = self._verify_resource(read(self.resource_path))
        self.runtime = self.repo / ".triage-runtime/vision-brev" / self.protocol["protocol_id"]
        self._verify_runtime_path()
        self._backend_value: Any | None = None

    def _verify_runtime_path(self) -> None:
        current = self.repo
        for part in (".triage-runtime", "vision-brev", self.protocol["protocol_id"]):
            current /= part
            if current.exists() and current.is_symlink():
                raise ValueError("canonical runtime 경로에 symlink가 있다")
        resolved = self.runtime.resolve()
        if resolved != self.repo and self.repo not in resolved.parents:
            raise ValueError("canonical runtime이 repo를 벗어났다")

    def _verify_bound_inputs(self) -> None:
        if sha256(self.protocol_path, lf=True) != self.protocol_sha:
            raise ValueError("실행 중 source protocol LF SHA가 바뀌었다")
        if sha256(self.resource_path) != self.resource_sha:
            raise ValueError("실행 중 resource proof 원본 SHA가 바뀌었다")

    def _verify_source(self) -> None:
        if not (self.repo / ".git").exists():
            raise ValueError("repo가 Git 저장소가 아니다")
        relative = self.protocol_path.relative_to(self.repo).as_posix()
        if _git(self.repo, "merge-base", "--is-ancestor", self.anchor, "HEAD", check=False).returncode:
            raise ValueError("source anchor가 현재 HEAD의 조상이 아니다")
        blob = _git(self.repo, "show", f"{self.anchor}:{relative}").stdout.encode().replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != self.protocol_sha:
            raise ValueError("source anchor의 protocol blob SHA가 다르다")
        for relative_path, expected in self.protocol["source_sha256_lf"].items():
            path = self.repo / relative_path
            if not path.is_file() or path.is_symlink() or sha256(path, lf=True) != expected:
                raise ValueError("repository source LF SHA가 다르다")
            committed = _git(self.repo, "show", f"{self.anchor}:{relative_path}").stdout.encode().replace(b"\r\n", b"\n")
            if hashlib.sha256(committed).hexdigest() != expected:
                raise ValueError("source anchor blob과 현재 source가 다르다")
        bands = self.repo / self.protocol["evaluation"]["bands_path"]
        if not bands.is_file() or sha256(bands) != self.protocol["evaluation"]["bands_sha256"]:
            raise ValueError("고정 평가 band SHA가 다르다")
        for relative_path, expected in self.protocol["recording"]["source_files_lf"].items():
            path = self.repo / relative_path
            if not path.is_file() or sha256(path, lf=True) != expected:
                raise ValueError("record source LF SHA가 다르다")
            committed = _git(self.repo, "show", f"{self.anchor}:{relative_path}").stdout.encode().replace(
                b"\r\n", b"\n"
            )
            if hashlib.sha256(committed).hexdigest() != expected:
                raise ValueError("record source anchor blob과 현재 source가 다르다")

    def _verify_resource(self, value: Mapping[str, Any]) -> dict[str, Any]:
        required = {"schema", "status", "resource_kind", "lease_id", "protocol_sha256",
                    "issued_at_utc", "expires_at_utc", "compute_started_at_utc",
                    "resource_id", "gpu_uuid",
                    "raw_gpu_receipt_path", "raw_gpu_receipt_sha256",
                    "budget_admission_path", "budget_admission_sha256",
                    "provider_inventory_path", "provider_inventory_sha256",
                    "nvidia_gpu_csv_path", "nvidia_gpu_csv_sha256",
                    "nvidia_process_csv_path", "nvidia_process_csv_sha256"}
        if (set(value) != required or value.get("schema") != RESOURCE_SCHEMA or value.get("status") != "granted"
                or value.get("resource_kind") != "remote_gpu"
                or not isinstance(value.get("lease_id"), str) or SLUG.fullmatch(value["lease_id"]) is None
                or value.get("resource_id") != self.protocol["resources"]["owned_instance_id"]
                or not isinstance(value.get("gpu_uuid"), str) or not value["gpu_uuid"].startswith("GPU-")
                or not isinstance(value.get("raw_gpu_receipt_path"), str)
                or not isinstance(value.get("raw_gpu_receipt_sha256"), str)
                or HEX64.fullmatch(value["raw_gpu_receipt_sha256"]) is None
                or not isinstance(value.get("budget_admission_path"), str)
                or not isinstance(value.get("budget_admission_sha256"), str)
                or HEX64.fullmatch(value["budget_admission_sha256"]) is None
                or value.get("protocol_sha256") != self.protocol_sha):
            raise ValueError("resource proof 결합이 다르다")
        REMOTE.verify_capture_proof({key: value[key] for key in REMOTE.CAPTURE_KEYS})
        started = _stamp(value["compute_started_at_utc"])
        issued, expires, current = _stamp(value["issued_at_utc"]), _stamp(value["expires_at_utc"]), self.now()
        if current.tzinfo is None:
            raise ValueError("현재 시각에 timezone이 없다")
        current_utc = current.astimezone(UTC_TZ)
        if (started > issued or issued > current_utc
                or expires < current_utc + timedelta(seconds=ACTION_AUTHORIZATION_S)
                or expires <= issued):
            raise ValueError("resource proof 유효 시간이 현재 실행을 포함하지 않는다")
        raw_path = Path(value["raw_gpu_receipt_path"])
        budget_path = Path(value["budget_admission_path"])
        for path, expected_sha, anchor in (
            (raw_path, value["raw_gpu_receipt_sha256"], "raw GPU receipt"),
            (budget_path, value["budget_admission_sha256"], "budget admission"),
        ):
            if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                    or path.stat().st_nlink != 1 or sha256(path) != expected_sha):
                raise ValueError(f"{anchor} 원본 파일·SHA가 다르다")
        common = {
            "status": "granted", "resource_id": value["resource_id"],
            "gpu_uuid": value["gpu_uuid"], "lease_id": value["lease_id"],
            "issued_at_utc": value["issued_at_utc"], "expires_at_utc": value["expires_at_utc"],
            "compute_started_at_utc": value["compute_started_at_utc"],
            "protocol_sha256": self.protocol_sha,
            "provider_inventory_sha256": value["provider_inventory_sha256"],
            "nvidia_gpu_csv_sha256": value["nvidia_gpu_csv_sha256"],
            "nvidia_process_csv_sha256": value["nvidia_process_csv_sha256"],
        }
        raw = read(raw_path)
        if raw != {"schema": RAW_GPU_RECEIPT_SCHEMA, **common}:
            raise ValueError("raw GPU receipt가 resource/lease/protocol과 다르다")
        budget = read(budget_path)
        expected_budget = {
            "schema": BUDGET_ADMISSION_SCHEMA, **common,
            "whole_batch_budget_usd": self.protocol["resources"]["whole_batch_budget_usd"],
            "compute_quote_usd_per_h": self.protocol["resources"]["compute_quote_usd_per_h"],
            "max_owned_instances": 1,
            "cost_scope": self.protocol["resources"]["cost_scope"],
        }
        if budget != expected_budget:
            raise ValueError("budget admission이 protocol 비용·자원과 다르다")
        return dict(value)

    def verify(self) -> dict[str, Any]:
        self._verify_bound_inputs()
        self._verify_source()
        self._verify_resource(read(self.resource_path))
        return {"status": "verified", "protocol_id": self.protocol["protocol_id"],
                "protocol_sha256_lf": self.protocol_sha, "source_anchor": self.anchor,
                "actions": len(self.protocol["source_action_order"]), "runtime": str(self.runtime)}

    def _initialize(self) -> None:
        self.runtime.mkdir(parents=True, exist_ok=True)
        freshness_path, manifest_path = self.runtime / "freshness.json", self.runtime / "manifest.json"
        seeds = sorted({row["seed"] for row in self.protocol["training_plan"]})
        if freshness_path.exists():
            freshness = read(freshness_path)
            if freshness.get("status") != "fresh" or freshness.get("seeds") != seeds:
                raise ValueError("초기 seed freshness receipt가 다르다")
            actions_root = self.runtime / "actions"
            if not actions_root.is_dir() or not any(actions_root.iterdir()):
                raise ValueError("첫 action 예약 없는 seed freshness receipt는 재사용할 수 없다")
        else:
            freshness = self.seed_guard(seeds, self.lab, repo=self.repo)
            if freshness.get("status") != "fresh" or freshness.get("seeds") != seeds:
                raise ValueError("seed freshness guard가 fresh receipt를 반환하지 않았다")
            write_new(freshness_path, freshness)
        body = {
            "schema": MANIFEST_SCHEMA, "protocol_id": self.protocol["protocol_id"],
            "protocol_path": self.protocol_path.relative_to(self.repo).as_posix(),
            "protocol_sha256_lf": self.protocol_sha, "source_anchor": self.anchor,
            "resource_path_sha256": self.resource_sha, "training_plan_sha256": digest(self.protocol["training_plan"]),
            "source_action_order_sha256": digest(self.protocol["source_action_order"]),
            "freshness_sha256": sha256(freshness_path),
        }
        if manifest_path.exists():
            if read(manifest_path) != body:
                raise ValueError("기존 source runtime manifest가 현재 결합과 다르다")
        else:
            write_new(manifest_path, body)

    def _backend(self) -> Any:
        if self._backend_value is None:
            backend_protocol = {
                **self.protocol, "protocol_path": str(self.protocol_path),
                "protocol_sha256_lf": self.protocol_sha,
            }
            self._backend_value = self.backend_factory(
                self.lab, self.python, backend_protocol, self.anchor,
                self.runtime, self.protocol["training_plan"], resource_proof=self.resource,
            )
        return self._backend_value

    def _action_folder(self, index: int, action: Mapping[str, Any]) -> Path:
        return self.runtime / "actions" / f"{index:02d}_{action['action_id']}"

    def _raw(self, index: int) -> dict[str, Any]:
        action = self.protocol["source_action_order"][index]
        folder = self._action_folder(index, action)
        receipt_path, raw_path = folder / "receipt.json", folder / "raw.json"
        intent_path = folder / "intent.json"
        if (not receipt_path.is_file() or receipt_path.is_symlink() or not raw_path.is_file()
                or raw_path.is_symlink() or not intent_path.is_file() or intent_path.is_symlink()):
            raise ValueError("완료 source action 파일이 없다")
        receipt = read(receipt_path)
        if (set(receipt) != {"schema", "action_id", "intent_sha256", "raw_sha256",
                             "cost_after_usd", "finished_at_utc"}
                or receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("action_id") != action["action_id"]
                or receipt.get("intent_sha256") != sha256(intent_path)
                or receipt.get("raw_sha256") != sha256(raw_path)):
            raise ValueError("source action receipt SHA 결합이 다르다")
        intent = read(intent_path)
        limit = self.protocol["resources"]["action_limits"][action["kind"]]
        if (set(intent) != {"schema", "action_id", "kind", "ordinal", "protocol_sha256_lf",
                            "resource_sha256", "cost_before_usd", "started_at_utc", "reserved_max_wall_s",
                            "reserved_max_compute_usd", "cost_scope"}
                or intent.get("schema") != INTENT_SCHEMA or intent.get("action_id") != action["action_id"]
                or intent.get("kind") != action["kind"] or intent.get("ordinal") != index
                or intent.get("protocol_sha256_lf") != self.protocol_sha
                or intent.get("resource_sha256") != self.resource_sha
                or intent.get("reserved_max_wall_s") != limit["max_wall_s"]
                or intent.get("reserved_max_compute_usd") != limit["max_compute_usd"]
                or intent.get("cost_scope") != self.protocol["resources"]["cost_scope"]):
            raise ValueError("source action intent 예약 결합이 다르다")
        raw = read(raw_path)
        if raw.get("kind") != action["kind"] or raw.get("action_id") != action["action_id"]:
            raise ValueError("source action raw identity가 다르다")
        if action["kind"] == "train":
            artifact = raw.get("artifact")
            if not isinstance(artifact, dict):
                raise TypeError("source training artifact가 object가 아니다")
            row = next(row for row in self.protocol["training_plan"] if row["job_id"] == action["job_id"])
            checkpoint = artifact.get("checkpoint")
            if (not isinstance(checkpoint, dict) or artifact.get("arm") != row["arm"]
                    or artifact.get("seed") != row["seed"] or not Path(checkpoint.get("path", "")).is_file()
                    or sha256(Path(checkpoint["path"])) != checkpoint.get("sha256")):
                raise ValueError("source training artifact가 변조됐다")
            telemetry = artifact.get("telemetry")
            manifest_path = Path(artifact.get("manifest_path", ""))
            if (not isinstance(telemetry, dict) or not Path(telemetry.get("path", "")).is_file()
                    or sha256(Path(telemetry["path"])) != telemetry.get("sha256")
                    or not manifest_path.is_file() or sha256(manifest_path) != artifact.get("manifest_sha256")):
                raise ValueError("source training telemetry/manifest가 변조됐다")
            if self._backend().train(action["job_id"]) != artifact:
                raise ValueError("backend cached training artifact가 raw와 다르다")
        elif action["kind"] == "evaluate":
            evaluation, gate = raw.get("evaluation"), raw.get("source_gate")
            if (not isinstance(evaluation, dict) or set(evaluation.get("reports", {})) != {"NONE", "F1"}
                    or not isinstance(evaluation.get("execution"), dict)
                    or evaluation["execution"].get("completed") is not True):
                raise ValueError("source evaluation raw가 완결되지 않았다")
            self._backend().verify_execution_gpu_observations(evaluation["execution"])
            bands = read(self.repo / self.protocol["evaluation"]["bands_path"])["bands"]
            measured = RECOVERY.recovery(evaluation["reports"]["F1"], evaluation["reports"]["NONE"], bands)
            if gate != measured:
                raise ValueError("source failure gate 재계산이 raw와 다르다")
        elif action["kind"] == "record":
            result = raw.get("record")
            if (not isinstance(result, dict) or result.get("status") != "capture_complete_gpu_smoke_pending"
                    or result.get("model_use_ready") is not False
                    or result.get("result", {}).get("zero_update_pixel_lag_confirmed") is not False
                    or result.get("result", {}).get("gpu_smoke_required_before_model_use") is not True):
                raise ValueError("record 결과가 render-sync smoke 전 상태를 보존하지 않았다")
            parent = raw.get("resource_parent")
            if (not isinstance(parent, dict) or set(parent) != {"path", "sha256"}
                    or not Path(parent["path"]).is_file() or sha256(Path(parent["path"])) != parent["sha256"]):
                raise ValueError("record resource parent receipt가 없다")
            train_index = next(
                position for position, candidate in enumerate(self.protocol["source_action_order"])
                if candidate["kind"] == "train" and candidate["job_id"] == action["source_job_id"]
            )
            artifact = self._raw(train_index)["artifact"]
            job, expected_parent = self._record_job(action, artifact)
            if expected_parent != parent or self._backend().record(job) != result:
                raise ValueError("backend cached record가 raw 또는 resource parent와 다르다")
        started_at = _stamp(intent["started_at_utc"])
        finished_at = _stamp(receipt["finished_at_utc"])
        cost_before = self._lifecycle_cost(started_at)
        cost_after = self._lifecycle_cost(finished_at)
        elapsed_wall = (finished_at - started_at).total_seconds()
        if (intent["cost_before_usd"] != cost_before or receipt["cost_after_usd"] != cost_after
                or elapsed_wall < 0 or elapsed_wall > float(limit["max_wall_s"])
                or self._cost([raw]) > float(limit["max_compute_usd"]) + 1e-9
                or cost_after > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9
                or _stamp(receipt["finished_at_utc"]) > _stamp(self.protocol["resources"]["deadline_utc"])):
            raise ValueError("source action 완료 비용·deadline 결합이 다르다")
        return raw

    def _cost(self, completed: Sequence[dict[str, Any]]) -> float:
        wall = 0.0
        for raw in completed:
            payload = raw.get({"train": "artifact", "evaluate": "evaluation", "record": "record"}[raw["kind"]], {})
            execution = payload.get("execution", {}) if isinstance(payload, dict) else {}
            value = execution.get("wall_s", 0.0) if isinstance(execution, dict) else 0.0
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("source action wall_s가 잘못됐다")
            wall += float(value)
        return wall * float(self.protocol["resources"]["compute_quote_usd_per_h"]) / 3600.0

    def _lifecycle_cost(self, observed_at: datetime) -> float:
        started = _stamp(self.resource["compute_started_at_utc"])
        current = observed_at.astimezone(UTC_TZ)
        elapsed = (current - started).total_seconds()
        if elapsed < 0:
            raise ValueError("compute 시작 전 시각으로 lifecycle 비용을 계산할 수 없다")
        return elapsed * float(self.protocol["resources"]["compute_quote_usd_per_h"]) / 3600.0

    def _record_job(self, action: Mapping[str, Any], artifact: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        checkpoint = artifact["checkpoint"]
        inputs = self.runtime / "record-inputs" / action["action_id"]
        inputs.mkdir(parents=True, exist_ok=True)
        protocol_path, admission_path = inputs / "protocol.json", inputs / "admission.json"
        parent_path = inputs / "resource_parent.json"
        protocol_body = {
            "schema": RECORD.PROTOCOL_SCHEMA, "method": RECORD.METHOD, "mode": RECORD.RECORD_MODE,
            "profile": RECORD.fixed_profile(),
            "source_commit": self.anchor, "source_files_lf": self.protocol["recording"]["source_files_lf"],
            "checkpoint_sha256": checkpoint["sha256"],
            "resource": {"resource_id": self.resource["resource_id"],
                         "gpu_uuid": self.resource["gpu_uuid"]},
        }
        if protocol_path.exists():
            if read(protocol_path) != protocol_body:
                raise ValueError("기존 record protocol이 다르다")
        else:
            write_new(protocol_path, protocol_body)
        output = self.runtime / "record" / action["job_id"] / "output"
        resource_parent_sha = self.resource_sha
        live_lease_proof_sha = self.resource["raw_gpu_receipt_sha256"]
        job = {
            "schema": RECORD.JOB_SCHEMA, "method": RECORD.METHOD, "mode": RECORD.RECORD_MODE,
            "job_id": action["job_id"],
            "profile": RECORD.fixed_profile(),
            "source": {"repo_root": str(self.repo), "commit": self.anchor,
                       "files_lf": self.protocol["recording"]["source_files_lf"]},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "checkpoint": {"path": checkpoint["path"], "sha256": checkpoint["sha256"],
                           "logical_id": action["checkpoint_logical_id"]},
            "resource": {"resource_id": self.resource["resource_id"],
                         "gpu_uuid": self.resource["gpu_uuid"]},
            "remote_gpu_admission": {"resource_parent_sha256": resource_parent_sha,
                                     "live_lease_proof_sha256": live_lease_proof_sha},
            "output_directory": str(output),
        }
        existing_admission = read(admission_path) if admission_path.exists() else None
        issued_at = (existing_admission.get("issued_at_utc")
                     if isinstance(existing_admission, dict) else self.now().isoformat())
        issued_stamp = _stamp(issued_at)
        resource_issued = _stamp(self.resource["issued_at_utc"])
        resource_expires = _stamp(self.resource["expires_at_utc"])
        if (issued_stamp < resource_issued
                or issued_stamp + timedelta(seconds=RECORD.EXECUTION_AUTHORIZATION_S) > resource_expires):
            raise ValueError("record admission 원본 시각이 resource lease 범위를 벗어났다")
        record_stage = self.runtime / "record" / action["job_id"]
        if existing_admission is not None and not (record_stage / "execution.json").is_file():
            age = (self.now().astimezone(UTC_TZ) - issued_stamp).total_seconds()
            if not 0.0 <= age <= RECORD.ADMISSION_MAX_AGE_S:
                raise ValueError("완료 stage가 없는 record admission이 현재 실행에는 오래됐다")
        admission = {
            "schema": RECORD.ADMISSION_SCHEMA, "status": "granted", "job_id": action["job_id"],
            "resource_kind": "remote_gpu", "lease_id": self.resource["lease_id"],
            "request_binding": RECORD.request_binding(job),
            "issued_at_utc": issued_at, "expires_at_utc": self.resource["expires_at_utc"],
            "resource_id": self.resource["resource_id"], "gpu_uuid": self.resource["gpu_uuid"],
            "resource_parent_sha256": resource_parent_sha,
            "live_lease_proof_sha256": live_lease_proof_sha,
        }
        if existing_admission is not None:
            if existing_admission != admission:
                raise ValueError("기존 record admission이 다르다")
        else:
            write_new(admission_path, admission)
        parent = {
            "schema": "vision_record_resource_parent_v1", "job_id": action["job_id"],
            "resource_proof_sha256": self.resource_sha, "resource_id": self.resource["resource_id"],
            "gpu_uuid": self.resource["gpu_uuid"],
            "raw_gpu_receipt_sha256": self.resource["raw_gpu_receipt_sha256"],
            "budget_admission_sha256": self.resource["budget_admission_sha256"],
            "compute_started_at_utc": self.resource["compute_started_at_utc"],
            "admission_sha256": sha256(admission_path),
        }
        if parent_path.exists():
            if read(parent_path) != parent:
                raise ValueError("기존 record resource parent가 다르다")
        else:
            write_new(parent_path, parent)
        job["remote_gpu_admission"].update(
            {"path": str(admission_path), "sha256": sha256(admission_path)}
        )
        return job, {"path": str(parent_path), "sha256": sha256(parent_path)}

    def _execute(self, action: Mapping[str, Any], completed: Sequence[dict[str, Any]]) -> dict[str, Any]:
        backend = self._backend()
        if action["kind"] == "train":
            artifact = backend.train(action["job_id"])
            if artifact.get("status") == "HOLD":
                return artifact
            return {"kind": "train", "action_id": action["action_id"], "artifact": artifact}
        by_job = {raw["job_id"]: raw["artifact"] for raw in completed if raw["kind"] == "train"}
        if action["kind"] == "evaluate":
            pair = {"NONE": by_job[action["none_job_id"]], "F1": by_job[action["f1_job_id"]]}
            evaluation = backend.evaluate(
                self.runtime / "evaluate" / action["action_id"], pair, action["run_name"]
            )
            bands = read(self.repo / self.protocol["evaluation"]["bands_path"])["bands"]
            gate = RECOVERY.recovery(evaluation["reports"]["F1"], evaluation["reports"]["NONE"], bands)
            return {"kind": "evaluate", "action_id": action["action_id"], "seed": action["seed"],
                    "evaluation": evaluation, "source_gate": gate}
        artifact = by_job[action["source_job_id"]]
        job, parent = self._record_job(action, artifact)
        recorded = backend.record(job)
        if recorded.get("model_use_ready") is not False:
            raise ValueError("render-sync smoke 전 record를 model-ready로 쓸 수 없다")
        return {"kind": "record", "action_id": action["action_id"],
                "source_job_id": action["source_job_id"], "record": recorded,
                "resource_parent": parent}

    def advance(self) -> dict[str, Any]:
        self._verify_bound_inputs()
        self._verify_source()
        self._verify_resource(read(self.resource_path))
        backend = self._backend()
        with backend.action_admission():
            return self._advance_admitted()

    def _advance_admitted(self) -> dict[str, Any]:
        completed: list[dict[str, Any]] = []
        actions = self.protocol["source_action_order"]
        for index, action in enumerate(actions):
            folder = self._action_folder(index, action)
            intent_path, receipt_path = folder / "intent.json", folder / "receipt.json"
            if receipt_path.exists():
                completed.append(self._raw(index))
                continue
            if intent_path.exists():
                self._initialize()
                return {"status": "HOLD", "reason": "source_action_pending",
                        "action_id": action["action_id"], "completed": len(completed)}
            limit = self.protocol["resources"]["action_limits"][action["kind"]]
            current = self.now().astimezone(UTC_TZ)
            cost = self._lifecycle_cost(current)
            deadline = _stamp(self.protocol["resources"]["deadline_utc"])
            if current + timedelta(seconds=limit["max_wall_s"]) > deadline:
                return {"status": "HOLD", "reason": "source_deadline_reached", "completed": len(completed)}
            if (cost + float(limit["max_compute_usd"])
                    > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9):
                return {"status": "HOLD", "reason": "whole_batch_budget_reached", "completed": len(completed)}
            self._initialize()
            folder.mkdir(parents=True, exist_ok=False)
            intent = {"schema": INTENT_SCHEMA, "action_id": action["action_id"], "kind": action["kind"],
                      "ordinal": index, "protocol_sha256_lf": self.protocol_sha,
                      "resource_sha256": self.resource_sha, "cost_before_usd": cost,
                      "started_at_utc": current.isoformat(),
                      "reserved_max_wall_s": limit["max_wall_s"],
                      "reserved_max_compute_usd": limit["max_compute_usd"],
                      "cost_scope": self.protocol["resources"]["cost_scope"]}
            write_new(intent_path, intent)
            raw = self._execute(action, completed)
            if raw.get("status") == "HOLD":
                return {"status": "HOLD", "reason": raw.get("reason", "backend_hold"),
                        "action_id": action["action_id"], "completed": len(completed)}
            raw["job_id"] = action.get("job_id", action.get("action_id"))
            raw_path = folder / "raw.json"
            write_new(raw_path, raw)
            action_compute = self._cost([raw])
            finished = self.now().astimezone(UTC_TZ)
            cost_after = self._lifecycle_cost(finished)
            elapsed_wall = (finished - current).total_seconds()
            if (action_compute > float(limit["max_compute_usd"]) + 1e-9
                    or elapsed_wall < 0 or elapsed_wall > float(limit["max_wall_s"])
                    or cost_after > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9
                    or finished > deadline):
                reason = ("action_wall_exceeded" if elapsed_wall < 0
                          or elapsed_wall > float(limit["max_wall_s"]) else
                          "action_compute_exceeded" if action_compute
                          > float(limit["max_compute_usd"]) + 1e-9 else
                          "whole_batch_budget_exceeded" if cost_after
                          > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9 else
                          "source_deadline_exceeded")
                write_new(folder / "invalid.json", {
                    "schema": "vision_brev_source_action_invalid_v1", "action_id": action["action_id"],
                    "reason": reason, "intent_sha256": sha256(intent_path),
                    "raw_sha256": sha256(raw_path), "cost_after_usd": cost_after,
                })
                return {"status": "HOLD", "reason": reason,
                        "action_id": action["action_id"], "completed": len(completed)}
            write_new(receipt_path, {"schema": RECEIPT_SCHEMA, "action_id": action["action_id"],
                                     "intent_sha256": sha256(intent_path), "raw_sha256": sha256(raw_path),
                                     "cost_after_usd": cost_after,
                                     "finished_at_utc": finished.isoformat()})
            completed.append(self._raw(index))
            return {"status": "advanced", "action_id": action["action_id"],
                    "completed": len(completed), "remaining": len(actions) - len(completed)}
        self._initialize()
        observed_at = self.now().astimezone(UTC_TZ)
        observed_cost = self._lifecycle_cost(observed_at)
        if observed_at > _stamp(self.protocol["resources"]["deadline_utc"]):
            return {"status": "HOLD", "reason": "source_deadline_reached",
                    "completed": len(completed)}
        if (observed_cost
                > float(self.protocol["resources"]["whole_batch_budget_usd"]) + 1e-9):
            return {"status": "HOLD", "reason": "whole_batch_budget_reached",
                    "completed": len(completed)}
        return {"status": "needs_render_sync_smoke", "model_use_ready": False,
                "source_data_ready": False, "completed": len(completed),
                "lifecycle_cost_status": "open",
                "cost_scope": self.protocol["resources"]["cost_scope"],
                "lifecycle_cost_observed_usd": observed_cost,
                "source_gates": [raw["source_gate"] for raw in completed if raw["kind"] == "evaluate"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "advance"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256-lf", required=True)
    parser.add_argument("--source-anchor", required=True)
    parser.add_argument("--resource-proof", required=True, type=Path)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--isaaclab", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args(argv)
    batch = SourceBatch(
        ROOT, args.protocol, args.protocol_sha256_lf, args.source_anchor,
        args.resource_proof, args.resource_sha256, args.isaaclab, args.python,
    )
    result = batch.verify() if args.command == "verify" else batch.advance()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] not in {"HOLD"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
