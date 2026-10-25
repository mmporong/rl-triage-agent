"""V04 readiness 네 건에서 V05의 24개 동결 선택 입력을 만든다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import selection_comparison_view_v1 as VIEW
import vision_evidence_contract_v1 as VISION
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as COMPARISON
import vision_selection_sources_v1 as INPUTS
import vision_selection_worker_v1 as WORKER

MANIFEST_SCHEMA = "vision_selection_actual_material_v1"
PROTOCOL_SCHEMA = "vision_selection_execution_protocol_v1"
SELF = "evals/vision_selection_materialize_v1.py"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
POLICY_ORDER = (
    ("source-01", "normal"),
    ("source-01", "registered-f1"),
    ("source-02", "normal"),
    ("source-02", "registered-f1"),
)
API = {
    "total_cap": 72,
    "logical_decisions": 24,
    "max_calls_per_decision": 3,
    "semantic_feedback": 1,
    "infra_retry": 1,
    "sdk_retries": 0,
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def _read(path: Path) -> dict[str, Any]:
    return INPUTS._read(INPUTS._regular(Path(path).resolve(strict=True)))


def _sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _inside(repo: Path, path: Path, *, existing: bool) -> Path:
    candidate = Path(path).resolve(strict=existing)
    candidate.relative_to(repo)
    if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents):
        raise ValueError("materializer 경로에 symlink가 있다")
    return candidate


def _reference(repo: Path, value: object, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    path = INPUTS._relative(repo, value["path"])
    if not isinstance(value["sha256"], str) or _sha(path) != value["sha256"]:
        raise ValueError(f"{label} SHA가 다르다")
    return path


def _ref(repo: Path, path: Path, *, frozen: bool = False) -> dict[str, str]:
    value = {"path": Path(path).resolve(strict=True).relative_to(repo).as_posix(), "sha256": _sha(path)}
    if frozen:
        value["status"] = "frozen"
    return value


def _protocol_ref(repo: Path, value: object, label: str) -> tuple[Path, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256_lf"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    path = INPUTS._relative(repo, value["path"])
    if not isinstance(value["sha256_lf"], str) or _sha(path, lf=True) != value["sha256_lf"]:
        raise ValueError(f"{label} LF SHA가 다르다")
    return path, deepcopy(value)


def _merge(repo: Path, target: dict[str, str], values: Mapping[str, Any], label: str) -> None:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} SHA map이 아니다")
    for name, digest in values.items():
        path = INPUTS._relative(repo, name)
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None or _sha(path, lf=True) != digest:
            raise ValueError(f"{label} 현재 LF bytes가 다르다: {name}")
        if name in target and target[name] != digest:
            raise ValueError(f"{label} 경로 SHA가 충돌한다: {name}")
        target[name] = digest

def _source_maps(repo: Path, method: dict[str, Any], source: dict[str, Any],
                 method_path: Path, source_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    sources: dict[str, str] = {}
    evidence: dict[str, str] = {}
    _merge(repo, sources, method.get("source_sha256_lf", {}), "method source")
    _merge(repo, sources, source.get("source_sha256_lf", {}), "source protocol source")
    recording = source.get("recording")
    if not isinstance(recording, dict):
        raise TypeError("source protocol recording이 없다")
    _merge(repo, sources, recording.get("source_files_lf", {}), "recording source")
    _merge(repo, evidence, method.get("evidence_sha256_lf", {}), "method evidence")
    for relative in {*WORKER.REQUIRED_SOURCES, SELF}:
        path = INPUTS._relative(repo, relative)
        digest = _sha(path, lf=True)
        if relative in sources and sources[relative] != digest:
            raise ValueError(f"runtime source 결합이 현재 bytes와 다르다: {relative}")
        sources[relative] = digest
    for path in (method_path, source_path):
        relative = path.relative_to(repo).as_posix()
        digest = _sha(path, lf=True)
        if relative in sources and sources[relative] != digest:
            raise ValueError("protocol 경로가 source map과 충돌한다")
        evidence[relative] = digest
    if set(sources) & set(evidence):
        raise ValueError("source/evidence 경계가 겹친다")
    return dict(sorted(sources.items())), dict(sorted(evidence.items()))


def _model_text(messages: list[dict[str, Any]]) -> str:
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(str(part.get("text")) for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return "\n".join(texts).lower()


def _visible_json(messages: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for message in messages:
        content = message.get("content")
        parts = content if isinstance(content, list) else [content]
        for part in parts:
            text = part.get("text") if isinstance(part, dict) and part.get("type") == "text" else part
            if not isinstance(text, str):
                continue
            try:
                values.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return values


def _reject_forbidden_keys(value: Any) -> None:
    forbidden = {"label", "labels", "arm", "seed", "metadata"}
    if isinstance(value, dict):
        keys = {re.sub(r"[^a-z0-9]", "", str(key).lower()) for key in value}
        if any(key in forbidden or "seed" in key
               or any(token in key for token in ("sourcerole", "groundtruth"))
               for key in keys):
            raise ValueError("모델 payload에 source label·계보·checkpoint metadata가 노출됐다")
        for nested in value.values():
            _reject_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_keys(nested)
    elif isinstance(value, str):
        try:
            nested = json.loads(value)
        except json.JSONDecodeError:
            return
        if isinstance(nested, (dict, list)):
            _reject_forbidden_keys(nested)


def _reject_leak(messages: list[dict[str, Any]], exact_metadata: Sequence[object]) -> None:
    text = _model_text(messages)
    for value in _visible_json(messages):
        _reject_forbidden_keys(value)
    if any(str(value).lower() in text for value in exact_metadata if value is not None):
        raise ValueError("모델 payload에 source label·계보·checkpoint metadata가 노출됐다")


def _actual_sources(repo: Path, manifest: dict[str, Any], source_protocol: dict[str, Any],
                    readiness_verifier: Callable[[Path, Path, dict], dict]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = manifest.get("policies")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("actual manifest에는 네 정책이 필요하다")
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for expected, row in zip(POLICY_ORDER, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "seed_group", "source_role", "model_readiness", "checkpoint", "snapshot",
        }:
            raise ValueError("actual policy row 형식이 다르다")
        key = (row["seed_group"], row["source_role"])
        if key != expected:
            raise ValueError("actual policy row 순서가 고정 design과 다르다")
        readiness = _reference(repo, row["model_readiness"], "model readiness")
        checkpoint = _reference(repo, row["checkpoint"], "checkpoint")
        snapshot_path = _reference(repo, row["snapshot"], "snapshot")
        proof = readiness_verifier(repo, readiness, source_protocol)
        if proof.get("source_role") != row["source_role"] or proof.get("checkpoint_sha256") != _sha(checkpoint):
            raise ValueError("readiness role/checkpoint와 actual manifest가 다르다")
        raw = _read(readiness)
        training_artifact_path = _reference(
            repo, raw.get("source_training_artifact"), "source training artifact",
        )
        training_artifact = _read(training_artifact_path)
        training_checkpoint = training_artifact.get("checkpoint")
        training_telemetry = training_artifact.get("telemetry")
        expected_arm = "NONE" if row["source_role"] == "normal" else "F1"
        if (not isinstance(training_checkpoint, dict) or not isinstance(training_telemetry, dict)
                or training_checkpoint.get("sha256") != proof["checkpoint_sha256"]
                or HEX64.fullmatch(str(training_telemetry.get("sha256"))) is None
                or training_artifact.get("seed") != proof["source_seed"]
                or training_artifact.get("arm") != expected_arm):
            raise ValueError("readiness source training artifact의 checkpoint·telemetry 결합이 다르다")
        snapshot = VIEW.checked(_read(snapshot_path))
        record = raw.get("record")
        if not isinstance(record, dict):
            raise TypeError("readiness record 원본 위치가 없다")
        materials = {name: _reference(repo, record[name], f"record {name}")
                     for name in ("trace", "video", "measurement_1", "measurement_2", "spec_1", "spec_2")}
        if (_sha(materials["trace"]) != proof["trace_sha256"]
                or _sha(materials["video"]) != proof["video_sha256"]
                or any(_sha(materials[f"measurement_{index}"]) != proof["measurement_sha256"][window]
                       for index, window in enumerate(COMPARISON.WINDOWS, 1))):
            raise ValueError("readiness proof와 Play trace/video/measurement가 다르다")
        found[key] = {"proof": proof, "readiness": readiness, "checkpoint": checkpoint,
                      "snapshot": snapshot, "snapshot_path": snapshot_path, "materials": materials,
                      "training_binding": {
                          "checkpoint_sha256": training_checkpoint["sha256"],
                          "telemetry_sha256": training_telemetry["sha256"],
                      }}
    seeds = {group: {found[(group, role)]["proof"]["source_seed"] for role in COMPARISON.ROLES}
             for group in COMPARISON.SEED_GROUPS}
    if any(len(values) != 1 for values in seeds.values()) or len({next(iter(v)) for v in seeds.values()}) != 2:
        raise ValueError("source seed group이 source protocol 결과와 다르다")
    for group in COMPARISON.SEED_GROUPS:
        reference = found[(group, "normal")]["training_binding"]
        for role in COMPARISON.ROLES:
            source = found[(group, role)]
            bindings = source["snapshot"].get("inputs", {}).get("packet", {}).get("input_bindings")
            expected = {"case": source["training_binding"], "reference": reference}
            if bindings != expected:
                raise ValueError("selection snapshot의 case/reference가 같은 seed 학습 artifact와 다르다")
    return found


def materialize(
    repo_root: Path,
    actual_manifest_path: Path,
    output_root: Path,
    protocol_output: Path,
    *,
    readiness_verifier: Callable[[Path, Path, dict], dict] = INPUTS.verify_model_readiness,
    input_verifier: Callable[[Path, dict, Path], dict] = INPUTS.verify_inputs,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve(strict=True)
    actual_path = _inside(repo, actual_manifest_path, existing=True)
    output = _inside(repo, output_root, existing=False)
    protocol_path = _inside(repo, protocol_output, existing=False)
    locations_path = output / "input_locations.json"
    if output.exists() or protocol_path.exists() or locations_path.exists():
        raise FileExistsError("materializer output은 새 경로여야 한다")
    manifest = _read(actual_path)
    if set(manifest) != {"schema", "protocol_id", "execution_id", "method_protocol",
                        "source_protocol", "policies"} or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("actual material manifest 형식이 다르다")
    method_path, method_ref = _protocol_ref(repo, manifest["method_protocol"], "method protocol")
    source_path, source_ref = _protocol_ref(repo, manifest["source_protocol"], "source protocol")
    method, source = _read(method_path), SOURCE.validate_protocol(_read(source_path))
    if (method.get("schema") != "vision_selection_comparison_method_v1"
            or method.get("inputs", {}).get("slots") != COMPARISON.design()):
        raise ValueError("V03 method의 slot/view 순서가 현재 design과 다르다")
    sources, evidence = _source_maps(repo, method, source, method_path, source_path)
    source["_verified_sha256_lf"] = source_ref["sha256_lf"]
    policies = _actual_sources(repo, manifest, source, readiness_verifier)
    method_hashes = {name: _sha(INPUTS._relative(repo, name)) for name in CALL._METHODS}
    rows: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    audits: dict[tuple[str, str], Path] = {}
    for slot in COMPARISON.design():
        key = (slot["seed_group"], slot["source_role"])
        policy = policies[key]
        policy_root = output / f"{slot['seed_group']}-{slot['source_role']}"
        if key not in audits:
            audit = policy_root / "camera_pose_audit.json"
            _write_new(audit, INPUTS.camera_pose_audit(policy["materials"]["trace"]))
            audits[key] = audit
        window_index = COMPARISON.WINDOWS.index(slot["window"]) + 1
        slot_root = output / slot["slot_id"]
        original_spec = _read(policy["materials"][f"spec_{window_index}"])
        local_spec = deepcopy(original_spec)
        local_spec["video"]["path"] = str(policy["materials"]["video"])
        local_spec["checkpoint"]["path"] = str(policy["checkpoint"])
        local_spec["measurement"]["path"] = str(policy["materials"][f"measurement_{window_index}"])
        spec_path = slot_root / "v01_local_spec.json"
        _write_new(spec_path, local_spec)
        contract_path = VISION.prepare(spec_path, slot_root / "v01")
        contract = VISION.verify(contract_path)
        provenance = contract["provenance"]
        if (provenance["checkpoint"]["sha256"] != policy["proof"]["checkpoint_sha256"]
                or provenance["video"]["sha256"] != policy["proof"]["video_sha256"]
                or provenance["measurement"]["sha256"]
                != policy["proof"]["measurement_sha256"][slot["window"]]):
            raise ValueError("V01 contract가 readiness Play 원본과 다르다")
        binding = {"contract_sha256": _sha(contract_path),
                   "source_lineage_id": provenance["source_lineage_id"]}
        manifest_path = slot_root / "input_manifest.json"
        _write_new(manifest_path, {
            "schema": INPUTS.INPUT_SCHEMA,
            "snapshot": _ref(repo, policy["snapshot_path"]),
            "contract": _ref(repo, contract_path),
            "trace": _ref(repo, policy["materials"]["trace"]),
            "camera_pose_audit": _ref(repo, audits[key]),
            "model_readiness": _ref(repo, policy["readiness"]),
        })
        for view in slot["view_order"]:
            call_id = f"choice-{len(rows) + 1:02d}"
            capsule_ref = CALL.freeze(
                policy["snapshot"], contract_path, binding, view, slot_root / f"capsule-{view}",
            )
            frozen = CALL.prepared(capsule_ref)
            snapshot_bindings = policy["snapshot"]["inputs"]["packet"]["input_bindings"]
            _reject_leak(frozen["messages"], (
                *(snapshot_bindings[side][field]
                  for side in ("case", "reference")
                  for field in ("checkpoint_sha256", "telemetry_sha256")),
                provenance["source_lineage_id"], provenance["parent_source_lineage_id"],
            ))
            capsule_path = Path(capsule_ref["path"])
            rows.append({
                "slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
                "source_role": slot["source_role"], "window": slot["window"],
                "view_kind": view, "call_id": call_id,
                "source_seed": policy["proof"]["source_seed"],
                "source_lineage_id": provenance["source_lineage_id"],
                "parent_lineage_id": provenance["parent_source_lineage_id"], "split": "comparison",
                "method_protocol_sha256": method_ref["sha256_lf"],
                "source_protocol_sha256": source_ref["sha256_lf"],
                "capsule_method_raw_sha256": method_hashes,
                "checkpoint_sha256": policy["proof"]["checkpoint_sha256"],
                "video_sha256": policy["proof"]["video_sha256"],
                "trace_sha256": policy["proof"]["trace_sha256"],
                "contract_sha256": _sha(contract_path),
                "snapshot_sha256": _sha(policy["snapshot_path"]),
                "capsule_sha256": _sha(capsule_path),
                "messages_sha256": frozen["capsule"]["messages_sha256"],
                "camera_pose_audit_sha256": _sha(audits[key]),
            })
            locations.append({"call_id": call_id, "input_manifest": _ref(repo, manifest_path, frozen=True),
                              "capsule": _ref(repo, capsule_path, frozen=True)})
    checked_rows = COMPARISON.validate_inputs(
        rows, method_protocol_sha256=method_ref["sha256_lf"],
        source_protocol_sha256=source_ref["sha256_lf"],
    )
    locations_body = {"schema": INPUTS.LOCATIONS_SCHEMA,
                      "execution_id": manifest["execution_id"], "rows": locations}
    _write_new(locations_path, locations_body)
    protocol = {
        "schema": PROTOCOL_SCHEMA, "protocol_id": manifest["protocol_id"],
        "execution_id": manifest["execution_id"], "phase": "comparison",
        "method_protocol": method_ref, "source_protocol": source_ref,
        "rows": checked_rows, "source_sha256_lf": sources, "evidence_sha256_lf": evidence,
        "input_locations_sha256": _sha(locations_path), "api": API,
    }
    verified = input_verifier(repo, protocol, locations_path)
    if verified.get("schema") != INPUTS.VERIFIED_SCHEMA or len(verified.get("rows", [])) != 24:
        raise ValueError("생성 입력이 production verifier를 통과하지 못했다")
    _write_new(protocol_path, protocol)
    return {"protocol": _ref(repo, protocol_path), "input_locations": _ref(repo, locations_path),
            "input_root": output.relative_to(repo).as_posix(), "rows": 24}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--actual-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--protocol-output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(materialize(
        args.repo, args.actual_manifest, args.output_root, args.protocol_output,
    ), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
