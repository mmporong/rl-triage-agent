# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""검증된 lineage source-v2 입력을 기존 paired runtime probe 경계에 연결한다.

이 모듈은 CPU 입력 봉인과 기존 probe/host 모듈의 격리 인스턴스만 제공한다.
승인, 원장 event, GPU 실행 또는 모델 호출을 새로 만들지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_source_batch_v2 as BATCH
import lineage_source_inputs_v2 as INPUT
import local_pair_probe_host_v1 as BASE_HOST
import optimizer_runtime_facts_v1 as RUNTIME
import runtime_observation_probe_v1 as BASE_PROBE

SCHEMA = "lineage_source_probe_input_v1"
SOURCE_KIND = "lineage_source_v2_observation"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_FAULTS = {
    "NONE": [], "F1": ["F1"], "F2": ["F2"], "F3": ["F3"],
    "F5": ["F5"], "COMPOUND": ["F1", "R1"],
}
SOURCE_FILES = tuple(sorted({
    *BASE_HOST.SOURCE_FILES,
    *BASE_PROBE.SOURCE_FILES,
    *BATCH.SOURCE_FILES,
    *BATCH.CHILD.SOURCE_FILES,
    "evals/lineage_source_inputs_v2.py",
    "tests/test_lineage_source_inputs_v2.py",
    "evals/lineage_source_probe_inputs_v1.py",
    "tests/test_lineage_source_probe_inputs_v1.py",
}))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def write_new(path: Path, value: object) -> None:
    path = Path(path).resolve()
    require(not path.exists(), "probe input은 fresh 파일이어야 한다")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def ref(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    require(resolved.is_file() and not resolved.is_symlink(), "안전한 source 파일 참조가 아니다")
    return {"path": str(resolved), "sha256": sha(resolved)}


def _checked_ref(value: object, name: str) -> Path:
    require(isinstance(value, dict) and set(value) == {"path", "sha256"}, f"{name} ref 형식")
    item = cast(dict[str, Any], value)
    path = Path(str(item["path"])).resolve(strict=True)
    require(path.is_file() and not path.is_symlink()
            and HEX64.fullmatch(str(item["sha256"])) is not None
            and sha(path) == item["sha256"], f"{name} ref SHA")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")))
        require(isinstance(value, dict), "source readback row는 JSON 객체여야 한다")
        rows.append(value)
    return rows


def _select(public: dict[str, Any], metadata: dict[str, Any], *,
            job_id: str | None, source_id: str | None) -> tuple[str, str, dict[str, Any]]:
    require((job_id is not None or source_id is not None)
            and (job_id is None or bool(job_id)) and (source_id is None or bool(source_id)),
            "job_id 또는 source_id가 필요하다")
    sources = metadata.get("sources")
    require(isinstance(sources, dict), "source-v2 private source index가 없다")
    typed = cast(dict[str, dict[str, Any]], sources)
    matches = [(job, row) for job, row in typed.items()
               if (job_id is None or job == job_id)
               and (source_id is None or row.get("public_id") == source_id)]
    require(len(matches) == 1, "job_id/source_id는 동일한 source 하나를 가리켜야 한다")
    selected_job, private = matches[0]
    selected_source_value = private.get("public_id")
    require(isinstance(selected_source_value, str)
            and public.get("sources", {}).get(selected_source_value) == private.get("binding"),
            "public source id와 private binding이 다르다")
    selected_source = cast(str, selected_source_value)
    return selected_job, selected_source, private


def _source_result(metadata: dict[str, Any], job_id: str) -> dict[str, Any]:
    result_path = _checked_ref(metadata.get("source_result"), "source result")
    result = read(result_path)
    records_value = result.get("records")
    require(result.get("schema") == BATCH.RESULT_SCHEMA and result.get("status") == "complete"
            and result.get("mode") == "all" and result.get("source_jobs_completed") == 20
            and result.get("fixed26_completed") == 20 and result.get("model_api") == 0
            and result.get("repairs") == 0 and result.get("runtime_facts_fabricated") is False
            and isinstance(records_value, list) and len(records_value) == 20,
            "source-v2 full batch 결과가 아니다")
    records = cast(list[Any], records_value)
    matches = [row for row in records if isinstance(row, dict) and row.get("job_id") == job_id]
    require(len(matches) == 1, "source-v2 result job row가 유일하지 않다")
    return cast(dict[str, Any], matches[0])


def _validated(index_path: Path, *, job_id: str | None,
               source_id: str | None) -> dict[str, Any]:
    index_path = Path(index_path).resolve(strict=True)
    public, metadata = INPUT.checked(index_path)
    selected_job, selected_source, private = _select(
        public, metadata, job_id=job_id, source_id=source_id)
    binding = private.get("binding")
    material = private.get("material")
    require(isinstance(binding, dict) and isinstance(material, dict),
            "source-v2 binding/material이 없다")
    binding = cast(dict[str, Any], binding)
    material = cast(dict[str, Any], material)
    stage = Path(str(material.get("stage", ""))).resolve(strict=True)
    require(stage.is_dir() and not stage.is_symlink(), "source-v2 stage가 안전한 폴더가 아니다")

    manifest_path = stage / "manifest.json"
    owned_path = stage / "owned-result.json"
    accepted_path = stage / "accepted.json"
    evidence_path = stage / "training" / BATCH.GUARD.EVIDENCE
    records_path = stage / "training" / BATCH.GUARD.RECORDS
    fixed_path = stage / "fixed-result.json"
    manifest = BATCH.CHILD.validate_manifest(read(manifest_path))
    require(manifest == private.get("manifest") and manifest.get("purpose") == "source"
            and manifest.get("job_id") == selected_job,
            "source-v2 manifest/index 결합이 다르다")
    evidence = BATCH.completed(stage, manifest, read(owned_path))
    checkpoint = evidence.get("checkpoint")
    require(isinstance(checkpoint, dict), "source-v2 checkpoint evidence가 없다")
    checkpoint = cast(dict[str, Any], checkpoint)
    checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve(strict=True)
    require(checkpoint_path.is_file() and not checkpoint_path.is_symlink()
            and sha(checkpoint_path) == checkpoint.get("sha256") == binding.get("checkpoint_sha256"),
            "source-v2 checkpoint bytes/index가 다르다")

    event_path = _checked_ref(material.get("event"), "source event")
    event_copy_path = _checked_ref(material.get("event_copy"), "source event copy")
    telemetry_path = _checked_ref(material.get("telemetry"), "source telemetry")
    extract_receipt_path = _checked_ref(material.get("receipt"), "telemetry extract receipt")
    require(event_path.parent == Path(str(evidence.get("log_dir", ""))).resolve()
            and sha(event_path) == sha(event_copy_path)
            and sha(telemetry_path) == binding.get("telemetry_sha256"),
            "source event/telemetry/checkpoint lineage가 다르다")
    INPUT.SELECT._telemetry_projection("case", read(telemetry_path))

    require(sha(manifest_path) == binding.get("manifest_sha256")
            and sha(evidence_path) == binding.get("evidence_sha256")
            and sha(records_path) == binding.get("records_sha256")
            and sha(fixed_path) == binding.get("fixed_result_sha256"),
            "source manifest/evidence/readback/fixed pin이 다르다")
    require(read(evidence_path) == evidence, "source evidence 재소비 결과가 다르다")
    rows = _rows(records_path)
    require(len(rows) == 1, "source readback은 완료 row 하나여야 한다")
    readback = rows[0]
    require(readback.get("checkpoint") == checkpoint
            and readback.get("update_count") == manifest.get("max_iterations") == 300
            and readback.get("ppo_readback") == evidence.get("ppo_readback")
            and readback.get("patch_readback") == evidence.get("patch_readback"),
            "source actual readback/evidence 결합이 다르다")

    result_row = _source_result(metadata, selected_job)
    accepted = read(accepted_path)
    require(accepted == result_row
            and result_row.get("checkpoint") == checkpoint
            and result_row.get("runtime_facts") is None
            and result_row.get("evidence_sha256") == sha(evidence_path)
            and result_row.get("records_sha256") == sha(records_path)
            and result_row.get("fixed26", {}).get("result_sha256") == sha(fixed_path),
            "source result/accepted/raw pin이 다르다")
    protocol_path = _checked_ref(metadata.get("protocol"), "source protocol")
    require(metadata.get("anchor") == manifest.get("source_commit")
            and sha(protocol_path, lf=True) == manifest.get("protocol_sha256_lf"),
            "source protocol/anchor/manifest 결합이 다르다")
    runtime_meta = metadata.get("runtime_observation")
    require(isinstance(runtime_meta, dict)
            and runtime_meta.get("fabricated_counts") is False
            and runtime_meta.get("actual_3_update_check")
            == "unavailable_requires_separate_gpu_run",
            "source runtime 결측 계약이 다르다")

    raw = {
        "manifest": ref(manifest_path), "owned_result": ref(owned_path),
        "accepted": ref(accepted_path), "evidence": ref(evidence_path),
        "readback": ref(records_path), "checkpoint": ref(checkpoint_path),
        "event": ref(event_path), "event_copy": ref(event_copy_path),
        "telemetry": ref(telemetry_path), "extract_receipt": ref(extract_receipt_path),
        "fixed_result": ref(fixed_path), "protocol": ref(protocol_path),
        "source_result": cast(dict[str, str], deepcopy(metadata["source_result"])),
    }
    return {
        "index": ref(index_path), "job_id": selected_job, "source_id": selected_source,
        "binding": deepcopy(binding), "raw": raw, "manifest": manifest,
        "evidence": evidence, "readback": readback,
        "effective_fault_metadata": deepcopy(private.get("effective_fault_metadata")),
    }


def _artifact(validated: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "index": validated["index"], "job_id": validated["job_id"],
        "source_id": validated["source_id"], "binding": validated["binding"],
        "raw": validated["raw"],
    }
    return {
        "schema": SCHEMA, "source_kind": SOURCE_KIND, **deepcopy(payload),
        "binding_digest": digest(payload),
        "runtime_observation": {
            "available": False, "facts": "role_scoped_false_and_null",
            "actual_3_update_check": "unavailable_requires_separate_gpu_run",
            "configured_or_effective_counts_used_as_actual": False,
        },
        "scope": {"cpu_input_bridge_only": True, "approval": False,
                  "execution_allowed": False, "gpu": False, "api": False},
    }


def collect(index_path: Path, out: Path, *, job_id: str | None = None,
            source_id: str | None = None) -> dict[str, Any]:
    """source-v2 index의 source 하나를 paired probe용 CPU 입력으로 봉인한다."""
    validated = _validated(index_path, job_id=job_id, source_id=source_id)
    result = _artifact(validated)
    write_new(out, result)
    return deepcopy(result)


def checked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """bridge 파일과 현재 source-v2 index/raw bytes를 모두 다시 검증한다."""
    path = Path(path).resolve(strict=True)
    value = read(path)
    require(value.get("schema") == SCHEMA and value.get("source_kind") == SOURCE_KIND,
            "lineage source probe input schema가 다르다")
    index = _checked_ref(value.get("index"), "source-v2 index")
    validated = _validated(index, job_id=value.get("job_id"), source_id=value.get("source_id"))
    require(value == _artifact(validated), "lineage source probe input 또는 원문이 바뀌었다")
    return deepcopy(value), validated


def _source(path: Path, lab: Path, role: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(role in {"case", "reference"}, "probe role은 case 또는 reference여야 한다")
    collected, validated = checked(path)
    manifest = validated["manifest"]
    BATCH.CHILD.verify_binding(manifest, repo=ROOT, lab=Path(lab).resolve())
    require(set(manifest["runtime_source_sha256_lf"]) == BASE_PROBE.RUNTIME_PINS,
            "source-v2 runtime source pin은 기존 9개와 같아야 한다")
    arm = str(manifest["arm"])
    require(arm in BASE_PROBE.P1.SOURCE_ARMS, "source-v2 probe arm이 등록 범위 밖이다")
    require(arm in SOURCE_FAULTS, "source-v2 private fault lineage가 등록되지 않았다")
    faults = deepcopy(SOURCE_FAULTS[arm])
    source = {
        "kind": SOURCE_KIND, "seed": manifest["seed"], "source_arm": arm,
        "checkpoint": {"path": validated["raw"]["checkpoint"]["path"],
                       "sha256": validated["binding"]["checkpoint_sha256"]},
        "telemetry_sha256": validated["binding"]["telemetry_sha256"],
        "binding_digest": collected["binding_digest"],
        "source_manifest_sha256": validated["raw"]["manifest"]["sha256"],
        "runtime_facts": RUNTIME.unavailable_facts(role),
        "original_faults": faults, "applied_corrections": [],
        "remaining_faults": deepcopy(faults),
        "readback": validated["raw"]["readback"]["path"],
    }
    metadata = {
        "schema": SCHEMA + "_verified_source", "job_id": collected["job_id"],
        "source_id": collected["source_id"], "lab": str(Path(lab).resolve()),
        "raw": deepcopy(validated["raw"]),
        "effective_fault_metadata": deepcopy(validated["effective_fault_metadata"]),
    }
    return collected, metadata, source


def _effective_fault(source: dict[str, Any]) -> str:
    arm_value = source.get("source_arm")
    require(isinstance(arm_value, str) and arm_value in BASE_PROBE.P1.LEGACY_FAULT,
            "legacy probe source arm이 등록되지 않았다")
    arm = cast(str, arm_value)
    return str(BASE_PROBE.P1.LEGACY_FAULT[arm])


def _residual_manifest(protocol: dict[str, Any], source: dict[str, Any], stage: Path,
                       anchor: str, protocol_path: Path) -> dict[str, Any]:
    arm = source["source_arm"]
    require(arm in BASE_PROBE.P1.OPTIMIZER_ARM, "optimizer probe source arm이 등록되지 않았다")
    value = {
        "schema": BASE_PROBE.P1.RESIDUAL.SCHEMA, "mode": "inspection",
        "arm": BASE_PROBE.P1.OPTIMIZER_ARM[arm], "seed": source["seed"],
        "run_name": f"runtime_observation_source_v2_s{source['seed']}_optimizer",
        "task": BASE_PROBE.P1.RESIDUAL.TASK, **BASE_PROBE.P1.RESIDUAL.FIXED,
        "max_iterations": 3, "source_commit_sha": anchor,
        "protocol_path": Path(protocol_path).resolve().relative_to(ROOT).as_posix(),
        "protocol_sha256_lf": sha(protocol_path, lf=True),
        "source_sha256_lf": protocol["source_sha256_lf"],
        "runtime_source_sha256_lf": protocol["runtime_source_sha256_lf"],
        "records_path": str((stage / "records.jsonl").resolve()),
        "readback_path": str((stage / "readback.json").resolve()),
        "source_checkpoint_sha256": source["checkpoint"]["sha256"],
        "original_faults": deepcopy(BASE_PROBE.P1.OPTIMIZER_FAULTS[arm]),
        "applied_corrections": [],
    }
    return BASE_PROBE.P1.RESIDUAL.validate_manifest(value)


def _isolated(module: ModuleType, name: str) -> ModuleType:
    path = Path(str(module.__file__)).resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    clone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clone)
    return clone


_PROBE = _isolated(BASE_PROBE, "_lineage_source_v2_runtime_probe_v1")
_PROBE.__dict__["SOURCE_FILES"] = SOURCE_FILES
_PROBE.__dict__["_source"] = _source
_PROBE.__dict__["_effective_fault"] = _effective_fault
_PROBE.__dict__["_residual_manifest"] = _residual_manifest
_HOST = _isolated(BASE_HOST, "_lineage_source_v2_local_pair_host_v1")
_HOST.__dict__["OBS"] = _PROBE

AUTH_SCHEMA = _PROBE.AUTH_SCHEMA
PROTOCOL_SCHEMA = _PROBE.PROTOCOL_SCHEMA


def prepare_protocol(base_probe_protocol: Path) -> dict[str, Any]:
    """기존 component와 bridge source closure를 결합한 protocol 본문을 만든다."""
    return cast(dict[str, Any], _PROBE.prepare_protocol(base_probe_protocol))


def verify_protocol(path: Path, lab: Path, anchor: str) -> dict[str, Any]:
    return cast(dict[str, Any], _PROBE.verify_protocol(path, lab, anchor))


def prepare_probe(collected_input: Path, probe: str, *, stage: Path, lab: Path,
                  protocol_path: Path, anchor: str, role: str) -> dict[str, Any]:
    return cast(dict[str, Any], _PROBE.prepare_probe(
        collected_input, probe, stage=stage, lab=lab, protocol_path=protocol_path,
        anchor=anchor, role=role))


def execute_probe(plan_path: Path, authorization_path: Path, *,
                  owned: Any = None, gpu_reader: Any = None) -> dict[str, Any]:
    return cast(dict[str, Any], _PROBE.execute_probe(
        plan_path, authorization_path, owned=owned, gpu_reader=gpu_reader))


def verify_receipt(plan_path: Path, receipt_path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], _PROBE.verify_receipt(plan_path, receipt_path))


def execute_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """기존 host 로직을 source-v2 probe 인스턴스로만 격리 실행한다."""
    return cast(dict[str, Any], _HOST.execute_pair(*args, **kwargs))


def verify_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return cast(dict[str, Any], _HOST.verify_pair(*args, **kwargs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("collect")
    build.add_argument("--index", type=Path, required=True)
    selector = build.add_mutually_exclusive_group(required=True)
    selector.add_argument("--job-id")
    selector.add_argument("--source-id")
    build.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check")
    check.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "collect":
        collect(args.index, args.out, job_id=args.job_id, source_id=args.source_id)
    else:
        checked(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
