"""first-class record readiness로 24개 영상 선택 입력을 검증한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import vision_record_source_readiness_v1 as READINESS  # pyright: ignore[reportMissingImports]
import vision_selection_materialize_v1 as MATERIALIZE  # pyright: ignore[reportMissingImports]
import vision_selection_sources_v1 as BASE  # pyright: ignore[reportMissingImports]


def verify_inputs(
    repo_root: Path,
    execution_protocol: dict[str, Any],
    input_locations_path: Path,
) -> dict[str, Any]:
    """기존 검증 전체를 보존하고 readiness 재계산기만 first-class 판으로 바꾼다."""
    repo = Path(repo_root).resolve(strict=True)
    locations_path = BASE._regular(Path(input_locations_path))
    locations_path.relative_to(repo)
    locations = BASE._exact(
        BASE._read(locations_path), {"schema", "execution_id", "rows"}, "입력 위치"
    )
    execution_id = execution_protocol["execution_id"]
    BASE._require(
        locations["schema"] == BASE.LOCATIONS_SCHEMA
        and locations["execution_id"] == execution_id,
        "입력 위치의 실행 결합이 다르다",
    )
    BASE._require(
        BASE._sha(locations_path) == execution_protocol["input_locations_sha256"],
        "입력 위치 SHA가 공개 사전등록과 다르다",
    )
    method_sha = execution_protocol["method_protocol"]["sha256_lf"]
    source_sha = execution_protocol["source_protocol"]["sha256_lf"]
    public_rows = BASE.COMPARISON.validate_inputs(
        execution_protocol["rows"],
        method_protocol_sha256=method_sha,
        source_protocol_sha256=source_sha,
    )
    BASE._require(
        isinstance(locations["rows"], list)
        and len(locations["rows"]) == len(public_rows),
        "입력 위치의 row 수가 다르다",
    )
    source_protocol = BASE._read(
        BASE._relative(repo, execution_protocol["source_protocol"]["path"])
    )
    source_protocol = BASE.SOURCE.validate_protocol(source_protocol)
    BASE._require(
        source_protocol.get("schema") == "vision_brev_source_protocol_v1",
        "source protocol schema가 다르다",
    )
    source_protocol_path = BASE._relative(
        repo, execution_protocol["source_protocol"]["path"]
    )
    BASE._require(
        hashlib.sha256(
            source_protocol_path.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        == source_sha,
        "source protocol 실제 LF SHA가 다르다",
    )
    source_protocol["_verified_sha256_lf"] = source_sha
    rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    proof_cache: dict[str, dict[str, Any]] = {}
    for public, location in zip(public_rows, locations["rows"], strict=True):
        BASE._exact(location, {"call_id", "input_manifest", "capsule"}, "입력 위치 row")
        BASE._require(
            location["call_id"] == public["call_id"], "입력 위치 순서가 다르다"
        )
        manifest_path = BASE._ref(repo, location["input_manifest"], frozen=True)
        manifest = BASE._exact(
            BASE._read(manifest_path),
            {
                "schema",
                "snapshot",
                "contract",
                "trace",
                "camera_pose_audit",
                "model_readiness",
            },
            "입력 원본 manifest",
        )
        BASE._require(manifest["schema"] == BASE.INPUT_SCHEMA, "입력 원본 schema가 다르다")
        paths = {
            key: BASE._ref(repo, manifest[key])
            for key in manifest
            if key != "schema"
        }
        for key, public_key in (
            ("snapshot", "snapshot_sha256"),
            ("contract", "contract_sha256"),
            ("trace", "trace_sha256"),
            ("camera_pose_audit", "camera_pose_audit_sha256"),
        ):
            BASE._require(
                BASE._sha(paths[key]) == public[public_key],
                f"{key} 공개 SHA가 다르다",
            )
        proof_key = BASE._sha(paths["model_readiness"])
        if proof_key not in proof_cache:
            proof_cache[proof_key] = READINESS.verify_model_readiness(
                repo, paths["model_readiness"], source_protocol
            )
        proof = proof_cache[proof_key]
        BASE._require(
            proof["trace_sha256"] == public["trace_sha256"]
            and proof["checkpoint_sha256"] == public["checkpoint_sha256"]
            and proof["video_sha256"] == public["video_sha256"]
            and proof["source_seed"] == public["source_seed"]
            and proof["source_role"] == public["source_role"],
            "원본 readiness와 공개 row가 다르다",
        )
        audit_key = public["trace_sha256"]
        if audit_key not in proof_cache:
            proof_cache[audit_key] = BASE.camera_pose_audit(paths["trace"])
        BASE._require(
            BASE._read(paths["camera_pose_audit"]) == proof_cache[audit_key],
            "camera 감사가 원본 trace와 다르다",
        )
        snapshot = BASE._read(paths["snapshot"])
        contract = BASE.CONTRACT.VISION.verify(paths["contract"])
        provenance = contract["provenance"]
        BASE._require(
            provenance["source_lineage_id"] == public["source_lineage_id"]
            and provenance["parent_source_lineage_id"] == public["parent_lineage_id"]
            and provenance["checkpoint"]["sha256"] == public["checkpoint_sha256"]
            and provenance["video"]["sha256"] == public["video_sha256"],
            "V01 공개 계보가 다르다",
        )
        BASE._require(
            provenance["measurement"] is not None
            and provenance["measurement"]["sha256"]
            == proof["measurement_sha256"][public["window"]],
            "V01 measurement 원본 SHA가 record artifact와 다르다",
        )
        BASE._verify_window(contract, BASE._read(paths["trace"]), public["window"])
        capsule_path = BASE._ref(repo, location["capsule"], frozen=True)
        BASE._require(
            BASE._sha(capsule_path) == public["capsule_sha256"],
            "공개 capsule SHA가 다르다",
        )
        capsule_ref = {
            "path": str(capsule_path),
            "sha256": public["capsule_sha256"],
            "status": "frozen",
        }
        frozen = BASE.CALL.prepared(capsule_ref)
        BASE._require(
            frozen["capsule"]["messages_sha256"] == public["messages_sha256"]
            and frozen["capsule"]["method_sha256"]
            == public["capsule_method_raw_sha256"]
            and frozen["view_kind"] == public["view_kind"],
            "공개 capsule method/view 결합이 다르다",
        )
        for relative, sha in public["capsule_method_raw_sha256"].items():
            BASE._require(
                BASE._sha(BASE._relative(repo, relative)) == sha,
                "현재 capsule method bytes가 다르다",
            )
        binding = {
            "contract_sha256": public["contract_sha256"],
            "source_lineage_id": public["source_lineage_id"],
        }
        BASE.CONTRACT.verify(frozen["bundle"], snapshot, paths["contract"], binding)
        BASE._require(
            BASE.CONTRACT.messages(
                frozen["bundle"], snapshot, paths["contract"], binding
            )
            == frozen["messages"],
            "동결 messages가 원본에서 재계산한 payload와 다르다",
        )
        rows.append(
            {
                "call_id": public["call_id"],
                "run_id": f"vision-selection-{execution_id}-{public['call_id']}",
                "capsule": capsule_ref,
            }
        )
        manifest_hashes[public["call_id"]] = BASE._sha(manifest_path)
    return {
        "schema": BASE.VERIFIED_SCHEMA,
        "execution_id": execution_id,
        "rows": rows,
        "input_manifest_sha256": manifest_hashes,
        "source_sha256_lf": execution_protocol["source_sha256_lf"],
        "evidence_sha256_lf": execution_protocol["evidence_sha256_lf"],
        "input_locations_sha256": BASE._sha(locations_path),
    }


def materialize(
    repo_root: Path,
    actual_manifest_path: Path,
    output_root: Path,
    protocol_output: Path,
) -> dict[str, Any]:
    """first-class readiness를 materializer와 생성 후 입력 검사 양쪽에 주입한다."""
    return MATERIALIZE.materialize(
        repo_root,
        actual_manifest_path,
        output_root,
        protocol_output,
        readiness_verifier=READINESS.verify_model_readiness,
        input_verifier=verify_inputs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--repo", type=Path, default=ROOT)
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--input-locations", type=Path, required=True)
    create = subparsers.add_parser("materialize")
    create.add_argument("--repo", type=Path, default=ROOT)
    create.add_argument("--actual-manifest", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--protocol-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        protocol = BASE._read(BASE._regular(args.protocol.resolve(strict=True)))
        result = verify_inputs(args.repo, protocol, args.input_locations)
    else:
        result = materialize(
            args.repo, args.actual_manifest, args.output_root, args.protocol_output
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
