# pyright: reportMissingImports=false
"""R5 actual runtime guard 근거를 경로·응답 원문 없는 공개 JSON으로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_five_policy_nim_transport_v1 as TRANSPORT
import lineage_five_policy_driver_v1 as DRIVER

SCHEMA = "lineage_runtime_guards_public_v1"
SUMMARY_SCHEMA = SCHEMA + "_summary"
CASES_SCHEMA = SCHEMA + "_cases"
PROVENANCE_SCHEMA = SCHEMA + "_provenance"
MANIFEST_SCHEMA = SCHEMA + "_manifest"
PUBLIC_FILES = ("summary.json", "cases.json", "provenance.json", "manifest.json")
EXPECTED_CASES = (
    "actual_model_sdk_receipts",
    "registered_input_missing",
    "registered_input_permission",
    "blur_frame_quarantine",
    "visual_numeric_conflict",
    "visual_numeric_unknown",
    "gpu_busy_pre_dispatch",
    "typed_transport_timeout",
    "pending_no_automatic_resend",
    "completed_receipt_reuse",
)
PRIVATE_VALUE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\|/(?:home|Users|mnt|tmp|root|private)/|"
    r"bench[\\/]private|Bearer\s+|nvapi-|sk-[A-Za-z0-9])",
    re.IGNORECASE,
)
SECRET_KEY = re.compile(
    r"^(?:api_?key|authorization|credential|secret|access_?token|pid|process_id)$",
    re.IGNORECASE,
)


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"비유한 JSON 숫자: {token}")),
    )
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def digest(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    require(not target.exists(), f"{target.name}은 fresh 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(json_bytes(value))


def _ref_path(value: object, label: str) -> Path:
    require(isinstance(value, Mapping), f"{label} ref가 없다")
    row = cast(Mapping[str, Any], value)
    path, expected = row.get("path"), row.get("sha256")
    require(isinstance(path, str) and isinstance(expected, str) and len(expected) == 64,
            f"{label} ref 형식이 다르다")
    target = Path(path).resolve(strict=True)
    require(target.is_file() and sha(target) == expected, f"{label} 현재 SHA가 다르다")
    return target


def _public_relative(path: Path, label: str) -> str:
    target = Path(path).resolve(strict=True)
    try:
        relative = target.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{label}은 저장소 공개 파일이어야 한다") from error
    require(not relative.startswith("bench/private/") and "/private/" not in relative,
            f"{label}은 공개 상대경로여야 한다")
    return relative


def _assert_public(*values: object) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                require(not (isinstance(key, str) and SECRET_KEY.fullmatch(key)),
                        "공개 JSON에 credential/PID key가 있다")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            require(PRIVATE_VALUE.search(value) is None,
                    "공개 JSON에 개인·private 경로 또는 credential 값이 있다")

    for value in values:
        visit(value)


def _matrix_manifest(matrix_path: Path, manifest_path: Path) -> dict[str, Any]:
    matrix_root = matrix_path.parent.resolve()
    manifest = read(manifest_path)
    require(manifest.get("schema") == "lineage_input_runtime_matrix_v1_manifest",
            "matrix manifest schema가 다르다")
    files = manifest.get("files")
    require(isinstance(files, dict) and files.get("matrix.json") == sha(matrix_path),
            "matrix manifest가 matrix 원문과 다르다")
    require(set(files) == {
        "matrix.json", "driver-busy/result.json", "fixture-transport/manifest.json"
    }, "matrix manifest inventory가 다르다")
    for name, expected in files.items():
        target = (matrix_root / name).resolve(strict=True)
        require(target.is_relative_to(matrix_root) and target.is_file()
                and isinstance(expected, str) and sha(target) == expected,
                f"matrix manifest 파일 SHA가 다르다: {name}")
    return manifest


def _design_case(protocol: dict[str, Any], case_id: str) -> dict[str, Any]:
    design_ref = protocol.get("study_design")
    require(isinstance(design_ref, dict), "study design ref가 없다")
    design_path = Path(str(design_ref.get("path", ""))).resolve(strict=True)
    require(design_path.is_file() and design_ref.get("sha256_lf") == sha(design_path, lf=True),
            "study design 현재 SHA가 protocol과 다르다")
    design = read(design_path)
    rows = design.get("case_plan")
    require(isinstance(rows, list), "study design case plan이 없다")
    matches = [row for row in rows if isinstance(row, dict) and row.get("case_id") == case_id]
    require(len(matches) == 1, f"study design {case_id}가 없다")
    return cast(dict[str, Any], matches[0])


def _verify_actual_call(row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    result = row.get("result")
    calls = result.get("calls") if isinstance(result, dict) else None
    require(isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict),
            "actual matrix call은 한 건이어야 한다")
    call = cast(dict[str, Any], calls[0])
    ref = call.get("call_ref")
    call_path = Path(str(ref.get("path", ""))).resolve(strict=True) if isinstance(ref, dict) else None
    require(isinstance(call_path, Path) and call_path.is_dir(), "actual call 경로가 없다")
    verified = TRANSPORT.verify_call(call_path)
    require(verified.get("schema") == TRANSPORT.VERIFIED_SCHEMA
            and verified.get("status") == "completed"
            and verified.get("outcome") == "semantic_valid"
            and verified.get("live") is True
            and verified.get("call_ref", {}).get("sha256") == ref.get("sha256")
            and call.get("status") == "completed" and call.get("outcome") == "semantic_valid"
            and call.get("live") is True,
            "matrix actual call 현재 verifier 결과가 다르다")
    return verified, call_path


def _inventory(call_path: Path) -> dict[str, str]:
    return {
        path.name: sha(path)
        for path in sorted(call_path.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _verify_cached_reuse(proof: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require(proof.get("schema") == "actual_completed_nim_receipt_reuse_v1"
            and proof.get("status") == "verified"
            and proof.get("evidence_class") == "actual_completed_receipt_CPU_reuse",
            "cached reuse proof schema/status가 다르다")
    expected_flags = {
        "same_verified_result": True,
        "source_call_files_unchanged": True,
        "request_index_unchanged": True,
        "transport_manifest_unchanged": True,
        "sender_callbacks": 0,
        "new_http_requests": 0,
        "new_dispatch": 0,
        "approval_granted": False,
        "gpu": 0,
        "other_live_batch_calls_may_advance": True,
    }
    require(all(proof.get(name) == expected for name, expected in expected_flags.items()),
            "cached reuse의 불변/무호출 경계가 다르다")
    existing = proof.get("existing_live_call")
    require(isinstance(existing, dict), "cached live call ref가 없다")
    call_path = Path(str(existing.get("path", ""))).resolve(strict=True)
    verified = TRANSPORT.verify_call(call_path)
    require(verified.get("schema") == TRANSPORT.VERIFIED_SCHEMA
            and verified.get("status") == "completed"
            and verified.get("outcome") == "semantic_valid"
            and verified.get("live") is True
            and verified.get("call_ref", {}).get("sha256") == existing.get("receipt_sha256")
            and digest(verified) == proof.get("verified_return_digest")
            and digest(verified.get("request")) == existing.get("request_digest"),
            "cached call 현재 verifier/반환 digest가 다르다")
    before, after = proof.get("selected_call_inventory_before"), proof.get(
        "selected_call_inventory_after")
    current = _inventory(call_path)
    require(isinstance(before, dict) and before == after == current and len(current) == 13,
            "cached selected call 13파일 inventory가 바뀌었다")
    require(current.get("receipt.json") == existing.get("receipt_sha256")
            and current.get("intent.json") == existing.get("intent_sha256"),
            "cached receipt/intent SHA가 proof와 다르다")
    request = verified.get("request")
    require(isinstance(request, dict) and isinstance(request.get("request_id"), str),
            "cached request가 없다")
    transport_root = call_path.parent.parent
    request_path = (transport_root / "requests" / f"{request['request_id']}.json").resolve(
        strict=True)
    request_index = read(request_path)
    require(request_index == {
        "request_id": request["request_id"], "ordinal": verified.get("ordinal"),
        "request_digest": digest(request),
    }, "cached request index가 verifier request digest와 다르다")
    manifest_path = (transport_root / "manifest.json").resolve(strict=True)
    return verified, {
        "selected_call_inventory_count": len(current),
        "selected_call_inventory_digest": digest(current),
        "request_index_sha256": sha(request_path),
        "transport_manifest_sha256": sha(manifest_path),
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    require(row.get("case") in EXPECTED_CASES
            and row.get("approval_granted") is False
            and row.get("dispatch_granted") is False
            and row.get("automatic_retry") is False,
            "matrix row 안전 상태가 다르다")
    result = row.get("result")
    require(isinstance(result, dict), "matrix row result가 없다")
    public: dict[str, Any] = {
        "case": row["case"], "evidence_class": row.get("evidence_class"),
        "status": row.get("status"), "approval_granted": False,
        "dispatch_granted": False, "automatic_retry": False,
        "decision": result.get("decision"), "reason": result.get("reason"),
        "stage": result.get("stage"),
    }
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        public["evidence"] = deepcopy(evidence)
    if row["case"] == "actual_model_sdk_receipts":
        call = result["calls"][0]
        public["evidence"] = {
            "existing_receipts_verified": 1, "status": call.get("status"),
            "outcome": call.get("outcome"), "live": call.get("live"),
            "new_http_requests": 0,
        }
    elif row["case"] == "blur_frame_quarantine":
        require(public.get("evidence", {}).get("image_count") == 0
                and "frame_refs=0" in str(row.get("notes")),
                "blur quarantine image/frame count가 다르다")
        public["evidence"]["frame_refs_count"] = 0
    elif row["case"] == "gpu_busy_pre_dispatch":
        public["evidence"]["measurement_kind"] = "controlled_injection_not_gpu_measurement"
    elif row["case"] == "completed_receipt_reuse":
        status = result.get("transport_status")
        require(result.get("actual_backend_admissible") is False
                and result.get("send_calls") == 1 and isinstance(status, dict)
                and status.get("automatic_retries") == 0 and status.get("completed") == 1,
                "fixture receipt reuse 경계가 다르다")
        public["evidence"] = {
            "fixture_send_calls": 1, "duplicate_send_calls": 0,
            "automatic_retries": 0, "actual_backend_admissible": False,
        }
    return public


def export(matrix_path: Path, matrix_manifest_path: Path, reuse_proof_path: Path,
           out: Path) -> dict[str, Any]:
    """actual 원문을 재검증하고 evals/results의 fresh 공개 tag를 생성한다."""
    matrix_path = Path(matrix_path).resolve(strict=True)
    matrix_manifest_path = Path(matrix_manifest_path).resolve(strict=True)
    reuse_proof_path = Path(reuse_proof_path).resolve(strict=True)
    output = Path(out).resolve()
    require(not output.exists(), "공개 output은 fresh 경로여야 한다")
    require(output.parent == (ROOT / "evals/results").resolve(),
            "공개 output은 evals/results/<fresh-tag>여야 한다")
    inputs_before = {path: sha(path) for path in (
        matrix_path, matrix_manifest_path, reuse_proof_path)}

    matrix = read(matrix_path)
    _matrix_manifest(matrix_path, matrix_manifest_path)
    proof = read(reuse_proof_path)
    require(matrix.get("schema") == "lineage_input_runtime_matrix_v1_result"
            and matrix.get("status") == "complete"
            and matrix.get("scope") == "actual_source_cpu_matrix"
            and matrix.get("row_count") == 10
            and matrix.get("actual_api_calls") == 0
            and matrix.get("gpu_operations") == 0
            and matrix.get("source_files_mutated") is False
            and matrix.get("fixture_results_promoted_to_actual") is False,
            "actual matrix 범위/무호출 계약이 다르다")
    require(matrix.get("claims") == {
        "actual_api_outage": False, "semantic_blur_recognition": False,
        "actual_gpu_busy": False, "model_effect": False,
    }, "actual matrix claims가 다르다")
    rows = matrix.get("rows")
    require(isinstance(rows, list) and len(rows) == 10
            and [row.get("case") for row in rows if isinstance(row, dict)]
            == list(EXPECTED_CASES), "matrix 10 row 순서가 다르다")
    typed_rows = cast(list[dict[str, Any]], rows)

    source_path = _ref_path(matrix.get("source"), "source")
    index_path = _ref_path(matrix.get("index"), "index")
    protocol_path = _ref_path(matrix.get("protocol"), "protocol")
    source = read(source_path)
    require(source.get("case_id") == "case-03"
            and source.get("qualification", {}).get("arm") == "F1"
            and source.get("index", {}).get("sha256") == sha(index_path),
            "matrix source는 case03 F1/index 결합이어야 한다")
    anchor = matrix.get("anchor")
    require(isinstance(anchor, str) and re.fullmatch(r"[0-9a-f]{40}", anchor) is not None,
            "matrix execution anchor 형식이 다르다")
    protocol = DRIVER.verify_protocol(protocol_path, anchor)
    case01, case03 = _design_case(protocol, "case-01"), _design_case(protocol, "case-03")
    require(case01.get("arm") == "NONE" and case03.get("arm") == "F1",
            "study design의 normal shadow/F1 맥락이 다르다")

    actual_row = typed_rows[0]
    actual_verified, actual_call_path = _verify_actual_call(actual_row)
    actual_request = actual_verified.get("request")
    require(isinstance(actual_request, dict)
            and actual_request.get("request_id") == "case_03_logs_only_logs_only_1"
            and actual_request.get("run_id") == "case_03_logs_only"
            and actual_request.get("arm") == "logs_only"
            and actual_call_path.name == "0005",
            "matrix actual call0005 F1 진단 맥락이 다르다")

    cached_verified, cached_files = _verify_cached_reuse(proof)
    cached_request = cached_verified.get("request")
    require(isinstance(cached_request, dict)
            and cached_request.get("request_id") == "case_01_logs_only_logs_only_1"
            and cached_request.get("run_id") == "case_01_logs_only"
            and cached_request.get("arm") == "logs_only"
            and cached_verified.get("ordinal") == 1,
            "cached call0001 normal shadow 맥락이 다르다")

    public_rows = [_public_row(row) for row in typed_rows]
    classes: dict[str, int] = {}
    for row in public_rows:
        name = cast(str, row["evidence_class"])
        classes[name] = classes.get(name, 0) + 1
    require(classes == {
        "actual_model_sdk_raw": 1, "controlled_perturbation": 7,
        "controlled_injection": 1, "fixture_transport": 1,
    }, "matrix evidence class 분포가 다르다")

    source_binding = source.get("source_binding")
    require(isinstance(source_binding, dict)
            and set(source_binding) == {"case", "reference"},
            "source checkpoint binding이 없다")
    checkpoint_sha = {
        role: cast(dict[str, Any], source_binding[role]).get("checkpoint_sha256")
        for role in ("case", "reference")
    }
    require(all(isinstance(value, str) and len(value) == 64
                for value in checkpoint_sha.values()), "checkpoint SHA 형식이 다르다")

    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete",
        "scope": "actual_source_runtime_guard_public_projection",
        "counts": {
            "matrix_rows": 10, "hold_rows": 8, "verified_rows": 2,
            "existing_model_responses_read_and_verified": 2,
            "matrix_new_http_requests": 0, "exporter_new_http_requests": 0,
            "cached_reuse_sender_callbacks": 0, "cached_reuse_new_http_requests": 0,
            "gpu_operations": 0,
        },
        "evidence_classes": classes,
        "contexts": {
            "diagnosis_input": {
                "case_id": "case-03", "source_type": "F1", "policy": "logs_only",
                "existing_call_ordinal": 5, "status": "completed",
                "outcome": "semantic_valid", "live_receipt": True,
                "role": "F1 진단 입력으로 이미 생성된 응답의 읽기 전용 검증",
            },
            "cached_normal_shadow": {
                "case_id": "case-01", "source_type": "NONE", "policy": "logs_only",
                "existing_call_ordinal": 1, "status": "completed",
                "outcome": "semantic_valid", "live_receipt": True,
                "same_verified_result": True, "sender_callbacks": 0,
                "new_http_requests": 0,
                "role": "정상 shadow의 완료 receipt를 같은 결과로 재사용한 CPU 검증",
            },
        },
        "interpretation_limits": {
            "actual_api_outage_induced": False,
            "api_outage_cause_or_rate_measured": False,
            "diagnostic_accuracy_measured": False,
            "model_effect_measured": False,
            "semantic_blur_recognition_measured": False,
            "gpu_busy_measured": False,
            "busy_is_controlled_injection": True,
            "fixture_transport_promoted_to_actual": False,
            "other_live_batch_calls_global_immutability_claimed": False,
        },
    }
    cases = {"schema": CASES_SCHEMA, "status": "complete", "rows": public_rows}
    provenance = {
        "schema": PROVENANCE_SCHEMA, "status": "complete",
        "execution_anchor": anchor,
        "public_protocol": {
            "path": _public_relative(protocol_path, "protocol"),
            "sha256": sha(protocol_path),
        },
        "sha256_only": {
            "matrix": sha(matrix_path), "matrix_manifest": sha(matrix_manifest_path),
            "cached_reuse_proof": sha(reuse_proof_path), "source": sha(source_path),
            "source_index": sha(index_path),
            "case_checkpoint": checkpoint_sha["case"],
            "reference_checkpoint": checkpoint_sha["reference"],
            "matrix_actual_receipt": actual_verified["call_ref"]["sha256"],
            "cached_actual_receipt": cached_verified["call_ref"]["sha256"],
            **cached_files,
        },
        "source_digest": matrix.get("source_digest"),
        "raw_provider_responses_included": False,
        "raw_metadata_included": False,
        "private_paths_included": False,
        "credentials_included": False,
        "pid_values_included": False,
        "fixture_promoted_to_actual": False,
    }
    _assert_public(summary, cases, provenance)

    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "summary.json", summary)
    write_new(output / "cases.json", cases)
    write_new(output / "provenance.json", provenance)
    inventory = {
        name: {"sha256": sha(output / name), "size_bytes": (output / name).stat().st_size}
        for name in PUBLIC_FILES if name != "manifest.json"
    }
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "complete",
        "release_schema": SCHEMA, "public_files": list(PUBLIC_FILES),
        "files": inventory, "overwrite_allowed": False,
    }
    _assert_public(manifest)
    write_new(output / "manifest.json", manifest)
    require(inputs_before == {path: sha(path) for path in inputs_before},
            "export 중 actual 입력 원문이 바뀌었다")
    verify_export(output)
    return deepcopy(manifest)


def verify_export(out: Path) -> dict[str, Any]:
    """공개 tag의 canonical bytes, inventory, privacy, fixture 경계를 검증한다."""
    root = Path(out).resolve(strict=True)
    manifest = read(root / "manifest.json")
    require(manifest.get("schema") == MANIFEST_SCHEMA
            and manifest.get("status") == "complete"
            and manifest.get("release_schema") == SCHEMA
            and manifest.get("public_files") == list(PUBLIC_FILES)
            and set(manifest.get("files", {})) == set(PUBLIC_FILES) - {"manifest.json"}
            and manifest.get("overwrite_allowed") is False,
            "공개 manifest schema/inventory가 다르다")
    actual = sorted(path.relative_to(root).as_posix()
                    for path in root.rglob("*") if path.is_file())
    require(actual == sorted(PUBLIC_FILES), "공개 tag에 미등록 파일이 있다")
    values = {}
    for name in PUBLIC_FILES:
        path = root / name
        value = read(path)
        require(path.read_bytes() == json_bytes(value), f"canonical JSON bytes가 아니다: {name}")
        if name != "manifest.json":
            item = manifest["files"][name]
            require(sha(path) == item.get("sha256")
                    and path.stat().st_size == item.get("size_bytes"),
                    f"공개 파일 SHA/size가 바뀌었다: {name}")
        values[name] = value
    _assert_public(*values.values())
    summary, cases, provenance = (
        values["summary.json"], values["cases.json"], values["provenance.json"])
    require(summary.get("schema") == SUMMARY_SCHEMA
            and summary.get("counts", {}).get("exporter_new_http_requests") == 0
            and summary.get("interpretation_limits", {}).get(
                "fixture_transport_promoted_to_actual") is False,
            "공개 summary 무호출/fixture 경계가 다르다")
    rows = cases.get("rows")
    require(cases.get("schema") == CASES_SCHEMA and isinstance(rows, list)
            and len(rows) == 10
            and [row.get("case") for row in rows if isinstance(row, dict)]
            == list(EXPECTED_CASES), "공개 cases 10 row가 다르다")
    fixture = next(row for row in rows if row["case"] == "completed_receipt_reuse")
    require(fixture.get("evidence_class") == "fixture_transport"
            and fixture.get("evidence", {}).get("actual_backend_admissible") is False,
            "fixture transport가 actual로 승격됐다")
    require(provenance.get("schema") == PROVENANCE_SCHEMA
            and provenance.get("private_paths_included") is False
            and provenance.get("raw_provider_responses_included") is False
            and provenance.get("fixture_promoted_to_actual") is False,
            "공개 provenance privacy/fixture 경계가 다르다")
    return deepcopy(manifest)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--matrix", type=Path, required=True)
    export_parser.add_argument("--matrix-manifest", type=Path, required=True)
    export_parser.add_argument("--reuse-proof", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export(args.matrix, args.matrix_manifest, args.reuse_proof, args.out)
    else:
        verify_export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
