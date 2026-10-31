"""완료된 lineage 비교 batch와 닫힌 NIM queue를 공개 artifact로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import lineage_comparison_batch_v1 as BATCH
import lineage_nim_transport_v1 as T

SCHEMA = "lineage_comparison_public_export_v1"
ROW_SCHEMA = "lineage_comparison_public_row_v1"
OBSERVATION_SCHEMA = "lineage_comparison_public_observation_v1"
SOURCE_SUMMARY_RELATIVE = "evals/results/lineage_source_fixed_dev_20261031_v1/summary.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[^A-Za-z0-9_])[A-Z]:[\\/]")
PRIVATE_MARKERS = ("bench/private", "bench\\private", "/home/", "/mnt/c/users/", "c:\\users\\")
SECRET_PATTERNS = (
    re.compile(r"(?i)\bnvapi-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*(?!null\b|none\b|redacted\b)[\"']?(?:bearer\s+)?[A-Za-z0-9._-]{8,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|apikey)\s*[:=]\s*[\"']?(?!null\b|none\b|redacted\b)[A-Za-z0-9._-]{8,}"),
)


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _repo_relative(path: Path, repo: Path) -> str:
    try:
        relative = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("공개 provenance 경로는 repo 내부여야 한다") from error
    _require(".." not in Path(relative).parts, "공개 provenance 상대경로가 안전하지 않다")
    return relative


def _privacy_scan(name: str, raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"공개 artifact는 UTF-8 텍스트여야 한다: {name}") from error
    normalized = text.replace("\\\\", "\\")
    lowered = normalized.lower()
    _require(not WINDOWS_ABSOLUTE.search(normalized), f"개인 Windows 절대경로가 있다: {name}")
    _require(not any(marker in lowered for marker in PRIVATE_MARKERS),
             f"private 경로 marker가 있다: {name}")
    _require(not any(pattern.search(text) for pattern in SECRET_PATTERNS),
             f"credential/key pattern이 있다: {name}")


def _safe_projection(value: Any) -> Any:
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            if isinstance(item, str) and (key.endswith("path") or Path(item).is_absolute()):
                path = Path(item)
                _require(path.is_file(), f"measurement 경로 원문이 없다: {key}")
                projected[key.removesuffix("path") + "file"] = {
                    "basename": path.name, "raw_text_sha256": _sha(path)}
            else:
                projected[key] = _safe_projection(item)
        return projected
    if isinstance(value, list):
        return [_safe_projection(item) for item in value]
    _require(value is None or isinstance(value, (str, int, float, bool)),
             "공개 measurement projection이 JSON 값이 아니다")
    return value


def _receipt_observation(receipt: dict, *, order: int, decision: int, role: str) -> tuple[dict, dict]:
    _require(receipt.get("role") in {None, role}, "paired receipt role이 다르다")
    status = receipt.get("status")
    measurement = receipt.get("measurement")
    _require(status in {"complete", "unknown"}, "paired receipt status가 다르다")
    if status == "complete":
        _require(isinstance(measurement, dict), "complete paired receipt measurement가 없다")
    elif measurement is not None:
        _require(isinstance(measurement, dict), "unknown measurement 형식이 다르다")
    execution = receipt.get("execution", {})
    _require(isinstance(execution, dict), "paired receipt execution 형식이 다르다")
    stub_keys = {"status", "measurement", "execution", "error_type", "role"}
    is_parent_unknown_stub = set(receipt) == stub_keys
    if is_parent_unknown_stub:
        _require(status == "unknown" and measurement is None and execution == {}
                 and receipt.get("role") == role
                 and isinstance(receipt.get("error_type"), str)
                 and bool(receipt["error_type"]),
                 "parent unknown stub 계약이 다르다")
    receipt_path = None
    plan = receipt.get("plan")
    if isinstance(plan, dict) and isinstance(plan.get("path"), str):
        candidate = Path(plan["path"]).resolve().parent / "probe_receipt.json"
        _require(candidate.is_file(), "paired receipt 원문이 없다")
        receipt_path = candidate
        _require(_read(candidate) == receipt, "paired receipt 원문과 embedded 값이 다르다")
    _require(receipt_path is not None or is_parent_unknown_stub,
             "paired receipt plan/원문 결합 또는 parent unknown stub이 없다")
    raw_refs = receipt.get("raw_refs", {})
    _require(isinstance(raw_refs, dict), "paired receipt raw_refs가 객체가 아니다")
    public_refs = {}
    for name, ref in raw_refs.items():
        _require(isinstance(ref, dict) and set(ref) == {"path", "sha256"},
                 "paired receipt raw ref 형식이 다르다")
        path = Path(ref["path"])
        _require(path.is_file() and _sha(path) == ref["sha256"], "paired raw ref SHA가 다르다")
        public_refs[name] = {"basename": path.name, "raw_sha256": ref["sha256"]}
    observation = {
        "schema": OBSERVATION_SCHEMA, "order": order, "decision": decision, "role": role,
        "status": status, "probe": receipt.get("probe"),
        "measurement": _safe_projection(measurement),
        "error_type": receipt.get("error_type"),
        "evidence_scope": ("parent_exception_without_probe_receipt"
                           if is_parent_unknown_stub else "probe_receipt"),
        "execution": {key: execution.get(key)
                      for key in ("exit_code", "natural_exit", "timed_out", "error_type",
                                  "cleanup_verified", "job_assigned_before_release", "wall_s")
                      if key in execution},
        "projection_rule": "numeric/string values preserved; path values become basename+raw_text_sha256",
        "raw_refs": public_refs,
    }
    provenance = {"receipt_sha256": _sha(receipt_path) if receipt_path else None,
                  "receipt_missing": receipt_path is None,
                  "raw_refs": public_refs}
    return observation, provenance


def _validate_rows(result: dict, manifest: dict, protocol: dict,
                   collected: dict, gate_rows: dict) -> tuple[list[dict], dict[str, bytes]]:
    rows = result.get("rows")
    _require(isinstance(rows, list) and len(rows) == 30, "batch result는 30행이어야 한다")
    _require(result.get("summary") == BATCH._summary(rows), "batch summary 재집계가 원문과 다르다")
    plans = manifest["execution_plan"]
    cases = {row["case_id"]: row for row in protocol["cases"]}
    _require(plans == protocol["latin_execution_plan"] and len(plans) == 30
             and len(cases) == 10, "manifest의 10-case Latin plan이 public protocol과 다르다")
    public_rows, observations = [], {}
    views: dict[str, set[str]] = {case_id: set() for case_id in cases}
    bindings: dict[str, list[dict]] = {case_id: [] for case_id in cases}
    for planned, row in zip(plans, rows, strict=True):
        _require(all(row.get(key) == planned[key] for key in (
            "order", "case_index", "policy_index", "case_id", "policy")),
            "result 행 순서가 등록 Latin plan과 다르다")
        case = cases[row["case_id"]]
        epoch = row.get("result")
        _require(isinstance(epoch, dict) and epoch.get("status") == "complete"
                 and epoch.get("evidence_scope") == "actual_backend"
                 and epoch.get("correction_executed") is False
                 and epoch.get("training_executed") is False
                 and epoch.get("epoch2_registered") is False,
                 "epoch 행은 actual backend의 진단 전용 완료 결과여야 한다")
        _require(epoch.get("case_id") == case["case_id"] and epoch.get("policy") == row["policy"]
                 and row.get("arm") == case["arm"]
                 and row.get("gate_label") == gate_rows[case["case_id"]]["label"],
                 "epoch 행의 case/policy/arm/gate 결합이 다르다")
        requests = epoch.get("requests")
        _require(type(requests) is int and 0 <= requests <= 3
                 and (row["policy"] == "model_adaptive" or requests == 0),
                 "정책별 모델 요청 수가 등록 최대 3회 계약과 다르다")
        raw_result = Path(str(row.get("result_path", ""))).resolve()
        expected_raw_result = (Path(manifest["output_root"]) / "execution" / "runs" /
                               f"{planned['order']:02d}_{case['case_id']}_{row['policy']}" /
                               "result.json").resolve()
        _require(raw_result == expected_raw_result and raw_result.is_file()
                 and _sha(raw_result) == row.get("result_sha256")
                 and _read(raw_result) == epoch, "epoch raw result/SHA/embedded JSON이 다르다")
        raw_row = Path(manifest["output_root"]) / "execution" / "rows" / f"{planned['order']:02d}.json"
        _require(raw_row.is_file() and _read(raw_row) == row, "batch raw row가 embedded 행과 다르다")
        expected_classification = BATCH._classification(case, epoch)
        _require(row.get("classification") == expected_classification,
                 "행 classification 재계산이 원문과 다르다")
        binding = epoch.get("source_binding")
        _require(isinstance(binding, dict) and set(binding) == {"case", "reference"},
                 "epoch source_binding role이 다르다")
        for role, job_field in (("case", "source_job"), ("reference", "reference_job")):
            expected = collected[case[job_field]]
            actual = binding[role]
            _require(actual == {"job_id": expected["job_id"], "seed": expected["seed"],
                                "checkpoint_sha256": expected["checkpoint_sha256"],
                                "telemetry_sha256": expected["telemetry_sha256"]},
                     f"{role} source CP/telemetry binding이 collected와 다르다")
        views[case["case_id"]].add(row.get("fresh_view_digest"))
        bindings[case["case_id"]].append(binding)

        public_steps = []
        receipt_provenance = []
        for step in epoch.get("steps", []):
            _require(isinstance(step, dict), "epoch step 형식이 다르다")
            paired = step.get("paired_receipts")
            _require(isinstance(paired, dict) and set(paired) == {"case", "reference"},
                     "epoch paired receipt role이 다르다")
            step_observations = {}
            for role in ("case", "reference"):
                observation, receipt_ref = _receipt_observation(
                    paired[role], order=planned["order"], decision=step["decision"], role=role)
                relative = f"observations/{planned['order']:02d}-{step['decision']:02d}-{role}.json"
                observations[relative] = _json_bytes(observation)
                step_observations[role] = {"path": relative,
                                           "sha256": _sha_bytes(observations[relative])}
                receipt_provenance.append({"decision": step["decision"], "role": role,
                                           **receipt_ref})
            case_receipt, reference_receipt = paired["case"], paired["reference"]
            if (case_receipt.get("status") == reference_receipt.get("status") == "complete"
                    and isinstance(case_receipt.get("measurement"), dict)
                    and isinstance(reference_receipt.get("measurement"), dict)):
                recomputed = BATCH.E.C.classify_raw(
                    step["probe"], case_receipt["measurement"], reference_receipt["measurement"])
                _require(recomputed == step.get("outcome"), "step raw classification이 원문과 다르다")
            else:
                _require(step.get("outcome") == "unknown", "unknown paired receipt outcome이 다르다")
            public_steps.append({"decision": step["decision"], "probe": step["probe"],
                                 "outcome": step["outcome"], "observations": step_observations})
        public_rows.append({
            "schema": ROW_SCHEMA, "order": planned["order"], "case_id": case["case_id"],
            "arm": case["arm"], "seed": case["seed"], "policy": row["policy"],
            "gate_label": row["gate_label"], "classification": deepcopy(row["classification"]),
            "source_binding": deepcopy(binding), "view_digest": row["fresh_view_digest"],
            "end_state": epoch.get("end_state"), "hold_reason": epoch.get("hold_reason"),
            "confirmed_subtype": epoch.get("confirmed_subtype"),
            "remaining_candidates": deepcopy(epoch.get("remaining_candidates")),
            "probe_order": deepcopy(epoch.get("probe_order")),
            "decision_count": epoch.get("decision_count"), "requests": epoch.get("requests"),
            "tokens": deepcopy(epoch.get("tokens")), "wall": deepcopy(epoch.get("wall")),
            "model_validity": deepcopy(epoch.get("model_validity")),
            "model_observation_use": deepcopy(epoch.get("model_observation_use")),
            "selected_by": deepcopy(epoch.get("selection_actor")), "steps": public_steps,
            "provenance": {"batch_row_sha256": _sha(raw_row),
                           "epoch_result_sha256": row["result_sha256"],
                           "probe_receipts": receipt_provenance},
        })
    for case_id in cases:
        _require(len(views[case_id]) == 1 and len(bindings[case_id]) == 3
                 and bindings[case_id].count(bindings[case_id][0]) == 3,
                 "case별 세 정책의 view/source 짝이 다르다")
    return public_rows, observations


def _transport_files(root: Path, status: dict, manifest: dict) -> list[Path]:
    session = T._session(root)
    _require(status.get("closed") is not None and status.get("requests") == status.get("responses")
             == status.get("dispatches") and status.get("pending") == 0
             and status.get("unknown") == 0 and status.get("failures") == 0,
             "transport는 closed이고 request/dispatch/response가 완결돼야 한다")
    _require(session.get("registration_anchor") == manifest["execution_anchor"]
             and session.get("plan") == manifest["transport_plan"]
             and session["plan"].get("global_request_cap") == 90,
             "transport session anchor/protocol/90-cap plan이 manifest와 다르다")
    worker_start, worker_result = root / "worker/start.json", root / "worker/result.json"
    _require(worker_start.is_file() and worker_result.is_file()
             and not (root / "worker/failure.json").exists(), "actual worker 완료 영수증이 없다")
    start, result = _read(worker_start), _read(worker_result)
    _require(start.get("schema") == "lineage_nim_transport_worker_start_v1"
             and start.get("evidence_scope") == "actual_wsl"
             and start.get("registration_anchor") == manifest["execution_anchor"]
             and start.get("session_sha256") == _sha(root / "session.json"),
             "actual worker start 결합이 다르다")
    _require(result.get("schema") == "lineage_nim_transport_worker_result_v1"
             and result.get("status") == "closed" and result.get("evidence_scope") == "actual_wsl"
             and result.get("posts") == status["dispatches"]
             and result.get("start_sha256") == _sha(worker_start)
             and result.get("queue_status") == status,
             "actual worker result/queue 결합이 다르다")
    files = [root / "session.json", root / "closed.json", worker_start, worker_result]
    for folder, suffix in (("requests", ".json"), ("dispatches", ".json"),
                           ("responses", ".json"), ("response-sha", ".sha256")):
        rows = sorted((root / folder).glob(f"*{suffix}"))
        _require(len(rows) == status["requests"], f"transport {folder} 수가 status와 다르다")
        files.extend(rows)
    return files


def _summary(result: dict, source_summary: dict) -> dict:
    return {
        "schema": "lineage_comparison_public_summary_v1", "status": "complete",
        "comparison": deepcopy(result["summary"]),
        "source_gate": {"planned_cases": source_summary["planned_cases"],
                        "valid_pairs": source_summary["valid_pairs"],
                        "label_counts": deepcopy(source_summary["label_counts"]),
                        "normal_host_gate": True},
        "scope": {"dataset": "fixed_development_categories", "heldout": False,
                  "new_policy_lineage": True, "correction_executed": False,
                  "recovery_claim": False, "general_advantage_claim": False,
                  "generalization_claim": False,
                  "timing": "API wall, owned child wall, and probe parent wall remain separate; total diagnosis wall was not measured",
                  "forbidden_claims": ["GPU savings", "total diagnosis speed"]},
    }


def export_completed(manifest_path: Path, result_path: Path, transport_root: Path,
                     out: Path, *, repo: Path = ROOT) -> dict:
    """완료된 actual batch만 privacy 검사 후 fresh 공개 폴더로 원문 보존 export한다."""
    manifest_path, result_path = Path(manifest_path).resolve(), Path(result_path).resolve()
    transport_root, out, repo = Path(transport_root).resolve(), Path(out).resolve(), Path(repo).resolve()
    _require(not out.exists(), "공개 export out은 fresh 경로여야 한다")
    manifest, protocol, collected, gate_rows = BATCH._validate_manifest(manifest_path)
    _require(transport_root == Path(manifest["transport_root"]).resolve(),
             "호출 transport root가 batch manifest와 다르다")
    result = _read(result_path)
    _require(result.get("schema") == BATCH.RESULT_SCHEMA and result.get("status") == "complete"
             and result.get("evidence_scope") == "actual_backend"
             and result.get("correction_executed") is False
             and result.get("training_executed") is False
             and result.get("epoch2_registered") is False,
             "actual complete 진단 batch만 공개할 수 있다")
    expected_result = Path(manifest["output_root"]) / "execution/result.json"
    _require(result_path == expected_result.resolve(), "result_path가 manifest output의 final result가 아니다")
    public_rows, files = _validate_rows(result, manifest, protocol, collected, gate_rows)
    status = T.status(transport_root)
    _require(sum(row["result"].get("requests", 0) for row in result["rows"])
             == status["requests"] == status["dispatches"] == status["responses"],
             "모델 행 요청 합계와 actual transport 수가 다르다")
    transport_files = _transport_files(transport_root, status, manifest)
    source_summary_path = repo / SOURCE_SUMMARY_RELATIVE
    source_summary = _read(source_summary_path)
    _require(source_summary.get("status") == "complete"
             and source_summary.get("planned_cases") == 10
             and source_summary.get("valid_pairs") == 10
             and source_summary.get("excluded_total") == 0
             and source_summary.get("diagnosis_executed") is False
             and source_summary.get("correction_executed") is False
             and source_summary.get("recovery_claim") is False
             and HEX64.fullmatch(str(source_summary.get("evaluation_result_sha256", ""))) is not None
             and source_summary.get("source_result_sha256") == manifest["source_result_sha256"]
             and source_summary.get("collected_index_sha256") == manifest["collected_index_sha256"]
             and source_summary.get("source_collection_anchor") == manifest["source_collection_anchor"]
             and source_summary.get("continuation_and_eval_anchor") == manifest["execution_anchor"]
             and source_summary.get("development_coupling", {}).get("heldout_claim") is False,
             "source public summary가 batch source/eval/index와 다르다")

    summary = _summary(result, source_summary)
    files["summary.json"] = _json_bytes(summary)
    for row in public_rows:
        files[f"rows/{row['order']:02d}.json"] = _json_bytes(row)
    nim_manifest = []
    for source in transport_files:
        relative = source.relative_to(transport_root).as_posix()
        raw = source.read_bytes()
        destination = f"nim/{relative}"
        files[destination] = raw
        nim_manifest.append({"path": destination, "sha256": _sha_bytes(raw), "size_bytes": len(raw)})
    provenance = {
        "schema": SCHEMA, "status": "complete", "rows": 30,
        "summary": {"path": "summary.json", "sha256": _sha_bytes(files["summary.json"])},
        "original_full_result_sha256": _sha(result_path),
        "batch_manifest_sha256": _sha(manifest_path),
        "source_result_sha256": manifest["source_result_sha256"],
        "fixed_evaluation_result_sha256": source_summary["evaluation_result_sha256"],
        "evaluation_gate_sha256": manifest["gate_sha256"],
        "collected_index_sha256": manifest["collected_index_sha256"],
        "source_public_summary_sha256": _sha(source_summary_path),
        "source_protocol": source_summary["source_protocol"],
        "source_protocol_sha256_lf": source_summary["source_protocol_sha256_lf"],
        "batch_protocol": _repo_relative(Path(manifest["protocol_path"]), repo),
        "batch_protocol_sha256": manifest["protocol_sha256"],
        "transport_protocol": _repo_relative(Path(manifest["transport_protocol"]), repo),
        "transport_protocol_sha256_lf": manifest["transport_protocol_sha256_lf"],
        "probe_protocol": _repo_relative(Path(manifest["probe_protocol"]), repo),
        "probe_protocol_sha256_lf": manifest["probe_protocol_sha256_lf"],
        "source_collection_anchor": manifest["source_collection_anchor"],
        "execution_anchor": manifest["execution_anchor"],
        "rows_manifest": [{"path": f"rows/{row['order']:02d}.json",
                           "sha256": _sha_bytes(files[f"rows/{row['order']:02d}.json"])}
                          for row in public_rows],
        "nim_raw_manifest": nim_manifest,
        "raw_artifact_git_note": "NIM raw artifacts require .gitattributes -text handling before commit",
    }
    files["manifest.json"] = _json_bytes(provenance)
    for name, raw in files.items():
        _privacy_scan(name, raw)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return deepcopy(provenance)


def verify_export(out: Path) -> dict:
    """공개 상대 참조·SHA와 privacy 경계를 CPU에서 재검증한다."""
    out = Path(out).resolve()
    manifest = _read(out / "manifest.json")
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete"
             and manifest.get("rows") == 30, "공개 export manifest가 다르다")
    refs = [manifest["summary"], *manifest["rows_manifest"], *manifest["nim_raw_manifest"]]
    for ref in refs:
        path = out / ref["path"]
        _require(path.is_file() and _sha(path) == ref["sha256"],
                 f"공개 export 상대 참조 SHA가 다르다: {ref['path']}")
        if "size_bytes" in ref:
            _require(path.stat().st_size == ref["size_bytes"], "공개 raw size가 다르다")
        _privacy_scan(ref["path"], path.read_bytes())
    _privacy_scan("manifest.json", (out / "manifest.json").read_bytes())
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("manifest", "result", "transport-root", "out", "repo"):
        export.add_argument(f"--{name}", type=Path, required=name != "repo")
    replay = commands.add_parser("verify")
    replay.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        export_completed(args.manifest, args.result, args.transport_root, args.out,
                         repo=args.repo or ROOT)
    else:
        verify_export(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
