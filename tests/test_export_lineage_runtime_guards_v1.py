# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / name) for name in ("bench", "evals", "src")]

import export_lineage_runtime_guards_v1 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.json_bytes(value))
    return path


def update_manifest(out: Path, name: str) -> None:
    manifest = M.read(out / "manifest.json")
    manifest["files"][name] = {
        "sha256": M.sha(out / name), "size_bytes": (out / name).stat().st_size}
    save(out / "manifest.tmp", manifest)
    (out / "manifest.json").unlink()
    (out / "manifest.tmp").rename(out / "manifest.json")


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path / "repo"
    (root / "evals/results").mkdir(parents=True)
    monkeypatch.setattr(M, "ROOT", root)
    anchor = "a" * 40
    design = {
        "case_plan": [
            {"case_id": "case-01", "arm": "NONE"},
            {"case_id": "case-03", "arm": "F1"},
        ]
    }
    design_path = save(root / "bench/protocols/design.json", design)
    protocol = {
        "source_anchor": anchor,
        "study_design": {"path": str(design_path), "sha256_lf": M.sha(design_path, lf=True)},
    }
    protocol_path = save(root / "bench/protocols/protocol.json", protocol)
    monkeypatch.setattr(M.DRIVER, "verify_protocol", lambda path, _anchor: M.read(path))
    index_path = save(tmp_path / "private/index.json", {"schema": "fixture-index"})
    source = {
        "case_id": "case-03", "qualification": {"arm": "F1"},
        "index": {"path": str(index_path), "sha256": M.sha(index_path)},
        "source_binding": {
            "case": {"checkpoint_sha256": "1" * 64},
            "reference": {"checkpoint_sha256": "2" * 64},
        },
    }
    source_path = save(tmp_path / "private/source.json", source)

    transport = tmp_path / "transport"
    save(transport / "manifest.json", {"schema": "transport"})
    calls: dict[str, dict] = {}
    inventory_names = (
        "dispatch-claim.json", "intent.json", "parent-exit.json", "parent-launch.json",
        "parent-process.json", "parent-stderr.txt", "parent-stdout.txt", "receipt.json",
        "response.json", "sent-claim.json", "worker-exit.json", "worker-process.json",
        "worker-response.json",
    )
    for ordinal, case in ((1, "01"), (5, "03")):
        call = transport / "calls" / f"{ordinal:04d}"
        for name in inventory_names:
            path = call / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{ordinal}:{name}\n", encoding="utf-8")
        request = {
            "schema": "local_pair_nim_request_v1",
            "request_id": f"case_{case}_logs_only_logs_only_1",
            "run_id": f"case_{case}_logs_only", "arm": "logs_only", "step": 1,
            "source_binding": {"pair_sha256": "3" * 64},
        }
        save(transport / "requests" / f"{request['request_id']}.json", {
            "request_id": request["request_id"], "ordinal": ordinal,
            "request_digest": M.digest(request),
        })
        calls[f"{ordinal:04d}"] = {
            "schema": M.TRANSPORT.VERIFIED_SCHEMA, "status": "completed",
            "outcome": "semantic_valid", "live": True, "ordinal": ordinal,
            "call_ref": {"path": str(call), "sha256": M.sha(call / "receipt.json")},
            "request": request, "request_ref": {}, "content": "{}", "usage": {},
            "wall_s": 1.0, "worker_evidence": {},
        }
    monkeypatch.setattr(
        M.TRANSPORT, "verify_call", lambda path: deepcopy(calls[Path(path).name]))

    classes = {
        "actual_model_sdk_receipts": "actual_model_sdk_raw",
        "registered_input_missing": "controlled_perturbation",
        "registered_input_permission": "controlled_perturbation",
        "blur_frame_quarantine": "controlled_perturbation",
        "visual_numeric_conflict": "controlled_perturbation",
        "visual_numeric_unknown": "controlled_perturbation",
        "gpu_busy_pre_dispatch": "controlled_injection",
        "typed_transport_timeout": "controlled_perturbation",
        "pending_no_automatic_resend": "controlled_perturbation",
        "completed_receipt_reuse": "fixture_transport",
    }
    reasons = {
        "registered_input_missing": "registered_input_unavailable",
        "registered_input_permission": "registered_input_unavailable",
        "blur_frame_quarantine": "visual_quality_unknown",
        "visual_numeric_conflict": "verified_visual_numeric_conflict",
        "visual_numeric_unknown": "visual_or_numeric_unknown",
        "gpu_busy_pre_dispatch": "gpu_resource_busy",
        "typed_transport_timeout": "typed_transport_error",
        "pending_no_automatic_resend": "transport_pending_no_resend",
    }
    rows = []
    for case in M.EXPECTED_CASES:
        result: dict = {
            "decision": "HOLD", "reason": reasons.get(case), "stage": "guard",
            "evidence": {},
        }
        notes = None
        status = "HOLD"
        if case == "actual_model_sdk_receipts":
            call = calls["0005"]
            result = {"status": "verified", "calls": [{
                "call_ref": deepcopy(call["call_ref"]), "status": "completed",
                "outcome": "semantic_valid", "live": True,
            }]}
            status = "verified"
        elif case == "blur_frame_quarantine":
            result["evidence"] = {
                "image_count": 0, "outcomes": ["quality_below_registered_floor"]}
            notes = "laplacian_variance=0.0; frame_refs=0; image_count=0"
        elif case in {"visual_numeric_conflict", "visual_numeric_unknown"}:
            result["evidence"] = {"visual_status": "unclear", "numeric_status": "unknown"}
        elif case == "gpu_busy_pre_dispatch":
            result["evidence"] = {"status": "busy", "observation_id": "injected-1"}
        elif case == "completed_receipt_reuse":
            result = {
                "status": "verified", "actual_backend_admissible": False,
                "send_calls": 1,
                "transport_status": {"automatic_retries": 0, "completed": 1},
            }
            status = "verified"
        rows.append({
            "case": case, "evidence_class": classes[case], "status": status,
            "approval_granted": False, "dispatch_granted": False,
            "automatic_retry": False, "result": result, "notes": notes,
        })

    matrix_dir = tmp_path / "actual/matrix-v1"
    save(matrix_dir / "driver-busy/result.json", {"status": "HOLD"})
    save(matrix_dir / "fixture-transport/manifest.json", {"fixture": True})
    matrix = {
        "schema": "lineage_input_runtime_matrix_v1_result", "status": "complete",
        "scope": "actual_source_cpu_matrix", "row_count": 10,
        "actual_api_calls": 0, "gpu_operations": 0, "source_files_mutated": False,
        "fixture_results_promoted_to_actual": False,
        "claims": {"actual_api_outage": False, "semantic_blur_recognition": False,
                   "actual_gpu_busy": False, "model_effect": False},
        "source": {"path": str(source_path), "sha256": M.sha(source_path)},
        "index": {"path": str(index_path), "sha256": M.sha(index_path)},
        "protocol": {"path": str(protocol_path), "sha256": M.sha(protocol_path)},
        "anchor": anchor, "source_digest": "4" * 64, "rows": rows,
    }
    matrix_path = save(matrix_dir / "matrix.json", matrix)
    manifest = {
        "schema": "lineage_input_runtime_matrix_v1_manifest",
        "files": {
            "matrix.json": M.sha(matrix_path),
            "driver-busy/result.json": M.sha(matrix_dir / "driver-busy/result.json"),
            "fixture-transport/manifest.json": M.sha(
                matrix_dir / "fixture-transport/manifest.json"),
        },
    }
    manifest_path = save(matrix_dir / "manifest.json", manifest)
    call1 = transport / "calls/0001"
    inventory = M._inventory(call1)
    verified1 = calls["0001"]
    proof = {
        "schema": "actual_completed_nim_receipt_reuse_v1", "status": "verified",
        "evidence_class": "actual_completed_receipt_CPU_reuse",
        "existing_live_call": {
            "path": str(call1), "receipt_sha256": M.sha(call1 / "receipt.json"),
            "intent_sha256": M.sha(call1 / "intent.json"),
            "request_digest": M.digest(verified1["request"]),
        },
        "verified_return_digest": M.digest(verified1), "same_verified_result": True,
        "selected_call_inventory_before": inventory,
        "selected_call_inventory_after": deepcopy(inventory),
        "source_call_files_unchanged": True, "request_index_unchanged": True,
        "transport_manifest_unchanged": True, "sender_callbacks": 0,
        "new_http_requests": 0, "new_dispatch": 0, "approval_granted": False,
        "gpu": 0, "other_live_batch_calls_may_advance": True,
    }
    proof_path = save(tmp_path / "actual/completed-reuse-v2.json", proof)
    return {"matrix": matrix_path, "manifest": manifest_path, "proof": proof_path,
            "out": root / "evals/results/tag", "call1": call1}


def test_export_and_verify_preserve_contexts_privacy_and_no_http(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    before = {name: M.sha(path) for name, path in paths.items() if name != "out" and path.is_file()}
    manifest = M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])
    assert manifest == M.verify_export(paths["out"])
    summary = M.read(paths["out"] / "summary.json")
    assert summary["counts"]["existing_model_responses_read_and_verified"] == 2
    assert summary["counts"]["exporter_new_http_requests"] == 0
    assert summary["contexts"]["diagnosis_input"]["source_type"] == "F1"
    assert summary["contexts"]["cached_normal_shadow"]["source_type"] == "NONE"
    public = "".join(path.read_text(encoding="utf-8") for path in paths["out"].iterdir())
    assert str(tmp_path) not in public and "bench/private" not in public
    after = {name: M.sha(path) for name, path in paths.items() if name != "out" and path.is_file()}
    assert before == after


def test_verify_rejects_tamper_extra_file_and_private_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])
    (paths["out"] / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="미등록"):
        M.verify_export(paths["out"])
    (paths["out"] / "extra.json").unlink()
    summary = M.read(paths["out"] / "summary.json")
    summary["leak"] = r"C:\Users\example\private.json"
    save(paths["out"] / "summary.tmp", summary)
    (paths["out"] / "summary.json").unlink()
    (paths["out"] / "summary.tmp").rename(paths["out"] / "summary.json")
    update_manifest(paths["out"], "summary.json")
    with pytest.raises(ValueError, match="개인·private"):
        M.verify_export(paths["out"])


def test_verify_rejects_fixture_promotion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])
    cases = M.read(paths["out"] / "cases.json")
    cases["rows"][-1]["evidence"]["actual_backend_admissible"] = True
    save(paths["out"] / "cases.tmp", cases)
    (paths["out"] / "cases.json").unlink()
    (paths["out"] / "cases.tmp").rename(paths["out"] / "cases.json")
    update_manifest(paths["out"], "cases.json")
    with pytest.raises(ValueError, match="승격"):
        M.verify_export(paths["out"])


def test_export_rejects_development_fixture_and_cached_inventory_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    matrix = M.read(paths["matrix"])
    matrix["scope"] = "development_fixture"
    save(paths["matrix"].with_suffix(".tmp"), matrix)
    paths["matrix"].unlink()
    paths["matrix"].with_suffix(".tmp").rename(paths["matrix"])
    manifest = M.read(paths["manifest"])
    manifest["files"]["matrix.json"] = M.sha(paths["matrix"])
    save(paths["manifest"].with_suffix(".tmp"), manifest)
    paths["manifest"].unlink()
    paths["manifest"].with_suffix(".tmp").rename(paths["manifest"])
    with pytest.raises(ValueError, match="actual matrix"):
        M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])

    paths = fixture(tmp_path / "drift", monkeypatch)
    (paths["call1"] / "response.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])


def test_output_is_fresh_and_cli_exposes_only_export_verify(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = fixture(tmp_path, monkeypatch)
    paths["out"].mkdir()
    with pytest.raises(ValueError, match="fresh"):
        M.export(paths["matrix"], paths["manifest"], paths["proof"], paths["out"])
    with pytest.raises(SystemExit):
        M.main(["export", "--allow-fixture"])
