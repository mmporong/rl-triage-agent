from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]
M = importlib.import_module("lineage_probe_failure_evidence_v3")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_post_child_permission_error_preserves_detail_and_original_unknown(tmp_path):
    stage = (tmp_path / "probe").resolve()
    plan = stage / "plan.json"
    authorization = tmp_path / "authorization.json"
    _write(plan, {"stage": str(stage)})
    _write(authorization, {"request_id": "request-1"})
    receipt = stage / "probe_receipt.json"
    owned = stage / "owned-result.json"
    _write(receipt, {"status": "unknown", "error_type": "PermissionError"})
    _write(owned, {"exit_code": 0, "natural_exit": True})
    receipt_before = receipt.read_bytes()
    target = stage / "records.jsonl"
    error = PermissionError(13, "Access is denied", str(target))
    error.winerror = 5

    with patch.object(M, "_ORIGINAL_EXECUTE_PROBE", side_effect=error), \
            pytest.raises(PermissionError) as caught:
        M.execute_probe(plan, authorization)

    assert caught.value is error
    assert receipt.read_bytes() == receipt_before
    detail = json.loads((stage / M.DETAIL_NAME).read_text(encoding="utf-8"))
    assert detail["error"] == {
        "type": "PermissionError",
        "message": str(error),
        "traceback": detail["error"]["traceback"],
        "errno": 13,
        "winerror": 5,
        "filename": str(target),
    }
    assert "PermissionError" in detail["error"]["traceback"]
    assert detail["plan"] == {"path": str(plan), "sha256": _sha(plan)}
    assert detail["authorization"] == {
        "path": str(authorization.resolve()), "sha256": _sha(authorization)
    }
    assert detail["probe_receipt"] == {"path": str(receipt), "sha256": _sha(receipt)}
    assert detail["owned_result"] == {"path": str(owned), "sha256": _sha(owned)}
    assert detail["scope"] == {
        "retry": False,
        "measurement_changed": False,
        "original_probe_receipt_changed": False,
    }


def test_existing_sidecar_rejects_duplicate_without_backend_retry(tmp_path):
    stage = (tmp_path / "probe").resolve()
    plan = stage / "plan.json"
    authorization = tmp_path / "authorization.json"
    _write(plan, {"stage": str(stage)})
    _write(authorization, {})
    detail = stage / M.DETAIL_NAME
    detail.write_text("preserved\n", encoding="utf-8")
    calls = []
    with patch.object(M, "_ORIGINAL_EXECUTE_PROBE", side_effect=lambda *_: calls.append(True)), \
            pytest.raises(FileExistsError, match="이미 존재"):
        M.execute_probe(plan, authorization)
    assert calls == [] and detail.read_text(encoding="utf-8") == "preserved\n"


def test_admission_base_exception_escapes_without_backend_sidecar(tmp_path):
    stage = (tmp_path / "probe").resolve()
    plan = stage / "plan.json"
    authorization = tmp_path / "authorization.json"
    _write(plan, {"stage": str(stage)})
    _write(authorization, {})
    blocked = M.A.ProbeAdmissionBlocked("gpu admission blocked")
    with patch.object(M, "_ORIGINAL_EXECUTE_PROBE", side_effect=blocked), \
            pytest.raises(M.A.ProbeAdmissionBlocked) as caught:
        M.execute_probe(plan, authorization)
    assert caught.value is blocked
    assert not (stage / M.DETAIL_NAME).exists()


def test_run_epoch_uses_admission_lifecycle_and_restores_backend(tmp_path):
    base = tmp_path / "base.json"
    base.write_text("{}", encoding="utf-8")
    seen = []

    def epoch(*args, **kwargs):
        assert M.A.E.P is M.A
        assert M.A.execute_probe is M.execute_probe
        assert "probe_executor" not in kwargs
        seen.append("inside")
        raise RuntimeError("epoch-stop")

    with patch.object(M, "ROOT", tmp_path), patch.object(M.A, "ROOT", tmp_path), \
            patch.object(M, "verify_protocol", return_value={
                "component_admission_protocol": {"path": "admission.json"}
            }), patch.object(M.A, "verify_protocol", return_value={
                "component_probe_protocol": {"path": "base.json"}
            }), patch.object(M.A.E, "run_epoch", side_effect=epoch), \
            pytest.raises(RuntimeError, match="epoch-stop"):
        M.run_epoch(failure_protocol=tmp_path / "failure.json", anchor="a" * 40,
                    probe_protocol=base)
    assert seen == ["inside"]
    assert M.A.E.P is M.A.P and M.A.execute_probe is M._ORIGINAL_EXECUTE_PROBE


def test_default_epoch_path_keeps_actual_backend_evidence_scope(tmp_path):
    base = tmp_path / "base.json"
    base.write_text("{}", encoding="utf-8")
    packet = {"remaining_candidates": ["reward_frame"]}
    envelope = {"schema": M.A.E.INPUT.PACKET_SCHEMA, "packet": packet}
    gate_case = {"case_id": "case-01", "label": "baseline_invalid"}
    with patch.object(M, "ROOT", tmp_path), patch.object(M.A, "ROOT", tmp_path), \
            patch.object(M, "verify_protocol", return_value={
                "component_admission_protocol": {"path": "admission.json"}
            }), patch.object(M.A, "verify_protocol", return_value={
                "component_probe_protocol": {"path": "base.json"}
            }), patch.object(M.A.E, "_gate", return_value=({}, gate_case)), \
            patch.object(M.A.E.INPUT, "make_selection_packet", return_value=envelope), \
            patch.object(M.A.E, "_source_binding", return_value={}):
        result = M.run_epoch(
            tmp_path / "out", tmp_path / "case.json", tmp_path / "reference.json",
            "registry_next", failure_protocol=tmp_path / "failure.json", anchor="a" * 40,
            run_id="run-01", lab=tmp_path, probe_protocol=base,
            gate_path=tmp_path / "gate.json", gate_sha256="b" * 64, case_id="case-01",
        )
    assert result["evidence_scope"] == "actual_backend"
    assert result["hold_reason"] == "source_gate_baseline_invalid"
    assert M.A.E.P is M.A.P and M.A.execute_probe is M._ORIGINAL_EXECUTE_PROBE


def test_logging_scope_is_reentrant_and_rejects_external_replacement():
    with M.logging_scope():
        assert M.A.execute_probe is M.execute_probe
        with M.logging_scope():
            assert M.A.execute_probe is M.execute_probe
    assert M.A.execute_probe is M._ORIGINAL_EXECUTE_PROBE
    replacement = lambda *_args, **_kwargs: None
    with patch.object(M.A, "execute_probe", replacement), \
            pytest.raises(ValueError, match="등록 원본"), M.logging_scope():
        pass


def test_run_epoch_rejects_caller_probe_executor(tmp_path):
    with pytest.raises(ValueError, match="교체할 수 없다"):
        M.run_epoch(failure_protocol=tmp_path / "failure.json", anchor="a" * 40,
                    probe_executor=lambda *_args: {})


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


def _registered_protocol(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    sources = tuple(dict.fromkeys((*M.A.SOURCE_FILES, *M.SOURCE_FILES)))
    for relative in sources:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative}\n", encoding="utf-8")
    base = repo / "bench/protocols/base.json"
    _write(base, {"schema": M.A.P.PROTOCOL_SCHEMA})
    _git(repo, "add", "--", *sources, "bench/protocols/base.json")
    _git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "code")
    admission = repo / "bench/protocols/admission.json"
    with patch.object(M, "ROOT", repo), patch.object(M.A, "ROOT", repo):
        _write(admission, M.A.prepare_protocol(base))
    _git(repo, "add", "--", "bench/protocols/admission.json")
    _git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "admission")
    admission_anchor = _git(repo, "rev-parse", "HEAD")
    failure = repo / "bench/protocols/failure.json"
    with patch.object(M, "ROOT", repo), patch.object(M.A, "ROOT", repo):
        _write(failure, M.prepare_protocol(admission))
    _git(repo, "add", "--", "bench/protocols/failure.json")
    _git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "failure")
    return repo, failure, admission_anchor, _git(repo, "rev-parse", "HEAD")


def test_protocol_binds_admission_component_new_sources_and_anchor(tmp_path):
    repo, failure, _prior, anchor = _registered_protocol(tmp_path)
    with patch.object(M, "ROOT", repo), patch.object(M.A, "ROOT", repo):
        value = M.verify_protocol(failure, anchor=anchor)
    assert set(value["source_sha256_lf"]) == set(M.SOURCE_FILES)
    assert value["component_admission_protocol"]["path"] == (
        "bench/protocols/admission.json"
    )
    assert value["failure_contract"]["retry"] is False


def test_protocol_rejects_source_and_anchor_without_registered_v3(tmp_path):
    repo, failure, prior, anchor = _registered_protocol(tmp_path)
    target = repo / M.SOURCE_FILES[0]
    original = target.read_text(encoding="utf-8")
    target.write_text("changed\n", encoding="utf-8")
    with patch.object(M, "ROOT", repo), patch.object(M.A, "ROOT", repo), \
            pytest.raises(ValueError, match="내용 또는 source|현재 source"):
        M.verify_protocol(failure, anchor=anchor)
    target.write_text(original, encoding="utf-8")
    with patch.object(M, "ROOT", repo), patch.object(M.A, "ROOT", repo), \
            pytest.raises(ValueError, match="Git commit/blob|anchor source"):
        M.verify_protocol(failure, anchor=prior)
