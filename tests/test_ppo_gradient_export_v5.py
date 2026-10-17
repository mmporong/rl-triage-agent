"""공개 원본으로 판정을 재생하고 묶음·경로·변조 경계를 확인한다."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))
import export_ppo_gradient_v5 as X


def public_payloads():
    return {p.relative_to(X.OUTPUT).as_posix(): p.read_bytes() for p in X.OUTPUT.rglob("*") if p.is_file()}


def test_public_bundle_recomputes_checks_without_private_files(monkeypatch):
    monkeypatch.setattr(X, "RUN", Path("nonexistent-source"))
    value = X.verify_public()
    assert value["files"] == 59 and value["checks"] == 7
    assert value["recomputed_arms"] == 8 and value["dry_run_verified"]
    result = X.Q.read(X.OUTPUT / "result.json")
    assert result["outcome"] == "confirmed" and all(result["checks"].values())
    bad = next(r for r in result["runs"] if r["arm"] == "INVERSE_RATIO")
    assert bad["inspection"]["primitives"]["optimizer_step_count"]["status"] == "normal"
    gradient = bad["inspection"]["primitives"]["ppo_surrogate_gradient"]
    assert gradient == {"status": "abnormal", "first_mismatch_update": 0, "batches": 60}
    assert result["behavior_recovery_measured"] is False


def test_v4_remains_cost_exceeded_and_cpu_counts_are_not_gpu_times():
    comparison = X.Q.read(X.OUTPUT / "v4_comparison.json")
    assert comparison["v4"]["outcome"] == "observer_cost_exceeded"
    assert all(p["update_wall_ratio"] > 1.5 for p in comparison["v4"]["pairs"])
    assert all(p["update_wall_ratio"] <= 1.5 for p in comparison["v5"]["pairs"])
    assert not comparison["same_seed_comparison"]
    profile = X.Q.read(X.OUTPUT / "cpu_profile.json")
    assert profile["device"] == "cpu" and profile["gpu_executions"] == 0
    assert [r["counts"]["item"] for r in profile["runs"]] == [726, 2211, 726]


def test_tampered_record_and_self_updated_manifest_are_rejected():
    payloads = public_payloads()
    path = "runs/inverse_ratio_s2144_on/records.jsonl"
    rows = [json.loads(line) for line in payloads[path].splitlines()]
    rows[0]["observed"]["optimizer_steps"] = 999
    payloads[path] = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    provenance = json.loads(payloads["provenance.json"])
    provenance["payload_sha256_lf"][path] = X.sha_bytes(payloads[path])
    payloads["provenance.json"] = X.encode(provenance)
    with pytest.raises(ValueError, match="canonical"):
        X.verify_payloads(payloads)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_inventory_is_required(change):
    payloads = public_payloads()
    if change == "missing":
        del payloads["dry_run_result.json"]
    else:
        payloads["extra.json"] = b"{}"
    with pytest.raises(ValueError, match="canonical"):
        X.verify_payloads(payloads)


def test_execution_projection_omits_host_metadata():
    row = {"job_assigned_before_release": True, "natural_exit": True, "timed_out": False,
        "error_type": None, "exit_code": 0, "cleanup_verified": True, "wall_s": 1,
        "owned_process_identities": {"123": 456}, "cwd": "sensitive-path"}
    value = X.execution_projection(row, "a" * 64)
    assert "owned_process_identities" not in value and "cwd" not in value
    assert value["source_execution_sha256_lf"] == "a" * 64


@pytest.mark.parametrize("text", ["C:" + "\\" + "Users\\person", "/" + "home/person/x", "bench/" + "private/data"])
def test_private_paths_are_rejected(text):
    with pytest.raises(ValueError, match="경로"):
        X.public_safe(X.encode({"nested": [text]}))


def test_export_does_not_overwrite_existing_folder_or_touch_source(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "OUTPUT", tmp_path)
    monkeypatch.setattr(X, "verify_and_build", lambda *args: pytest.fail("기존 결과는 build 전에 거부"))
    with pytest.raises(FileExistsError):
        X.export(tmp_path)


def test_source_inventory_requires_pinned_result_and_all_artifacts(tmp_path):
    result = {"artifact_sha256_lf": {"evidence.json": X.sha_bytes(b"{}")}}
    (tmp_path / "evidence.json").write_bytes(b"{}")
    (tmp_path / "result.json").write_bytes(X.encode(result))
    pinned = X.Q.sha(tmp_path / "result.json")
    X.validate_source_inventory(tmp_path, pinned)
    (tmp_path / "evidence.json").write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError, match="artifact"):
        X.validate_source_inventory(tmp_path, pinned)
