from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import core_seed_guard_v3 as G  # pyright: ignore[reportMissingImports]


def _write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bytes(path: Path, value: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    for path in (repo / "bench/private/cpu-proof", repo / "bench/protocols", repo / "evals",
                 lab / "logs"):
        path.mkdir(parents=True)
    source_receipts = {}
    for relative in (
        "evals/registered_epoch_recovery_execution_v1.py",
        "evals/r1_repair_execution_v2.py",
        "evals/r1_repair_execution.py",
        "bench/run_runtime_readback_smoke.py",
        "bench/registered_epoch_recovery_training_v1.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
        raw = path.read_bytes()
        source_receipts[relative] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
        }
    folder = repo / "bench/private/cpu-proof/actual-v16-preflight-run01"
    helper = folder.parent / "actual_request_preflight_v13.py"
    helper.write_text("cpu helper\n", encoding="utf-8")
    binding = _write(repo / "bench/private/binding.json", {"seed": 1})
    generated = _write(folder / ".training-environment-binding.json", {"mode": "train"})
    manifest = _write(folder / "training/manifest.json", {"seed": 6666})
    launch_path = folder / "training/owned/launch.json"
    launch = _write(launch_path, {"args": ["python", "worker"]})
    execution = _write(folder / "training/execution.json", {
        "job_assigned_before_release": False, "natural_exit": False, "timed_out": False,
        "error_type": "SdkBoundaryReached", "exit_code": None, "cleanup_verified": False,
        "owned_process_identities": {}, "wall_s": 0.01,
    })
    outer = _bytes(folder / "outer-traceback.txt", "expected boundary\n")
    stack = _bytes(folder / "sdk-boundary-stack.txt", "guarded_popen\nsubprocess.Popen\n")
    proof = {
        "schema": G.PROOF_SCHEMA, "status": "pass", "binding": binding,
        "output": str(folder), "original_request_id": "request", "effective_run_name": "run_s6666",
        "training_seed": 6666,
        "isaaclab_preflight": {"processes_started": 0, "gpu_queries": 0},
        "git_head_before": "a" * 40, "git_head_after": "a" * 40,
        "source_sha256_before": source_receipts, "source_sha256_after": source_receipts,
        "outer_error": {"type": "RuntimeError",
                        "message": "학습 실행 또는 첫 3 update 계측이 완결되지 않았다"},
        "boundary": {"count": 1, "command": ["python", "worker", "--_worker", str(launch_path)]},
        "checks": {name: True for name in G.EXPECTED_CHECKS},
        "refs": {"binding": binding, "generated_environment_binding": generated,
                 "manifest": manifest, "launch": launch, "execution": execution,
                 "outer_traceback": outer, "sdk_boundary_stack": stack},
        "resource_counts": {"sdk_launches": 0, "gpu_queries": 0, "api_calls": 0,
                            "fixture_gpu_snapshots": 1},
    }
    proof_path = repo / "bench/private/cpu-proof/result.json"
    proof_ref = _write(proof_path, proof)
    return repo, lab, helper, proof_ref


def _enable_fixture_helper(monkeypatch: pytest.MonkeyPatch, helper: Path) -> None:
    monkeypatch.setattr(G, "KNOWN_HELPER_SHA256", hashlib.sha256(helper.read_bytes()).hexdigest())


def test_verified_cpu_boundary_proof_is_the_only_skipped_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lab, helper, proof_ref = _fixture(tmp_path)
    _enable_fixture_helper(monkeypatch, helper)
    result = G.assert_fresh_seeds(
        [6666], lab, repo=repo, nonexecution_proofs={"epoch_cpu_boundary": proof_ref},
    )
    assert result["status"] == "fresh"
    assert result["schema"] == G.SCHEMA
    assert result["isolated_skipped_record_count"] == 1
    assert result["verified_nonexecution_proofs"][0]["reason"] == \
        "verified_cpu_sdk_boundary_nonexecution"


def test_sibling_record_with_same_seed_is_still_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lab, helper, proof_ref = _fixture(tmp_path)
    _enable_fixture_helper(monkeypatch, helper)
    _write(repo / "bench/private/cpu-proof/sibling/result.json", {"training_seed": 6666})
    with pytest.raises(G.V1.ORIGINAL.SeedReuseError, match="6666"):
        G.assert_fresh_seeds(
            [6666], lab, repo=repo, nonexecution_proofs={"epoch_cpu_boundary": proof_ref},
        )


def test_zero_metadata_without_full_receipts_is_not_exempted(tmp_path: Path) -> None:
    repo, lab, _helper, _proof_ref = _fixture(tmp_path)
    fake_ref = _write(repo / "bench/private/fake/result.json", {"schema": G.PROOF_SCHEMA, "SDK": 0})
    with pytest.raises(ValueError, match="schema 또는 필드"):
        G.assert_fresh_seeds([6666], lab, repo=repo, nonexecution_proofs={"fake": fake_ref})


def test_positive_execution_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lab, helper, proof_ref = _fixture(tmp_path)
    _enable_fixture_helper(monkeypatch, helper)
    proof = json.loads(Path(proof_ref["path"]).read_text(encoding="utf-8"))
    proof["checkpoint"] = {"sha256": "0" * 64}
    changed = _write(repo / "bench/private/cpu-proof/positive/result.json", proof)
    with pytest.raises(ValueError, match="schema 또는 필드|양성 증거"):
        G.assert_fresh_seeds([6666], lab, repo=repo, nonexecution_proofs={"positive": changed})


def test_forbidden_proof_path_is_rejected_before_filesystem_lookup(tmp_path: Path) -> None:
    repo, lab, _helper, _proof_ref = _fixture(tmp_path)
    forbidden = repo / G.V1.EXCLUDED_RELATIVE / "missing.json"
    with pytest.raises(ValueError, match="제외 provider"):
        G.assert_fresh_seeds(
            [6666], lab, repo=repo,
            nonexecution_proofs={"forbidden": {"path": str(forbidden), "sha256": "0" * 64}},
        )


def test_referenced_receipt_sha_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lab, helper, proof_ref = _fixture(tmp_path)
    _enable_fixture_helper(monkeypatch, helper)
    proof = json.loads(Path(proof_ref["path"]).read_text(encoding="utf-8"))
    stack = Path(proof["refs"]["sdk_boundary_stack"]["path"])
    stack.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA가 원문과 다르다"):
        G.assert_fresh_seeds(
            [6666], lab, repo=repo, nonexecution_proofs={"changed": proof_ref},
        )


def test_failed_validation_restores_original_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lab, helper, proof_ref = _fixture(tmp_path)
    _enable_fixture_helper(monkeypatch, helper)
    old_assert = G.V1.ORIGINAL.assert_fresh_seeds
    old_files = G.V1.ORIGINAL._files
    _write(repo / "bench/private/used/result.json", {"training_seed": 6666})
    with pytest.raises(G.V1.ORIGINAL.SeedReuseError):
        G.assert_fresh_seeds(
            [6666], lab, repo=repo, nonexecution_proofs={"epoch_cpu_boundary": proof_ref},
        )
    assert G.V1.ORIGINAL.assert_fresh_seeds is old_assert
    assert G.V1.ORIGINAL._files is old_files
