# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals")]

import run_five_policy_recovery_resume_v4 as V


def save(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def old_phase(tmp_path: Path) -> tuple[Path, Path]:
    phase = tmp_path / "old"
    for name in V.OLD_FILES:
        path = phase / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    save(phase / "actual-v1/parent-error.json", {
        "schema": "five_policy_recovery_parent_error_v1", "error_type": "ValueError",
        "error": "다른 arm 또는 출력 checkpoint의 부모"})
    save(phase / "exit.json", {"exit_code": 1})
    save(phase / "dispatch.json", {"outer_anchor": "a" * 40})
    save(phase / "actual-v1-git-proof-cache.json", {"execution_anchor": "a" * 40})
    save(phase / "transport-v1/manifest.json", {
        "schema": V.C.RUN.TRANSPORT.MANIFEST_SCHEMA, "api_cap": 32})
    lease = save(tmp_path / "closed.json", {
        "state": "failed", "exitCode": 1, "owner": V.C.RUN.LEASE.OWNER,
        "leaseId": "old-lease", "acquiredUtc": "2026-11-02T03:22:17.5939459Z",
        "completedUtc": "2026-11-02T03:28:35.3845525Z"})
    return phase, lease


def test_prior_failure_charges_time_and_rejects_any_request_or_sdk_file(tmp_path: Path) -> None:
    phase, lease = old_phase(tmp_path)
    value = V._old_state(phase, lease, "a" * 40)
    assert value["old_api_calls"] == value["old_sdk_executions"] == 0
    assert 377 < value["old_wall_upper_bound_s"] < 378
    assert value["remaining_wall_s"] + value["old_wall_upper_bound_s"] == 43200
    save(phase / "transport-v1/requests/new-request.json", {"status": "intent"})
    with pytest.raises(ValueError, match="원문 집합"):
        V._old_state(phase, lease, "a" * 40)


def test_wrong_failure_or_open_lease_cannot_authorize_retry(tmp_path: Path) -> None:
    phase, lease = old_phase(tmp_path)
    save(phase / "actual-v1/parent-error.json", {"error": "different error"})
    with pytest.raises(ValueError, match="부모 결합 오류"):
        V._old_state(phase, lease, "a" * 40)
    phase, lease = old_phase(tmp_path)
    save(lease, {"state": "running", "exitCode": None, "owner": V.C.RUN.LEASE.OWNER})
    with pytest.raises(ValueError, match="종료된 원 lease"):
        V._old_state(phase, lease, "a" * 40)


def test_remaining_wall_is_bounded_and_restored_on_failure() -> None:
    original = V.C.RUN.SECOND_WALL_S
    with (pytest.raises(RuntimeError, match="body"), V.remaining_wall_scope(42000)):
        assert V.C.RUN.SECOND_WALL_S == 42000
        raise RuntimeError("body")
    assert V.C.RUN.SECOND_WALL_S == original == 43200
    with (pytest.raises(ValueError, match="남은 wall"), V.remaining_wall_scope(43201)):
        pass


def test_external_wall_change_is_detected_after_restore() -> None:
    with (pytest.raises(ValueError, match="wall이 바뀌었다"),
          V.remaining_wall_scope(42000)):
        V.C.RUN.SECOND_WALL_S = 43000
    assert V.C.RUN.SECOND_WALL_S == 43200


def test_registered_source_requires_exact_packet_digest(tmp_path: Path) -> None:
    sources = []
    digests = {}
    for case, policy in V.C.RUN.TARGETS:
        source = {"case_id": case, "policy": policy, "packet": {"value": case}}
        p = save(tmp_path / f"{case}_{policy}.json", source)
        sources.append({"case_id": case, "policy": policy,
                        "status": "registered", "source": V.C._ref(p)})
        digests[f"{case}:{policy}"] = V.C.SECOND.digest(source)
    loop = save(tmp_path / "loop.json", {"source_digests": digests})
    plan = save(tmp_path / "plan.json", {"schema": V.C.RUN.SECOND_PLAN_SCHEMA,
                                        "loop_protocol": V.C._ref(loop), "records": sources})
    assert len(V.registered_sources(plan)) == 4
    digests[next(iter(digests))] = "0" * 64
    save(loop, {"source_digests": digests})
    p = json.loads(plan.read_text(encoding="utf-8"))
    p["loop_protocol"] = V.C._ref(loop)
    save(plan, p)
    with pytest.raises(ValueError, match="source digest"):
        V.registered_sources(plan)
