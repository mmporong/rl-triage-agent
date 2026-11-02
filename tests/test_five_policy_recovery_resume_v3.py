# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals")]

import run_five_policy_recovery_resume_v3 as V


def test_registered_overlay_checks_real_git_and_mutable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(V, "ROOT", tmp_path)
    monkeypatch.setattr(V.CONT, "ROOT", tmp_path)
    monkeypatch.setattr(V, "SOURCE_FILES", ("bench/frozen.py",))
    (tmp_path / "bench").mkdir()
    frozen = tmp_path / "bench/frozen.py"
    frozen.write_bytes(b"# frozen\n")
    plan = tmp_path / "private-plan.json"
    plan.write_text(json.dumps({"schema": V.CONT.RUN.SECOND_PLAN_SCHEMA}), encoding="utf-8")
    compatibility = tmp_path / "bench/compatibility.json"
    compatibility.write_text(json.dumps({
        "schema": V.CONT.SECOND_COMPATIBILITY_PROTOCOL_SCHEMA,
        "second_plan_sha256": V.CONT._sha(plan),
    }), encoding="utf-8")
    out = tmp_path / "bench/cache.json"
    V.build_protocol(second_plan=plan, compatibility_protocol=compatibility, out=out)
    for argv in (["init", "-q"], ["config", "user.name", "Fixture"],
                 ["config", "user.email", "fixture@example.invalid"],
                 ["add", "bench/frozen.py", "bench/compatibility.json", "bench/cache.json"],
                 ["commit", "-qm", "fixture"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    anchor = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                            capture_output=True, text=True, check=True).stdout.strip()
    assert V.verify_protocol(out, anchor=anchor, second_plan=plan,
                             compatibility_protocol=compatibility)["policy"] == V.POLICY
    frozen.write_bytes(b"# changed\n")
    with pytest.raises(ValueError, match="current source"):
        V.verify_protocol(out, anchor=anchor, second_plan=plan,
                          compatibility_protocol=compatibility)
    frozen.write_bytes(b"# frozen\n")
    plan.write_text('{"schema":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="policy/plan/compatibility"):
        V.verify_protocol(out, anchor=anchor, second_plan=plan,
                          compatibility_protocol=compatibility)


@pytest.mark.parametrize("fail", [False, True])
def test_frozen_runner_is_scoped_and_evidence_written_after_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool,
) -> None:
    events = []
    protocol = tmp_path / "registered.json"
    protocol.write_bytes(b"{}\n")
    out = tmp_path / "actual"
    monkeypatch.setattr(V, "verify_protocol", lambda *a, **kw: events.append("verify"))

    @contextmanager
    def scope(**kwargs):
        events.append("enter")
        try:
            yield {"hits": 3, "cached_material": False}
        finally:
            events.append("restored")

    def run(plan_path, **kwargs):
        assert events == ["verify", "enter"]
        assert kwargs["out"] == out
        events.append("frozen")
        if fail:
            raise RuntimeError("original failure")
        return {"schema": "unchanged", "status": "complete"}

    monkeypatch.setattr(V.CACHE, "proof_scope", scope)
    monkeypatch.setattr(V.CONT, "run_second_operational", run)
    kwargs = {"host_protocol": protocol, "host_anchor": "a" * 40,
              "compatibility_protocol": protocol, "compatibility_anchor": "a" * 40,
              "cache_protocol": protocol, "cache_anchor": "a" * 40,
              "transport_root": tmp_path / "transport", "lab": tmp_path / "lab",
              "isaac_python": tmp_path / "python", "out": out,
              "approver": V.CONT.RUN.APPROVER}
    if fail:
        with pytest.raises(RuntimeError, match="original failure"):
            V.run_second_operational(tmp_path / "plan", **kwargs)
    else:
        assert V.run_second_operational(tmp_path / "plan", **kwargs) == {
            "schema": "unchanged", "status": "complete"}
    assert events == ["verify", "enter", "frozen", "restored"]
    evidence = json.loads((tmp_path / "actual-git-proof-cache.json").read_text(encoding="utf-8"))
    assert evidence["stats"] == {"hits": 3, "cached_material": False}
    assert evidence["protocol"]["sha256"] == V.CONT._sha(protocol)
    with pytest.raises(ValueError, match="새 Git proof evidence"):
        V.run_second_operational(tmp_path / "plan", **kwargs)


def test_different_anchor_is_rejected_before_original_runner(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outer anchors"):
        V.run_second_operational(
            tmp_path, host_protocol=tmp_path, host_anchor="a" * 40,
            compatibility_protocol=tmp_path, compatibility_anchor="a" * 40,
            cache_protocol=tmp_path, cache_anchor="b" * 40,
            transport_root=tmp_path, lab=tmp_path, isaac_python=tmp_path,
            out=tmp_path, approver=V.CONT.RUN.APPROVER)


def test_cli_preserves_original_error_and_parent_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("original parent failure")

    monkeypatch.setattr(V, "run_second_operational", fail)
    out = tmp_path / "actual"
    argv = ["run-second"]
    for name in ("plan", "host-protocol", "compatibility-protocol", "cache-protocol",
                 "transport-root", "lab", "isaac-python"):
        argv.extend(["--" + name, str(tmp_path / name)])
    for name in ("host-anchor", "compatibility-anchor", "cache-anchor"):
        argv.extend(["--" + name, "a" * 40])
    argv.extend(["--out", str(out), "--approver", V.CONT.RUN.APPROVER])
    with pytest.raises(RuntimeError, match="original parent failure"):
        V.main(argv)
    error = json.loads((out / "parent-error.json").read_text(encoding="utf-8"))
    assert error["error"] == "original parent failure"
    assert error["automatic_retry"] is False
