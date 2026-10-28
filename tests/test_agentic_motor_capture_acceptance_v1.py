"""motor capture 회수 수용 adapter를 실제 형태 trace·sidecar fixture로 CPU에서 검증한다(실제 회수 자료가 아니다)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src", ROOT / "tests"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_motor_capture_acceptance_v1 as A  # pyright: ignore[reportMissingImports]
import test_agentic_state_facts_v1 as FT  # pyright: ignore[reportMissingImports]
from test_agentic_play_job_v2 import make_dir_link  # pyright: ignore[reportMissingImports]

R, SIDECAR, JOB1, RUN1, FACTS = A.R, A.SIDECAR, A.JOB1, A.RUN1, A.FACTS
REMOTE = PurePosixPath("/remote/agentic-capture")
V5_PHASE = "bench/protocols/agentic_motor_capture_phase_20261028_v5.json"  # fixture 원본으로 쓰는 공개 등록
Mutation = Callable[["Run"], object]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _dump(path, value)


@dataclass
class Run:
    """CPU fixture. 공개 등록 ID·CP SHA로 만든 형태이며 실제 회수 자료가 아니다."""

    tmp: Path
    repo: Path
    portable: Path
    phase_sha: str
    run: Path
    evidence: Path
    backup: Path
    policies: dict[str, dict[str, Any]]

    def stage(self, identity: str) -> Path:
        return self.evidence / A.stage_name(identity)

    def evaluate(self) -> dict[str, Any]:
        return A.evaluate(self.run, self.portable, phase=V5_PHASE, phase_sha256_lf=self.phase_sha, repo=self.repo)


def _registered_repo(tmp: Path, *, protocol_change: Callable[[dict[str, Any]], None] | None = None,
                     portable_change: Callable[[dict[str, Any]], None] | None = None) -> tuple[Path, Path, str]:
    """실제 공개 phase·protocol을 fixture repo로 복사하고 fixture portable manifest SHA를 phase에 등록한다."""
    repo = tmp / "repo"
    phase = json.loads((ROOT / V5_PHASE).read_text(encoding="utf-8"))
    files = []
    for batch in phase["vms"][0]["batches"]:
        source = ROOT / batch["protocol"]
        target = repo / batch["protocol"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        if protocol_change is not None:
            _rewrite(target, protocol_change)
            batch["protocol_sha256_lf"] = R.sha256_file(target, lf=True)
        protocol = json.loads(target.read_text(encoding="utf-8"))
        files += [{"archive_path": row["isaaclab_relative"], "bytes": 983787, "sha256": row["sha256"]}
                  for row in protocol["checkpoints"]]
    portable_value = {"schema": A.PORTABLE_SCHEMA, "archive_sha256": "0" * 64, "files": files}
    if portable_change is not None:
        portable_change(portable_value)
    portable = _dump(tmp / "inputs" / "portable-manifest.json", portable_value)
    phase["input_manifest_sha256"] = _sha(portable)
    phase_path = _dump(repo / V5_PHASE, phase)
    return repo, portable, R.sha256_file(phase_path, lf=True)


def _write_stage(evidence: Path, policy: dict[str, Any], scratch: Path, *, frames: int = R.FRAME_COUNT,
                 video: str | None = None) -> None:
    identity, job_id, checkpoint = policy["logical_id"], policy["job_id"], policy["checkpoint_sha256"]
    name = A.stage_name(identity)
    stage = evidence / name
    scratch.mkdir(parents=True)
    sidecar_source, trace_source, _native = FT.build_inputs(scratch, sidecar_checkpoint=checkpoint, source_id=job_id,
                                                            frames=frames)
    sidecar = JOB1.sidecar_directory_for(stage)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sidecar_source), str(sidecar))
    trace = json.loads(trace_source.read_text(encoding="utf-8"))
    video = video or hashlib.sha256(identity.encode()).hexdigest()
    trace.update(job_id=job_id, checkpoint_sha256=checkpoint, video_sha256=video)
    trace["timing_receipt"]["source_sha256"] = video
    trace_path = _dump(stage / "output" / "evidence" / "raw_trace.json", trace)
    native_path = _dump(stage / "output" / "native_capture_manifest.json",
                        {"checkpoint_sha256": checkpoint, "frame_count": R.FRAME_COUNT})
    remote = REMOTE / name
    job_path = _dump(stage / "job.json", {
        "schema": JOB1.JOB_SCHEMA, "mode": JOB1.MODE, "job_id": job_id,
        "protocol": {"path": f"/remote/repo/{policy['protocol']}", "sha256_lf": policy["protocol_sha256_lf"]},
        "checkpoint": {"path": "/remote/lab/model_299.pt", "sha256": checkpoint, "logical_id": identity},
        "output_directory": str(remote / "output"),
        "sidecar": {"directory": str(REMOTE / (name + JOB1.SIDECAR_SUFFIX)), "source_id": job_id,
                    "error_policy": "abort_recording", "expected_frames": R.FRAME_COUNT},
    })
    try:
        join: Any = SIDECAR.join_with_trace(sidecar, trace_path, native_path)
    except SIDECAR.SidecarError:  # 타 frame fixture: 생산자가 결합 결과를 잘못 봉인한 경우를 흉내 낸다
        join = {"joined": True, "frame_count": R.FRAME_COUNT, "checkpoint_sha256": checkpoint,
                "trace_sha256": _sha(trace_path), "native_manifest_sha256": _sha(native_path),
                "sidecar_manifest_sha256": _sha(sidecar / SIDECAR.MANIFEST_NAME)}
    _dump(stage / RUN1.FINAL_NAME, {
        "schema": RUN1.FINAL_SCHEMA, "job_sha256": _sha(job_path), "checkpoint_logical_id": identity,
        "checkpoint_sha256": checkpoint, "fault_id": policy["fault_id"], "pair": policy["pair"],
        "native_technical_complete": True, "sidecar": {"status": "sealed"}, "join": join,
        "playback_completed": True, "actuator_clip_consistent_with_registration": True,
        "lease": {"active_at_finalize": True, "live_lock_proof": True},
        "final_technical_complete": True, "live_evidence": True, "acceptable_as_live_evidence": True,
        "behavior_label": None, "used_for_decision": False,
    })
    _dump(stage / "unit-result.json", {"schema": A.UNIT_RESULT_SCHEMA, "mode": "record", "completed": True})
    _dump(evidence / (name + "-admission") / "resource.json", {"fixture": True})
    _dump(evidence / (name + "-lease-root.json"), {"fixture": True})


def _seal(run: Path, backup: Path) -> None:
    """host v5 collect·collect_verified와 같은 봉인 형태(inventory·D 사본·collection.json)."""
    evidence = run / "retrieved" / "evidence"
    rows = [{"path": path.relative_to(evidence).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(evidence.rglob("*")) if path.is_file()]
    _dump(evidence / A.INVENTORY_NAME, {"files": rows})
    for path in evidence.rglob("*"):
        if path.is_file():
            target = backup / path.relative_to(evidence)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    archive = run / A.ARCHIVE_NAME
    archive.write_bytes(b"fixture archive")
    _dump(run / A.COLLECTION_NAME, {
        "path": "/remote/agentic-capture.tar.gz", "sha256": _sha(archive), "files": len(rows), "local": str(archive),
        "archive_sha_verified": True,
        "copy_verification": {"files_verified": len(rows) + 1, "local": str(evidence), "backup": str(backup),
                              "all_match": True},
    })


def build_run(tmp: Path, *, before_seal: Mutation | None = None, frames: dict[str, int] | None = None,
              videos: dict[str, str] | None = None) -> Run:
    repo, portable, phase_sha = _registered_repo(tmp)
    policies = A.load_registration(portable, phase=V5_PHASE, phase_sha256_lf=phase_sha, repo=repo)["policies"]
    run = tmp / "run-j1kxncj4e"
    evidence = run / "retrieved" / "evidence"
    counts: dict[str, int] = frames or {}
    for identity, policy in sorted(policies.items()):
        _write_stage(evidence, policy, tmp / "scratch" / identity, frames=counts.get(identity, 625),
                     video=(videos or {}).get(identity))
    _dump(evidence / "state-smoke-motor-s2033-none" / "unit-result.json", {"fixture": True})
    result = Run(tmp, repo, portable, phase_sha, run, evidence, tmp / "backup", policies)
    if before_seal is not None:
        before_seal(result)
    _seal(run, result.backup)
    return result


NONE_2033, F2_2033 = "motor-s2033-none", "motor-s2033-f2"


# --- 성공 ---------------------------------------------------------------------------------------


def test_four_exact_registrations_bind_into_public_facts_and_write_once(tmp_path: Path) -> None:
    world = build_run(tmp_path)
    registered = [(ROOT / path).read_bytes() for path in (V5_PHASE, *[row["protocol"] for row in
                                                                     world.policies.values()])]
    result = world.evaluate()
    summary, public, metadata = result["summary"], result["public"], result["metadata"]
    assert summary["status"] == "technical_complete" and summary["errors"] == []
    assert [(row["logical_id"], row["fault_id"], row["technical_accepted"]) for row in summary["policies"]] == [
        ("motor-s2033-f2", "F2", True), ("motor-s2033-none", "NONE", True),
        ("motor-s2046-f2", "F2", True), ("motor-s2046-none", "NONE", True)]
    assert public is not None and public["policy_count"] == 4
    assert all(item["schema"] == FACTS.SCHEMA and item["thresholds_created"] == 0 for item in public["policies"])
    assert "w1.command_tracking_error_norm" in public["policies"][0]["facts"]
    text = json.dumps(public, ensure_ascii=False)
    for banned in (*world.policies, *(row["job_id"] for row in world.policies.values()), str(tmp_path), "F2", "NONE"):
        assert banned not in text
    assert summary["fixed_flags"] == public["fixed_flags"] == dict.fromkeys(A.FIXED_FLAGS, False)
    assert summary["final_acceptance_owner"] == "main" and summary["behavior_label"] is None
    rows = {row["logical_id"]: row for row in metadata["policies"]}
    assert rows[F2_2033]["pair"] == rows[NONE_2033]["pair"] and rows[F2_2033]["role"] == "case"
    assert {row["binding_digest"] for row in rows.values()} == {
        item["binding"]["binding_digest"] for item in public["policies"]}
    assert metadata["collection"]["c_d_rehashed"] is True and metadata["public_sha256"] == R.digest(public)
    paths = A.write_outputs(result, tmp_path / "out")
    assert set(paths) == {"summary", "metadata", "public"}
    assert json.loads(paths["public"].read_text(encoding="utf-8")) == public
    with pytest.raises(FileExistsError):
        A.write_outputs(result, tmp_path / "out")
    assert registered == [(ROOT / path).read_bytes() for path in (V5_PHASE, *[row["protocol"] for row in
                                                                            world.policies.values()])]


# --- HOLD·등록 ----------------------------------------------------------------------------------


def test_hold_before_collection_and_failure_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    world = build_run(tmp_path)
    (world.run / A.COLLECTION_NAME).unlink()
    with pytest.raises(A.Hold):
        world.evaluate()
    assert A.main(["--run", str(world.run), "--portable-manifest", str(world.portable), "--phase", V5_PHASE,
                   "--phase-sha256-lf", world.phase_sha, "--repo", str(world.repo),
                   "--out", str(tmp_path / "out")]) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "HOLD" and not (tmp_path / "out").exists()
    _dump(world.run / A.COLLECTION_FAILURE_NAME, {"type": "ValueError"})
    with pytest.raises(A.AcceptanceError, match="회수 실패"):
        world.evaluate()


@pytest.mark.parametrize("case", ["portable-sha", "protocol-sha", "approval-true", "portable-cp"])
def test_registration_binding_is_exact(tmp_path: Path, case: str) -> None:
    def approve(value: dict[str, Any]) -> None:
        value["preregistration"]["fixed_flags"]["capture_approval"] = True

    def other_cp(value: dict[str, Any]) -> None:
        value["files"][0]["sha256"] = "0" * 64

    repo, portable, phase_sha = _registered_repo(
        tmp_path, protocol_change=approve if case == "approval-true" else None,
        portable_change=other_cp if case == "portable-cp" else None)
    if case == "portable-sha":
        _rewrite(portable, lambda value: value.update(archive_sha256="1" * 64))
    elif case == "protocol-sha":
        target = repo / "bench/protocols/agentic_motor_capture_s2046_20261028_v5.json"
        _rewrite(target, lambda value: value["pairs"][0].update(seed=1))
    with pytest.raises(A.AcceptanceError):
        A.load_registration(portable, phase=V5_PHASE, phase_sha256_lf=phase_sha, repo=repo)


# --- 회수 게이트 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["archive-sha", "not-sealed", "unlisted-file", "duplicate-row", "escape-path",
                                  "c-root-moved", "other-run", "link"])
def test_collection_gate_rejects_before_policies(tmp_path: Path, case: str) -> None:
    world = build_run(tmp_path)
    inventory = world.evidence / A.INVENTORY_NAME
    collection = world.run / A.COLLECTION_NAME
    if case == "archive-sha":
        (world.run / A.ARCHIVE_NAME).write_bytes(b"other archive")
    elif case == "not-sealed":
        _rewrite(collection, lambda value: value["copy_verification"].update(all_match=False))
    elif case == "unlisted-file":
        _dump(world.stage(NONE_2033) / "late.json", {"generated_after_seal": True})
    elif case == "duplicate-row":
        _rewrite(inventory, lambda value: value["files"].append(dict(value["files"][0])))
    elif case == "escape-path":
        _rewrite(inventory, lambda value: value["files"][0].update(path="../outside.json"))
    elif case == "c-root-moved":
        _rewrite(collection, lambda value: value["copy_verification"].update(local=str(world.tmp / "elsewhere")))
    elif case == "other-run":
        world.run = world.run.rename(world.run.with_name("run-other"))
    else:
        sidecar = JOB1.sidecar_directory_for(world.stage(NONE_2033))
        moved = shutil.move(str(sidecar), str(world.tmp / "outside-sidecar"))
        make_dir_link(sidecar, Path(moved))
    with pytest.raises(A.AcceptanceError):
        world.evaluate()


# --- 정책 ---------------------------------------------------------------------------------------


def _final(identity: str, change: Callable[[dict[str, Any]], None]) -> Mutation:
    return lambda world: _rewrite(world.stage(identity) / RUN1.FINAL_NAME, change)


def _job_other_cp(world: Run) -> None:
    stage = world.stage(F2_2033)
    _rewrite(stage / "job.json", lambda value: value["checkpoint"].update(sha256="0" * 64))
    _rewrite(stage / RUN1.FINAL_NAME, lambda value: value.update(job_sha256=_sha(stage / "job.json")))


POLICY_CASES: dict[str, tuple[Mutation | None, str]] = {
    "incomplete": (_final(F2_2033, lambda v: v.update(acceptable_as_live_evidence=False)), "live evidence"),
    "missing": (lambda world: shutil.rmtree(world.stage(F2_2033)), "회수되지 않았다"),
    "other-cp": (_job_other_cp, "job checkpoint"),
    "other-pair": (_final(F2_2033, lambda v: v["pair"].update(pair_id="motor-s2046")), "pair"),
    "join-tamper": (_final(F2_2033, lambda v: v["join"].update(trace_sha256="0" * 64)), "등록 SHA"),
    "unit-failure": (lambda world: _dump(world.stage(F2_2033) / "unit-failure.json", {}), "unit 실패"),
    "other-frame": (None, "frame"),
}


@pytest.mark.parametrize("case", sorted(POLICY_CASES))
def test_one_bad_policy_is_rejected_and_public_facts_are_withheld(tmp_path: Path, case: str) -> None:
    mutate, reason = POLICY_CASES[case]
    world = build_run(tmp_path, before_seal=mutate, frames={F2_2033: R.FRAME_COUNT - 1} if case == "other-frame" else None)
    result = world.evaluate()
    rows = {row["logical_id"]: row for row in result["summary"]["policies"]}
    assert result["summary"]["status"] == "rejected" and result["public"] is None
    assert rows[F2_2033]["technical_accepted"] is False and reason in rows[F2_2033]["reason"]
    assert all(row["technical_accepted"] for identity, row in rows.items() if identity != F2_2033)
    paths = A.write_outputs(result, tmp_path / "out")
    assert set(paths) == {"summary", "metadata"} and result["summary"]["fixed_flags"]["capture_approval"] is False


@pytest.mark.parametrize("copy", ["C", "D"])
def test_c_or_d_copy_change_after_seal_rejects_that_policy(tmp_path: Path, copy: str) -> None:
    world = build_run(tmp_path)
    relative = Path(A.stage_name(NONE_2033)) / "output" / "native_capture_manifest.json"
    root = world.evidence if copy == "C" else world.backup
    (root / relative).write_bytes((root / relative).read_bytes() + b" ")
    rows = {row["logical_id"]: row for row in world.evaluate()["summary"]["policies"]}
    assert rows[NONE_2033]["technical_accepted"] is False and f"{copy} 사본" in rows[NONE_2033]["reason"]


@pytest.mark.parametrize("case", ["extra-stage", "duplicate-video"])
def test_extra_record_stage_or_duplicate_binding_rejects_bundle(tmp_path: Path, case: str) -> None:
    if case == "extra-stage":
        world = build_run(tmp_path, before_seal=lambda w: _dump(w.evidence / "record-motor-s2033-none-retry" / "job.json", {}))
    else:
        world = build_run(tmp_path, videos={F2_2033: hashlib.sha256(NONE_2033.encode()).hexdigest()})
    summary = world.evaluate()["summary"]
    assert summary["status"] == "rejected" and summary["errors"]
    assert all(row["technical_accepted"] for row in summary["policies"])


def test_host_path_maps_only_wsl_and_windows_drive_roots() -> None:
    if os.name == "nt":
        assert A.host_path("/mnt/d/rl-triage-artifacts/x") == Path("D:/rl-triage-artifacts/x")
    else:
        assert A.host_path("D:\\rl-triage-artifacts\\x") == Path("/mnt/d/rl-triage-artifacts/x")
    with pytest.raises(A.AcceptanceError):
        A.host_path("relative/path")
