from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import agentic_fault_playback_v1 as PLAYBACK  # pyright: ignore[reportMissingImports]
import agentic_play_job_v1 as J  # pyright: ignore[reportMissingImports]
import record_agentic_state_sidecar_v1 as SIDECAR  # pyright: ignore[reportMissingImports]

TARGET = {"resource_id": "agentic-l40s-id", "resource_name": "agentic-recording-vm",
          "instance_type": "l40s-48gb.1x"}
F2_ID, NONE_ID = "go2-s2033-f2", "go2-s2033-none"
OMIT = object()
_OPEN: list[ExitStack] = []


@pytest.fixture(autouse=True)
def _close_leases() -> Iterator[None]:
    """테스트가 연 lease를 끝에서 모두 닫는다(heartbeat thread 정지·lock 해제)."""
    yield
    while _OPEN:
        _OPEN.pop().close()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8", newline="\n")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


@dataclass
class World:
    """CPU fixture. 실제 checkpoint·GPU·원격 자원이 아니며 실제 evidence로 쓰지 않는다."""

    root: Path
    now: datetime
    repo: Path
    commit: str
    sources: dict[str, str]
    lab: Path
    runtime: dict[str, str]
    rows: list[dict[str, Any]]
    protocol_path: Path
    protocol: dict[str, Any]
    proof: Path
    lease: dict[str, str]
    lease_stack: ExitStack


def commit_protocol(repo: Path, name: str, value: dict[str, Any]) -> Path:
    path = _write(repo / "bench" / "protocols" / name, value)
    _git(repo, "add", path.relative_to(repo).as_posix())
    _git(repo, "commit", "-q", "-m", f"register {name}")
    return path


def open_lease(protocol_path: Path, repo: Path, **kwargs: Any) -> tuple[dict[str, str], ExitStack]:
    stack = ExitStack()
    _OPEN.append(stack)
    return stack.enter_context(J.resource_lease(protocol_path, repo=repo, **kwargs)), stack


def other_protocol(world: World, resource_id: str) -> Path:
    """같은 source·catalog에 recording 자원만 다른 등록 protocol(커밋)."""
    value = J.build_protocol(**_build_args(world, protocol_id=f"agentic-play-{resource_id}",
                                           recording_target=dict(TARGET, resource_id=resource_id)))
    return commit_protocol(world.repo, f"{resource_id}.json", value)


def _repo(root: Path) -> tuple[Path, str, dict[str, str]]:
    repo = root / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.com")
    _git(repo, "config", "user.name", "fixture")
    _git(repo, "config", "core.autocrlf", "false")
    for relative in sorted(J.REQUIRED_SOURCE_FILES):
        if relative == SIDECAR.SOURCE_PATH:  # sidecar worker는 실제 파일 SHA 결합을 확인한다
            _write(repo / relative, (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
        else:
            _write(repo / relative, f"# fixture {relative}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    sources = {relative: J._sha(repo / relative, lf=True) for relative in J.REQUIRED_SOURCE_FILES}
    return repo, _git(repo, "rev-parse", "HEAD"), sources


def _lab(root: Path) -> tuple[Path, dict[str, str], list[dict[str, Any]]]:
    lab = root / "isaaclab"
    runtime = {}
    for relative in sorted(J.REQUIRED_RUNTIME_FILES):
        runtime[relative] = J._sha(_write(lab / relative, f"# runtime {relative}\n"), lf=True)
    rows = []
    for identity, run in ((F2_ID, "2026-10-13_19-53-47_h06_s2033"),
                          (NONE_ID, "2026-10-13_19-44-24_baseline_p0c_s2033")):
        path = _write(lab / "logs" / "rsl_rl" / "unitree_go2_flat" / run / "model_299.pt", identity.encode())
        rows.append({"logical_id": identity, "isaaclab_relative": path.relative_to(lab).as_posix(),
                     "sha256": J._sha(path)})
    return lab, runtime, rows


def build_world(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, now: datetime | None = None,
    policy: str = SIDECAR.ABORT, budget: str = "2.00",
) -> World:
    now = now or datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10
    repo, commit, sources = _repo(root)
    monkeypatch.setattr(PLAYBACK, "ROOT", repo.resolve())
    monkeypatch.setattr(J, "LEASE_ROOT", root / "leases")
    monkeypatch.setattr(J.COMMON, "_python_executable", lambda path: Path(path))
    lab, runtime, rows = _lab(root)
    protocol = J.build_protocol(
        protocol_id="agentic-play-20261028a",
        registered_at_utc=(now - timedelta(minutes=1)).isoformat(),
        source_commit=commit, source_sha256_lf=sources, runtime_source_sha256_lf=runtime,
        recording_target=TARGET, compute_quote_usd_per_h="1.74", batch_budget_usd=budget,
        deadline_utc=(now + timedelta(hours=2)).isoformat(),
        checkpoints=rows, faults={F2_ID: "F2", NONE_ID: "NONE"},
        pairs=[{"pair_id": "go2-s2033", "seed": 2033, "case": F2_ID, "reference": NONE_ID}],
        sidecar_error_policy=policy,
    )
    protocol_path = commit_protocol(repo, "agentic_play_fixture.json", protocol)
    lease, stack = open_lease(protocol_path, repo)
    proof = write_proof(root, protocol_path, observed=now, lease=lease)
    return World(root, now, repo, commit, sources, lab, runtime, rows, protocol_path, protocol, proof,
                 lease, stack)


def write_proof(root: Path, protocol_path: Path, *, observed: datetime, lease: Any, process_text: str = "",
                target: dict[str, str] | None = None, name: str = "resource.json") -> Path:
    target = target or TARGET
    provider = _write(root / "admission" / "provider.json", {"workspaces": [{
        "build_status": "READY", "gpu": "L40S", "health_status": "HEALTHY",
        "id": target["resource_id"], "instance_kind": "gpu", "instance_type": target["instance_type"],
        "name": target["resource_name"], "shell_status": "READY", "status": "RUNNING",
    }]})
    gpu = _write(root / "admission" / "gpu.csv", "GPU-AGENTIC\n")
    processes = _write(root / "admission" / "processes.csv", process_text)
    capture = {
        "resource_id": target["resource_id"], "gpu_uuid": "GPU-AGENTIC",
        "provider_inventory_path": str(provider.resolve()), "provider_inventory_sha256": J._sha(provider),
        "nvidia_gpu_csv_path": str(gpu.resolve()), "nvidia_gpu_csv_sha256": J._sha(gpu),
        "nvidia_process_csv_path": str(processes.resolve()), "nvidia_process_csv_sha256": J._sha(processes),
    }
    value: dict[str, Any] = {
        "schema": J.RESOURCE_SCHEMA, "status": "granted",
        "observed_at_utc": observed.isoformat(),
        "expires_at_utc": (observed + timedelta(hours=1)).isoformat(),
        "recording_protocol_sha256_lf": J._sha(protocol_path, lf=True),
        "recording_target": target, "capture": capture,
    }
    if lease is not OMIT:
        value["lease"] = lease
    return _write(root / name, value)


def write_job(world: World, identity: str = F2_ID, *, stage_name: str | None = None,
              **changes: Any) -> Path:
    stage = world.root / (stage_name or identity)
    row = next(item for item in world.rows if item["logical_id"] == identity)
    job_id = f"{world.protocol['protocol_id']}-{identity}"
    job: dict[str, Any] = {
        "schema": J.JOB_SCHEMA, "mode": J.MODE, "job_id": job_id,
        "source": {"repo_root": str(world.repo.resolve()), "commit": world.commit,
                   "files_lf": world.protocol["source_sha256_lf"]},
        "protocol": {"path": str(world.protocol_path.resolve()),
                     "sha256_lf": J._sha(world.protocol_path, lf=True)},
        "resource_proof": {"path": str(world.proof.resolve()), "sha256": J._sha(world.proof)},
        "isaaclab": str(world.lab.resolve()), "python": str(Path(sys.executable).absolute()),
        "output_directory": str(stage.absolute() / "output"), "profile": J.RECORD.fixed_profile(),
        "checkpoint": {"path": str((world.lab / row["isaaclab_relative"]).resolve()),
                       "sha256": row["sha256"], "logical_id": identity},
        "sidecar": {"directory": str(J.sidecar_directory_for(stage)), "source_id": job_id,
                    "error_policy": world.protocol["sidecar"]["error_policy"],
                    "expected_frames": J.TECHNICAL_LIMITS["frames_per_policy"]},
        "lease": world.lease,
    }
    job.update(changes)
    return _write(stage / "job.json", job)


def _rewrite_protocol(world: World, mutate: Any) -> None:
    value = json.loads(world.protocol_path.read_text(encoding="utf-8"))
    mutate(value)
    _write(world.protocol_path, value)


# --- 기본 결합 -----------------------------------------------------------------------


def test_f2_and_none_jobs_bind_registration_pair_and_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    for identity, fault in ((F2_ID, "F2"), (NONE_ID, "NONE")):
        checked = J.validate_job(write_job(world, identity), now=world.now)
        assert checked["playback_registration"] == PLAYBACK.registration(
            identity, checked["checkpoint_sha256"], fault)
        assert checked["protocol"]["playback_registrations"][identity]["fault_id"] == fault
        assert checked["pair"] == {"pair_id": "go2-s2033", "seed": 2033, "case": F2_ID, "reference": NONE_ID}
        assert checked["sidecar_directory"] == tmp_path / f"{identity}-sidecar"
        assert checked["resource"]["resource_id"] == TARGET["resource_id"]
    assert world.protocol["interpretation"]["single_leg_fault"] is False
    assert world.protocol["sidecar"] == {"error_policy": SIDECAR.ABORT, "required": True,
                                         "source_files": list(SIDECAR.SOURCE_FILES)}
    assert world.protocol["limits"]["max_policies"] == 2


def test_validation_never_reads_f1_historical_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("F1 historical catalog를 읽었다")

    monkeypatch.setattr(J.TRANSFERRED, "_historical", forbidden)
    world = build_world(tmp_path, monkeypatch)
    J.validate_job(write_job(world), now=world.now)
    assert J.TRANSFERRED.CHECKPOINT_IDS == ("source-s2220-none", "source-s2220-f1")


def test_create_job_writes_sidecar_rule_outside_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    path = J.create_job(world.protocol_path, world.proof, J._sha(world.proof), world.lab,
                        Path(sys.executable), tmp_path / "remote" / NONE_ID, NONE_ID, repo=world.repo)
    job = json.loads(path.read_text(encoding="utf-8"))
    assert job["sidecar"]["directory"] == str(tmp_path / "remote" / f"{NONE_ID}-sidecar")
    assert job["sidecar"]["source_id"] == f"agentic-play-20261028a-{NONE_ID}"


# --- catalog·checkpoint --------------------------------------------------------------


def test_unregistered_checkpoint_or_changed_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    job = write_job(world, checkpoint={"path": "x", "sha256": "0" * 64, "logical_id": "go2-s9999-f2"})
    with pytest.raises(ValueError, match="catalog 등록 밖"):
        J.validate_job(job, now=world.now)
    other = write_job(world, stage_name="other", job_id="wrong-id")
    with pytest.raises(ValueError, match="job ID"):
        J.validate_job(other, now=world.now)
    _write(world.lab / world.rows[0]["isaaclab_relative"], b"tampered")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        J.validate_job(write_job(world, stage_name="third"), now=world.now)


@pytest.mark.parametrize("relative", ["other/model.pt", "logs/rsl_rl/x/model.txt", "logs/rsl_rl/../x.pt"])
def test_catalog_path_outside_isaaclab_logs_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    world = build_world(tmp_path, monkeypatch)
    rows = [dict(world.rows[0], isaaclab_relative=relative), world.rows[1]]
    with pytest.raises(ValueError):
        J.build_protocol(**_build_args(world, checkpoints=rows))


def _build_args(world: World, **changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "protocol_id": "agentic-play-20261028b",
        "registered_at_utc": (world.now - timedelta(minutes=1)).isoformat(),
        "source_commit": world.commit, "source_sha256_lf": world.sources,
        "runtime_source_sha256_lf": world.runtime, "recording_target": TARGET,
        "compute_quote_usd_per_h": "1.74", "batch_budget_usd": "2.00",
        "deadline_utc": (world.now + timedelta(hours=2)).isoformat(),
        "checkpoints": world.rows, "faults": {F2_ID: "F2", NONE_ID: "NONE"},
        "pairs": [{"pair_id": "go2-s2033", "seed": 2033, "case": F2_ID, "reference": NONE_ID}],
    }
    args.update(changes)
    return args


@pytest.mark.parametrize("change", [
    {"faults": {F2_ID: "F2", NONE_ID: "F2"}},
    {"faults": {F2_ID: "F1", NONE_ID: "NONE"}},
    {"pairs": [{"pair_id": "go2-s2033", "seed": 2033, "case": NONE_ID, "reference": F2_ID}]},
    {"pairs": []},
    {"batch_budget_usd": "0.579"},
    {"sidecar_error_policy": "ignore"},
], ids=["two-f2", "f1-fault", "swapped-pair", "no-pair", "budget", "sidecar-policy"])
def test_build_rejects_pair_fault_budget_or_sidecar_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any]
) -> None:
    world = build_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        J.build_protocol(**_build_args(world, **change))


def test_registration_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    original = world.protocol_path.read_text(encoding="utf-8")

    def hidden(value: dict[str, Any]) -> None:
        value["playback_registrations"][F2_ID]["hidden_faults_sha256_lf"] = "0" * 64

    _rewrite_protocol(world, hidden)
    with pytest.raises(ValueError, match="playback registration"):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)
    _write(world.protocol_path, original)

    def flip(value: dict[str, Any]) -> None:  # 유효한 NONE 등록이지만 pair가 F2·NONE 쌍이 아니게 된다
        value["playback_registrations"][F2_ID] = PLAYBACK.registration(F2_ID, world.rows[0]["sha256"], "NONE")

    _rewrite_protocol(world, flip)
    with pytest.raises(ValueError, match="pair"):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)


@pytest.mark.parametrize("key", ["interpretation", "limits", "record_profile"])
def test_interpretation_limits_and_profile_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    world = build_world(tmp_path, monkeypatch)

    def mutate(value: dict[str, Any]) -> None:
        if key == "interpretation":
            value[key]["single_leg_fault"] = True
        elif key == "limits":
            value[key]["max_policies"] = 4
        else:
            value[key]["value"]["fps"] = 30

    _rewrite_protocol(world, mutate)
    with pytest.raises(ValueError):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)


# --- source·runtime pin ---------------------------------------------------------------


def test_source_closure_requires_new_sidecar_playback_and_own_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    assert J.HELPER_SOURCE_FILES <= J.REQUIRED_SOURCE_FILES
    for missing in (SIDECAR.SOURCE_PATH, "bench/agentic_fault_playback_v1.py", "bench/hidden_faults.py",
                    "bench/run_agentic_play_record_v1.py", "tests/test_agentic_play_job_v1.py",
                    *sorted(J.HELPER_SOURCE_FILES)):
        sources = {key: value for key, value in world.sources.items() if key != missing}
        with pytest.raises(ValueError, match="source closure"):
            J.build_protocol(**_build_args(world, source_sha256_lf=sources))


@pytest.mark.parametrize("changed", [
    "bench/run_agentic_play_record_v1.py",
    "evals/vision_brev_backend_v1.py",  # OWNED: OS lock·owner start·cleanup
    "evals/vision_remote_gpu_v1.py",  # REMOTE: provider·busy GPU
])
def test_source_change_after_commit_or_foreign_commit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    world = build_world(tmp_path, monkeypatch)
    J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)  # 변경 전에는 통과
    _write(world.repo / changed, "# changed after commit\n")
    with pytest.raises(ValueError):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)
    _rewrite_protocol(world, lambda value: value.update(source_commit="f" * 40))
    with pytest.raises(ValueError, match="HEAD 조상"):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now)


def test_runtime_pin_requires_dcmotor_and_rejects_changed_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    runtime = {key: value for key, value in world.runtime.items()
               if key != "source/isaaclab/isaaclab/actuators/actuator_pd.py"}
    with pytest.raises(ValueError, match="runtime pin"):
        J.build_protocol(**_build_args(world, runtime_source_sha256_lf=runtime))
    _write(world.lab / "source/isaaclab/isaaclab/actuators/actuator_pd.py", "# other install\n")
    with pytest.raises(ValueError, match="runtime source SHA"):
        J.validate_job(write_job(world), now=world.now)


# --- 자원 입장 ---------------------------------------------------------------------


@pytest.mark.parametrize("case", ["other-resource", "stale", "proof-file-missing", "busy-gpu", "sha", "future"])
def test_resource_admission_rejects_other_stale_missing_or_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    world = build_world(tmp_path, monkeypatch)
    proof = world.proof
    sha = J._sha(proof)
    lease = world.lease
    if case == "other-resource":
        other = dict(TARGET, resource_id="old-l40s-id")
        proof = write_proof(tmp_path, world.protocol_path, observed=world.now, lease=lease, target=other,
                            name="other.json")
        sha = J._sha(proof)
    elif case == "stale":
        proof = write_proof(tmp_path, world.protocol_path, observed=world.now - timedelta(seconds=31),
                            lease=lease, name="stale.json")
        sha = J._sha(proof)
    elif case == "proof-file-missing":
        proof = tmp_path / "absent-proof.json"
    elif case == "busy-gpu":
        proof = write_proof(tmp_path, world.protocol_path, observed=world.now, lease=lease,
                            process_text="GPU-AGENTIC, 77\n", name="busy.json")
        sha = J._sha(proof)
    elif case == "sha":
        sha = "0" * 64
    else:
        proof = write_proof(tmp_path, world.protocol_path, observed=world.now + timedelta(minutes=5),
                            lease=lease, name="future.json")
        sha = J._sha(proof)
    job = write_job(world, resource_proof={"path": str(proof.absolute()), "sha256": sha})
    with pytest.raises(ValueError):
        J.validate_job(job, now=world.now)


def test_deadline_and_registration_time_are_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="deadline"):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now + timedelta(hours=3))
    with pytest.raises(ValueError, match="deadline"):
        J.validate_protocol(world.protocol_path, repo=world.repo, now=world.now - timedelta(hours=1))


def test_uncommitted_or_outside_protocol_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    loose = _write(world.repo / "bench" / "protocols" / "not_committed.json", world.protocol)
    with pytest.raises(ValueError, match="커밋"):
        J.validate_protocol(loose, repo=world.repo, now=world.now)
    outside = _write(tmp_path / "outside.json", world.protocol)
    with pytest.raises(ValueError, match="커밋"):
        J.validate_protocol(outside, repo=world.repo, now=world.now)


# --- 실제 임대(lease) ------------------------------------------------------------------


def _verify(world: World, reference: Any, protocol_path: Path | None = None, **kwargs: Any) -> dict[str, Any]:
    path = protocol_path or world.protocol_path
    protocol = J.validate_protocol(path, repo=world.repo, now=world.now)
    return J.verify_lease(reference, protocol=protocol, protocol_sha=J._sha(path, lf=True), **kwargs)


def test_active_lease_is_verified_and_bound_to_proof_and_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    info = _verify(world, world.lease)
    assert info["resource_id"] == TARGET["resource_id"] and info["owner"]["pid"] == os.getpid()
    assert info["live_lock_proof"] is (os.name == "posix")  # Windows probe는 fixture 증거로만 남는다
    checked = J.validate_job(write_job(world), now=world.now)
    assert checked["lease"]["lease_id"] == info["lease_id"]
    snapshot = json.loads(Path(world.lease["path"]).read_text(encoding="utf-8"))
    assert snapshot["lock_path"] == str(tmp_path / "leases" / f"{TARGET['resource_id']}.lock")


def test_proof_without_lease_or_with_replaced_lease_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    no_lease = write_proof(tmp_path, world.protocol_path, observed=world.now, lease=OMIT, name="no-lease.json")
    job = write_job(world, stage_name="a", resource_proof={"path": str(no_lease), "sha256": J._sha(no_lease)})
    with pytest.raises(ValueError, match="필드"):
        J.validate_job(job, now=world.now)
    snapshot = json.loads(Path(world.lease["path"]).read_text(encoding="utf-8"))
    forged_path = _write(Path(world.lease["path"]).parent / f"{TARGET['resource_id']}-forged.json",
                         dict(snapshot, lease_id="f" * 32))
    forged = {"path": str(forged_path), "sha256": J._sha(forged_path)}
    with pytest.raises(J.LeaseError, match="heartbeat"):
        _verify(world, forged)
    with pytest.raises(J.LeaseError, match="SHA"):
        _verify(world, dict(world.lease, sha256="0" * 64))
    other_proof = write_proof(tmp_path, world.protocol_path, observed=world.now, lease=forged, name="forged.json")
    job = write_job(world, stage_name="b", resource_proof={"path": str(other_proof), "sha256": J._sha(other_proof)})
    with pytest.raises(J.LeaseError):
        J.validate_job(job, now=world.now)
    job = write_job(world, stage_name="c", lease=forged)
    with pytest.raises(J.LeaseError, match="job lease"):
        J.validate_job(job, now=world.now)


def test_stale_heartbeat_and_expired_lease_are_rejected_even_without_freshness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    later = datetime.now(timezone.utc) + timedelta(seconds=J.HEARTBEAT_MAX_AGE_S + 5)  # noqa: UP017
    with pytest.raises(J.LeaseError, match="heartbeat"):
        _verify(world, world.lease, now=later)
    with pytest.raises(J.LeaseError, match="heartbeat"):
        J.validate_job(write_job(world), now=later, require_fresh=False)  # 30초 원문 기한을 꺼도 lease는 확인
    with pytest.raises(J.LeaseError, match="유효 기간"):
        _verify(world, world.lease, now=world.now + timedelta(hours=1, minutes=1))


def test_dead_owner_lease_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    protocol = other_protocol(world, "dead-owner-l40s")
    paths = [str(ROOT / name) for name in ("src", "evals", "bench")]
    script = "\n".join([
        "import json, os, sys",
        "from pathlib import Path",
        "sys.dont_write_bytecode = True",
        f"[sys.path.insert(0, p) for p in {paths!r}]",
        "import agentic_fault_playback_v1 as PLAYBACK, agentic_play_job_v1 as J",
        f"J.LEASE_ROOT = Path({str(tmp_path / 'leases')!r})",
        f"PLAYBACK.ROOT = Path({str(world.repo.resolve())!r})",
        f"context = J.resource_lease(Path({str(protocol)!r}), repo=Path({str(world.repo)!r}))",
        "print(json.dumps(context.__enter__()), flush=True)",
        "os._exit(0)  # 정리 없이 종료: OS가 lock을 풀고 heartbeat는 active로 남는다",
    ])
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            check=True, timeout=120)
    reference = json.loads(result.stdout.strip().splitlines()[-1])
    with pytest.raises(J.LeaseError, match="owner"):
        _verify(world, reference, protocol)


def test_unlocked_lease_with_forged_heartbeat_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    protocol = other_protocol(world, "unlocked-l40s")
    reference, stack = open_lease(protocol, world.repo)
    stack.close()
    heartbeat = tmp_path / "leases" / "unlocked-l40s.heartbeat.json"
    value = json.loads(heartbeat.read_text(encoding="utf-8"))
    value.update(active=True, updated_at_utc=datetime.now(timezone.utc).isoformat())  # noqa: UP017
    _write(heartbeat, value)
    with pytest.raises(J.LeaseError, match="lock"):
        _verify(world, reference, protocol)


def test_same_resource_contender_is_refused_and_other_resource_runs_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    with pytest.raises(J.LeaseError, match="이미"):
        open_lease(world.protocol_path, world.repo)
    protocol = other_protocol(world, "parallel-l40s")
    reference, _stack = open_lease(protocol, world.repo)
    assert _verify(world, reference, protocol)["resource_id"] == "parallel-l40s"
    assert _verify(world, world.lease)["resource_id"] == TARGET["resource_id"]


def test_closing_lease_stops_heartbeat_marks_inactive_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path, monkeypatch)
    protocol = other_protocol(world, "cleanup-l40s")
    reference, stack = open_lease(protocol, world.repo)
    lease_id = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))["lease_id"]
    assert any(thread.name == f"agentic-play-lease-{lease_id}" for thread in threading.enumerate())
    stack.close()
    assert not any(thread.name == f"agentic-play-lease-{lease_id}" for thread in threading.enumerate())
    heartbeat = json.loads((tmp_path / "leases" / "cleanup-l40s.heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["active"] is False and heartbeat["lease_id"] == lease_id
    with pytest.raises(J.LeaseError):
        _verify(world, reference, protocol)
    again, _stack = open_lease(protocol, world.repo)  # 해제 뒤 같은 자원을 다시 잡을 수 있다
    assert _verify(world, again, protocol)["lease_id"] != lease_id


def test_proc_locks_parser_keeps_only_active_flock_writers() -> None:
    lines = [
        "1: FLOCK  ADVISORY  WRITE 4242 08:01:131074 0 EOF",
        "1: -> FLOCK  ADVISORY  WRITE 5151 08:01:131074 0 EOF",
        "2: POSIX  ADVISORY  WRITE 4242 08:01:999 0 EOF",
        "3: FLOCK  ADVISORY  READ  4242 08:01:777 0 EOF",
        "4: FLOCK  ADVISORY  WRITE 7 fd:00:12 0 EOF",
    ]
    assert J.proc_lock_owners("\n".join(lines)) == {(4242, 8, 1, 131074), (7, 253, 0, 12)}


# --- job 경계 --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["inside-stage", "policy", "source-id", "frames"])
def test_job_sidecar_must_follow_protocol_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    world = build_world(tmp_path, monkeypatch)
    stage = tmp_path / F2_ID
    sidecar = {"directory": str(J.sidecar_directory_for(stage)), "source_id": f"agentic-play-20261028a-{F2_ID}",
               "error_policy": SIDECAR.ABORT, "expected_frames": 625}
    if field == "inside-stage":
        sidecar["directory"] = str(stage / "sidecar")
    elif field == "policy":
        sidecar["error_policy"] = SIDECAR.ISOLATE
    elif field == "source-id":
        sidecar["source_id"] = "other"
    else:
        sidecar["expected_frames"] = 1
    with pytest.raises(ValueError, match="sidecar"):
        J.validate_job(write_job(world, sidecar=sidecar), now=world.now)


def test_existing_sidecar_or_output_blocks_fresh_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    job = write_job(world)
    (tmp_path / f"{F2_ID}-sidecar").mkdir()
    with pytest.raises(FileExistsError):
        J.validate_job(job, now=world.now)
    checked = J.validate_job(job, now=world.now, require_fresh=False)
    assert checked["sidecar_directory"].exists()


def test_f1_transferred_job_schema_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world = build_world(tmp_path, monkeypatch)
    job = write_job(world, schema="vision_transferred_pull_job_v3")
    with pytest.raises(ValueError, match="schema"):
        J.validate_job(job, now=world.now)


# --- 메인 hand-off helper ----------------------------------------------------------------


def test_motor_catalog_and_batch_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = "C:\\Lab\\IsaacLab"

    def policy(run: str, sha: str) -> dict[str, Any]:
        return {"checkpoint": {"path": f"{lab}\\logs\\rsl_rl\\unitree_go2_flat\\{run}\\model_299.pt", "sha256": sha}}

    audit = {"schema": "ri_agentic_motor_sources_audit_v1", "candidates": [
        {"fault": "F2", "robot": "Go2", "seed": seed,
         "policies": {"case": policy(f"h06_s{seed}", f"{index}" * 64),
                      "reference": policy(f"baseline_s{seed}", f"{index + 2}" * 64)}}
        for index, seed in ((1, 2033), (3, 2046))
    ]}
    catalog = J.motor_catalog(audit, isaaclab_root=lab)
    assert [row["logical_id"] for row in catalog["checkpoints"]] == [
        "go2-s2033-f2", "go2-s2033-none", "go2-s2046-f2", "go2-s2046-none"]
    assert catalog["checkpoints"][0]["isaaclab_relative"] == "logs/rsl_rl/unitree_go2_flat/h06_s2033/model_299.pt"
    assert catalog["pairs"][1] == {"pair_id": "go2-s2046", "seed": 2046,
                                   "case": "go2-s2046-f2", "reference": "go2-s2046-none"}
    world = build_world(tmp_path, monkeypatch)
    layout = J.batch_layout(world.protocol, "go2-s2033", "/home/ubuntu/agentic-play")
    assert layout["max_policies"] == 2
    assert layout["policies"][0] == {
        "role": "case", "logical_id": F2_ID, "fault_id": "F2",
        "stage": f"/home/ubuntu/agentic-play/agentic-play-20261028a/{F2_ID}",
        "job": f"/home/ubuntu/agentic-play/agentic-play-20261028a/{F2_ID}/job.json",
        "output": f"/home/ubuntu/agentic-play/agentic-play-20261028a/{F2_ID}/output",
        "sidecar": f"/home/ubuntu/agentic-play/agentic-play-20261028a/{F2_ID}-sidecar",
    }
    with pytest.raises(ValueError):
        J.batch_layout(world.protocol, "go2-s2033", "relative/root")


def test_cpu_import_does_not_load_isaac_or_torch() -> None:
    script = (
        "import sys; sys.dont_write_bytecode = True; "
        f"[sys.path.insert(0, p) for p in {[str(ROOT / name) for name in ('src', 'evals', 'bench')]!r}]; "
        "import agentic_play_job_v1, run_agentic_play_record_v1; "
        "heavy = ('isaaclab', 'isaacsim', 'omni', 'carb', 'pxr', 'torch', 'gymnasium', 'rsl_rl'); "
        "print(sorted(name for name in sys.modules if name.split('.')[0] in heavy))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True,
                            check=True, timeout=120)
    assert result.stdout.strip() == "[]"
