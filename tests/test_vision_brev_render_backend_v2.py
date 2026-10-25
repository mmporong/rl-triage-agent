from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    sys.path.insert(0, str(folder))

M = importlib.import_module("vision_brev_render_backend_v2")
R = importlib.import_module("record_vision_evidence_v1")


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> list[dict]:
    return [
        {
            "job_id": f"s{seed}-{arm.lower()}",
            "arm": arm,
            "seed": seed,
            "run_name": f"s{seed}-{arm.lower()}",
            "stage_relative": f"training/s{seed}-{arm.lower()}",
            "iterations": 300,
            "num_envs": 4096,
            "source_job_id": f"s{seed}-f1" if arm == "CORRECTED" else None,
        }
        for seed in (301, 302)
        for arm in ("NONE", "F1", "CORRECTED")
    ]


class Reader:
    def __init__(self, root: Path, *, busy: bool = False):
        self.root = root
        self.busy = busy
        self.count = 0
        self.last_receipt: dict[str, str] | None = None

    def __call__(self) -> list[int]:
        self.count += 1
        path = self.root / f"gpu-{self.count}.json"
        _write(path, {"count": self.count})
        self.last_receipt = {"path": str(path.resolve()), "sha256": _sha(path)}
        return [99] if self.busy else []


class Recovery:
    def __init__(self, root: Path, popen, reader: Reader):
        self.lab = root
        self.python = Path(sys.executable)
        self.popen = popen
        self.read_gpu = reader
        self.snapshot = {"fixture": "stable"}
        now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 regression
        self.protocol = {
            "protocol_id": "attested-test",
            "protocol_sha256_lf": "a" * 64,
            "training_plan": _plan(),
            "resources": {
                "deadline_utc": (now + timedelta(hours=1)).isoformat(),
                "compute_quote_usd_per_h": 2.0,
                "action_limits": {"record": {"max_wall_s": 900, "max_compute_usd": 0.5}},
            },
            "recording": {"source_files_lf": {}},
        }
        self.vision_resource_proof = {
            "resource_id": "instance-1",
            "gpu_uuid": "GPU-fixture",
            "lease_id": "lease-1",
            "issued_at_utc": now.isoformat(),
            "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
        }

    def _source_snapshot(self) -> dict[str, str]:
        return dict(self.snapshot)


def _job(repo: Path, runtime: Path, mode: str = R.RENDER_SYNC_SMOKE_MODE) -> dict:
    now = datetime.now(timezone.utc)  # noqa: UP017 - Python 3.10 regression
    protocol = repo / "capture-protocol.json"
    checkpoint = repo / "model.pt"
    admission = repo / "admission.json"
    source = repo / "source.py"
    source.write_bytes(b"value = 1\n")
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    source_sha = _sha(source)
    checkpoint.write_bytes(b"model")
    resource = {"resource_id": "instance-1", "gpu_uuid": "GPU-fixture"}
    _write(
        protocol,
        {
            "schema": R.PROTOCOL_SCHEMA,
            "method": R.METHOD,
            "mode": mode,
            "profile": R.fixed_profile(),
            "source_commit": commit,
            "source_files_lf": {"source.py": source_sha},
            "checkpoint_sha256": _sha(checkpoint),
            "resource": resource,
        },
    )
    base = {
        "schema": R.JOB_SCHEMA,
        "method": R.METHOD,
        "mode": mode,
        "job_id": "smoke-1" if mode == R.RENDER_SYNC_SMOKE_MODE else "record-1",
        "profile": R.fixed_profile(),
        "source": {"repo_root": str(repo), "commit": commit, "files_lf": {"source.py": source_sha}},
        "protocol": {"path": str(protocol.resolve()), "sha256": _sha(protocol)},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint), "logical_id": "policy"},
        "resource": resource,
        "remote_gpu_admission": {
            "path": str(admission.resolve()),
            "sha256": "pending",
            "resource_parent_sha256": "c" * 64,
            "live_lease_proof_sha256": "d" * 64,
        },
        "output_directory": str((runtime / "record" / ("smoke-1" if mode == R.RENDER_SYNC_SMOKE_MODE else "record-1") / "output").resolve()),
    }
    _write(
        admission,
        {
            "schema": R.ADMISSION_SCHEMA,
            "status": "granted",
            "job_id": base["job_id"],
            "resource_kind": "remote_gpu",
            "lease_id": "lease-1",
            "request_binding": R.request_binding(base),
            "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
            "issued_at_utc": now.isoformat(),
            "resource_id": resource["resource_id"],
            "gpu_uuid": resource["gpu_uuid"],
            "resource_parent_sha256": base["remote_gpu_admission"]["resource_parent_sha256"],
            "live_lease_proof_sha256": base["remote_gpu_admission"]["live_lease_proof_sha256"],
        },
    )
    base["remote_gpu_admission"]["sha256"] = _sha(admission)
    return base


def _worker_script(path: Path, *, behavior: str = "ok") -> None:
    source = """
import hashlib,json,sys,time
from pathlib import Path
job_path=Path(sys.argv[-1]); job=json.loads(job_path.read_text(encoding='utf-8')); output=Path(job['output_directory'])
if job['mode']=='record':
 sys.path.insert(0,TESTS_DIR)
 from test_vision_brev_backend_v1 import FakeProcess
 FakeProcess(job).wait(900)
 (output/'worker_argv.json').write_text(json.dumps(sys.argv)+'\\n')
 raise SystemExit(0)
BEHAVIOR
output.mkdir(parents=True)
(output/'worker_argv.json').write_text(json.dumps(sys.argv)+'\\n')
rgb_a=b'a'*(640*480*3); rgb_b=b'b'*(640*480*3)
rows=[]
for index,(label,data,eye) in enumerate([('warmup',rgb_a,[3.0,3.0,2.5]),('A',rgb_a,[3.0,3.0,2.5]),('B',rgb_b,[-3.0,3.0,2.5]),('A',rgb_a,[3.0,3.0,2.5])]):
 read_times={'durationDenominator':50,'durationNumerator':1,'externalTimeOfSimNs':1000000000,'frameNumber':50,'rationalTimeOfSimDenominator':50,'rationalTimeOfSimNumerator':50,'sampleTimeOffsetInSimFrames':0,'simulationTime':1.0,'simulationTimeMonotonic':1.0,'swhFrameNumber':50,'systemTime':1001.0}
 rows.append({'label':label,'rgb_sha256':hashlib.sha256(data).hexdigest(),'simulation_time_s':1.0,'read_times':read_times,'reference_time':{'referenceTimeNumerator':1,'referenceTimeDenominator':1},'camera':{'eye_world_m':eye,'lookat_world_m':[0.0,0.0,0.85]}})
artifacts={}
report=output/'render_sync_smoke.json'; report.write_text(json.dumps({'schema':'vision_render_sync_smoke_v1','status':'pass','capture_order':['warmup_discard','A','B','A'],'policy_steps':0,'capture_phase_physics_steps':0,'environment_reset_performed':True,'delta_time_s':0.0,'rt_subframes':1,'pause_timeline':False,'renderer_combination_only':True,'universal_zero_lag_claim':False,'captures':rows},sort_keys=True)+'\\n')
artifacts['render_sync_smoke']=report
for index,data in enumerate((rgb_a,rgb_b,rgb_a),1):
 p=output/f'audit_rgb_{index}.rgb'; p.write_bytes(data); artifacts[f'audit_rgb_{index}']=p
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
binding={'method':job['method'],'mode':job['mode'],'job_id':job['job_id'],'profile':job['profile'],'source':job['source'],'protocol_sha256':job['protocol']['sha256'],'checkpoint_sha256':job['checkpoint']['sha256'],'resource':job['resource'],'resource_parent_sha256':job['remote_gpu_admission']['resource_parent_sha256'],'live_lease_proof_sha256':job['remote_gpu_admission']['live_lease_proof_sha256']}
admission=json.loads(Path(job['remote_gpu_admission']['path']).read_text(encoding='utf-8'))
result={'schema':'vision_play_record_result_v1','completed':True,'job_sha256':sha(job_path),'protocol_sha256':job['protocol']['sha256'],'checkpoint_sha256':job['checkpoint']['sha256'],'request_binding':digest(binding),'resource':job['resource'],'resource_parent_sha256':job['remote_gpu_admission']['resource_parent_sha256'],'live_lease_proof_sha256':job['remote_gpu_admission']['live_lease_proof_sha256'],'admission_sha256':job['remote_gpu_admission']['sha256'],'admission_issued_at_utc':admission['issued_at_utc'],'admission_expires_at_utc':admission['expires_at_utc'],'profile_sha256':digest(job['profile']),'mode':job['mode'],'primary_artifact_sha256':sha(report),'video_sha256':None,'artifacts':{key:{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for key,p in artifacts.items()},'fixed_evaluation_used_as_behavior_ground_truth':False,'independent_visual_evidence_source_count':1,'zero_update_pixel_lag_confirmed':False,'gpu_smoke_required_before_model_use':False}
(output/'result.json').write_text(json.dumps(result,sort_keys=True)+'\\n')
"""
    behavior_source = {
        "ok": "pass",
        "exit": "sys.exit(7)",
        "missing": "sys.exit(0)",
        "hang": "time.sleep(60)",
    }[behavior]
    path.write_text(
        source.replace("TESTS_DIR", repr(str(ROOT / "tests"))).replace("BEHAVIOR", behavior_source),
        encoding="utf-8",
    )


def _install_harness(tmp_path: Path, monkeypatch, *, behavior: str = "ok"):
    repo = tmp_path / "repo"
    repo.mkdir()
    runtime = repo / ".triage-runtime/vision-brev/attested-test"
    fixture = repo / "fixture_worker.py"
    _worker_script(fixture, behavior=behavior)
    launches: list[tuple[list[str], dict]] = []

    def popen(argv, **kwargs):
        launch_argv = argv
        if os.name == "nt":
            windows_gate = (
                "import runpy,sys\n"
                "command=sys.argv[1:]\n"
                "if sys.stdin.buffer.readline()!=b'GO\\n': raise SystemExit(97)\n"
                "sys.argv=command\n"
                "runpy.run_path(command[0],run_name='__main__')\n"
            )
            launch_argv = [sys.executable, "-c", windows_gate, str(fixture), "--job", argv[-1]]
        process = subprocess.Popen(
            launch_argv,
            cwd=kwargs["cwd"],
            env=kwargs["env"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            stdin=kwargs["stdin"],
            start_new_session=kwargs["start_new_session"],
        )
        kwargs["fixture_process"] = process
        launches.append((list(argv), kwargs))
        return process

    reader = Reader(repo)
    recovery = Recovery(repo, popen, reader)
    monkeypatch.setattr(M, "ROOT", repo)
    monkeypatch.setattr(M, "VISION_WORKER", fixture)
    monkeypatch.setattr(M.V1, "ROOT", repo)
    order: list[str] = []

    def attest(job_path, provider, python, output, execute, **kwargs):
        order.append("source_before")
        provider()
        order.append("context")
        context = output / "context"
        context.mkdir(parents=True)
        _write(context / M.CONTEXT.RECEIPT_FILE, {"fixture": True})
        order.append("execute")
        result = execute()
        provider()
        order.append("seal")
        _write(output / M.ATTEST.ATTESTATION_FILE, {"fixture": True, "result": result})
        return output / M.ATTEST.ATTESTATION_FILE

    monkeypatch.setattr(M.ATTEST, "attest_capture", attest)

    def verify_capture(repo_root, refs, mode, *args):
        paths = {key: repo_root / value["path"] for key, value in refs.items()}
        for key, path in paths.items():
            assert path.is_file(), key
            if _sha(path) != refs[key]["sha256"]:
                raise ValueError("원본 파일 SHA가 다르다")
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        artifact_keys = (
            M.SOURCES._RECORD_ARTIFACTS
            if mode == R.RECORD_MODE
            else M.SOURCES._SMOKE_ARTIFACTS
        )
        artifact_paths = {key: paths[key] for key in artifact_keys}
        for key, path in artifact_paths.items():
            if _sha(path) != result["artifacts"][key]["sha256"]:
                raise ValueError("실행 artifact 이름·바이트·크기가 다르다")
        return {"result": result, "artifact_paths": artifact_paths}

    monkeypatch.setattr(M.SOURCES, "_verify_capture", verify_capture)
    backend = M.AttestedVisionBrevBackend(recovery, runtime, _plan())
    return backend, runtime, launches, order, recovery


def test_smoke_runs_fixed_argv_attests_before_return_and_cache_executes_zero_children(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, order, _ = _install_harness(tmp_path, monkeypatch)
    job = _job(M.ROOT, runtime)
    frozen_validate = M.R.validate_job
    validated_shapes: list[set[str]] = []

    def validate_with_shape(path: Path) -> dict:
        validated = frozen_validate(path)
        validated_shapes.append(set(validated))
        assert validated["job"]["mode"] == R.RENDER_SYNC_SMOKE_MODE
        return validated

    monkeypatch.setattr(M.R, "validate_job", validate_with_shape)
    result = backend.smoke(job)
    assert result["status"] == "render_sync_smoke_complete"
    assert result["model_use_ready"] is False
    assert order == ["source_before", "context", "execute", "seal"]
    assert launches[0][0] == [
        str(sys.executable), "-I", "-S", "-c", M.OWNED_WORKER_GATE,
        str(sys.executable), "-B", str(M.VISION_WORKER), "--job",
        str(runtime / "record/smoke-1/job.json"),
    ]
    assert launches[0][1]["start_new_session"] is True
    worker_argv = json.loads((runtime / "record/smoke-1/output/worker_argv.json").read_text(encoding="utf-8"))
    assert worker_argv == [str(M.VISION_WORKER), "--job", str(runtime / "record/smoke-1/job.json")]
    assert set(result["execution"]) == set(M.ATTEST.EXECUTION_KEYS)
    assert validated_shapes and {"job", "output_directory", "request_binding"} <= validated_shapes[0]
    before = len(launches)
    cached = backend.smoke(job)
    assert cached["result"] == result["result"] and len(launches) == before


def test_record_uses_frozen_validator_and_verifier_then_cache_runs_zero_children(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, order, _ = _install_harness(tmp_path, monkeypatch)
    job = _job(M.ROOT, runtime, R.RECORD_MODE)
    frozen_validate = M.R.validate_job
    validated_jobs: list[dict] = []

    def validate_record(path: Path) -> dict:
        validated = frozen_validate(path)
        validated_jobs.append(validated["job"])
        return validated

    monkeypatch.setattr(M.R, "validate_job", validate_record)
    result = backend.record(job)
    assert result["status"] == "capture_complete_gpu_smoke_pending"
    assert result["model_use_ready"] is False
    assert result["result"]["mode"] == R.RECORD_MODE
    assert result["result"]["video_sha256"] == result["result"]["primary_artifact_sha256"]
    assert validated_jobs and all(value["mode"] == R.RECORD_MODE for value in validated_jobs)
    assert order == ["source_before", "context", "execute", "seal"]
    worker_argv = json.loads(
        (runtime / "record/record-1/output/worker_argv.json").read_text(encoding="utf-8"),
    )
    assert worker_argv == [str(M.VISION_WORKER), "--job", str(runtime / "record/record-1/job.json")]
    before = len(launches)
    cached = backend.record(job)
    assert cached["result"] == result["result"] and len(launches) == before


@pytest.mark.parametrize("behavior", ["exit", "missing"])
def test_child_exit_or_missing_result_seals_failure(tmp_path: Path, monkeypatch, behavior: str) -> None:
    backend, runtime, _, _, _ = _install_harness(tmp_path, monkeypatch, behavior=behavior)
    job = _job(M.ROOT, runtime)
    with pytest.raises(M.B.BackendError):
        backend.smoke(job)
    receipt = json.loads((runtime / "record/smoke-1/execution.json").read_text(encoding="utf-8"))
    assert receipt["completed"] is False
    assert receipt["cleanup_verified"] is True


def test_hung_cpu_child_is_cleaned_and_sealed(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, _, _ = _install_harness(tmp_path, monkeypatch, behavior="hang")
    job = _job(M.ROOT, runtime)
    monkeypatch.setattr(backend, "_action_deadline", lambda _: time.monotonic() + 2.0)
    with pytest.raises(M.B.BackendError):
        backend.smoke(job)
    receipt = json.loads((runtime / "record/smoke-1/execution.json").read_text(encoding="utf-8"))
    assert receipt["timed_out"] is True
    assert receipt["cleanup_verified"] is True
    assert isinstance(receipt["launch"], dict)
    assert launches[0][1]["fixture_process"].poll() is not None


def test_identity_failure_terminates_waiting_gate_before_worker_exec(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, _, _ = _install_harness(tmp_path, monkeypatch)
    job = _job(M.ROOT, runtime)
    monkeypatch.setattr(
        M.V1,
        "_process_identity",
        lambda _pid: (_ for _ in ()).throw(ValueError("identity fixture failure")),
    )
    with pytest.raises(M.B.BackendError, match="ValueError"):
        backend.smoke(job)
    process = launches[0][1]["fixture_process"]
    assert process.poll() is not None
    assert not (runtime / "record/smoke-1/output/worker_argv.json").exists()
    receipt = json.loads((runtime / "record/smoke-1/execution.json").read_text(encoding="utf-8"))
    assert receipt["launch"] is None
    assert receipt["cleanup_verified"] is True
    if os.name == "posix":
        assert M.V1._group_members(process.pid) == []


def test_owned_worker_gate_eof_exits_without_starting_command(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-run.txt"
    command = f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')"
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", M.OWNED_WORKER_GATE, sys.executable, "-c", command],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait(timeout=10) == 97
    assert not marker.exists()


def test_partial_and_unattested_complete_cache_never_rerun(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, _, _ = _install_harness(tmp_path, monkeypatch)
    job = _job(M.ROOT, runtime)
    stage = runtime / "record/smoke-1"
    stage.mkdir(parents=True)
    _write(stage / "job.json", job)
    assert backend.smoke(job)["reason"] == "render_pending"
    assert launches == []
    _write(stage / "execution.json", {"completed": True})
    with pytest.raises(ValueError, match="attestation"):
        backend.smoke(job)
    assert launches == []


def test_mode_and_original_artifact_tamper_are_rejected(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, _, _ = _install_harness(tmp_path, monkeypatch)
    smoke = _job(M.ROOT, runtime)
    with pytest.raises(ValueError, match="mode"):
        backend.record(smoke)
    backend.smoke(smoke)
    artifact = runtime / "record/smoke-1/output/audit_rgb_2.rgb"
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact"):
        backend.smoke(smoke)
    assert len(launches) == 1


def test_busy_gpu_rejected_before_child_start(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, launches, _, recovery = _install_harness(tmp_path, monkeypatch)
    recovery.read_gpu.busy = True
    job = _job(M.ROOT, runtime)
    with pytest.raises(M.B.BackendError, match="ForeignGPUBusy"):
        backend.smoke(job)
    assert launches == []


def test_capture_references_rejects_hardlinks(tmp_path: Path, monkeypatch) -> None:
    backend, runtime, _, _, _ = _install_harness(tmp_path, monkeypatch)
    job = _job(M.ROOT, runtime)
    backend.smoke(job)
    result = runtime / "record/smoke-1/output/result.json"
    os.link(result, M.ROOT / "result-hardlink.json")
    with pytest.raises(ValueError, match="hardlink"):
        backend.capture_references(job)
