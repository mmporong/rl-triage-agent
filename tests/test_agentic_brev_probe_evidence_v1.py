"""등록 POSIX 원문을 쓰는 CPU fixture 회귀. GPU/API/네트워크 호출은 없다."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src"), str(ROOT / "tests")]
import agentic_brev_probe_evidence_v1 as E
from test_brev_recovery_backend_v4 import _optimizer_probe_result
from test_optimizer_observability import observed
from test_selection_comparison_controller_v1 import measurement


def fixture(root: Path, probe: str = "P_episode", *, kind: str = E.FIXTURE) -> tuple[dict, list]:
    roots = [{"remote": "/" + name, "local": str((root / name).resolve())} for name in ("stage", "source", "control")]
    for row in roots:
        Path(row["local"]).mkdir(parents=True)

    def local(logical: str) -> Path:
        return root / logical.removeprefix("/")

    def blob(logical: str, raw: bytes) -> dict:
        path = local(logical)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"path": logical, "sha256": E.J.file_sha(path)}

    def save(logical: str, value: dict) -> dict:
        return blob(logical, E.J._bytes(value))

    protocol = save("/control/bench/protocols/probe.json", {"schema": "cpu_protocol_fixture"})
    execution_protocol = save("/control/execution.json", {"schema": "cpu_execution_fixture"})
    checkpoint = blob("/source/model_299.pt", b"cpu-source-checkpoint")
    telemetry = save("/source/telemetry.json", {"summary": {"reward": 0.5}})
    hashes = {name: "d" * 64 for name in E.B.RESIDUAL.MINIMUM_BOUND_SOURCES}
    common = {"source_commit_sha": "a" * 40, "protocol_path": "bench/protocols/probe.json",
        "protocol_sha256_lf": protocol["sha256"], "source_sha256_lf": hashes,
        "runtime_source_sha256_lf": {"scripts/train.py": "b" * 64}}
    if probe == E.B.OPTIMIZER_PROBE:
        source, _ = _optimizer_probe_result()
        training = {**source["manifest"], **common, "seed": 2221,
            "records_path": "/source/train.records.jsonl", "readback_path": "/source/train.readback.json"}
    else:
        training = {"seed": 2221, "arm": "F1"}
    manifest = save("/source/training.manifest.json", training)
    artifact = save("/source/adapter_artifact.json", {"arm": "F1", "seed": 2221,
        "checkpoint": checkpoint, "telemetry": telemetry, "manifest_path": manifest["path"]})
    job = {"schema": "brev_recovery_worker_job_v1", "mode": "probe", "stage": "/stage", "lab": "/lab",
        "repo": "/control", **common, "protocol_path": protocol["path"],
        "python_package_source_sha256_lf": {"rsl_rl/ppo.py": "c" * 64}, "arm": "F1", "seed": 2221,
        "run_name": "registered_probe", "probe": probe, "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"], "report_path": "/stage/report.json",
        "training_manifest_path": manifest["path"], "inspection_manifest_path": "/stage/inspection.manifest.json",
        "records_path": "/stage/records.jsonl", "readback_path": "/stage/readback.json"}
    environment = {"isaaclab_sha": "e" * 40, "python": "3.10.20", "numpy": "fixture",
                   "cuda_available": True, "cuda_device_count": 1}
    expected_source = {"lab_head": environment["isaaclab_sha"], "lab_tracked_clean": True,
        "source_anchor": job["source_commit_sha"], "protocol": {"path": common["protocol_path"], "sha256_lf": protocol["sha256"]},
        "environment": environment, "selected_interpreter": {"python": "3.10.20", "packages": {"numpy": "fixture"}},
        **{key: job[key] for key in E.HASHES}}
    capture = {"resource_id": "fixture-vm", "gpu_uuid": "GPU-fixture"}
    inventory = {"workspaces": [{key: ("fixture-vm" if key == "id" else "RUNNING" if key == "status" else "fixture")
                                  for key in E.GPU.PROVIDER_ROW_KEYS}]}
    for prefix, reference in (
        ("provider_inventory", save("/control/provider.json", inventory)),
        ("nvidia_gpu_csv", blob("/control/gpu.csv", b"GPU-fixture\n")),
        ("nvidia_process_csv", blob("/control/process.csv", b""))):
        capture.update({prefix + "_path": reference["path"], prefix + "_sha256": reference["sha256"]})
    proof = save("/control/resource.json", {"schema": "brev_resource_proof_v1", "status": "granted", **capture,
        "lease_id": "registered-fixture-lease", "protocol_sha256": execution_protocol["sha256"],
        "issued_at_utc": "2026-10-28T00:00:00Z", "expires_at_utc": "2026-10-28T01:00:00Z"})
    observations = []
    for ordinal in (1, 2):
        gpu = blob(f"/control/observations/{ordinal:06d}.gpu.csv", b"GPU-fixture\n")
        processes = blob(f"/control/observations/{ordinal:06d}.process.csv", b"")
        observations.append(save(f"/control/observations/{ordinal:06d}.receipt.json", {
            "schema": "vision_remote_gpu_observation_v1", "ordinal": ordinal,
            "resource_id": "fixture-vm", "gpu_uuid": "GPU-fixture", "gpu_command": E.GPU.GPU_COMMAND,
            "process_command": E.GPU.PROCESS_COMMAND, "gpu_exit_code": 0, "process_exit_code": 0,
            "gpu_stdout_file": Path(gpu["path"]).name, "gpu_stdout_sha256": gpu["sha256"], "gpu_stderr": "",
            "process_stdout_file": Path(processes["path"]).name, "process_stdout_sha256": processes["sha256"], "process_stderr": ""}))
    execution = {"schema": "brev_recovery_posix_execution_v1", "mode": "probe", "completed": True,
        "started_at": "2026-10-28T00:00:10Z", "finished_at": "2026-10-28T00:00:20Z", "exit_code": 0,
        "timed_out": False, "cleanup_verified": True,
        "launch": {"pid": 123, "pgid": 123, "sid": 123, "start_ticks": 98765, "posix_group_verified": True},
        "gpu_before": [], "gpu_after": [], "wall_s": 10.0,
        "source_before": expected_source, "source_after": deepcopy(expected_source),
        "gpu_before_observation": observations[0], "gpu_after_observation": observations[1]}
    files: dict = {}
    if probe == E.B.OPTIMIZER_PROBE:
        inspection = {**training, "mode": "inspection", "max_iterations": 3, "run_name": job["run_name"],
            "source_checkpoint_sha256": checkpoint["sha256"], "records_path": job["records_path"], "readback_path": job["readback_path"]}
        descriptor = save(job["inspection_manifest_path"], inspection)
        observer_manifest = E.B.RESIDUAL.observer_manifest(inspection)
        ppo, rows, observer = observed(epochs=5, bound=observer_manifest)
        ppo.num_mini_batches = 4
        try:
            for _ in range(3):
                ppo.update()
        finally:
            observer.remove()
        files["records"] = blob(job["records_path"], b"".join(E.J._bytes(row) for row in rows))
        files["inspection"] = descriptor
        files["readback"] = save(job["readback_path"], {"fixture": "source readback"})
        report = E.B.OPTIMIZER.assemble_inspection(rows, observer_manifest)
        measured = deepcopy(report)
    else:
        descriptor = None
        measured = measurement(probe, "normal")
        report = {"name": job["run_name"], "checkpoint": {"file": "model_299.pt", "sha256": checkpoint["sha256"]},
            "task": E.B.TASK, "seed": 2221, "num_envs": 4096,
            "probes": {probe: {"exit": 0, "measurement": measured}}}
        files["report_jobs"] = save("/stage/report.jobs.json", {"fault": "F1", "seed": 2221, "task": E.B.TASK,
            "jobs": [{"name": job["run_name"], "checkpoint": checkpoint["path"], "probes": [probe], "output": job["report_path"]}]})
    files["report"] = save(job["report_path"], report)
    worker = {"schema": "brev_recovery_probe_worker_v1", "completed": True, "probe": probe, "measurement": measured,
        "report": files["report"], "inspection_manifest": descriptor, "wall_s": 9.0}
    backend = {"execution": execution, "report": report, "measurement": measured}
    if descriptor:
        backend["inspection_manifest"] = {**descriptor, "body": inspection}
    files.update(job=save("/stage/job.json", job), worker=save("/stage/worker.json", worker),
        execution=save("/stage/execution.json", execution), backend_result=save("/stage/backend_result.json", backend))
    request = {"schema": E.REQUEST_SCHEMA, "evidence_kind": kind, "action": "probe", "operation_id": "1" * 64,
        "probe": probe, "side": "case", "arm": "F1", "seed": 2221, "run_name": job["run_name"],
        "stage": "/stage", "max_wall_s": 300, "expected_source": expected_source,
        "source": {"artifact": artifact, "checkpoint": checkpoint, "telemetry": telemetry,
                   "manifest": manifest, "lineage_id": "registered-source-s2221"},
        "resource": {"proof": proof, "protocol": execution_protocol, "resource_id": "fixture-vm", "gpu_uuid": "GPU-fixture",
            "lease_id": "registered-fixture-lease", "protocol_sha256": execution_protocol["sha256"], "capture": capture},
        "files": files}
    return request, roots


def build(request, roots, *, allow_fixture=True):
    return E.build(request, roots, expected_request_sha256=E.digest(request),
                   expected_roots_sha256=E.digest(roots), allow_fixture=allow_fixture)


def rewrite(root, request, name, mutate):
    reference = request["files"][name]
    path = root / reference["path"].removeprefix("/")
    value = E.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(E.J._bytes(value))
    reference["sha256"] = E.J.file_sha(path)


@pytest.mark.parametrize("probe", E.PROBES)
def test_all_registered_probes_recompute_without_raw_rewrite(tmp_path, probe):
    request, roots = fixture(tmp_path, probe)
    before = {str(p): E.J.file_sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    evidence = build(request, roots)
    assert E.verify(evidence, roots, expected_request_sha256=E.digest(request),
        expected_roots_sha256=E.digest(roots), allow_fixture=True) == evidence
    assert evidence["probe"] == probe and evidence["gpu_contract"] == "brev_posix_v1"
    assert evidence["checkpoint_sha256"] == request["source"]["checkpoint"]["sha256"]
    assert evidence["seed"] == 2221 and not evidence["approval_granted"] and not evidence["repair_supported"]
    assert before == {str(p): E.J.file_sha(p) for p in tmp_path.rglob("*") if p.is_file()}


def test_fixture_is_explicit_and_actual_is_default_guard(tmp_path):
    request, roots = fixture(tmp_path)
    with pytest.raises(ValueError, match="fixture requires explicit"):
        E.build(request, roots, expected_request_sha256=E.digest(request), expected_roots_sha256=E.digest(roots))
    request, roots = fixture(tmp_path / "actual-schema", kind=E.ACTUAL)
    assert build(request, roots, allow_fixture=False)["evidence_kind"] == E.ACTUAL  # CPU schema fixture only
    assert E.action_support("repair") == {"status": "HOLD", "reason": "repair_unsupported",
                                         "execution_permitted": False, "approval_granted": False}


@pytest.mark.parametrize("field,value", [("seed", 2222), ("probe", "P_reward"), ("checkpoint_sha256", "9" * 64),
                                        ("run_name", "another_run"), ("stage", "/different")])
def test_wrong_checkpoint_seed_probe_and_run_are_rejected(tmp_path, field, value):
    request, roots = fixture(tmp_path)
    rewrite(tmp_path, request, "job", lambda row: row.update({field: value}))
    with pytest.raises((ValueError, FileNotFoundError)):
        build(request, roots)


@pytest.mark.parametrize("mutate", [lambda row: row.update(completed=False),
    lambda row: row.update(cleanup_verified=False), lambda row: row.update(timed_out=True),
    lambda row: row.update(exit_code=1), lambda row: row["launch"].update(start_ticks=None),
    lambda row: row["launch"].update(pgid=456), lambda row: row["source_after"].update(source_anchor="f" * 40),
    lambda row: row.update(gpu_after=[99]), lambda row: row.pop("gpu_before_observation"),
    lambda row: row.update(gpu_after_observation=row["gpu_before_observation"])])
def test_incomplete_unowned_cleanup_source_gpu_evidence_is_rejected(tmp_path, mutate):
    request, roots = fixture(tmp_path)
    rewrite(tmp_path, request, "execution", mutate)
    with pytest.raises((ValueError, KeyError)):
        build(request, roots)


def test_raw_mutation_missing_duplicate_and_measurement_tampering(tmp_path):
    request, roots = fixture(tmp_path)
    evidence = build(request, roots)
    evidence["measurement"]["timeout_ratio"] = 0.1
    with pytest.raises(ValueError, match="recomputation"):
        E.verify(evidence, roots, expected_request_sha256=E.digest(request),
                 expected_roots_sha256=E.digest(roots), allow_fixture=True)
    path = tmp_path / "stage/report.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        build(request, roots)
    path.unlink()
    with pytest.raises(ValueError, match="regular"):
        build(request, roots)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "duplicate_json", "duplicate_jobs"])
def test_raw_inventory_and_duplicate_json_or_jobs_are_rejected(tmp_path, mutation):
    request, roots = fixture(tmp_path)
    if mutation == "missing":
        request["files"].pop("worker")
    elif mutation == "duplicate":
        request["files"]["worker"] = request["files"]["job"]
    elif mutation == "duplicate_json":
        path = tmp_path / "stage/worker.json"
        path.write_text('{"completed":true,"completed":true}', encoding="utf-8")
        request["files"]["worker"]["sha256"] = E.J.file_sha(path)
    else:
        rewrite(tmp_path, request, "report_jobs", lambda row: row["jobs"].append(row["jobs"][0]))
    with pytest.raises(ValueError):
        build(request, roots)


@pytest.mark.parametrize("field,value", [("status", "pending"), ("lease_id", "other-lease"),
    ("expires_at_utc", "2026-10-28T00:00:15Z"), ("gpu_uuid", "GPU-other")])
def test_lease_and_resource_binding_cannot_be_replaced(tmp_path, field, value):
    request, roots = fixture(tmp_path)
    reference = request["resource"]["proof"]
    path = tmp_path / "control/resource.json"
    row = E.loads(path.read_text(encoding="utf-8"))
    row[field] = value
    path.write_bytes(E.J._bytes(row))
    reference["sha256"] = E.J.file_sha(path)
    with pytest.raises(ValueError, match="resource/lease"):
        build(request, roots)


def test_registered_root_map_only_no_fallback_or_path_traversal(tmp_path):
    request, roots = fixture(tmp_path)
    with pytest.raises(ValueError, match="root-map SHA"):
        E.build(request, roots, expected_request_sha256=E.digest(request), expected_roots_sha256="0" * 64, allow_fixture=True)
    roots.append({"remote": "/stage/nested", "local": roots[0]["local"]})
    with pytest.raises(ValueError, match="overlapping"):
        build(request, roots)
    roots.pop()
    request["files"]["worker"]["path"] = "/stage/../source/worker.json"
    with pytest.raises(ValueError, match="POSIX"):
        build(request, roots)


def test_optimizer_incomplete_or_duplicate_updates_fail(tmp_path):
    request, roots = fixture(tmp_path, E.B.OPTIMIZER_PROBE)
    path = tmp_path / "stage/records.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([rows[0], rows[0], rows[2]]) + "\n", encoding="utf-8")
    request["files"]["records"]["sha256"] = E.J.file_sha(path)
    with pytest.raises(ValueError, match="duplicate optimizer"):
        build(request, roots)


def test_existing_dotnet_lease_timestamp_parser_is_reused(tmp_path):
    request, roots = fixture(tmp_path)
    path = tmp_path / "control/resource.json"
    row = E.loads(path.read_text(encoding="utf-8"))
    row["issued_at_utc"] = "2026-10-28T00:00:00.1234567Z"
    row["expires_at_utc"] = "2026-10-28T01:00:00.1234567+00:00"
    path.write_bytes(E.J._bytes(row))
    request["resource"]["proof"]["sha256"] = E.J.file_sha(path)
    assert build(request, roots)["measurement"] == measurement("P_episode", "normal")
