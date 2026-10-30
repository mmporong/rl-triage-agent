from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
ROOT = next(path for path in HERE.parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(CANDIDATE / "bench"), str(CANDIDATE / "evals"),
                str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]
M = importlib.import_module("run_agentic_brev_probe_parent_v10")
ACTUAL = (ROOT / "bench/private/claude_reward_track_20261013/codex_main_20261025/vision_goal"
          / "ri_native_relative_lifecycle_retry_170/posix-parent-v17-r1/final-graph-v1/producer-inputs/binding.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="")


def projection(tmp_path: Path) -> tuple[dict, dict, Path]:
    binding = json.loads(ACTUAL.read_text(encoding="utf-8"))
    home, repo = tmp_path / "home", tmp_path / "home/repo"
    repo.mkdir(parents=True)
    protocol = {"layout": {"repo": "repo", "isaaclab": "lab", "python": "python"},
                "execution_root": ".triage-runtime/fresh-pair-ri170-h-v1/execution"}
    registration = tmp_path / "registration"
    registration.mkdir()
    registration_rows = [row for row in binding["roots"] if row["remote"].endswith("/registration")]
    assert len(registration_rows) == 1
    registration_rows[0]["local"] = str(registration)
    operations = binding["expected_request"]["execution"]["operations"]
    operation_ids = {operations[arm]["operation_id"] for arm in M.V1.ORDER}
    for index, row in enumerate(binding["roots"]):
        operation_id = next((value for value in operation_ids
                             if value in row["remote"] and row["remote"].endswith("/gpu-observations")), None)
        if operation_id is not None:
            row["local"] = str(registration / "gpu-observations" / operation_id)
        elif row is not registration_rows[0]:
            local = tmp_path / "existing" / str(index)
            local.mkdir(parents=True)
            row["local"] = str(local)
    for arm in M.V1.ORDER:
        for index, row in enumerate(operations[arm]["source_roots"]):
            local = tmp_path / "sources" / arm / str(index)
            local.mkdir(parents=True)
            row["local"] = str(local)
    return protocol, binding, home


def capture_proof(tmp_path: Path) -> dict[str, Any]:
    provider, gpu, process = tmp_path / "provider.json", tmp_path / "gpu.csv", tmp_path / "process.csv"
    row: dict[str, Any] = {key: None for key in M.V1.E.GPU.PROVIDER_ROW_KEYS}
    row.update({"id": "instance-1", "name": "fixture", "status": "RUNNING"})
    write(provider, json.dumps({"workspaces": [row]}) + "\n")
    write(gpu, "GPU-fixture\n")
    write(process, "")
    return {"resource_id": "instance-1", "gpu_uuid": "GPU-fixture",
            "provider_inventory_path": str(provider.resolve()), "provider_inventory_sha256": sha(provider),
            "nvidia_gpu_csv_path": str(gpu.resolve()), "nvidia_gpu_csv_sha256": sha(gpu),
            "nvidia_process_csv_path": str(process.resolve()), "nvidia_process_csv_sha256": sha(process)}


def test_safe_recovery_backend_preserves_logical_receipts_until_resolver(tmp_path: Path, monkeypatch) -> None:
    protocol, binding, home = projection(tmp_path)
    mapping = M.ensure_observation_roots(protocol, binding, home=home)
    operation = binding["expected_request"]["execution"]["operations"][M.V1.ORDER[0]]
    value = mapping[operation["operation_id"]]
    capture = capture_proof(tmp_path / "capture")

    def fake_gpu(command, **_kwargs):
        stdout = "GPU-fixture\n" if command == M.V1.E.GPU.GPU_COMMAND else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    native = M.V1.E.GPU.StrictRemoteGpuReader(capture, value["local"], run=fake_gpu)
    reader = M.ReboundReader(native, value["local"], value["remote"])

    class Process:
        pid = os.getpid()
        returncode = 0

        def wait(self, timeout):
            assert timeout == M.V1.U.OWNED.B.TIMEOUTS["probe"]
            return 0

    def popen(args, **_kwargs):
        output = Path(args[args.index("--output") + 1])
        write(output, json.dumps({"completed": True}) + "\n")
        return Process()

    backend = object.__new__(M.V1.U.OWNED.SafeRecoveryBackend)
    backend.lab, backend.python = tmp_path, Path(sys.executable)
    backend.read_gpu, backend.popen = reader, popen
    backend._source_snapshot = lambda: {"source": "fixed"}
    monkeypatch.setattr(M.V1.U.OWNED, "_process_identity", lambda pid: {
        "pid": pid, "pgid": pid, "sid": pid, "start_ticks": 1, "posix_group_verified": True,
    })
    monkeypatch.setattr(M.V1.U.OWNED, "_safe_cleanup", lambda _identity, _process: True)

    stage = tmp_path / "stage"
    worker, receipt = backend._execute(stage, "probe", {"schema": "fixture"})
    assert worker == {"completed": True} and receipt["completed"] is True
    assert not Path(receipt["gpu_before_observation"]["path"]).exists()

    roots = binding["roots"]
    resolver = M.V1.E.Resolver(roots, M.V1.E.digest(roots))
    before = resolver.observation(receipt["gpu_before_observation"], {
        "resource_id": "instance-1", "gpu_uuid": "GPU-fixture",
    })
    after = resolver.observation(receipt["gpu_after_observation"], {
        "resource_id": "instance-1", "gpu_uuid": "GPU-fixture",
    })
    assert (before["ordinal"], after["ordinal"]) == (1, 2)


def test_actual_run_rejects_custom_services_before_dispatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(M, "_inputs", lambda _ref: ({}, {}))
    monkeypatch.setattr(M, "ensure_observation_roots", lambda *_args, **_kwargs: {})
    called = []
    monkeypatch.setattr(M.V1, "run", lambda *_args, **_kwargs: called.append(True))
    try:
        M.run({"path": "unused", "sha256": "0" * 64}, services=object())
    except ValueError as error:
        assert "actual custom services forbidden" in str(error)
    else:
        raise AssertionError("actual custom services가 허용됐다")
    assert called == []


def test_registered_roots_reject_stale_pre_pair_files(tmp_path: Path) -> None:
    protocol, binding, home = projection(tmp_path)
    mapping = M.ensure_observation_roots(protocol, binding, home=home)
    local = Path(next(iter(mapping.values()))["local"])
    write(local / "stale.txt", "stale")
    try:
        M.ensure_observation_roots(protocol, binding, home=home)
    except ValueError as error:
        assert "stale GPU observation root" in str(error)
    else:
        raise AssertionError("pair 시작 전 stale observation을 수용했다")
