from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]

ROOT = next(path for path in Path(__file__).resolve().parents if (path / "AGENTS.md").is_file())
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import agentic_pipeline_preflight_v1 as M  # pyright: ignore[reportMissingImports]

E, V1 = M.E, M.V1
LIFECYCLE = "bench/protocols/agentic_runtime_text_s2221_20261030_v47_lifecycle.json"
V17 = "bench/protocols/agentic_runtime_text_s2221_pair_fresh_20261030_v17.json"
V17_ANCHOR = "21bc54e9d69eae490d508484bee6ac32dfa1193a"
# Windows 260자 경로 제한 안에 두려고 ID·폴더 이름을 줄였다. 실제 host는 POSIX다.
IDS = {"NONE": "a" * 24, "F1": "b" * 24}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict[str, Any]) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"path": str(path), "sha256": sha(path)}


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def fixture(tmp_path: Path, *, drop_observation: bool = False, absent_transport: bool = False,
            registration_inside_unit: bool = False, source_outside: bool = False) -> dict[str, Any]:
    """synthetic_cpu 등록 한 벌. 코드·lifecycle은 실제 저장소 blob을, 실행 배치는 tmp_path를 쓴다."""
    tmp_path = tmp_path.resolve()
    home = tmp_path / "h"
    repo, lab = home / "r", home / "l"
    lab.mkdir(parents=True)
    repo.mkdir()
    (home / "p").write_bytes(b"")
    execution_root = ".t/x"
    units = repo / execution_root / "operations"
    registration = units / IDS["NONE"] / "registration" if registration_inside_unit else tmp_path / "registration"
    if not registration_inside_unit:
        registration.mkdir()
    transport = tmp_path / "transport"
    if not absent_transport:
        transport.mkdir()
    anchor = head()
    backend_root = repo / ".triage-runtime/inputs/backend"
    backend = write(backend_root / "probe-protocol.json", {
        "environment": {"python": "3.10.15"}, "source_sha256_lf": {"bench/brev_recovery_worker_v1.py": "0" * 64},
        "runtime_source_sha256_lf": {name: "1" * 64 for name in (
            "scripts/reinforcement_learning/rsl_rl/train.py", "source/isaaclab/isaaclab/envs/mdp/rewards.py")},
        "python_package_source_sha256_lf": {name: "2" * 64 for name in (
            "rsl_rl/algorithms/ppo.py", "rsl_rl/storage/rollout_storage.py", "rsl_rl/runners/on_policy_runner.py")}})
    roots = [{"remote": "/fixture/transport", "local": str(transport)},
             {"remote": "/fixture/registration", "local": str(registration)},
             {"remote": "/fixture/backend", "local": str(backend_root)}]
    operations = {}
    for arm in V1.ORDER:
        operation_id = IDS[arm]
        if not (drop_observation and arm == "F1"):
            roots.append({"remote": f"/fixture/operations/{operation_id}/gpu-observations",
                          "local": str(registration / "gpu-observations" / operation_id)})
        local = (tmp_path / "old-workspace" if source_outside else repo / ".triage-runtime/inputs") / arm
        local.mkdir(parents=True)
        (local / "model.pt").write_bytes(b"checkpoint-" + arm.encode())
        manifest = write(local / "training.manifest.json", {"arm": arm, "seed": 2221})
        operations[arm] = {"operation_id": operation_id, "arm": arm, "seed": 2221, "probe": "P_reward",
            "run_name": "fixture-" + arm.lower(), "remote_stage": f"/fixture/operations/{operation_id}/raw",
            "source_roots": [{"remote": "/fixture/source/" + arm, "local": str(local)}],
            "source": {"checkpoint": {"path": f"/fixture/source/{arm}/model.pt", "sha256": sha(local / "model.pt")},
                       "manifest": {"path": f"/fixture/source/{arm}/training.manifest.json",
                                    "sha256": manifest["sha256"]}}}
    binding = {"schema": M.PARENT.SCHEMA + "_binding", "evidence_kind": E.FIXTURE, "roots": roots,
               "expected_request": {"budget_gpu_s": 600, "execution": {"pair_id": "c" * 24, "operations": operations,
                   "backend": {"protocol": {"path": "/fixture/backend/probe-protocol.json",
                                            "sha256": backend["sha256"]}, "source_anchor": anchor}}}}
    files = tmp_path / "registration-files"
    binding_ref = write(files / "binding.json", binding)
    protocol = {"schema": M.PARENT.SCHEMA + "_protocol", "registered_before_execution": True,
                "binding_sha256": binding_ref["sha256"], "execution_root": execution_root,
                "layout": {"repo": "r", "isaaclab": "l", "python": "p"},
                "sources_lf": {name: V1.C.digest_lf_file(ROOT / name) for name in M.REMOTE.SOURCES}}
    request = write(files / "request.json", {"schema": M.PARENT.SCHEMA + "_request", "action": "probe",
        "protocol": write(files / "protocol.json", protocol), "binding": binding_ref, "source_anchor": anchor})
    phase = json.loads((ROOT / LIFECYCLE).read_text(encoding="utf-8"))
    contract = write(tmp_path / "contract.json", {"deadline_utc": phase["deadline_utc"], "lifecycle_protocol": {
        "path": LIFECYCLE, "sha256": sha(ROOT / LIFECYCLE)}})
    return {"home": home, "repo": repo, "request": request, "contract": contract, "protocol": protocol,
            "binding": binding, "anchor": anchor, "phase": phase, "tmp": tmp_path}


def run(value: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    deadline = E._stamp(value["phase"]["deadline_utc"])
    arguments = {"request_ref": value["request"], "contract_ref": value["contract"], "runtime_anchor": value["anchor"],
                 "home": value["home"], "allow_fixture": True, "now": deadline - timedelta(hours=1)}
    return M.check(**{**arguments, **overrides})


@pytest.fixture
def registered_root(monkeypatch: pytest.MonkeyPatch):
    """실제 판은 layout repo가 실행 저장소 ROOT와 같아야 한다. fixture는 tmp repo로 그 등식을 만든다."""
    def apply(value: dict[str, Any]) -> dict[str, Any]:
        monkeypatch.setattr(M.OWNED.B, "ROOT", value["repo"])
        return value
    return apply


def test_fixed_fixture_passes_then_claims_and_relooks_up_in_same_scratch(tmp_path: Path, registered_root) -> None:
    value = registered_root(fixture(tmp_path))
    result = run(value)
    assert result["status"] == "PASS" and result["failed_stage"] is None
    assert set(result["scope"].values()) == {"verified"} and result["fixture_scope"] == "synthetic_cpu"
    assert result["checked"]["backend_args"]["validated_jobs"] == list(V1.ORDER)
    assert result["execution_permitted"] is False and result["gpu_start_authorized"] is False

    protocol, binding, home = value["protocol"], value["binding"], value["home"]
    places = M.layout(protocol, binding, home)
    operations = binding["expected_request"]["execution"]["operations"]
    stage_row = {"remote": operations["NONE"]["remote_stage"], "local": str(places["stages"]["NONE"])}
    with pytest.raises(FileNotFoundError):  # 준비 시점에는 미래 stage root를 해석하지 못한다.
        E.Resolver(binding["roots"] + [stage_row], E.digest(binding["roots"] + [stage_row]))
    observations = M.PARENT.ensure_observation_roots(protocol, binding, home=home)
    E.Resolver(binding["roots"], E.digest(binding["roots"]))
    places["folder"].mkdir(parents=True, exist_ok=False)  # Parent1 run의 pair/unit claim과 같은 호출
    for unit in places["units"].values():
        unit.mkdir(parents=True, exist_ok=False)
    for arm in V1.ORDER:
        operation = operations[arm]
        proof = places["folder"] / operation["operation_id"]
        proof.mkdir(exist_ok=False)
        stage = places["stages"][arm]
        stage.mkdir()
        raw = write(stage / "job.json", {"fixture": "synthetic_cpu"})
        receipt = write(Path(observations[operation["operation_id"]]["local"]) / "00001.receipt.json", {"arm": arm})
        roots = operation["source_roots"] + binding["roots"] + [
            {"remote": "/parent/" + operation["operation_id"], "local": str(proof)},
            {"remote": operation["remote_stage"], "local": str(stage)}]
        plan = E.Resolver(roots, E.digest(roots))
        assert plan.file({"path": operation["remote_stage"] + "/job.json", "sha256": raw["sha256"]}) == stage / "job.json"
        remote = observations[operation["operation_id"]]["remote"]
        assert plan.file({"path": remote + "/00001.receipt.json", "sha256": receipt["sha256"]}) == Path(receipt["path"])
        plan.file(operation["source"]["checkpoint"])
    assert M.PARENT.registered_observation_roots(protocol, binding, home=home) == observations


def test_missing_observation_root_map_is_rejected_by_parent10(tmp_path: Path) -> None:
    result = run(fixture(tmp_path, drop_observation=True))
    assert (result["status"], result["failed_stage"]) == ("FAIL", "root_map")
    assert result["reason"] == "registered stable GPU observation root"
    assert result["scope"]["prepare_layout"] == "not_run"


def test_prepare_time_resolver_rejects_root_that_does_not_exist_yet(tmp_path: Path) -> None:
    result = run(fixture(tmp_path, absent_transport=True))
    assert (result["status"], result["failed_stage"], result["error_type"]) == (
        "FAIL", "prepare_layout", "FileNotFoundError")
    assert "transport" in result["reason"]


def test_observation_precreation_collides_with_exclusive_unit_claim(tmp_path: Path) -> None:
    value = fixture(tmp_path, registration_inside_unit=True)
    result = run(value)
    assert (result["failed_stage"], result["error_type"]) == ("prepare_layout", "ValueError")
    assert result["reason"].startswith("prepare-time root inside exclusive claim unit")
    places = M.layout(value["protocol"], value["binding"], value["home"])
    M.PARENT.ensure_observation_roots(value["protocol"], value["binding"], home=value["home"])
    with pytest.raises(FileExistsError):  # Parent1 run의 exist_ok=False unit claim
        places["units"]["NONE"].mkdir(parents=True, exist_ok=False)


def test_absolute_lifecycle_reference_is_rejected_by_service6(tmp_path: Path) -> None:
    value = fixture(tmp_path)
    contract = json.loads(Path(value["contract"]["path"]).read_text(encoding="utf-8"))
    contract["lifecycle_protocol"]["path"] = str(ROOT / LIFECYCLE)
    value["contract"] = write(Path(value["contract"]["path"]).with_name("absolute.json"), contract)
    result = run(value, request_ref=None)
    assert (result["status"], result["failed_stage"]) == ("FAIL", "lifecycle")
    assert result["reason"] == "public lifecycle repo-relative path required"


def test_backend_file_argument_outside_worker_workspace_is_rejected(tmp_path: Path, registered_root) -> None:
    result = run(registered_root(fixture(tmp_path, source_outside=True)))
    assert (result["status"], result["failed_stage"]) == ("FAIL", "backend_args")
    assert result["reason"] == "입력 경로가 허용된 workspace를 벗어났다"


def test_deadline_and_stop_reserve_mismatch_are_rejected_by_service9(tmp_path: Path, registered_root) -> None:
    value = registered_root(fixture(tmp_path))
    deadline = E._stamp(value["phase"]["deadline_utc"])
    reserve = int(value["phase"]["stop_reserve_s"])
    late = run(value, now=deadline - timedelta(seconds=600 + reserve - 1))
    assert (late["failed_stage"], late["reason"]) == ("budget", "deadline must preserve requested work and Stop reserve")
    assert run(value, now=deadline - timedelta(seconds=600 + reserve))["status"] == "PASS"
    contract = json.loads(Path(value["contract"]["path"]).read_text(encoding="utf-8"))
    contract["deadline_utc"] = (deadline + timedelta(hours=1)).isoformat()
    value["contract"] = write(Path(value["contract"]["path"]).with_name("later.json"), contract)
    moved = run(value)
    assert (moved["failed_stage"], moved["reason"]) == ("budget", "contract/lifecycle deadline differs")


def test_fixture_requires_explicit_flag_and_unknown_schema_is_unsupported(tmp_path: Path) -> None:
    value = fixture(tmp_path)
    assert run(value, allow_fixture=False)["reason"] == "fixture mode must be explicit"
    request = json.loads(Path(value["request"]["path"]).read_text(encoding="utf-8"))
    request["schema"] = "agentic_brev_probe_pair_parent_v99_request"
    value["request"] = write(Path(value["request"]["path"]).with_name("future.json"), request)
    result = run(value, contract_ref=None)
    assert result["status"] == "UNSUPPORTED" and result["failed_stage"] is None
    assert result["unsupported"]["registration"].startswith("unknown parent request schema")


def test_registered_v17_protocol_lacks_remote10_entry_closure() -> None:
    path = ROOT / V17
    result = M.check(protocol_ref={"path": str(path), "sha256": sha(path)}, source_anchor=V17_ANCHOR)
    assert (result["status"], result["failed_stage"]) == ("FAIL", "registration")
    assert result["reason"].startswith("registered source closure lacks 6 Remote10 entry sources")


def test_check_cli_returns_short_failure_json(tmp_path: Path) -> None:
    value = fixture(tmp_path, drop_observation=True)
    completed = subprocess.run([sys.executable, str(ROOT / M.SOURCE), "--check", "--allow-fixture",
        "--request", value["request"]["path"], "--request-sha256", value["request"]["sha256"],
        "--home", str(value["home"])], capture_output=True, text=True, encoding="utf-8", check=False)
    assert completed.returncode == 1, completed.stderr
    result = json.loads(completed.stdout)
    assert {"status", "failed_stage", "inputs", "scope", "next"} <= result.keys()
    assert (result["failed_stage"], result["inputs"]["request"]) == ("root_map", value["request"]["path"])
    assert result["next"] == M.NEXT["root_map"] and result["fixture_scope"] == "synthetic_cpu"
