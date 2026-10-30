"""lineage source probe backend의 CPU fixture 회귀."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench"), str(ROOT / "src")]
import lineage_probe_backend_v1 as M


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def protocol() -> dict:
    paths = {name: "1" * 64 for name in {
        "evals/lineage_probe_backend_v1.py", "evals/probes.py", "bench/hidden_faults.py",
        *M.RESIDUAL.MINIMUM_BOUND_SOURCES,
    }}
    return {"schema": M.PROTOCOL_SCHEMA,
            "source_sha256_lf": paths, "runtime_source_sha256_lf": {"runtime.py": "2" * 64},
            "watchdogs_s": {"legacy": 600, "optimizer": 900},
            "isaac_python_relative": "_isaac_sim/python.bat", "scope": "fixture"}


def source(tmp_path: Path, arm: str, seed: int = 2301) -> tuple[dict, dict, dict]:
    checkpoint = tmp_path / f"{arm}.pt"
    checkpoint.write_bytes(arm.encode())
    manifest_path = save(tmp_path / f"{arm}.manifest.json", {"fixture": arm})
    collected_path = save(tmp_path / f"{arm}.collected.json", {"fixture": arm})
    collected = {"private_source_map": {"manifest": str(manifest_path)},
                 "source_binding": {"checkpoint_sha256": M.sha(checkpoint),
                                    "telemetry_sha256": hashlib_sha(f"telemetry-{arm}")}}
    manifest = {"purpose": "source", "arm": arm, "seed": seed, "job_id": f"{arm.lower()}-s{seed}"}
    evidence = {"checkpoint": {"path": str(checkpoint.resolve()), "sha256": M.sha(checkpoint)}}
    return {"path": collected_path, "value": collected}, manifest, evidence


def hashlib_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def prepared(tmp_path: Path, arm: str, probe: str, *, role: str = "case") -> tuple[dict, Path]:
    source_root = tmp_path / f"source-{role}-{arm}-{probe}"
    source_root.mkdir()
    collected, manifest, evidence = source(source_root, arm)
    if probe == "P_value":
        Path(evidence["checkpoint"]["path"]).with_name(
            M.C.RC.registry()["probes"][probe]["args"]["from"]
        ).write_bytes(f"first-{role}-{arm}".encode())
    lab = tmp_path / "lab"
    python = lab / "_isaac_sim/python.bat"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    stage = tmp_path / f"stage-{role}-{arm}-{probe}"
    with patch.object(M, "verify_protocol", return_value=protocol()), \
            patch.object(M, "_source", return_value=(collected["value"], manifest, evidence)):
        plan = M.prepare_probe(collected["path"], probe, stage=stage, lab=lab,
                               protocol_path=Path(M.__file__), anchor="a" * 40, role=role)
    return plan, stage / "plan.json"


def authorization(tmp_path: Path, plan: dict, plan_path: Path, *, state: str = "running",
                  checkpoint: str | None = None) -> Path:
    ledger_path = tmp_path / f"ledger-{plan['source_arm']}-{plan['probe']}-{state}.jsonl"
    packet = M.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={
            "case": {"checkpoint_sha256": checkpoint or plan["checkpoint"]["sha256"],
                     "telemetry_sha256": plan["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": "3" * 64, "telemetry_sha256": "4" * 64},
        },
        execution_budget_s=1200,
    )
    snapshot = M.C.V.build_snapshot(packet, "lineage_backend_fixture", "registry_next",
                                    M.C.V.file_digest(M.C.V.PROTOCOL), ledger_path)
    choice = M.C.choose(snapshot, "registry_next")
    if choice["probe"] != plan["probe"]:
        choice = {**choice, "probe": plan["probe"]}
        choice["action"] = "continue"
        choice["selected_by"] = "registered_rule"
        choice["reason"] = "registered_policy"
    spec = M.C.request_spec(snapshot, choice)
    ledger = M.C.S.M.RegistryLedger(ledger_path, M.C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], int(spec["budget_gpu_s"]))
    if state in {"approved", "running", "done"}:
        ledger.approve(request["request_id"], "fixture-owner")
    if state in {"running", "done"}:
        ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    if state == "done":
        ledger.receipt(request["request_id"], "unknown", {}, 1.0, 1)
    path = tmp_path / f"authorization-{arm_key(plan)}-{state}.json"
    save(path, {"schema": M.AUTH_SCHEMA, "plan_sha256": M.sha(plan_path), "snapshot": snapshot,
                "choice": choice, "request_spec": spec, "ledger_path": str(ledger_path),
                "request_id": request["request_id"]})
    return path


def paired_authorizations(tmp_path: Path, case: tuple[dict, Path], reference: tuple[dict, Path]) -> Path:
    case_plan, case_path = case
    reference_plan, reference_path = reference
    ledger_path = tmp_path / "paired/probes.ledger.jsonl"
    packet = M.C.S.build_packet(
        epoch=1, applied_corrections=[], prior_receipt=None,
        facts={"telemetry:case:Train/mean_reward": 1.0},
        input_bindings={
            "case": {"checkpoint_sha256": case_plan["checkpoint"]["sha256"],
                     "telemetry_sha256": case_plan["telemetry_sha256"]},
            "reference": {"checkpoint_sha256": reference_plan["checkpoint"]["sha256"],
                          "telemetry_sha256": reference_plan["telemetry_sha256"]},
        }, execution_budget_s=1200)
    snapshot = M.C.V.build_snapshot(packet, "paired_fixture", "registry_next",
                                    M.C.V.file_digest(M.C.V.PROTOCOL), ledger_path)
    choice = M.C.choose(snapshot, "registry_next")
    if choice["probe"] != case_plan["probe"]:
        choice = {**choice, "probe": case_plan["probe"], "action": "continue",
                  "selected_by": "registered_rule", "reason": "registered_policy"}
    spec = M.C.request_spec(snapshot, choice)
    ledger = M.C.S.M.RegistryLedger(ledger_path, M.C.RC.registry())
    request = ledger.propose(spec["prereg"], spec["probe"], int(spec["budget_gpu_s"]))
    ledger.approve(request["request_id"], "paired-owner")
    ledger.consume(request["request_id"], spec["prereg"], spec["probe"])
    common = {"schema": M.AUTH_SCHEMA, "snapshot": snapshot, "choice": choice,
              "request_spec": spec, "ledger_path": str(ledger_path), "request_id": request["request_id"]}
    return save(tmp_path / "paired/authorization.json", {
        **common, "plan_sha256": {"case": M.sha(case_path), "reference": M.sha(reference_path)}})


def arm_key(plan: dict) -> str:
    return f"{plan['source_arm']}-{plan['probe']}"


def execution() -> dict:
    return {"exit_code": 0, "natural_exit": True, "timed_out": False, "error_type": None,
            "cleanup_verified": True, "job_assigned_before_release": True, "wall_s": 1.0}


def legacy_measurement(probe: str) -> dict:
    return {"probe_value": 1.0, "probe": probe}


def legacy_owned(plan: dict, *, probe_exit: int = 0):
    def run(folder, argv, cwd, timeout):
        assert argv == plan["command"] and cwd == Path(plan["lab"]) and timeout == plan["watchdog_s"]
        assert folder.is_dir()
        measurement = legacy_measurement(plan["probe"])
        save(Path(plan["raw_files"]["report"]), {
            "name": plan["run_name"], "task": M.FROZEN.TASK, "seed": plan["seed"],
            "checkpoint": {"file": Path(plan["checkpoint"]["path"]).name,
                           "sha256": plan["checkpoint"]["sha256"]},
            "probes": {plan["probe"]: {"measurement": measurement if probe_exit == 0 else None,
                                         "exit": probe_exit, "elapsed_s": 1.0}},
        })
        return execution()
    return run


def runtime_checks():
    return (lambda: {"leaseId": "lease-fixture"}), (lambda _stage: {"accepted": True})


def test_protocol_registration_anchor_has_no_self_commit_cycle(tmp_path):
    repo, lab = tmp_path / "repo", tmp_path / "lab"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    required = {"evals/lineage_probe_backend_v1.py", "evals/probes.py", "bench/hidden_faults.py",
                *M.RESIDUAL.MINIMUM_BOUND_SOURCES}
    for index, relative in enumerate(sorted(required)):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--", *sorted(required)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "code A"], check=True)
    code_anchor = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                 capture_output=True, text=True).stdout.strip()
    runtime = lab / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("runtime\n", encoding="utf-8")
    python = lab / "_isaac_sim/python.bat"
    python.parent.mkdir()
    python.touch()
    protocol_path = repo / "bench/protocols/lineage_probe.json"
    body = {"schema": M.PROTOCOL_SCHEMA,
            "source_sha256_lf": {name: M.sha(repo / name, lf=True) for name in required},
            "runtime_source_sha256_lf": {"runtime.py": M.sha(runtime, lf=True)},
            "watchdogs_s": {"legacy": 600, "optimizer": 900},
            "isaac_python_relative": "_isaac_sim/python.bat", "scope": "scratch registration"}
    save(protocol_path, body)
    subprocess.run(["git", "-C", str(repo), "add", "--", "bench/protocols/lineage_probe.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "protocol B"], check=True)
    registration = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    assert registration != code_anchor and "source_commit_sha" not in body
    with patch.object(M, "ROOT", repo):
        assert M.verify_protocol(protocol_path, lab, registration) == body


@pytest.mark.parametrize("arm,expected", [
    ("NONE", "NONE"), ("F1", "F1"), ("F2", "F2"), ("F3", "F3"),
    ("F5", "F5"), ("COMPOUND", "F1"),
])
def test_legacy_prepare_maps_all_source_faults_to_frozen_probe(tmp_path, arm, expected):
    plan, _path = prepared(tmp_path, arm, "P_reward")
    assert plan["projection"]["fault"] == expected
    assert plan["projection"]["compound_projection"] == ("F1_only" if arm == "COMPOUND" else None)
    assert read_jobs(plan)["jobs"][0]["probes"] == ["P_reward"]
    assert plan["scope"]["approval_created"] is False
    assert plan["auxiliary_inputs"] == []


def test_p_value_binds_frozen_from_checkpoint_path_sha_and_size(tmp_path):
    plan, _path = prepared(tmp_path, "F3", "P_value")
    frozen_from = M.C.RC.registry()["probes"]["P_value"]["args"]["from"]
    auxiliary = plan["auxiliary_inputs"]
    assert frozen_from == "model_0.pt" and len(auxiliary) == 1
    assert auxiliary[0] == {
        "role": "P_value:from",
        "path": str(Path(plan["checkpoint"]["path"]).with_name(frozen_from).resolve()),
        "sha256": M.sha(Path(auxiliary[0]["path"])),
        "size_bytes": Path(auxiliary[0]["path"]).stat().st_size,
    }


@pytest.mark.parametrize("from_value", ["F3.pt", "../other/model_0.pt"])
def test_p_value_rejects_final_checkpoint_alias_or_other_directory(tmp_path, from_value):
    collected, manifest, evidence = source(tmp_path, "F3")
    other = tmp_path.parent / "other/model_0.pt"
    other.parent.mkdir(exist_ok=True)
    other.write_bytes(b"other")
    registry = deepcopy(M.C.RC.registry())
    registry["probes"]["P_value"]["args"]["from"] = from_value
    lab = tmp_path / "lab"
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").touch()
    with patch.object(M.C.RC, "registry", return_value=registry), \
            patch.object(M, "verify_protocol", return_value=protocol()), \
            patch.object(M, "_source", return_value=(collected["value"], manifest, evidence)), \
            pytest.raises(ValueError, match="같은 디렉터리|최종 checkpoint"):
        M.prepare_probe(collected["path"], "P_value", stage=tmp_path / "stage", lab=lab,
                        protocol_path=Path(M.__file__), anchor="a" * 40)


def test_normal_reference_source_is_a_separate_none_process_plan(tmp_path):
    normal, _ = prepared(tmp_path, "NONE", "P_noise")
    case, _ = prepared(tmp_path, "F1", "P_noise")
    normal_jobs, case_jobs = read_jobs(normal), read_jobs(case)
    assert normal_jobs["fault"] == "NONE" and case_jobs["fault"] == "F1"
    assert len(normal_jobs["jobs"]) == len(case_jobs["jobs"]) == 1
    assert normal["stage"] != case["stage"] and normal["command"] != case["command"]
    assert set(M.PROBES) == set(M.C.RC.registry()["probes"])


def read_jobs(plan: dict) -> dict:
    return M.read(Path(plan["raw_files"]["jobs"]))


@pytest.mark.parametrize("arm,inspection_arm,faults,reason", [
    ("NONE", "NONE", [], "source_fault_family"),
    ("F1", "F1", ["F1"], "source_fault_family"),
    ("F2", "NONE", [], "count_code_only_NONE"),
    ("F3", "NONE", [], "count_code_only_NONE"),
    ("F5", "NONE", [], "count_code_only_NONE"),
    ("COMPOUND", "COMPOUND", ["F1", "R1"], "source_fault_family"),
])
def test_optimizer_prepare_projection_and_checkpoint_load_false(tmp_path, arm, inspection_arm, faults, reason):
    plan, _path = prepared(tmp_path, arm, M.OPTIMIZER_PROBE)
    assert plan["projection"] == {"backend": "optimizer_observability_3_update", "source_arm": arm,
        "inspection_arm": inspection_arm, "original_faults": faults, "checkpoint_loaded": False,
        "projection_reason": reason}
    manifest = M.RESIDUAL.validate_manifest(M.read(Path(plan["raw_files"]["manifest"])))
    assert manifest["source_checkpoint_sha256"] == plan["checkpoint"]["sha256"]
    assert manifest["max_iterations"] == 3


def test_legacy_execute_validates_running_authorization_and_seals_raw_report(tmp_path):
    plan, plan_path = prepared(tmp_path, "F2", "P_reward")
    auth = authorization(tmp_path, plan, plan_path)
    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=plan):
        receipt = M.execute_probe(plan_path, auth, owned=legacy_owned(plan), check_lease=lease, check_gpu=gpu)
    assert receipt["schema"] == M.RECEIPT_SCHEMA and receipt["status"] == "complete"
    assert receipt["measurement"] == legacy_measurement("P_reward")
    assert receipt["scope"]["ledger_receipt_created"] is False
    assert set(receipt["raw_refs"]) == {"jobs", "report", "owned_result"}
    with patch.object(M, "_verify_plan", return_value=plan), pytest.raises(ValueError, match="cache"):
        M.execute_probe(plan_path, auth, owned=legacy_owned(plan), check_lease=lease, check_gpu=gpu)


@pytest.mark.parametrize("mutation", ["delete", "replace"])
def test_p_value_auxiliary_checkpoint_is_rechecked_before_owned_call(tmp_path, mutation):
    plan, plan_path = prepared(tmp_path, "F3", "P_value")
    auth = authorization(tmp_path, plan, plan_path)
    auxiliary = Path(plan["auxiliary_inputs"][0]["path"])
    if mutation == "delete":
        auxiliary.unlink()
    else:
        auxiliary.write_bytes(b"replaced-before-owned")
    called = False

    def owned(*_args):
        nonlocal called
        called = True
        return execution()

    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=plan), \
            pytest.raises(ValueError, match="보조 checkpoint|보조 입력 결합"):
        M.execute_probe(plan_path, auth, owned=owned, check_lease=lease, check_gpu=gpu)
    assert called is False


def test_p_value_mid_execution_auxiliary_mutation_is_sealed_unknown(tmp_path):
    plan, plan_path = prepared(tmp_path, "F5", "P_value")
    auth = authorization(tmp_path, plan, plan_path)
    auxiliary = Path(plan["auxiliary_inputs"][0]["path"])

    def owned(folder, argv, cwd, timeout):
        result = legacy_owned(plan)(folder, argv, cwd, timeout)
        auxiliary.write_bytes(b"mutated-during-owned")
        return result

    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=plan), \
            pytest.raises(ValueError, match="보조 입력 결합"):
        M.execute_probe(plan_path, auth, owned=owned, check_lease=lease, check_gpu=gpu)
    receipt = M.read(Path(plan["stage"]) / "probe_receipt.json")
    assert receipt["status"] == "unknown" and receipt["measurement"] is None
    assert receipt["error_type"] == "ValueError"
    assert receipt["source_refs"]["auxiliary_inputs"] == plan["auxiliary_inputs"]


@pytest.mark.parametrize("state", ["approved", "done"])
def test_unconsumed_or_completed_authorization_is_rejected(tmp_path, state):
    plan, plan_path = prepared(tmp_path, "F3", "P_value")
    auth = authorization(tmp_path, plan, plan_path, state=state)
    with pytest.raises(ValueError):
        M._authorization(auth, plan_path, plan)


def test_plan_tamper_and_wrong_checkpoint_authorization_are_rejected(tmp_path):
    plan, plan_path = prepared(tmp_path, "F5", "P_episode")
    auth = authorization(tmp_path, plan, plan_path)
    changed = deepcopy(plan)
    changed["watchdog_s"] += 1
    save(plan_path, changed)
    with pytest.raises(ValueError, match="plan SHA"):
        M._authorization(auth, plan_path, changed)

    other, other_path = prepared(tmp_path, "F1", "P_noise")
    wrong_auth = authorization(tmp_path, other, other_path, checkpoint="f" * 64)
    with pytest.raises(ValueError, match="checkpoint"):
        M._authorization(wrong_auth, other_path, other)


def test_false_owned_exit_zero_with_failed_probe_stays_unknown(tmp_path):
    plan, plan_path = prepared(tmp_path, "COMPOUND", "P_physics")
    auth = authorization(tmp_path, plan, plan_path)
    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=plan), pytest.raises((ValueError, TypeError)):
        M.execute_probe(plan_path, auth, owned=legacy_owned(plan, probe_exit=1), check_lease=lease, check_gpu=gpu)
    receipt = M.read(Path(plan["stage"]) / "probe_receipt.json")
    assert receipt["status"] == "unknown" and receipt["measurement"] is None
    assert receipt["execution"]["exit_code"] == 0 and receipt["error_type"] in {"ValueError", "TypeError"}


def optimizer_rows(manifest: dict) -> list[dict]:
    observer = M.RESIDUAL.observer_manifest(manifest)
    rows = []
    for index in range(3):
        rows.append({"schema": M.OPTIMIZER.SCHEMA,
            "manifest_sha256": M.OPTIMIZER.manifest_fingerprint(observer),
            "binding": M.OPTIMIZER.manifest_binding(observer), "update_index": index,
            "expected": {"optimizer_steps": 20, "minibatches": 20},
            "runtime_configuration": {"learning_epochs": 5, "mini_batches": 4},
            "observed": {"optimizer_steps": 20, "minibatches": 20,
                "adam_step_before": {"p": index * 20}, "adam_step_after": {"p": (index + 1) * 20},
                "adam_step_delta": {"p": 20}, "gradient_present_steps": {"p": 20},
                "pre_step_gradient_norm": [1.0] * 20, "learning_rates": [[0.001]] * 20,
                "approx_kl": None, "clip_fraction": None},
            "unavailable": {"approx_kl": "fixture", "clip_fraction": "fixture"}, "error": None})
    return rows


def test_optimizer_owned_rows_are_reassembled_by_original_parser(tmp_path):
    plan, plan_path = prepared(tmp_path, "COMPOUND", M.OPTIMIZER_PROBE)
    auth = authorization(tmp_path, plan, plan_path)
    manifest = M.read(Path(plan["raw_files"]["manifest"]))

    def owned(folder, _argv, _cwd, _timeout):
        assert folder.is_dir()
        rows = optimizer_rows(manifest)
        Path(plan["raw_files"]["records"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        save(Path(plan["raw_files"]["readback"]), {
            "observer_manifest": M.RESIDUAL.observer_manifest(manifest)})
        return execution()

    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=plan):
        receipt = M.execute_probe(plan_path, auth, owned=owned, check_lease=lease, check_gpu=gpu)
    assert receipt["measurement"]["schema"] == "optimizer_observability_inspection_v1"
    assert receipt["measurement"]["status"] == "normal" and receipt["measurement"]["records"] == 3


def test_same_consumed_request_runs_case_and_none_reference_once_each(tmp_path):
    case = prepared(tmp_path, "F1", "P_value")
    reference = prepared(tmp_path, "NONE", "P_value", role="reference")
    paired_auth = paired_authorizations(tmp_path, case, reference)
    lease, gpu = runtime_checks()
    with patch.object(M, "_verify_plan", return_value=case[0]):
        case_receipt = M.execute_probe(case[1], paired_auth, owned=legacy_owned(case[0]),
                                       check_lease=lease, check_gpu=gpu)
    with patch.object(M, "_verify_plan", return_value=reference[0]):
        reference_receipt = M.execute_probe(reference[1], paired_auth, owned=legacy_owned(reference[0]),
                                            check_lease=lease, check_gpu=gpu)
    assert case_receipt["authorization"]["request_id"] == reference_receipt["authorization"]["request_id"]
    assert case_receipt["checkpoint_sha256"] != reference_receipt["checkpoint_sha256"]
    assert {case_receipt["role"], reference_receipt["role"]} == {"case", "reference"}
    assert Path(case_receipt["claim"]["path"]).name == "case.json"
    assert Path(reference_receipt["claim"]["path"]).name == "reference.json"
    assert case_receipt["source_refs"]["auxiliary_inputs"] == case[0]["auxiliary_inputs"]
    assert reference_receipt["source_refs"]["auxiliary_inputs"] == reference[0]["auxiliary_inputs"]
    assert case[0]["auxiliary_inputs"][0]["path"] != reference[0]["auxiliary_inputs"][0]["path"]
    assert case_receipt["claim"]["body"]["auxiliary_inputs"] == case[0]["auxiliary_inputs"]
    assert reference_receipt["claim"]["body"]["auxiliary_inputs"] == reference[0]["auxiliary_inputs"]


def test_reference_rejects_non_none_and_wrong_binding(tmp_path):
    with pytest.raises(ValueError, match="NONE source"):
        prepared(tmp_path, "F1", "P_reward", role="reference")
    case = prepared(tmp_path, "F2", "P_reward")
    reference = prepared(tmp_path, "NONE", "P_reward", role="reference")
    reference_auth = paired_authorizations(tmp_path, case, reference)
    for name in ("telemetry", "checkpoint"):
        changed = deepcopy(reference[0])
        if name == "telemetry":
            changed["telemetry_sha256"] = "f" * 64
        else:
            changed["checkpoint"]["sha256"] = "f" * 64
        changed_path = Path(changed["stage"]) / f"changed-{name}-plan.json"
        save(changed_path, changed)
        raw = M.read(reference_auth)
        raw["plan_sha256"]["reference"] = M.sha(changed_path)
        changed_auth = save(tmp_path / f"paired/changed-reference-{name}.authorization.json", raw)
        with pytest.raises(ValueError, match="checkpoint 결합"):
            M._authorization(changed_auth, changed_path, changed)


def test_role_claim_blocks_new_stage_and_pending_retransmission(tmp_path):
    case = prepared(tmp_path, "F3", "P_value")
    reference = prepared(tmp_path, "NONE", "P_value", role="reference")
    case_auth = paired_authorizations(tmp_path, case, reference)
    authorization_value = M._authorization(case_auth, case[1], case[0])
    first_path, first = M._claim(case[0], case[1], authorization_value)
    duplicate = deepcopy(case[0])
    duplicate_stage = tmp_path / "replicated-stage"
    duplicate_stage.mkdir()
    duplicate["stage"] = str(duplicate_stage)
    duplicate_path = save(duplicate_stage / "plan.json", duplicate)
    raw = M.read(case_auth)
    raw["plan_sha256"]["case"] = M.sha(duplicate_path)
    duplicate_auth = save(tmp_path / "paired/duplicate.authorization.json", raw)
    duplicate_authorization = M._authorization(duplicate_auth, duplicate_path, duplicate)
    with pytest.raises(ValueError, match="claim이 이미"):
        M._claim(duplicate, duplicate_path, duplicate_authorization)
    assert M.read(first_path) == first and first["status"] == "unknown_until_parent_settlement"


def test_existing_stage_is_rejected(tmp_path):
    collected, manifest, evidence = source(tmp_path, "NONE")
    lab = tmp_path / "lab"
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").touch()
    stage = tmp_path / "exists"
    stage.mkdir()
    with patch.object(M, "verify_protocol", return_value=protocol()), \
            patch.object(M, "_source", return_value=(collected["value"], manifest, evidence)), \
            pytest.raises(ValueError, match="새 경로"):
        M.prepare_probe(collected["path"], "P_reward", stage=stage, lab=lab,
                        protocol_path=Path(M.__file__), anchor="a" * 40)
