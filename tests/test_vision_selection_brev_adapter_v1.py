from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "evals", ROOT / "bench", ROOT / "src"):
    sys.path.insert(0, str(folder))
A = importlib.import_module("vision_selection_brev_adapter_v1")
C = importlib.import_module("vision_selection_e2e_caller_v1")
X = importlib.import_module("vision_selection_execution_v1")
UTC_TZ = timezone.utc  # noqa: UP017 - Python 3.10 compatibility


class FixedFixtureDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = datetime(2026, 10, 25, 1, tzinfo=UTC_TZ)
        return value if tz is None else value.astimezone(tz)


@pytest.fixture(autouse=True)
def fixed_adapter_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(A, "datetime", FixedFixtureDateTime)


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": A.file_sha(path)}


def context(tmp_path: Path, *, expires: str = "2027-01-01T02:00:00+00:00",
            group_one: dict | None = None, groups: list[dict] | None = None,
            source_plan_sha256: str = "1" * 64, selection_plan_sha256: str = "2" * 64,
            source_plan_path: Path | None = None, selection_plan_path: Path | None = None,
            service_file: Path | None = None, corrected_groups: list[dict] | None = None) -> dict:
    probe_protocol = tmp_path / "bench/protocols/brev_recovery_v4.json"
    probe_protocol.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "bench/protocols/brev_recovery_v4.json", probe_protocol)
    source_protocol = tmp_path / "bench/protocols/vision_brev_source_20261025_v9.json"
    shutil.copyfile(ROOT / "bench/protocols/vision_brev_source_20261025_v9.json", source_protocol)
    correction_backend = tmp_path / "evals/vision_brev_backend_v1.py"
    record_backend = tmp_path / "evals/vision_brev_render_backend_v2.py"
    record_verifier = tmp_path / "evals/vision_brev_render_backend_v4.py"
    evidence_contract = tmp_path / "evals/vision_evidence_contract_v1.py"
    fixed_eval_protocol = tmp_path / "bench/protocols/fixed_eval_v1.json"
    correction_backend.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "evals/vision_brev_backend_v1.py", correction_backend)
    shutil.copyfile(ROOT / "evals/vision_brev_render_backend_v2.py", record_backend)
    shutil.copyfile(ROOT / "evals/vision_brev_render_backend_v4.py", record_verifier)
    shutil.copyfile(ROOT / "evals/vision_evidence_contract_v1.py", evidence_contract)
    shutil.copyfile(ROOT / "bench/protocols/fixed_eval_v1.json", fixed_eval_protocol)
    corrected_plan = write(tmp_path / "corrected_render_plan.json", {
        "schema": "vision_selection_corrected_render_plan_fixture_v1",
        "groups": corrected_groups or [],
    })
    controller_paths = [
        "evals/vision_selection_brev_adapter_v1.py",
        "evals/vision_selection_e2e_caller_v1.py",
        "evals/run_vision_selection_e2e_v1.py",
        "evals/vision_selection_e2e_v1.py",
        "evals/vision_selection_call_v1.py",
    ]
    for relative in controller_paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "fixture@example.invalid"],
                   check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "fixture"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", *controller_paths], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"], check=True)
    source_anchor = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    if source_plan_path is not None:
        source_plan_body = json.loads(source_plan_path.read_text(encoding="utf-8"))
        if source_plan_body.get("schema") == "vision_selection_e2e_plan_v1":
            source_plan_body["source_anchor"] = source_anchor
            write(source_plan_path, source_plan_body)
            source_plan_sha256 = A.file_sha(source_plan_path)
    if source_plan_path is None:
        source_plan_path = write(tmp_path / "source-plan-placeholder.json", {"kind": "source"})
        source_plan_sha256 = A.file_sha(source_plan_path)
    if selection_plan_path is None:
        selection_plan_path = write(tmp_path / "selection-plan-placeholder.json", {"kind": "selection"})
        selection_plan_sha256 = A.file_sha(selection_plan_path)
    correction_ref = ({"path": service_file.relative_to(tmp_path).as_posix(),
                       "sha256_lf": A.file_sha(service_file, lf=True),
                       "class": "FixtureService"} if service_file else
                      {"path": "evals/vision_brev_backend_v1.py",
                       "sha256_lf": A.file_sha(correction_backend, lf=True),
                       "class": "VisionBrevBackend"})
    correction_service_ref = {"path": correction_ref["path"],
                              "sha256_lf": correction_ref["sha256_lf"]}
    record_backend_ref = ({"path": service_file.relative_to(tmp_path).as_posix(),
                           "sha256_lf": A.file_sha(service_file, lf=True)} if service_file else
                          {"path": "evals/vision_brev_render_backend_v2.py",
                           "sha256_lf": A.file_sha(record_backend, lf=True)})
    record_verifier_ref = ({"path": service_file.relative_to(tmp_path).as_posix(),
                            "sha256_lf": A.file_sha(service_file, lf=True),
                            "class": "FixtureService"} if service_file else
                           {"path": "evals/vision_brev_render_backend_v4.py",
                            "sha256_lf": A.file_sha(record_verifier, lf=True),
                            "class": "SupervisedVisionBrevBackend"})
    protocol = {
        "schema": A.PROTOCOL_SCHEMA, "protocol_id": "v05-actual-fixture",
        "registered_before_execution": True, "deadline_utc": "2027-01-01T03:00:00+00:00",
        "source_anchor": source_anchor,
        "controller_sources_lf": {relative: A.file_sha(tmp_path / relative, lf=True)
                                  for relative in controller_paths},
        "source_plan_sha256": source_plan_sha256, "selection_plan_sha256": selection_plan_sha256,
        "plan_references": {"source": ref(tmp_path, source_plan_path),
                            "selection": ref(tmp_path, selection_plan_path)},
        "probe_execution": {"path": "bench/protocols/brev_recovery_v4.json",
                            "sha256_lf": A.file_sha(probe_protocol, lf=True)},
        "correction": {"id": "restore_body_frame_reward", "mode": "restart"},
        "execution_bindings": {
            "source_protocol": {"path": "bench/protocols/vision_brev_source_20261025_v9.json",
                                "sha256_lf": A.file_sha(source_protocol, lf=True)},
            "correction_backend": correction_ref,
            "correction_service": correction_service_ref,
            "record_backend": record_backend_ref,
            "record_verifier": record_verifier_ref,
            "evidence_contract": {"path": "evals/vision_evidence_contract_v1.py",
                                  "sha256_lf": A.file_sha(evidence_contract, lf=True)},
            "fixed_eval_protocol": {"path": "bench/protocols/fixed_eval_v1.json",
                                    "sha256_lf": A.file_sha(fixed_eval_protocol, lf=True)},
            "corrected_render_plan": ref(tmp_path, corrected_plan),
        },
        "limits": {"probe_pairs": 2, "gpu_probe_runs": 4, "corrected_runs": 2,
                   "fixed_evaluations": 2, "api_requests": 32},
        "resources": {"compute_quote_usd_per_h": 1.8, "whole_batch_budget_usd": 100.0},
        "groups": groups or [
            group_one or {"group": "source-01", "seed": 2196, "source_protocol_sha256_lf": "3" * 64,
             "checkpoints": {"NONE": "4" * 64, "F1": "5" * 64},
             "corrected_job_id": "vision-s2196-corrected", "probe_budget_gpu_s": 120,
             "action_limits": {name: {"max_wall_s": 3600, "max_compute_usd": 1.8}
                               for name in ("probe_pair", "correction", "record")},
             "execution": {"NONE": {"stage": "ref-stage-1", "run_name": "ref-run-1"},
                           "F1": {"stage": "case-stage-1", "run_name": "case-run-1"}}},
            {"group": "source-02", "seed": 2197, "source_protocol_sha256_lf": "6" * 64,
             "checkpoints": {"NONE": "7" * 64, "F1": "8" * 64},
             "corrected_job_id": "vision-s2197-corrected", "probe_budget_gpu_s": 120,
             "action_limits": {name: {"max_wall_s": 3600, "max_compute_usd": 1.8}
                               for name in ("probe_pair", "correction", "record")},
             "execution": {"NONE": {"stage": "ref-stage-2", "run_name": "ref-run-2"},
                           "F1": {"stage": "case-stage-2", "run_name": "case-run-2"}}},
        ],
    }
    protocol_path = write(tmp_path / "execution_protocol.json", protocol)
    protocol_sha = A.file_sha(protocol_path)
    resource = write(tmp_path / "resource.json", {
        "schema": "brev_resource_proof_v1", "status": "granted", "lease_id": "lease-new",
        "issued_at_utc": "2026-01-01T00:00:00+00:00", "expires_at_utc": expires,
        "compute_started_at_utc": "2026-10-25T00:00:00+00:00",
        "protocol_sha256": protocol_sha,
    })
    budget = write(tmp_path / "budget.json", {
        "schema": "brev_budget_admission_v1", "status": "granted", "lease_id": "lease-new",
        "issued_at_utc": "2026-01-01T00:00:00+00:00", "expires_at_utc": expires,
        "compute_started_at_utc": "2026-10-25T00:00:00+00:00",
        "protocol_sha256": protocol_sha, "whole_batch_budget_usd": 100.0,
        "compute_quote_usd_per_h": 1.8,
        "cost_scope": "owned_instance_elapsed_wall_includes_idle_setup_transfer_excludes_storage_v1",
    })
    lease = write(tmp_path / "lease.json", {
        "schema": A.LEASE_SCHEMA, "status": "granted", "execution_id": "v05-actual-fixture",
        "protocol_sha256": protocol_sha, "resource": ref(tmp_path, resource),
        "budget": ref(tmp_path, budget),
        "issued_at_utc": "2026-01-01T00:00:00+00:00", "expires_at_utc": expires,
    })
    return A.verify_context(tmp_path, ref(tmp_path, protocol_path), ref(tmp_path, lease),
                            now=datetime(2026, 10, 25, 1, tzinfo=UTC_TZ))


def test_context_rejects_expired_or_old_protocol_resource(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="만료"):
        context(tmp_path, expires="2026-10-24T00:30:00+00:00")


def test_each_action_receipt_must_stay_inside_lease_deadline_and_cost(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = context(tmp_path, expires="2026-10-25T01:30:00+00:00")
    receipt = {
        "completed": True, "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [], "source_before": {"anchor": "a"},
        "source_after": {"anchor": "a"}, "wall_s": 60.0,
        "finished_at": "2027-01-01T02:01:00+00:00",
    }
    with pytest.raises(ValueError, match="lease/deadline"):
        A._execution_gpu_s(receipt, "fixture action", ctx)
    entry = ctx["protocol"]["groups"][0]
    with pytest.raises(ValueError, match="상한"):
        A._action_cost(ctx, entry, "record", 3601.0)

    class NearExpiry(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 10, 25, 1, 29, tzinfo=UTC_TZ)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(A, "datetime", NearExpiry)
    with pytest.raises(ValueError, match="예약 창"):
        A.require_action_window(ctx, entry, "record")


def test_correction_service_never_trains_missing_none_cache() -> None:
    class Backend:
        def __init__(self) -> None:
            self.train_calls = 0
            self.evaluate_calls = 0

        def _completed(self, _job_id: str):
            return None

        def train(self, *_args, **_kwargs):
            self.train_calls += 1
            raise AssertionError("NONE cache miss must not train")

        def evaluate(self, *_args, **_kwargs):
            self.evaluate_calls += 1
            raise AssertionError("missing NONE cache must not evaluate")

    backend = Backend()
    service = A.CorrectionService(backend, Path("runtime"), "none-job", "fixed")
    assert service.preflight() is False
    with pytest.raises(ValueError, match="preflight"):
        service.evaluate({})
    assert backend.train_calls == backend.evaluate_calls == 0


def test_probe_rechecks_action_window_before_second_arm(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = context(tmp_path)
    group = ctx["protocol"]["groups"][0]
    stage = tmp_path / "stage"
    commands = {
        arm: {"argv": ["fixture", str(stage / arm.lower() / "descriptor.json")],
              "descriptor": str(stage / arm.lower() / "descriptor.json")}
        for arm in ("NONE", "F1")
    }
    request = {
        "schema": A.PAIR_SCHEMA, "group": "source-01", "probe": "P_reward",
        "seed": group["seed"], "arms": ["NONE", "F1"], "checkpoints": group["checkpoints"],
        "selection": {"terminal_sha256": "1" * 64, "activation_bridge_sha256": "2" * 64},
        "protocols": {"source": group["source_protocol_sha256_lf"],
                      "probe": ctx["protocol"]["probe_execution"]},
        "execution": group["execution"], "commands": commands,
        "commands_sha256": A.digest(commands),
        "budget_gpu_s": group["probe_budget_gpu_s"],
        "resource_sha256": ctx["lease"]["resource"]["sha256"],
    }
    reference = C.PairLedger(tmp_path / "pair.jsonl").authorize(request, "fixture")
    windows = {"count": 0}
    dispatches = {"count": 0}

    def window(*_args, **_kwargs):
        windows["count"] += 1
        if windows["count"] == 2:
            raise ValueError("expired before second arm")

    def runner(argv, **_kwargs):
        dispatches["count"] += 1
        descriptor = Path(argv[1]); descriptor.parent.mkdir(parents=True, exist_ok=True)
        write(descriptor, {"fixture": True})
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(A, "require_action_window", window)
    monkeypatch.setattr(A.EXECUTION, "_brev_actual_probe", lambda *_args, side, **_kwargs: {
        "seed": group["seed"],
        "stage": group["execution"]["NONE" if side == "reference" else "F1"]["stage"],
        "run_name": group["execution"]["NONE" if side == "reference" else "F1"]["run_name"],
        "measurement": {}, "gpu_s": 1.0,
    })
    result = A.run_probe_pair(ctx, reference, commands, {}, stage, runner=runner)
    assert result["status"] == "HOLD" and result["reason"] == "next_arm_window_insufficient"
    assert dispatches["count"] == 1


def test_record_result_materializes_bound_v01_contract_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = tmp_path / "evals/fixture_record_service.py"
    service.parent.mkdir(parents=True)
    service.write_text(
        "class FixtureService:\n"
        "    def record(self, job):\n"
        "        raise AssertionError('materializer must not dispatch record')\n",
        encoding="utf-8",
    )
    ctx = context(tmp_path, service_file=service)
    evidence_source = tmp_path / "evals/vision_evidence_contract_v1.py"
    evidence_spec = importlib.util.spec_from_file_location("fixture_evidence_contract", evidence_source)
    assert evidence_spec is not None and evidence_spec.loader is not None
    evidence_module = importlib.util.module_from_spec(evidence_spec)
    evidence_spec.loader.exec_module(evidence_module)
    monkeypatch.setattr(A, "VISION", evidence_module)
    checkpoint = tmp_path / "checkpoint/model.pt"
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b"corrected-checkpoint")
    checkpoint_sha = A.file_sha(checkpoint)
    record_job = {
        "job_id": "record-corrected-one", "source_job_id": "vision-s2196-corrected",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
    }
    stage = tmp_path / "record-stage"
    stage.mkdir()
    write(stage / "job.json", record_job)
    video = stage / "same_play.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=duration=6:size=64x64:rate=2", "-c:v", "mpeg4", "-y", str(video)],
        check=True, capture_output=True,
    )
    run_id = "record-corrected-one-w1"
    spec = write(stage / "window_1_v01_spec.json", {
        "schema": "vision_evidence_prepare_v1", "play_run_id": run_id,
        "source_lineage_id": "record-corrected-one-same-play",
        "parent_source_lineage_id": None,
        "video": {"path": str(video), "logical_id": "same_play_video", "play_run_id": run_id},
        "checkpoint": {"path": str(checkpoint), "logical_id": "corrected_policy",
                       "play_run_id": run_id},
        "frames": [{"timestamp_s": timestamp, "play_run_id": run_id,
                    "video_sha256": A.file_sha(video)}
                   for timestamp in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)],
        "measurement": None,
        "play": {"task": "Isaac-Velocity-Flat-Unitree-Go2-v0", "camera": "follow_camera",
                 "command_name": "base_velocity", "command_frame": "body",
                 "motion_frame": "body", "settings_sha256": "a" * 64},
    })
    result_path = write(stage / "result.json", {
        "completed": True, "checkpoint_sha256": checkpoint_sha,
        "video_sha256": A.file_sha(video),
        "artifacts": {"spec_1": {"file": spec.name, "sha256": A.file_sha(spec),
                                  "bytes": spec.stat().st_size}},
    })
    execution_path = write(stage / "execution.json", {
        "completed": True, "exit_code": 0, "timed_out": False, "cleanup_verified": True,
        "gpu_before": [], "gpu_after": [], "source_before": {"anchor": "fixture"},
        "source_after": {"anchor": "fixture"}, "wall_s": 2.0,
        "finished_at": "2026-10-25T01:00:00+00:00",
        "result_sha256": A.file_sha(result_path),
    })
    value = {"execution": ref(tmp_path, execution_path), "result": ref(tmp_path, result_path)}
    output = tmp_path / ".triage-runtime/vision-selection-e2e/materialize/evidence"
    contract = A.materialize_record_contract(
        ctx, value, record_job=record_job, output_dir=output,
    )
    verified = A.VISION.verify(contract)
    assert verified["provenance"]["checkpoint"]["sha256"] == checkpoint_sha
    assert verified["provenance"]["video"]["sha256"] == A.file_sha(video)
    assert A.materialize_record_contract(
        ctx, value, record_job=record_job, output_dir=output,
    ) == contract
    (output / "artifacts/frames/frame_01.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="JPEG|SHA"):
        A.materialize_record_contract(ctx, value, record_job=record_job, output_dir=output)


@pytest.mark.skipif(not os.environ.get("RL_TRIAGE_BREV_PROBE_FIXTURE"),
                    reason="actual Brev probe fixture가 지정되지 않았다")
def test_actual_pair_dispatch_validates_six_raw_files_and_cache_is_zero_side_effect(tmp_path: Path) -> None:
    stored = Path(os.environ["RL_TRIAGE_BREV_PROBE_FIXTURE"]).resolve(strict=True)
    jobs = {side: json.loads((stored / side / "job.json").read_text(encoding="utf-8"))
            for side in ("case", "reference")}
    group = {"group": "source-01", "seed": 2196, "source_protocol_sha256_lf": "3" * 64,
             "checkpoints": {"NONE": jobs["reference"]["checkpoint_sha256"],
                             "F1": jobs["case"]["checkpoint_sha256"]},
             "corrected_job_id": "vision-s2196-corrected", "probe_budget_gpu_s": 120,
             "action_limits": {name: {"max_wall_s": 3600, "max_compute_usd": 1.8}
                               for name in ("probe_pair", "correction", "record")},
             "execution": {"NONE": {"stage": jobs["reference"]["stage"],
                                       "run_name": jobs["reference"]["run_name"]},
                           "F1": {"stage": jobs["case"]["stage"],
                                    "run_name": jobs["case"]["run_name"]}}}
    ctx = context(tmp_path, group_one=group)
    request = {
        "schema": A.PAIR_SCHEMA, "group": "source-01", "probe": "P_episode", "seed": 2196,
        "arms": ["NONE", "F1"],
        "checkpoints": {"NONE": jobs["reference"]["checkpoint_sha256"],
                        "F1": jobs["case"]["checkpoint_sha256"]},
        "selection": {"terminal_sha256": "1" * 64, "activation_bridge_sha256": "2" * 64},
        "protocols": {"source": "3" * 64, "probe": ctx["protocol"]["probe_execution"]},
        "execution": {"NONE": {"stage": jobs["reference"]["stage"],
                                "run_name": jobs["reference"]["run_name"]},
                      "F1": {"stage": jobs["case"]["stage"],
                             "run_name": jobs["case"]["run_name"]}},
        "budget_gpu_s": 120, "resource_sha256": ctx["lease"]["resource"]["sha256"],
    }
    stage = tmp_path / "runtime/probes/source-01"
    script = tmp_path / "copy_actual.py"
    # 작은 실제 subprocess worker를 사용해 원문을 복사하고 portable descriptor를 만든다.
    script.write_text(
        "import hashlib,json,shutil,sys\nfrom pathlib import Path\n"
        "src=Path(sys.argv[1]); dst=Path(sys.argv[2]); side=sys.argv[3]; probe=sys.argv[4]; checkpoint=sys.argv[5]\n"
        "dst.mkdir(parents=True,exist_ok=True)\n"
        "names={'job':'job.json','worker':'worker.json','execution':'execution.json','backend_result':'backend_result.json','report_jobs':'report.jobs.json','report':'report.json'}\n"
        "[shutil.copyfile(src/name,dst/name) for name in names.values()]\n"
        "sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "root=Path(sys.argv[6]); desc={'schema':'vision_selection_brev_probe_evidence_v1','side':side,'probe':probe,'checkpoint_sha256':checkpoint,'sources':{k:{'path':(dst/v).relative_to(root).as_posix(),'sha256':sha(dst/v)} for k,v in names.items()}}\n"
        "Path(sys.argv[7]).write_text(json.dumps(desc,sort_keys=True)+'\\n',encoding='utf-8')\n",
        encoding="utf-8",
    )
    commands = {}
    for arm, side in (("NONE", "reference"), ("F1", "case")):
        raw = stage / arm.lower() / "raw"
        descriptor = stage / arm.lower() / "descriptor.json"
        commands[arm] = {"argv": [sys.executable, str(script), str(stored / side), str(raw), side,
                                   "P_episode", jobs[side]["checkpoint_sha256"], str(tmp_path),
                                   str(descriptor)], "descriptor": str(descriptor)}
    request["commands"] = commands
    request["commands_sha256"] = A.digest(commands)
    ledger = C.PairLedger(tmp_path / "pair.ledger.jsonl")
    authorization = ledger.authorize(request, "fixture-approver")
    expected = X._probe_execution_source(tmp_path, {
        "source_anchor": "fa0c8f2707eaee93c84ff273a4510c164ef3fd90",
        "protocol": ctx["protocol"]["probe_execution"],
    })
    result = A.run_probe_pair(ctx, authorization, commands, expected, stage)
    assert result["status"] == "complete" and result["gpu_runs"] == 2
    assert result["NONE"]["outcome"] == result["F1"]["outcome"] == "normal"
    cached = A.run_probe_pair(
        ctx, authorization, commands, expected, stage,
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache dispatch")),
    )
    assert cached == result


def test_partial_stage_holds_without_dispatch(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    group = ctx["protocol"]["groups"][0]
    request = {
        "schema": A.PAIR_SCHEMA, "group": "source-01", "probe": "P_reward", "seed": group["seed"],
        "arms": ["NONE", "F1"], "checkpoints": group["checkpoints"],
        "selection": {"terminal_sha256": "1" * 64, "activation_bridge_sha256": "2" * 64},
        "protocols": {"source": group["source_protocol_sha256_lf"],
                      "probe": ctx["protocol"]["probe_execution"]},
        "execution": group["execution"], "commands": {}, "commands_sha256": A.digest({}),
        "budget_gpu_s": group["probe_budget_gpu_s"],
        "resource_sha256": ctx["lease"]["resource"]["sha256"],
    }
    ledger = C.PairLedger(tmp_path / "pair.ledger.jsonl")
    authorization = ledger.authorize(request, "fixture")
    stage = tmp_path / "partial"; stage.mkdir(); (stage / "orphan.json").write_text("{}")
    result = A.run_probe_pair(
        ctx, authorization, {"NONE": {}, "F1": {}}, {}, stage,
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("partial dispatch")),
    )
    assert result == {"schema": A.PAIR_RESULT_SCHEMA, "status": "HOLD",
                      "reason": "partial_stage", "request_sha256": A.digest(request)}


def test_pair_rejects_checkpoint_or_resource_not_registered_in_execution_protocol(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    group = ctx["protocol"]["groups"][0]
    request = {
        "schema": A.PAIR_SCHEMA, "group": "source-01", "probe": "P_reward", "seed": group["seed"],
        "arms": ["NONE", "F1"], "checkpoints": group["checkpoints"],
        "selection": {"terminal_sha256": "1" * 64, "activation_bridge_sha256": "2" * 64},
        "protocols": {"source": group["source_protocol_sha256_lf"],
                      "probe": ctx["protocol"]["probe_execution"]},
        "execution": group["execution"], "commands": {}, "commands_sha256": A.digest({}),
        "budget_gpu_s": group["probe_budget_gpu_s"],
        "resource_sha256": "9" * 64,
    }
    authorization = C.PairLedger(tmp_path / "pair.ledger.jsonl").authorize(request, "fixture")
    with pytest.raises(ValueError, match="등록 seed/checkpoint/protocol/resource"):
        A.run_probe_pair(ctx, authorization, {"NONE": {}, "F1": {}}, {}, tmp_path / "stage")


def test_registered_controller_source_change_invalidates_existing_protocol_and_lease(
        tmp_path: Path) -> None:
    ctx = context(tmp_path)
    controller = tmp_path / "evals/vision_selection_brev_adapter_v1.py"
    controller.write_text(controller.read_text(encoding="utf-8") + "\n# changed after registration\n",
                          encoding="utf-8")
    with pytest.raises(ValueError, match="controller"):
        A.refresh_context(ctx)


def test_probe_command_digest_rejects_argv_change_before_subprocess(
        tmp_path: Path) -> None:
    ctx = context(tmp_path)
    group = ctx["protocol"]["groups"][0]
    stage = tmp_path / "command-bound-stage"
    original = {
        arm: {"argv": ["registered", arm],
              "descriptor": str(stage / arm.lower() / "descriptor.json")}
        for arm in ("NONE", "F1")
    }
    request = {
        "schema": A.PAIR_SCHEMA, "group": "source-01", "probe": "P_reward",
        "seed": group["seed"], "arms": ["NONE", "F1"], "checkpoints": group["checkpoints"],
        "selection": {"terminal_sha256": "1" * 64, "activation_bridge_sha256": "2" * 64},
        "protocols": {"source": group["source_protocol_sha256_lf"],
                      "probe": ctx["protocol"]["probe_execution"]},
        "execution": group["execution"], "commands": original,
        "commands_sha256": A.digest(original),
        "budget_gpu_s": group["probe_budget_gpu_s"],
        "resource_sha256": ctx["lease"]["resource"]["sha256"],
    }
    authorization = C.PairLedger(tmp_path / "command.ledger.jsonl").authorize(request, "fixture")
    changed = json.loads(json.dumps(original))
    changed["F1"]["argv"] = ["foreign", "--side-effect"]
    calls = {"count": 0}
    def runner(*_args, **_kwargs):
        calls["count"] += 1
        raise AssertionError("unregistered argv must not run")
    with pytest.raises(ValueError, match="request"):
        A.run_probe_pair(ctx, authorization, changed, {}, stage, runner=runner)
    assert calls["count"] == 0
