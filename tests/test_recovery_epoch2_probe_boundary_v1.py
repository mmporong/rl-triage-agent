# pyright: reportMissingImports=false
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "tests", "src")]

import five_policy_recovery_epoch2_v1 as SECOND
import recovery_epoch2_parent_binding_v1 as BIND
import recovery_epoch2_probe_boundary_v1 as M
import test_local_pair_probe_host_v1 as HOST_FIXTURE


def save(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_typed_protocol_preserves_parent_and_separates_runtime_view(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    selection = save(root / "bench/protocols/selection.json", {"schema": "selection"})
    runtime = save(root / "bench/protocols/runtime.json", {"schema": "runtime"})
    for name in M.SOURCE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(name + "\n", encoding="utf-8")
    parent_path = save(root / "bench/protocols/parent.json", {})
    anchor = "a" * 40
    parent = {
        "schema": SECOND.LOOP_SCHEMA,
        "purpose": "R4",
        "api_cap": 32,
        "model": "frozen-model",
        "sources_lf": {"old.py": "1" * 64},
        "source_digests": {"case-03:fixed_eligible": "2" * 64},
        "second_seeds": {"case-03:fixed_eligible": 9001},
        "targets": ["case-03:fixed_eligible"],
        "view_protocol": {"path": selection.relative_to(root).as_posix(),
                          "sha256_lf": SECOND.sha(selection, lf=True)},
    }
    save(parent_path, parent)
    calls: list[tuple[str, Path, str]] = []

    def original(path: Path, checked_anchor: str, *, lab: Path | None = None) -> dict:
        calls.append(("parent", Path(path).resolve(), checked_anchor))
        assert lab is None
        return deepcopy(parent)

    monkeypatch.setattr(M, "ROOT", root)
    monkeypatch.setattr(M.RV, "PROTOCOL", runtime)
    monkeypatch.setattr(M, "_ORIGINAL_VERIFY", original)
    value = M.prepare_protocol(parent_path, anchor)
    assert value["selection_view_protocol"] == parent["view_protocol"]
    assert value["view_protocol"] == {
        "path": runtime.relative_to(root).as_posix(),
        "sha256_lf": SECOND.sha(runtime, lf=True),
    }
    assert value["parent_loop_protocol"]["anchor"] == anchor
    assert {key: value[key] for key in parent if key != "view_protocol"} == {
        key: value[key] for key in parent if key != "view_protocol"
    }
    protocol_path = save(root / "bench/protocols/typed.json", value)
    monkeypatch.setattr(M.OLD.TRANSPORT, "_verify_protocol", lambda *_args: deepcopy(value))
    monkeypatch.setattr(M.RV, "verify_protocol", lambda path, checked_anchor: calls.append(
        ("runtime", Path(path).resolve(), checked_anchor)) or {"verified": True})
    assert M.verify_loop_protocol(protocol_path, anchor) == value
    assert calls.count(("parent", parent_path.resolve(), anchor)) == 2
    assert calls[-1] == ("runtime", runtime.resolve(), anchor)

    changed = deepcopy(value)
    changed["view_protocol"] = deepcopy(changed["selection_view_protocol"])
    monkeypatch.setattr(M.OLD.TRANSPORT, "_verify_protocol", lambda *_args: deepcopy(changed))
    with pytest.raises(ValueError, match="파생|runtime"):
        M.verify_loop_protocol(protocol_path, anchor)


def epoch2_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict[str, dict]]:
    case_checkpoint = tmp_path / "case-model.pt"
    reference_checkpoint = tmp_path / "reference-model.pt"
    case_checkpoint.write_bytes(b"case checkpoint")
    reference_checkpoint.write_bytes(b"reference checkpoint")
    case_telemetry = save(tmp_path / "case-telemetry.json", {"case": True})
    reference_telemetry = save(tmp_path / "reference-telemetry.json", {"reference": True})
    bindings = {
        "case": {"checkpoint_sha256": SECOND.sha(case_checkpoint),
                 "telemetry_sha256": SECOND.sha(case_telemetry)},
        "reference": {"checkpoint_sha256": SECOND.sha(reference_checkpoint),
                      "telemetry_sha256": SECOND.sha(reference_telemetry)},
    }
    receipt = save(tmp_path / "first-receipt.json", {"status": "not_recovered"})
    epoch = save(tmp_path / "epoch2.json", {"status": "input_only"})
    correction = "restore_body_frame_reward"
    prior = {"verdict": "not_recovered", "correction_id": correction,
             "result_sha256": SECOND.sha(receipt),
             "checkpoint_sha256": bindings["case"]["checkpoint_sha256"]}
    public_facts = HOST_FIXTURE.packet(False)["facts"]
    backend_facts = HOST_FIXTURE.packet(True)["facts"]
    public = SECOND.DRIVER.C.V.S.build_packet(
        epoch=2, applied_corrections=[correction], prior_receipt=prior,
        facts=public_facts, input_bindings=bindings, budget_left=3)
    backend = SECOND.DRIVER.C.V.S.build_packet(
        epoch=2, applied_corrections=[correction], prior_receipt=prior,
        facts=backend_facts, input_bindings=bindings, budget_left=3)
    source = {
        "schema": SECOND.SOURCE_SCHEMA, "case_id": "case-03", "policy": "fixed_eligible",
        "packet": public, "backend_packet": backend,
        "qualification": {"arm": "COMPOUND", "label": "eligible", "diagnosis_eligible": True},
        "source_binding": {
            "epoch2_input_sha256": SECOND.sha(epoch),
            "first_receipt_sha256": SECOND.sha(receipt),
            "first_component_sha256_lf": "a" * 64,
            "case_checkpoint_sha256": bindings["case"]["checkpoint_sha256"],
            "reference_checkpoint_sha256": bindings["reference"]["checkpoint_sha256"],
        },
        "index": {"path": str(epoch), "sha256": SECOND.sha(epoch)},
        "inputs": {"epoch2_input": SECOND.ref(epoch), "first_receipt": SECOND.ref(receipt)},
        "lab": str(tmp_path / "lab"), "first_correction": correction,
        "original_faults": ["F1", "R1"], "remaining_faults": ["R1"],
        "providers": None, "visual_evidence": None,
        "approval_granted": False, "api_authorized": False,
    }
    monkeypatch.setattr(SECOND, "_prepare_source_value", lambda *_args, **_kwargs: deepcopy(source))
    source_rows: dict[str, dict] = {}
    for role, checkpoint, telemetry in (
        ("case", case_checkpoint, case_telemetry),
        ("reference", reference_checkpoint, reference_telemetry),
    ):
        manifest = save(tmp_path / f"{role}-manifest.json", {"role": role})
        readback = save(tmp_path / f"{role}-readback.json", {"role": role})
        prefix = f"runtime:{role}:"
        source_rows[role] = {
            "kind": "registered_training_observation",
            "seed": 2307,
            "source_arm": "FIRST" if role == "case" else "NONE",
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": SECOND.sha(checkpoint)},
            "telemetry_sha256": SECOND.sha(telemetry),
            "binding_digest": ("b" if role == "case" else "c") * 64,
            "source_manifest_sha256": SECOND.sha(manifest),
            "runtime_facts": {key: item for key, item in backend_facts.items()
                              if key.startswith(prefix)},
            "original_faults": ["F1", "R1"] if role == "case" else [],
            "applied_corrections": [correction] if role == "case" else [],
            "remaining_faults": ["R1"] if role == "case" else [],
            "readback": str(readback.resolve()),
        }
    return source, source_rows


class BoundaryReached(BaseException):
    pass


def test_actual_host_reaches_owned_boundary_with_runtime_view_and_backend_parent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, source_rows = epoch2_source(tmp_path, monkeypatch)
    anchor = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    lab = tmp_path / "lab"
    (lab / "_isaac_sim").mkdir(parents=True)
    (lab / "_isaac_sim/python.bat").touch()
    component = ROOT / "evals/lineage_probe_backend_v1.py"
    base_protocol = {
        "isaac_python_relative": "_isaac_sim/python.bat",
        "watchdogs_s": {"legacy": 600, "optimizer": 900},
        "source_sha256_lf": {},
        "runtime_source_sha256_lf": {},
    }
    probe_protocol = save(tmp_path / "probe-protocol.json", {"fixture": True})
    case_input = save(tmp_path / "case-input.json", {"role": "case"})
    reference_input = save(tmp_path / "reference-input.json", {"role": "reference"})
    monkeypatch.setattr(SECOND._PROBE, "verify_protocol", lambda *_args, **_kwargs: {
        "component_probe_protocol": {"path": component.relative_to(ROOT).as_posix(),
                                     "sha256_lf": SECOND.sha(component, lf=True)}})
    monkeypatch.setattr(SECOND._PROBE.P1, "verify_protocol", lambda *_args, **_kwargs: deepcopy(base_protocol))
    monkeypatch.setattr(SECOND._PROBE, "_source", lambda _path, _lab, role: (
        {}, {}, deepcopy(source_rows[role])))
    monkeypatch.setattr(SECOND._PROBE.POST, "_admit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(SECOND._PROBE.POST, "_lease_reader", lambda _stage: (
        lambda: {"leaseId": "0" * 36}))
    monkeypatch.setattr(SECOND._PROBE.P1, "_auxiliary_inputs", lambda *_args, **_kwargs: {})

    owned_calls: list[dict] = []

    def owned(stage: Path, command: list[str], owned_lab: Path, watchdog: int) -> dict:
        owned_calls.append({"stage": stage, "command": command, "lab": owned_lab,
                            "watchdog": watchdog})
        raise BoundaryReached

    gpu_calls: list[Path] = []

    def gpu_reader(stage: Path) -> dict:
        gpu_calls.append(Path(stage))
        return {"physical_busy": False, "processes": 0}

    approvals: list[dict] = []

    def approve(request: dict) -> dict:
        approvals.append(deepcopy(request))
        response: dict[str, object] = {"approved": True}
        if request["phase"] == "execute":
            response.update(owned=owned, gpu_reader=gpu_reader)
        return response

    with BIND.epoch2_snapshot_scope([source]) as parents:
        run_id = "r4_case_03_fixed_eligible"
        public_snapshot = SECOND._DRIVER.C.V.build_snapshot(
            source["packet"], run_id, "fixed_eligible",
            SECOND.DRIVER.C.V.file_digest(SECOND.DRIVER.C.V.PROTOCOL),
            tmp_path / "public-ledger.jsonl")
        choice = SECOND.DRIVER.C.choose(public_snapshot, "fixed_eligible")
        with M.backend_parent_scope(
            [source], parents=parents,
        ) as boundary, pytest.raises(BoundaryReached):
            SECOND.execute_pair(
                public_snapshot, choice, backend_packet=source["backend_packet"],
                case_input=case_input, reference_input=reference_input,
                probe_protocol=probe_protocol, view_protocol=M.RV.PROTOCOL,
                anchor=anchor, lab=lab, out=tmp_path / "host-output",
                verify_context=approve)
        registered = boundary["case-03:fixed_eligible"]
    assert [row["phase"] for row in approvals] == ["prepare", "prepare", "execute"]
    assert len(owned_calls) == len(gpu_calls) == 1
    authorization = json.loads(
        (tmp_path / "host-output/probe/authorization.json").read_text(encoding="utf-8"))
    assert authorization["snapshot"]["inputs"]["parent"] == registered["backend_parent"]
    assert authorization["snapshot"]["inputs"]["run_id"] == registered["backend_run_id"]
    assert authorization["snapshot"]["binding"]["parent"] == registered["backend_parent"]
    assert SECOND._HOST is M._ORIGINAL_HOST


def test_backend_registration_rejects_cross_parent_checkpoint_and_budget(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, _rows = epoch2_source(tmp_path, monkeypatch)
    target = "case-03:fixed_eligible"
    protocol = {
        "source_digests": {target: SECOND.digest(source)},
        "selection_view_protocol": {
            "path": SECOND.DRIVER.C.V.PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256_lf": SECOND.sha(SECOND.DRIVER.C.V.PROTOCOL, lf=True),
        },
    }
    parent = {
        "run_id": "r4_case_03_fixed_eligible", "policy": "fixed_eligible", "epoch": 1,
        **deepcopy(source["packet"]["prior_receipt"]),
    }
    assert M._backend_registrations([source], {target: parent}, protocol)
    wrong_parent = deepcopy(parent)
    wrong_parent["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="public parent"):
        M._backend_registrations([source], {target: wrong_parent}, protocol)

    changed = deepcopy(source)
    changed["backend_packet"]["input_bindings"]["case"]["checkpoint_sha256"] = "e" * 64
    changed_protocol = deepcopy(protocol)
    changed_protocol["source_digests"][target] = SECOND.digest(changed)
    with pytest.raises(ValueError, match="public/backend"):
        M._backend_registrations([changed], {target: parent}, changed_protocol)
    changed = deepcopy(source)
    changed["backend_packet"]["execution_budget_s"] += 1
    changed_protocol["source_digests"][target] = SECOND.digest(changed)
    with pytest.raises(ValueError, match="public/backend"):
        M._backend_registrations([changed], {target: parent}, changed_protocol)
