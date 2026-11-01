# pyright: reportMissingImports=false
"""RI373 source index를 production same-Play 녹화 체인에 결합한다.

CPU parent는 익명 ``source_id``와 원천 SHA를 다시 검증한다. Isaac SDK import와
F2 class patch는 ``sdk-child`` full 실행 안에서만 일어나며 복원 receipt가 필수다.
F5 post-create material lifecycle은 이 버전에서 실행하지 않고 HOLD로 남긴다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import agentic_fault_playback_v1 as PLAYBACK
import lineage_source_inputs_v2 as INPUTS
import local_same_play_capture_v1 as V1
import local_same_play_capture_v2 as V2
import local_same_play_capture_v3 as V3

SCHEMA = "lineage_same_play_capture_protocol_v1"
RESULT_SCHEMA = "lineage_same_play_capture_execution_v1"
FINAL_SCHEMA = "lineage_same_play_capture_result_v1"
ARTIFACT_SCHEMA = "lineage_same_play_capture_artifact_v1"
INDEX_LOGICAL_ID = "lineage_source_inputs_v2_collected_index"
SOURCE_RESULT_LOGICAL_ID = "lineage_source_batch_v2_result"
SOURCE_FILES = (
    "bench/lineage_same_play_capture_v1.py",
    "tests/test_lineage_same_play_capture_v1.py",
)
EXTRA_SOURCE_FILES = (
    *V2.SOURCE_FILES,
    *V3.SOURCE_FILES,
    "evals/lineage_source_inputs_v2.py",
    "tests/test_lineage_source_inputs_v2.py",
    "bench/agentic_fault_playback_v1.py",
    "bench/hidden_faults.py",
)
PLAYBACK_RUNTIME_FILES = (
    "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
)
NO_INJECTION_ARMS = frozenset({"NONE", "F1", "COMPOUND"})
HEX40 = frozenset("0123456789abcdef")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _hex40(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX40


def source_files() -> tuple[str, ...]:
    """기존 transitive recorder closure에 lineage adapter 경계를 더한다."""
    return tuple(
        sorted(set(V1.source_files()) | set(EXTRA_SOURCE_FILES) | set(SOURCE_FILES))
    )


def _source_map() -> dict[str, str]:
    return {name: V1._sha(ROOT / name, lf=True) for name in source_files()}


def _runtime_pins(lab: Path, names: Sequence[str], *, lf: bool) -> dict[str, str]:
    pins: dict[str, str] = {}
    for name in names:
        path = Path(lab) / name
        _require(
            path.is_file() and not path.is_symlink(), f"runtime pin이 없다: {name}"
        )
        pins[name] = V1._sha(path, lf=lf)
    return dict(sorted(pins.items()))


def _source_context(index_path: Path, source_id: str, lab: Path) -> dict[str, Any]:
    """collected checker 뒤에 source 원문 6개 서명을 다시 결합한다."""
    public, metadata = INPUTS.checked(index_path)
    public_binding = public.get("sources", {}).get(source_id)
    _require(isinstance(public_binding, dict), "등록되지 않은 익명 source_id")
    matches = [
        (job_id, row)
        for job_id, row in metadata["sources"].items()
        if row.get("public_id") == source_id
    ]
    _require(len(matches) == 1, "익명 source_id의 private source가 하나가 아니다")
    job_id, private = matches[0]
    binding = private.get("binding")
    _require(binding == public_binding, "public/private source binding이 다르다")
    manifest = private.get("manifest")
    _require(
        isinstance(manifest, dict) and manifest.get("job_id") == job_id,
        "source manifest job 결합이 다르다",
    )
    source_result_ref = metadata.get("source_result")
    _require(isinstance(source_result_ref, dict), "source result ref가 없다")
    source_result_path = Path(source_result_ref.get("path", "")).resolve()
    _require(
        source_result_path.is_file()
        and V1._sha(source_result_path) == source_result_ref.get("sha256"),
        "source result SHA가 다르다",
    )
    source_result = V1._read(source_result_path)
    records = source_result.get("records")
    _require(
        source_result.get("status") == "complete"
        and source_result.get("source_jobs_completed") == 20
        and source_result.get("fixed26_completed") == 20
        and isinstance(records, list)
        and len(records) == 20,
        "actual source20/fixed20 완료 결과가 아니다",
    )
    result_rows = [
        row
        for row in cast(list[Any], records)
        if isinstance(row, dict) and row.get("job_id") == job_id
    ]
    _require(len(result_rows) == 1, "source result에서 job record가 하나가 아니다")
    result_row = result_rows[0]
    stage = source_result_path.parent / job_id
    paths = {
        "manifest": stage / "manifest.json",
        "evidence": stage / "training" / "training_evidence.json",
        "records": stage / "training" / "records.jsonl",
        "fixed_result": stage / "fixed-result.json",
        "accepted": stage / "accepted.json",
    }
    _require(
        all(path.is_file() and not path.is_symlink() for path in paths.values()),
        "source stage 원문이 빠졌다",
    )
    _require(
        V1._read(paths["manifest"]) == manifest
        and V1._read(paths["accepted"]) == result_row,
        "manifest/accepted 원문이 collected source와 다르다",
    )
    expected = {
        "manifest_sha256": V1._sha(paths["manifest"]),
        "evidence_sha256": V1._sha(paths["evidence"]),
        "records_sha256": V1._sha(paths["records"]),
        "fixed_result_sha256": V1._sha(paths["fixed_result"]),
    }
    _require(
        all(binding.get(key) == digest for key, digest in expected.items()),
        "source manifest/evidence/records/fixed20 서명이 다르다",
    )
    _require(
        result_row.get("evidence_sha256") == expected["evidence_sha256"]
        and result_row.get("records_sha256") == expected["records_sha256"]
        and result_row.get("fixed26", {}).get("conditions") == 26
        and result_row.get("fixed26", {}).get("result_sha256")
        == expected["fixed_result_sha256"],
        "source result의 fixed20 record 결합이 다르다",
    )
    checkpoint = result_row.get("checkpoint")
    _require(isinstance(checkpoint, dict), "source checkpoint ref가 없다")
    checked_checkpoint = cast(dict[str, Any], checkpoint)
    checkpoint_path = Path(checked_checkpoint.get("path", "")).resolve()
    _require(
        checkpoint_path.is_file()
        and not checkpoint_path.is_symlink()
        and V1._sha(checkpoint_path) == checked_checkpoint.get("sha256")
        and checked_checkpoint.get("sha256") == binding.get("checkpoint_sha256")
        and checked_checkpoint.get("iteration") == 299
        and checked_checkpoint.get("size_bytes") == checkpoint_path.stat().st_size,
        "source checkpoint exact SHA가 다르다",
    )
    training_pins = cast(dict[str, str], manifest.get("runtime_source_sha256_lf"))
    _require(
        isinstance(training_pins, dict) and bool(training_pins),
        "source training runtime pin이 없다",
    )
    for name, digest in training_pins.items():
        _require(
            V1._sha(Path(lab) / name, lf=True) == digest,
            f"source training runtime pin이 바뀌었다: {name}",
        )
    arm = manifest.get("arm")
    _require(arm in {*NO_INJECTION_ARMS, "F2", "F5"}, "지원 범위 밖 source arm")
    return {
        "public": deepcopy(public),
        "metadata": deepcopy(metadata),
        "source_id": source_id,
        "job_id": job_id,
        "arm": arm,
        "binding": deepcopy(binding),
        "manifest": deepcopy(manifest),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checked_checkpoint["sha256"],
        },
        "source_result": {
            "path": str(source_result_path),
            "sha256": source_result_ref["sha256"],
        },
        "index": V1._ref(index_path),
        "training_runtime_pins": dict(training_pins),
    }


def _playback(arm: str, source_id: str, checkpoint_sha256: str) -> dict[str, Any]:
    if arm == "F2":
        return {
            "kind": "class_pre_env",
            "status": "registered",
            "registration": PLAYBACK.registration(source_id, checkpoint_sha256, "F2"),
            "all_go2_joints_required": 12,
            "restore_required": True,
        }
    if arm == "F5":
        return {
            "kind": "environment_post_create",
            "status": "hold_unsupported",
            "reason": "current recorder child has no verified post-create material restore lifecycle",
            "friction_scale": 0.25,
            "replacement_arm_allowed": False,
        }
    return {"kind": "none", "status": "registered", "restore_required": False}


def prepare_protocol(
    index: Path,
    source_id: str,
    lab: Path,
    out: Path,
    *,
    source_anchor: str | None = None,
) -> dict[str, Any]:
    """검증된 source20 중 익명 source 하나를 write-once capture protocol로 만든다."""
    if source_anchor is not None:
        _require(_hex40(source_anchor), "source anchor 형식")
    context = _source_context(index, source_id, lab)
    arm = cast(str, context["arm"])
    playback = _playback(arm, source_id, context["checkpoint"]["sha256"])
    status = (
        "hold_unsupported"
        if arm == "F5"
        else (
            "registered_before_local_gpu"
            if source_anchor is not None
            else "candidate_not_registered"
        )
    )
    value = {
        "schema": SCHEMA,
        "status": status,
        "source_anchor": source_anchor,
        "source_files_lf": _source_map(),
        "runtime_pins": _runtime_pins(lab, V1.RUNTIME_PINS, lf=False),
        "playback_runtime_pins": _runtime_pins(lab, PLAYBACK_RUNTIME_FILES, lf=True),
        "training_runtime_pins": context["training_runtime_pins"],
        "source_input": {
            "index": {
                "logical_id": INDEX_LOGICAL_ID,
                "sha256": context["index"]["sha256"],
            },
            "source_id": source_id,
            "binding": context["binding"],
        },
        "source_result": {
            "logical_id": SOURCE_RESULT_LOGICAL_ID,
            "sha256": context["source_result"]["sha256"],
        },
        "checkpoint": {
            "logical_id": source_id,
            "sha256": context["checkpoint"]["sha256"],
            "iteration": 299,
        },
        "private": {
            "job_id": context["job_id"],
            "arm": arm,
            "fault_metadata_model_visible": False,
        },
        "playback": playback,
        "platform": {"os": "windows", "isaacsim": "4.5.0", "gpu": "RTX 3060"},
        "gpu": {
            "lease_helper": "Use-GpuLease.ps1",
            "owner": V1.OWNER,
            "cloud_lease": False,
            "exclusive": True,
        },
        "play": {
            "task": V1.TASK,
            "num_envs": 1,
            "fps": V1.FPS,
            "full_frames": V1.FULL_FRAMES,
            "smoke_frames": 1,
            "camera_profile": "same_play_fixed_v1",
            "kit_args": V1.KIT_ARGS,
        },
        "shutdown": {
            "smoke": "record_agentic_state_sidecar_v2",
            "full": "vision_shutdown_policy_v1",
            "terminal_count": 1,
        },
        "layout": dict(V2.LAYOUT),
        "authority": {
            "gpu": False,
            "api": False,
            "model": False,
            "capture_approved": False,
            "used_for_decision": False,
        },
    }
    V1._write_new(out, value)
    return value


def _verify_source_anchor(repo: Path, anchor: str, sources: Mapping[str, str]) -> None:
    for relative, expected in sources.items():
        path = repo / relative
        _require(
            path.is_file() and V1._sha(path, lf=True) == expected,
            f"current source changed: {relative}",
        )
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{anchor}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        normalized = blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(
            hashlib.sha256(normalized).hexdigest() == expected,
            f"source anchor blob changed: {relative}",
        )


def validate_protocol(
    path: Path,
    index: Path,
    lab: Path,
    *,
    execution_anchor: str | None = None,
    verify_anchor: bool = True,
    repo: Path = ROOT,
) -> dict[str, Any]:
    value = V1._read(path)
    expected = {
        "schema",
        "status",
        "source_anchor",
        "source_files_lf",
        "runtime_pins",
        "playback_runtime_pins",
        "training_runtime_pins",
        "source_input",
        "source_result",
        "checkpoint",
        "private",
        "playback",
        "platform",
        "gpu",
        "play",
        "shutdown",
        "layout",
        "authority",
    }
    _require(
        set(value) == expected and value.get("schema") == SCHEMA,
        "lineage capture protocol shape",
    )
    arm = value.get("private", {}).get("arm")
    expected_status = (
        "hold_unsupported" if arm == "F5" else "registered_before_local_gpu"
    )
    _require(value.get("status") == expected_status, "lineage capture protocol status")
    source_id = value.get("source_input", {}).get("source_id")
    _require(isinstance(source_id, str), "anonymous source_id")
    context = _source_context(index, source_id, lab)
    _require(
        value["source_input"]
        == {
            "index": {
                "logical_id": INDEX_LOGICAL_ID,
                "sha256": context["index"]["sha256"],
            },
            "source_id": source_id,
            "binding": context["binding"],
        }
        and value["source_result"]
        == {
            "logical_id": SOURCE_RESULT_LOGICAL_ID,
            "sha256": context["source_result"]["sha256"],
        }
        and value["checkpoint"]
        == {
            "logical_id": source_id,
            "sha256": context["checkpoint"]["sha256"],
            "iteration": 299,
        }
        and value["private"]
        == {
            "job_id": context["job_id"],
            "arm": context["arm"],
            "fault_metadata_model_visible": False,
        },
        "lineage collected source binding",
    )
    _require(
        value["playback"]
        == _playback(context["arm"], source_id, context["checkpoint"]["sha256"]),
        "playback registration binding",
    )
    _require(value["source_files_lf"] == _source_map(), "recorder source closure")
    _require(
        len(value["source_files_lf"]) >= 235,
        "기존 recorder closure보다 작은 source closure",
    )
    for names, key, lf in (
        (V1.RUNTIME_PINS, "runtime_pins", False),
        (PLAYBACK_RUNTIME_FILES, "playback_runtime_pins", True),
    ):
        _require(value[key] == _runtime_pins(lab, names, lf=lf), f"{key} readback")
    _require(
        value["training_runtime_pins"] == context["training_runtime_pins"],
        "training runtime readback",
    )
    _require(
        value["platform"] == {"os": "windows", "isaacsim": "4.5.0", "gpu": "RTX 3060"}
        and value["gpu"]
        == {
            "lease_helper": "Use-GpuLease.ps1",
            "owner": V1.OWNER,
            "cloud_lease": False,
            "exclusive": True,
        }
        and value["play"]
        == {
            "task": V1.TASK,
            "num_envs": 1,
            "fps": V1.FPS,
            "full_frames": V1.FULL_FRAMES,
            "smoke_frames": 1,
            "camera_profile": "same_play_fixed_v1",
            "kit_args": V1.KIT_ARGS,
        }
        and value["shutdown"]
        == {
            "smoke": "record_agentic_state_sidecar_v2",
            "full": "vision_shutdown_policy_v1",
            "terminal_count": 1,
        }
        and value["layout"] == V2.LAYOUT,
        "same-Play runtime contract",
    )
    _require(
        value["authority"]
        == {
            "gpu": False,
            "api": False,
            "model": False,
            "capture_approved": False,
            "used_for_decision": False,
        },
        "capture authority contract",
    )
    anchor = value.get("source_anchor")
    _require(_hex40(anchor), "source anchor가 등록되지 않았다")
    if verify_anchor:
        _require(_hex40(execution_anchor), "execution anchor 형식")
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                cast(str, anchor),
                cast(str, execution_anchor),
            ],
            check=False,
        )
        _require(
            ancestry.returncode == 0, "source anchor가 execution anchor의 조상이 아니다"
        )
        _verify_source_anchor(repo, cast(str, anchor), value["source_files_lf"])
        V1._verify_protocol_blob(
            path, repo, cast(str, anchor), cast(str, execution_anchor)
        )
    return value


def verify_protocol(
    path: Path,
    index: Path,
    lab: Path,
    *,
    execution_anchor: str | None = None,
    verify_anchor: bool = True,
    repo: Path = ROOT,
) -> dict[str, Any]:
    return validate_protocol(
        path,
        index,
        lab,
        execution_anchor=execution_anchor,
        verify_anchor=verify_anchor,
        repo=repo,
    )


def build_job(
    protocol_path: Path,
    index_path: Path,
    lab: Path,
    stage: Path,
    *,
    mode: str,
    execution_anchor: str,
    verify_anchor: bool = True,
) -> dict[str, Any]:
    protocol = validate_protocol(
        protocol_path,
        index_path,
        lab,
        execution_anchor=execution_anchor,
        verify_anchor=verify_anchor,
    )
    source_id = cast(str, protocol["source_input"]["source_id"])
    context = _source_context(index_path, source_id, lab)
    _require(
        protocol["status"] != "hold_unsupported",
        "F5 post-create friction lifecycle은 HOLD이며 capture를 실행할 수 없다",
    )
    _require(mode in {"smoke", "full"} and not stage.exists(), "mode/fresh stage")
    _require(
        not (protocol["private"]["arm"] == "F2" and mode != "full"),
        "F2 playback은 actual method sidecar를 남기는 full mode만 허용한다",
    )
    sidecar = V2.sibling_sidecar(stage)
    _require(
        not sidecar.exists() and not sidecar.is_symlink(),
        "sidecar sibling은 fresh여야 한다",
    )
    stage.mkdir(parents=True)
    frames = 1 if mode == "smoke" else V1.FULL_FRAMES
    checkpoint = protocol["checkpoint"]
    job = {
        "schema": V1.JOB_SCHEMA,
        "mode": mode,
        "frames": frames,
        "fps": V1.FPS,
        "protocol": V1._ref(protocol_path),
        "source_anchor": protocol["source_anchor"],
        "execution_anchor": execution_anchor,
        "checkpoint_path": context["checkpoint"]["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_logical_id": checkpoint["logical_id"],
        "source": {
            "commit": protocol["source_anchor"],
            "files_lf": dict(protocol["source_files_lf"]),
        },
        "output_directory": str((stage / "capture").resolve()),
        "sidecar_directory": str(sidecar),
        "job_path": str((stage / "job.json").resolve()),
        "source_id": f"lineage-{checkpoint['logical_id']}-{mode}-v1",
        "source_input": deepcopy(protocol["source_input"]),
        "source_index": deepcopy(context["index"]),
        "source_result": deepcopy(context["source_result"]),
        "source_lab": str(Path(lab).resolve()),
        "playback": deepcopy(protocol["playback"]),
    }
    V1._write_new(stage / "job.json", job)
    return {**job, "job_sha256": V1._sha(stage / "job.json")}


def _validated_job(job_path: Path) -> dict[str, Any]:
    value = V3._validated_job(job_path)
    protocol_ref = value.get("protocol")
    _require(isinstance(protocol_ref, Mapping), "job protocol ref")
    checked_protocol_ref = cast(Mapping[str, Any], protocol_ref)
    protocol_path = Path(cast(str, checked_protocol_ref["path"]))
    _require(
        V1._sha(protocol_path) == checked_protocol_ref.get("sha256"), "job protocol SHA"
    )
    protocol = V1._read(protocol_path)
    _require(protocol.get("schema") == SCHEMA, "job lineage protocol")
    source_id = value["checkpoint_logical_id"]
    source_index = _checked_ref(value.get("source_index"), "job source index")
    source_result = _checked_ref(value.get("source_result"), "job source result")
    source_lab_value = value.get("source_lab")
    _require(isinstance(source_lab_value, str), "job source lab")
    source_lab = Path(cast(str, source_lab_value)).resolve()
    context = _source_context(Path(source_index["path"]), source_id, source_lab)
    _require(
        protocol["source_input"]["source_id"] == source_id
        and protocol["checkpoint"]["sha256"] == value["checkpoint_sha256"]
        and protocol["source_input"]["index"]
        == {"logical_id": INDEX_LOGICAL_ID, "sha256": source_index["sha256"]}
        and protocol["source_result"]
        == {"logical_id": SOURCE_RESULT_LOGICAL_ID, "sha256": source_result["sha256"]},
        "job/protocol lineage binding",
    )
    _require(
        Path(cast(Path | str, value.get("checkpoint_path"))).resolve()
        == Path(context["checkpoint"]["path"]).resolve(),
        "job checkpoint resolution",
    )
    _require(value.get("source_index") == context["index"], "job index resolution")
    _require(
        value.get("source_result") == context["source_result"],
        "job source result resolution",
    )
    _require(
        value.get("source_input") == protocol["source_input"]
        and value.get("playback") == protocol["playback"],
        "job public descriptor binding",
    )
    value["protocol"] = {
        **protocol,
        "source_sha256_lf": protocol["source_files_lf"],
        "playback_registrations": (
            {source_id: protocol["playback"]["registration"]}
            if protocol["private"]["arm"] == "F2"
            else {}
        ),
    }
    value["repo"] = ROOT
    return value


def _full_worker(
    validated: Mapping[str, Any],
    app: Any,
    modules: Mapping[str, Any],
    report_path: Path,
) -> Mapping[str, Any]:
    worker = modules["worker"]
    arm = validated["protocol"]["private"]["arm"]
    maker = modules["render"].render_sync_capture_factory(
        modules["capture"].make_capture_factory
    )
    selected_worker: Callable[..., Any] = worker.run_policy_record
    if arm == "F2":
        registration = validated["protocol"]["playback"]["registration"]

        def playback_worker(
            value: Mapping[str, Any], worker_app: Any, **kwargs: Any
        ) -> Any:
            return worker.run_policy_record(
                value,
                worker_app,
                run_real=PLAYBACK.make_run_real(registration),
                **kwargs,
            )

        selected_worker = playback_worker

    selected = modules["sidecar"].with_sidecar_worker(
        Path(validated["sidecar_directory"]),
        source_id=validated["source_id"],
        expected_checkpoint_sha256=validated["checkpoint_sha256"],
        error_policy=modules["sidecar"].ABORT,
        expected_frames=validated["frames"],
        run_worker=selected_worker,
        capture_factory_maker=maker,
        reader_factory=modules["sidecar"]._default_reader_factory,
    )
    return selected(
        validated, app, seal_report=lambda result: V1._write_new(report_path, result)
    )


def sdk_child(job_path: Path) -> int:
    validated = _validated_job(job_path)
    if validated["mode"] != "full":
        sidecar2 = importlib.import_module("record_agentic_state_sidecar_v2")
        return int(
            sidecar2.run_state_dry_run(
                checkpoint_path=Path(validated["checkpoint_path"]),
                checkpoint_sha256=validated["checkpoint_sha256"],
                source_id=validated["source_id"],
                out_dir=Path(validated["output_directory"]),
                frames=1,
            )
        )
    modules = {
        name: importlib.import_module(module)
        for name, module in {
            "worker": "vision_pull_record_worker_v3",
            "capture": "vision_pull_record_capture_v2",
            "render": "run_agentic_play_record_v4",
            "sidecar": "record_agentic_state_sidecar_v1",
            "shutdown": "vision_shutdown_policy_v1",
        }.items()
    }
    stage = Path(job_path).resolve().parent
    app = V1._full_app(validated, stage, modules["shutdown"])
    report_path = Path(validated["output_directory"]) / modules["worker"].REPORT_NAME

    def run_worker() -> Mapping[str, Any]:
        return _full_worker(validated, app, modules, report_path)

    exit_code, _ = V1.run_full_terminal(
        validated,
        app,
        run_worker,
        before_close=lambda result: V1._seal_preclose_proof(validated, result, modules),
    )
    return exit_code


def _read_playback_proof(stage: Path, validated: Mapping[str, Any]) -> dict[str, Any]:
    arm = validated["protocol"]["private"]["arm"]
    receipt_path = stage / "playback-receipt.json"
    if arm != "F2":
        _require(
            not receipt_path.exists(), "no-injection source에 playback receipt가 생겼다"
        )
        return {"kind": "none", "receipt": None, "all_go2_joints_verified": None}
    receipt = V1._read(receipt_path)
    registration = validated["protocol"]["playback"]["registration"]
    installation = receipt.get("installation", {})
    _require(
        receipt.get("status") == "completed"
        and receipt.get("registration") == registration
        and receipt.get("restore_completed") is True
        and installation.get("fault_id") == "F2"
        and installation.get("class_method_changed") is True,
        "F2 playback 설치·복원 receipt",
    )
    after = installation.get("after")
    _require(isinstance(after, dict), "F2 actual method receipt")
    actual_method = f"{after.get('module')}.{after.get('qualname')}"
    sidecar = importlib.import_module("record_agentic_state_sidecar_v1")
    manifest, _ = sidecar.load_sidecar(Path(validated["sidecar_directory"]))
    joints = manifest.get("joint_names")
    actuators = manifest.get("actuators")
    _require(
        isinstance(joints, list) and len(joints) == 12 and len(set(joints)) == 12,
        "F2 sidecar Go2 joint 12개 결합",
    )
    _require(
        isinstance(actuators, Mapping) and bool(actuators),
        "F2 actuator sidecar metadata",
    )
    groups = [group for group in actuators.values() if isinstance(group, Mapping)]
    _require(
        bool(groups)
        and all(group.get("clip_effort_function") == actual_method for group in groups),
        "F2 sidecar actual clip method",
    )
    _require(
        any(group.get("joint_names") == joints for group in groups),
        "F2 torque curve가 Go2 전체 12 joint에 결합되지 않았다",
    )
    return {
        "kind": "class_pre_env",
        "receipt": V1._ref(receipt_path),
        "actual_clip_effort_function": actual_method,
        "joint_count": 12,
        "all_go2_joints_verified": True,
        "restore_completed": True,
    }


def _playback_proof(
    stage: Path, validated: Mapping[str, Any], *, seal: bool = True
) -> dict[str, Any]:
    proof = _read_playback_proof(stage, validated)
    if seal and proof["kind"] == "class_pre_env":
        V1._write_new(stage / "playback-proof.json", proof)
    return proof


def verify_result(
    stage: Path, job: Mapping[str, Any], execution: Mapping[str, Any]
) -> dict[str, Any]:
    base = V3.verify_result(stage, job, execution)
    validated = _validated_job(stage / "job.json")
    playback = _playback_proof(stage, validated)
    value = {
        "schema": FINAL_SCHEMA,
        "status": "technical_complete",
        "mode": "full",
        "frames": V1.FULL_FRAMES,
        "fps": V1.FPS,
        "source_id": validated["checkpoint_logical_id"],
        "checkpoint_sha256": validated["checkpoint_sha256"],
        "source_anchor": validated["source_anchor"],
        "base_v3": deepcopy(base),
        "playback": playback,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
    }
    V1._write_new(stage / "parent-finalized-lineage-result-v1.json", value)
    return value


def _checked_ref(value: object, label: str) -> dict[str, str]:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "sha256"}, f"{label} ref"
    )
    ref = cast(Mapping[str, Any], value)
    path = Path(cast(str, ref["path"])).resolve()
    _require(
        path.is_file() and not path.is_symlink() and V1._sha(path) == ref.get("sha256"),
        f"{label} SHA",
    )
    return {"path": str(path), "sha256": cast(str, ref["sha256"])}


def export_artifact(
    protocol_path: Path,
    index_path: Path,
    lab: Path,
    stage: Path | None,
    *,
    execution_anchor: str,
    verify_anchor: bool = True,
) -> dict[str, Any]:
    """pair provider가 소비할 기술 증거만 내보내고 private arm은 제외한다."""
    protocol = validate_protocol(
        protocol_path,
        index_path,
        lab,
        execution_anchor=execution_anchor,
        verify_anchor=verify_anchor,
    )
    common = {
        "schema": ARTIFACT_SCHEMA,
        "source_id": protocol["source_input"]["source_id"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "source_binding": deepcopy(protocol["source_input"]["binding"]),
        "fps": V1.FPS,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
    }
    if protocol["status"] == "hold_unsupported":
        _require(stage is None, "HOLD source에는 capture stage를 결합할 수 없다")
        return {
            **common,
            "status": "hold_unsupported",
            "frames": None,
            "reason": "post_create_material_restore_lifecycle_unverified",
            "replacement_capture": False,
            "proofs": None,
            "v01_specs": [],
            "quality": {
                "status": "unavailable_hold",
                "receipt": None,
                "capture_approved": False,
                "model_ready": False,
                "used_for_decision": False,
            },
        }
    _require(
        stage is not None and Path(stage).is_dir(), "available capture stage가 없다"
    )
    root = Path(cast(Path, stage)).resolve()
    job_ref = V1._ref(root / "job.json")
    execution_ref = V1._ref(root / "execution-result-lineage-v1.json")
    parent_ref = V1._ref(root / "parent-finalized-lineage-result-v1.json")
    job = V1._read(root / "job.json")
    execution = V1._read(root / "execution-result-lineage-v1.json")
    parent = V1._read(root / "parent-finalized-lineage-result-v1.json")
    validated = _validated_job(root / "job.json")
    playback = _playback_proof(root, validated, seal=False)
    protocol_ref = V1._ref(protocol_path)
    source_id = common["source_id"]
    checkpoint_sha = common["checkpoint_sha256"]
    _require(
        job.get("protocol") == protocol_ref
        and job.get("checkpoint_logical_id") == source_id
        and job.get("checkpoint_sha256") == checkpoint_sha
        and execution.get("schema") == RESULT_SCHEMA
        and execution.get("status") == "technical_complete"
        and execution.get("mode") == "full"
        and execution.get("protocol") == protocol_ref
        and execution.get("source_id") == source_id
        and execution.get("checkpoint_sha256") == checkpoint_sha,
        "capture job/execution source binding",
    )
    base = parent.get("base_v3")
    _require(
        parent.get("schema") == FINAL_SCHEMA
        and parent.get("status") == "technical_complete"
        and parent.get("mode") == "full"
        and parent.get("frames") == V1.FULL_FRAMES
        and parent.get("fps") == V1.FPS
        and parent.get("source_id") == source_id
        and parent.get("checkpoint_sha256") == checkpoint_sha
        and parent.get("capture_approved") is False
        and parent.get("model_ready") is False
        and parent.get("used_for_decision") is False
        and parent.get("playback") == playback
        and isinstance(base, Mapping),
        "lineage parent technical result",
    )
    checked_base = cast(Mapping[str, Any], base)
    proofs = checked_base.get("proofs")
    required = {
        "worker_report",
        "native_manifest",
        "raw_trace",
        "sidecar_manifest",
        "sidecar_records",
        "render_sync",
        "same_pts",
        "shutdown",
        "app_close_entered",
    }
    _require(
        checked_base.get("schema") == V1.RESULT_SCHEMA
        and checked_base.get("status") == "technical_complete"
        and checked_base.get("frames") == V1.FULL_FRAMES
        and checked_base.get("fps") == V1.FPS
        and checked_base.get("same_pts") is True
        and isinstance(proofs, Mapping)
        and required <= set(proofs),
        "base v3 full625/50 proof set",
    )
    checked_proofs = {
        name: _checked_ref(ref, f"proof {name}")
        for name, ref in cast(Mapping[str, object], proofs).items()
    }
    raw_trace = Path(checked_proofs["raw_trace"]["path"])
    specs = [raw_trace.parent / f"window_{number}_v01_spec.json" for number in (1, 2)]
    _require(
        all(path.is_file() and not path.is_symlink() for path in specs),
        "V01 spec 2개가 없다",
    )
    worker = V1._read(Path(checked_proofs["worker_report"]["path"]))
    artifacts = worker.get("artifacts")
    _require(isinstance(artifacts, Mapping), "worker artifact descriptors")
    descriptors = cast(Mapping[str, Mapping[str, Any]], artifacts)
    for number, spec in enumerate(specs, start=1):
        descriptor = descriptors.get(f"spec_{number}")
        _require(
            isinstance(descriptor, Mapping)
            and descriptor.get("sha256") == V1._sha(spec),
            f"V01 spec {number} worker binding",
        )
    return {
        **common,
        "status": "available",
        "frames": V1.FULL_FRAMES,
        "reason": None,
        "replacement_capture": False,
        "protocol": protocol_ref,
        "job": job_ref,
        "execution": execution_ref,
        "parent_result": parent_ref,
        "proofs": checked_proofs,
        "v01_specs": [V1._ref(path) for path in specs],
        "quality": {
            "status": "technical_complete_not_approved",
            "receipt": parent_ref,
            "capture_approved": False,
            "model_ready": False,
            "used_for_decision": False,
        },
    }


def verify_artifact(
    value: Mapping[str, Any],
    protocol_path: Path,
    index_path: Path,
    lab: Path,
    *,
    execution_anchor: str,
) -> dict[str, Any]:
    """export를 원천에서 재계산하고 exact object equality로 확인한다."""
    _require(
        isinstance(value, Mapping) and value.get("schema") == ARTIFACT_SCHEMA,
        "lineage capture artifact schema",
    )
    status = value.get("status")
    _require(
        status in {"available", "hold_unsupported"}, "lineage capture artifact status"
    )
    stage: Path | None = None
    if status == "available":
        job = _checked_ref(value.get("job"), "artifact job")
        stage = Path(job["path"]).parent
    expected = export_artifact(
        protocol_path,
        index_path,
        lab,
        stage,
        execution_anchor=execution_anchor,
        verify_anchor=True,
    )
    _require(dict(value) == expected, "lineage capture artifact가 원천 재계산과 다르다")
    return deepcopy(expected)


def execute(
    protocol_path: Path,
    index_path: Path,
    execution_anchor: str,
    lab: Path,
    stage: Path,
    isaac_python: Path,
    *,
    mode: str,
    runner: Callable[[Path, list[str], Path, float], Mapping[str, Any]] | None = None,
    lease_reader: Callable[[Path | None], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _require(Path(isaac_python).is_file(), "Isaac Python launcher")
    run_owned = (
        runner or importlib.import_module("run_runtime_readback_smoke").run_owned
    )
    read_lease = (
        lease_reader
        or importlib.import_module("lineage_lease_context_v2").lease_context
    )
    before = V1._lease_identity(
        dict(read_lease(stage.parent / f"{stage.name}-lease-before"))
    )
    job = build_job(
        protocol_path,
        index_path,
        lab,
        stage,
        mode=mode,
        execution_anchor=execution_anchor,
    )
    owned = stage / "owned"
    owned.mkdir()
    command = [
        str(isaac_python),
        str(Path(__file__).resolve()),
        "sdk-child",
        str(stage / "job.json"),
        *V1.KIT_ARGS,
    ]
    execution = dict(
        run_owned(owned, command, ROOT, 600.0 if mode == "smoke" else 2400.0)
    )
    failure = Path(job["output_directory"]) / "failure.json"
    if failure.is_file():
        raw = V1._read(failure)
        raise RuntimeError(
            f"owned child failure: {raw.get('error_type')}: {raw.get('error_message')}"
        )
    V1.verify_owned(execution)
    after = V1._lease_identity(dict(read_lease(stage / "lease-after")))
    _require(after == before, "실행 중 공용 GPU lease가 바뀌었다")
    if mode == "full":
        child = verify_result(stage, job, execution)
    else:
        sidecar = importlib.import_module("record_agentic_state_sidecar_v2")
        child = sidecar.verify_shutdown_evidence(
            Path(job["output_directory"]),
            checkpoint_sha256=job["checkpoint_sha256"],
            source_id=job["source_id"],
            frames=1,
        )
    result = {
        "schema": RESULT_SCHEMA,
        "status": "technical_complete",
        "mode": mode,
        "protocol": V1._ref(protocol_path),
        "source_id": job["checkpoint_logical_id"],
        "source_anchor": job["source_anchor"],
        "execution_anchor": execution_anchor,
        "checkpoint_sha256": job["checkpoint_sha256"],
        "layout": {"stage": str(stage.resolve()), "sidecar": job["sidecar_directory"]},
        "lease": {"owner": before[0], "lease_id": before[1]},
        "owned": execution,
        "child": child,
        "capture_approved": False,
        "model_ready": False,
        "used_for_decision": False,
    }
    V1._write_new(stage / "execution-result-lineage-v1.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-protocol")
    for name in ("index", "lab", "out"):
        prepare.add_argument(f"--{name}", required=True, type=Path)
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument("--source-anchor")
    verify = sub.add_parser("verify-protocol")
    for name in ("protocol", "index", "lab"):
        verify.add_argument(f"--{name}", required=True, type=Path)
    verify.add_argument("--anchor", required=True)
    run = sub.add_parser("execute")
    for name in ("protocol", "index", "lab", "stage", "isaac-python"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--anchor", required=True)
    run.add_argument("--mode", required=True, choices=("smoke", "full"))
    child = sub.add_parser("sdk-child")
    child.add_argument("job", type=Path)
    args, extras = parser.parse_known_args(argv)
    _require(
        extras == (V1.KIT_ARGS if args.command == "sdk-child" else []),
        "허용되지 않은 CLI 인자",
    )
    if args.command == "prepare-protocol":
        value = prepare_protocol(
            args.index,
            args.source_id,
            args.lab,
            args.out,
            source_anchor=args.source_anchor,
        )
    elif args.command == "verify-protocol":
        value = verify_protocol(
            args.protocol, args.index, args.lab, execution_anchor=args.anchor
        )
    elif args.command == "execute":
        value = execute(
            args.protocol,
            args.index,
            args.anchor,
            args.lab,
            args.stage,
            args.isaac_python,
            mode=args.mode,
        )
    else:
        return sdk_child(args.job)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
