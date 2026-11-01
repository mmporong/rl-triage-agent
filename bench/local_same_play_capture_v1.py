"""Windows 로컬 same-Play RGB+state 녹화의 등록·검증 경계.

모듈 import 시 Isaac/torch를 불러오지 않는다. 실제 SDK import는 ``sdk-child`` 뒤에만 수행한다.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

SCHEMA = "local_same_play_capture_protocol_v1"
JOB_SCHEMA = "local_same_play_capture_job_v1"
RESULT_SCHEMA = "local_same_play_capture_result_v1"
INSPECTION_SCHEMA = "local_same_play_capture_actual_input_inspection_v1"
OWNER = "rl-triage-lineage-sources-v1"
TASK = "Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
FPS = 50
FULL_FRAMES = 625
KIT_ARGS = ["--/app/vulkan=false", "--/app/window/hideUi=true"]
LOCAL_SOURCE_FILES = (
    "bench/local_same_play_capture_v1.py",
    "tests/test_local_same_play_capture_v1.py",
    "bench/vision_pull_record_worker_v3.py",
    "bench/vision_pull_record_capture_v2.py",
    "bench/run_agentic_play_record_v4.py",
    "bench/vision_play_worker_v1.py",
    "bench/vision_shutdown_policy_v1.py",
    "bench/record_agentic_state_sidecar_v1.py",
    "bench/record_agentic_state_sidecar_v2.py",
    "bench/run_runtime_readback_smoke.py",
    "evals/lineage_lease_context_v2.py",
    "evals/lineage_source_batch_v1.py",
)
RUNTIME_PINS = (
    "source/isaaclab/isaaclab/app/app_launcher.py",
    "source/isaaclab/isaaclab/sim/simulation_context.py",
    "source/isaaclab/isaaclab/envs/manager_based_env.py",
    "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/flat_env_cfg.py",
    "source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
)


def source_files() -> tuple[str, ...]:
    """기존 pull-record 등록 closure에 이 로컬 adapter 경계를 더한다."""
    registered = importlib.import_module("vision_pull_record_job_v3").REQUIRED_SOURCE_FILES
    seeds = set(registered) | set(LOCAL_SOURCE_FILES) | {
        "bench/vision_pull_record_worker_v2.py",
        "bench/vision_pull_record_capture_v1.py",
        "bench/record_vision_evidence_v1.py",
        "bench/run_vision_prepared_record_v1.py",
        "evals/lineage_lease_context_v2.py",
        "tests/test_lineage_lease_context_v2.py",
    }
    pending = list(seeds)
    while pending:
        relative = pending.pop()
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
        for name in names:
            module_path = name.replace(".", "/")
            candidates = [f"bench/{module_path}.py", f"evals/{module_path}.py",
                          f"src/{module_path}.py", f"src/{module_path}/__init__.py"]
            for candidate in candidates:
                if candidate not in seeds and (ROOT / candidate).is_file():
                    seeds.add(candidate)
                    pending.append(candidate)
    return tuple(sorted(seeds))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file() and not path.is_symlink(), f"원문 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _verify_protocol_blob(path: Path, repo: Path, source_anchor: str,
                          execution_anchor: str) -> None:
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", source_anchor, execution_anchor],
        check=False, capture_output=True)
    _require(ancestry.returncode == 0, "source anchor가 execution anchor의 조상이 아니다")
    head_ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", execution_anchor, "HEAD"],
        check=False, capture_output=True)
    _require(head_ancestry.returncode == 0, "execution anchor가 HEAD의 조상이 아니다")
    try:
        relative_protocol = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 repository 안에 있어야 한다") from error
    protocol_blob = subprocess.run(
        ["git", "-C", str(repo), "show", f"{execution_anchor}:{relative_protocol}"],
        check=True, capture_output=True).stdout
    current_lf = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    _require(protocol_blob.replace(b"\r\n", b"\n").replace(b"\r", b"\n") == current_lf,
             "execution anchor protocol blob이 현재 파일과 다르다")


def _source_stage(source_batch: Mapping[str, Any], logical_id: str) -> Mapping[str, Any]:
    stages = source_batch.get("stages")
    _require(source_batch.get("schema") == "prospective_diagnosis_batch_result_v1"
             and source_batch.get("status") == "complete" and isinstance(stages, list),
             "source batch가 complete 계약이 아니다")
    matches = [row for row in cast(list[Any], stages)
               if isinstance(row, Mapping) and row.get("stage") == logical_id]
    _require(len(matches) == 1 and isinstance(matches[0].get("checkpoint"), Mapping),
             "요청한 source checkpoint가 하나가 아니다")
    return matches[0]


def _training_runtime_pins(row: Mapping[str, Any], lab: Path) -> dict[str, str]:
    result_ref = row.get("result")
    _require(isinstance(result_ref, Mapping) and _sha(Path(result_ref["path"])) == result_ref.get("sha256"),
             "source training result SHA")
    checked_result = cast(Mapping[str, Any], result_ref)
    training_ref = _read(Path(checked_result["path"])).get("training")
    _require(isinstance(training_ref, Mapping) and _sha(Path(training_ref["path"])) == training_ref.get("sha256"),
             "source training receipt SHA")
    checked_training = cast(Mapping[str, Any], training_ref)
    manifest_ref = _read(Path(checked_training["path"])).get("manifest")
    _require(isinstance(manifest_ref, Mapping) and _sha(Path(manifest_ref["path"])) == manifest_ref.get("sha256"),
             "source training manifest SHA")
    manifest = _read(Path(cast(Mapping[str, Any], manifest_ref)["path"]))
    pins = manifest.get("runtime_source_sha256_lf")
    _require(isinstance(pins, dict) and bool(pins), "source training runtime pins")
    checked = cast(dict[str, str], pins)
    for relative, expected in checked.items():
        _require(_sha(lab / relative, lf=True) == expected, f"training runtime pin changed: {relative}")
    return dict(sorted(checked.items()))


def inspect_actual_none(source_batch_path: Path, lab: Path, out: Path,
                        *, logical_id: str = "s2320-none") -> dict[str, Any]:
    """기존 NONE CP를 실행 없이 읽어 로컬 입력 형태만 검증한다."""
    source_batch_path, lab, out = map(Path, (source_batch_path, lab, out))
    source = _read(source_batch_path)
    row = _source_stage(source, logical_id)
    checkpoint = row["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve()
    _require(checkpoint_path.is_file() and _sha(checkpoint_path) == checkpoint.get("sha256"),
             "NONE checkpoint 원문 SHA가 다르다")
    _require(checkpoint.get("iteration") == 299 and checkpoint.get("size_bytes") == checkpoint_path.stat().st_size,
             "NONE checkpoint 메타데이터가 다르다")
    pins = {}
    for relative in RUNTIME_PINS:
        path = lab / relative
        _require(path.is_file() and not path.is_symlink(), f"Isaac Lab runtime pin이 없다: {relative}")
        pins[relative] = _sha(path)
    training_pins = _training_runtime_pins(row, lab)
    receipt = {
        "schema": INSPECTION_SCHEMA,
        "status": "cpu_input_shape_verified",
        "source_batch": _ref(source_batch_path),
        "logical_id": logical_id,
        "checkpoint": _ref(checkpoint_path),
        "checkpoint_iteration": 299,
        "runtime_pins": pins,
        "training_runtime_pins": training_pins,
        "sdk_executed": False,
        "gpu_actions": 0,
        "model_use_approved": False,
        "behavior_label_created": False,
    }
    _write_new(out, receipt)
    return receipt


def prepare_protocol(inspection_path: Path, out: Path, *, anchor: str | None = None) -> dict[str, Any]:
    """CPU inspection에서 등록 전 후보 또는 commit 뒤 실행 protocol을 만든다."""
    inspection = _read(inspection_path)
    _require(inspection.get("schema") == INSPECTION_SCHEMA
             and inspection.get("status") == "cpu_input_shape_verified"
             and inspection.get("sdk_executed") is False, "actual input inspection")
    if anchor is not None:
        _require(len(anchor) == 40 and all(char in "0123456789abcdef" for char in anchor),
                 "source anchor")
    value = {
        "schema": SCHEMA,
        "status": "registered_before_local_gpu" if anchor is not None else "candidate_not_registered",
        "source_anchor": anchor,
        "source_files_lf": {name: _sha(ROOT / name, lf=True) for name in source_files()},
        "runtime_pins": dict(inspection["runtime_pins"]),
        "training_runtime_pins": dict(inspection["training_runtime_pins"]),
        "source_batch": {"logical_id": "prospective_source_batch_20261101_v1",
                         "sha256": inspection["source_batch"]["sha256"]},
        "checkpoint": {"sha256": inspection["checkpoint"]["sha256"],
                       "logical_id": inspection["logical_id"], "iteration": 299},
        "platform": {"os": "windows", "isaacsim": "4.5.0", "gpu": "RTX 3060"},
        "gpu": {"lease_helper": "Use-GpuLease.ps1", "owner": OWNER,
                "cloud_lease": False, "exclusive": True},
        "play": {"task": TASK, "num_envs": 1, "fps": FPS, "full_frames": FULL_FRAMES,
                 "smoke_frames": 1, "camera_profile": "same_play_fixed_v1", "kit_args": KIT_ARGS,
                 "new_playback_injection": False},
        "shutdown": {"smoke": "record_agentic_state_sidecar_v2",
                     "full": "vision_shutdown_policy_v1", "terminal_count": 1},
    }
    _write_new(out, value)
    return value


def validate_protocol(path: Path, lab: Path, inspection_path: Path, *, repo: Path = ROOT,
                      execution_anchor: str | None = None,
                      verify_anchor: bool = True) -> dict[str, Any]:
    value = _read(path)
    expected = {"schema", "status", "source_anchor", "source_files_lf", "runtime_pins",
                "training_runtime_pins",
                "source_batch", "checkpoint", "platform", "gpu", "play", "shutdown"}
    _require(set(value) == expected and value.get("schema") == SCHEMA
             and value.get("status") == "registered_before_local_gpu", "local protocol schema/status")
    anchor = value.get("source_anchor")
    _require(isinstance(anchor, str) and len(anchor) == 40
             and all(char in "0123456789abcdef" for char in anchor), "source anchor")
    _require(value.get("platform") == {"os": "windows", "isaacsim": "4.5.0", "gpu": "RTX 3060"},
             "local platform contract")
    _require(value.get("gpu") == {"lease_helper": "Use-GpuLease.ps1", "owner": OWNER,
                                  "cloud_lease": False, "exclusive": True}, "GPU lease contract")
    play = value.get("play")
    _require(play == {"task": TASK, "num_envs": 1, "fps": FPS, "full_frames": FULL_FRAMES,
                      "smoke_frames": 1, "camera_profile": "same_play_fixed_v1",
                      "kit_args": KIT_ARGS, "new_playback_injection": False}, "same-Play contract")
    _require(value.get("shutdown") == {"smoke": "record_agentic_state_sidecar_v2",
                                       "full": "vision_shutdown_policy_v1",
                                       "terminal_count": 1}, "shutdown boundary")
    source_map = value.get("source_files_lf")
    _require(isinstance(source_map, dict) and set(source_map) == set(source_files()), "source closure")
    checked_sources = cast(dict[str, str], source_map)
    for relative, expected_sha in checked_sources.items():
        source_path = repo / relative
        _require(source_path.is_file() and _sha(source_path, lf=True) == expected_sha,
                 f"source LF SHA changed: {relative}")
        if verify_anchor:
            shown = subprocess.run(["git", "-C", str(repo), "show", f"{anchor}:{relative}"],
                                   check=True, capture_output=True).stdout
            _require(hashlib.sha256(shown.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
                     == expected_sha, f"anchor blob changed: {relative}")
    if verify_anchor:
        _require(isinstance(execution_anchor, str) and len(execution_anchor) == 40
                 and all(char in "0123456789abcdef" for char in execution_anchor),
                 "execution anchor")
        _verify_protocol_blob(path, repo, cast(str, anchor), cast(str, execution_anchor))
    pins = value.get("runtime_pins")
    _require(isinstance(pins, dict) and set(pins) == set(RUNTIME_PINS), "runtime pin set")
    for relative, expected_sha in cast(dict[str, str], pins).items():
        _require(_sha(lab / relative) == expected_sha, f"runtime pin changed: {relative}")
    training_pins = value.get("training_runtime_pins")
    _require(isinstance(training_pins, dict) and bool(training_pins), "training runtime pin set")
    for relative, expected_sha in cast(dict[str, str], training_pins).items():
        _require(_sha(lab / relative, lf=True) == expected_sha, f"training runtime pin changed: {relative}")
    inspection = _read(inspection_path)
    _require(inspection.get("schema") == INSPECTION_SCHEMA
             and inspection.get("status") == "cpu_input_shape_verified", "actual input inspection")
    _require(pins == inspection.get("runtime_pins")
             and training_pins == inspection.get("training_runtime_pins"),
             "protocol runtime pins가 actual training inspection과 다르다")
    batch_ref, cp_ref = value["source_batch"], value["checkpoint"]
    raw_batch, raw_cp = inspection.get("source_batch"), inspection.get("checkpoint")
    _require(isinstance(raw_batch, Mapping) and isinstance(raw_cp, Mapping)
             and batch_ref == {"logical_id": "prospective_source_batch_20261101_v1",
                               "sha256": raw_batch.get("sha256")}
             and _sha(Path(raw_batch["path"])) == raw_batch.get("sha256"), "source batch SHA")
    checked_batch = cast(Mapping[str, Any], raw_batch)
    checked_cp = cast(Mapping[str, Any], raw_cp)
    row = _source_stage(_read(Path(checked_batch["path"])), cp_ref.get("logical_id"))
    _require(cp_ref == {"sha256": checked_cp.get("sha256"), "logical_id": inspection.get("logical_id"),
                        "iteration": 299}
             and row["checkpoint"]["sha256"] == cp_ref.get("sha256")
             and _sha(Path(checked_cp["path"])) == cp_ref.get("sha256"), "checkpoint/source binding")
    return value


def build_job(protocol_path: Path, lab: Path, inspection_path: Path, stage: Path, *, mode: str,
              execution_anchor: str) -> dict[str, Any]:
    _require(mode in {"smoke", "full"}, "mode")
    protocol = validate_protocol(protocol_path, lab, inspection_path, execution_anchor=execution_anchor)
    inspection = _read(inspection_path)
    _require(not stage.exists(), "stage는 fresh 경로여야 한다")
    stage.mkdir(parents=True)
    frames = 1 if mode == "smoke" else FULL_FRAMES
    job = {"schema": JOB_SCHEMA, "mode": mode, "frames": frames, "fps": FPS,
           "protocol": _ref(protocol_path), "source_anchor": protocol["source_anchor"],
           "execution_anchor": execution_anchor,
           "checkpoint_path": inspection["checkpoint"]["path"],
           "checkpoint_sha256": protocol["checkpoint"]["sha256"],
           "checkpoint_logical_id": protocol["checkpoint"]["logical_id"],
           "source": {"commit": protocol["source_anchor"],
                      "files_lf": dict(protocol["source_files_lf"])},
           "output_directory": str((stage / "capture").resolve()),
           "sidecar_directory": str((stage / "sidecar").resolve()),
           "job_path": str((stage / "job.json").resolve()),
           "source_id": f"local-{protocol['checkpoint']['logical_id']}-{mode}"}
    _write_new(stage / "job.json", job)
    job["job_sha256"] = _sha(stage / "job.json")
    return job


def compose_full_worker(validated: Mapping[str, Any], app: Any, *, worker: Any, capture: Any,
                        render: Any, sidecar: Any, seal_report: Callable[[Mapping[str, Any]], None],
                        reader_factory: Callable[..., Any]) -> dict[str, Any]:
    """기존 worker→render-sync→sidecar 순서를 한 곳에서 결합한다."""
    maker = render.render_sync_capture_factory(capture.make_capture_factory)
    selected = sidecar.with_sidecar_worker(
        Path(validated["sidecar_directory"]), source_id=validated["source_id"],
        expected_checkpoint_sha256=validated["checkpoint_sha256"], error_policy=sidecar.ABORT,
        expected_frames=validated["frames"], run_worker=worker.run_policy_record,
        capture_factory_maker=maker, reader_factory=reader_factory,
    )
    return selected(validated, app, seal_report=seal_report)


def verify_owned(execution: Mapping[str, Any]) -> None:
    _require(execution.get("job_assigned_before_release") is True
             and execution.get("natural_exit") is True and execution.get("timed_out") is False
             and execution.get("exit_code") == 0 and execution.get("cleanup_verified") is True
             and execution.get("error_type") is None, "owned child exit/cleanup contract")


def _lease_identity(value: Mapping[str, Any]) -> tuple[str, str]:
    _require(value.get("owner") == OWNER and value.get("state") == "running", "공용 GPU lease owner/state")
    identifier = value.get("leaseId")
    _require(isinstance(identifier, str) and bool(identifier), "공용 GPU lease ID")
    return OWNER, cast(str, identifier)


def execute(protocol_path: Path, execution_anchor: str, lab: Path, inspection_path: Path, stage: Path,
            isaac_python: Path, *, mode: str,
            runner: Callable[[Path, list[str], Path, float], Mapping[str, Any]] | None = None,
            lease_reader: Callable[[Path | None], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """등록 protocol 아래에서 공용 lease와 owned child를 결합한다."""
    protocol_path, lab, inspection_path, stage, isaac_python = map(
        Path, (protocol_path, lab, inspection_path, stage, isaac_python))
    _require(mode in {"smoke", "full"}, "mode")
    _require(isaac_python.is_file(), "Isaac Python launcher")
    run_owned = runner or cast(
        Callable[[Path, list[str], Path, float], Mapping[str, Any]],
        importlib.import_module("run_runtime_readback_smoke").run_owned)
    read_lease = lease_reader or cast(
        Callable[[Path | None], Mapping[str, Any]],
        importlib.import_module("lineage_lease_context_v2").lease_context)
    first_lease = dict(read_lease(stage.parent / f"{stage.name}-lease-before"))
    lease_identity = _lease_identity(first_lease)
    job = build_job(protocol_path, lab, inspection_path, stage, mode=mode,
                    execution_anchor=execution_anchor)
    owned = stage / "owned"
    owned.mkdir()
    command = [str(isaac_python), str(Path(__file__).resolve()), "sdk-child", str(stage / "job.json"),
               *KIT_ARGS]
    execution = dict(run_owned(owned, command, ROOT, 600.0 if mode == "smoke" else 2400.0))
    verify_owned(execution)
    second_lease = dict(read_lease(stage / "lease-after"))
    _require(_lease_identity(second_lease) == lease_identity, "실행 중 공용 GPU lease가 바뀌었다")
    if mode == "full":
        child = verify_result(stage, job, execution)
    else:
        sidecar = importlib.import_module("record_agentic_state_sidecar_v2")
        output = Path(job["output_directory"])
        child = sidecar.verify_shutdown_evidence(
            output, checkpoint_sha256=job["checkpoint_sha256"],
            source_id=job["source_id"], frames=1)
    result = {
        "schema": "local_same_play_capture_execution_v1", "status": "technical_complete",
        "mode": mode, "protocol": _ref(protocol_path), "execution_anchor": execution_anchor,
        "source_anchor": job["source_anchor"], "checkpoint_sha256": job["checkpoint_sha256"],
        "lease": {"owner": lease_identity[0], "lease_id": lease_identity[1]},
        "owned": execution, "child": child, "capture_approved": False,
        "model_ready": False, "used_for_decision": False,
    }
    _write_new(stage / "execution-result.json", result)
    return result


def verify_result(stage: Path, job: Mapping[str, Any], execution: Mapping[str, Any]) -> dict[str, Any]:
    verify_owned(execution)
    validated = _validated_job(stage / "job.json")
    modules = {name: importlib.import_module(module) for name, module in {
        "worker": "vision_pull_record_worker_v3", "sidecar": "record_agentic_state_sidecar_v1",
        "render": "run_agentic_play_record_v4", "shutdown": "vision_shutdown_policy_v1",
    }.items()}
    proof = _verify_preclose_proof(validated, modules)
    terminal = Path(validated["output_directory"]) / "result.json"
    modules["shutdown"].verify(stage / "job.json", validated, terminal)
    entered = _read(stage / "local-app-close-entered.json")
    _require(entered == {"schema": "local_same_play_app_close_v1", "status": "entered",
                         "job_sha256": validated["job_sha256"],
                         "checkpoint_sha256": validated["checkpoint_sha256"]}, "app close entered proof")
    _require(not (stage / "local-app-close-error.json").exists(), "app close error proof")
    returned_path = stage / "local-app-close-returned.json"
    if returned_path.exists():
        returned = _read(returned_path)
        _require(returned.get("schema") == "local_same_play_app_close_v1"
                 and returned.get("status") == "returned"
                 and returned.get("entered_sha256") == _sha(stage / "local-app-close-entered.json"),
                 "app close returned proof")
    final_proofs = {**dict(proof["proofs"]),
                    "shutdown": _ref(stage / modules["shutdown"].SIDECAR),
                    "app_close_entered": _ref(stage / "local-app-close-entered.json")}
    if returned_path.exists():
        final_proofs["app_close_returned"] = _ref(returned_path)
    result = {"schema": RESULT_SCHEMA, "status": "technical_complete", "mode": "full",
              "frames": FULL_FRAMES, "fps": FPS, "checkpoint_sha256": job["checkpoint_sha256"],
              "source_anchor": job["source_anchor"], "proofs": final_proofs, "same_pts": True,
              "app_close_returned": returned_path.exists(), "capture_approved": False,
              "model_ready": False, "used_for_decision": False}
    _write_new(stage / "parent-finalized-result.json", result)
    return result


class FullAppProxy:
    """검증된 full shutdown proxy 호출의 진입·반환·오류를 별도 원문으로 남긴다."""

    def __init__(self, app: Any, stage: Path, validated: Mapping[str, Any]) -> None:
        self._app = app
        self._stage = Path(stage)
        self._validated = validated

    def update(self) -> Any:
        return self._app.update()

    def close(self) -> Any:
        entered = self._stage / "local-app-close-entered.json"
        _write_new(entered, {"schema": "local_same_play_app_close_v1", "status": "entered",
                             "job_sha256": self._validated["job_sha256"],
                             "checkpoint_sha256": self._validated["checkpoint_sha256"]})
        try:
            value = self._app.close()
        except BaseException as error:
            _write_new(self._stage / "local-app-close-error.json", {
                "schema": "local_same_play_app_close_v1", "status": "error",
                "error_type": type(error).__name__, "error_message": str(error)})
            raise
        _write_new(self._stage / "local-app-close-returned.json", {
            "schema": "local_same_play_app_close_v1", "status": "returned",
            "entered_sha256": _sha(entered)})
        return value


def _validated_job(job_path: Path) -> dict[str, Any]:
    job = _read(job_path)
    _require(job.get("schema") == JOB_SCHEMA and _sha(Path(job["checkpoint_path"]))
             == job.get("checkpoint_sha256"), "child job/checkpoint")
    return {**job, "job_path": Path(job_path).resolve(), "job_sha256": _sha(job_path),
            "output_directory": Path(job["output_directory"]),
            "sidecar_directory": Path(job["sidecar_directory"]),
            "job": {"mode": "local_same_play_record",
                    "checkpoint": {"logical_id": job["checkpoint_logical_id"]},
                    "source": job["source"]}}


def run_full_terminal(validated: Mapping[str, Any], app: FullAppProxy,
                      run_worker: Callable[[], Mapping[str, Any]],
                      before_close: Callable[[Mapping[str, Any]], None]) -> tuple[int, Mapping[str, Any] | None]:
    """worker terminal을 먼저 봉인하고 어떤 경로에서도 SDK close를 시도한다."""
    output = Path(validated["output_directory"])
    stage = Path(validated["job_path"]).parent
    primary: BaseException | None = None
    report: Mapping[str, Any] | None = None
    try:
        report = run_worker()
        _require(report.get("technical_recording_complete") is True, "worker technical completion")
        before_close(report)
        _write_new(stage / "worker-cleanup-returned.json", {
            "schema": "local_same_play_worker_cleanup_v1", "status": "returned",
            "job_sha256": validated["job_sha256"], "checkpoint_sha256": validated["checkpoint_sha256"]})
        _write_new(output / "result.json", {
            "schema": "local_same_play_worker_terminal_v1", "completed": True,
            "technical_recording_complete": True, "job_sha256": validated["job_sha256"],
            "checkpoint_sha256": validated["checkpoint_sha256"], "capture_approved": False,
            "model_ready": False, "used_for_decision": False})
    except BaseException as error:  # noqa: BLE001 - terminal을 봉인하고 SDK close를 이어간다.
        primary = error
        output.mkdir(parents=True, exist_ok=True)
        _write_new(output / "failure.json", {
            "schema": "local_same_play_worker_failure_v1", "completed": False,
            "technical_recording_complete": False, "job_sha256": validated["job_sha256"],
            "checkpoint_sha256": validated["checkpoint_sha256"],
            "error_type": type(error).__name__, "error_message": str(error),
            "capture_approved": False, "model_ready": False, "used_for_decision": False})
    finally:
        try:
            app.close()
        except BaseException as close_error:  # noqa: BLE001 - 원래 worker 실패를 우선 보존한다.
            primary = primary or close_error
            traceback.print_exception(close_error)
    return (0 if primary is None else 1), report


def _full_app(validated: Mapping[str, Any], stage: Path, shutdown: Any) -> FullAppProxy:
    from isaaclab.app import AppLauncher  # pyright: ignore[reportMissingImports]

    app = AppLauncher(headless=True, enable_cameras=True, fast_shutdown=False).app
    try:
        import carb.settings  # pyright: ignore[reportMissingImports]

        return FullAppProxy(shutdown.AppProxy(app, validated, carb.settings.get_settings()), stage, validated)
    except BaseException:
        app.close(wait_for_replicator=False)
        raise


def _descriptor_path(root: Path, descriptor: Mapping[str, Any], label: str) -> Path:
    _require(set(descriptor) == {"file", "sha256", "bytes"}
             and isinstance(descriptor.get("file"), str), f"{label} descriptor")
    matches = [path for path in root.rglob(cast(str, descriptor["file"]))
               if path.is_file() and not path.is_symlink()
               and path.stat().st_size == descriptor.get("bytes") and _sha(path) == descriptor.get("sha256")]
    _require(len(matches) == 1, f"{label} descriptor가 원문 하나에 결합되지 않는다")
    return matches[0]


def _verify_preclose_proof(validated: Mapping[str, Any], modules: Mapping[str, Any]) -> dict[str, Any]:
    stage, output = Path(validated["job_path"]).parent, Path(validated["output_directory"])
    sidecar = Path(validated["sidecar_directory"])
    worker, sidecar_module, render = (modules[name] for name in ("worker", "sidecar", "render"))
    report_path = output / worker.REPORT_NAME
    report = _read(report_path)
    _require(report.get("schema") == worker.SCHEMA and report.get("completed") is True
             and report.get("technical_recording_complete") is True
             and report.get("job_sha256") == validated["job_sha256"]
             and report.get("checkpoint_sha256") == validated["checkpoint_sha256"], "worker report binding")
    descriptors = report.get("artifacts")
    _require(isinstance(descriptors, Mapping), "worker artifact descriptors")
    artifacts = {name: _descriptor_path(output, descriptor, f"worker {name}")
                 for name, descriptor in cast(Mapping[str, Mapping[str, Any]], descriptors).items()}
    _require(worker.PREVIOUS.RECORD_VERIFY._record_artifacts(validated, artifacts),
             "worker raw artifact recomputation")
    native_descriptor = report.get("native_capture_manifest")
    _require(isinstance(native_descriptor, Mapping), "native manifest descriptor")
    native_path = _descriptor_path(output, cast(Mapping[str, Any], native_descriptor), "native manifest")
    sidecar_module.load_sidecar(sidecar)
    same_pts = sidecar_module.join_with_trace(sidecar, artifacts["raw_trace"], native_path)
    render.verify_render_sync(stage, frames=validated["frames"])
    same_pts_path = stage / "same-pts.json"
    if same_pts_path.exists():
        _require(_read(same_pts_path) == same_pts, "same-PTS proof changed")
    else:
        _write_new(same_pts_path, same_pts)
    refs = {name: _ref(path) for name, path in {
        "worker_report": report_path,
        "native_manifest": native_path,
        "raw_trace": artifacts["raw_trace"],
        "sidecar_manifest": sidecar / sidecar_module.MANIFEST_NAME,
        "sidecar_records": sidecar / sidecar_module.RECORDS_NAME,
        "render_sync": stage / render.RENDER_SYNC_NAME,
        "same_pts": same_pts_path,
    }.items()}
    proof = {"schema": "local_same_play_preclose_proof_v1", "status": "raw_proofs_verified",
             "job_sha256": validated["job_sha256"], "checkpoint_sha256": validated["checkpoint_sha256"],
             "frames": validated["frames"], "proofs": refs, "capture_approved": False,
             "model_ready": False, "used_for_decision": False}
    proof_path = stage / "preclose-proof.json"
    if proof_path.exists():
        _require(_read(proof_path) == proof, "preclose proof changed")
    else:
        _write_new(proof_path, proof)
    return proof


def _seal_preclose_proof(validated: Mapping[str, Any], report: Mapping[str, Any],
                         modules: Mapping[str, Any]) -> None:
    report_path = Path(validated["output_directory"]) / modules["worker"].REPORT_NAME
    _require(_read(report_path) == report, "worker returned/sealed report mismatch")
    _verify_preclose_proof(validated, modules)


def sdk_child(job_path: Path) -> int:
    validated = _validated_job(job_path)
    if validated["mode"] == "full":
        modules = {name: importlib.import_module(module) for name, module in {
            "worker": "vision_pull_record_worker_v3",
            "capture": "vision_pull_record_capture_v2",
            "render": "run_agentic_play_record_v4",
            "sidecar": "record_agentic_state_sidecar_v1",
            "shutdown": "vision_shutdown_policy_v1",
        }.items()}
        stage = Path(job_path).resolve().parent
        app = _full_app(validated, stage, modules["shutdown"])
        report_path = Path(validated["output_directory"]) / modules["worker"].REPORT_NAME

        def run_worker() -> Mapping[str, Any]:
            return compose_full_worker(
                validated, app, worker=modules["worker"], capture=modules["capture"],
                render=modules["render"], sidecar=modules["sidecar"],
                seal_report=lambda value: _write_new(report_path, value),
                reader_factory=modules["sidecar"]._default_reader_factory,
            )

        exit_code, _report = run_full_terminal(
            validated, app, run_worker,
            before_close=lambda value: _seal_preclose_proof(validated, value, modules))
        return exit_code
    sidecar2 = importlib.import_module("record_agentic_state_sidecar_v2")
    return int(sidecar2.run_state_dry_run(
        checkpoint_path=Path(validated["checkpoint_path"]),
        checkpoint_sha256=validated["checkpoint_sha256"], source_id=validated["source_id"],
        out_dir=Path(validated["output_directory"]), frames=1,
    ))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-actual-none")
    inspect.add_argument("--source-batch", required=True, type=Path)
    inspect.add_argument("--lab", required=True, type=Path)
    inspect.add_argument("--out", required=True, type=Path)
    prepare = sub.add_parser("prepare-protocol")
    prepare.add_argument("--inspection", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    prepare.add_argument("--anchor")
    run = sub.add_parser("execute")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--anchor", required=True)
    run.add_argument("--lab", required=True, type=Path)
    run.add_argument("--inspection", required=True, type=Path)
    run.add_argument("--stage", required=True, type=Path)
    run.add_argument("--isaac-python", required=True, type=Path)
    run.add_argument("--mode", required=True, choices=("smoke", "full"))
    child = sub.add_parser("sdk-child")
    child.add_argument("job", type=Path)
    args, extras = parser.parse_known_args(argv)
    _require(extras == (KIT_ARGS if args.command == "sdk-child" else []),
             "허용되지 않은 CLI 인자")
    if args.command == "inspect-actual-none":
        print(json.dumps(inspect_actual_none(args.source_batch, args.lab, args.out), sort_keys=True))
        return 0
    if args.command == "prepare-protocol":
        print(json.dumps(prepare_protocol(args.inspection, args.out, anchor=args.anchor), sort_keys=True))
        return 0
    if args.command == "execute":
        print(json.dumps(execute(args.protocol, args.anchor, args.lab, args.inspection, args.stage,
                                 args.isaac_python, mode=args.mode), sort_keys=True))
        return 0
    return sdk_child(args.job)


if __name__ == "__main__":
    raise SystemExit(main())
