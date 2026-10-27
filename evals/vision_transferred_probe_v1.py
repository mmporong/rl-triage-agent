"""Transferred vision checkpoints의 등록 P_reward 확인 검사 실행기.

학습 provenance는 보존된 source-training manifest로 검증하고, 검사 실행
provenance는 새 승인 원장과 Brev worker 원문으로 분리한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import brev_recovery_backend_v4 as BACKEND
import selection_comparison_controller_v1 as CONTROLLER
import vision_selection_execution_v1 as CONSUMER

SCHEMA = "vision_transferred_probe_protocol_v1"
LEDGER_SCHEMA = "vision_transferred_probe_ledger_event_v1"
RESULT_SCHEMA = "vision_transferred_probe_pair_result_v1"
PROBE = "P_reward"
ARMS = ("NONE", "F1")
GROUPS = ("source-01", "source-02")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} path가 문자열이 아니다")
    path = Path(value)
    if path.is_absolute() or "\\" in value or ".." in path.parts or not value:
        raise ValueError(f"{label} path가 portable 형식이 아니다")
    return path.as_posix()


def _reference(root: Path, value: object, label: str, *, lf: bool = False) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    relative = _safe_relative(value.get("path"), label)
    target = (Path(root) / relative).resolve(strict=True)
    target.relative_to(Path(root).resolve(strict=True))
    if file_sha(target, lf=lf) != value.get("sha256"):
        raise ValueError(f"{label} SHA가 다르다")
    return target


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).relative_to(Path(root))
    except ValueError:
        return False
    return True


def _ref(root: Path, path: Path) -> dict[str, str]:
    repository = Path(root).resolve(strict=True)
    target = Path(path).resolve(strict=True)
    target.relative_to(repository)
    return {"path": target.relative_to(repository).as_posix(), "sha256": file_sha(target)}


def validate_protocol(path: Path, *, require_bound: bool = False) -> dict[str, Any]:
    protocol_path = Path(path).resolve(strict=True)
    protocol = read(protocol_path)
    required = {
        "schema", "protocol_id", "registered_before_execution", "deadline_utc", "probe",
        "measurement_type", "source_training", "probe_runtime", "groups", "limits",
        "resources", "predictions", "classification", "wording", "execution_root",
        "source_binding",
    }
    if set(protocol) != required or protocol.get("schema") != SCHEMA:
        raise ValueError("transferred probe protocol 필드가 다르다")
    if (protocol.get("registered_before_execution") is not True or protocol.get("probe") != PROBE
            or protocol.get("measurement_type") != "preregistered_registered_fault_confirmation"):
        raise ValueError("probe 등록 목적이 다르다")
    if _safe_relative(protocol.get("execution_root"), "execution root") != (
            ".triage-runtime/vision-transferred-probe-v1"):
        raise ValueError("probe execution root가 고정값과 다르다")
    deadline = datetime.fromisoformat(str(protocol["deadline_utc"]).replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("deadline timezone이 없다")
    source = protocol.get("source_training")
    runtime = protocol.get("probe_runtime")
    if (not isinstance(source, Mapping) or set(source) != {"summary", "archive_layout", "source_protocol"}
            or not isinstance(runtime, Mapping)
            or set(runtime) != {"source_anchor", "source_protocol", "backend", "worker"}
            or not HEX40.fullmatch(str(runtime.get("source_anchor", "")))):
        raise ValueError("source-training/probe-runtime 분리 등록값이 다르다")
    for name in ("summary", "source_protocol"):
        value = source[name]
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError(f"source_training.{name} reference가 다르다")
        _safe_relative(value["path"], f"source_training.{name}")
        if not HEX64.fullmatch(str(value["sha256"])):
            raise ValueError(f"source_training.{name} SHA가 다르다")
    for name in ("source_protocol", "backend", "worker"):
        value = runtime[name]
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256_lf"}:
            raise ValueError(f"probe_runtime.{name} reference가 다르다")
        _safe_relative(value["path"], f"probe_runtime.{name}")
        if not HEX64.fullmatch(str(value["sha256_lf"])):
            raise ValueError(f"probe_runtime.{name} SHA가 다르다")
    groups = protocol.get("groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("두 source seed group이 필요하다")
    seen: set[str] = set()
    for row in groups:
        if (not isinstance(row, Mapping)
                or set(row) != {"group", "seed", "checkpoints", "training_manifests", "execution"}
                or row.get("group") not in GROUPS or row["group"] in seen
                or type(row.get("seed")) is not int):
            raise ValueError("source group 형식이 다르다")
        seen.add(str(row["group"]))
        for field in ("checkpoints", "training_manifests", "execution"):
            if not isinstance(row[field], Mapping) or set(row[field]) != set(ARMS):
                raise ValueError(f"{row['group']} {field} arm 집합이 다르다")
        for arm in ARMS:
            ref = row["checkpoints"][arm]
            if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
                    or not HEX64.fullmatch(str(ref.get("sha256", "")))):
                raise ValueError("checkpoint reference가 다르다")
            _safe_relative(ref["path"], "checkpoint")
            _safe_relative(row["training_manifests"][arm], "training manifest")
            plan = row["execution"][arm]
            if (not isinstance(plan, Mapping) or set(plan) != {"stage", "run_name"}
                    or not isinstance(plan["run_name"], str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}", plan["run_name"])):
                raise ValueError("probe execution plan이 다르다")
            _safe_relative(plan["stage"], "execution stage")
    limits = protocol.get("limits")
    resources = protocol.get("resources")
    if (limits != {"probe_pairs": 2, "gpu_probe_runs": 4, "total_gpu_s": 600,
                   "pair_gpu_s": 300, "api_requests": 0, "training_runs": 0}
            or not isinstance(resources, Mapping)
            or set(resources) != {"instance_id", "instance_type", "compute_quote_usd_per_h",
                                  "additional_brev_cap_usd", "program_cap_usd"}
            or resources.get("additional_brev_cap_usd") != 3
            or resources.get("program_cap_usd") != 35):
        raise ValueError("실행 횟수·비용 상한이 다르다")
    binding = protocol.get("source_binding")
    if (not isinstance(binding, Mapping)
            or set(binding) != {"status", "source_anchor", "source_sha256_lf"}
            or binding.get("status") not in {"draft", "bound"}
            or not isinstance(binding.get("source_sha256_lf"), Mapping)):
        raise ValueError("adapter source binding 형식이 다르다")
    if require_bound:
        if binding["status"] != "bound" or not HEX40.fullmatch(str(binding["source_anchor"])):
            raise ValueError("실행 전 committed adapter source binding이 필요하다")
        expected = {"evals/vision_transferred_probe_v1.py",
                    "tests/test_vision_transferred_probe_v1.py"}
        if set(binding["source_sha256_lf"]) != expected or any(
                not HEX64.fullmatch(str(value)) for value in binding["source_sha256_lf"].values()):
            raise ValueError("adapter source closure가 완전하지 않다")
    return protocol


def verify_repository_binding(root: Path, protocol: Mapping[str, Any]) -> None:
    repository = Path(root).resolve(strict=True)
    binding = protocol["source_binding"]
    anchor = str(binding["source_anchor"])
    if subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor", anchor, "HEAD"],
                      check=False).returncode:
        raise ValueError("adapter source anchor가 현재 HEAD의 조상이 아니다")
    for relative, expected in binding["source_sha256_lf"].items():
        target = repository / relative
        if not target.is_file() or file_sha(target, lf=True) != expected:
            raise ValueError(f"adapter source hash가 다르다: {relative}")
        blob = subprocess.run(["git", "-C", str(repository), "show", f"{anchor}:{relative}"],
                              check=True, capture_output=True).stdout.replace(b"\r\n", b"\n")
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"adapter source anchor blob이 다르다: {relative}")


def verify_source_inputs(root: Path, archive: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    repository = Path(root).resolve(strict=True)
    archive_root = Path(archive).resolve(strict=True)
    summary_path = _reference(repository, protocol["source_training"]["summary"], "source summary")
    summary = read(summary_path)
    runtime_path = _reference(repository, {
        "path": protocol["probe_runtime"]["source_protocol"]["path"],
        "sha256": protocol["probe_runtime"]["source_protocol"]["sha256_lf"],
    }, "probe runtime protocol", lf=True)
    runtime_protocol = read(runtime_path)
    source_protocol_path = _reference(repository, protocol["source_training"]["source_protocol"],
                                      "source training protocol", lf=True)
    source_protocol = read(source_protocol_path)
    training = {(int(row["seed"]), str(row["arm"])): row for row in summary.get("training", [])
                if isinstance(row, Mapping)}
    if len(training) != 4:
        raise ValueError("source summary의 네 training arm이 없다")
    checked: dict[str, Any] = {"summary": summary, "runtime_protocol": runtime_protocol,
                               "source_protocol": source_protocol, "groups": {}}
    for row in protocol["groups"]:
        group: dict[str, Any] = {"seed": row["seed"], "arms": {}}
        for arm in ARMS:
            checkpoint = (archive_root / _safe_relative(row["checkpoints"][arm]["path"], "checkpoint")).resolve(strict=True)
            checkpoint.relative_to(archive_root)
            manifest = (archive_root / _safe_relative(row["training_manifests"][arm], "manifest")).resolve(strict=True)
            manifest.relative_to(archive_root)
            if file_sha(checkpoint) != row["checkpoints"][arm]["sha256"]:
                raise ValueError(f"{row['group']} {arm} checkpoint SHA가 다르다")
            body = read(manifest)
            source_row = training.get((row["seed"], arm))
            if (source_row is None or source_row.get("checkpoint_sha256") != file_sha(checkpoint)
                    or body.get("seed") != row["seed"] or body.get("arm") != arm
                    or body.get("source_commit_sha") != summary.get("source_commit")
                    or body.get("protocol_sha256_lf") != protocol["source_training"]["source_protocol"]["sha256"]):
                raise ValueError(f"{row['group']} {arm} source-training 계보가 다르다")
            group["arms"][arm] = {"checkpoint": checkpoint, "manifest": manifest, "body": body}
        checked["groups"][row["group"]] = group
    return checked


def expected_probe_source(protocol: Mapping[str, Any], source_inputs: Mapping[str, Any]) -> dict[str, Any]:
    runtime = source_inputs["runtime_protocol"]
    environment = dict(runtime["environment"])
    environment.update(cuda_available=True, cuda_device_count=1)
    registered = protocol["probe_runtime"]
    return {
        "source_anchor": registered["source_anchor"],
        "protocol": {"path": registered["source_protocol"]["path"],
                     "sha256_lf": registered["source_protocol"]["sha256_lf"]},
        "environment": environment,
        "source_sha256_lf": runtime["source_sha256_lf"],
        "runtime_source_sha256_lf": runtime["runtime_source_sha256_lf"],
        "python_package_source_sha256_lf": runtime["python_package_source_sha256_lf"],
        "task": next(iter(source_inputs["groups"].values()))["arms"]["NONE"]["body"]["task"],
    }


def _append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(dict(value)).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def authorize_pair(stage: Path, group: Mapping[str, Any], protocol_sha: str,
                   approver: str) -> dict[str, Any]:
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError("probe pair approver가 비어 있다")
    request = {
        "group": group["group"], "seed": group["seed"], "probe": PROBE,
        "checkpoints": {arm: group["checkpoints"][arm]["sha256"] for arm in ARMS},
        "execution": group["execution"], "budget_gpu_s": 300,
        "protocol_sha256": protocol_sha,
    }
    request_id = digest(request)[:16]
    ledger = Path(stage) / "approval.ledger.jsonl"
    at = datetime.now(timezone.utc).isoformat()
    common = {"schema": LEDGER_SCHEMA, "request_id": request_id,
              "request_sha256": digest(request), "at": at}
    _append(ledger, {**common, "event": "proposed", "request": request})
    _append(ledger, {**common, "event": "approved", "approver": approver})
    _append(ledger, {**common, "event": "consumed"})
    authorization = {"schema": "vision_transferred_probe_authorization_v1",
                     "request_id": request_id, "request": request,
                     "ledger_prefix_sha256": digest([read_line for read_line in _ledger(ledger)]),
                     "events": ["proposed", "approved", "consumed"]}
    auth_path = Path(stage) / "authorization.json"
    write_new(auth_path, authorization)
    return {"ledger": ledger, "authorization": auth_path, "request": request,
            "request_id": request_id}


def _ledger(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _descriptor(root: Path, stage: Path, side: str, checkpoint_sha: str) -> Path:
    names = {"job": "job.json", "worker": "worker.json", "execution": "execution.json",
             "backend_result": "backend_result.json", "report_jobs": "report.jobs.json",
             "report": "report.json"}
    paths = {name: Path(stage) / filename for name, filename in names.items()}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("probe six-source 원문이 완전하지 않다")
    value = {"schema": "vision_selection_brev_probe_evidence_v1", "side": side,
             "probe": PROBE, "checkpoint_sha256": checkpoint_sha,
             "sources": {name: _ref(root, path) for name, path in paths.items()}}
    target = Path(stage) / "descriptor.json"
    write_new(target, value)
    return target


def _run_arm(backend: BACKEND.Backend, root: Path, arm_input: Mapping[str, Any], arm: str,
             plan: Mapping[str, Any], output: Path) -> Path:
    stage = Path(output) / str(plan["stage"])
    artifact = {"checkpoint": {"path": str(arm_input["checkpoint"]),
                                "sha256": file_sha(arm_input["checkpoint"])},
                "arm": arm, "seed": arm_input["body"]["seed"],
                "manifest_path": str(arm_input["manifest"])}
    result = backend.probe(stage, artifact, PROBE, str(plan["run_name"]))
    write_new(stage / "backend_result.json", result)
    return _descriptor(root, stage, "reference" if arm == "NONE" else "case",
                       artifact["checkpoint"]["sha256"])


def validate_pair(root: Path, protocol: Mapping[str, Any], source_inputs: Mapping[str, Any],
                  stage: Path) -> dict[str, Any]:
    output = Path(stage).resolve(strict=True)
    authorization = read(output / "authorization.json")
    request = authorization.get("request")
    rows = _ledger(output / "approval.ledger.jsonl")
    if (authorization.get("schema") != "vision_transferred_probe_authorization_v1"
            or not isinstance(request, Mapping) or [row.get("event") for row in rows] !=
            ["proposed", "approved", "consumed", "receipt"]
            or any(row.get("schema") != LEDGER_SCHEMA for row in rows)
            or any(row.get("request_id") != authorization.get("request_id") for row in rows)):
        raise ValueError("probe 승인→소비→receipt 원장이 완결되지 않았다")
    group = next((row for row in protocol["groups"] if row["group"] == request.get("group")), None)
    expected_request = None if group is None else {
        "group": group["group"], "seed": group["seed"], "probe": PROBE,
        "checkpoints": {arm: group["checkpoints"][arm]["sha256"] for arm in ARMS},
        "execution": group["execution"], "budget_gpu_s": 300,
        "protocol_sha256": file_sha(Path(protocol["_path"])),
    }
    request_sha = digest(request)
    if (group is None or dict(request) != expected_request
            or authorization.get("request_id") != request_sha[:16]
            or authorization.get("events") != ["proposed", "approved", "consumed"]
            or authorization.get("ledger_prefix_sha256") != digest(rows[:3])
            or rows[0].get("request") != request
            or any(row.get("request_sha256") != request_sha for row in rows[1:])
            or not isinstance(rows[1].get("approver"), str) or not rows[1]["approver"].strip()):
        raise ValueError("probe request가 protocol group과 다르다")
    expected = expected_probe_source(protocol, source_inputs)
    actual: dict[str, dict[str, Any]] = {}
    descriptors: dict[str, dict[str, str]] = {}
    for arm, side in (("NONE", "reference"), ("F1", "case")):
        descriptor_path = output / str(group["execution"][arm]["stage"]) / "descriptor.json"
        descriptor = read(descriptor_path)
        actual[arm] = CONSUMER._brev_actual_probe(
            Path(root), descriptor, side=side, probe=PROBE,
            checkpoint_sha256=group["checkpoints"][arm]["sha256"], expected_source=expected,
        )
        descriptors[arm] = _ref(root, descriptor_path)
    outcomes = {"NONE": CONTROLLER.classify_raw(PROBE, actual["NONE"]["measurement"],
                                                 actual["NONE"]["measurement"]),
                "F1": CONTROLLER.classify_raw(PROBE, actual["F1"]["measurement"],
                                               actual["NONE"]["measurement"])}
    gpu_s = sum(float(actual[arm]["gpu_s"]) for arm in ARMS)
    receipt = rows[-1]
    if (receipt.get("outcomes") != outcomes or receipt.get("gpu_s") != gpu_s
            or receipt.get("descriptors") != descriptors or gpu_s > 300):
        raise ValueError("probe receipt가 actual 원문 재계산과 다르다")
    return {"schema": RESULT_SCHEMA, "status": "complete", "group": group["group"],
            "seed": group["seed"], "probe": PROBE, "outcomes": outcomes,
            "measurements": {arm: actual[arm]["measurement"] for arm in ARMS},
            "gpu_s": gpu_s, "authorization": _ref(root, output / "authorization.json"),
            "ledger": _ref(root, output / "approval.ledger.jsonl"), "descriptors": descriptors,
            "usable": outcomes["NONE"] == "normal" and outcomes["F1"] == "abnormal"}


def run_pair(root: Path, protocol_path: Path, archive: Path, output: Path, group_name: str,
             lab: Path, python: Path, approver: str,
             *, backend_factory: Callable[..., BACKEND.Backend] = BACKEND.Backend) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, require_bound=True)
    deadline = datetime.fromisoformat(str(protocol["deadline_utc"]).replace("Z", "+00:00"))
    if datetime.now(timezone.utc) >= deadline.astimezone(timezone.utc):
        raise ValueError("probe protocol deadline이 지났다")
    protocol["_path"] = str(Path(protocol_path).resolve(strict=True))
    verify_repository_binding(root, protocol)
    repository = Path(root).resolve(strict=True)
    lab_root = Path(lab).resolve(strict=True)
    archive_root = Path(archive).resolve(strict=True)
    if not any(_is_relative_to(archive_root, allowed) for allowed in (repository, lab_root)):
        raise ValueError("원격 source archive는 repo 또는 Isaac Lab 아래여야 한다")
    inputs = verify_source_inputs(root, archive, protocol)
    group = next((row for row in protocol["groups"] if row["group"] == group_name), None)
    if group is None:
        raise ValueError("등록되지 않은 source group")
    stage = Path(output).resolve()
    expected_stage = repository / str(protocol["execution_root"]) / group_name
    if stage != expected_stage.resolve():
        raise ValueError("probe pair output이 등록 execution root와 다르다")
    if stage.exists():
        raise FileExistsError("probe pair output은 새 경로여야 한다")
    stage.mkdir(parents=True)
    auth = authorize_pair(stage, group, file_sha(Path(protocol_path)), approver)
    runtime = inputs["runtime_protocol"]
    backend = backend_factory(Path(lab), Path(python), {
        **runtime, "protocol_path": str((Path(root) / protocol["probe_runtime"]["source_protocol"]["path"]).resolve()),
        "protocol_sha256_lf": protocol["probe_runtime"]["source_protocol"]["sha256_lf"],
    }, protocol["probe_runtime"]["source_anchor"])
    descriptors: dict[str, dict[str, str]] = {}
    actual: dict[str, dict[str, Any]] = {}
    expected = expected_probe_source(protocol, inputs)
    try:
        for arm, side in (("NONE", "reference"), ("F1", "case")):
            path = _run_arm(backend, Path(root), inputs["groups"][group_name]["arms"][arm], arm,
                            group["execution"][arm], stage)
            descriptors[arm] = _ref(Path(root), path)
            actual[arm] = CONSUMER._brev_actual_probe(
                Path(root), read(path), side=side, probe=PROBE,
                checkpoint_sha256=group["checkpoints"][arm]["sha256"], expected_source=expected,
            )
    except BaseException as error:
        known_gpu_s = sum(float(value["gpu_s"]) for value in actual.values())
        if isinstance(error, BACKEND.BackendError) and isinstance(error.receipt.get("wall_s"), (int, float)):
            known_gpu_s += float(error.receipt["wall_s"])
        _append(auth["ledger"], {"schema": LEDGER_SCHEMA, "event": "receipt",
                "request_id": auth["request_id"], "request_sha256": digest(auth["request"]),
                "at": datetime.now(timezone.utc).isoformat(), "outcomes": {"NONE": "unknown", "F1": "unknown"},
                "gpu_s": known_gpu_s, "descriptors": descriptors,
                "status": "failed", "error_type": type(error).__name__})
        raise
    outcomes = {"NONE": CONTROLLER.classify_raw(PROBE, actual["NONE"]["measurement"],
                                                 actual["NONE"]["measurement"]),
                "F1": CONTROLLER.classify_raw(PROBE, actual["F1"]["measurement"],
                                               actual["NONE"]["measurement"])}
    gpu_s = sum(float(actual[arm]["gpu_s"]) for arm in ARMS)
    if gpu_s > 300:
        raise ValueError("probe pair GPU 시간이 등록 상한을 넘었다")
    _append(auth["ledger"], {"schema": LEDGER_SCHEMA, "event": "receipt",
            "request_id": auth["request_id"],
            "request_sha256": digest(auth["request"]), "at": datetime.now(timezone.utc).isoformat(),
            "outcomes": outcomes, "gpu_s": gpu_s, "descriptors": descriptors})
    result = validate_pair(Path(root), protocol, inputs, stage)
    write_new(stage / "result.json", result)
    return result


def bind_source(protocol_path: Path, source_anchor: str) -> dict[str, Any]:
    if not HEX40.fullmatch(source_anchor):
        raise ValueError("source anchor SHA 형식이 다르다")
    path = Path(protocol_path).resolve(strict=True)
    protocol = validate_protocol(path)
    if protocol["source_binding"]["status"] != "draft":
        raise ValueError("draft protocol만 source binding할 수 있다")
    hashes = {relative: file_sha(ROOT / relative, lf=True) for relative in (
        "evals/vision_transferred_probe_v1.py", "tests/test_vision_transferred_probe_v1.py")}
    protocol["source_binding"] = {"status": "bound", "source_anchor": source_anchor,
                                  "source_sha256_lf": hashes}
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind-source")
    bind.add_argument("--protocol", type=Path, required=True)
    bind.add_argument("--source-anchor", required=True)
    check = commands.add_parser("check")
    check.add_argument("--protocol", type=Path, required=True)
    check.add_argument("--archive", type=Path, required=True)
    run = commands.add_parser("run-pair")
    for item in (check, run):
        item.add_argument("--repo", type=Path, default=ROOT)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--group", choices=GROUPS, required=True)
    run.add_argument("--lab", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--approver", required=True)
    args = parser.parse_args(argv)
    if args.command == "bind-source":
        value = bind_source(args.protocol, args.source_anchor)
    elif args.command == "check":
        protocol = validate_protocol(args.protocol, require_bound=True)
        verify_repository_binding(args.repo, protocol)
        checked = verify_source_inputs(args.repo, args.archive, protocol)
        value = {"status": "ready", "groups": sorted(checked["groups"]), "probe": PROBE}
    else:
        value = run_pair(args.repo, args.protocol, args.archive, args.output, args.group,
                         args.lab, args.python, args.approver)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
