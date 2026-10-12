"""검증된 교정 학습 결과를 공개 수치 재생 묶음으로 내보낸다."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    sys.path.insert(0, str(directory))

import run_training_correction as H  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402


RESULTS = ROOT / "evals/results"
SCHEMA = "training_correction_public_v1"
REPLAY_SCOPE = "numeric_replay_only_not_execution_identity_attestation"
SOURCE_COMMIT_SCOPE = "experiment_execution_and_verdict_code_only"
PUBLISHER_SOURCES = {
    "evals/publish_training_correction.py",
    "evals/run_training_correction.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
}
PARENT_WALL_SCOPE = "parent_launch_to_exit_and_owned_job_empty_verification"
PUBLIC_FILES = {
    "reports/clean_reference.json", "reports/fault_sham.json", "reports/corrected.json",
    "original_reports/clean_reference.json", "original_reports/fault.json",
    "initial_states.json", "training_evidence.json", "source_inputs.json", "result.json",
}
FORBIDDEN_KEYS = {"pid", "process_id", "process_identity", "owned_process_identities", "processes",
                  "process_list", "stdout", "stderr", "isaaclab_root", "output_dir", "log_dir", "path"}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("public artifact must be a JSON object")
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _publisher_sources() -> dict[str, str]:
    result = {}
    for relative in sorted(PUBLISHER_SOURCES):
        raw = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        result[relative] = hashlib.sha256(raw).hexdigest()
    if set(result) != PUBLISHER_SOURCES:
        raise AssertionError("publisher source set differs")
    return result


def _public_value(value: object, key: str | None = None) -> None:
    if key is not None and key.lower() in FORBIDDEN_KEYS:
        raise ValueError(f"private execution field is not publishable: {key}")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ValueError("public object keys must be strings")
            _public_value(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _public_value(child)
    elif isinstance(value, str):
        if (PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
                or "bench/private" in value.replace("\\", "/").lower()):
            raise ValueError("absolute or private path leaked into public output")


def _initial_public(initial: dict) -> dict:
    checkpoint = initial["resume_checkpoint"]
    def state_summary(value: dict) -> dict:
        required = {"sha256", "tensor_count", "scalar_count"}
        if not isinstance(value, dict) or not required <= set(value):
            raise ValueError("initial state digest summary is incomplete")
        return {key: value[key] for key in ("sha256", "tensor_count", "scalar_count")}

    return {
        "policy_state": state_summary(initial["policy_state"]),
        "optimizer_state": state_summary(initial["optimizer_state"]),
        "rng_state_before_learn": deepcopy(initial["rng_state_before_learn"]),
        "runtime_versions": deepcopy(initial["runtime_versions"]),
        "resume_checkpoint": {"file": Path(checkpoint["path"]).name,
                              "sha256": checkpoint["sha256"],
                              "load_optimizer": checkpoint["load_optimizer"]},
        "resume_iteration": initial["resume_iteration"],
    }


def _receipt_public(receipt: dict, receipt_sha256: str) -> dict:
    return {
        "receipt_sha256": receipt_sha256,
        "checkpoint": {"file": Path(receipt["checkpoint"]["path"]).name,
                       "sha256": receipt["checkpoint"]["sha256"]},
        "configs": {key: {"file": Path(row["path"]).name, "sha256": row["sha256"]}
                    for key, row in receipt["configs"].items()},
        "resume_checkpoint": {"file": Path(receipt["resume_checkpoint"]["path"]).name,
                              "sha256": receipt["resume_checkpoint"]["sha256"]},
        "resume_iteration": receipt["resume_iteration"],
        "final_iteration": receipt["final_iteration"],
        "actual_control_steps": receipt["actual_control_steps"],
        "aggregate_env_steps": receipt["aggregate_env_steps"],
        "budget": deepcopy(receipt["budget"]),
        "observed": deepcopy(receipt["observed"]),
        "runtime_versions": deepcopy(receipt["runtime_versions"]),
    }


def _parent_wall_cost(execution: dict, eval_execution: dict, parent: dict) -> dict:
    if any(row.get("wall_scope") != PARENT_WALL_SCOPE for row in (execution, eval_execution)):
        raise ValueError("owned execution parent wall scope differs")

    def number(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("parent wall time must be a finite non-negative number")
        return float(value)

    train = number(execution.get("wall_s"))
    evaluation = number(eval_execution.get("wall_s"))
    if number(parent.get("train_parent_wall_s")) != train or number(parent.get("eval_parent_wall_s")) != evaluation:
        raise ValueError("parent wall time differs from the owned execution record")
    total = train + evaluation
    if not math.isfinite(total):
        raise ValueError("parent wall time total is not finite")
    return {"scope": PARENT_WALL_SCOPE, "train_s": train, "fixed_evaluation_s": evaluation,
            "total_s": total}


def _public_costs(evidence_doc: dict) -> dict[str, dict]:
    result = {}
    for arm in H.ARMS:
        row = evidence_doc["arms"][arm].get("parent_wall_s")
        if not isinstance(row, dict) or set(row) != {"scope", "train_s", "fixed_evaluation_s", "total_s"}:
            raise ValueError("public parent wall time schema differs")
        if row["scope"] != PARENT_WALL_SCOPE:
            raise ValueError("public parent wall time scope differs")
        values = [row[key] for key in ("train_s", "fixed_evaluation_s", "total_s")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("public parent wall time is invalid")
        if not math.isclose(float(row["total_s"]), float(row["train_s"]) + float(row["fixed_evaluation_s"]),
                            rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("public parent wall time total differs")
        result[arm] = {key: row[key] for key in ("train_s", "fixed_evaluation_s", "total_s")}
    return result


def _summarize_public(reports: dict, initials: dict, originals: dict, source_inputs: dict) -> dict:
    fault = source_inputs["fault"]
    result = H.summarize(reports, initials, originals, source_inputs["bands"],
                         source_inputs["runs"][fault]["original_failed"])
    if result["status"] == "invalid":
        raise ValueError("invalid result cannot be published")
    return result


def export(tag: str, isaaclab: Path) -> dict:
    private = H.folder_for(tag)
    request = H.read(private / "request.json")
    protocol, commit = H.verify_protocol(isaaclab)
    if (request.get("tag") != tag or request.get("source_commit_sha") != commit
            or request.get("protocol_sha256_lf") != H.sha(H.PROTOCOL, True)
            or Path(request.get("isaaclab_root", "")).resolve() != isaaclab.resolve()):
        raise ValueError("request is not bound to the current verified source")
    case = request.get("case")
    if case not in H.CASES:
        raise ValueError("unregistered case")
    fault = H.CASES[case]["fault"]

    reports, initials, evidence = {}, {}, {}
    for arm in H.ARMS:
        training = private / arm / "train"
        evaluation = private / arm / "eval"
        manifest, arm_protocol = H.validate_manifest(training / "manifest.json")
        if arm_protocol != protocol or manifest["arm"] != arm or manifest["case"] != case:
            raise ValueError("arm manifest is not bound to this experiment")
        execution = H.read(training / "execution.json")
        receipt = H.verify_training(private, arm, execution)
        eval_execution = H.read(evaluation / "execution.json")
        if not H.execution_valid(eval_execution):
            raise ValueError("fixed evaluation execution is incomplete")
        report_path = evaluation / "report.json"
        report = C.validate_fixed_report(H.read(report_path), {
            "name": arm, "checkpoint_file": f"model_{H.FIXED['final_iteration']}.pt",
            "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "action_scale": H.FIXED["action_scale"],
            "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"],
        })
        parent = H.read(private / arm / "parent.json")
        if (parent.get("status") != "pass"
                or parent.get("training_receipt_sha256") != H.sha(training / "receipt.json")
                or parent.get("fixed_report_sha256") != H.sha(report_path)):
            raise ValueError("arm parent binding differs")
        reports[arm] = report
        initial = H.read(training / "initial.json")
        initials[arm] = initial
        evidence[arm] = {**_receipt_public(receipt, H.sha(training / "receipt.json")),
                         "parent_wall_s": _parent_wall_cost(execution, eval_execution, parent)}

    originals = {
        "clean_reference": H.read(H.source_paths(isaaclab, "NONE")["fixed_report"]),
        "fault": H.read(H.source_paths(isaaclab, fault)["fixed_report"]),
    }
    source = {
        "fault": fault, "bands": deepcopy(protocol["resume_inputs"]["bands"]),
        "runtime_versions": deepcopy(protocol["resume_inputs"]["runtime_versions"]),
        "summary_sha256_lf": protocol["resume_inputs"]["summary_sha256_lf"],
        "runs": {key: deepcopy(protocol["resume_inputs"]["runs"][key]) for key in ("NONE", fault)},
        "fixed_eval_sha256_lf": protocol["fixed_eval_sha256_lf"],
    }
    numeric = _summarize_public(reports, initials, originals, source)
    private_result = H.read(private / "result.json")
    if (private_result.get("status") != numeric["status"] or private_result.get("case") != case
            or private_result.get("source_commit_sha") != commit):
        raise ValueError("private result differs from replayed evidence")

    publisher_sources = _publisher_sources()
    artifacts = {
        **{f"reports/{arm}.json": reports[arm] for arm in H.ARMS},
        "original_reports/clean_reference.json": originals["clean_reference"],
        "original_reports/fault.json": originals["fault"],
        "initial_states.json": {"schema": SCHEMA, "arms": {arm: _initial_public(initials[arm]) for arm in H.ARMS}},
        "training_evidence.json": {"schema": SCHEMA, "arms": evidence},
        "source_inputs.json": {"schema": SCHEMA, **source},
        "result.json": {"schema": SCHEMA, "replay_scope": REPLAY_SCOPE, "tag": tag, "case": case,
                        "source_commit_sha": commit, "protocol_sha256_lf": request["protocol_sha256_lf"],
                        "source_commit_scope": SOURCE_COMMIT_SCOPE,
                        "publisher_sources": publisher_sources, "numeric": numeric},
    }
    if set(artifacts) != PUBLIC_FILES:
        raise AssertionError("public artifact set differs")
    for value in artifacts.values():
        _public_value(value)
    encoded = {name: _json_bytes(value) for name, value in artifacts.items()}
    manifest = {"schema": SCHEMA, "replay_scope": REPLAY_SCOPE,
                "publisher_sources": publisher_sources,
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(encoded.items())}}
    _public_value(manifest)
    output = RESULTS / tag
    if output.exists():
        raise FileExistsError("public result folder already exists")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in encoded.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    with (output / "artifact_manifest.json").open("xb") as stream:
        stream.write(_json_bytes(manifest))
    return {"status": numeric["status"], "folder": f"evals/results/{tag}", "replay_scope": REPLAY_SCOPE,
            "artifact_count": len(encoded)}


def replay(folder: Path) -> dict:
    if folder.is_symlink():
        raise ValueError("public result folder is missing or unsafe")
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError("public result folder is missing or unsafe")
    manifest_path = folder / "artifact_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("public artifact path is unsafe")
    manifest = _read(manifest_path)
    if (set(manifest) != {"schema", "replay_scope", "publisher_sources", "files"}
            or manifest.get("schema") != SCHEMA or manifest.get("replay_scope") != REPLAY_SCOPE
            or set(manifest.get("files", {})) != PUBLIC_FILES):
        raise ValueError("public artifact manifest differs")
    published_sources = manifest.get("publisher_sources")
    if (not isinstance(published_sources, dict) or set(published_sources) != PUBLISHER_SOURCES
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in published_sources.values())
            or published_sources != _publisher_sources()):
        raise ValueError("publisher source LF-SHA binding differs")
    actual_files = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual_files != PUBLIC_FILES | {"artifact_manifest.json"}:
        raise ValueError("public artifact file set differs")
    documents = {}
    for name, expected in manifest["files"].items():
        path = folder / name
        try:
            path.resolve(strict=True).relative_to(folder)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError("public artifact path is unsafe") from exc
        parents = list(path.parents)
        if path.is_symlink() or any(parent.is_symlink() for parent in parents[:parents.index(folder)]):
            raise ValueError("public artifact path is unsafe")
        if H.sha(path) != expected:
            raise ValueError("public artifact hash differs")
        documents[name] = _read(path)
        _public_value(documents[name])
    reports = {arm: documents[f"reports/{arm}.json"] for arm in H.ARMS}
    originals = {key: documents[f"original_reports/{key}.json"] for key in ("clean_reference", "fault")}
    initial_doc = documents["initial_states.json"]
    evidence_doc = documents["training_evidence.json"]
    source = documents["source_inputs.json"]
    result = documents["result.json"]
    if any(document.get("schema") != SCHEMA for document in (initial_doc, evidence_doc, source, result)):
        raise ValueError("public artifact schema differs")
    if (set(initial_doc) != {"schema", "arms"} or set(initial_doc.get("arms", {})) != set(H.ARMS)
            or set(evidence_doc) != {"schema", "arms"} or set(evidence_doc.get("arms", {})) != set(H.ARMS)):
        raise ValueError("public arm evidence schema differs")
    if (result.get("source_commit_scope") != SOURCE_COMMIT_SCOPE
            or result.get("publisher_sources") != published_sources):
        raise ValueError("result and manifest publisher source bindings differ")
    costs = _public_costs(evidence_doc)
    initials = initial_doc["arms"]
    original_sha = H.sha
    try:
        H.sha = lambda path, lf=False: source["fixed_eval_sha256_lf"] if Path(path) == H.FIXED_EVAL else original_sha(path, lf)
        numeric = _summarize_public(reports, initials, originals, source)
    finally:
        H.sha = original_sha
    if result.get("replay_scope") != REPLAY_SCOPE or result.get("numeric") != numeric:
        raise ValueError("published result does not replay from public numeric evidence")
    total_parent_wall = sum(row["total_s"] for row in costs.values())
    if not math.isfinite(total_parent_wall):
        raise ValueError("public aggregate parent wall time is invalid")
    return {"status": numeric["status"], "replay_scope": REPLAY_SCOPE,
            "behavior": numeric["behavior"], "verdicts": numeric["verdicts"],
            "parent_wall_scope": PARENT_WALL_SCOPE, "parent_wall_s_by_arm": costs,
            "total_parent_wall_s": total_parent_wall}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("tag")
    export_parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    result = export(args.tag, args.isaaclab) if args.command == "export" else replay(args.folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
