"""검증된 v4 회복 방식 seed 결과를 공개 수치 재생 묶음으로 내보낸다.

비공개 실행을 실험 커밋 기준으로 다시 검증한 뒤 절대 경로·PID·로그를 뺀 고정 평가 원자료와
상태 지문 요약만 쓴다. replay는 공개 파일의 해시와 수치 판정을 다시 계산할 뿐 실행 프로세스의
신원을 재인증하지 않는다. v3 publisher는 공개 결과의 소스 결합 때문에 바꾸지 않고 함수만 재사용한다.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import publish_training_correction as Q  # noqa: E402
import run_training_correction_v4 as H  # noqa: E402
from rl_triage import grounded_recovery as C  # noqa: E402


RESULTS = ROOT / "evals/results"
SCHEMA = "training_correction_v4_public_v1"
REPLAY_SCOPE = Q.REPLAY_SCOPE
SOURCE_COMMIT_SCOPE = Q.SOURCE_COMMIT_SCOPE
PARENT_WALL_SCOPE = Q.PARENT_WALL_SCOPE
PUBLISHER_SOURCES = {
    "evals/publish_training_correction_v4.py",
    "evals/publish_training_correction.py",
    "evals/run_training_correction_v4.py",
    "evals/run_training_correction.py",
    "src/rl_triage/grounded_recovery.py",
    "src/rl_triage/recovery.py",
}
COMMON_FILES = {
    "original_reports/clean_reference.json", "original_reports/fault.json",
    "initial_states.json", "training_evidence.json", "source_inputs.json", "result.json",
}
PUBLISHABLE_STATUS = ("complete", "partial")


def public_files(completed: list[str]) -> set[str]:
    return COMMON_FILES | {f"reports/{arm}.json" for arm in completed}


def _publisher_sources() -> dict[str, str]:
    return {relative: hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for relative in sorted(PUBLISHER_SOURCES)}


def _blob_lf(repo: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                               check=True, capture_output=True)
    return hashlib.sha256(completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


class _FrozenCommit:
    """H.P 대역. 결합 파일이 실험 커밋과 HEAD에서 같으면 HEAD 대신 실험 커밋을 돌려준다."""

    def __init__(self, module: Any, source_commit: str):
        self._module, self._source_commit = module, source_commit

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def verify_committed(self, repo: Path, paths: set[str]) -> str:
        head = self._module.verify_committed(repo, paths)
        if head == self._source_commit:
            return head
        ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                   self._source_commit, head], capture_output=True)
        if ancestor.returncode != 0:
            raise ValueError("experiment source commit is not an ancestor of HEAD")
        for relative in sorted(paths):
            if _blob_lf(repo, self._source_commit, relative) != _blob_lf(repo, head, relative):
                raise ValueError(f"bound source differs between the experiment commit and HEAD: {relative}")
        return self._source_commit


@contextlib.contextmanager
def frozen_source(source_commit: str) -> Iterator[None]:
    """실험 뒤 HEAD가 전진해도 결합 파일이 그대로면 v4 검증을 실험 커밋 기준으로 수행한다."""
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("experiment source commit is malformed")
    original = H.P
    H.P = _FrozenCommit(original, source_commit)
    try:
        yield
    finally:
        H.P = original


def _state_summary(value: dict) -> dict:
    if not isinstance(value, dict) or not {"sha256", "tensor_count", "scalar_count"} <= set(value):
        raise ValueError("initial state digest summary is incomplete")
    return {key: value[key] for key in ("sha256", "tensor_count", "scalar_count")}


def _checkpoint_public(row: dict | None, *, optimizer: bool) -> dict | None:
    if row is None:
        return None
    result = {"file": Path(row["path"]).name, "sha256": row["sha256"]}
    if optimizer:
        result["load_optimizer"] = row["load_optimizer"]
    return result


def _initial_public(initial: dict) -> dict:
    return {
        "recovery_mode": initial["recovery_mode"],
        "policy_state": _state_summary(initial["policy_state"]),
        "optimizer_state": _state_summary(initial["optimizer_state"]),
        "rng_state_before_learn": deepcopy(initial["rng_state_before_learn"]),
        "runtime_versions": deepcopy(initial["runtime_versions"]),
        "resume_checkpoint": _checkpoint_public(initial["resume_checkpoint"], optimizer=True),
        "resume_iteration": initial["resume_iteration"],
        "start_iteration": initial["start_iteration"],
        "source_checkpoint_state": deepcopy(initial["source_checkpoint_state"]),
        "std_reset": deepcopy(initial["std_reset"]),
    }


def _receipt_public(receipt: dict, receipt_sha256: str) -> dict:
    return {
        "receipt_sha256": receipt_sha256, "recovery_mode": receipt["recovery_mode"],
        "checkpoint": {"file": Path(receipt["checkpoint"]["path"]).name, "sha256": receipt["checkpoint"]["sha256"]},
        "configs": {key: {"file": Path(row["path"]).name, "sha256": row["sha256"]}
                    for key, row in receipt["configs"].items()},
        "resume_checkpoint": _checkpoint_public(receipt["resume_checkpoint"], optimizer=False),
        **{key: deepcopy(receipt[key]) for key in (
            "resume_iteration", "start_iteration", "final_iteration", "checkpoint_load_calls",
            "std_reset_applied", "resume_scope", "actual_control_steps", "aggregate_env_steps",
            "budget", "observed", "runtime_versions")},
    }


def _costs(evidence_doc: dict, completed: list[str]) -> dict[str, dict]:
    """v3 공개 비용과 같은 규칙(범위, 유한·비음수, 합계)을 v4 arm 이름에 적용한다."""
    result = {}
    for arm in completed:
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


def _summarize_public(reports: dict, initials: dict, originals: dict, source: dict, failed_arm: str | None) -> dict:
    result = H.summarize(source["seed"], reports, initials, originals, source["bands"],
                         source["runs"][H.FAULT]["original_failed"], failed_arm)
    if result["status"] not in PUBLISHABLE_STATUS:
        raise ValueError("invalid seed result cannot be published")
    return result


def _completed_arms(private_result: dict) -> tuple[list[str], str | None]:
    completed, failed_arm = private_result.get("completed_arms"), private_result.get("failed_arm")
    if (not isinstance(completed, list) or not completed or completed != list(H.ARMS[:len(completed)])
            or (failed_arm is not None and failed_arm != (H.ARMS[len(completed)] if len(completed) < len(H.ARMS) else None))
            or (failed_arm is None and len(completed) != len(H.ARMS))):
        raise ValueError("private result arm completion record differs")
    return completed, failed_arm


def export(tag: str, isaaclab: Path) -> dict:
    private = H.folder_for(tag)
    request = H.read(private / "request.json")
    commit = request.get("source_commit_sha")
    seed = request.get("seed")
    reports, initials, evidence = {}, {}, {}
    with frozen_source(commit):
        protocol, verified = H.verify_protocol(isaaclab)
        if (verified != commit or request.get("tag") != tag or seed not in H.SEEDS
                or request.get("protocol_sha256_lf") != H.sha(H.PROTOCOL, True)
                or Path(request.get("isaaclab_root", "")).resolve() != isaaclab.resolve()):
            raise ValueError("request is not bound to the verified experiment source")
        private_result = H.read(private / "result.json")
        if private_result.get("status") not in PUBLISHABLE_STATUS:
            raise ValueError("invalid seed result cannot be published")
        completed, failed_arm = _completed_arms(private_result)
        failure = private / "failure.json"
        if (failed_arm is None) == failure.exists() or (failed_arm and H.read(failure).get("arm") != failed_arm):
            raise ValueError("failure record differs from the private result")
        for arm in completed:
            training, evaluation = private / arm / "train", private / arm / "eval"
            manifest, arm_protocol = H.validate_manifest(training / "manifest.json")
            if arm_protocol != protocol or manifest["arm"] != arm or manifest["seed"] != seed:
                raise ValueError("arm manifest is not bound to this experiment")
            execution = H.read(training / "execution.json")
            receipt = H.verify_training(private, arm, execution)
            eval_execution = H.read(evaluation / "execution.json")
            if not H.execution_valid(eval_execution):
                raise ValueError("fixed evaluation execution is incomplete")
            report_path = evaluation / "report.json"
            reports[arm] = C.validate_fixed_report(H.read(report_path), {
                "name": arm, "checkpoint_file": f"model_{H.FINAL_ITERATION[H.ARM_SPECS[arm]['recovery_mode']]}.pt",
                "checkpoint_sha256": receipt["checkpoint"]["sha256"], "action_scale": H.FIXED["action_scale"],
                "protocol_sha256_lf": protocol["fixed_eval_sha256_lf"],
            })
            parent = H.read(private / arm / "parent.json")
            if (parent.get("status") != "pass"
                    or parent.get("training_receipt_sha256") != H.sha(training / "receipt.json")
                    or parent.get("fixed_report_sha256") != H.sha(report_path)):
                raise ValueError("arm parent binding differs")
            initials[arm] = _initial_public(H.read(training / "initial.json"))
            evidence[arm] = {**_receipt_public(receipt, H.sha(training / "receipt.json")),
                             "parent_wall_s": Q._parent_wall_cost(execution, eval_execution, parent)}
    originals = {"clean_reference": H.read(H.source_paths(isaaclab, seed, "NONE")["fixed_report"]),
                 "fault": H.read(H.source_paths(isaaclab, seed, H.FAULT)["fixed_report"])}
    inputs = protocol["resume_inputs"]
    source = {"seed": seed, "fault": H.FAULT, "bands": deepcopy(inputs["bands"]),
              "runtime_versions": deepcopy(inputs["runtime_versions"]),
              "summary_sha256_lf": inputs["summary_sha256_lf"],
              "runs": deepcopy(inputs["seeds"][str(seed)]["runs"]),
              "init_noise_std": inputs["init_noise_std"],
              "fixed_eval_sha256_lf": protocol["fixed_eval_sha256_lf"]}
    numeric = _summarize_public(reports, initials, originals, source, failed_arm)
    if (any(private_result.get(key) != numeric[key]
            for key in ("status", "seed", "labels", "label_reasons", "completed_arms", "failed_arm"))
            or private_result.get("tag") != tag or private_result.get("source_commit_sha") != commit
            or private_result.get("protocol_sha256_lf") != request["protocol_sha256_lf"]):
        raise ValueError("private result differs from replayed evidence")
    publisher_sources = _publisher_sources()
    artifacts = {
        **{f"reports/{arm}.json": reports[arm] for arm in completed},
        "original_reports/clean_reference.json": originals["clean_reference"],
        "original_reports/fault.json": originals["fault"],
        "initial_states.json": {"schema": SCHEMA, "arms": initials},
        "training_evidence.json": {"schema": SCHEMA, "arms": evidence},
        "source_inputs.json": {"schema": SCHEMA, **source},
        "result.json": {"schema": SCHEMA, "replay_scope": REPLAY_SCOPE, "tag": tag, "seed": seed,
                        "source_commit_sha": commit, "protocol_sha256_lf": request["protocol_sha256_lf"],
                        "source_commit_scope": SOURCE_COMMIT_SCOPE,
                        "publisher_sources": publisher_sources, "numeric": numeric},
    }
    if set(artifacts) != public_files(completed):
        raise AssertionError("public artifact set differs")
    for value in artifacts.values():
        Q._public_value(value)
    encoded = {name: Q._json_bytes(value) for name, value in artifacts.items()}
    manifest = {"schema": SCHEMA, "replay_scope": REPLAY_SCOPE, "publisher_sources": publisher_sources,
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(encoded.items())}}
    Q._public_value(manifest)
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
        stream.write(Q._json_bytes(manifest))
    return {"status": numeric["status"], "seed": seed, "labels": numeric["labels"],
            "folder": f"evals/results/{tag}", "replay_scope": REPLAY_SCOPE, "artifact_count": len(encoded)}


def _safe_document(folder: Path, name: str, expected: str) -> dict:
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
    document = Q._read(path)
    Q._public_value(document)
    return document


def replay(folder: Path) -> dict:
    if folder.is_symlink():
        raise ValueError("public result folder is missing or unsafe")
    folder = folder.resolve()
    if not folder.is_dir() or (folder / "artifact_manifest.json").is_symlink():
        raise ValueError("public result folder is missing or unsafe")
    manifest = Q._read(folder / "artifact_manifest.json")
    files = manifest.get("files")
    if (set(manifest) != {"schema", "replay_scope", "publisher_sources", "files"}
            or manifest.get("schema") != SCHEMA or manifest.get("replay_scope") != REPLAY_SCOPE
            or not isinstance(files, dict) or not COMMON_FILES <= set(files)):
        raise ValueError("public artifact manifest differs")
    published_sources = manifest.get("publisher_sources")
    if (not isinstance(published_sources, dict) or set(published_sources) != PUBLISHER_SOURCES
            or published_sources != _publisher_sources()):
        raise ValueError("publisher source LF-SHA binding differs")
    result = _safe_document(folder, "result.json", files["result.json"])
    numeric = result.get("numeric")
    completed = numeric.get("completed_arms") if isinstance(numeric, dict) else None
    if (not isinstance(completed, list) or not completed or completed != list(H.ARMS[:len(completed)])
            or set(files) != public_files(completed)):
        raise ValueError("public artifact set differs from the completed arms")
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}
    if actual != set(files) | {"artifact_manifest.json"}:
        raise ValueError("public artifact file set differs")
    documents = {name: _safe_document(folder, name, digest) for name, digest in files.items()}
    reports = {arm: documents[f"reports/{arm}.json"] for arm in completed}
    originals = {key: documents[f"original_reports/{key}.json"] for key in ("clean_reference", "fault")}
    initial_doc, evidence_doc = documents["initial_states.json"], documents["training_evidence.json"]
    source = documents["source_inputs.json"]
    if any(document.get("schema") != SCHEMA for document in (initial_doc, evidence_doc, source, result)):
        raise ValueError("public artifact schema differs")
    if (set(initial_doc) != {"schema", "arms"} or list(initial_doc.get("arms", {})) != completed
            or set(evidence_doc) != {"schema", "arms"} or list(evidence_doc.get("arms", {})) != completed):
        raise ValueError("public arm evidence schema differs")
    if (result.get("source_commit_scope") != SOURCE_COMMIT_SCOPE or result.get("publisher_sources") != published_sources
            or result.get("replay_scope") != REPLAY_SCOPE or result.get("seed") != source.get("seed")):
        raise ValueError("result and manifest bindings differ")
    costs = _costs(evidence_doc, completed)
    original_sha = H.sha
    try:
        H.sha = lambda path, lf=False: source["fixed_eval_sha256_lf"] if Path(path) == H.FIXED_EVAL else original_sha(path, lf)
        recomputed = _summarize_public(reports, initial_doc["arms"], originals, source, numeric.get("failed_arm"))
    finally:
        H.sha = original_sha
    if recomputed != numeric:
        raise ValueError("published result does not replay from public numeric evidence")
    return {"status": numeric["status"], "seed": numeric["seed"], "replay_scope": REPLAY_SCOPE,
            "labels": numeric["labels"], "label_reasons": numeric["label_reasons"],
            "behavior": numeric["behavior"], "verdicts": numeric["verdicts"],
            "mechanical_signal": numeric["mechanical_signal"]["values"],
            "recovery_mode_rule": numeric["recovery_mode_rule"],
            "parent_wall_scope": PARENT_WALL_SCOPE, "parent_wall_s_by_arm": costs,
            "total_parent_wall_s": sum(row["total_s"] for row in costs.values())}


def aggregate(folders: list[Path]) -> dict:
    """각 seed 공개 묶음을 replay한 뒤 arm별 라벨·예측·규칙 적중을 모은다."""
    rows = []
    for folder in folders:
        replay(folder)
        result = Q._read(folder.resolve() / "result.json")
        rows.append({**result["numeric"], "protocol_sha256_lf": result["protocol_sha256_lf"]})
    return {**H.aggregate(rows), "replay_scope": REPLAY_SCOPE}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("tag")
    export_parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("folder", type=Path)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export(args.tag, args.isaaclab)
    elif args.command == "replay":
        result = replay(args.folder)
    else:
        result = aggregate(args.folders)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
