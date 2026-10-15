"""결합 파일 blob anchor를 유지하며 두 seed의 재설정 개입을 실행한다."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "bench", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import record_evidence_media as M  # noqa: E402
import run_runtime_readback_smoke as P  # noqa: E402
import runtime_readback_smoke as S  # noqa: E402
import run_recovery_reset as V1  # noqa: E402

_V1_VALIDATE_MANIFEST = V1.validate_manifest
_V1_VERIFY_TRAINING = V1.verify_training
_V1_VALIDATE_RESULT_PATH = V1._validate_result_path

read, write_new, sha, digest = V1.read, V1.write_new, V1.sha, V1.digest
validate_readback = V1.validate_readback
PROTOCOL = ROOT / "bench/protocols/recovery_reset_v2.json"
FIXED_EVAL = V1.FIXED_EVAL
PRIVATE = ROOT / "bench/private/codex_support_20261015/task6/reset_v2/results"
SEEDS = (2026, 2028)
ARMS = V1.ARMS
CODE = V1.CODE | {
    "evals/run_recovery_reset_v2.py",
    "bench/train_with_recovery_reset_v2.py",
}
OLD_FOLDER = ROOT / "bench/private/codex_support_20261015/task1/runner/results/reward_reset_v1_s2026_20261015"
OLD_CONSUMED = ROOT / "bench/private/codex_support_20261015/task1/runner/results/consumed/70d1a9ada10226cb0b7b0c8da4f3694cd97e0de4ef2325cd1f6e867713344231.json"
OLD_EVIDENCE_PATHS = {
    "request": OLD_FOLDER / "request.json",
    "approval": OLD_FOLDER / "approval.json",
    "failure": OLD_FOLDER / "failure.json",
    "result": OLD_FOLDER / "result.json",
    "arm_consumption": OLD_FOLDER / "optimizer_only/consumed.json",
    "gpu_before": OLD_FOLDER / "optimizer_only/gpu_before.json",
    "manifest": OLD_FOLDER / "optimizer_only/train/manifest.json",
    "launch": OLD_FOLDER / "optimizer_only/train/launch.json",
    "optimizer_execution": OLD_FOLDER / "optimizer_only/train/execution.json",
    "optimizer_receipt": OLD_FOLDER / "optimizer_only/train/receipt.json",
    "intervention": OLD_FOLDER / "optimizer_only/train/intervention.json",
    "initial": OLD_FOLDER / "optimizer_only/train/initial.json",
    "correction": OLD_FOLDER / "optimizer_only/train/correction.json",
    "consumption": OLD_CONSUMED,
    "v1_protocol": V1.PROTOCOL,
}
OLD_EVIDENCE_SHA256 = {
    "request": "aaeeacafc656c40f9d6ecc496e76d744b6e4d084bc7c66b7d02f6df390b6c733",
    "approval": "018b22993494e158800094b69b4e845eae684bc146f0106ae6360a3356be42de",
    "failure": "571661aea50a25695e7e11b81c0e256735fd81d3ff2f0ab643736d3b7ae078f2",
    "result": "8ccd69d6e2c2b5841a128a9964ca0b577051c25576527358d066ca907391cf85",
    "arm_consumption": "cc06aa19267d3a8aec90431250c19467f036ce606eb34831d3cd449e355a11d3",
    "gpu_before": "e18e098dab154a1171545ce54c537331ee56060f309a5ef4def7eb43dc2dd6ad",
    "manifest": "0de206409a81e7789eb51517e2d0bcf08bc48770adfa6c517d14457f779ff679",
    "launch": "56a93aa6e77e81e70867dd35d23108b743a4a7c4297d5212b4555bc2f1134c2c",
    "optimizer_execution": "3d2210c42db83892e467953114fbec69f45538ce14971df5fc570adea92fd999",
    "optimizer_receipt": "3181d48bfd35c2eeff5299a48230982557906c427c657d9e9844a525ed183936",
    "intervention": "e9b6d0d2b37f908a168e80234c21f297e84a37428ef56d3463f7ef21c746597c",
    "initial": "f4ae7610e307607be17cbc551e5b32a544df8a51906223b7f070ebdfee2f60e3",
    "correction": "064aaa23ac2aa48185dc4a5dcfc6369bec4adeb439ad09123461e4010de5f608",
    "consumption": "6e135df75b69ba1c517c12f5d9900c6472f51807ed84d013f2d4147c11e06629",
    "v1_protocol": "767e7f293eacc15575899dd3d3b2152a7da90297a5afcc4756c0fa53170df153",
}

FIXED = {
    **deepcopy(V1.FIXED),
    "schema": "recovery_reset_protocol_v2",
    "revision": 2,
    "scope": "seen_two_seed_reward_fault_reset_intervention_development_v2",
    "design_disclosure": "v4 결과와 v1 seed 2026의 계약 invalid를 본 뒤 만든 실행 계약 보정이다",
    "registered_execution": {
        "2026": "v1 결과·소비를 보존하고 새 v2 namespace에서 세 arm 전체를 한 번 실행",
        "2028": "새 v2 namespace에서 세 arm 전체를 처음 실행",
    },
    "v1_invalid_reason": ("결합 밖 커밋으로 HEAD가 바뀌어 optimizer_only 300회 정상 종료 뒤 검증이 중단됐다. "
                          "고정 평가는 실행되지 않았고 다른 두 arm도 실행되지 않았다."),
    "selection_limit": "고정 평가 결과를 본 뒤 좋은 결과만 고르는 재시도가 아니라 실행 계약 invalid 보정이다",
    "v1_absent_artifacts": [
        "optimizer_only/eval", "optimizer_only/parent.json", "optimizer_only/gpu_after.json",
        "critic_reset", "actor_last_layer_reset",
    ],
    "source_anchor_policy": ("요청 commit이 현재 HEAD의 조상이고 결합 파일의 anchor·HEAD·index blob 및 "
                             "LF 정규화 작업 내용이 같을 때만 실행을 유지한다"),
    "semantic_retries": 0,
}
DYNAMIC = V1.DYNAMIC | {"v1_invalid_evidence"}


def _lf_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _git(repo: Path, *args: str, text: bool = False) -> Any:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=text, encoding="utf-8" if text else None,
                          errors="strict" if text else None).stdout


def _bound_relatives() -> tuple[str, ...]:
    return tuple(sorted(CODE | {PROTOCOL.relative_to(ROOT).as_posix(),
                                FIXED_EVAL.relative_to(ROOT).as_posix()}))


def _blob(repo: Path, revision: str, relative: str) -> str:
    return _git(repo, "rev-parse", f"{revision}:{relative}", text=True).strip()


def _bundle_sha(commit: str, blobs: dict[str, str]) -> str:
    material = json.dumps({"commit": commit, "blobs": blobs}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def verify_bound_at_head(repo: Path = ROOT, relatives: tuple[str, ...] | None = None) -> str:
    relatives = relatives or _bound_relatives()
    head = _git(repo, "rev-parse", "HEAD", text=True).strip()
    for relative in relatives:
        path = repo / relative
        _git(repo, "ls-files", "--error-unmatch", "--", relative)
        head_blob = _blob(repo, "HEAD", relative)
        index_blob = _blob(repo, "", relative)
        if index_blob != head_blob:
            raise ValueError(f"결합 파일 staged content가 HEAD와 다르다: {relative}")
        committed = _git(repo, "show", f"HEAD:{relative}")
        if not path.is_file() or _lf_bytes(path.read_bytes()) != _lf_bytes(committed):
            raise ValueError(f"결합 파일 working content가 HEAD와 다르다: {relative}")
    return head


def make_source_anchor(commit: str, repo: Path = ROOT,
                       relatives: tuple[str, ...] | None = None) -> dict[str, Any]:
    relatives = relatives or _bound_relatives()
    blobs = {relative: _blob(repo, commit, relative) for relative in relatives}
    return {"schema": "recovery_reset_source_anchor_v2", "commit": commit,
            "git_blobs": blobs, "bundle_sha256": _bundle_sha(commit, blobs)}


def verify_source_anchor(anchor: Any, repo: Path = ROOT,
                         relatives: tuple[str, ...] | None = None) -> dict[str, Any]:
    relatives = relatives or _bound_relatives()
    if not isinstance(anchor, dict) or set(anchor) != {"schema", "commit", "git_blobs", "bundle_sha256"}:
        raise ValueError("source anchor 형식이 다르다")
    commit, blobs = anchor.get("commit"), anchor.get("git_blobs")
    if anchor.get("schema") != "recovery_reset_source_anchor_v2" or not isinstance(commit, str):
        raise ValueError("source anchor schema나 commit이 다르다")
    expected_paths = set(relatives)
    if not isinstance(blobs, dict) or set(blobs) != expected_paths or anchor.get("bundle_sha256") != _bundle_sha(commit, blobs):
        raise ValueError("source anchor blob 집합이나 digest가 다르다")
    try:
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", commit, "HEAD"],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("source anchor가 없거나 현재 HEAD의 조상이 아니다") from error
    for relative in relatives:
        anchor_blob = _blob(repo, commit, relative)
        head_blob = _blob(repo, "HEAD", relative)
        index_blob = _blob(repo, "", relative)
        if blobs[relative] != anchor_blob or head_blob != anchor_blob or index_blob != anchor_blob:
            raise ValueError(f"결합 파일 Git blob이 anchor와 다르다: {relative}")
        anchor_bytes = _git(repo, "show", f"{commit}:{relative}")
        path = repo / relative
        if not path.is_file() or _lf_bytes(path.read_bytes()) != _lf_bytes(anchor_bytes):
            raise ValueError(f"결합 파일 working content가 anchor와 다르다: {relative}")
    return anchor


def _reject_public_absolute_paths(value: Any, location: str = "protocol") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_public_absolute_paths(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_public_absolute_paths(item, f"{location}[{index}]")
    elif isinstance(value, str):
        absolute = (re.match(r"^[A-Za-z]:[\\/]", value)
                    or re.match(r"^/(?:home|Users|mnt/[A-Za-z]/Users)/", value)
                    or re.match(r"^(?:\\\\|//)[^\\/]+[\\/]", value))
        if absolute:
            raise ValueError(f"공개 protocol에 사용자 절대경로가 있다: {location}")


def invalid_evidence(isaaclab: Path) -> dict[str, Any]:
    isaaclab = isaaclab.resolve()
    rows = {}
    for name, path in OLD_EVIDENCE_PATHS.items():
        if not path.is_file() or sha(path) != OLD_EVIDENCE_SHA256[name]:
            raise ValueError(f"v1 invalid 원본 증거가 없거나 바뀌었다: {name}")
        rows[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}
    execution, receipt = read(OLD_EVIDENCE_PATHS["optimizer_execution"]), read(OLD_EVIDENCE_PATHS["optimizer_receipt"])
    failure, result = read(OLD_EVIDENCE_PATHS["failure"]), read(OLD_EVIDENCE_PATHS["result"])
    completed_checkpoint = receipt.get("checkpoint", {})
    checkpoint_path = Path(completed_checkpoint.get("path", ""))
    try:
        checkpoint_relative = checkpoint_path.resolve().relative_to(isaaclab)
    except ValueError as error:
        raise ValueError("v1 completed checkpoint가 등록 Isaac Lab 밖에 있다") from error
    absent = [OLD_FOLDER / relative for relative in FIXED["v1_absent_artifacts"]]
    if (execution.get("natural_exit") is not True or execution.get("exit_code") != 0
            or execution.get("cleanup_verified") is not True or receipt.get("completed") is not True
            or receipt.get("actual_control_steps") != 7200 or receipt.get("aggregate_env_steps") != 29491200
            or receipt.get("final_iteration") != 598
            or completed_checkpoint.get("sha256") != "07b1b362e3cfe3c3d0a880412910daa5390d9dc7dd8fdeb7f599b8f2f491031b"
            or not checkpoint_path.is_file() or sha(checkpoint_path) != completed_checkpoint["sha256"]
            or failure != {"arm": "optimizer_only", "stage": "training", "error_type": "ValueError",
                           "detail": "request source commitment changed"}
            or result.get("status") != "partial" or result.get("failed_arm") != "optimizer_only"
            or set(result.get("labels", {}).values()) != {"invalid"}
            or any(path.exists() for path in absent)):
        raise ValueError("v1 invalid 실행 범위 증거가 감사 문서와 다르다")
    rows["completed_checkpoint"] = {
        "path": "<isaaclab>/" + checkpoint_relative.as_posix(),
        "sha256": completed_checkpoint["sha256"],
    }
    return rows


def source_maps(isaaclab: Path) -> dict:
    maps = V1.V4.source_maps(isaaclab)
    maps["source_files"] = {name: sha(ROOT / name, True) for name in sorted(CODE)}
    return maps


def prepare(isaaclab: Path) -> dict[str, Any]:
    v4 = read(V1.V4.PROTOCOL)
    contract = v4["correction_source_contract"]
    V1.check_contract(contract, isaaclab)
    document = {
        **deepcopy(FIXED), **source_maps(isaaclab),
        "correction_source_contract": contract,
        "fixed_eval_sha256_lf": sha(FIXED_EVAL, True),
        "resume_inputs": V1.V4.source_inputs(isaaclab),
        "v4_reference_hashes": V1.reference_hashes(),
        "v1_invalid_evidence": invalid_evidence(isaaclab),
    }
    _reject_public_absolute_paths(document)
    write_new(PROTOCOL, document)
    return {"status": "prepared_commit_before_execution", "path": str(PROTOCOL)}


def verify_protocol(isaaclab: Path) -> tuple[dict[str, Any], str]:
    protocol = read(PROTOCOL)
    if set(protocol) != set(FIXED) | DYNAMIC:
        raise ValueError("v2 protocol fields differ")
    if any(type(protocol.get(key)) is not type(value) or protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("v2 fixed contract changed")
    V1.check_contract(protocol["correction_source_contract"], isaaclab)
    if any(protocol.get(key) != value for key, value in source_maps(isaaclab).items()):
        raise ValueError("v2 training source hashes changed")
    if (protocol.get("fixed_eval_sha256_lf") != sha(FIXED_EVAL, True)
            or protocol.get("resume_inputs") != V1.V4.source_inputs(isaaclab)
            or protocol.get("v4_reference_hashes") != V1.reference_hashes()
            or protocol.get("v1_invalid_evidence") != invalid_evidence(isaaclab)):
        raise ValueError("v2 bound inputs changed")
    _reject_public_absolute_paths(protocol)
    if S._git_head(isaaclab) != FIXED["isaaclab_commit"] or not S._git_tracked_clean(isaaclab):
        raise ValueError("Isaac Lab checkout differs")
    return protocol, verify_bound_at_head()


@contextmanager
def _v1_context(anchor_commit: str | None = None) -> Iterator[None]:
    values = {"PROTOCOL": PROTOCOL, "PRIVATE": PRIVATE, "SEEDS": SEEDS, "CODE": CODE, "FIXED": FIXED}
    saved = {name: getattr(V1, name) for name in values}
    saved_verify = V1.verify_protocol
    for name, value in values.items():
        setattr(V1, name, value)
    if anchor_commit is not None:
        V1.verify_protocol = lambda isaaclab: (verify_protocol(isaaclab)[0], anchor_commit)
    try:
        yield
    finally:
        V1.verify_protocol = saved_verify
        for name, value in saved.items():
            setattr(V1, name, value)


def intervention_for_arm(arm: str) -> str:
    return V1.intervention_for_arm(arm)


def reset_seed(seed: int, intervention: str) -> int:
    return V1.reset_seed(seed, intervention)


def folder_for(tag: str) -> Path:
    if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", tag):
        raise ValueError("invalid recovery reset v2 tag")
    folder = PRIVATE / tag
    if folder.is_symlink() or folder.resolve().parent != PRIVATE.resolve():
        raise ValueError("training output escapes its v2 root")
    return folder


def manifest_for(folder: Path, request: dict, arm: str) -> dict:
    with _v1_context():
        return V1.manifest_for(folder, request, arm)


def validate_manifest(path: Path) -> tuple[dict, dict]:
    manifest = read(path)
    folder = folder_for(manifest.get("tag"))
    request = read(folder / "request.json")
    anchor = verify_source_anchor(request.get("source_anchor"))
    if request.get("source_commit_sha") != anchor["commit"] or manifest.get("source_commit_sha") != anchor["commit"]:
        raise ValueError("request/manifest source anchor가 다르다")
    if request.get("protocol_sha256_lf") != sha(PROTOCOL, True):
        raise ValueError("request v2 protocol hash가 다르다")
    with _v1_context(anchor["commit"]):
        return _V1_VALIDATE_MANIFEST(path)


def validate_training_configuration(log_dir: Path, manifest: dict) -> None:
    with _v1_context():
        V1.validate_training_configuration(log_dir, manifest)


def verify_training(folder: Path, arm: str, execution: dict) -> dict:
    saved = V1.validate_manifest
    with _v1_context():
        V1.validate_manifest = validate_manifest
        try:
            return _V1_VERIFY_TRAINING(folder, arm, execution)
        finally:
            V1.validate_manifest = saved


def _validate_report(report: dict, name: str, checkpoint_sha: str) -> dict:
    with _v1_context():
        return V1._validate_report(report, name, checkpoint_sha)


def validate_gpu_receipt(snapshot: Any, *, require_idle: bool) -> dict[str, Any]:
    if (not isinstance(snapshot, dict) or type(snapshot.get("busy")) is not bool
            or snapshot.get("admission_policy") != FIXED["gpu_admission_profile"]
            or not isinstance(snapshot.get("processes"), list)):
        raise ValueError("GPU receipt 형식이나 profile이 다르다")
    if require_idle and snapshot["busy"]:
        raise ValueError("GPU 입장 receipt가 busy다")
    return snapshot


def summarize(seed: int, reports: dict, failed_arm: str | None = None) -> dict:
    with _v1_context():
        return V1.summarize(seed, reports, failed_arm)


def execute(tag: str, seed: int, isaaclab: Path) -> dict[str, Any]:
    isaaclab = isaaclab.resolve()
    protocol, commit = verify_protocol(isaaclab)
    if seed not in SEEDS:
        raise ValueError("seed is not registered in v2")
    folder = folder_for(tag)
    key = digest({"protocol": sha(PROTOCOL, True), "seed": seed, "namespace": "recovery_reset_v2"})
    consumption = PRIVATE / "consumed" / f"{key}.json"
    if folder.exists() or consumption.exists():
        raise FileExistsError("v2 seed or tag has already been consumed")
    log_root = isaaclab / "logs/rsl_rl/unitree_go2_flat"
    for arm in ARMS:
        if any(path.name.endswith(f"_{tag}_{arm}") for path in log_root.iterdir()):
            raise FileExistsError("an SDK run already exists for this v2 arm")
    for path in (ROOT, isaaclab, folder):
        if any(character in str(path) for character in '\r\n"%&^'):
            raise ValueError("unsafe Windows batch path")
    admission = validate_gpu_receipt(M.query_gpu(FIXED["gpu_admission_profile"]), require_idle=False)
    if admission["busy"]:
        return {"status": "gpu_busy_no_launch", "sdk_jobs_launched": 0}
    anchor = make_source_anchor(commit)
    verify_source_anchor(anchor)
    request = {"tag": tag, "seed": seed, "source_commit_sha": commit,
               "source_anchor": anchor, "protocol_sha256_lf": sha(PROTOCOL, True),
               "isaaclab_root": str(isaaclab.resolve()),
               "selection_source": "registered_v2_contract_invalid_remediation_not_result_selection",
               "bundle_key": key}
    write_new(folder / "request.json", request)
    write_new(folder / "approval.json", {"request_sha256": sha(folder / "request.json"),
                                         "actor": FIXED["approval_actor"], "identity_attested": False})
    write_new(consumption, {"request_sha256": sha(folder / "request.json"), "tag": tag,
                            "source_anchor_sha256": anchor["bundle_sha256"]})
    reports, costs, failed_arm = {}, {}, None
    for arm in ARMS:
        stage = "preflight"
        try:
            verify_source_anchor(anchor)
            current, _ = verify_protocol(isaaclab)
            if current != protocol:
                raise ValueError("bound protocol changed during v2 experiment")
            before = validate_gpu_receipt(M.query_gpu(FIXED["gpu_admission_profile"]), require_idle=True)
            arm_dir = folder / arm
            training, evaluation = arm_dir / "train", arm_dir / "eval"
            write_new(arm_dir / "gpu_before.json", before)
            write_new(arm_dir / "consumed.json", {"approval_sha256": sha(folder / "approval.json"),
                                                  "arm": arm, "train_limit": 1, "eval_limit": 1})
            write_new(training / "manifest.json", manifest_for(folder, request, arm))
            argv = [str(isaaclab / "_isaac_sim/python.bat"),
                    str(ROOT / "bench/train_with_recovery_reset_v2.py"),
                    "--manifest", str(training / "manifest.json")]
            stage = "training"
            execution = P.run_owned(training, argv, isaaclab, FIXED["train_watchdog_s"])
            write_new(training / "execution.json", execution)
            receipt = verify_training(folder, arm, execution)
            evaluation.mkdir()
            write_new(evaluation / "jobs.json", {"action_scale": FIXED["action_scale"], "jobs": [{
                "name": arm, "checkpoint": receipt["checkpoint"]["path"],
                "output": str(evaluation / "report.json")} ]})
            validate_gpu_receipt(M.query_gpu(FIXED["gpu_admission_profile"]), require_idle=True)
            stage = "fixed_evaluation"
            argv = [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"),
                    "--protocol", str(FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"]
            eval_execution = P.run_owned(evaluation, argv, isaaclab, FIXED["eval_watchdog_s"])
            write_new(evaluation / "execution.json", eval_execution)
            if not V1.execution_valid(eval_execution):
                raise ValueError("fixed evaluation did not complete")
            reports[arm] = _validate_report(read(evaluation / "report.json"), arm,
                                            receipt["checkpoint"]["sha256"])
            verify_source_anchor(anchor)
            if sha(Path(receipt["checkpoint"]["path"])) != receipt["checkpoint"]["sha256"]:
                raise ValueError("checkpoint changed during evaluation")
            after = validate_gpu_receipt(M.query_gpu(FIXED["gpu_admission_profile"]), require_idle=False)
            write_new(arm_dir / "gpu_after.json", after)
            owned = set(execution["owned_process_identities"]) | set(eval_execution["owned_process_identities"])
            if any(str(row["pid"]) in owned for row in after["processes"]):
                raise ValueError("an owned GPU process remains")
            costs[arm] = {"train_parent_wall_s": execution["wall_s"],
                          "eval_parent_wall_s": eval_execution["wall_s"]}
            write_new(arm_dir / "parent.json", {
                "status": "pass", "source_anchor_sha256": anchor["bundle_sha256"],
                "train_launch_sha256": sha(training / "launch.json"),
                "eval_launch_sha256": sha(evaluation / "launch.json"),
                "gpu_before_sha256": sha(arm_dir / "gpu_before.json"),
                "gpu_after_sha256": sha(arm_dir / "gpu_after.json"),
                "training_receipt_sha256": sha(training / "receipt.json"),
                "intervention_sha256": sha(training / "intervention.json"),
                "fixed_report_sha256": sha(evaluation / "report.json"), **costs[arm]})
        except Exception as error:
            failed_arm = arm
            reports.pop(arm, None)
            write_new(folder / "failure.json", {"arm": arm, "stage": stage,
                                                "error_type": type(error).__name__, "detail": str(error)})
            break
    result = summarize(seed, reports, failed_arm)
    result.update(tag=tag, costs=costs, model_calls=0, source_commit_sha=commit,
                  source_anchor=anchor, protocol_sha256_lf=request["protocol_sha256_lf"],
                  consumption_sha256=sha(consumption))
    result["arm_evidence"] = {arm: {"parent_sha256": sha(folder / arm / "parent.json"),
                                    **read(folder / arm / "parent.json")}
                              for arm in ARMS if (folder / arm / "parent.json").is_file()}
    write_new(folder / "result.json", result)
    return result


def _validate_result_path(path: Path) -> tuple[dict, dict]:
    row = read(path.resolve())
    folder = folder_for(row.get("tag"))
    request = read(folder / "request.json")
    anchor = verify_source_anchor(request.get("source_anchor"))
    if (row.get("source_anchor") != anchor or row.get("source_commit_sha") != anchor["commit"]
            or request.get("source_commit_sha") != anchor["commit"]):
        raise ValueError("aggregate request/result source anchor가 다르다")
    with _v1_context(anchor["commit"]):
        validated, artifact = _V1_VALIDATE_RESULT_PATH(path)
    seed = validated["seed"]
    expected_key = digest({"protocol": sha(PROTOCOL, True), "seed": seed,
                           "namespace": "recovery_reset_v2"})
    if request.get("bundle_key") != expected_key:
        raise ValueError("aggregate request v2 consumption key가 다르다")
    consumption_path = PRIVATE / "consumed" / f"{expected_key}.json"
    expected_consumption = {"request_sha256": sha(folder / "request.json"), "tag": folder.name,
                            "source_anchor_sha256": anchor["bundle_sha256"]}
    if read(consumption_path) != expected_consumption:
        raise ValueError("aggregate v2 global consumption이 다르다")
    if validated.get("consumption_sha256") != sha(consumption_path):
        raise ValueError("aggregate result v2 consumption SHA가 다르다")
    isaaclab = Path(request["isaaclab_root"])
    for arm in ARMS:
        if validated["arm_evidence"][arm].get("source_anchor_sha256") != anchor["bundle_sha256"]:
            raise ValueError("aggregate parent source anchor가 다르다")
        arm_dir = folder / arm
        training, evaluation = arm_dir / "train", arm_dir / "eval"
        manifest_path = training / "manifest.json"
        train_launch = read(training / "launch.json")
        expected_train_launch = {
            "args": [str(isaaclab / "_isaac_sim/python.bat"),
                     str(ROOT / "bench/train_with_recovery_reset_v2.py"),
                     "--manifest", str(manifest_path)],
            "cwd": str(isaaclab),
        }
        eval_launch = read(evaluation / "launch.json")
        expected_eval_launch = {
            "args": [str(isaaclab / "_isaac_sim/python.bat"), str(ROOT / "evals/fixed_eval.py"),
                     "--protocol", str(FIXED_EVAL), "--jobs", str(evaluation / "jobs.json"), "--headless"],
            "cwd": str(isaaclab),
        }
        before, after = read(arm_dir / "gpu_before.json"), read(arm_dir / "gpu_after.json")
        validate_gpu_receipt(before, require_idle=True)
        validate_gpu_receipt(after, require_idle=False)
        if train_launch != expected_train_launch or eval_launch != expected_eval_launch:
            raise ValueError("aggregate v2 launch argv/cwd가 다르다")
        parent = read(arm_dir / "parent.json")
        expected_hashes = {
            "train_launch_sha256": sha(training / "launch.json"),
            "eval_launch_sha256": sha(evaluation / "launch.json"),
            "gpu_before_sha256": sha(arm_dir / "gpu_before.json"),
            "gpu_after_sha256": sha(arm_dir / "gpu_after.json"),
        }
        if any(parent.get(key) != value for key, value in expected_hashes.items()):
            raise ValueError("aggregate v2 launch/GPU evidence SHA가 다르다")
    return validated, artifact


def aggregate(result_paths: list[Path]) -> dict[str, Any]:
    validated = [_validate_result_path(path) for path in result_paths]
    rows = [item[0] for item in validated]
    if sorted(row["seed"] for row in rows) != list(SEEDS) or len(rows) != len(SEEDS):
        raise ValueError("aggregate requires both v2 seeds once")
    if len({row["protocol_sha256_lf"] for row in rows}) != 1:
        raise ValueError("aggregate v2 results use different protocols")
    bundle_maps = [row["source_anchor"]["git_blobs"] for row in rows]
    if bundle_maps[0] != bundle_maps[1]:
        raise ValueError("aggregate v2 bound source blobs differ")
    v4 = {}
    for seed in SEEDS:
        numeric = read(V1._reference_paths(seed)["result"])["numeric"]
        v4[str(seed)] = {arm: {"label": numeric["labels"][arm], "behavior": numeric["behavior"][arm],
                              "verdict": numeric["verdicts"][arm]}
                         for arm in FIXED["registered_v4_comparisons"]}
    return {"status": "complete", "seeds": list(SEEDS), "scope": FIXED["scope"],
            "source_anchors": {str(row["seed"]): row["source_anchor"] for row in rows},
            "bound_git_blobs": bundle_maps[0], "protocol_sha256_lf": rows[0]["protocol_sha256_lf"],
            "result_artifacts": {str(row["seed"]): artifact for row, (_, artifact) in zip(rows, validated)},
            "arms": {arm: {str(row["seed"]): row["labels"][arm] for row in rows} for arm in ARMS},
            "v4_comparisons": v4,
            "recovery_mode_change_criteria": FIXED["recovery_mode_change_criteria"],
            "causal_mechanism_identified": False, "generalization_proven": False,
            "gpu_saving_claimed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute", "aggregate"))
    parser.add_argument("--isaaclab", type=Path, default=Path.home() / "IsaacLab")
    parser.add_argument("--tag")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--result", type=Path, action="append")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.isaaclab)
    elif args.command == "verify":
        _, commit = verify_protocol(args.isaaclab)
        result = {"status": "pass", "current_head": commit}
    elif args.command == "execute":
        if args.tag is None or args.seed is None:
            parser.error("execute requires --tag and --seed")
        result = execute(args.tag, args.seed, args.isaaclab)
    else:
        if not args.result or len(args.result) != 2:
            parser.error("aggregate requires --result twice")
        result = aggregate(args.result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command in {"execute", "aggregate"} and result.get("status") != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
