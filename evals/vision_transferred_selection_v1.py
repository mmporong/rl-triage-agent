"""전송된 native-pull 24개 입력을 기존 V05 선택 worker에 연결한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _folder in (ROOT / "bench", ROOT / "evals", ROOT / "src"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import run_vision_brev_batch_v1 as SOURCE  # pyright: ignore[reportMissingImports]
import vision_selection_call_v1 as CALL
import vision_selection_comparison_v1 as METHOD
import vision_selection_execution_v1 as EXECUTION
import vision_selection_worker_v1 as WORKER
import vision_transferred_input_acceptance_v3 as ACCEPT

SCHEMA = "vision_transferred_selection_locations_v1"
INPUT_SCHEMA = "vision_transferred_selection_input_v1"
PROTOCOL_SCHEMA = "vision_selection_execution_protocol_v1"
SELF = "evals/vision_transferred_selection_v1.py"
SOURCE_PROTOCOL = "bench/protocols/vision_brev_source_20261026_v8.json"
METHOD_PROTOCOL = "bench/protocols/vision_selection_comparison_v1.json"
API = {
    "total_cap": 72,
    "logical_decisions": 24,
    "max_calls_per_decision": 3,
    "semantic_feedback": 1,
    "infra_retry": 1,
    "sdk_retries": 0,
}
SEED_GROUP = {2220: "source-01", 2221: "source-02"}
ROLE = {"NONE": "normal", "F1": "registered-f1"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _read(path: Path) -> Any:
    target = Path(path).resolve(strict=True)
    if not target.is_file() or target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError(f"canonical regular file이 아니다: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def _sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, value: Any) -> None:
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _relative(repo: Path, value: str) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute() or "\\" in value or ".." in Path(value).parts:
        raise ValueError("repo-relative 경로가 아니다")
    path = (repo / value).resolve(strict=True)
    path.relative_to(repo)
    return path


def _ref(repo: Path, path: Path, *, frozen: bool = False) -> dict[str, str]:
    target = Path(path).resolve(strict=True)
    value = {"path": target.relative_to(repo).as_posix(), "sha256": _sha(target)}
    if frozen:
        value["status"] = "frozen"
    return value


def _absolute_ref(value: object, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} reference 형식이 다르다")
    path = Path(str(value["path"])).resolve(strict=True)
    if _sha(path) != value["sha256"]:
        raise ValueError(f"{label} SHA가 다르다")
    return path


def _protocol_ref(repo: Path, relative: str) -> tuple[Path, dict[str, str]]:
    path = _relative(repo, relative)
    return path, {"path": relative, "sha256_lf": _sha(path, lf=True)}


def _merge_hashes(repo: Path, target: dict[str, str], values: object, label: str) -> None:
    if not isinstance(values, dict):
        raise TypeError(f"{label} SHA map이 아니다")
    for name, digest in values.items():
        path = _relative(repo, name)
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None or _sha(path, lf=True) != digest:
            raise ValueError(f"{label} 현재 LF bytes가 다르다: {name}")
        if name in target and target[name] != digest:
            raise ValueError(f"{label} 경로 SHA가 충돌한다: {name}")
        target[name] = digest


def _source_maps(repo: Path, method: dict[str, Any], source: dict[str, Any],
                 method_path: Path, source_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    sources: dict[str, str] = {}
    evidence: dict[str, str] = {}
    _merge_hashes(repo, sources, method.get("source_sha256_lf"), "method source")
    _merge_hashes(repo, evidence, method.get("evidence_sha256_lf"), "method evidence")
    _merge_hashes(repo, sources, source.get("source_sha256_lf"), "training source")
    recording = source.get("recording")
    if not isinstance(recording, dict):
        raise TypeError("training source recording이 없다")
    _merge_hashes(repo, sources, recording.get("source_files_lf"), "training recording source")
    runtime = set(WORKER.REQUIRED_SOURCES) | {
        SELF,
        "evals/vision_selection_execution_v1.py",
        "evals/vision_transferred_input_acceptance_v1.py",
        "evals/vision_transferred_input_acceptance_v2.py",
        "evals/vision_transferred_input_acceptance_v3.py",
    }
    for relative in runtime:
        path = _relative(repo, relative)
        digest = _sha(path, lf=True)
        if relative in sources and sources[relative] != digest:
            raise ValueError(f"runtime source 결합이 다르다: {relative}")
        sources[relative] = digest
    for path in (method_path, source_path):
        relative = path.relative_to(repo).as_posix()
        digest = _sha(path, lf=True)
        if relative in sources and sources[relative] != digest:
            raise ValueError(f"parent protocol 결합이 다르다: {relative}")
        evidence[relative] = digest
    # 두 부모가 같은 bytes를 각각 runtime source와 review evidence로 분류한 경우에는
    # production worker가 요구하는 단일 경계에서 runtime source 쪽을 보존한다.
    overlap = set(sources) & set(evidence)
    if any(sources[name] != evidence[name] for name in overlap):
        raise ValueError("source/evidence 중복 경로의 SHA가 충돌한다")
    for name in overlap:
        del evidence[name]
    return dict(sorted(sources.items())), dict(sorted(evidence.items()))


def _verified_acceptance(receipt_path: Path, *, verify_raw_rgb: bool,
                         accept_pair: Callable[..., dict[str, Any]] = ACCEPT.accept_pair) -> dict[str, Any]:
    receipt = _read(receipt_path)
    if (not isinstance(receipt, dict) or receipt.get("schema") != "vision_transferred_input_acceptance_v3"
            or receipt.get("cpu_preparation") != "PASS" or receipt.get("policy_count") != 4
            or receipt.get("payload_count") != 24 or receipt.get("complete_four_policy_set") is not True):
        raise ValueError("acceptance v3 완전한 4 policy/24 payload 영수증이 아니다")
    use = receipt.get("use_scope_attestation")
    domains = receipt.get("mechanical_verification")
    if (not isinstance(domains, dict) or set(domains) != set(ACCEPT.MECHANICAL_DOMAINS)
            or any(value is not True for value in domains.values())
            or not isinstance(use, dict) or use.get("scopes") != list(ACCEPT.USE_SCOPE)
            or use.get("behavior_labels_created") is not False
            or use.get("behavior_ground_truth_assigned") is not False
            or use.get("model_use_authorized") is not False
            or use.get("used_for_decision") is not False):
        raise ValueError("acceptance v3 사용 경계가 다르다")
    fresh = []
    for pair in receipt.get("pairs", []):
        fresh.append(accept_pair(
            _absolute_ref(pair.get("protocol"), "acceptance protocol"),
            _absolute_ref(pair.get("capture_receipt"), "capture receipt"),
            _absolute_ref(pair.get("training_receipt"), "training receipt"),
            seed=pair.get("seed"), verify_raw_rgb=verify_raw_rgb,
        ))
    if fresh != receipt.get("pairs"):
        raise ValueError("acceptance v3 host 재검증 결과가 영수증과 다르다")
    seeds = [pair["seed"] for pair in fresh]
    if seeds != [2220, 2221]:
        raise ValueError("전송 입력 seed 순서가 2220, 2221이 아니다")
    return receipt


def _policy_catalog(acceptance: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in acceptance["pairs"]:
        group = SEED_GROUP.get(pair["seed"])
        if group is None:
            raise ValueError("등록되지 않은 source seed다")
        capture = _read(_absolute_ref(pair["capture_receipt"], "capture receipt"))
        captures = {row["logical_id"]: row for row in capture.get("policies", [])}
        for policy in pair["policies"]:
            role = ROLE.get(policy.get("source_role"))
            if role is None or policy["logical_id"] not in captures:
                raise ValueError("policy role 또는 capture logical ID가 다르다")
            raw_capture = captures[policy["logical_id"]]
            source_inputs = raw_capture.get("source_inputs", {})
            trace = _absolute_ref(source_inputs.get("raw_trace"), "raw trace")
            video = _absolute_ref(source_inputs.get("video"), "video")
            found[group, role] = {**policy, "seed": pair["seed"], "trace": trace, "video": video}
    expected = {(group, role) for group in METHOD.SEED_GROUPS for role in METHOD.ROLES}
    if set(found) != expected:
        raise ValueError("4 policy catalog가 V05 design과 다르다")
    return found


def _payload(policy: dict[str, Any], window: str, view: str) -> dict[str, Any]:
    number = METHOD.WINDOWS.index(window) + 1
    matches = [row for row in policy["payloads"] if row.get("window") == number and row.get("view") == view]
    if len(matches) != 1:
        raise ValueError("window/view payload가 유일하지 않다")
    return matches[0]


def _lineage(policy: dict[str, Any], window: str) -> dict[str, Any]:
    number = METHOD.WINDOWS.index(window) + 1
    matches = [row for row in policy["nested_lineage"] if row.get("window") == number]
    if len(matches) != 1:
        raise ValueError("window lineage가 유일하지 않다")
    return matches[0]


def _freeze_payload(repo: Path, payload: dict[str, Any], output: Path) -> dict[str, str]:
    bundle_path = _absolute_ref(payload.get("bundle"), "selection bundle")
    messages_path = _absolute_ref(payload.get("messages"), "selection messages")
    bundle, messages = _read(bundle_path), _read(messages_path)
    if (bundle.get("schema") != "vision_selection_bundle_v1" or bundle.get("view_kind") != payload.get("view")
            or not isinstance(messages, list)):
        raise ValueError("selection bundle/messages 형식이 다르다")
    methods = {name: _sha(_relative(repo, name)) for name in CALL._METHODS}
    _write_new(output / "bundle.json", bundle)
    _write_new(output / "messages.json", {"messages": messages})
    capsule = {"schema": CALL.CAPSULE_SCHEMA, "bundle_sha256": _sha(output / "bundle.json"),
               "messages_sha256": _sha(output / "messages.json"), "method_sha256": methods}
    _write_new(output / "capsule.json", capsule)
    capsule_path = (output / "capsule.json").resolve(strict=True)
    reference = {"path": str(capsule_path), "sha256": _sha(capsule_path), "status": "frozen"}
    CALL.prepared(reference)
    return reference


def materialize(repo_root: Path, acceptance_receipt: Path, output_root: Path, protocol_output: Path,
                *, execution_id: str, protocol_id: str, model_use_authorized: bool = False,
                approval_id: str | None = None, verify_raw_rgb: bool = True,
                accept_pair: Callable[..., dict[str, Any]] = ACCEPT.accept_pair) -> dict[str, Any]:
    """검증된 24 payload를 고정 V05 순서의 capsule과 실행 protocol로 만든다."""
    repo = Path(repo_root).resolve(strict=True)
    output = Path(output_root).resolve()
    protocol_path = Path(protocol_output).resolve()
    output.relative_to(repo)
    protocol_path.relative_to(repo)
    if output.exists() or protocol_path.exists():
        raise FileExistsError("materialization 출력은 새 경로여야 한다")
    if model_use_authorized != bool(approval_id):
        raise ValueError("model use 승인과 approval ID를 함께 제공해야 한다")
    acceptance_path = Path(acceptance_receipt).resolve(strict=True)
    acceptance = _verified_acceptance(
        acceptance_path, verify_raw_rgb=verify_raw_rgb, accept_pair=accept_pair,
    )
    policies = _policy_catalog(acceptance)
    method_path, method_ref = _protocol_ref(repo, METHOD_PROTOCOL)
    source_path, source_ref = _protocol_ref(repo, SOURCE_PROTOCOL)
    method = _read(method_path)
    source = SOURCE.validate_protocol(_read(source_path))
    if method.get("inputs", {}).get("slots") != METHOD.design():
        raise ValueError("V05 method design이 현재 코드와 다르다")
    sources, evidence = _source_maps(repo, method, source, method_path, source_path)
    rows: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    audit_paths: dict[tuple[str, str], Path] = {}
    for slot in METHOD.design():
        key = slot["seed_group"], slot["source_role"]
        policy = policies[key]
        if key not in audit_paths:
            audit_path = output / f"{key[0]}-{key[1]}" / "camera_pose_audit.json"
            # 기존 verifier의 계산을 재사용하되 옛 readiness 승인 필드는 만들지 않는다.
            import vision_selection_sources_v1 as OLD_INPUTS
            _write_new(audit_path, OLD_INPUTS.camera_pose_audit(policy["trace"]))
            audit_paths[key] = audit_path
        lineage = _lineage(policy, slot["window"])
        for view in slot["view_order"]:
            payload = _payload(policy, slot["window"], view)
            call_id = f"choice-{len(rows) + 1:02d}"
            capsule = _freeze_payload(repo, payload, output / slot["slot_id"] / f"capsule-{view}")
            frozen = CALL.prepared(capsule)
            bundle = frozen["bundle"]
            source_meta = bundle.get("source", {})
            if (frozen["checkpoint_sha256"] != policy["checkpoint_sha256"]
                    or source_meta.get("video_sha256") != _sha(policy["video"])):
                raise ValueError("capsule checkpoint/video가 acceptance와 다르다")
            input_manifest = output / slot["slot_id"] / f"input-{view}.json"
            _write_new(input_manifest, {
                "schema": INPUT_SCHEMA, "acceptance": _ref(repo, acceptance_path),
                "logical_id": policy["logical_id"], "window": payload["window"], "view": view,
                "bundle_source_sha256": payload["bundle"]["sha256"],
                "messages_source_sha256": payload["messages"]["sha256"],
                "capsule": capsule,
            })
            row = {
                "slot_id": slot["slot_id"], "seed_group": slot["seed_group"],
                "source_role": slot["source_role"], "window": slot["window"], "view_kind": view,
                "call_id": call_id, "source_seed": policy["seed"],
                "source_lineage_id": lineage["source_lineage_id"],
                "parent_lineage_id": lineage["parent_lineage_id"], "split": "comparison",
                "method_protocol_sha256": method_ref["sha256_lf"],
                "source_protocol_sha256": source_ref["sha256_lf"],
                "capsule_method_raw_sha256": deepcopy(frozen["capsule"]["method_sha256"]),
                "checkpoint_sha256": policy["checkpoint_sha256"],
                "video_sha256": source_meta["video_sha256"], "trace_sha256": _sha(policy["trace"]),
                "contract_sha256": source_meta["contract_sha256"],
                "snapshot_sha256": policy["snapshot"]["sha256"],
                "capsule_sha256": capsule["sha256"],
                "messages_sha256": frozen["capsule"]["messages_sha256"],
                "camera_pose_audit_sha256": _sha(audit_paths[key]),
            }
            rows.append(row)
            locations.append({"call_id": call_id,
                              "run_id": f"vision-selection-{execution_id}-{call_id}",
                              "input_manifest": _ref(repo, input_manifest, frozen=True),
                              "capsule": capsule})
    checked = METHOD.validate_inputs(rows, method_protocol_sha256=method_ref["sha256_lf"],
                                     source_protocol_sha256=source_ref["sha256_lf"])
    locations_path = output / "input_locations.json"
    _write_new(locations_path, {
        "schema": SCHEMA, "execution_id": execution_id,
        "acceptance": _ref(repo, acceptance_path),
        "use_attestation": {"model_use_authorized": model_use_authorized,
                            "approval_id": approval_id, "used_for_decision": False},
        "rows": locations,
    })
    protocol = {
        "schema": PROTOCOL_SCHEMA, "protocol_id": protocol_id, "execution_id": execution_id,
        "phase": "comparison", "method_protocol": method_ref, "source_protocol": source_ref,
        "rows": checked, "source_sha256_lf": sources, "evidence_sha256_lf": evidence,
        "input_locations_sha256": _sha(locations_path), "api": deepcopy(API),
    }
    _write_new(protocol_path, protocol)
    verify_input_verifier(repo, protocol, locations_path, verify_raw_rgb=verify_raw_rgb,
                          accept_pair=accept_pair)
    return {"protocol": _ref(repo, protocol_path), "input_locations": _ref(repo, locations_path),
            "rows": len(rows), "model_use_authorized": model_use_authorized}


def verify_input_verifier(root: Path, protocol: dict[str, Any], input_locations: Path, *,
                          verify_raw_rgb: bool = True,
                          accept_pair: Callable[..., dict[str, Any]] = ACCEPT.accept_pair) -> dict[str, Any]:
    """WORKER.prepare에 주입하는 acceptance-v3 host attestation verifier다."""
    repo = Path(root).resolve(strict=True)
    path = Path(input_locations).resolve(strict=True)
    path.relative_to(repo)
    value = _read(path)
    if (not isinstance(value, dict) or set(value) != {"schema", "execution_id", "acceptance",
            "use_attestation", "rows"} or value.get("schema") != SCHEMA
            or value.get("execution_id") != protocol.get("execution_id")
            or _sha(path) != protocol.get("input_locations_sha256")):
        raise ValueError("transferred input locations 결합이 다르다")
    acceptance_path = _relative(repo, value["acceptance"]["path"])
    if _sha(acceptance_path) != value["acceptance"]["sha256"]:
        raise ValueError("acceptance receipt SHA가 다르다")
    acceptance = _verified_acceptance(acceptance_path, verify_raw_rgb=verify_raw_rgb,
                                      accept_pair=accept_pair)
    if acceptance.get("complete_four_policy_set") is not True:
        raise ValueError("acceptance payload가 완전하지 않다")
    plan_rows = value.get("rows")
    protocol_rows = protocol.get("rows")
    if not isinstance(plan_rows, list) or not isinstance(protocol_rows, list) or len(plan_rows) != 24:
        raise ValueError("24개 input row가 아니다")
    verified_rows = []
    manifests: dict[str, str] = {}
    for expected, location in zip(protocol_rows, plan_rows, strict=True):
        if not isinstance(location, dict) or set(location) != {"call_id", "run_id", "input_manifest", "capsule"}:
            raise ValueError("input location row 형식이 다르다")
        call_id = expected["call_id"]
        if location["call_id"] != call_id or location["run_id"] != f"vision-selection-{protocol['execution_id']}-{call_id}":
            raise ValueError("call/run 순서가 protocol과 다르다")
        manifest_path = _relative(repo, location["input_manifest"]["path"])
        if (_sha(manifest_path) != location["input_manifest"]["sha256"]
                or location["input_manifest"].get("status") != "frozen"):
            raise ValueError("input manifest가 동결되지 않았다")
        manifest = _read(manifest_path)
        if manifest.get("schema") != INPUT_SCHEMA or manifest.get("acceptance") != value["acceptance"]:
            raise ValueError("input manifest acceptance 결합이 다르다")
        frozen = CALL.prepared(location["capsule"])
        if (manifest.get("capsule") != location["capsule"]
                or frozen["reference"]["sha256"] != expected["capsule_sha256"]
                or frozen["view_kind"] != expected["view_kind"]
                or frozen["checkpoint_sha256"] != expected["checkpoint_sha256"]):
            raise ValueError("frozen capsule과 protocol row가 다르다")
        manifests[call_id] = _sha(manifest_path)
        verified_rows.append({"call_id": call_id, "run_id": location["run_id"],
                              "capsule": deepcopy(location["capsule"])})
    return {"schema": WORKER.VERIFIED_INPUTS_SCHEMA, "execution_id": protocol["execution_id"],
            "rows": verified_rows, "input_manifest_sha256": manifests,
            "source_sha256_lf": deepcopy(protocol["source_sha256_lf"]),
            "evidence_sha256_lf": deepcopy(protocol["evidence_sha256_lf"]),
            "input_locations_sha256": _sha(path)}


def _verifier(*, verify_raw_rgb: bool = True) -> Callable[[Path, dict[str, Any], Path], dict[str, Any]]:
    return lambda root, protocol, locations: verify_input_verifier(
        root, protocol, locations, verify_raw_rgb=verify_raw_rgb,
    )


def prepare(protocol: Path, protocol_sha256_lf: str, source_anchor: str, input_locations: Path,
            *, root: Path = ROOT, verify_raw_rgb: bool = True) -> dict[str, Any]:
    return WORKER.prepare(protocol, protocol_sha256_lf, source_anchor, input_locations,
                          root=root, input_verifier=_verifier(verify_raw_rgb=verify_raw_rgb))


def advance(protocol: Path, protocol_sha256_lf: str, source_anchor: str, input_locations: Path,
            call_id: str, *, root: Path = ROOT, verify_raw_rgb: bool = True) -> tuple[dict[str, Any], int]:
    locations = _read(input_locations)
    use = locations.get("use_attestation", {})
    if use != {"model_use_authorized": True, "approval_id": use.get("approval_id"),
               "used_for_decision": False} or not use.get("approval_id"):
        return {"status": "terminal_hold", "reason": "model_use_approval_pending",
                "call_id": call_id, "provider_calls": 0}, WORKER.EXIT_CODES["terminal_hold"]
    return WORKER.advance(protocol, protocol_sha256_lf, source_anchor, input_locations, call_id,
                          root=root, input_verifier=_verifier(verify_raw_rgb=verify_raw_rgb))


def inspect(protocol: Path, protocol_sha256_lf: str, source_anchor: str, input_locations: Path,
            *, root: Path = ROOT, verify_raw_rgb: bool = True) -> dict[str, Any]:
    prepared = prepare(protocol, protocol_sha256_lf, source_anchor, input_locations,
                       root=root, verify_raw_rgb=verify_raw_rgb)
    repository, stage, manifest = CALL._plan(prepared["plan"])
    context = {"repository": repository, "stage": stage, "manifest": manifest,
               "prepared": prepared, "protocol_path": Path(protocol).resolve(strict=True),
               "protocol_sha256_lf": protocol_sha256_lf, "source_anchor": source_anchor,
               "input_locations": Path(input_locations).resolve(strict=True)}
    result = EXECUTION.inspect_run(context)
    if result.get("status") in {"blocked_pending", "paused_infra"}:
        result["decision"] = "HOLD"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("materialize")
    make.add_argument("--repo", type=Path, default=ROOT)
    make.add_argument("--acceptance", type=Path, required=True)
    make.add_argument("--output-root", type=Path, required=True)
    make.add_argument("--protocol-output", type=Path, required=True)
    make.add_argument("--execution-id", required=True)
    make.add_argument("--protocol-id", required=True)
    make.add_argument("--approve-model-use", action="store_true")
    make.add_argument("--approval-id")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--protocol", type=Path, required=True)
    common.add_argument("--protocol-sha256-lf", required=True)
    common.add_argument("--source-anchor", required=True)
    common.add_argument("--input-locations", type=Path, required=True)
    sub.add_parser("verify", parents=[common])
    sub.add_parser("prepare", parents=[common])
    run = sub.add_parser("advance", parents=[common])
    run.add_argument("--call-id", required=True)
    sub.add_parser("inspect", parents=[common])
    args = parser.parse_args(argv)
    if args.command == "materialize":
        result = materialize(args.repo, args.acceptance, args.output_root, args.protocol_output,
                             execution_id=args.execution_id, protocol_id=args.protocol_id,
                             model_use_authorized=args.approve_model_use, approval_id=args.approval_id)
    elif args.command in {"verify", "prepare"}:
        result = prepare(args.protocol, args.protocol_sha256_lf, args.source_anchor, args.input_locations)
        result = {"status": "verified", "plan": result["plan"]}
    elif args.command == "advance":
        result, code = advance(args.protocol, args.protocol_sha256_lf, args.source_anchor,
                               args.input_locations, args.call_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
        return code
    else:
        result = inspect(args.protocol, args.protocol_sha256_lf, args.source_anchor, args.input_locations)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
