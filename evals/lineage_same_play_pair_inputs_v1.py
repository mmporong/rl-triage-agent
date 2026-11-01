"""새 lineage source를 동일 Play 수치/영상 pair 입력으로 봉인한다.

결함명·seed·평가 라벨은 모델 입력에서 제외하고, 캡처가 없거나 품질 검사를
통과하지 못한 영상 arm은 HOLD로 남긴다. 이 모듈은 모델/API/GPU를 실행하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("evals", "bench", "src")]

import agentic_frame_quality_guard_v1 as QUALITY
import agentic_investigation_contract_v2 as A2
import lineage_five_policy_driver_v1 as DRIVER
import lineage_same_play_capture_v1 as CAPTURE  # pyright: ignore[reportMissingImports]
import lineage_source_inputs_v2 as INPUT
import local_same_play_pair_inputs_v1 as PAIR_INPUT
import local_same_play_pair_provider_v1 as PROVIDER
import selection_comparison_controller_v1 as C

SCHEMA = "lineage_same_play_pair_inputs_protocol_v1"
RESULT_SCHEMA = "lineage_same_play_pair_inputs_result_v1"
QUALITY_FLOOR = 10.0
QUALITY_POLICY = {
    "schema": QUALITY.POLICY_SCHEMA,
    "metric": QUALITY.METRIC,
    "unit": QUALITY.UNIT,
    "min_laplacian_variance": QUALITY_FLOOR,
}
PROJECTION = {
    "training_facts": 48, "runtime_facts_excluded": 16,
    "same_play_numeric_facts": 54, "image_counts": [0, 6],
    "source_indices_per_role": [0, 2, 5],
}
AUTHORITY = {"cpu_input_preparation": True, "api": False, "gpu": False,
             "probe": False, "correction": False, "retraining": False}
SOURCE_FILES = tuple(sorted({
    *PROVIDER.SOURCE_FILES, *CAPTURE.source_files(),
    "evals/lineage_same_play_pair_inputs_v1.py",
    "tests/test_lineage_same_play_pair_inputs_v1.py",
    "evals/lineage_five_policy_driver_v1.py",
    "tests/test_lineage_five_policy_driver_v1.py",
    "evals/lineage_source_inputs_v2.py", "tests/test_lineage_source_inputs_v2.py",
    "evals/agentic_frame_quality_guard_v1.py",
    "tests/test_agentic_frame_quality_guard_v1.py",
}))


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "JSON object required")
    return value


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(C.V.canonical(value)).hexdigest()


def _write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(C.V.canonical(value) + b"\n")


def _ref(path: Path) -> dict[str, str]:
    checked = Path(path).resolve(strict=True)
    return {"path": str(checked), "sha256": _sha(checked)}


def protocol_value(index_path: Path, case_id: str, *, source_anchor: str | None = None) -> dict[str, Any]:
    envelope = INPUT.make_case_packet(index_path, case_id, budget_left=3)
    return {
        "schema": SCHEMA,
        "status": "registered_before_pair_preparation" if source_anchor else "candidate_not_registered",
        "source_anchor": source_anchor,
        "sources_lf": {name: _sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "case_id": case_id,
        "source_index_sha256": _sha(index_path),
        "source_binding_sha256": _digest(envelope["source_binding"]),
        "projection": deepcopy(PROJECTION), "frame_selection": deepcopy(PROVIDER.FRAME_SELECTION),
        "quality_policy": deepcopy(QUALITY_POLICY), "authority": deepcopy(AUTHORITY),
        "claims": {"heldout": False, "recovery": False, "video_benefit": False,
                   "model_improvement": False, "capture_approved": False,
                   "used_for_decision": False},
    }


def prepare_protocol(index_path: Path, case_id: str, output: Path, *,
                     source_anchor: str | None = None) -> dict[str, Any]:
    value = protocol_value(index_path, case_id, source_anchor=source_anchor)
    _write_new(output, value)
    return value


def verify_protocol(path: Path, index_path: Path, case_id: str, execution_anchor: str,
                    *, repo: Path = ROOT) -> dict[str, Any]:
    value = _read(path)
    expected = protocol_value(index_path, case_id, source_anchor=cast(str, value.get("source_anchor")))
    _require(value == expected and value["status"] == "registered_before_pair_preparation",
             "registered pair-input protocol changed")
    anchor = value["source_anchor"]
    _require(isinstance(anchor, str) and len(anchor) == 40 and len(execution_anchor) == 40,
             "pair-input anchors")
    for older, newer in ((anchor, execution_anchor), (execution_anchor, "HEAD")):
        _require(subprocess.run(["git", "merge-base", "--is-ancestor", older, newer],
                                cwd=repo, capture_output=True, check=False).returncode == 0,
                 "pair-input anchor ancestry")
    for name, digest in value["sources_lf"].items():
        raw = subprocess.run(["git", "show", f"{anchor}:{name}"], cwd=repo,
                             capture_output=True, check=True).stdout
        normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _require(hashlib.sha256(normalized).hexdigest() == digest, "pair-input source blob")
    relative = Path(path).resolve().relative_to(repo.resolve()).as_posix()
    registered = subprocess.run(["git", "show", f"{execution_anchor}:{relative}"], cwd=repo,
                                capture_output=True, check=True).stdout
    _require(registered.replace(b"\r\n", b"\n") == Path(path).read_bytes().replace(b"\r\n", b"\n"),
             "pair-input protocol blob")
    return value


def _quality_receipt(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    policy_sha = _digest(QUALITY_POLICY)
    rows = []
    frame_refs = []
    for row in chosen:
        raw = QUALITY._image_bytes([row["label"], row["image"]])
        content_sha = hashlib.sha256(raw).hexdigest()
        try:
            value = QUALITY.laplacian_variance(raw)
            outcome = "available" if value >= QUALITY_FLOOR else "quality_below_registered_floor"
            metric: dict[str, Any] | None = {"name": QUALITY.METRIC, "value": value,
                                             "unit": QUALITY.UNIT}
        except ValueError:
            outcome, metric = "quarantined_decode_unavailable", None
        rows.append({"schema": QUALITY.SCHEMA + "_receipt", "policy_sha256": policy_sha,
                     "frame_content_sha256": content_sha, "metric": metric, "outcome": outcome})
        frame_refs.append({"frame_id": row["frame_id"], "sha256": content_sha})
    available = bool(rows) and all(row["outcome"] == "available" for row in rows)
    return {
        "schema": QUALITY.SCHEMA + "_evidence_receipt", "approval_granted": False,
        "visual_outcome": "available" if available else "unknown",
        "frame_refs": frame_refs if available else [], "image_count": 6 if available else 0,
        "quality_receipts": rows, "policy_sha256": policy_sha,
    }


def _capture(value: Mapping[str, Any], protocol: Path, index: Path, lab: Path,
             execution_anchor: str, verifier: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    return verifier(value, protocol, index, lab, execution_anchor=execution_anchor)


def _provenance(protocol_path: Path, index_path: Path, capture_protocols: Mapping[str, Path],
                artifacts: Mapping[str, Mapping[str, Any]], lab: Path,
                execution_anchor: str, *, fixture: bool) -> dict[str, Any]:
    base = {
        "schema": "lineage_same_play_pair_provenance_v1",
        "execution_anchor": execution_anchor, "lab_root": str(Path(lab).resolve()),
        "artifacts": {role: {"value": deepcopy(dict(value)),
                             "canonical_sha256": _digest(value)}
                      for role, value in artifacts.items()},
        "fixture": fixture,
    }
    if fixture:
        return base
    return {**base, "pair_protocol": _ref(protocol_path), "source_index": _ref(index_path),
            "capture_protocols": {role: _ref(path) for role, path in capture_protocols.items()}}


def _verify_provenance(result: Mapping[str, Any], index_path: Path, *, repo: Path = ROOT,
                       pair_protocol_verifier: Callable[..., dict[str, Any]] = verify_protocol,
                       capture_verifier: Callable[..., dict[str, Any]] = CAPTURE.verify_artifact) -> dict[str, Any]:
    provenance = result.get("provenance")
    _require(isinstance(provenance, Mapping)
             and provenance.get("schema") == "lineage_same_play_pair_provenance_v1"
             and provenance.get("fixture") is False, "production pair provenance")
    checked = cast(Mapping[str, Any], provenance)
    source_ref = cast(Mapping[str, Any], checked.get("source_index"))
    pair_ref = cast(Mapping[str, Any], checked.get("pair_protocol"))
    execution_anchor = checked.get("execution_anchor")
    _require(isinstance(execution_anchor, str) and len(execution_anchor) == 40
             and isinstance(source_ref, Mapping) and isinstance(pair_ref, Mapping),
             "pair provenance refs")
    source_path, pair_path = Path(cast(str, source_ref["path"])), Path(cast(str, pair_ref["path"]))
    _require(source_path.resolve() == Path(index_path).resolve()
             and _sha(source_path) == source_ref.get("sha256")
             and _sha(pair_path) == pair_ref.get("sha256"), "pair protocol/index source changed")
    pair_protocol_verifier(pair_path, source_path, result["case_id"], execution_anchor, repo=repo)
    lab = Path(cast(str, checked.get("lab_root"))).resolve()
    protocol_refs = checked.get("capture_protocols")
    artifacts = checked.get("artifacts")
    _require(isinstance(protocol_refs, Mapping) and isinstance(artifacts, Mapping)
             and set(protocol_refs) == {"case", "reference"}
             and set(artifacts) == {"case", "reference"}, "capture provenance roles")
    checked_protocol_refs = cast(Mapping[str, Any], protocol_refs)
    checked_artifacts = cast(Mapping[str, Any], artifacts)
    verified = {}
    for role in ("case", "reference"):
        protocol_ref = cast(Mapping[str, Any], checked_protocol_refs[role])
        protocol_path = Path(cast(str, protocol_ref["path"]))
        _require(_sha(protocol_path) == protocol_ref.get("sha256"), f"{role} capture protocol changed")
        record = cast(Mapping[str, Any], checked_artifacts[role])
        value = record.get("value")
        _require(isinstance(value, Mapping) and _digest(value) == record.get("canonical_sha256"),
                 f"{role} capture artifact changed")
        current = capture_verifier(value, protocol_path, source_path, lab,
                                   execution_anchor=execution_anchor)
        _require(current == value, f"{role} capture source proof changed")
        verified[role] = current
    return verified


def _assert_model_messages_safe(providers: Mapping[str, Mapping[str, Any]]) -> None:
    """모델에 직렬화되는 messages에 private 분류·경로가 없는지 확인한다."""
    forbidden_keys = {str(token).lower() for token in INPUT.PRIVATE_TOKENS}
    for provider in providers.values():
        messages = provider.get("messages")
        _require(isinstance(messages, list), "provider messages")
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        lowered = raw.lower()
        _require(not any(f'"{key}"' in lowered for key in forbidden_keys),
                 "private source key exposed to model")
        _require("bench/private" not in lowered and "\\users\\" not in lowered
                 and "/users/" not in lowered and "/home/" not in lowered,
                 "private or user path exposed to model")


def prepare_pair(index_path: Path, case_id: str, case_capture: Mapping[str, Any],
                 reference_capture: Mapping[str, Any], output: Path, *, protocol_path: Path,
                 execution_anchor: str, capture_protocols: Mapping[str, Path], lab: Path,
                 repo: Path = ROOT, allow_fixture: bool = False,
                 capture_verifier: Callable[..., dict[str, Any]] = CAPTURE.verify_artifact,
                 materializer: Callable[..., tuple[list[dict[str, Any]], dict[str, Any]]] = PROVIDER._materialize_frames,
                 quality_builder: Callable[[list[dict[str, Any]]], dict[str, Any]] = _quality_receipt) -> dict[str, Any]:
    protocol = (_read(protocol_path) if allow_fixture else
                verify_protocol(protocol_path, index_path, case_id, execution_anchor, repo=repo))
    _require(protocol.get("case_id") == case_id and set(capture_protocols) == {"case", "reference"},
             "pair/capture protocol binding")
    source = DRIVER.prepare_source(index_path, case_id)
    artifacts = {}
    for role, value in (("case", case_capture), ("reference", reference_capture)):
        artifacts[role] = (dict(value) if allow_fixture else _capture(
            value, capture_protocols[role], index_path, lab, execution_anchor, capture_verifier))
    provenance = _provenance(protocol_path, index_path, capture_protocols, artifacts, lab,
                             execution_anchor, fixture=allow_fixture)
    if any(item.get("status") != "available" for item in artifacts.values()):
        result = {
            "schema": RESULT_SCHEMA, "status": "hold_visual_unavailable",
            "case_id": case_id, "source_binding_sha256": _digest(source["source_binding"]),
            "providers": None, "visual_evidence": None,
            "numeric_training_fact_count": len(source["packet"]["facts"]),
            "replacement_capture": False, "approval_granted": False, "api_authorized": False,
            "claims": deepcopy(protocol["claims"]), "provenance": provenance,
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_new(output / "result.json", result)
        return result
    for role, artifact in artifacts.items():
        _require(artifact.get("frames") == 625 and artifact.get("fps") == 50
                 and artifact.get("capture_approved") is False
                 and artifact.get("model_ready") is False
                 and artifact.get("used_for_decision") is False,
                 f"{role} capture boundary")
    output.mkdir(parents=True, exist_ok=False)
    ledger = output / "history.jsonl"
    ledger.write_bytes(b"")
    snapshot = C.V.build_snapshot(source["packet"], f"lineage_pair_{case_id.replace('-', '_')}",
                                  "model_adaptive",
                                  C.V.file_digest(C.V.PROTOCOL), ledger)
    pending = {"source": {"captures": artifacts, "training": {
        role: {"checkpoint": {"sha256": artifact["checkpoint_sha256"]}}
        for role, artifact in artifacts.items()}}}
    chosen, evidence_refs = materializer(pending, output)
    play = {}
    for role in ("case", "reference"):
        play.update(PAIR_INPUT._play_facts(role, artifacts[role]["v01_specs"][0],
                                           artifacts[role]["proofs"]["raw_trace"]))
    _require(len(play) == 54, "same-Play numeric facts54")
    frame_facts = {f"vision:frame:{row['frame_id']}": {
        "evidence_type": "raw_frame", "timestamp_s": row["timestamp_s"],
        "source_group": "visual_source_1",
    } for row in chosen}
    context = A2.prepare(snapshot, goal=PROVIDER.PAIR_GOAL, evidence={}, revealed=[],
                         extra_facts={**play, **frame_facts}, decisions_left=3)
    messages = A2.messages(context)
    messages[1]["content"] = [{"type": "text", "text": messages[1]["content"]},
                              *[block for row in chosen for block in (row["label"], row["image"])]]
    prepared = {"schema": "fresh_frame_decision_prepared_v1",
                "status": "ready_for_cpu_provider_inputs_api_not_authorized",
                "context": context, "messages": messages,
                "approval_granted": False, "api_authorized": False}
    prepared_path = output / "prepared.json"
    _write_new(prepared_path, prepared)
    pair_dir = output / "pair"
    PROVIDER._prepare_pair(prepared_path, pair_dir)
    providers = {name: _read(pair_dir / f"{name}.provider.json") for name in DRIVER.MODEL_ARMS}
    _assert_model_messages_safe(providers)
    receipt = quality_builder(chosen)
    checked = DRIVER.prepare_source(index_path, case_id, providers=providers,
                                    visual_evidence={"quality_receipt": receipt})
    DRIVER.checked_source(checked, fixture=allow_fixture)
    result = {
        "schema": RESULT_SCHEMA, "status": "prepared" if receipt["visual_outcome"] == "available"
        else "prepared_frames_quarantined", "case_id": case_id,
        "source_binding_sha256": _digest(source["source_binding"]),
        "pair": _ref(pair_dir / "pair.json"), "providers": {name: _ref(
            pair_dir / f"{name}.provider.json") for name in DRIVER.MODEL_ARMS},
        "visual_evidence": receipt, "evidence_refs": evidence_refs,
        "numeric_training_fact_count": 48, "same_play_numeric_fact_count": 54,
        "image_count": 6 if receipt["visual_outcome"] == "available" else 0,
        "replacement_capture": False, "approval_granted": False, "api_authorized": False,
        "labels_exposed": False, "claims": deepcopy(protocol["claims"]),
        "provenance": provenance,
    }
    _write_new(output / "result.json", result)
    return result


def load_pair(result_path: Path, index_path: Path, *, allow_fixture: bool = False,
              repo: Path = ROOT,
              pair_protocol_verifier: Callable[..., dict[str, Any]] = verify_protocol,
              capture_verifier: Callable[..., dict[str, Any]] = CAPTURE.verify_artifact) -> dict[str, Any]:
    result = _read(result_path)
    _require(result.get("schema") == RESULT_SCHEMA
             and isinstance(result.get("case_id"), str) and bool(result["case_id"]), "pair result")
    if not allow_fixture:
        _verify_provenance(result, index_path, repo=repo,
                           pair_protocol_verifier=pair_protocol_verifier,
                           capture_verifier=capture_verifier)
    source = DRIVER.prepare_source(index_path, result["case_id"])
    if result["providers"] is None:
        _require(result["status"] == "hold_visual_unavailable", "visual HOLD result")
        return source
    providers = {name: _read(Path(ref["path"])) for name, ref in result["providers"].items()}
    for name, ref in result["providers"].items():
        _require(_sha(Path(ref["path"])) == ref["sha256"], f"provider SHA: {name}")
    checked = DRIVER.prepare_source(index_path, result["case_id"], providers=providers,
                                    visual_evidence={"quality_receipt": result["visual_evidence"]})
    return DRIVER.checked_source(checked, fixture=allow_fixture)
