"""V01/V03 원문을 재검증하는 읽기 전용 CPU 근거 색인. 반환 색인은 호스트 전용이다."""
from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src"), str(ROOT / "bench")]
import vision_transferred_input_acceptance_v2 as TRANSFER

V01, SELECT = TRANSFER.V01, TRANSFER.SELECT
SPEC_SCHEMA = "agentic_evidence_index_spec_v1"
SCHEMA = "agentic_evidence_index_v1"
REF_FIELDS = {"root", "path", "sha256"}
ENTRY_FIELDS = {"evidence_id", "split", "already_seen", "binding", "contract", "snapshot",
                "training", "same_play_trace", "window"}
Roots = Mapping[str, Path]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _exact(value: Any, fields: set[str], anchor: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == fields, f"{anchor}: fields")
    return value


def _sha(value: Any) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "invalid SHA256")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    def invalid(_value: str) -> Any:
        raise ValueError("nonfinite JSON")

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                       parse_constant=invalid)
    _require(isinstance(value, dict), "JSON object required")
    return value


def index_sha256(value: dict[str, Any]) -> str:
    """호출자가 등록·보관할 canonical index SHA. 원문 artifact SHA와 구분한다."""
    return hashlib.sha256(V01.canonical(value)).hexdigest()


def _inside(root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative) and "\\" not in relative
             and ":" not in relative and "\x00" not in relative, "unsafe relative path")
    parts = relative.split("/")
    _require(not PurePosixPath(relative).is_absolute()
             and all(part not in {"", ".", ".."} and part.rstrip(" .") == part for part in parts),
             "unsafe relative path")
    base = Path(root).resolve(strict=True)
    _require(base.is_dir(), "root is not directory")
    target = base
    for part in parts:
        target = target / part
        info = target.lstat()
        _require(not target.is_symlink() and not getattr(target, "is_junction", lambda: False)()
                 and not (getattr(info, "st_file_attributes", 0) & 0x400), "linked path")
    target.resolve(strict=True).relative_to(base)
    info = target.stat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "not a single regular file")
    return target


def _ref(value: Any, roots: Roots) -> Path:
    ref = _exact(value, REF_FIELDS, "artifact ref")
    _sha(ref["sha256"])
    _require(isinstance(ref["root"], str) and ref["root"] in roots, "unknown root")
    target = _inside(roots[ref["root"]], ref["path"])
    _require(V01.sha256(target) == ref["sha256"], "raw SHA256 mismatch")
    return target


def _contract(ref: dict[str, Any], roots: Roots) -> tuple[Path, dict[str, Any]]:
    path = _ref(ref, roots)
    body = _read(path)
    provenance = body["provenance"]
    artifacts = [provenance["video"], provenance["checkpoint"], *body["frames"]]
    if provenance["measurement"] is not None:
        artifacts.append(provenance["measurement"])
    for artifact in artifacts:
        local = _inside(path.parent, artifact["path"])
        _require(V01.sha256(local) == _sha(artifact["sha256"]), "contract artifact SHA mismatch")
        if local.suffix == ".json":
            _read(local)
    # 프레임 추출·같은 Play 시각·측정·계보 검증은 기존 계약이 소유한다.
    return path, V01.verify(path)


def _snapshot(ref: dict[str, Any], roots: Roots) -> tuple[dict[str, Any], dict[str, Any] | None]:
    snapshot = _read(_ref(ref, roots))
    ledger = Path(snapshot["inputs"]["ledger_path"])
    _require(ledger.is_absolute() and ".." not in ledger.parts, "unsafe snapshot ledger path")
    for name, root in roots.items():
        base = Path(root).resolve(strict=True)
        if not ledger.is_relative_to(base):
            continue
        relative = ledger.relative_to(base).as_posix()
        target = base
        # 초기 상태에는 원장 파일·부모 폴더가 아직 없을 수 있다. 생성하지 않고 빈 상태로 둔다.
        for part in ledger.relative_to(base).parts:
            target = target / part
            if not target.exists() and not target.is_symlink():
                return snapshot, None
            info = target.lstat()
            _require(not target.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "linked snapshot ledger path")
        target = _inside(base, relative)
        return snapshot, {"root": name, "path": relative, "sha256": V01.sha256(target)}
    raise ValueError("snapshot ledger outside registered roots")


def _training(entry: dict[str, Any], snapshot: dict[str, Any], roots: Roots) -> list[dict]:
    training = _exact(entry["training"], {"case", "reference"}, "training")
    packet = snapshot["inputs"]["packet"]
    rows = []
    for role in ("case", "reference"):
        pair = _exact(training[role], {"artifact", "telemetry", "checkpoint"}, "training role")
        artifact = _read(_ref(pair["artifact"], roots))
        telemetry_path = _ref(pair["telemetry"], roots)
        telemetry = _read(telemetry_path)
        binding = packet["input_bindings"][role]
        _ref(pair["checkpoint"], roots)
        _require(artifact["checkpoint"]["sha256"] == binding["checkpoint_sha256"]
                 == pair["checkpoint"]["sha256"],
                 "training checkpoint mismatch")
        _require(artifact["telemetry"]["sha256"] == binding["telemetry_sha256"]
                 == pair["telemetry"]["sha256"]
                 and artifact["telemetry"]["summary"] == telemetry["summary"],
                 "training telemetry binding mismatch")
        for metric in SELECT.CONTROLLER.V.TELEMETRY_FIELDS:
            fact_id = f"telemetry:{role}:{metric}"
            _require(packet["facts"].get(fact_id) == telemetry["summary"][metric]["mean_last_20pct"],
                     "training fact mismatch")
        rows.append({"role": role, "artifact": deepcopy(pair["artifact"]),
                     "telemetry": deepcopy(pair["telemetry"]),
                     "checkpoint_sha256": binding["checkpoint_sha256"]})
    return rows


def _entry(entry: Any, roots: Roots) -> dict[str, Any]:
    entry = _exact(entry, ENTRY_FIELDS, "entry")
    _require(isinstance(entry["evidence_id"], str)
             and re.fullmatch(r"evidence-[0-9]{2,6}", entry["evidence_id"]) is not None,
             "opaque evidence ID required")
    _require(type(entry["already_seen"]) is bool and entry["split"] in {"development", "evaluation"}
             and not (entry["already_seen"] and entry["split"] != "development"), "invalid split")
    binding = _exact(entry["binding"], {"source_lineage_id", "checkpoint_sha256"}, "binding")
    path, contract = _contract(entry["contract"], roots)
    provenance = contract["provenance"]
    _require(binding["source_lineage_id"] == provenance["source_lineage_id"]
             and _sha(binding["checkpoint_sha256"]) == provenance["checkpoint"]["sha256"],
             "source/checkpoint mismatch")
    training: list[dict] = []
    ledger = None
    views = ["video_only"]
    if entry["snapshot"] is not None:
        snapshot, ledger = _snapshot(entry["snapshot"], roots)
        # V03가 case checkpoint, 상태 원장, 공개 사실 allowlist를 검사한다.
        bundle = SELECT.prepare(snapshot, path, {"contract_sha256": entry["contract"]["sha256"],
                                "source_lineage_id": binding["source_lineage_id"]}, "text_and_video")
        _require(not any(key.startswith("telemetry:") for key in bundle["selection_view"]["evidence_facts"])
                 or entry["training"] is not None, "training provenance required")
        if entry["training"] is not None:
            training = _training(entry, snapshot, roots)
        views = sorted(SELECT.VIEW_KINDS)
    else:
        _require(entry["training"] is None, "training requires snapshot")
    if entry["same_play_trace"] is not None:
        trace = _read(_ref(entry["same_play_trace"], roots))
        _require(trace["checkpoint_sha256"] == binding["checkpoint_sha256"]
                 and trace["video_sha256"] == provenance["video"]["sha256"], "trace source mismatch")
        TRANSFER.BASE.SOURCES._verify_window(contract, trace, entry["window"])
    else:
        _require(entry["window"] is None, "window requires raw trace")
    return {"evidence_id": entry["evidence_id"], "binding": deepcopy(binding),
            "parent_source_lineage_id": provenance["parent_source_lineage_id"],
            "video_sha256": provenance["video"]["sha256"],
            "frame_ids": [row["frame_id"] for row in contract["frames"]],
            "timestamps_s": [row["timestamp_s"] for row in contract["frames"]],
            "same_play_measurement": "available" if provenance["measurement"] else "unavailable",
            "same_play_raw_trace": "available" if entry["same_play_trace"] else "unavailable",
            "training": training, "snapshot_ledger": ledger, "available_views": views,
            "behavior_ground_truth": "unavailable", "joint_contact_ground_truth": "unavailable",
            "fixed_evaluation_is_same_play_ground_truth": False,
            "model_interpretation_is_independent_measurement": False}


def build_index(spec: dict[str, Any], roots: Roots) -> dict[str, Any]:
    """원문 참조를 검증해 호스트 색인을 반환한다. 파일·원장·모델 호출 부수효과는 없다."""
    _exact(spec, {"schema", "entries"}, "spec")
    _require(spec["schema"] == SPEC_SCHEMA and isinstance(spec["entries"], list)
             and bool(spec["entries"]), "invalid index spec")
    entries = [_entry(entry, roots) for entry in spec["entries"]]
    _require(len({row["evidence_id"] for row in entries}) == len(entries), "duplicate evidence ID")
    lineages: dict[str, tuple[str, str]] = {}
    for entry in entries:
        identity = (entry["binding"]["checkpoint_sha256"], entry["video_sha256"])
        for lineage in (entry["binding"]["source_lineage_id"], entry["parent_source_lineage_id"]):
            if lineage is not None:
                _require(lineage not in lineages or lineages[lineage] == identity, "mixed source lineage")
                lineages[lineage] = identity
    return {"schema": SCHEMA, "scope": "host_only_cpu_evidence_index",
            "spec": deepcopy(spec), "entries": entries,
            "model_api_requests": 0, "gpu_executions": 0, "independent_confirmation_sources_added": 0}


def verify_index(index: dict[str, Any], roots: Roots, *, expected_sha256: str) -> dict[str, Any]:
    """호출자가 고정한 index SHA와 현재 원문·계보를 함께 재검증한다."""
    _require(index_sha256(index) == _sha(expected_sha256), "registered index SHA mismatch")
    expected = build_index(index["spec"], roots)
    _require(V01.canonical(expected) == V01.canonical(index), "index/current evidence mismatch")
    return expected


def _selected(index: dict[str, Any], evidence_id: str, roots: Roots,
              expected_sha256: str) -> tuple[dict[str, Any], Path]:
    checked = verify_index(index, roots, expected_sha256=expected_sha256)
    entries = [row for row in checked["spec"]["entries"] if row["evidence_id"] == evidence_id]
    _require(len(entries) == 1, "unknown evidence ID")
    return entries[0], _ref(entries[0]["contract"], roots)


def _safe(messages: list[dict], entry: dict, contract: dict, roots: Roots) -> None:
    # 원문 라벨·파일명·계보·경로는 기존 검증기에서 차단한다. 이미지 바이트는 검사 문구와 분리한다.
    hidden = [entry["binding"]["source_lineage_id"], entry["binding"]["checkpoint_sha256"],
              contract["provenance"]["play_run_id"], *[str(path) for path in roots.values()]]
    TRANSFER.check_messages(messages, hidden)
    texts = TRANSFER.MATERIAL._model_text(messages)
    _require(re.search(r"(?i)(?:[a-z]:[\\/]|/home/|/users/|bench[/\\]private|"
                       r"\b(?:healthy|unhealthy|recovered|F[1-9][0-9]*|NONE)\b|"
                       r"restore_[a-z_]+|[^\s\"]+\.(?:json|mp4|pt|npz)\b)", texts) is None,
             "model view answer/path leak")


def model_view(index: dict[str, Any], evidence_id: str, roots: Roots, *, expected_sha256: str,
               view_kind: str = "text_and_video") -> dict[str, Any]:
    """V03 메시지 또는 snapshot 없는 V01 영상 메시지만 반환한다. 호스트 메타데이터는 제외한다."""
    entry, path = _selected(index, evidence_id, roots, expected_sha256)
    contract = V01.verify(path)
    if entry["snapshot"] is None:
        _require(view_kind == "video_only", "snapshot unavailable for requested view")
        messages = V01.public_model_view(path)["messages"]
    else:
        snapshot = _read(_ref(entry["snapshot"], roots))
        binding = {"contract_sha256": entry["contract"]["sha256"],
                   "source_lineage_id": entry["binding"]["source_lineage_id"]}
        bundle = SELECT.prepare(snapshot, path, binding, view_kind)
        messages = SELECT.messages(bundle, snapshot, path, binding)
    _safe(messages, entry, contract, roots)
    return {"schema": "agentic_evidence_model_view_v1", "messages": messages,
            "independent_confirmation_sources_added": 0}


def frame_view(index: dict[str, Any], evidence_id: str, frame_id: str, roots: Roots,
               *, expected_sha256: str) -> dict[str, Any]:
    """기존 V01 payload의 등록 프레임·시각·이미지 한 쌍을 조회한다."""
    entry, path = _selected(index, evidence_id, roots, expected_sha256)
    contract = V01.verify(path)
    ids = [row["frame_id"] for row in contract["frames"]]
    _require(frame_id in ids, "unknown frame ID")
    public = V01.public_model_view(path)
    content = public["messages"][1]["content"]
    offset = 2 + 2 * ids.index(frame_id)
    messages = [{"role": "user", "content": deepcopy(content[offset:offset + 2])}]
    _safe(messages, entry, contract, roots)
    return {"schema": "agentic_evidence_frame_view_v1", "messages": messages,
            "independent_confirmation_sources_added": 0}


def inventory_media(manifest_ref: dict[str, Any], roots: Roots) -> dict[str, Any]:
    """기존 Go2 시연 clip의 SHA를 검증한다. 같은 Play 상태 미등록 자료는 model view 대상에서 보류한다."""
    path = _ref(manifest_ref, roots)
    manifest = _read(path)
    _require(manifest.get("schema") == "training_correction_media_artifact_v2",
             "unsupported media manifest")
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("missing media clips")
    related = []
    for source in manifest.get("provenance", {}).get("public_sources", []):
        source_ref = {"root": manifest_ref["root"], "path": source["path"], "sha256": source["sha256"]}
        _ref(source_ref, roots)
        related.append(source_ref)
    rows = []
    for clip in clips:
        video = _inside(path.parent, clip["file"])
        _require(V01.sha256(video) == _sha(clip["sha256"])
                 and video.stat().st_size == clip["bytes"], "media raw SHA/size mismatch")
        rows.append({"clip_id": clip["id"], "role": clip["role"],
                     "video_sha256": clip["sha256"], "bytes": clip["bytes"],
                     "checkpoint_sha256": _sha(clip["checkpoint_sha256"]),
                     "checkpoint_verification": "manifest_hash_only_raw_checkpoint_not_supplied",
                     "same_play_state": "unavailable", "behavior_ground_truth": "unavailable",
                     "model_view_available": False})
    _require(len({row["clip_id"] for row in rows}) == len(rows), "duplicate media clip")
    return {"schema": "agentic_evidence_media_inventory_v1", "manifest": deepcopy(manifest_ref),
            "split": "development", "already_seen": True, "clips": rows,
            "related_public_artifacts": related,
            "missing": ["V01 contract", "same-Play state/command/pose", "same-Play joint/contact truth"],
            "fixed_evaluation_is_same_play_ground_truth": False,
            "model_interpretation_is_independent_measurement": False}
