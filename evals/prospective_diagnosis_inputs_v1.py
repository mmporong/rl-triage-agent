"""새 seed 학습을 결함 정답을 노출하지 않는 진단 입력으로 봉인한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "bench"), str(ROOT / "evals"), str(ROOT / "src")]

import optimizer_runtime_facts_v1 as FACT
import prospective_diagnosis_training_v1 as TRAIN
import runtime_training_inputs_v1 as BASE

SCHEMA = "prospective_diagnosis_collected_v1"
METADATA_SCHEMA = "prospective_diagnosis_collected_metadata_v1"
PACKET_SCHEMA = "prospective_diagnosis_packet_v1"
COLLECTED = "collected.json"
METADATA = "metadata.json"
_PRIVATE_TOKENS = {"F1", "R1", "R1_ONLY", "COMPOUND", "original_faults",
                   "applied_corrections", "remaining_faults"}


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda token: (
        _ for _ in ()).throw(ValueError(f"비유한 JSON 숫자: {token}")))
    _require(isinstance(value, dict), f"{Path(path).name}은 객체여야 한다")
    return value


def _write(path: Path, value: object) -> None:
    _require(not path.exists(), f"{path.name}은 fresh 파일이어야 한다")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _ref(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    _require(path.is_file(), f"근거 파일이 없다: {path.name}")
    return {"path": str(path), "sha256": _sha(path)}


def _category(metadata: dict[str, Any]) -> str:
    private = metadata.get("private_training")
    _require(isinstance(private, dict), "base private training metadata가 없다")
    private = cast(dict[str, Any], private)
    matches = [name for name, contract in TRAIN.CATEGORIES.items()
               if private.get("arm") == contract["arm"]
               and private.get("original_faults") == contract["original_faults"]
               and private.get("applied_corrections") == []]
    _require(len(matches) == 1, "학습 metadata가 등록 prospective 범주와 일치하지 않는다")
    return matches[0]


def collect(training_result_path: Path, *, category: str, out: Path) -> dict[str, Any]:
    """기존 수집·runtime fact를 재검증하고 새 목적 envelope를 write-once로 만든다."""
    training_result_path, out = Path(training_result_path).resolve(), Path(out).resolve()
    _require(not out.exists(), "prospective input out은 fresh 경로여야 한다")
    training = _read(training_result_path)
    for key in ("collected", "runtime_facts", "checkpoint"):
        _require(isinstance(training.get(key), dict), f"training result에 {key}가 없다")
    collected_ref = cast(dict[str, Any], training["collected"])
    _require(_sha(Path(collected_ref["path"])) == collected_ref.get("sha256"),
             "base collected SHA가 다르다")
    base, base_metadata = BASE.checked(Path(collected_ref["path"]))
    actual = _category(base_metadata)
    _require(category == actual, "호출 category와 실제 manifest 범주가 다르다")
    checkpoint = cast(dict[str, Any], training["checkpoint"])
    _require(base["source_binding"]["checkpoint_sha256"] == checkpoint["sha256"],
             "base collected와 checkpoint가 다르다")
    runtime_refs = cast(dict[str, dict[str, str]], training["runtime_facts"])
    for role in ("case", "reference"):
        item = runtime_refs.get(role)
        _require(isinstance(item, dict) and _sha(Path(item["path"])) == item.get("sha256"),
                 f"{role} runtime fact ref가 다르다")
        item = cast(dict[str, str], item)
        FACT.checked(Path(item["path"]), expected_checkpoint_sha256=checkpoint["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": METADATA_SCHEMA, "category": category,
        "training_result": _ref(training_result_path), "base_collected": deepcopy(collected_ref),
        "base_metadata": _ref(Path(collected_ref["path"]).parent / BASE.METADATA),
        "runtime_facts": deepcopy(runtime_refs), "checkpoint": deepcopy(checkpoint),
    }
    _write(out / METADATA, metadata)
    public = {
        "schema": SCHEMA,
        "source_binding": {
            "checkpoint_sha256": checkpoint["sha256"],
            "telemetry_sha256": base["source_binding"]["telemetry_sha256"],
            "base_binding_digest": base["source_binding"]["binding_digest"],
            "metadata_sha256": _sha(out / METADATA),
        },
        "scope": {"purpose": "prospective_diagnosis_input", "approval": False,
                  "execution_allowed": False, "fault_label_exposed": False},
    }
    _write(out / COLLECTED, public)
    return deepcopy(public)


def checked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    value = _read(path)
    _require(value.get("schema") == SCHEMA and set(value) == {"schema", "source_binding", "scope"},
             "prospective collected shape가 다르다")
    metadata_path = path.parent / METADATA
    metadata = _read(metadata_path)
    binding = value.get("source_binding")
    _require(isinstance(binding, dict) and binding.get("metadata_sha256") == _sha(metadata_path),
             "private metadata ref가 다르다")
    binding = cast(dict[str, Any], binding)
    training_ref = metadata.get("training_result")
    _require(isinstance(training_ref, dict)
             and _sha(Path(training_ref["path"])) == training_ref.get("sha256"),
             "training result 원문이 바뀌었다")
    training_ref = cast(dict[str, Any], training_ref)
    training = _read(Path(training_ref["path"]))
    base_ref = metadata.get("base_collected")
    _require(isinstance(base_ref, dict) and training.get("collected") == base_ref
             and _sha(Path(base_ref["path"])) == base_ref.get("sha256"),
             "base collected 결합이 다르다")
    base_ref = cast(dict[str, Any], base_ref)
    base, base_metadata = BASE.checked(Path(base_ref["path"]))
    category = _category(base_metadata)
    _require(metadata.get("schema") == METADATA_SCHEMA and metadata.get("category") == category,
             "private category가 실제 manifest와 다르다")
    checkpoint = training.get("checkpoint")
    _require(isinstance(checkpoint, dict) and metadata.get("checkpoint") == checkpoint
             and base["source_binding"]["checkpoint_sha256"] == checkpoint.get("sha256"),
             "checkpoint lineage가 다르다")
    checkpoint = cast(dict[str, Any], checkpoint)
    runtime_refs = metadata.get("runtime_facts")
    _require(isinstance(runtime_refs, dict) and runtime_refs == training.get("runtime_facts"),
             "runtime fact refs가 다르다")
    runtime_refs = cast(dict[str, dict[str, str]], runtime_refs)
    for role in ("case", "reference"):
        item = runtime_refs.get(role)
        _require(isinstance(item, dict) and _sha(Path(item["path"])) == item.get("sha256"),
                 f"{role} runtime fact bytes가 바뀌었다")
        item = cast(dict[str, str], item)
        FACT.checked(Path(item["path"]), expected_checkpoint_sha256=checkpoint["sha256"])
    _require(binding == {
        "checkpoint_sha256": checkpoint["sha256"],
        "telemetry_sha256": base["source_binding"]["telemetry_sha256"],
        "base_binding_digest": base["source_binding"]["binding_digest"],
        "metadata_sha256": _sha(metadata_path)}, "public source binding이 다르다")
    _require(value.get("scope") == {"purpose": "prospective_diagnosis_input", "approval": False,
                                     "execution_allowed": False, "fault_label_exposed": False},
             "prospective input이 권한을 만들었다")
    return deepcopy(value), deepcopy(metadata)


def _model_safe(value: object) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _require(not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", encoded)
                     for token in _PRIVATE_TOKENS), "model packet에 fault 정답 필드가 노출됐다")
    _require(not re.search(r"(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users)/)", encoded),
             "model packet에 물리 경로가 노출됐다")


def make_packet(
    case_path: Path, reference_path: Path, *, budget_left: int = 3,
    remaining_candidates: list[str] | None = None,
) -> dict[str, Any]:
    """base telemetry/runtime projection을 쓰되 모델 view에는 private label/path를 넣지 않는다."""
    case, case_meta = checked(case_path)
    reference, reference_meta = checked(reference_path)
    case_base = cast(dict[str, Any], case_meta["base_collected"])
    reference_base = cast(dict[str, Any], reference_meta["base_collected"])
    case_runtime = cast(dict[str, dict[str, str]], case_meta["runtime_facts"])["case"]
    reference_runtime = cast(dict[str, dict[str, str]], reference_meta["runtime_facts"])["reference"]
    base = BASE.make_packet(
        Path(case_base["path"]), Path(reference_base["path"]),
        case_runtime_path=Path(case_runtime["path"]),
        reference_runtime_path=Path(reference_runtime["path"]), budget_left=budget_left,
        remaining_candidates=remaining_candidates, applied_corrections=[])
    packet = deepcopy(base["packet"])
    _model_safe(packet)
    return {
        "schema": PACKET_SCHEMA, "packet": packet,
        "source_binding": {"case": deepcopy(case["source_binding"]),
                           "reference": deepcopy(reference["source_binding"])},
        "private_metadata": {
            "case_input": _ref(case_path), "reference_input": _ref(reference_path),
            "base_packet_metadata": deepcopy(base["private_metadata"]),
        },
        "scope": {"approval": False, "execution_allowed": False,
                  "fault_label_exposed_to_model": False,
                  "physical_path_exposed_to_model": False},
    }


__all__ = ["COLLECTED", "METADATA", "PACKET_SCHEMA", "SCHEMA", "checked", "collect",
           "make_packet"]
