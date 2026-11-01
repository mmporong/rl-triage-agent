# pyright: reportArgumentType=false, reportMissingImports=false
# pyright: reportOptionalMemberAccess=false, reportOptionalSubscript=false
"""검증된 source-v2 20건을 경로·credential 없는 공개 collection으로 내보낸다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("bench", "evals", "src")]

import lineage_source_inputs_v2 as INPUT

SCHEMA = "lineage_source_collection_public_v2"
SOURCE_FILES = tuple(sorted({
    *INPUT.BATCH.source_files(),
    "evals/lineage_source_inputs_v2.py",
    "tests/test_lineage_source_inputs_v2.py",
    "evals/export_lineage_source_collection_v2.py",
    "tests/test_export_lineage_source_collection_v2.py",
}))
SOURCE_PUBLIC_FILES = tuple(
    f"sources/source-{index:02d}/{name}"
    for index in range(1, 21)
    for name in ("fixed26.json", "training-curves.json", "provenance.json")
)
PUBLIC_FILES = ("summary.json", "provenance.json", *SOURCE_PUBLIC_FILES, "manifest.json")
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\[^\\/]+[\\/])")
POSIX_PRIVATE = re.compile(r"/(?:home|Users|mnt|root|private|tmp)/", re.IGNORECASE)
SECRET_PATTERNS = (
    re.compile(r"(?i)(?:api[_-]?key|authorization|bearer|client[_-]?secret)\s*[:=]"),
    re.compile(r"\b(?:nvapi-|sk-)[A-Za-z0-9_-]{8,}"),
)
PRIVATE_KEYS = {
    "path", "stage", "log_dir", "output_root", "checkpoint_file", "checkpoint_path",
    "source_job", "reference_job", "job_id", "run_name",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           ValueError(f"비유한 JSON 숫자: {token}")))
    require(isinstance(value, dict), f"{Path(path).name}은 JSON 객체여야 한다")
    return value


def sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, value: object) -> None:
    target = Path(path).resolve()
    require(not target.exists(), f"{target.name}은 fresh 파일이어야 한다")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(json_bytes(value))


def _privacy_scan(name: str, raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"공개 파일은 UTF-8 JSON이어야 한다: {name}") from error
    normalized = text.replace("\\\\", "\\")
    require(WINDOWS_ABSOLUTE.search(normalized) is None
            and POSIX_PRIVATE.search(normalized) is None,
            f"공개 파일에 절대/private 경로가 있다: {name}")
    require(not any(pattern.search(text) for pattern in SECRET_PATTERNS),
            f"공개 파일에 credential 표식이 있다: {name}")


def _sanitize(value: Any, *, alias: str) -> Any:
    """측정값을 보존하고 물리 경로·실행 식별자만 제거한다."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS or key.endswith(("_path", "_dir")):
                continue
            if key == "name":
                result[key] = alias
            else:
                result[key] = _sanitize(item, alias=alias)
        return result
    if isinstance(value, list):
        return [_sanitize(item, alias=alias) for item in value]
    require(value is None or isinstance(value, (str, int, float, bool)),
            "공개 projection은 JSON scalar/list/object여야 한다")
    if isinstance(value, float):
        require(math.isfinite(value), "공개 수치는 유한해야 한다")
    if isinstance(value, str):
        require(WINDOWS_ABSOLUTE.search(value) is None and POSIX_PRIVATE.search(value) is None,
                "공개 projection 문자열에 절대/private 경로가 있다")
    return value


def _owned_wall(value: object, name: str) -> float:
    require(isinstance(value, dict), f"{name} owned 실행이 없다")
    item = cast(dict[str, Any], value)
    wall = item.get("wall_s")
    require(isinstance(wall, (int, float)) and not isinstance(wall, bool)
            and math.isfinite(float(wall)) and wall >= 0, f"{name} wall_s가 없다")
    return float(wall)


def _series(telemetry: dict[str, Any], alias: str) -> dict[str, Any]:
    fields = INPUT.SELECT.V.TELEMETRY_FIELDS
    series, summary = telemetry.get("series"), telemetry.get("summary")
    require(isinstance(series, dict) and isinstance(summary, dict),
            f"{alias} telemetry series/summary가 없다")
    result_series: dict[str, list[int | float]] = {}
    result_summary: dict[str, Any] = {}
    for field in fields:
        values = series.get(field)
        require(isinstance(values, list) and len(values) == 300,
                f"{alias} {field}는 원래 300-point series여야 한다")
        values = cast(list[Any], values)
        require(all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(float(item)) for item in values),
                f"{alias} {field} series 수치가 유한하지 않다")
        row = summary.get(field)
        require(isinstance(row, dict) and row.get("n") == 300,
                f"{alias} {field} summary n이 300이 아니다")
        result_series[field] = deepcopy(values)
        result_summary[field] = deepcopy(row)
    return {"schema": SCHEMA + "_training_curves", "source_id": alias,
            "iterations": 300, "fields": list(fields),
            "series": result_series, "summary": result_summary}


def _committed_exporter() -> dict[str, str]:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    for relative in SOURCE_FILES:
        raw = subprocess.run(["git", "-C", str(ROOT), "show", f"{head}:{relative}"],
                             capture_output=True, check=True).stdout
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        require(hashlib.sha256(raw).hexdigest() == sha(ROOT / relative, lf=True),
                f"export source closure는 현재 HEAD에 커밋되어야 한다: {relative}")
    return {"analysis_head": head, "exporter_sha256_lf": sha(Path(__file__), lf=True)}


def _callbacks(allow_fixture: bool,
               validators: dict[str, Callable] | None) -> tuple[Callable, Callable]:
    if not allow_fixture:
        require(validators is None, "actual export validator는 교체할 수 없다")
        return INPUT.validate_batch, INPUT.checked
    require(isinstance(validators, dict), "fixture export는 명시 validator 두 개가 필요하다")
    validators = cast(dict[str, Callable], validators)
    require(set(validators) == {"validate_batch", "checked"}
            and all(callable(item) for item in validators.values()),
            "fixture export는 명시 validator 두 개가 필요하다")
    return validators["validate_batch"], validators["checked"]


def export(source_root: Path, protocol_path: Path, index_path: Path, *, lab: Path,
           anchor: str, out: Path, allow_fixture: bool = False,
           validators: dict[str, Callable] | None = None) -> dict[str, Any]:
    """actual20/fixed20/index를 재검증하고 공개 tag 폴더를 write-once 생성한다."""
    output = Path(out).resolve()
    require(not output.exists(), "공개 export tag 폴더는 fresh여야 한다")
    if not allow_fixture:
        try:
            relative_output = output.relative_to((ROOT / "evals/results").resolve())
        except ValueError as error:
            raise ValueError("actual 공개 export는 evals/results 새 tag 폴더여야 한다") from error
        require(bool(relative_output.parts) and ".." not in relative_output.parts,
                "actual 공개 export tag 경로가 안전하지 않다")
    validate_batch, checked = _callbacks(allow_fixture, validators)
    context = validate_batch(source_root, protocol_path, lab=lab, anchor=anchor)
    public, metadata = checked(index_path)
    require(public.get("schema") == INPUT.SCHEMA and public.get("source_count") == 20
            and public.get("planned_cases") == 10
            and metadata.get("schema") == INPUT.METADATA_SCHEMA
            and len(context.get("sources", {})) == 20 and len(context.get("cases", [])) == 10,
            "source-v2 actual 20/10 index가 아니다")
    source_result = context.get("result")
    require(isinstance(source_result, dict)
            and source_result.get("source_jobs_completed") == 20
            and source_result.get("fixed26_completed") == 20
            and source_result.get("model_api") == 0 and source_result.get("repairs") == 0
            and source_result.get("runtime_facts_fabricated") is False,
            "source-v2 SDK 완료/API/repair/runtime 계약이 다르다")
    if allow_fixture:
        require(metadata.get("fixture") in {False, True}, "fixture metadata 표식")
        committed = {"analysis_head": None,
                     "exporter_sha256_lf": sha(Path(__file__), lf=True)}
    else:
        require(public.get("fixture") is False and metadata.get("fixture") is False,
                "actual export에 fixture index를 쓸 수 없다")
        committed = _committed_exporter()
    require(metadata.get("anchor") == anchor
            and metadata.get("protocol") == INPUT._ref(Path(protocol_path))
            and metadata.get("source_result") == INPUT._ref(Path(source_root) / "result.json")
            and metadata.get("cases") == context["cases"]
            and metadata.get("duplicate_checkpoint_groups")
            == context["duplicate_checkpoint_groups"],
            "batch/index/protocol/current data 결합이 다르다")
    private_sources = metadata.get("sources")
    require(isinstance(private_sources, dict) and set(private_sources) == set(context["sources"]),
            "batch/index source 집합이 다르다")
    private_sources = cast(dict[str, dict[str, Any]], private_sources)
    ordered_jobs = [row["job_id"] for row in context["protocol"]["training_jobs"]]
    require(len(ordered_jobs) == 20 and len(set(ordered_jobs)) == 20,
            "source-v2 training job 순서가 20개가 아니다")

    output.mkdir(parents=True, exist_ok=False)
    source_inventory = []
    alias_by_job: dict[str, str] = {}
    training_wall = fixed_wall = 0.0
    for index, job_id in enumerate(ordered_jobs, 1):
        alias = f"source-{index:02d}"
        alias_by_job[job_id] = alias
        source, private = context["sources"][job_id], private_sources[job_id]
        binding = private["binding"]
        require(public["sources"].get(private["public_id"]) == binding
                and source["checkpoint"]["sha256"] == binding["checkpoint_sha256"],
                f"{alias} public/private/current checkpoint 결합")
        material = private["material"]
        telemetry_path = Path(material["telemetry"]["path"])
        telemetry = read(telemetry_path)
        curves = _series(telemetry, alias)
        fixed_raw = read(source["stage"] / "fixed-result.json")
        fixed_public = {
            "schema": SCHEMA + "_fixed26", "source_id": alias, "conditions": 26,
            "report": _sanitize(fixed_raw["report"], alias=alias),
            "execution": _sanitize(fixed_raw["execution"], alias=alias),
        }
        training_owned = read(source["stage"] / "owned-result.json")
        training_wall += _owned_wall(training_owned, alias + " training")
        fixed_wall += _owned_wall(fixed_raw["execution"], alias + " fixed26")
        provenance = {
            "schema": SCHEMA + "_source_provenance", "source_id": alias,
            "sha256_only": {
                "checkpoint": binding["checkpoint_sha256"],
                "event": material["event"]["sha256"],
                "event_copy": material["event_copy"]["sha256"],
                "telemetry": binding["telemetry_sha256"],
                "extract_receipt": material["receipt"]["sha256"],
                "manifest": binding["manifest_sha256"],
                "training_evidence": binding["evidence_sha256"],
                "training_records": binding["records_sha256"],
                "fixed_result": binding["fixed_result_sha256"],
                "owned_result": sha(source["stage"] / "owned-result.json"),
            },
            "checkpoint_copied": False, "event_copied": False,
            "raw_provenance_copied": False,
        }
        base = output / "sources" / alias
        write_new(base / "fixed26.json", fixed_public)
        write_new(base / "training-curves.json", curves)
        write_new(base / "provenance.json", provenance)
        source_inventory.append({
            "source_id": alias, "fixed26": f"sources/{alias}/fixed26.json",
            "training_curves": f"sources/{alias}/training-curves.json",
            "provenance": f"sources/{alias}/provenance.json",
        })

    cases = context["cases"]
    labels: dict[str, int] = {}
    references: dict[str, int] = {}
    for row in cases:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        relation = row["reference_relation"]
        references[relation] = references.get(relation, 0) + 1
    valid = sum(bool(row["eligible_after_dedup"]) for row in cases)
    arms = [row["arm"] for row in context["protocol"]["cases"]]
    case_plan = [{
        "case_id": row["case_id"], "source_id": alias_by_job[row["source_job"]],
        "reference_id": alias_by_job[row["reference_job"]], "source_type": row["arm"],
        "classification": row["label"], "valid_after_dedup": row["eligible_after_dedup"],
        "exclusion_reason": row.get("exclusion_reason"),
        "reference_relation": row["reference_relation"],
    } for row in cases]
    batch_wall = source_result.get("wall_s")
    require(isinstance(batch_wall, (int, float)) and not isinstance(batch_wall, bool)
            and math.isfinite(float(batch_wall)) and batch_wall >= 0,
            "source batch parent wall_s가 없다")
    summary = {
        "schema": SCHEMA + "_summary", "status": "complete",
        "scope": "development_fixture" if allow_fixture else "actual_source_collection",
        "actual": not allow_fixture,
        "counts": {"planned_cases": 10, "valid_after_dedup": valid,
                   "excluded": 10 - valid, "anonymous_sources": 20,
                   "training_sdk_completed": 20, "fixed26_sdk_completed": 20,
                   "actual_model_api_calls": source_result["model_api"],
                   "repairs": source_result["repairs"]},
        "case_plan": case_plan,
        "classification_labels": dict(sorted(labels.items())),
        "known_source_types": sorted(set(arms)),
        "known_fault_types": sorted(set(arms) - {"NONE"}),
        "seed_semantics": "새 source seed는 독립 반복이며 새 fault type을 뜻하지 않는다.",
        "reference_relationships": dict(sorted(references.items())),
        "runtime_observation": {"facts_per_case": 16, "observed": 0,
                                "available_flags": 0, "fabricated": False,
                                "status": "unavailable_requires_separate_gpu_run"},
        "wall_clock_s": {"batch_parent": float(batch_wall),
                         "training_parent_sum": training_wall,
                         "fixed26_parent_sum": fixed_wall,
                         "scope": "parent_launch_to_owned_completion_not_sdk_completion_count"},
        "sdk_completion_scope": "20 training과 20 fixed26 완료 receipt 수; wall-clock과 별도",
        "sources": source_inventory,
    }
    provenance = {
        "schema": SCHEMA + "_provenance", **committed,
        "collection_anchor": anchor, "protocol_sha256": sha(Path(protocol_path)),
        "source_result_sha256": sha(Path(source_root) / "result.json"),
        "source_index_sha256": sha(Path(index_path)),
        "exporter_sources_lf": {name: sha(ROOT / name, lf=True) for name in SOURCE_FILES},
        "checkpoint_files_copied": 0, "event_files_copied": 0,
        "raw_private_files_copied": 0, "credentials_copied": 0,
    }
    write_new(output / "summary.json", summary)
    write_new(output / "provenance.json", provenance)
    inventory = {name: {"sha256": sha(output / name),
                        "size_bytes": (output / name).stat().st_size}
                 for name in PUBLIC_FILES if name != "manifest.json"}
    manifest = {"schema": SCHEMA + "_manifest", "status": "complete",
                "scope": summary["scope"], "public_files": list(PUBLIC_FILES),
                "files": inventory}
    write_new(output / "manifest.json", manifest)
    verify_export(output, index_path=index_path, allow_fixture=allow_fixture,
                  index_checker=checked)
    return deepcopy(summary)


def verify_export(out: Path, *, index_path: Path | None = None,
                  allow_fixture: bool = False,
                  index_checker: Callable = INPUT.checked) -> dict[str, Any]:
    root = Path(out).resolve(strict=True)
    manifest = read(root / "manifest.json")
    expected_scope = "development_fixture" if allow_fixture else "actual_source_collection"
    require(manifest.get("schema") == SCHEMA + "_manifest"
            and manifest.get("status") == "complete" and manifest.get("scope") == expected_scope
            and manifest.get("public_files") == list(PUBLIC_FILES)
            and set(manifest.get("files", {})) == set(PUBLIC_FILES) - {"manifest.json"},
            "public manifest/file inventory가 다르다")
    actual_files = sorted(path.relative_to(root).as_posix()
                          for path in root.rglob("*") if path.is_file())
    require(actual_files == sorted(PUBLIC_FILES), "public tag에 미등록 파일이 있다")
    for name, item in manifest["files"].items():
        path = root / name
        require(sha(path) == item.get("sha256") and path.stat().st_size == item.get("size_bytes"),
                f"public file SHA/size가 바뀌었다: {name}")
    for name in PUBLIC_FILES:
        _privacy_scan(name, (root / name).read_bytes())
    if index_path is not None:
        public, metadata = index_checker(index_path)
        del public, metadata
        provenance = read(root / "provenance.json")
        require(provenance.get("source_index_sha256") == sha(Path(index_path)),
                "현재 source-v2 index SHA가 export 시점과 다르다")
    return deepcopy(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    export(args.source_root, args.protocol, args.index, lab=args.lab,
           anchor=args.anchor, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
