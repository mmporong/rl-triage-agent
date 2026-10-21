"""외부 telemetry를 검증 가능한 단일 실행 작업공간으로 준비하고 검사한다."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rl_triage import grounded_evidence as G
from rl_triage import single_run as S
from rl_triage import triage_tools as T


IMPORT_SCHEMA = "rl_triage_run_import_v1"
REFERENCE = {"reference_status": "absent", "summary": {}, "series": {}}
_MANIFEST_KEYS = {
    "schema", "case_id", "reference_status", "source_telemetry_sha256", "files",
}
_SAFE_CASE = S._SAFE_FACT_ID


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(path: Path, label: str) -> tuple[bytes, object]:
    if path.is_symlink():
        raise PermissionError(f"심볼릭 링크 입력은 허용되지 않습니다: {label}")
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {label}")
    raw = path.read_bytes()

    def reject_constant(value: str):
        raise ValueError(f"유한하지 않은 JSON 숫자는 허용되지 않습니다: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"중복 JSON 키는 허용되지 않습니다: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"UTF-8 JSON이 아닙니다: {label}") from exc
    return raw, value


def _safe_case_id(case_id: str) -> str:
    if (not isinstance(case_id, str) or not case_id or not _SAFE_CASE.fullmatch(case_id)
            or case_id in (".", "..")):
        raise ValueError(f"안전하지 않은 case id입니다: {case_id!r}")
    return case_id


def _prepared_telemetry(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("telemetry JSON 최상위 값은 객체여야 합니다")
    summary, series = value.get("summary"), value.get("series")
    if not isinstance(summary, dict) or not isinstance(series, dict):
        raise ValueError("telemetry JSON에는 summary와 series 객체가 필요합니다")
    clean_summary = {}
    for tag, row in summary.items():
        if not isinstance(tag, str) or not isinstance(row, dict):
            raise ValueError("telemetry summary는 문자열 tag와 객체 행으로 구성되어야 합니다")
        clean_summary[tag] = {
            field: row[field] for field in G.SUMMARY_FIELDS if field in row
        }
    has_summary_observation = any(
        value is not None for row in clean_summary.values() for value in row.values()
    )
    has_series_observation = any(
        isinstance(values, list) and bool(values) for values in series.values()
    )
    if not has_summary_observation and not has_series_observation:
        raise ValueError("telemetry에는 하나 이상의 관측 지표가 필요합니다")
    return {"summary": clean_summary, "series": series}


def _encode(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@contextmanager
def _workspace(path: Path):
    previous = T.WORKSPACE
    T.WORKSPACE = path
    try:
        yield
    finally:
        T.WORKSPACE = previous


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _relative_files(case_id: str, with_context: bool) -> list[str]:
    files = ["reference/telemetry.json", f"cases/{case_id}/telemetry.json"]
    if with_context:
        files.extend([
            f"cases/{case_id}/configuration.json",
            f"cases/{case_id}/context.json",
        ])
    return files


def prepare_run(telemetry: Path, workspace: Path, case_id: str,
                configuration: Path | None = None,
                facts: Path | None = None) -> dict[str, Any]:
    """입력을 먼저 임시 작업공간에서 검증한 뒤 새 작업공간으로 게시한다."""
    case_id = _safe_case_id(case_id)
    telemetry = Path(telemetry)
    workspace = Path(workspace)
    if (configuration is None) != (facts is None):
        raise ValueError("configuration과 facts는 함께 제공하거나 둘 다 생략해야 합니다")
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError("대상 workspace가 이미 존재합니다")
    workspace.parent.mkdir(parents=True, exist_ok=True)

    source_raw, source_value = _json_bytes(telemetry, "telemetry")
    prepared = _prepared_telemetry(source_value)
    prepared_raw = _encode(prepared)
    configuration_raw = None
    facts_value = None
    if configuration is not None and facts is not None:
        configuration_raw, configuration_value = _json_bytes(Path(configuration), "configuration")
        _, facts_value = _json_bytes(Path(facts), "facts")
        if not isinstance(configuration_value, dict):
            raise ValueError("configuration JSON 최상위 값은 객체여야 합니다")
        if not isinstance(facts_value, list) or not facts_value:
            raise ValueError("facts는 비어 있지 않은 목록이어야 합니다")

    temporary = Path(tempfile.mkdtemp(prefix=f".{workspace.name}.prepare-", dir=workspace.parent))
    try:
        reference_raw = _encode(REFERENCE)
        _write(temporary / "reference" / "telemetry.json", reference_raw)
        case_dir = temporary / "cases" / case_id
        _write(case_dir / "telemetry.json", prepared_raw)
        if configuration_raw is not None:
            _write(case_dir / "configuration.json", configuration_raw)
            context = {
                "schema": "single_run_context_v1",
                "telemetry_sha256": _sha(prepared_raw),
                "configuration_sha256": _sha(configuration_raw),
                "facts": facts_value,
            }
            _write(case_dir / "context.json", _encode(context))

        with _workspace(temporary):
            S.telemetry_overview(case_id)
            S.get_context(case_id)
            G.evidence_catalog(case_id)

        relative_files = _relative_files(case_id, configuration_raw is not None)
        file_hashes = {
            relative: _sha((temporary / relative).read_bytes()) for relative in relative_files
        }
        manifest = {
            "schema": IMPORT_SCHEMA,
            "case_id": case_id,
            "reference_status": "absent",
            "source_telemetry_sha256": _sha(source_raw),
            "files": file_hashes,
        }
        manifest_raw = _encode(manifest)

        workspace.mkdir(exist_ok=False)
        os.replace(temporary / "reference", workspace / "reference")
        os.replace(temporary / "cases", workspace / "cases")
        with (workspace / "import.json").open("xb") as output:
            output.write(manifest_raw)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    return {
        "status": "ready_for_assessment",
        "case_id": case_id,
        "reference_status": "absent",
        "model_evaluation": "not_run",
        "source_telemetry_sha256": _sha(source_raw),
        "prepared_telemetry_sha256": _sha(prepared_raw),
        "import_sha256": _sha(manifest_raw),
    }


def _manifest(workspace: Path, case_id: str) -> tuple[bytes, dict[str, Any]]:
    raw, value = _json_bytes(workspace / "import.json", "import.json")
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ValueError("import.json 최상위 키가 계약과 다릅니다")
    if value["schema"] != IMPORT_SCHEMA:
        raise ValueError("지원하지 않는 import schema입니다")
    if value["case_id"] != case_id:
        raise ValueError("import.json의 case_id가 요청과 다릅니다")
    if value["reference_status"] != "absent":
        raise ValueError("import.json의 reference_status가 absent가 아닙니다")
    if (not isinstance(value["source_telemetry_sha256"], str)
            or not S._LOWER_SHA256.fullmatch(value["source_telemetry_sha256"])):
        raise ValueError("source_telemetry_sha256이 올바르지 않습니다")
    files = value["files"]
    if not isinstance(files, dict):
        raise ValueError("import.json files는 객체여야 합니다")
    with_context = f"cases/{case_id}/context.json" in files
    expected = set(_relative_files(case_id, with_context))
    if set(files) != expected:
        raise ValueError("import.json의 허용 파일 목록이 정확하지 않습니다")
    for relative, digest in files.items():
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts:
            raise PermissionError("import.json에 workspace 경계 밖 경로가 있습니다")
        path = workspace.joinpath(*parts)
        if any(parent.is_symlink() for parent in path.parents if parent != workspace.parent):
            raise PermissionError(f"manifest 파일의 상위 경로가 심볼릭 링크입니다: {relative}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"manifest 파일이 없거나 심볼릭 링크입니다: {relative}")
        if not isinstance(digest, str) or not S._LOWER_SHA256.fullmatch(digest):
            raise ValueError(f"manifest SHA-256이 올바르지 않습니다: {relative}")
        if _sha(path.read_bytes()) != digest:
            raise ValueError(f"manifest SHA-256 불일치: {relative}")
    actual = {
        str(path.relative_to(workspace))
        for root in (workspace / "reference", workspace / "cases")
        for path in root.rglob("*") if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise ValueError("workspace의 실제 파일 목록이 manifest와 다릅니다")
    return raw, value


def inspect_run(case_id: str, workspace: Path,
                report: Path | None = None) -> dict[str, Any]:
    """게시된 입력의 해시와 기존 단일 실행 검증기를 다시 확인한다."""
    case_id = _safe_case_id(case_id)
    workspace = Path(workspace)
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("workspace는 심볼릭 링크가 아닌 기존 디렉터리여야 합니다")
    if report is not None:
        report_resolved = Path(report).resolve()
        input_roots = ((workspace / "reference").resolve(), (workspace / "cases").resolve())
        if any(root == report_resolved or root in report_resolved.parents for root in input_roots):
            raise ValueError("report는 reference/cases 입력 트리 안에 저장할 수 없습니다")
    manifest_raw, manifest = _manifest(workspace, case_id)
    with _workspace(workspace):
        overview = S.telemetry_overview(case_id)
        context = S.get_context(case_id)
        catalog = G.evidence_catalog(case_id)
    counts = {"present": 0, "null": 0, "missing": 0}
    for item in catalog["evidence"]:
        counts[item["state"]] += 1
    result = {
        "status": "ready_for_assessment",
        "case_id": case_id,
        "reference_status": "absent",
        "model_evaluation": "not_run",
        "import_sha256": _sha(manifest_raw),
        "source_telemetry_sha256": manifest["source_telemetry_sha256"],
        "prepared_telemetry_sha256": manifest["files"][f"cases/{case_id}/telemetry.json"],
        "catalog_sha256": catalog["catalog_sha256"],
        "metrics": overview["rows"],
        "configuration": {
            "status": context["status"],
            "facts": context["facts"],
        },
        "evidence_state_counts": counts,
    }
    final_manifest_raw, _ = _manifest(workspace, case_id)
    if final_manifest_raw != manifest_raw:
        raise ValueError("검사 중 import.json이 변경되었습니다")
    if report is not None:
        report_path = Path(report)
        report_raw = _encode(result)
        with report_path.open("xb") as output:
            output.write(report_raw)
        result["report_sha256"] = _sha(report_raw)
    return result
