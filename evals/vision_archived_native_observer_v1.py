"""회수된 native V01 근거를 같은 모델 계약으로 관찰하는 개발용 caller.

GPU 호스트에서는 API를 호출하지 않는다. 검증된 archive를 Stop 뒤 로컬 WSL에서
읽고, 두 사례가 공유하는 API6 원장에 grounded observer 호출만 기록한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "evals"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import grounded_vision_observer_v1 as OBSERVER
import model_decision_journal_v1 as JOURNAL
import vision_evidence_contract_v1 as V01

SCHEMA = "vision_archived_native_observer_protocol_v1"
REGISTRATION_SCHEMA = "vision_archived_native_observer_registration_v1"
PREPARED_SCHEMA = "vision_archived_native_observer_prepared_v1"
RESULT_SCHEMA = "vision_archived_native_observer_result_v1"
GROUPS = {"source-01": 2220, "source-02": 2221}
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha(path: Path, *, lf: bool = False) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if lf else data).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{target.name}은 regular file이어야 한다")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{target.name}은 JSON object여야 한다")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _relative(value: Any, label: str) -> Path:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError(f"{label} 경로 형식 불일치")
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise ValueError(f"{label} 경로 탈출")
    return Path(*posix.parts)


def _bound(archive_root: Path, ref: Any, label: str) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{label} 참조 형식 불일치")
    if HEX64.fullmatch(str(ref.get("sha256"))) is None:
        raise ValueError(f"{label} SHA 형식 불일치")
    root = Path(archive_root).resolve(strict=True)
    path = (root / _relative(ref["path"], label)).resolve(strict=True)
    path.relative_to(root)
    if path.is_symlink() or file_sha(path) != ref["sha256"]:
        raise ValueError(f"{label} archive 결합 불일치")
    return path


def _source_context(repo: Path, protocol: dict[str, Any]) -> None:
    anchor = protocol.get("source_commit")
    if HEX40.fullmatch(str(anchor)) is None:
        raise ValueError("source commit 형식 불일치")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(anchor), current],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("등록 source commit이 현재 HEAD의 조상이 아니다")
    sources = protocol.get("source_sha256_lf")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source SHA 결합 누락")
    for relative, expected in sources.items():
        path = (repo / _relative(relative, "source")).resolve(strict=True)
        path.relative_to(repo.resolve(strict=True))
        if (
            HEX64.fullmatch(str(expected)) is None
            or file_sha(path, lf=True) != expected
        ):
            raise ValueError(f"source 변경: {relative}")


def _git_registration(repo: Path, path: Path, anchor: str) -> None:
    if HEX40.fullmatch(anchor) is None:
        raise ValueError("registration anchor 형식 불일치")
    repository = Path(repo).resolve(strict=True)
    target = Path(path).resolve(strict=True)
    relative = target.relative_to(repository).as_posix()
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, current],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("registration anchor가 현재 HEAD의 조상이 아니다")
    blob = subprocess.run(
        ["git", "show", f"{anchor}:{relative}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(blob).hexdigest() != file_sha(target):
        raise ValueError("공개 registration의 committed blob과 현재 파일이 다르다")


def evidence_fingerprint(contract: dict[str, Any]) -> dict[str, Any]:
    provenance = contract["provenance"]
    commands = provenance["frame_extraction"]["commands"]
    frames = contract["frames"]
    if len(frames) != 6 or len(commands) != 6:
        raise ValueError("V01 여섯 frame 계약 불일치")
    return {
        "video_sha256": provenance["video"]["sha256"],
        "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        "measurement_sha256": (
            None
            if provenance["measurement"] is None
            else provenance["measurement"]["sha256"]
        ),
        "state_digest": digest(contract["synchronized_samples"]),
        "frames": [
            {
                "frame_id": frame["frame_id"],
                "sha256": frame["sha256"],
                "source_frame_index": command["source_frame_index"],
                "source_frame_pts_s": command["source_frame_pts_s"],
            }
            for frame, command in zip(frames, commands, strict=True)
        ],
    }


def _archive_manifest(archive_root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _bound(
        archive_root, protocol["archive"]["manifest"], "archive manifest"
    )
    manifest = _read(manifest_path)
    if (
        manifest.get("schema") != "vision_transferred_e2e_close_manifest_v1"
        or manifest.get("archive_repo_prefix") != "repo"
        or manifest.get("source_cache_rearchived") is not False
    ):
        raise ValueError("close archive manifest 계약 불일치")
    return manifest


def _manifest_binding(
    manifest: dict[str, Any],
    archived_path: Path,
    archive_root: Path,
    source_ref: dict[str, Any],
) -> None:
    archive_relative = archived_path.relative_to(
        (archive_root / "tree").resolve()
    ).as_posix()
    matches = [
        row
        for row in manifest.get("files", [])
        if row.get("archive_path") == archive_relative
        and row.get("source_path") == source_ref.get("path")
        and row.get("sha256") == source_ref.get("sha256")
    ]
    if len(matches) != 1:
        raise ValueError("원격 원문과 archive member 결합 불일치")


def validate_protocol(
    protocol_path: Path,
    archive_root: Path,
    registration_path: Path,
    registration_anchor: str,
    *,
    repo: Path = ROOT,
    verify_source: bool = True,
) -> dict[str, Any]:
    path = Path(protocol_path).resolve(strict=True)
    protocol = _read(path)
    required = {
        "schema",
        "protocol_id",
        "registered_before_api",
        "development_scope",
        "source_commit",
        "source_sha256_lf",
        "archive",
        "model",
        "limits",
        "prior_execution",
        "rows",
        "preregistration",
    }
    if set(protocol) != required or protocol.get("schema") != SCHEMA:
        raise ValueError("archived observer protocol 형식 불일치")
    if (
        protocol.get("registered_before_api") is not True
        or protocol.get("development_scope")
        != "known_healthy_corrected_policy_auxiliary_observation"
        or set(protocol.get("archive", {}))
        != {"manifest", "archive_repo_prefix", "raw_archive_sha256"}
        or protocol["archive"].get("archive_repo_prefix") != "repo"
        or HEX64.fullmatch(str(protocol["archive"].get("raw_archive_sha256"))) is None
        or protocol.get("model")
        != {"name": OBSERVER.MODEL, "params": OBSERVER.REQUEST_PARAMS}
        or protocol.get("limits")
        != {"new_api_requests": 6, "cases": 2, "max_calls_per_case": 3}
        or protocol.get("prior_execution")
        != {
            "initial_model_api_requests": 2,
            "remote_post_observation_api_requests": 0,
            "combined_api_requests_after_upper_bound": 8,
            "original_e2e_api_cap": 32,
        }
    ):
        raise ValueError("observer 범위·모델·API 상한 불일치")
    expected_preregistration = {
        "prediction": "두 known-healthy 교정 정책의 실제 native 프레임에서 walking=yes, falling=no 관찰을 예측한다.",
        "valid": "grounded 계약이 유효하면 보조 관찰로 기록하고 회복 판정에는 사용하지 않는다.",
        "hold": "전송·형식·근거 검증이 끝나지 않으면 observed_HOLD로 기록하며 유효 관찰로 바꾸지 않는다.",
        "label_separation": "known-healthy 개발 관찰이며 새 회복률·일반화 근거로 사용하지 않는다.",
    }
    if protocol.get("preregistration") != expected_preregistration:
        raise ValueError("결과 전 고정 문구 불일치")
    if verify_source:
        _source_context(Path(repo).resolve(strict=True), protocol)
    archive = Path(archive_root).resolve(strict=True)
    raw_archive = archive / "retrieved" / "e2e-close-raw.tar.gz"
    if (
        raw_archive.is_symlink()
        or not raw_archive.is_file()
        or file_sha(raw_archive) != protocol["archive"]["raw_archive_sha256"]
    ):
        raise ValueError("회수 archive SHA 결합 불일치")
    manifest = _archive_manifest(archive, protocol)
    rows = protocol.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("두 관찰 row가 필요하다")
    checked = []
    for row in rows:
        group = row.get("group") if isinstance(row, dict) else None
        if (
            group not in GROUPS
            or row.get("seed") != GROUPS[group]
            or row.get("label") != "known_healthy_corrected_policy"
            or row.get("window") != 1
            or row.get("provenance_transition")
            != "absolute_remote_reference_to_verified_archive_member_only"
            or row.get("used_for_decision") is not False
        ):
            raise ValueError("관찰 row identity 불일치")
        receipt_path = _bound(
            archive, row.get("native_receipt"), f"{group} native receipt"
        )
        receipt = _read(receipt_path)
        if (
            receipt.get("schema") != "vision_corrected_native_record_receipt_v1"
            or receipt.get("status") != "verified_native_record"
            or receipt.get("seed") != GROUPS[group]
            or receipt.get("checkpoint_sha256") != row.get("checkpoint_sha256")
            or any(
                receipt.get(name) is not False
                for name in ("capture_approved", "model_ready", "used_for_decision")
            )
        ):
            raise ValueError("native receipt identity 또는 사용 경계 불일치")
        contract_rows = [
            item for item in receipt.get("v01_contracts", []) if item.get("window") == 1
        ]
        if len(contract_rows) != 1 or contract_rows[0].get("verified") is not True:
            raise ValueError("native window-1 V01 참조 불일치")
        contract_path = _bound(archive, row.get("contract"), f"{group} V01 contract")
        source_ref = contract_rows[0].get("contract")
        if (
            not isinstance(source_ref, dict)
            or source_ref.get("sha256") != row["contract"]["sha256"]
        ):
            raise ValueError("native receipt와 archived V01 SHA 불일치")
        _manifest_binding(manifest, contract_path, archive, source_ref)
        contract = V01.verify(contract_path)
        fingerprint = evidence_fingerprint(contract)
        messages_sha = JOURNAL.digest(OBSERVER.request_view(contract_path)["messages"])
        if (
            fingerprint != row.get("evidence")
            or fingerprint["checkpoint_sha256"] != row["checkpoint_sha256"]
            or row.get("request_messages_sha256") != messages_sha
            or row.get("original_request_messages_sha256") != messages_sha
        ):
            raise ValueError(
                "V01 frame/video/state/checkpoint 또는 model view 결합 불일치"
            )
        checked.append({**deepcopy(row), "contract_path": contract_path})
    if {row["group"] for row in checked} != set(GROUPS):
        raise ValueError("관찰 group 중복 또는 누락")
    protocol_sha = file_sha(path)
    registration_file = Path(registration_path).resolve(strict=True)
    registration_file.relative_to(Path(repo).resolve(strict=True))
    registration = _read(registration_file)
    expected_registration = {
        "schema": REGISTRATION_SCHEMA,
        "protocol_id": protocol["protocol_id"],
        "bound_protocol_sha256": protocol_sha,
        "source_commit": protocol["source_commit"],
        "source_sha256_lf": protocol["source_sha256_lf"],
        "archive_sha256": protocol["archive"]["raw_archive_sha256"],
        "rows": [
            {
                "group": row["group"],
                "seed": row["seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "contract_sha256": row["contract"]["sha256"],
                "request_messages_sha256": row["request_messages_sha256"],
            }
            for row in protocol["rows"]
        ],
        "model": protocol["model"],
        "api": {
            "new_api_cap": 6,
            "remote_post_observation_api_requests": 0,
            "prior_initial_model_api_requests": 2,
            "combined_api_requests_after_upper_bound": 8,
            "original_e2e_api_cap": 32,
        },
        "auxiliary_used_for_decision": False,
    }
    if registration != expected_registration:
        raise ValueError("공개 registration과 비공개 bound protocol 불일치")
    if any(
        token in json.dumps(registration, ensure_ascii=False)
        for token in ("bench/private", ".triage-runtime", "tree/repo", "/home/", "C:\\")
    ):
        raise ValueError("공개 registration에 private·runtime·절대 경로가 있다")
    if verify_source:
        _git_registration(Path(repo), registration_file, registration_anchor)
    return {
        "path": path,
        "sha256": protocol_sha,
        "value": protocol,
        "rows": checked,
        "registration": registration,
        "registration_sha256": file_sha(registration_file),
        "registration_anchor": registration_anchor,
    }


def prepare(
    protocol_path: Path,
    archive_root: Path,
    registration_path: Path,
    registration_anchor: str,
    output: Path,
) -> dict[str, Any]:
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    checked = validate_protocol(
        protocol_path, archive_root, registration_path, registration_anchor
    )
    target.mkdir(parents=True)
    value = {
        "schema": PREPARED_SCHEMA,
        "protocol_sha256": checked["sha256"],
        "registration_sha256": checked["registration_sha256"],
        "registration_anchor": checked["registration_anchor"],
        "archive_manifest_sha256": checked["value"]["archive"]["manifest"]["sha256"],
        "groups": [row["group"] for row in checked["rows"]],
        "api_cap": 6,
        "used_for_decision": False,
    }
    _write_new(target / "prepared.json", value)
    return value


def _prepared(output: Path, checked: dict[str, Any]) -> dict[str, Any]:
    value = _read(Path(output) / "prepared.json")
    if (
        value.get("schema") != PREPARED_SCHEMA
        or value.get("protocol_sha256") != checked["sha256"]
        or value.get("registration_sha256") != checked["registration_sha256"]
        or value.get("registration_anchor") != checked["registration_anchor"]
        or value.get("groups") != [row["group"] for row in checked["rows"]]
        or value.get("api_cap") != 6
        or value.get("used_for_decision") is not False
    ):
        raise ValueError("prepared receipt와 현재 protocol 불일치")
    return value


def advance(
    protocol_path: Path,
    archive_root: Path,
    registration_path: Path,
    registration_anchor: str,
    output: Path,
    group: str,
    complete: Callable[[list[dict], dict], dict] = OBSERVER.default_complete,
) -> dict[str, Any]:
    checked = validate_protocol(
        protocol_path, archive_root, registration_path, registration_anchor
    )
    _prepared(output, checked)
    rows = [row for row in checked["rows"] if row["group"] == group]
    if len(rows) != 1:
        raise ValueError("등록 group이 아니다")
    row = rows[0]
    binding = OBSERVER.binding_template(
        row["contract_path"],
        run_id=f"archived-native-observer-{group}-v1",
        source_anchor=checked["value"]["source_commit"],
        protocol_sha256=checked["sha256"],
    )
    folder = Path(output) / "cases" / group
    result = OBSERVER.run(
        folder,
        row["contract_path"],
        binding,
        complete,
        api_root=Path(output) / "shared-api",
        api_cap=6,
    )
    terminal = result.get("status") in {"observation_ready", "terminal_hold"}
    if not terminal:
        return result
    value = {
        "schema": RESULT_SCHEMA,
        "group": group,
        "seed": row["seed"],
        "label": row["label"],
        "status": result["status"],
        "observer": result,
        "protocol_sha256": checked["sha256"],
        "contract_sha256": row["contract"]["sha256"],
        "request_messages_sha256": row["request_messages_sha256"],
        "used_for_decision": False,
    }
    result_path = folder / "result.json"
    if result_path.exists():
        if _read(result_path) != value:
            raise ValueError("기존 관찰 결과와 replay가 다르다")
    else:
        _write_new(result_path, value)
    return value


def inspect(
    protocol_path: Path,
    archive_root: Path,
    registration_path: Path,
    registration_anchor: str,
    output: Path,
) -> dict[str, Any]:
    checked = validate_protocol(
        protocol_path, archive_root, registration_path, registration_anchor
    )
    _prepared(output, checked)
    rows = []
    for row in checked["rows"]:
        result = _read(Path(output) / "cases" / row["group"] / "result.json")
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("group") != row["group"]
            or result.get("protocol_sha256") != checked["sha256"]
            or result.get("contract_sha256") != row["contract"]["sha256"]
            or result.get("used_for_decision") is not False
            or result.get("status") not in {"observation_ready", "terminal_hold"}
        ):
            raise ValueError("관찰 결과 결합 불일치")
        rows.append({"group": row["group"], "status": result["status"]})
    reservations = sorted((Path(output) / "shared-api" / "reservations").glob("*.json"))
    responses = sorted((Path(output) / "shared-api" / "responses").glob("*.json"))
    if len(reservations) > 6 or len(reservations) != len(responses):
        raise ValueError("shared API6 원장 pending 또는 상한 초과")
    return {
        "status": "complete",
        "rows": rows,
        "api": {"reserved": len(reservations), "pending": 0, "cap": 6},
        "used_for_decision": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "advance", "inspect"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-anchor", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group", choices=tuple(GROUPS))
    args = parser.parse_args()
    if args.operation == "prepare":
        result = prepare(
            args.protocol,
            args.archive_root,
            args.registration,
            args.registration_anchor,
            args.output,
        )
    elif args.operation == "advance":
        if args.group is None:
            parser.error("advance에는 --group이 필요하다")
        result = advance(
            args.protocol,
            args.archive_root,
            args.registration,
            args.registration_anchor,
            args.output,
            args.group,
        )
    else:
        result = inspect(
            args.protocol,
            args.archive_root,
            args.registration,
            args.registration_anchor,
            args.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
