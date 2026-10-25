"""metadata-only GPU 수명주기 원문과 비용을 resource proof 없이 검증한다."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath
from typing import Any

UTC = timezone.utc  # noqa: UP017 - Python 3.10 호환
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
METADATA_ONLY_PATHS = frozenset({
    "admission/before.json",
    "admission/provider-bootstrap-sidecar.json",
    "admission/provider-bootstrap.json",
    "admission/provider.json",
    "admission/relocation.json",
    "admission/start.json",
    "admission/start.stderr",
    "admission/start.stdout",
    "admission/verify_atomic_paced_admission.py",
    "admission/vision-source-20261025j-stop.json",
    "admission/vision-source-20261025k-stop.json",
    "admission/vision-source-20261025l-stop.json",
    "admission/vision_atomic_compatibility_20261025_v3.json",
    "admission/vision_brev_render_20261025_v14.json",
    "admission/vision_brev_source_20261025_v13.json",
    "bootstrap.json",
})


@dataclass(frozen=True)
class MetadataLifecycleCost:
    run_id: str
    started_at_utc: str
    stopped_at_utc: str
    wall_seconds: Decimal
    cost_usd: Decimal


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"원문이 단일 regular file이 아니다: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object가 아니다: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _under_repo(repo: Path, candidate: Path, label: str) -> Path:
    """repo부터 대상까지 lexical 구성요소에 symlink가 없는지 확인한다."""
    lexical = candidate.absolute()
    relative = lexical.relative_to(repo)
    current = repo
    if current.is_symlink():
        raise ValueError("repo가 symlink다")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} 경로 구성요소가 symlink다")
    resolved = lexical.resolve(strict=True)
    resolved.relative_to(repo)
    return resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} 필드가 다르다")


def _decimal(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"{label}은 십진 문자열이어야 한다")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"{label}은 유한값이어야 한다")
    return result


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label}은 문자열이어야 한다")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))  # noqa: FURB162
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label}에 timezone이 없다")
    return parsed.astimezone(UTC)


def _reference(repo: Path, value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise TypeError(f"{label} ref가 object가 아니다")
    _exact_keys(value, {"path", "sha256"}, f"{label} ref")
    raw = value["path"]
    if not isinstance(raw, str) or not isinstance(value["sha256"], str):
        raise TypeError(f"{label} ref 값이 문자열이 아니다")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts \
            or relative.parts[:3] != (".triage-runtime", "vision-brev", "lifecycle"):
        raise ValueError(f"{label} ref가 허용 상대경로 밖이다")
    path = _under_repo(repo, repo.joinpath(*relative.parts), label)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"{label} ref가 단일 regular file이 아니다")
    if _sha(path) != value["sha256"]:
        raise ValueError(f"{label} raw SHA가 다르다")
    return path


def _transport(
    repo: Path, receipt_ref: Any, stdout_ref: Any, stderr_ref: Any,
    *, command: list[str], label: str,
) -> tuple[dict[str, Any], Path, Path]:
    receipt_path = _reference(repo, receipt_ref, f"{label} receipt")
    stdout = _reference(repo, stdout_ref, f"{label} stdout")
    stderr = _reference(repo, stderr_ref, f"{label} stderr")
    receipt = _read_object(receipt_path)
    _exact_keys(receipt, {
        "command", "request_before_utc", "response_after_utc", "exit_code",
        "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
    }, f"{label} receipt")
    if receipt["command"] != command or type(receipt["exit_code"]) is not int \
            or receipt["exit_code"] != 0:
        raise ValueError(f"{label} command 또는 exit code가 다르다")
    if not all(
        isinstance(receipt[key], str) and receipt[key]
        for key in ("stdout_path", "stderr_path")
    ):
        raise TypeError(f"{label} native stream path가 문자열이 아니다")
    if receipt["stdout_sha256"] != _sha(stdout) \
            or receipt["stderr_sha256"] != _sha(stderr):
        raise ValueError(f"{label} native stream SHA와 envelope ref가 다르다")
    requested = _utc(receipt["request_before_utc"], f"{label} request")
    responded = _utc(receipt["response_after_utc"], f"{label} response")
    if responded < requested:
        raise ValueError(f"{label} transport 시각이 역전됐다")
    return receipt, stdout, stderr


def _owned_status(raw: Path, owner: Mapping[str, str], status: str, label: str) -> None:
    body = _read_object(raw)
    rows = body.get("workspaces")
    if not isinstance(rows, list):
        raise TypeError(f"{label} workspaces가 list가 아니다")
    by_id = [row for row in rows if isinstance(row, dict) and row.get("id") == owner["id"]]
    by_name = [row for row in rows if isinstance(row, dict) and row.get("name") == owner["name"]]
    if len(by_id) != 1 or len(by_name) != 1 or by_id[0] != by_name[0] \
            or by_id[0].get("status") != status \
            or by_id[0].get("instance_type") != owner["instance_type"]:
        raise ValueError(f"{label} exact owner {status} 원문이 다르다")


def _verify_metadata_capture(
    repo: Path, capture: Mapping[str, Any], owner: Mapping[str, str],
) -> None:
    _exact_keys(capture, {"inventory", "root", "files"}, "metadata capture")
    if capture["files"] != 16 or not isinstance(capture["root"], str):
        raise ValueError("metadata capture 파일 수 또는 root가 다르다")
    inventory_path = _reference(repo, capture["inventory"], "metadata inventory")
    root_relative = PurePosixPath(capture["root"])
    if root_relative.is_absolute() or ".." in root_relative.parts \
            or root_relative.parts[:3] != (".triage-runtime", "vision-brev", "lifecycle"):
        raise ValueError("metadata root가 허용 상대경로 밖이다")
    root = _under_repo(repo, repo.joinpath(*root_relative.parts), "metadata root")
    inventory = _read_object(inventory_path)
    _exact_keys(inventory, {"schema", "files", "remote_root", "resource_id"}, "metadata inventory")
    if inventory["schema"] != "paced_hold_metadata_inventory_v1" \
            or inventory["resource_id"] != owner["id"] \
            or not isinstance(inventory["remote_root"], str) \
            or not inventory["remote_root"].startswith("/home/ubuntu/workspace/") \
            or not isinstance(inventory["files"], list) \
            or len(inventory["files"]) != capture["files"]:
        raise ValueError("metadata inventory identity 또는 파일 수가 다르다")
    seen: set[str] = set()
    for row in inventory["files"]:
        if not isinstance(row, dict):
            raise TypeError("metadata inventory file row가 object가 아니다")
        _exact_keys(row, {"path", "bytes", "sha256"}, "metadata file row")
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise ValueError("metadata file 상대경로가 잘못됐거나 중복됐다")
        seen.add(relative.as_posix())
        path = _under_repo(repo, root.joinpath(*relative.parts), "metadata file")
        path.relative_to(root)
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("metadata file이 단일 regular file이 아니다")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) \
                or row["bytes"] != path.stat().st_size or row["sha256"] != _sha(path):
            raise ValueError("metadata file bytes 또는 SHA가 다르다")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("metadata root에 symlink가 있다")
    actual = {
        path.relative_to(root).as_posix() for path in entries if path.is_file()
    }
    if actual != seen or seen != METADATA_ONLY_PATHS:
        raise ValueError("metadata root 실제 파일 집합이 inventory와 다르다")


def validate_metadata_lifecycle(
    envelope_path: Path, *, repo: Path, expected_owner: Mapping[str, str],
    expected_hourly_rate_usd: Decimal,
    allowed_inspection_script_sha256: set[str],
) -> MetadataLifecycleCost:
    """봉인된 metadata-only 수명주기 한 건을 검증해 누적 가능한 비용을 반환한다."""
    if set(expected_owner) != {"id", "name", "instance_type"} or not all(
        isinstance(expected_owner[key], str) and expected_owner[key]
        for key in ("id", "name", "instance_type")
    ):
        raise ValueError("expected owner가 다르다")
    if not expected_hourly_rate_usd.is_finite() or expected_hourly_rate_usd <= 0:
        raise ValueError("expected hourly rate가 유효하지 않다")
    if not allowed_inspection_script_sha256 or any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in allowed_inspection_script_sha256
    ):
        raise ValueError("허용 inspection script SHA 집합이 다르다")
    repo_source = Path(repo)
    if repo_source.is_symlink():
        raise ValueError("repo가 symlink다")
    repo = repo_source.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("repo가 실제 directory가 아니다")
    envelope_path = _under_repo(repo, Path(envelope_path), "envelope")
    envelope = _read_object(envelope_path)
    _exact_keys(envelope, {
        "schema", "run_id", "owner", "quote", "period", "reported",
        "claims", "evidence", "metadata_capture",
    }, "metadata lifecycle envelope")
    if envelope["schema"] != "vision_metadata_lifecycle_v1" \
            or not isinstance(envelope["run_id"], str) or not envelope["run_id"] \
            or envelope["owner"] != dict(expected_owner):
        raise ValueError("metadata lifecycle identity가 다르다")
    quote = envelope["quote"]
    if not isinstance(quote, dict):
        raise TypeError("quote가 object가 아니다")
    _exact_keys(quote, {"currency", "hourly_rate_usd", "rounding", "quantum_usd"}, "quote")
    hourly = _decimal(quote["hourly_rate_usd"], "hourly rate")
    if quote["currency"] != "USD" or quote["rounding"] != "ROUND_HALF_UP" \
            or quote["quantum_usd"] != "0.000001" or hourly != expected_hourly_rate_usd:
        raise ValueError("quote가 protocol 허용값과 다르다")
    claims = envelope["claims"]
    expected_claims = {"resource_proof_issued", "lease_issued", "measurement_resumed"}
    if not isinstance(claims, dict) or set(claims) != expected_claims \
            or any(claims[key] is not False for key in expected_claims):
        raise ValueError("metadata-only claim이 다르다")
    evidence = envelope["evidence"]
    if not isinstance(evidence, dict):
        raise TypeError("evidence가 object가 아니다")
    expected_evidence = {
        "start_observation", "before_inventory", "start_receipt", "start_stdout",
        "start_stderr", "inspection_intent", "inspection_script", "idle_receipt",
        "idle_stdout", "idle_stderr", "stop_receipt", "stop_stdout", "stop_stderr",
        "final_inventory_receipt", "final_inventory_stdout", "final_inventory_stderr",
    }
    _exact_keys(evidence, expected_evidence, "evidence")

    owner = dict(expected_owner)
    before = _reference(repo, evidence["before_inventory"], "before inventory")
    _owned_status(before, owner, "STOPPED", "before inventory")
    start_observation_path = _reference(
        repo, evidence["start_observation"], "start observation"
    )
    start_observation = _read_object(start_observation_path)
    _exact_keys(start_observation, {
        "schema", "resource_id", "resource_name", "before_inventory_path",
        "before_inventory_sha256", "command", "request_before_utc",
        "response_after_utc", "exit_code", "stdout_path", "stdout_sha256",
        "stderr_path", "stderr_sha256",
    }, "start observation")
    start_command = ["brev", "start", owner["name"], "--detached"]
    if start_observation["schema"] != "v04_start_observation_v1" \
            or start_observation["resource_id"] != owner["id"] \
            or start_observation["resource_name"] != owner["name"] \
            or not isinstance(start_observation["before_inventory_path"], str) \
            or start_observation["before_inventory_sha256"] != _sha(before):
        raise ValueError("start observation identity 또는 before SHA가 다르다")
    start_receipt, start_stdout, start_stderr = _transport(
        repo, evidence["start_receipt"], evidence["start_stdout"],
        evidence["start_stderr"], command=start_command, label="start",
    )
    if any(start_observation[key] != start_receipt[key] for key in start_receipt) \
            or start_observation["stdout_sha256"] != _sha(start_stdout) \
            or start_observation["stderr_sha256"] != _sha(start_stderr) \
            or start_stderr.read_bytes():
        raise ValueError("start observation과 native start 원문이 다르다")

    intent_path = _reference(repo, evidence["inspection_intent"], "inspection intent")
    script = _reference(repo, evidence["inspection_script"], "inspection script")
    intent = _read_object(intent_path)
    _exact_keys(intent, {
        "schema", "source_path", "source_sha256", "snapshot_path", "command",
    }, "inspection intent")
    inspection_sha = _sha(script)
    if intent["schema"] != "vision_atomic_exec_script_intent_v1" \
            or not isinstance(intent["source_path"], str) or not intent["source_path"] \
            or not isinstance(intent["snapshot_path"], str) or not intent["snapshot_path"] \
            or intent["source_sha256"] != inspection_sha \
            or inspection_sha not in allowed_inspection_script_sha256:
        raise ValueError("inspection script 결합 또는 허용 SHA가 다르다")
    idle_command = intent["command"]
    if not isinstance(idle_command, list) or len(idle_command) != 4 \
            or idle_command[:3] != ["brev", "exec", owner["name"]] \
            or idle_command[3] != "@" + intent["snapshot_path"]:
        raise ValueError("inspection command가 다르다")
    idle_receipt, idle_stdout, idle_stderr = _transport(
        repo, evidence["idle_receipt"], evidence["idle_stdout"],
        evidence["idle_stderr"], command=idle_command, label="idle inspection",
    )
    if idle_stdout.read_bytes() != (owner["name"] + "\n").encode() \
            or idle_stderr.read_bytes():
        raise ValueError("idle 출력에 footer 외 내용이 있다")

    stop_command = ["brev", "stop", owner["name"]]
    stop_receipt, stop_stdout, stop_stderr = _transport(
        repo, evidence["stop_receipt"], evidence["stop_stdout"],
        evidence["stop_stderr"], command=stop_command, label="stop",
    )
    if stop_stdout.read_bytes() != (owner["name"] + "\n").encode() \
            or stop_stderr.read_bytes():
        raise ValueError("stop 출력이 다르다")
    final_receipt, final_stdout, final_stderr = _transport(
        repo, evidence["final_inventory_receipt"],
        evidence["final_inventory_stdout"], evidence["final_inventory_stderr"],
        command=["brev", "ls", "--json"], label="final inventory",
    )
    if final_stderr.read_bytes():
        raise ValueError("final inventory stderr가 비어 있지 않다")
    _owned_status(final_stdout, owner, "STOPPED", "final inventory")

    start_request = _utc(start_receipt["request_before_utc"], "start request")
    start_response = _utc(start_receipt["response_after_utc"], "start response")
    idle_request = _utc(idle_receipt["request_before_utc"], "idle request")
    idle_response = _utc(idle_receipt["response_after_utc"], "idle response")
    stop_request = _utc(stop_receipt["request_before_utc"], "stop request")
    stop_response = _utc(stop_receipt["response_after_utc"], "stop response")
    final_request = _utc(final_receipt["request_before_utc"], "final request")
    final_response = _utc(final_receipt["response_after_utc"], "final response")
    if not start_request <= start_response <= idle_request <= idle_response \
            <= stop_request <= stop_response <= final_request <= final_response:
        raise ValueError("metadata lifecycle 시각 순서가 다르다")

    period = envelope["period"]
    reported = envelope["reported"]
    if not isinstance(period, dict) or not isinstance(reported, dict):
        raise TypeError("period 또는 reported가 object가 아니다")
    _exact_keys(period, {"started_at_utc", "stopped_at_utc"}, "period")
    _exact_keys(reported, {"wall_seconds", "cost_usd"}, "reported")
    if _utc(period["started_at_utc"], "period start") != start_request \
            or _utc(period["stopped_at_utc"], "period stop") != final_response:
        raise ValueError("period가 start/final 원문과 다르다")
    wall = _decimal(reported["wall_seconds"], "wall seconds")
    cost = _decimal(reported["cost_usd"], "cost")
    duration = final_response - start_request
    actual_wall = (
        Decimal(duration.days * 86400 + duration.seconds)
        + Decimal(duration.microseconds) / Decimal(1_000_000)
    )
    expected_cost = (
        hourly * actual_wall / Decimal(3600)
    ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    if wall != actual_wall or cost != expected_cost:
        raise ValueError("metadata lifecycle 시간 또는 비용 계산이 다르다")
    _verify_metadata_capture(repo, envelope["metadata_capture"], owner)
    return MetadataLifecycleCost(
        run_id=envelope["run_id"], started_at_utc=period["started_at_utc"],
        stopped_at_utc=period["stopped_at_utc"], wall_seconds=wall, cost_usd=cost,
    )
