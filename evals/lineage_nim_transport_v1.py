"""Windows host와 기존 WSL worker 사이의 write-once NIM message transport."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import selection_comparison_call_v2 as CALL2

F = CALL2.F
SCHEMA = "lineage_nim_transport_protocol_v1"
SESSION_SCHEMA = "lineage_nim_transport_session_v1"
REQUEST_SCHEMA = "lineage_nim_transport_request_v1"
RESPONSE_SCHEMA = "lineage_nim_transport_response_v1"
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")
MAX_WORKER_WALL_S = 21600.0
RESULT_KEYS = {
    "response_id", "provider_model", "content", "finish_reason", "reasoning_content",
    "input_tokens", "output_tokens", "total_tokens", "wall_s", "error_type",
    "status_code", "infra_error",
}
SOURCE_FILES = (
    "evals/lineage_nim_transport_v1.py",
    "tests/test_lineage_nim_transport_v1.py",
    "evals/selection_comparison_call_v2.py",
    "evals/selection_comparison_controller_v1.py",
    "evals/selection_comparison_view_v1.py",
    "evals/residual_model_call_v1.py",
    "evals/feedback_model_call.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _inside(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError("queue 상대경로 형식이 잘못됐다")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError("queue 경로가 root 밖을 가리킨다")
    return path


def _git(*args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git source/protocol 결합이 확인되지 않았다")
    return result.stdout


def _plan(value: Any) -> dict:
    if not isinstance(value, dict) or set(value) != {
            "case_ids", "logical_slots", "global_request_cap", "per_slot_max"}:
        raise ValueError("transport plan 필드가 다르다")
    cases, slots = value["case_ids"], value["logical_slots"]
    if (not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)
            or not all(isinstance(row, str) and IDENTIFIER.fullmatch(row) for row in cases)):
        raise ValueError("정식 case ID가 비어 있거나 중복됐다")
    if not isinstance(slots, list) or not slots:
        raise ValueError("logical slot 사전등록이 없다")
    found: set[str] = set()
    for row in slots:
        if (not isinstance(row, dict) or set(row) != {"slot_id", "case_id"}
                or not isinstance(row["slot_id"], str) or not IDENTIFIER.fullmatch(row["slot_id"])
                or row["slot_id"] in found or row["case_id"] not in cases):
            raise ValueError("logical slot ID/case 결합이 다르거나 중복됐다")
        found.add(row["slot_id"])
    if (type(value["global_request_cap"]) is not int or value["global_request_cap"] <= 0
            or value["per_slot_max"] != 3):
        raise ValueError("global cap 또는 slot당 최대 3회 계약이 다르다")
    return json.loads(json.dumps(value))


def _verify_sources(protocol_path: Path, anchor: str) -> dict:
    if not HEX40.fullmatch(anchor):
        raise ValueError("source commit 형식이 잘못됐다")
    protocol_path = Path(protocol_path).resolve()
    try:
        relative = protocol_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("protocol은 repo 내부여야 한다") from exc
    value = _read(protocol_path)
    if (value.get("schema") != SCHEMA or "source_commit" in value
            or value.get("model") != CALL2.MODEL
            or value.get("accepted_provider_models") != sorted(CALL2.ACCEPTED_PROVIDER_MODELS)
            or value.get("params") != F.request_params(CALL2.MODEL)
            or value.get("operator_approved") is not True
            or not isinstance(value.get("source_sha256_lf"), dict)
            or set(value["source_sha256_lf"]) != set(SOURCE_FILES)):
        raise ValueError("transport protocol의 모델·source·operator 결합이 다르다")
    _plan(value.get("plan"))
    _git("cat-file", "-e", f"{anchor}^{{commit}}")
    _git("merge-base", "--is-ancestor", anchor, "HEAD")
    bound = {relative: _file_sha(protocol_path, lf=True), **value["source_sha256_lf"]}
    for name, expected in bound.items():
        path = _inside(ROOT, name)
        if not HEX64.fullmatch(str(expected)) or not path.is_file() or _file_sha(path, lf=True) != expected:
            raise ValueError(f"현재 source SHA가 protocol과 다르다: {name}")
        committed = _git("show", f"{anchor}:{name}").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"source commit blob이 protocol과 다르다: {name}")
    return value


def initialize(root: Path, protocol_path: Path, anchor: str, plan: dict) -> dict:
    """사전등록 계획과 commit/blob을 검증하고 fresh queue를 만든다."""
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError("transport root는 fresh 경로여야 한다")
    protocol = _verify_sources(protocol_path, anchor)
    checked_plan = _plan(plan)
    if protocol["plan"] != checked_plan:
        raise ValueError("호출 plan이 등록 protocol과 다르다")
    root.mkdir(parents=True)
    for name in ("requests", "dispatches", "responses", "response-sha", "failures", "locks", "worker"):
        (root / name).mkdir()
    try:
        protocol_relative = Path(protocol_path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("protocol은 repo 내부여야 한다") from exc
    receipt = {
        "schema": SESSION_SCHEMA, "protocol_relative": protocol_relative,
        "protocol_sha256_lf": _file_sha(protocol_path, lf=True), "registration_anchor": anchor,
        "plan": checked_plan, "model": CALL2.MODEL,
        "accepted_provider_models": sorted(CALL2.ACCEPTED_PROVIDER_MODELS),
        "params": F.request_params(CALL2.MODEL),
    }
    _write_new(root / "session.json", receipt)
    return receipt


def _session(root: Path) -> dict:
    root = Path(root).resolve()
    value = _read(root / "session.json")
    if (value.get("schema") != SESSION_SCHEMA or value.get("model") != CALL2.MODEL
            or value.get("accepted_provider_models") != sorted(CALL2.ACCEPTED_PROVIDER_MODELS)
            or value.get("params") != F.request_params(CALL2.MODEL)
            or value.get("plan") != _plan(value.get("plan"))
            or _file_sha(_inside(ROOT, value.get("protocol_relative", "")), lf=True)
            != value.get("protocol_sha256_lf")):
        raise ValueError("transport session이 protocol/model과 다르다")
    protocol = _verify_sources(_inside(ROOT, value["protocol_relative"]), value["registration_anchor"])
    if protocol["plan"] != value["plan"]:
        raise ValueError("session plan이 protocol과 다르다")
    return value


@contextmanager
def _lock(root: Path, name: str) -> Iterator[None]:
    path = Path(root) / "locks" / name
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("병렬 transport 전송은 허용되지 않는다") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        if path.exists() and path.read_text(encoding="ascii") == str(os.getpid()):
            path.unlink()


def _inventory(root: Path) -> tuple[list[Path], list[Path]]:
    requests = sorted((Path(root) / "requests").iterdir())
    responses = sorted((Path(root) / "responses").iterdir())
    if any(path.suffix != ".json" or not path.is_file() for path in (*requests, *responses)):
        raise ValueError("queue에 등록되지 않은 entry가 있다")
    return requests, responses


def _messages(value: Any) -> list[dict]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(row, dict) or set(row) != {"role", "content"}
                   or row["role"] not in {"system", "user", "assistant"}
                   or not isinstance(row["content"], str) for row in value)):
        raise ValueError("canonical chat messages 형식이 다르다")
    return json.loads(json.dumps(value))


def _closed(root: Path) -> dict | None:
    path = Path(root) / "closed.json"
    if not path.exists():
        return None
    value = _read(path)
    if (set(value) != {"schema", "reason", "closed_at"}
            or value.get("schema") != "lineage_nim_transport_closed_v1"
            or not isinstance(value.get("reason"), str) or not value["reason"]
            or not isinstance(value.get("closed_at"), str) or not value["closed_at"]):
        raise ValueError("transport close receipt가 다르다")
    return value


def _require_open(root: Path) -> None:
    if _closed(root) is not None:
        raise RuntimeError("종료된 transport session은 새 enqueue/dispatch를 허용하지 않는다")


def close(root: Path, reason: str) -> dict:
    """새 intent와 dispatch를 닫고 기존 pending/unknown 원문은 보존한다."""
    root = Path(root).resolve()
    _session(root)
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise ValueError("close reason은 1~500자 문자열이어야 한다")
    with _lock(root, "enqueue.lock"), _lock(root, "dispatch-gate.lock"):
        if _closed(root) is not None:
            raise FileExistsError("transport session은 이미 종료됐다")
        receipt = {"schema": "lineage_nim_transport_closed_v1", "reason": reason,
                   "closed_at": datetime.now(timezone.utc).isoformat()}
        _write_new(root / "closed.json", receipt)
        return receipt


def status(root: Path) -> dict:
    """write 없이 request/dispatch/response/pending/unknown 상태를 재집계한다."""
    root = Path(root).resolve()
    _session(root)
    requests, responses = _inventory(root)
    dispatches = sorted((root / "dispatches").iterdir())
    failures = sorted((root / "failures").iterdir())
    sha_receipts = sorted((root / "response-sha").iterdir())
    if any(path.suffix != ".json" or not path.is_file() for path in (*dispatches, *failures)):
        raise ValueError("dispatch/failure 원장에 등록되지 않은 entry가 있다")
    if any(path.suffix != ".sha256" or not path.is_file() for path in sha_receipts):
        raise ValueError("response SHA 원장에 등록되지 않은 entry가 있다")
    request_ids = {path.stem for path in requests}
    response_ids = {path.stem for path in responses}
    dispatch_ids = {path.stem for path in dispatches}
    failure_ids = {path.stem for path in failures}
    sha_ids = {path.stem for path in sha_receipts}
    if (not response_ids <= dispatch_ids or not dispatch_ids <= request_ids
            or not failure_ids <= dispatch_ids or sha_ids != response_ids):
        raise ValueError("request/dispatch/response/failure 원장 관계가 다르다")
    request_sha = {path.stem: _file_sha(path) for path in requests}
    dispatch_sha = {}
    for path in dispatches:
        row = _read(path)
        if (set(row) != {"schema", "request_id", "request_sha256", "claimed_at"}
                or row.get("schema") != "lineage_nim_transport_dispatch_v1"
                or row.get("request_id") != path.stem
                or row.get("request_sha256") != request_sha[path.stem]
                or not isinstance(row.get("claimed_at"), str) or not row["claimed_at"]):
            raise ValueError("dispatch claim이 request와 다르다")
        dispatch_sha[path.stem] = _file_sha(path)
    for path in failures:
        row = _read(path)
        if (set(row) != {"schema", "request_id", "request_sha256", "dispatch_sha256",
                         "error_type", "wall_s"}
                or row.get("schema") != "lineage_nim_transport_failure_v1"
                or row.get("request_id") != path.stem
                or row.get("request_sha256") != request_sha[path.stem]
                or row.get("dispatch_sha256") != dispatch_sha[path.stem]):
            raise ValueError("failure가 request/dispatch와 다르다")
    for path in responses:
        row, sha_receipt = _read(path), _read(root / "response-sha" / f"{path.stem}.sha256")
        if (row.get("schema") != RESPONSE_SCHEMA or row.get("request_id") != path.stem
                or row.get("request_sha256") != request_sha[path.stem]
                or row.get("dispatch_sha256") != dispatch_sha.get(path.stem)
                or sha_receipt != {"request_id": path.stem, "response_sha256": _file_sha(path)}):
            raise ValueError("response가 request/dispatch/SHA와 다르다")
    rows = []
    for path in requests:
        request_id = path.stem
        dispatched, responded = request_id in dispatch_ids, request_id in response_ids
        state = "response" if responded else "unknown" if dispatched else "pending"
        rows.append({"request_id": request_id, "state": state, "dispatched": dispatched,
                     "response": responded, "failure": request_id in failure_ids})
    return {"schema": "lineage_nim_transport_status_v1", "closed": _closed(root),
            "requests": len(requests), "dispatches": len(dispatches), "responses": len(responses),
            "pending": sum(row["state"] == "pending" for row in rows),
            "unknown": sum(row["state"] == "unknown" for row in rows),
            "failures": len(failures), "rows": rows}


def enqueue(root: Path, slot_id: str, messages: list[dict], params: dict,
            context_digest: str, view_digest: str) -> dict:
    """전송 전 intent를 write-once로 기록한다. 기존 pending은 재전송하지 않는다."""
    root = Path(root).resolve()
    with _lock(root, "enqueue.lock"):
        session = _session(root)
        _require_open(root)
        slots = {row["slot_id"]: row for row in session["plan"]["logical_slots"]}
        if slot_id not in slots:
            raise ValueError("등록되지 않은 logical slot")
        if params != session["params"]:
            raise ValueError("model/request params가 동결 계약과 다르다")
        if not HEX64.fullmatch(context_digest) or not HEX64.fullmatch(view_digest):
            raise ValueError("context/view digest 형식이 잘못됐다")
        sent = _messages(messages)
        requests, responses = _inventory(root)
        response_ids = {path.stem for path in responses}
        pending = [path for path in requests if path.stem not in response_ids]
        if pending:
            raise RuntimeError("미완료 intent가 있어 새 요청·자동 재전송을 거부한다")
        slot_requests = [path for path in requests if _read(path).get("slot_id") == slot_id]
        if len(requests) >= session["plan"]["global_request_cap"]:
            raise RuntimeError("사전등록 global request cap을 넘는다")
        if len(slot_requests) >= session["plan"]["per_slot_max"]:
            raise RuntimeError("logical slot당 최대 3회 요청을 넘는다")
        ordinal = len(slot_requests) + 1
        base = {"schema": REQUEST_SCHEMA, "slot_id": slot_id, "case_id": slots[slot_id]["case_id"],
                "ordinal": ordinal, "messages": sent, "params": params,
                "context_digest": context_digest, "view_digest": view_digest,
                "created_at": datetime.now(timezone.utc).isoformat()}
        request_id = f"{slot_id}-{ordinal}-{_digest(base)[:16]}"
        body = {**base, "request_id": request_id}
        relative = f"requests/{request_id}.json"
        path = _inside(root, relative)
        _write_new(path, body)
        return {"request_id": request_id, "relative_path": relative, "request_sha256": _file_sha(path)}


def _request(root: Path, request_ref: dict) -> tuple[Path, dict]:
    if not isinstance(request_ref, dict) or set(request_ref) != {
            "request_id", "relative_path", "request_sha256"}:
        raise ValueError("request ref 필드가 다르다")
    request_id = request_ref["request_id"]
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("request ID 형식이 잘못됐다")
    expected_relative = f"requests/{request_id}.json"
    if request_ref["relative_path"] != expected_relative:
        raise ValueError("request ref 경로가 ID와 다르다")
    path = _inside(root, expected_relative)
    if not path.is_file() or _file_sha(path) != request_ref["request_sha256"]:
        raise ValueError("request 원문 SHA가 다르다")
    return path, _read(path)


def _validate_result(result: Any, session: dict) -> dict:
    if not isinstance(result, dict) or not set(result).issubset(RESULT_KEYS):
        raise ValueError("provider 완료 body shape가 F.openai_complete 계약 밖이다")
    if result.get("content") is not None and result.get("provider_model") not in session["accepted_provider_models"]:
        raise ValueError("등록되지 않은 provider model")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in result and (type(result[key]) is not int or result[key] < 0):
            raise ValueError("usage token 형식이 잘못됐다")
    if "wall_s" in result and (isinstance(result["wall_s"], bool)
                               or not isinstance(result["wall_s"], (int, float))
                               or result["wall_s"] < 0):
        raise ValueError("provider wall_s 형식이 잘못됐다")
    return json.loads(json.dumps(result, allow_nan=False))


def _actual_wsl_ready() -> None:
    release = platform.release().lower()
    proc = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower() \
        if Path("/proc/version").is_file() else ""
    if os.name != "posix" or ("microsoft" not in release and "microsoft" not in proc
                              and "wsl_interop" not in os.environ):
        raise RuntimeError("실제 provider worker는 WSL에서만 실행할 수 있다")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("WSL NVIDIA_API_KEY가 없다")


def serve_one(root: Path, request_ref: dict, *, complete: Callable | None = None) -> dict:
    """한 pending intent만 소비한다. transport 자체 재시도는 없다."""
    root = Path(root).resolve()
    with _lock(root, "server.lock"):
        session = _session(root)
        _require_open(root)
        path, request = _request(root, request_ref)
        slots = {row["slot_id"]: row for row in session["plan"]["logical_slots"]}
        if (request.get("schema") != REQUEST_SCHEMA or request.get("request_id") != request_ref["request_id"]
                or request.get("slot_id") not in slots
                or request.get("case_id") != slots[request["slot_id"]]["case_id"]
                or request.get("params") != session["params"]
                or not HEX64.fullmatch(str(request.get("context_digest")))
                or not HEX64.fullmatch(str(request.get("view_digest")))):
            raise ValueError("request의 slot/model/context/view 결합이 다르다")
        _messages(request.get("messages"))
        identity = {key: value for key, value in request.items() if key != "request_id"}
        expected_id = f"{request['slot_id']}-{request['ordinal']}-{_digest(identity)[:16]}"
        if request["request_id"] != expected_id:
            raise ValueError("request ID가 canonical context/view intent와 다르다")
        response_path = root / "responses" / path.name
        sha_path = root / "response-sha" / f"{path.stem}.sha256"
        dispatch_path = root / "dispatches" / path.name
        failure_path = root / "failures" / path.name
        if response_path.exists() or sha_path.exists():
            raise FileExistsError("request는 이미 완료됐거나 응답이 불완전하다")
        if dispatch_path.exists() or failure_path.exists():
            raise RuntimeError("기존 dispatch claim/failure가 있어 request를 재전송하지 않는다")
        if complete is None:
            _actual_wsl_ready()
            complete = F.openai_complete
        with _lock(root, "dispatch-gate.lock"):
            _require_open(root)
            claim = {"schema": "lineage_nim_transport_dispatch_v1", "request_id": path.stem,
                     "request_sha256": request_ref["request_sha256"],
                     "claimed_at": datetime.now(timezone.utc).isoformat()}
            _write_new(dispatch_path, claim)
            started = time.perf_counter()
            try:
                raw = complete(request["messages"], request["params"])
                result = _validate_result(raw, session)
            except BaseException as error:
                failure = {"schema": "lineage_nim_transport_failure_v1", "request_id": path.stem,
                           "request_sha256": request_ref["request_sha256"],
                           "dispatch_sha256": _file_sha(dispatch_path),
                           "error_type": type(error).__name__, "wall_s": time.perf_counter() - started}
                _write_new(failure_path, failure)
                raise
        response = {"schema": RESPONSE_SCHEMA, "request_id": path.stem,
                    "request_sha256": request_ref["request_sha256"], "result": result,
                    "dispatch_sha256": _file_sha(dispatch_path),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "transport_wall_s": time.perf_counter() - started}
        _write_new(response_path, response)
        response_sha = _file_sha(response_path)
        _write_new(sha_path, {"request_id": path.stem, "response_sha256": response_sha})
        return {"request_ref": request_ref, "response_relative": f"responses/{path.name}",
                "response_sha256": response_sha}


def file_completion(root: Path, slot_id: str, context_digest: str,
                    view_digest: str, timeout_s: float) -> Callable[[list[dict], dict], dict]:
    """CALL2.engine용 유한 대기 complete callback을 만든다."""
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 900:
        raise ValueError("timeout_s는 0초 초과 900초 이하로 고정해야 한다")
    root = Path(root).resolve()

    def complete(messages: list[dict], params: dict) -> dict:
        ref = enqueue(root, slot_id, messages, params, context_digest, view_digest)
        response_path = root / "responses" / f"{ref['request_id']}.json"
        sha_path = root / "response-sha" / f"{ref['request_id']}.sha256"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if response_path.is_file() and sha_path.is_file():
                receipt = _read(sha_path)
                if (set(receipt) != {"request_id", "response_sha256"}
                        or receipt["request_id"] != ref["request_id"]
                        or _file_sha(response_path) != receipt["response_sha256"]):
                    raise ValueError("response SHA/ID가 변조됐다")
                response = _read(response_path)
                if (response.get("schema") != RESPONSE_SCHEMA
                        or response.get("request_id") != ref["request_id"]
                        or response.get("request_sha256") != ref["request_sha256"]):
                    raise ValueError("response가 request intent와 다르다")
                return _validate_result(response.get("result"), _session(root))
            time.sleep(0.01)
        if response_path.exists() or sha_path.exists():
            raise ValueError("부분 응답은 완료로 소비하지 않는다")
        raise TimeoutError("WSL response 유한 대기 시간이 끝났다; pending intent는 보존했다")

    return complete


def _next_pending(root: Path) -> dict:
    requests, responses = _inventory(root)
    response_ids = {path.stem for path in responses}
    dispatch_ids = {path.stem for path in (Path(root) / "dispatches").iterdir()}
    pending = [path for path in requests if path.stem not in response_ids | dispatch_ids]
    if len(pending) != 1:
        raise RuntimeError("serve-next는 정확히 한 pending intent만 허용한다")
    path = pending[0]
    return {"request_id": path.stem, "relative_path": f"requests/{path.name}",
            "request_sha256": _file_sha(path)}


def run_worker(root: Path, *, max_wall_s: float, poll_s: float = 0.25,
               complete: Callable | None = None) -> dict:
    """한 queue를 유한 시간 소비하며 close 또는 실제 global cap 소진까지 기다린다."""
    if (isinstance(max_wall_s, bool) or not isinstance(max_wall_s, (int, float))
            or not 0 < max_wall_s <= MAX_WORKER_WALL_S):
        raise ValueError("max_wall_s는 0초 초과 21600초 이하여야 한다")
    if (isinstance(poll_s, bool) or not isinstance(poll_s, (int, float))
            or not 0.01 <= poll_s <= 60 or poll_s > max_wall_s):
        raise ValueError("poll_s는 0.01~60초이며 max_wall_s 이하여야 한다")
    root = Path(root).resolve()
    evidence_scope = "actual_wsl" if complete is None else "cpu_fixture"
    with _lock(root, "worker.lock"):
        session = _session(root)
        worker_root = root / "worker"
        if any((worker_root / name).exists() for name in ("start.json", "result.json", "failure.json")):
            raise FileExistsError("worker 영수증이 이미 있어 재실행하지 않는다")
        started_at = datetime.now(timezone.utc).isoformat()
        start = {"schema": "lineage_nim_transport_worker_start_v1",
                 "registration_anchor": session["registration_anchor"],
                 "session_sha256": _file_sha(root / "session.json"),
                 "evidence_scope": evidence_scope, "max_wall_s": max_wall_s,
                 "poll_s": poll_s, "started_at": started_at}
        _write_new(worker_root / "start.json", start)
        started = time.monotonic()
        posts = 0
        try:
            if complete is None:
                _actual_wsl_ready()
            while True:
                measured = status(root)
                if measured["closed"] is not None:
                    outcome = "closed"
                    break
                if measured["unknown"] or measured["failures"]:
                    raise RuntimeError("claimed unknown/failure가 있어 추가 POST를 거부한다")
                if measured["pending"] > 1:
                    raise RuntimeError("worker는 정확히 한 unclaimed pending만 소비할 수 있다")
                if measured["pending"] == 1:
                    serve_one(root, _next_pending(root), complete=complete)
                    posts += 1
                    continue
                if measured["requests"] >= session["plan"]["global_request_cap"]:
                    outcome = "global_cap_exhausted"
                    break
                if time.monotonic() - started >= max_wall_s:
                    outcome = "deadline"
                    break
                remaining = max_wall_s - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(min(poll_s, remaining))
            final = status(root)
            result = {"schema": "lineage_nim_transport_worker_result_v1", "status": outcome,
                      "evidence_scope": evidence_scope, "posts": posts,
                      "start_sha256": _file_sha(worker_root / "start.json"),
                      "queue_status": final, "wall_s": time.monotonic() - started,
                      "finished_at": datetime.now(timezone.utc).isoformat()}
            _write_new(worker_root / "result.json", result)
            return result
        except BaseException as error:
            failure = {"schema": "lineage_nim_transport_worker_failure_v1",
                       "evidence_scope": evidence_scope, "posts": posts,
                       "start_sha256": _file_sha(worker_root / "start.json"),
                       "error_type": type(error).__name__, "wall_s": time.monotonic() - started,
                       "failed_at": datetime.now(timezone.utc).isoformat()}
            _write_new(worker_root / "failure.json", failure)
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "initialize", "serve-next", "once", "close", "status", "worker"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--request-ref", type=Path)
    parser.add_argument("--reason")
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--poll-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "initialize":
        if args.protocol is None or args.anchor is None or args.plan is None:
            parser.error("initialize에는 --protocol --anchor --plan이 필요하다")
        initialize(args.root, args.protocol, args.anchor, _read(args.plan))
        return 0
    if args.command == "close":
        if args.reason is None:
            parser.error("close에는 --reason이 필요하다")
        close(args.root, args.reason)
        return 0
    if args.command == "status":
        print(_canonical(status(args.root)).decode("utf-8"))
        return 0
    if args.command == "worker":
        if args.max_wall_s is None:
            parser.error("worker에는 --max-wall-s가 필요하다")
        result = run_worker(args.root, max_wall_s=args.max_wall_s, poll_s=args.poll_s)
        return 2 if result["status"] == "deadline" else 0
    ref = _next_pending(args.root) if args.command == "serve-next" else (
        _read(args.request_ref) if args.request_ref is not None else parser.error("once에는 --request-ref가 필요하다"))
    serve_one(args.root, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
