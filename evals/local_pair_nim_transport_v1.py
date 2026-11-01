"""새 local pair 비교를 위한 write-once Nemotron 3.5 NIM transport."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import agentic_investigation_call_v2 as CALL
import selection_comparison_call_v2 as MODEL_CALL

PROTOCOL_SCHEMA = "local_pair_four_arm_protocol_v1"
REQUEST_SCHEMA = "local_pair_nim_request_v1"
MANIFEST_SCHEMA = "local_pair_nim_manifest_v1"
INTENT_SCHEMA = "local_pair_nim_intent_v1"
RESULT_SCHEMA = "local_pair_nim_result_v1"
RECEIPT_SCHEMA = "local_pair_nim_receipt_v1"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT_S = 90.0
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}\Z")
SOURCE_FILES = (
    "evals/local_pair_nim_transport_v1.py",
    "tests/test_local_pair_nim_transport_v1.py",
    "evals/agentic_investigation_call_v2.py",
    "evals/agentic_investigation_call_v1.py",
    "evals/agentic_investigation_contract_v2.py",
    "evals/selection_comparison_call_v2.py",
    "evals/feedback_model_call.py",
)
AUTHORITY = {"api": True, "gpu": True, "probe": True,
             "correction": False, "retraining": False}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path, *, lf: bool = False) -> str:
    raw = Path(path).read_bytes()
    if lf:
        raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{Path(path).name}은 JSON object여야 한다")
    return value


def _directory_sync(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path: Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{target.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    temporary = target.parent / suffix
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except OSError:
            if os.name != "nt":
                raise
            os.rename(temporary, target)
        else:
            temporary.unlink()
        _directory_sync(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _directory_sync(target.parent)


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git protocol/source 결합을 확인할 수 없다")
    return result.stdout


def _relative(repo: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    except ValueError as error:
        raise ValueError("protocol은 repo 내부여야 한다") from error


def _source_path(repo: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError("source 상대 경로가 잘못됐다")
    path = (Path(repo).resolve() / relative).resolve()
    if Path(repo).resolve() not in path.parents:
        raise ValueError("source가 repo 밖을 가리킨다")
    return path


def _verify_protocol(protocol_path: Path, anchor: str) -> dict[str, Any]:
    repo = ROOT.resolve()
    path = Path(protocol_path).resolve()
    relative = _relative(repo, path)
    if not HEX40.fullmatch(anchor):
        raise ValueError("execution anchor 형식이 잘못됐다")
    value = _read(path)
    required = {"schema", "api_cap", "model", "accepted_provider_models", "base_params",
                "sources_lf", "backend0prefix", "authority", "automatic_retries"}
    if not required <= set(value):
        raise ValueError("local pair protocol 필수 필드가 없다")
    if (value.get("schema") != PROTOCOL_SCHEMA
            or value.get("api_cap") != 12
            or value.get("model") != CALL.PARAMS["model"]
            or value.get("accepted_provider_models") != sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS)
            or value.get("base_params") != CALL.PARAMS
            or value.get("backend0prefix") != 0
            or value.get("authority") != AUTHORITY
            or value.get("automatic_retries") != 0
            or not isinstance(value.get("sources_lf"), dict)
            or not set(SOURCE_FILES) <= set(value["sources_lf"])):
        raise ValueError("local pair protocol 모델·상한·authority·source 결합이 다르다")
    _git(repo, "cat-file", "-e", f"{anchor}^{{commit}}")
    _git(repo, "merge-base", "--is-ancestor", anchor, "HEAD")
    bindings = {relative: _sha(path, lf=True), **value["sources_lf"]}
    for name, expected in bindings.items():
        source = _source_path(repo, name)
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError("source SHA 형식이 잘못됐다")
        if not source.is_file() or _sha(source, lf=True) != expected:
            raise ValueError(f"현재 source SHA가 protocol과 다르다: {name}")
        committed = _git(repo, "show", f"{anchor}:{name}")
        committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"anchor source blob이 protocol과 다르다: {name}")
    return deepcopy(value)


def initialize(root: Path, *, protocol_path: Path, execution_anchor: str, api_cap: int) -> dict:
    """등록 protocol을 검증하고 baseline 0인 fresh transport를 만든다."""
    target = Path(root).resolve()
    if target.exists():
        raise FileExistsError("transport root는 fresh 경로여야 한다")
    protocol = _verify_protocol(protocol_path, execution_anchor)
    if api_cap != 12 or api_cap != protocol["api_cap"]:
        raise ValueError("호출 API cap이 등록 protocol과 다르다")
    target.mkdir(parents=True)
    for name in ("calls", "requests", "locks"):
        (target / name).mkdir()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": {"relative": _relative(ROOT, protocol_path),
                     "sha256_lf": _sha(protocol_path, lf=True)},
        "execution_anchor": execution_anchor,
        "api_cap": api_cap,
        "backend0prefix": 0,
        "model": MODEL_CALL.MODEL,
        "accepted_provider_models": sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(CALL.PARAMS),
        "automatic_retries": 0,
    }
    _write_new(target / "manifest.json", manifest)
    return deepcopy(manifest)


def _manifest(root: Path) -> dict[str, Any]:
    target = Path(root).resolve()
    value = _read(target / "manifest.json")
    protocol_path = ROOT / value.get("protocol", {}).get("relative", "")
    protocol = _verify_protocol(protocol_path, value.get("execution_anchor", ""))
    expected = {
        "schema": MANIFEST_SCHEMA,
        "protocol": {"relative": _relative(ROOT, protocol_path),
                     "sha256_lf": _sha(protocol_path, lf=True)},
        "execution_anchor": value.get("execution_anchor"),
        "api_cap": protocol["api_cap"],
        "backend0prefix": 0,
        "model": MODEL_CALL.MODEL,
        "accepted_provider_models": sorted(MODEL_CALL.ACCEPTED_PROVIDER_MODELS),
        "base_params": deepcopy(CALL.PARAMS),
        "automatic_retries": 0,
    }
    if value != expected:
        raise ValueError("transport manifest가 protocol과 다르다")
    return value


def _request(value: Any) -> dict[str, Any]:
    keys = {"schema", "run_id", "arm", "step", "context_digest", "messages", "params",
            "source_binding", "request_id"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("NIM request 필드가 다르다")
    if (value.get("schema") != REQUEST_SCHEMA
            or any(not isinstance(value.get(key), str) or not IDENTIFIER.fullmatch(value[key])
                   for key in ("run_id", "request_id"))
            or value.get("arm") not in {"logs_only", "logs_plus_frames"}
            or type(value.get("step")) is not int or not 1 <= value["step"] <= 3
            or value.get("request_id") != f"{value.get('run_id')}_{value.get('arm')}_{value.get('step')}"
            or not HEX64.fullmatch(str(value.get("context_digest")))):
        raise ValueError("NIM request ID/context 형식이 다르다")
    binding = value.get("source_binding")
    binding_keys = {"pair_sha256", "case_checkpoint_sha256", "reference_checkpoint_sha256"}
    if (not isinstance(binding, dict) or set(binding) != binding_keys
            or any(not isinstance(item, str) or not HEX64.fullmatch(item)
                   for item in binding.values())):
        raise ValueError("NIM request pair/source/checkpoint 결합이 다르다")
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise TypeError("NIM request messages 형식이 다르다")
    expected_params = CALL.params_for_messages(messages)
    if value.get("params") != expected_params or value["params"].get("model") != MODEL_CALL.MODEL:
        raise ValueError("NIM request messages/params/model이 다르다")
    return deepcopy(value)


@contextmanager
def _lock(root: Path, name: str, timeout_s: float = 5.0) -> Iterator[None]:
    path = Path(root) / "locks" / name
    token = f"{os.getpid()}-{time.monotonic_ns()}"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"transport lock timeout: {name}")
            time.sleep(0.01)
    try:
        os.write(descriptor, token.encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        if path.is_file() and path.read_text(encoding="ascii") == token:
            path.unlink()


def _allocate(root: Path, request: dict[str, Any], manifest: dict[str, Any]) -> tuple[Path, bool]:
    request_digest = _digest(request)
    index_path = Path(root) / "requests" / f"{request['request_id']}.json"
    with _lock(root, "allocate.lock"):
        if index_path.exists():
            index = _read(index_path)
            if index != {"request_id": request["request_id"], "request_digest": request_digest,
                         "ordinal": index.get("ordinal")}:
                raise ValueError("같은 request_id의 원문이 바뀌었다")
            call = Path(root) / "calls" / f"{index['ordinal']:04d}"
            if not call.is_dir():
                raise ValueError("request index의 call이 없다")
            return call, False
        calls = sorted(path for path in (Path(root) / "calls").iterdir() if path.is_dir())
        if len(calls) >= manifest["api_cap"]:
            raise RuntimeError("NIM API cap을 초과했다")
        ordinal = len(calls) + 1
        call = Path(root) / "calls" / f"{ordinal:04d}"
        call.mkdir()
        intent = {"schema": INTENT_SCHEMA, "ordinal": ordinal, "request": request,
                  "request_digest": request_digest, "params_sha256": _digest(request["params"]),
                  "wire_body_sha256": _digest(wire_body(request)),
                  "protocol_sha256_lf": manifest["protocol"]["sha256_lf"],
                  "execution_anchor": manifest["execution_anchor"]}
        _write_new(call / "intent.json", intent)
        _write_new(index_path, {"request_id": request["request_id"],
                                "request_digest": request_digest, "ordinal": ordinal})
        return call, True


def _worker_evidence(call: Path, raw: dict[str, Any]) -> bool:
    """live는 worker의 write-once claim과 완료 원문이 같은 경우에만 인정한다."""
    if raw.get("live") is not True:
        return False
    claim_path = call / "sent-claim.json"
    response_path = call / "worker-response.json"
    error_path = call / "worker-error.json"
    process_path = call / "worker-process.json"
    exit_path = call / "worker-exit.json"
    if (not claim_path.is_file() or not process_path.is_file() or not exit_path.is_file()
            or response_path.is_file() == error_path.is_file()):
        raise ValueError("live worker claim/completion 증거가 불완전하다")
    claim = _read(claim_path)
    intent = _read(call / "intent.json")
    expected_claim = {
        "schema": "local_pair_nim_sent_claim_v1", "ordinal": intent["ordinal"],
        "intent_sha256": _sha(call / "intent.json"), "endpoint": ENDPOINT,
        "timeout_s": TIMEOUT_S, "worker_source_sha256_lf": _sha(Path(__file__), lf=True),
        "manifest_sha256": _sha(call.parents[1] / "manifest.json"),
        "protocol_sha256_lf": intent["protocol_sha256_lf"],
        "execution_anchor": intent["execution_anchor"],
        "wire_body_sha256": intent["wire_body_sha256"],
        "call_relative": f"calls/{call.name}",
        "worker_argv_contract": ["--worker", "--root", "--call-dir", "--protocol", "--anchor"],
    }
    if claim != expected_claim:
        raise ValueError("live worker claim 결합이 다르다")
    process = _read(process_path)
    exited = _read(exit_path)
    expected_exit_code = 0 if response_path.is_file() else 1
    if (set(process) != {"pid", "proc_start_ticks", "wsl"}
            or process.get("wsl") is not True
            or type(process.get("pid")) is not int or process["pid"] <= 0
            or not isinstance(process.get("proc_start_ticks"), str)
            or not process["proc_start_ticks"].isdigit()
            or exited != {"identity": process, "exit_code": expected_exit_code,
                          "worker_source_sha256_lf": _sha(Path(__file__), lf=True)}):
        raise ValueError("live worker process/exit identity 결합이 다르다")
    recorded = _read(response_path if response_path.is_file() else error_path)
    if recorded != raw:
        raise ValueError("live worker 반환과 write-once 원문이 다르다")
    return True


def _flatten_wire_body(params_value: dict[str, Any], messages: list[Any]) -> dict[str, Any]:
    params = deepcopy(params_value)
    extra = params.pop("extra_body", None)
    if not isinstance(extra, dict):
        raise TypeError("NIM params extra_body는 object여야 한다")
    collisions = set(extra) & ({*params, "messages"})
    if collisions:
        raise ValueError(f"NIM extra_body top-level 충돌: {sorted(collisions)}")
    return {**params, **extra, "messages": deepcopy(messages)}


def wire_body(request: dict[str, Any]) -> dict[str, Any]:
    """OpenAI SDK의 extra_body 병합 규칙을 urllib JSON body로 명시한다."""
    checked = _request(request)
    return _flatten_wire_body(checked["params"], checked["messages"])


def _provider_result(raw: Any, *, wall_s: float, live: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("send 결과는 object여야 한다")
    allowed = {"live", "http_status", "provider_response", "wall_s", "error_type", "error_message"}
    if not set(raw) <= allowed or raw.get("live") is not live:
        raise ValueError("send 결과 필드/live 표기가 다르다")
    observed_wall = raw.get("wall_s", wall_s)
    if isinstance(observed_wall, bool) or not isinstance(observed_wall, (int, float)) or observed_wall < 0:
        raise ValueError("send wall_s 형식이 다르다")
    status = raw.get("http_status")
    provider = raw.get("provider_response")
    base = {"live": live, "http_status": status, "wall_s": observed_wall,
            "error_type": raw.get("error_type"), "error_message": raw.get("error_message"),
            "provider_response": deepcopy(provider)}
    if raw.get("error_type") or status != 200 or not isinstance(provider, dict):
        return {**base, "outcome": "infrastructure_unknown", "content": None,
                "finish_reason": None, "provider_model": None, "usage": None}
    choices = provider.get("choices")
    usage = provider.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return {**base, "outcome": "infrastructure_unknown", "content": None,
                "finish_reason": None, "provider_model": provider.get("model"),
                "usage": deepcopy(usage)}
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return {**base, "outcome": "infrastructure_unknown", "content": None,
                "finish_reason": choices[0].get("finish_reason"),
                "provider_model": provider.get("model"), "usage": deepcopy(usage)}
    content = message.get("content")
    finish = choices[0].get("finish_reason")
    model = provider.get("model")
    if not isinstance(content, str) or not content.strip() or finish == "length":
        return {**base, "outcome": "infrastructure_unknown", "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": finish, "provider_model": model, "usage": deepcopy(usage)}
    semantic_valid = model in MODEL_CALL.ACCEPTED_PROVIDER_MODELS and finish == "stop"
    return {**base, "outcome": "semantic_valid" if semantic_valid else "semantic_invalid",
            "content": content, "reasoning_content": message.get("reasoning_content"),
            "finish_reason": finish, "provider_model": model, "usage": deepcopy(usage)}


def verify_call(call_path: Path) -> dict[str, Any]:
    """intent→dispatch→response→receipt SHA 사슬과 분류를 다시 계산한다."""
    call = Path(call_path).resolve()
    root = call.parents[1]
    manifest = _manifest(root)
    if call.parent != root / "calls" or not re.fullmatch(r"[0-9]{4}", call.name):
        raise ValueError("NIM call 경로가 transport calls 밖이다")
    intent_path, dispatch_path = call / "intent.json", call / "dispatch-claim.json"
    response_path, receipt_path = call / "response.json", call / "receipt.json"
    intent, dispatch, response, receipt = map(_read, (intent_path, dispatch_path, response_path, receipt_path))
    request = _request(intent.get("request"))
    if (intent.get("schema") != INTENT_SCHEMA or intent.get("request_digest") != _digest(request)
            or intent.get("ordinal") != int(call.name) or not 1 <= intent["ordinal"] <= manifest["api_cap"]
            or intent.get("protocol_sha256_lf") != manifest["protocol"]["sha256_lf"]
            or intent.get("execution_anchor") != manifest["execution_anchor"]
            or intent.get("params_sha256") != _digest(request["params"])
            or intent.get("wire_body_sha256") != _digest(wire_body(request))
            or dispatch != {"schema": "local_pair_nim_dispatch_v1",
                            "ordinal": intent.get("ordinal"),
                            "intent_sha256": _sha(intent_path), "automatic_retry": 0,
                            "params_sha256": intent.get("params_sha256"),
                            "wire_body_sha256": intent.get("wire_body_sha256")}
            or response.get("schema") != RESULT_SCHEMA
            or response.get("ordinal") != intent.get("ordinal")
            or response.get("intent_sha256") != _sha(intent_path)
            or response.get("dispatch_sha256") != _sha(dispatch_path)
            or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("ordinal") != intent.get("ordinal")
            or receipt.get("intent_sha256") != _sha(intent_path)
            or receipt.get("dispatch_sha256") != _sha(dispatch_path)
            or receipt.get("response_sha256") != _sha(response_path)
            or receipt.get("request_id") != request["request_id"]
            or receipt.get("outcome") != response.get("result", {}).get("outcome")
            or type(receipt.get("global_api_calls")) is not int
            or not 1 <= receipt["global_api_calls"] <= manifest["api_cap"]):
        raise ValueError("NIM call SHA 사슬 또는 request 결합이 다르다")
    result = response.get("result")
    if not isinstance(result, dict) or result.get("outcome") not in {
            "semantic_valid", "semantic_invalid", "infrastructure_unknown"}:
        raise ValueError("NIM call 결과 분류가 다르다")
    replay_raw = {key: deepcopy(result.get(key)) for key in (
        "live", "http_status", "provider_response", "wall_s", "error_type", "error_message")}
    if result.get("live") is True:
        worker_path = call / ("worker-response.json" if (call / "worker-response.json").is_file()
                              else "worker-error.json")
        replay_raw = _read(worker_path)
        if not _worker_evidence(call, replay_raw):
            raise ValueError("NIM live worker 증거가 확인되지 않았다")
    elif any((call / name).exists() for name in (
            "sent-claim.json", "worker-response.json", "worker-error.json",
            "worker-process.json", "worker-exit.json")):
        raise ValueError("fixture 결과에 live worker 파일이 섞였다")
    replayed = _provider_result(replay_raw, wall_s=float(result.get("wall_s", 0)),
                                live=result.get("live") is True)
    if replayed != result:
        raise ValueError("NIM provider 원문 재분류가 저장 결과와 다르다")
    status = {"semantic_valid": "completed", "semantic_invalid": "unknown",
              "infrastructure_unknown": "infra_error"}[result["outcome"]]
    worker_evidence = None
    if result.get("live") is True:
        worker_evidence = {"process_sha256": _sha(call / "worker-process.json"),
                           "exit_sha256": _sha(call / "worker-exit.json")}
    return {"schema": "local_pair_nim_verified_call_v1", "status": status,
            "content": result.get("content"),
            "call_ref": {"path": str(call), "sha256": _sha(receipt_path)},
            "live": result.get("live"), "usage": deepcopy(result.get("usage")),
            "wall_s": result.get("wall_s"), "ordinal": intent["ordinal"],
            "request": deepcopy(request), "request_ref": {"request_id": request["request_id"],
                                                             "digest": intent["request_digest"]},
            "outcome": result["outcome"], "worker_evidence": worker_evidence}


def complete(root: Path, request: dict, *, send: Callable[[dict, Path], dict]) -> dict:
    """새 request는 한 번만 보내고, 완료된 동일 request는 검증된 receipt를 재사용한다."""
    target = Path(root).resolve()
    manifest = _manifest(target)
    checked = _request(request)
    call, created = _allocate(target, checked, manifest)
    if not created:
        if (call / "receipt.json").is_file():
            return verify_call(call)
        raise RuntimeError("기존 dispatch가 미완료이므로 request를 다시 보내지 않는다")
    intent_path = call / "intent.json"
    intent = _read(intent_path)
    dispatch = {"schema": "local_pair_nim_dispatch_v1", "ordinal": intent["ordinal"],
                "intent_sha256": _sha(intent_path), "automatic_retry": 0,
                "params_sha256": intent["params_sha256"],
                "wire_body_sha256": intent["wire_body_sha256"]}
    _write_new(call / "dispatch-claim.json", dispatch)
    body = wire_body(checked)
    started = time.perf_counter()
    try:
        raw = send(body, call)
        live = _worker_evidence(call, raw) if isinstance(raw, dict) else False
        result = _provider_result(raw, wall_s=time.perf_counter() - started,
                                  live=live)
    except Exception as error:  # noqa: BLE001 - callback 실패도 재전송 없이 원장에 봉인한다.
        result = _provider_result({"live": False, "http_status": None,
                                   "error_type": type(error).__name__,
                                   "error_message": str(error),
                                   "wall_s": time.perf_counter() - started},
                                  wall_s=time.perf_counter() - started, live=False)
    response = {"schema": RESULT_SCHEMA, "ordinal": intent["ordinal"],
                "intent_sha256": _sha(intent_path),
                "dispatch_sha256": _sha(call / "dispatch-claim.json"), "result": result}
    _write_new(call / "response.json", response)
    receipt = {"schema": RECEIPT_SCHEMA, "ordinal": intent["ordinal"],
               "request_id": checked["request_id"], "intent_sha256": _sha(intent_path),
               "dispatch_sha256": _sha(call / "dispatch-claim.json"),
               "response_sha256": _sha(call / "response.json"), "outcome": result["outcome"],
               "live": result["live"], "usage": deepcopy(result.get("usage")),
               "provider_wall_s": result["wall_s"],
               "global_api_calls": len(list((target / "calls").glob("[0-9][0-9][0-9][0-9]/dispatch-claim.json")))}
    _write_new(call / "receipt.json", receipt)
    return verify_call(call)


def _actual_wsl_ready() -> None:
    release = Path("/proc/sys/kernel/osrelease").read_text(
        encoding="utf-8", errors="replace").lower() \
        if Path("/proc/sys/kernel/osrelease").is_file() else ""
    version = Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower() \
        if Path("/proc/version").is_file() else ""
    if (os.name != "posix" or not Path("/proc/self/stat").is_file()
            or "microsoft" not in f"{release}\n{version}"
            and "wsl_interop" not in os.environ):
        raise RuntimeError("실제 provider worker는 WSL에서만 실행할 수 있다")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("WSL NVIDIA_API_KEY가 없다")


def _process_identity(*, actual: bool) -> dict[str, Any]:
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = stat.rfind(")")
    fields = stat[closing + 2:].split() if closing >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit():
        raise RuntimeError("worker /proc start identity를 읽을 수 없다")
    return {"pid": pid, "proc_start_ticks": fields[19], "wsl": actual}


def _worker(root: Path, call_dir: Path, protocol: Path, anchor: str, *, live: bool = False) -> int:
    target, call = Path(root).resolve(), Path(call_dir).resolve()
    manifest = _manifest(target)
    if Path(protocol).resolve() != (ROOT / manifest["protocol"]["relative"]).resolve():
        raise ValueError("worker protocol 경로가 manifest와 다르다")
    if anchor != manifest["execution_anchor"] or call.parent != target / "calls":
        raise ValueError("worker anchor/call 경계가 다르다")
    intent = _read(call / "intent.json")
    _request(intent.get("request"))
    ordinal = int(call.name)
    if call.name != f"{ordinal:04d}" or intent.get("ordinal") != ordinal:
        raise ValueError("worker ordinal이 다르다")
    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("WSL NVIDIA_API_KEY가 없다")
    claim = {"schema": "local_pair_nim_sent_claim_v1", "ordinal": ordinal,
             "intent_sha256": _sha(call / "intent.json"), "endpoint": ENDPOINT,
             "timeout_s": TIMEOUT_S, "worker_source_sha256_lf": _sha(Path(__file__), lf=True),
             "manifest_sha256": _sha(target / "manifest.json"),
             "protocol_sha256_lf": intent["protocol_sha256_lf"],
             "execution_anchor": intent["execution_anchor"],
             "wire_body_sha256": intent["wire_body_sha256"],
             "call_relative": f"calls/{call.name}",
             "worker_argv_contract": ["--worker", "--root", "--call-dir", "--protocol", "--anchor"]}
    _write_new(call / "sent-claim.json", claim)
    body = wire_body(intent["request"])
    request = urllib.request.Request(ENDPOINT, data=_canonical(body), method="POST",
                                     headers={"Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
                                              "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.read()
            value = {"live": live, "http_status": response.status,
                     "provider_response": json.loads(raw), "wall_s": time.perf_counter() - started}
        _write_new(call / "worker-response.json", value)
        return 0
    except Exception as error:  # noqa: BLE001 - HTTP/JSON/timeout 실패를 한 worker 원장에 봉인한다.
        status = error.code if isinstance(error, urllib.error.HTTPError) else None
        value = {"live": live, "http_status": status, "error_type": type(error).__name__,
                 "error_message": str(error), "wall_s": time.perf_counter() - started}
        _write_new(call / "worker-error.json", value)
        return 1


def _run_worker_cli(root: Path, call_dir: Path, protocol: Path, anchor: str) -> int:
    target, call = Path(root).resolve(), Path(call_dir).resolve()
    manifest = _manifest(target)
    if (Path(protocol).resolve() != (ROOT / manifest["protocol"]["relative"]).resolve()
            or anchor != manifest["execution_anchor"] or call.parent != target / "calls"):
        raise ValueError("worker CLI root/call/protocol/anchor 경계가 다르다")
    _actual_wsl_ready()
    identity = _process_identity(actual=True)
    process_path = call / "worker-process.json"
    _write_new(process_path, identity)
    exit_code = 1
    try:
        exit_code = _worker(target, call, protocol, anchor, live=True)
        return exit_code
    finally:
        current = _process_identity(actual=True)
        _write_new(call / "worker-exit.json",
                   {"identity": current,
                    "exit_code": exit_code,
                    "worker_source_sha256_lf": _sha(Path(__file__), lf=True)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--call-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--anchor")
    args = parser.parse_args(argv)
    if not args.worker or any(value is None for value in (args.root, args.call_dir, args.protocol, args.anchor)):
        parser.error("worker 인자 전체가 필요하다")
    return _run_worker_cli(args.root, args.call_dir, args.protocol, args.anchor)


if __name__ == "__main__":
    raise SystemExit(main())
