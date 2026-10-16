"""보조 영상 관찰을 위한 6-frame packet, 요청 원장, WSL 호출기.

영상 관찰은 고정 평가를 대체하지 않는다. 모델에는 여섯 프레임과 동결된
관찰 질문만 보내며 checkpoint/fixed-eval 결합값은 로컬 packet에만 남긴다.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "evals") not in sys.path:
    sys.path.insert(0, str(ROOT / "evals"))

import feedback_model_call as MODEL_CALL
import nemotron_bottleneck_ledger_v2 as TRANSPORT
import nemotron_bottleneck_vision_v2 as VISION
import nemotron_endpoint_ledger_v3 as ENDPOINT
import vl_aux_worker_v1 as WORKER


MODEL = VISION.MODEL
PROVIDER_ALIAS = ENDPOINT.PRIVATE_CANDIDATE_ALIAS
TOTAL_API_CAP = 8
MAX_ATTEMPTS_PER_CLIP = 2
EXPECTED_POLICY_COUNT = 3
EXPECTED_API_CAP = EXPECTED_POLICY_COUNT * MAX_ATTEMPTS_PER_CLIP
MAX_POLICY_COUNT = TOTAL_API_CAP // MAX_ATTEMPTS_PER_CLIP
PACKET_SCHEMA = "vl_aux_observation_packet_v1"
RECORD_SCHEMA = "vl_aux_observation_record_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BINDING_KEYS = frozenset({"checkpoint_sha256", "fixed_eval_sha256"})


class LedgerStop(RuntimeError):
    """예약 상한, 중단 예약, 중복 packet 때문에 전송할 수 없다."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object가 필요하다")
    return value


def _write_new(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_binding(binding: dict[str, Any]) -> dict[str, str]:
    if not isinstance(binding, dict) or set(binding) != _BINDING_KEYS:
        raise ValueError("binding에는 checkpoint/fixed-eval SHA만 필요하다")
    if any(not isinstance(binding[key], str) or not _SHA256.fullmatch(binding[key])
           for key in _BINDING_KEYS):
        raise ValueError("binding SHA 형식이 잘못됐다")
    return {key: binding[key] for key in sorted(_BINDING_KEYS)}


def prepare(video: Path, outdir: Path, binding: dict[str, Any]) -> Path:
    """영상에서 여섯 프레임을 만들고 비전 입력과 로컬 provenance를 결합한다."""
    source = Path(video).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    private_binding = _validate_binding(binding)
    output = Path(outdir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frames_dir = output / "frames"
    frames_dir.mkdir()
    frames = VISION._extract_frames(source, frames_dir)
    source_sha = _sha(source)
    clip_id = _digest({"source_sha256": source_sha, "binding": private_binding})[:24]
    packet = {
        "schema": PACKET_SCHEMA,
        "clip_id": clip_id,
        "source": {"sha256": source_sha, "bytes": source.stat().st_size},
        "frames": [{**frame, "file": f"frames/{frame['file']}"} for frame in frames],
        "binding": private_binding,
        "prompt": {"system": VISION.VIDEO_SYSTEM, "question": VISION.VIDEO_QUESTION},
        "request_params": VISION._request_params(),
    }
    packet_path = output / "packet.json"
    _write_new(packet_path, packet)
    return packet_path


def _case(packet_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    packet_path = Path(packet_path).resolve()
    packet = _read(packet_path)
    if (packet.get("schema") != PACKET_SCHEMA or not isinstance(packet.get("clip_id"), str)
            or packet.get("prompt") != {"system": VISION.VIDEO_SYSTEM, "question": VISION.VIDEO_QUESTION}
            or packet.get("request_params") != VISION._request_params()):
        raise ValueError("동결 packet 계약이 다르다")
    binding = _validate_binding(packet.get("binding"))
    source = packet.get("source")
    if (not isinstance(source, dict) or not _SHA256.fullmatch(str(source.get("sha256", "")))
            or not isinstance(source.get("bytes"), int) or isinstance(source.get("bytes"), bool)
            or source["bytes"] <= 0):
        raise ValueError("source 결합이 잘못됐다")
    if packet["clip_id"] != _digest({"source_sha256": source["sha256"], "binding": binding})[:24]:
        raise ValueError("clip_id 결합이 다르다")
    frames = packet.get("frames")
    if not isinstance(frames, list) or len(frames) != VISION.FRAME_COUNT:
        raise ValueError("여섯 프레임이 필요하다")
    content: list[dict[str, Any]] = [{"type": "text", "text": VISION.VIDEO_QUESTION}]
    public_frames = []
    prior_timestamp = -1.0
    for index, frame in enumerate(frames, 1):
        if not isinstance(frame, dict) or set(frame) != {"file", "timestamp_s", "sha256", "bytes"}:
            raise ValueError("frame metadata가 잘못됐다")
        relative = Path(frame["file"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != f"frames/frame_{index:02d}.jpg":
            raise ValueError("frame 경로가 잘못됐다")
        timestamp = frame["timestamp_s"]
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp <= prior_timestamp):
            raise ValueError("frame timestamp가 잘못됐다")
        if (not isinstance(frame["sha256"], str) or not _SHA256.fullmatch(frame["sha256"])
                or not isinstance(frame["bytes"], int) or isinstance(frame["bytes"], bool)
                or not 0 < frame["bytes"] <= VISION.MAX_JPEG_BYTES):
            raise ValueError("frame hash/bytes가 잘못됐다")
        prior_timestamp = float(timestamp)
        path = packet_path.parent / relative
        content.append({"type": "image_url", "image_url": {
            "url": VISION._data_url(path, frame["sha256"], frame["bytes"]), "detail": "low"}})
        public_frames.append({key: frame[key] for key in ("timestamp_s", "sha256", "bytes")})
    messages = [{"role": "system", "content": VISION.VIDEO_SYSTEM},
                {"role": "user", "content": content}]
    case = {"id": packet["clip_id"], "kind": "vision", "messages": messages,
            "request_params": packet["request_params"], "packet": {"preflight": True}}
    public = {"clip_id": packet["clip_id"], "source": source, "frames": public_frames,
              "messages_sha256": _digest(messages)}
    return case, public


class RequestLedger:
    """모든 API 시도를 전송 전에 연속 파일로 예약하는 작은 write-once 원장."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _lock(self) -> Path:
        lock = self.root / "ledger.lock"
        _write_new(lock, {"pid": os.getpid(), "created_at": _now()})
        return lock

    def audit(self) -> list[dict[str, Any]]:
        reservations = sorted((self.root / "reservations").glob("*.json"))
        responses = {path.name: path for path in (self.root / "responses").glob("*.json")}
        if len(reservations) > TOTAL_API_CAP:
            raise ValueError("VL 보조 관찰 요청 상한 초과")
        rows = []
        for sequence, path in enumerate(reservations, 1):
            reservation = _read(path)
            if (path.name != f"{sequence:04d}.json" or reservation.get("sequence") != sequence
                    or reservation.get("schema") != "vl_aux_reservation_v1"
                    or reservation.get("model") != MODEL
                    or not _SHA256.fullmatch(str(reservation.get("packet_sha256", "")))
                    or not _SHA256.fullmatch(str(reservation.get("request_sha256", "")))
                    or reservation.get("attempt") not in (1, 2)
                    or reservation.get("previous_sha256") != (_digest(rows[-1]["reservation"]) if rows else None)):
                raise ValueError("VL 예약 원장 연결이 다르다")
            response_path = responses.pop(path.name, None)
            response = _read(response_path) if response_path else None
            if response is not None and (response.get("sequence") != sequence
                                         or response.get("reservation_sha256") != _digest(reservation)
                                         or response.get("transport_status") not in {
                                             "ok", "infra_error", "call_error"}):
                raise ValueError("VL 응답과 예약 결합이 다르다")
            rows.append({"reservation": reservation, "response": response})
        if responses:
            raise ValueError("예약 없는 VL 응답이 있다")
        return rows

    def reserve(self, packet_sha256: str, clip_id: str, attempt: int, request_sha256: str) -> dict[str, Any]:
        lock = self._lock()
        try:
            rows = self.audit()
            if any(row["response"] is None for row in rows):
                raise LedgerStop("응답 없는 예약을 crash-unknown으로 보존한다")
            if any(row["reservation"]["packet_sha256"] == packet_sha256
                   and row["reservation"]["attempt"] == attempt for row in rows):
                raise LedgerStop("같은 packet 시도를 다시 보내지 않는다")
            if attempt == 1 and any(row["reservation"]["packet_sha256"] == packet_sha256 for row in rows):
                raise LedgerStop("같은 packet을 다시 실행하지 않는다")
            prior = [row for row in rows if row["reservation"]["packet_sha256"] == packet_sha256]
            if attempt == 2 and (len(prior) != 1 or prior[0]["reservation"]["attempt"] != 1
                                 or prior[0]["response"].get("transport_status") != "infra_error"):
                raise LedgerStop("인프라 오류 뒤에만 두 번째 시도를 예약한다")
            if len(rows) >= TOTAL_API_CAP:
                raise LedgerStop("VL 보조 관찰 전체 요청 상한에 도달했다")
            reservation = {"schema": "vl_aux_reservation_v1", "sequence": len(rows) + 1,
                           "packet_sha256": packet_sha256, "clip_id": clip_id, "attempt": attempt,
                           "model": MODEL, "request_sha256": request_sha256,
                           "previous_sha256": _digest(rows[-1]["reservation"]) if rows else None,
                           "reserved_at": _now()}
            _write_new(self.root / "reservations" / f"{reservation['sequence']:04d}.json", reservation)
            return reservation
        finally:
            lock.unlink(missing_ok=True)

    def finish(self, reservation: dict[str, Any], result: dict[str, Any], status: str) -> None:
        response = {"schema": "vl_aux_response_v1", "sequence": reservation["sequence"],
                    "reservation_sha256": _digest(reservation), "transport_status": status,
                    "result": result, "finished_at": _now()}
        _write_new(self.root / "responses" / f"{reservation['sequence']:04d}.json", response)


def _safe_result(raw: dict[str, Any]) -> dict[str, Any]:
    fields = ("response_id", "provider_model", "provider_model_raw", "provider_identity_registered",
              "identity_policy", "content", "finish_reason", "input_tokens", "output_tokens",
              "total_tokens", "wall_s", "error_type", "status_code", "infra_error")
    return {key: raw[key] for key in fields if key in raw}


def _verified_provider_result(saved: dict[str, Any]) -> dict[str, Any]:
    """저장된 raw provider ID에서 등록 alias view를 다시 계산한다."""
    if not isinstance(saved, dict):
        raise ValueError("저장된 model result가 object가 아니다")
    measured = saved.get("measured_wall_s")
    if (isinstance(measured, bool) or not isinstance(measured, (int, float))
            or not math.isfinite(measured) or measured < 0):
        raise ValueError("model wall time이 잘못됐다")
    raw = {key: saved[key] for key in (
        "response_id", "content", "finish_reason", "input_tokens", "output_tokens", "total_tokens",
        "wall_s", "error_type", "status_code", "infra_error") if key in saved}
    if "provider_model" in saved or saved.get("provider_model_raw") is not None:
        raw["provider_model"] = saved.get("provider_model_raw", saved.get("provider_model"))
    view = ENDPOINT.provider_view(raw, MODEL)
    expected = _safe_result(view)
    expected["measured_wall_s"] = measured
    if saved != expected:
        raise ValueError("provider alias view 또는 저장 model result가 다르다")
    return view


def verify_record(packet_path: Path, output_path: Path, ledger_root: Path) -> dict[str, Any]:
    """worker record를 packet과 write-once 요청 원장에서 다시 계산해 검증한다."""
    packet_path, output_path = Path(packet_path).resolve(), Path(output_path).resolve()
    record = _read(output_path)
    case, public = _case(packet_path)
    packet_sha = _sha(packet_path)
    params = {"model": MODEL, **case["request_params"]}
    request_sha = _digest({"messages": case["messages"], **params})
    rows = RequestLedger(ledger_root).audit()
    if any(row["response"] is None for row in rows):
        raise ValueError("완료 응답 없는 원장 예약이 있다")
    selected = [row for row in rows if row["reservation"]["packet_sha256"] == packet_sha]
    if not 1 <= len(selected) <= MAX_ATTEMPTS_PER_CLIP or any(row["response"] is None for row in selected):
        raise ValueError("packet 요청 원장 수 또는 완료 상태가 다르다")
    attempts = []
    final_transport = None
    final_view = None
    for attempt, row in enumerate(selected, 1):
        reservation, response = row["reservation"], row["response"]
        if (reservation.get("clip_id") != public["clip_id"] or reservation.get("attempt") != attempt
                or reservation.get("request_sha256") != request_sha):
            raise ValueError("packet 요청 예약 의미가 다르다")
        final_view = _verified_provider_result(response.get("result"))
        final_transport = TRANSPORT.transport_status(final_view, MODEL)
        if response.get("transport_status") != final_transport:
            raise ValueError("저장 transport status가 재계산과 다르다")
        attempts.append({"attempt": attempt, "reservation_sha256": _digest(reservation),
                         "transport_status": final_transport, "result": response["result"]})
    if (len(attempts) == 2) != (attempts[0]["transport_status"] == "infra_error"):
        raise ValueError("clip별 인프라 재시도 계약이 다르다")
    verdict = None
    status = final_transport
    if final_transport == "ok":
        verdict = VISION.validate_response(case, final_view.get("content"))
        status = "valid" if verdict["valid"] else "invalid"
    expected = {"schema": RECORD_SCHEMA, **public, "model": MODEL,
                "provider_identity_policy": ENDPOINT.IDENTITY_POLICY,
                "packet_sha256": packet_sha, "request_sha256": request_sha,
                "status": status, "verdict": verdict, "attempts": attempts,
                "requests": len(attempts)}
    if record != expected:
        raise ValueError("VL observation record가 packet/원장 재계산과 다르다")
    return record


def call(packet_path: Path, output_path: Path, ledger_root: Path,
         *, complete: Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    """한 clip을 관찰한다. 의미 오류는 재시도하지 않고 인프라 오류만 한 번 재시도한다."""
    packet_path = Path(packet_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    case, public = _case(packet_path)
    packet_sha = _sha(packet_path)
    params = {"model": MODEL, **case["request_params"]}
    request_sha = _digest({"messages": case["messages"], **params})
    ledger = RequestLedger(ledger_root)
    sender = complete or MODEL_CALL.openai_complete
    attempts = []
    final_status = "infra_error"
    verdict = None
    for attempt in range(1, MAX_ATTEMPTS_PER_CLIP + 1):
        reservation = ledger.reserve(packet_sha, public["clip_id"], attempt, request_sha)
        started = time.perf_counter()
        try:
            raw = sender(case["messages"], params)
            if not isinstance(raw, dict):
                raise TypeError("model result must be object")
        except Exception as error:
            raw = {"error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
                   "infra_error": type(error).__name__ in {
                       "APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "TimeoutError"}}
        raw["measured_wall_s"] = time.perf_counter() - started
        view = ENDPOINT.provider_view(raw, MODEL)
        final_status = TRANSPORT.transport_status(view, MODEL)
        safe = _safe_result(view)
        safe["measured_wall_s"] = raw["measured_wall_s"]
        ledger.finish(reservation, safe, final_status)
        attempts.append({"attempt": attempt, "reservation_sha256": _digest(reservation),
                         "transport_status": final_status, "result": safe})
        if final_status == "infra_error" and attempt < MAX_ATTEMPTS_PER_CLIP:
            continue
        if final_status == "ok":
            verdict = VISION.validate_response(case, view.get("content"))
            final_status = "valid" if verdict["valid"] else "invalid"
        break
    record = {"schema": RECORD_SCHEMA, **public, "model": MODEL,
              "provider_identity_policy": ENDPOINT.IDENTITY_POLICY,
              "packet_sha256": packet_sha, "request_sha256": request_sha,
              "status": final_status, "verdict": verdict, "attempts": attempts,
              "requests": len(attempts)}
    _write_new(output_path, record)
    return verify_record(packet_path, output_path, ledger_root)


def invoke(packet_path: Path, output_path: Path, ledger_root: Path, *, runner=subprocess.run) -> dict[str, Any]:
    """Windows controller에서 기존 WSL 자격증명 환경의 단일 worker를 실행한다."""
    WORKER.invoke_worker("observation", Path(packet_path), Path(output_path), Path(ledger_root), runner=runner)
    return verify_record(packet_path, output_path, ledger_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--video", type=Path, required=True)
    prepare_parser.add_argument("--outdir", type=Path, required=True)
    prepare_parser.add_argument("--checkpoint-sha256", required=True)
    prepare_parser.add_argument("--fixed-eval-sha256", required=True)
    call_parser = sub.add_parser("call")
    call_parser.add_argument("--packet", type=Path, required=True)
    call_parser.add_argument("--output", type=Path, required=True)
    call_parser.add_argument("--ledger-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        path = prepare(args.video, args.outdir, {"checkpoint_sha256": args.checkpoint_sha256,
                                                 "fixed_eval_sha256": args.fixed_eval_sha256})
        print(json.dumps({"packet": str(path)}, ensure_ascii=False))
    else:
        result = call(args.packet, args.output, args.ledger_root)
        print(json.dumps({"status": result["status"], "requests": result["requests"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
