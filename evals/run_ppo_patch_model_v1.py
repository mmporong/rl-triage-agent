"""등록된 PPO 설정 어댑터 패치 요청을 NVIDIA에 한 번 보내고 원문을 봉인한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "bench")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def call(protocol_path: Path, request_path: Path, out: Path, reservations: Path) -> dict:
    if not sys.platform.startswith("linux") or not os.environ.get("WSL_DISTRO_NAME"):
        raise ValueError("등록 WSL에서만 모델을 호출한다")
    if out.exists():
        raise FileExistsError(out)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    anchor = protocol["source_anchor"]
    subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT, check=True)
    blob = subprocess.check_output(["git", "show", "HEAD:" + protocol_path.relative_to(ROOT).as_posix()], cwd=ROOT)
    if blob.replace(b"\r\n", b"\n") != protocol_path.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("실행 프로토콜이 커밋과 다르다")
    for rel, expected in protocol["source_sha256_lf"].items():
        source_blob = subprocess.check_output(["git", "show", anchor + ":" + rel], cwd=ROOT)
        if sha(ROOT / rel) != expected or hashlib.sha256(source_blob.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError("등록된 source SHA 불일치: " + rel)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["schema"] != "ppo_patch_model_request_v1" or request["model"] != protocol["model"]:
        raise ValueError("모델 요청 계약 불일치")
    context = request["context"]
    if request["context_sha256"] not in protocol["contexts_sha256"]:
        raise ValueError("사전등록 입력이 아니다")
    actual = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if actual != request["context_sha256"]:
        raise ValueError("요청 입력 SHA 불일치")
    import feedback_model_call as MODEL
    import ppo_patch_cases_v1 as CASES
    import ppo_patch_contract_v1 as CONTRACT

    messages = CASES.messages(context)
    if request["messages"] != messages:
        raise ValueError("실제 메시지가 입력·등록 프롬프트와 다르다")
    expected_reservations = ROOT / "workspace/ppo-patch-generation" / protocol["campaign_id"] / "reservations"
    if reservations.resolve() != expected_reservations.resolve():
        raise ValueError("요청 원장의 등록 경로가 다르다")
    reservations.mkdir(parents=True, exist_ok=True)
    for ordinal in range(1, protocol["API_cap"] + 1):
        try:
            write(reservations / f"reservation-{ordinal:02d}.json", {
                "request_sha256": sha(request_path), "source_sha256": context["source_sha256"],
                "ordinal": ordinal, "consumed_before_send": True})
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("API 요청 상한 소진")
    raw = MODEL.openai_complete(messages, MODEL.request_params(protocol["model"]))
    record = {"schema": "ppo_patch_model_response_v1", "request_sha256": sha(request_path),
              "context_sha256": actual, "provider_raw": raw, "status": "invalid", "proposal": None,
              "error": None, "reservation": ordinal}
    try:
        if raw.get("error_type") or raw.get("infra_error"):
            raise ValueError("NVIDIA provider 요청 실패")
        if raw.get("provider_model") not in protocol["accepted_models"] or raw.get("finish_reason") != "stop":
            raise ValueError("등록 모델 또는 종료 조건 불일치")
        content = raw.get("content")
        if not isinstance(content, str):
            raise TypeError("모델 텍스트 없음")
        proposal = json.loads(content)
        record["proposal"] = CONTRACT.validate_proposal(proposal, context)
        record["status"] = "valid"
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    write(out, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "request", "out", "reservations"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = call(args.protocol.resolve(), args.request.resolve(), args.out.resolve(), args.reservations.resolve())
    print(json.dumps({"status": result["status"], "reservation": result["reservation"]}))


if __name__ == "__main__":
    main()
