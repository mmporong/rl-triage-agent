"""사전등록한 진단·패치·한 번의 피드백 수정만 NVIDIA에 전송한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)] + [str(ROOT / p) for p in ("evals", "bench", "src")]
import diagnostic_patch_bridge_v1 as BRIDGE
import diagnostic_patch_prompts_v1 as PROMPTS
import ppo_patch_contract_v1 as CONTRACT


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path, *, lf: bool = False) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n") if lf else raw).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def artifact(ref: dict, root: Path) -> dict:
    if type(ref) is not dict or set(ref) != {"scope", "path", "sha256"}:
        raise ValueError("근거 참조 형식이 다르다")
    base = {"run": root, "repo": ROOT}.get(ref["scope"])
    relative = PurePosixPath(ref["path"])
    if base is None or relative.is_absolute() or ".." in relative.parts or ":" in str(relative) or "\\" in str(relative):
        raise ValueError("근거 상대경로가 잘못됐다")
    path = (base / str(relative)).resolve()
    if not path.is_relative_to(base.resolve()) or sha(path) != ref["sha256"]:
        raise ValueError("근거 파일 경계 또는 SHA가 다르다")
    return read(path)


def native_proposal(response: dict, request: dict, *, kind: str) -> dict:
    if response.get("status") != "valid" or response.get("context_sha256") != BRIDGE.digest(request["context"]):
        raise ValueError("상위 모델 응답이 유효하지 않다")
    raw = json.loads(response["provider_raw"]["content"])
    validator = BRIDGE.validate_choice if kind == "diagnosis" else CONTRACT.validate_proposal
    checked = validator(raw, request["context"])
    if checked != response["proposal"]:
        raise ValueError("모델 원문과 검증 제안이 다르다")
    return checked


def validate_request(protocol: dict, request: dict, root: Path) -> tuple[dict, list[dict]]:
    if request.get("schema") != "diagnostic_patch_request_v1" or request.get("model") != protocol["model"]:
        raise ValueError("등록 모델 요청 계약이 다르다")
    case = next((c for c in protocol["cases"] if c["case_id"] == request["case_id"]), None)
    if case is None:
        raise ValueError("사전등록되지 않은 사례다")
    kind, chain = request["kind"], request["chain"]
    if kind == "diagnosis":
        if case["mode"] == "runtime_pairs":
            if chain != {"runtime_inputs": case["runtime_inputs"]}:
                raise ValueError("실제 정책 쌍 입력이 등록과 다르다")
            expected = BRIDGE.runtime_diagnosis_context(case["source"], protocol["CPU_configurations"][0],
                case_id=case["case_id"], runtime=artifact(chain["runtime_inputs"], root))
        else:
            if case["mode"] != "new_adapter" or chain:
                raise ValueError("진단 최초 입력의 계보가 다르다")
            expected = BRIDGE.diagnosis_context(case["source"], protocol["CPU_configurations"][0], case_id=case["case_id"])
    elif kind == "patch":
        if case["mode"] not in {"new_adapter", "runtime_pairs"}:
            raise ValueError("저장 응답에 새 최초 패치를 덧붙일 수 없다")
        parent = artifact(chain["diagnosis_request"], root)
        validate_request(protocol, parent, root)
        if parent["kind"] != "diagnosis" or parent["case_id"] != request["case_id"]:
            raise ValueError("패치와 상위 진단 사례가 다르다")
        response = artifact(chain["diagnosis_response"], root)
        choice = native_proposal(response, parent, kind="diagnosis")
        baseline, reference = artifact(chain["baseline"], root), artifact(chain["reference"], root)
        authorization = artifact(chain["authorization"], root)
        confirmation = artifact(chain["confirmation"], root)
        if confirmation != BRIDGE.bind_confirmation(parent["context"], choice, baseline, reference, authorization):
            raise ValueError("확인 검사 영수증을 재계산할 수 없다")
        expected = BRIDGE.patch_context(parent["context"], confirmation, baseline, reference)
    elif kind == "revision":
        import ppo_patch_feedback_v1 as FEEDBACK
        parent = artifact(chain["first_request"], root)
        response = artifact(chain["first_response"], root)
        if case["mode"] == "historical_feedback":
            if chain["first_request"] != case["first_request"] or chain["first_response"] != case["first_response"]:
                raise ValueError("저장 최초 응답 원문이 등록과 다르다")
        else:
            if parent["kind"] != "patch" or parent["case_id"] != request["case_id"]:
                raise ValueError("재수정은 최초 패치 한 번에만 연결된다")
            validate_request(protocol, parent, root)
        first = native_proposal(response, parent, kind="patch")
        expected = FEEDBACK.build_feedback_context(parent["context"], first, artifact(chain["validation"], root))
    else:
        raise ValueError("등록되지 않은 모델 단계다")
    if request["context"] != expected or request["context_sha256"] != BRIDGE.digest(expected):
        raise ValueError("요청 context가 원문 계보와 다르다")
    messages = PROMPTS.messages(kind, expected)
    if request["messages"] != messages:
        raise ValueError("실제 메시지가 등록된 프롬프트와 다르다")
    return expected, messages


def reserve(protocol: dict, request: dict, request_path: Path, root: Path, out: Path) -> int:
    directory = ROOT / "workspace/ppo-patch-generation" / protocol["campaign_id"] / "reservations"
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / f"slot-{request['case_id']}-{request['kind']}.lock"
    with lock.open("x", encoding="utf-8") as handle:
        try:
            prior = [read(p) for p in sorted(directory.glob("reservation-*.json"))]
            prior = [v for v in prior if v["case_id"] == request["case_id"] and v["kind"] == request["kind"]]
            if prior:
                if len(prior) >= 3:
                    raise ValueError("단계별 인프라 재시도 상한을 소진했다")
                previous = prior[-1]
                old_out = Path(previous["out_path"]).resolve()
                if previous["request_sha256"] != sha(request_path) or not old_out.is_relative_to(root) or not old_out.is_file():
                    raise ValueError("같은 요청의 봉인된 인프라 실패만 재시도할 수 있다")
                raw = read(old_out)["provider_raw"]
                infra = raw.get("infra_error") or (not raw.get("error_type") and (
                    not (raw.get("content") or "").strip() or raw.get("finish_reason") == "length"))
                if not infra:
                    raise ValueError("의미 결과 뒤 같은 모델 단계를 다시 호출할 수 없다")
            for ordinal in range(1, protocol["API_cap"] + 1):
                try:
                    write(directory / f"reservation-{ordinal:03d}.json", {
                        "request_sha256": sha(request_path), "case_id": request["case_id"],
                        "kind": request["kind"], "ordinal": ordinal, "consumed_before_send": True,
                        "out_path": str(out), "attempt": len(prior) + 1})
                    return ordinal
                except FileExistsError:
                    continue
            raise ValueError("전송 전 API 상한을 소진했다")
        finally:
            handle.close()
            lock.unlink()


def call(protocol_path: Path, request_path: Path, root: Path, out: Path) -> dict:
    if not sys.platform.startswith("linux") or not os.environ.get("WSL_DISTRO_NAME"):
        raise ValueError("기존 등록 WSL에서만 NIM을 호출한다")
    if out.exists():
        raise FileExistsError(out)
    protocol, request = read(protocol_path), read(request_path)
    anchor = protocol["source_anchor"]
    subprocess.run(["git", "merge-base", "--is-ancestor", anchor, "HEAD"], cwd=ROOT, check=True)
    blob = subprocess.check_output(["git", "show", "HEAD:" + protocol_path.relative_to(ROOT).as_posix()], cwd=ROOT)
    if hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest() != sha(protocol_path, lf=True):
        raise ValueError("프로토콜이 커밋 blob과 다르다")
    for rel, expected in protocol["source_sha256_lf"].items():
        source = subprocess.check_output(["git", "show", anchor + ":" + rel], cwd=ROOT)
        if sha(ROOT / rel, lf=True) != expected or hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError("등록 source가 다르다: " + rel)
    context, messages = validate_request(protocol, request, root)
    import feedback_model_call as MODEL
    params = MODEL.request_params(protocol["model"])
    if params != protocol["params"]:
        raise ValueError("모델 설정이 등록과 다르다")
    if not out.is_relative_to(root):
        raise ValueError("응답 출력이 작업 원문 경계를 벗어났다")
    ordinal = reserve(protocol, request, request_path, root, out)
    raw = MODEL.openai_complete(messages, params)
    record: dict[str, Any] = {"schema": "diagnostic_patch_response_v1", "kind": request["kind"],
        "request_sha256": sha(request_path), "context_sha256": BRIDGE.digest(context),
        "provider_raw": raw, "status": "invalid", "proposal": None, "error": None, "reservation": ordinal}
    try:
        if raw.get("error_type") or raw.get("infra_error") or raw.get("provider_model") not in protocol["accepted_models"] or raw.get("finish_reason") != "stop":
            raise ValueError("provider 모델·종료·전송 조건이 다르다")
        value = json.loads(raw["content"])
        validator = BRIDGE.validate_choice if request["kind"] == "diagnosis" else CONTRACT.validate_proposal
        record["proposal"] = validator(value, context)
        record["status"] = "valid"
    except (ValueError, TypeError, KeyError) as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    write(out, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("protocol", "request", "root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = call(args.protocol.resolve(), args.request.resolve(), args.root.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "reservation": result["reservation"]}))


if __name__ == "__main__":
    main()
