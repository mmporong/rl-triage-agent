"""클라우드의 공통뷰를 WSL에서 호출한다. 자격증명은 전송하지 않는다."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import brev_recovery_contract_v1 as P
import selection_comparison_controller_v1 as C

CALL = C.CALL
MC = CALL.MC


def audit(root: Path) -> dict:
    reservations = sorted((root / "reservations").glob("*.json"))
    responses = sorted((root / "responses").glob("*.json"))
    names = [path.name for path in reservations]
    if names != [f"{index:04d}.json" for index in range(1, len(names) + 1)] or len(names) > 20:
        raise ValueError("API 예약 순서 또는 상한")
    response_names = {path.name for path in responses}
    if not response_names <= set(names):
        raise ValueError("예약 없는 응답")
    rows = []
    for reservation in reservations:
        request = P.read(reservation)
        response = root / "responses" / reservation.name
        if response.exists():
            reply = P.read(response)
            if reply["reservation_sha256"] != P.sha(reservation) or reply["request"] != request:
                raise ValueError("API 응답 결합")
            rows.append(reply)
    return {"reserved": len(names), "completed": len(rows),
            "pending": len(names) - len(rows), "remaining": 20 - len(names), "rows": rows}


def rebind_snapshot(envelope: dict, folder: Path) -> dict:
    remote = envelope["snapshot"]
    if P.digest(remote) != envelope["snapshot_sha256"]:
        raise ValueError("원격 snapshot 해시")
    ledger = folder / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("x", encoding="utf-8", newline="\n") as stream:
        for event in envelope["ledger_events"]:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    inputs = {**deepcopy(remote["inputs"]), "ledger_path": str(ledger.resolve())}
    local = C.V.build_snapshot(**inputs)
    if any(local[key] != remote[key] for key in remote if key != "inputs"):
        raise ValueError("전송 뒤 공통뷰 또는 원장 의미 변경")
    remote_inputs = {key: value for key, value in remote["inputs"].items() if key != "ledger_path"}
    if {key: value for key, value in local["inputs"].items() if key != "ledger_path"} != remote_inputs:
        raise ValueError("전송 뒤 입력 결합 변경")
    return local


def decide(envelope: dict, folder: Path, api_root: Path, anchor: str, *, complete=None) -> dict:
    protocol = P.verify(anchor)
    if (envelope["schema"] != "brev_recovery_model_request_v1"
            or envelope["anchor"] != anchor or envelope["protocol_sha256_lf"] != P.sha(P.PROTOCOL, True)
            or envelope["tag"] != protocol["tag"]):
        raise ValueError("모델 요청 실행 결합")
    snapshot = rebind_snapshot(envelope, folder)
    inputs = C.dialogue_input(snapshot)
    if not inputs["needs_model"] or inputs["messages_digest"] != envelope["messages_digest"]:
        raise ValueError("닫힌 상태 또는 다른 모델 입력")
    if complete is None:
        if not CALL.is_wsl():
            raise ValueError("실제 모델 호출은 WSL에서만 허용")
        complete = CALL.F.openai_complete
    results = []

    def bounded(messages, params):
        P.verify(anchor)
        if params != CALL.F.request_params(protocol["selection"]["model"]):
            raise ValueError("모델 매개변수 불일치")
        with MC._active_lease(api_root):
            before = audit(api_root)
            if before["pending"] or before["remaining"] <= 0:
                raise MC.BudgetStop("API 미완료 예약 또는 요청 상한")
            index = before["reserved"] + 1
            path = api_root / "reservations" / f"{index:04d}.json"
            request = {"schema": "brev_recovery_api_reservation_v1", "index": index,
                       "snapshot_sha256": envelope["snapshot_sha256"], "anchor": anchor,
                       "protocol_sha256_lf": envelope["protocol_sha256_lf"],
                       "messages_digest": P.digest(messages), "params": params}
            P.write_new(path, request)
            # 예외 문자열과 HTTP 헤더는 저장하지 않는다.
            result = CALL.safe_result(complete(messages, params))
            P.write_new(api_root / "responses" / path.name,
                        {"request": request, "reservation_sha256": P.sha(path), "result": result})
        results.append(result)
        return result

    engine = CALL.engine(snapshot, inputs["view"], inputs["messages"], "production", bounded)
    result = {"schema": "brev_recovery_model_response_v1", "anchor": anchor,
              "protocol_sha256_lf": envelope["protocol_sha256_lf"],
              "snapshot_sha256": envelope["snapshot_sha256"],
              "messages_digest": inputs["messages_digest"], "safe_results": results,
              "local_engine": engine, "api_audit": audit(api_root)}
    result["stop_reason"] = engine["stop_reason"]
    if result["api_audit"]["pending"]:
        raise ValueError("모델 호출 뒤 미완료 예약")
    P.write_new(folder / "response.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    args = parser.parse_args()
    try:
        result = decide(P.read(args.envelope), args.output_folder, args.api_root, args.anchor)
        print(json.dumps({"status": "recorded", "requests": len(result["safe_results"]),
                          "model_status": result["local_engine"]["selection"]["model_status"]}))
        return 0
    except Exception as error:  # noqa: BLE001 - 자격증명 없는 실패 종류만 출력한다.
        print(json.dumps({"status": "invalid", "error_type": type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
