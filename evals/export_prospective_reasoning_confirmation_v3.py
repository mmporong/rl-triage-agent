"""실행 폴더의 fact ledger에 결합된 규칙 arm을 재생해 V3 결과를 공개한다."""
# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "evals"), str(ROOT / "src")]

import export_prospective_reasoning_confirmation_v1 as BASE
import export_prospective_reasoning_confirmation_v2 as RELEASE2


def require(ok: object, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _runtime_rule_initial(root: Path, corpus_initial: dict[str, Any],
                          epoch: dict[str, Any]) -> dict[str, Any]:
    """동결 packet과 실행 시 봉인된 ledger로 실제 초기 state를 다시 만든다."""
    root = Path(root).resolve()
    ledger = root / "fact-ledger.json"
    BASE.R.SEL._state(corpus_initial)
    ledger_value = BASE.R.SEL.LEDGER.checked(ledger)
    corpus_ledger = corpus_initial.get("fact_ledger")
    require(isinstance(corpus_ledger, dict)
            and corpus_ledger.get("sha256") == BASE._sha(ledger)
            and corpus_ledger.get("source_binding") == ledger_value.get("source_binding"),
            "rule runtime ledger와 corpus ledger 내용 결합 불일치")
    packet = corpus_initial.get("initial_packet")
    require(isinstance(packet, dict), "rule corpus 초기 packet 누락")
    packet = cast(dict[str, Any], packet)
    expected = BASE.R.SEL.initialize(
        deepcopy(packet), ledger, budget_left=cast(int, corpus_initial["budget_left"]))
    BASE.R.SEL._state(expected)
    source_binding = epoch.get("source_binding")
    input_bindings = packet.get("input_bindings")
    require(isinstance(source_binding, dict) and isinstance(input_bindings, dict),
            "rule epoch/packet source binding 누락")
    source_binding = cast(dict[str, Any], source_binding)
    input_bindings = cast(dict[str, Any], input_bindings)
    for role in ("case", "reference"):
        epoch_role, input_role = source_binding.get(role), input_bindings.get(role)
        require(isinstance(epoch_role, dict) and isinstance(input_role, dict)
                and epoch_role.get("checkpoint_sha256") == input_role.get("checkpoint_sha256")
                and epoch_role.get("telemetry_sha256") == input_role.get("telemetry_sha256"),
                f"rule {role} epoch source와 초기 packet 불일치")
    require(expected["state_digest"] == epoch.get("initial_state_digest"),
            "rule runtime 초기 state digest 불일치")
    return expected


def _verify_rule_epoch(root: Path, corpus_initial: dict[str, Any], row: dict[str, Any], *,
                       case_id: str, category: str) -> dict[str, Any]:
    epoch = BASE._read(Path(root) / "result.json")
    initial = _runtime_rule_initial(root, corpus_initial, epoch)
    return BASE._verify_rule_epoch(
        root, initial, row, case_id=case_id, category=category)


def _collect(result_path: Path, source_batch: Path, corpus: Path,
             bridge_root: Path, design: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                                  list[dict[str, Any]],
                                                                  dict[str, Any]]:
    """V1 collect와 같되 규칙 arm 초기 state를 실행 ledger에서 재생한다."""
    result = BASE._read(result_path)
    rows = result.get("rows")
    require(result.get("schema") == BASE.R.RESULT_SCHEMA and result.get("status") == "complete"
            and isinstance(rows, list) and len(rows) == 24, "완료된 24행 result가 아니다")
    rows = cast(list[dict[str, Any]], rows)
    expected_order = [(f"c{i:02d}", policy) for i in range(1, 9)
                      for policy in BASE.R.POLICIES]
    require([(row.get("case_id"), row.get("policy")) for row in rows] == expected_order,
            "parent row 순서/정책 불일치")
    source_rows = BASE.R._source_rows(source_batch, design)
    _corpus_value, corpus_rows = BASE.R._corpus(corpus)
    BASE.R._bind_rows(source_rows, corpus_rows)
    by_case = {row["case_id"]: row for row in source_rows}
    public_rows: list[dict[str, Any]] = []
    public_calls: list[dict[str, Any]] = []
    parent_wall = child_wall = 0.0
    for raw in rows:
        case_id, policy = cast(str, raw["case_id"]), cast(str, raw["policy"])
        source = by_case[case_id]
        if case_id in BASE.R.NORMAL_CASES:
            BASE._normal_row(raw, case_id, policy)
            detail = {"parent_probe_wall_s": 0.0, "owned_child_wall_s": 0.0}
        else:
            epoch_root = result_path.parent / case_id / policy
            initial = corpus_rows[case_id]["state"]
            if policy in BASE.MODEL_POLICIES:
                detail, calls = BASE._verify_model_epoch(
                    epoch_root, initial, raw, case_id=case_id, policy=policy,
                    category=source["category"])
                public_calls.extend(calls)
            else:
                detail = _verify_rule_epoch(
                    epoch_root, initial, raw, case_id=case_id, category=source["category"])
        parent_wall += detail["parent_probe_wall_s"]
        child_wall += detail["owned_child_wall_s"]
        public_rows.append({"schema": BASE.CASE_SCHEMA, **deepcopy(raw),
                            "seed": source["seed"], "category": source["category"],
                            "checkpoint_sha256": source["checkpoint"]["sha256"]})
    BASE._verify_parent_aggregate(result, rows, design)
    bridge_requests = sorted((Path(bridge_root) / "requests").glob("[0-9][0-9][0-9].json"))
    bridge_responses = sorted((Path(bridge_root) / "responses").glob("[0-9][0-9][0-9].json"))
    require(len(bridge_requests) == len(bridge_responses) == len(public_calls)
            == result["model_requests"], "bridge request/response/model 분모 불일치")
    require({call["slot"] for call in public_calls}
            == {BASE._read(path)["slot"] for path in bridge_requests},
            "bridge slot 소비 불일치")
    timing = {"api_wall_s": sum(call["api_wall_s"] for call in public_calls),
              "parent_probe_wall_s": parent_wall, "owned_child_wall_s": child_wall,
              "scope": "component_sums_not_whole_execution_wall"}
    return public_rows, public_calls, timing


def _base_export(result_path: Path, protocol_path: Path, design_path: Path,
                 bridge_path: Path, source_batch: Path, corpus: Path,
                 bridge_root: Path, *, anchor: str, out: Path) -> dict[str, Any]:
    # 반복되는 commit-blob 증명만 scoped cache에 둔다. 현재 파일과 receipt 검증은
    # 각 소비 함수가 그대로 수행하며 scope 종료 뒤 cache는 남지 않는다.
    with BASE.R.COMP.CACHE.proof_scope():
        return BASE.export(result_path, protocol_path, design_path, bridge_path,
                           source_batch, corpus, bridge_root, anchor=anchor, out=out,
                           collect_fn=_collect)


def export(execution_root: Path, protocol_path: Path, design_path: Path,
           bridge_path: Path, source_batch: Path, corpus: Path,
           bridge_root: Path, failure_summary: Path, rule_audit: Path, *,
           anchor: str, out: Path) -> dict[str, Any]:
    """V2 release provenance를 보존하고 runtime-ledger 검증만 교체한다."""
    return RELEASE2.export(execution_root, protocol_path, design_path, bridge_path,
                           source_batch, corpus, bridge_root, failure_summary, rule_audit,
                           anchor=anchor, out=out, base_export_fn=_base_export)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--source-batch", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--failure-summary", type=Path, required=True)
    parser.add_argument("--rule-audit", type=Path, required=True)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = export(args.execution_root, args.protocol, args.design, args.bridge,
                   args.source_batch, args.corpus, args.bridge_root,
                   args.failure_summary, args.rule_audit,
                   anchor=args.anchor, out=args.out)
    print(json.dumps({"status": value["status"], "assessment": value["assessment"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
